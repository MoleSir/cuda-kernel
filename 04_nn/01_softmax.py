import triton
import triton.language as tl
import torch
from torch import Tensor


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_ROW': 16, 'BLOCK_COL': 16}, num_warps=4),
    ],
    key=["ROWS", "COLS"]
)
@triton.jit
def softmax_kernel(
    x_ptr, y_ptr, 
    ROWS, COLS, 
    stride_x1, stride_x2,
    stride_y1, stride_y2,
    BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr,
):
    pid_r = tl.program_id(0)

    offs_r = pid_r * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    mask_r = offs_r < ROWS

    # BLOCK online softmax
    m_i = tl.full((BLOCK_ROW,), value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_ROW,), dtype=tl.float32)   # 改成 BLOCK_ROW

    for col_base in range(0, COLS, BLOCK_COL):
        offs_c = col_base + tl.arange(0, BLOCK_COL)

        # 加载 x
        x_ptrs = x_ptr + offs_r[:, None] * stride_x1 + offs_c[None, :] * stride_x2
        x_mask = mask_r & (offs_c < COLS) 
        x = tl.load(x_ptrs, mask=x_mask, other=float('-inf')) # (BLOCK_ROW, BLOCK_COL)

        # 计算当前最大值
        x_max = tl.max(x, axis=-1) # (BLOCK_ROW)
        m_new = tl.maximum(x_max, m_i) # (BLOCK_ROW)

        # 修正因子
        alpha = tl.exp(m_i - m_new) # (BLOCK_ROW)

        # 这块的 exp
        x_exp = tl.exp(x - m_new[:, None]) # (BLOCK_ROW, BLOCK_COL)
        x_exp_sum = tl.sum(x_exp, axis=-1) # (BLOCK_ROW)

        # 修正、增加累积值
        l_i = alpha * l_i + x_exp_sum
        m_i = m_new
    
    for col_base in range(0, COLS, BLOCK_COL):
        offs_c = col_base + tl.arange(0, BLOCK_COL)

        # 加载 x
        x_ptrs = x_ptr + offs_r[:, None] * stride_x1 + offs_c[None, :] * stride_x2
        x_mask = mask_r & (offs_c < COLS) 
        x = tl.load(x_ptrs, mask=x_mask, other=float('-inf')) # (BLOCK_ROW, BLOCK_COL)

        x_exp = tl.exp(x - m_i[:, None]) # (BLOCK_ROW, BLOCK_COL)
        x_exp_scale = x_exp / l_i[:, None]

        y_ptrs = y_ptr + offs_r[:, None] * stride_y1 + offs_c[None, :] * stride_y2
        y_mask = mask_r & (offs_c < COLS) 
        tl.store(y_ptrs, x_exp_scale, mask=y_mask)



def softmax(x: Tensor) -> Tensor:
    """
    x: (batch, dim) 
    y: (batch, dim)
    """
    rows, cols = x.shape
    y = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_ROW']), )
    softmax_kernel[grid](
        x, y, rows, cols,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )

    return y


def softmax_math(x: Tensor) -> Tensor:
    """
    x: (batch, dim) 
    y: (batch, dim)
    """
    x_max = torch.max(x, dim=-1, keepdim=True) # (batch, 1)
    x_uif = x - x_max # (batch, dim)

    x_exp = torch.exp(x_uif) # (batch, dim)
    x_sum = torch.sum(x_exp, dim=-1, keepdim=True) # (batch, 1)

    return x_exp / x_sum # (batch, dim)
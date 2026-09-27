import torch
from torch import Tensor
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "TILE_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "TILE_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, 
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TILE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, TILE_K):
        offs_k = tl.arange(0, TILE_K) + k

        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A = tl.load(A_ptrs, mask=A_mask, other=0.0)

        B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        B_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        B = tl.load(B_ptrs, mask=B_mask, other=0.0)

        acc += tl.dot(A, B)

    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    assert len(a.shape) == 2 and len(b.shape) == 2
    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2
    K = K1

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_kernel[grid](
        a, b, c, 
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c



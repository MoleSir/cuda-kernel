import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    c = a + b
    tl.store(C_ptr, c, mask=mask)


def vector_add(a: Tensor, b: Tensor, BLOCK_SIZE: int) -> Tensor:
    assert a.is_cuda
    assert b.is_cuda
    assert a.shape == a.shape

    out = torch.empty_like(a)
    N = out.numel()

    grid = lambda meta: (N / meta['BLOCK_SIZE'])
    vector_add[grid](a, b, out, N, BLOCK_SIZE=BLOCK_SIZE)

    return out
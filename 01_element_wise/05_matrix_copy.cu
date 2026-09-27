#include "include/common.cuh"

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCK = 1024;

__global__ void matrix_copy(const float* __restrict__ A, float* __restrict__ B, int N) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const size_t N4 = N / 4;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    
    for (size_t i = tid; i < N4; i += stride) {
        B4[i] = __ldg(A4 + i);
    }

    for (size_t i = N4 * 4 + tid; i < N; i += stride) {
        B[i] = A[i];
    }
}

void matrix_copy(const float* A, float* B, int N) {
    const size_t NN = N * N;
    int num_blocks = CEIL(NN / 4, BLOCK_SIZE);
    if (num_blocks < 1) {
        num_blocks = 1;
    }
    if (num_blocks > MAX_BLOCK) {
        num_blocks = MAX_BLOCK;
    }
    matrix_copy<<<num_blocks, BLOCK_SIZE>>>(A, B, N);
}
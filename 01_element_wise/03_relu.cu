#include "include/common.cuh"

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCKS = 1024;

__device__ __forceinline__ float4 relu4(float4 v) {
    return make_float4(fmaxf(v.x, 0.f), fmaxf(v.y, 0.f), fmaxf(v.z, 0.f), fmaxf(v.w, 0.f));
}

__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    for (size_t i = tid; i < N / 4; i += stride) {
        output4[i] = relu4(input4[i]);
    }

    for (size_t i = N + tid; i < N; i += stride) {
        output[i] = fmaxf(input[i], 0.f);
    }
}


void relu(const float* input, float* output, int N) {
    size_t num_block = CEIL(N / 4, BLOCK_SIZE);
    if (num_block < 1)
        num_block = 1;
    if (num_block > MAX_BLOCKS) {
        num_block = MAX_BLOCKS;
    }
    relu_kernel<<<num_block, BLOCK_SIZE>>>(input, output, N);
    cudaDeviceSynchronize();
}
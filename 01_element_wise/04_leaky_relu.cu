#include "include/common.cuh"

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCKS = 1024;

__device__ __forceinline__ float  leaky(float x, float alpha) { return fmaxf(x, alpha * x); }
__device__ __forceinline__ float4 leaky4(float4 v, float alpha) { 
    return make_float4(
        leaky(v.x, alpha), 
        leaky(v.y, alpha), 
        leaky(v.z, alpha), 
        leaky(v.w, alpha)
    ); 
}

__global__ void leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float alpha,
    int N
) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    for (size_t i = tid; i < N / 4; i += stride) {
        output4[i] = leaky4(input4[i], alpha);
    }

    for (size_t i = N + tid; i < N; i += stride) {
        output[i] = leaky(input[i], alpha);
    }
}


void leaky_relu(const float* input, float* output, float alpha, int N) {
    size_t num_block = CEIL(N / 4, BLOCK_SIZE);
    if (num_block < 1)
        num_block = 1;
    if (num_block > MAX_BLOCKS) {
        num_block = MAX_BLOCKS;
    }
    leaky_relu_kernel<<<num_block, BLOCK_SIZE>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}
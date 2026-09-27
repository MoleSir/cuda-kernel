#include "include/common.cuh"

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCK = 1024;

__device__ __forceinline__ float _sigmoid(const float x) {
    return 1.0f / (1.0f + __expf(-x)); 
}

__device__ __forceinline__ float _silu(const float x) {
    return x * _sigmoid(x);
}

__device__ __forceinline__ float4 _silu(const float4 v) {
    return make_float4(_silu(v.x), _silu(v.y), _silu(v.z), _silu(v.w));
}

__global__ void silu_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int N4 = N / 4;
    
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    for (int i = tid; i < N4; i += stride) {
        output4[i] = _silu(input4[i]);
    }

    for (int i = N4 * 4 + tid; i < N; i += stride) {
        output[i] = _silu(input[i]);
    }
}

void silu(const float* input, float* output, int N) {
    const int N4 = N / 4; // 每个 thread 处理 float4
    int num_blocks = CEIL(N4, BLOCK_SIZE);
    if (num_blocks < 1) {
        num_blocks = 1;
    } 
    if (num_blocks > MAX_BLOCK) {
        num_blocks = MAX_BLOCK;
    }
    silu_kernel<<<num_blocks, BLOCK_SIZE>>>(input, output, N);
}

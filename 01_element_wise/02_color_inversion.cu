#include "include/common.cuh"
#include "include/bench.cuh"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>


constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__device__ __forceinline__ uchar4 invert_color(uchar4 v) {
    return make_uchar4(255 - v.x, 255 - v.y, 255 - v.z, v.w); 
}

__global__ void color_invert_kernel(unsigned char* __restrict__ image, int n_pix) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    uchar4* p4 = reinterpret_cast<uchar4*>(image);
    for (size_t i = tid; i < (size_t)n_pix; i += stride) {
        p4[i] = invert_color(p4[i]);
    }
} 

void invert_color(unsigned char* image, int width, int height) {
    const int N_PIX = width * height;
    int blocks = CEIL(N_PIX, THREADS);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > MAX_BLOCKS) {
        blocks = MAX_BLOCKS;
    }

    color_invert_kernel<<<blocks, THREADS>>>(image, N_PIX);
    cudaDeviceSynchronize();
}

static void invert_cpu(unsigned char* img, int n_pix) {
    for (int i = 0; i < n_pix; ++i) {
        img[i * 4 + 0] = 255 - img[i * 4 + 0];
        img[i * 4 + 1] = 255 - img[i * 4 + 1];
        img[i * 4 + 2] = 255 - img[i * 4 + 2];
    }
}

static void test(int w, int h) {
    const int n = w * h * 4;
    std::vector<unsigned char> in(n);
    for (int i = 0; i < n; ++i)
        in[i] = (unsigned char)(i * 17);

    std::vector<unsigned char> ref = in;
    invert_cpu(ref.data(), w * h);

    unsigned char* d = nullptr;
    cudaMalloc(&d, n);
    cudaMemcpy(d, in.data(), n, cudaMemcpyHostToDevice);
    invert_color(d, w, h);
    std::vector<unsigned char> got(n);
    cudaMemcpy(got.data(), d, n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        if (got[i] != ref[i]) {
            printf("FAIL color %dx%d  [%d]  %d vs %d\n", w, h, i, got[i], ref[i]);
            std::exit(1);
        }
    }
    printf("OK  color %dx%d\n", w, h);
    cudaFree(d);
}

int main() {
    test(1, 1);
    test(2, 3);
    test(17, 19);

    const int w = 2048, h = 2048;
    const int n = w * h * 4;
    unsigned char* d = nullptr;
    cudaMalloc(&d, n);
    cudaMemset(d, 1, n);
    float ms = time_ms([&] { invert_color(d, w, h); }, 20, 40);
    printf("color %dx%d   %.3f ms  %.0f GB/s\n", w, h, ms, 2.0 * n / (ms * 1e6));
    cudaFree(d);
}

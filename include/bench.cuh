#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <vector>
#include <functional>
#include <cuda_runtime.h>

template <typename F>
float time_ms(F f, size_t warmup = 3, size_t runs = 10) {
    // 热身
    for (size_t i = 0; i < warmup; ++i) {
        f();
    }

    // 保证之前的 stream 调用完成
    cudaDeviceSynchronize();

    // 构建事件 a/b
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    
    // 将 a 提交 stream
    cudaEventRecord(a);
    // 执行 runs 次 f()
    for (size_t i = 0; i < runs; ++i) {
        f();
    }
    // 将 b 提交 stream
    cudaEventRecord(b);
    // [a, f, f, f, f.., f, b]

    // 等待 b 完成，即所有的 f 都完成
    cudaEventSynchronize(b);

    // 计算两个事件的间隔
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);

    return ms / runs;
}

template <typename F>
float time_ms_median(F f, size_t warmup = 5, size_t trials = 21) {
    for (size_t i = 0; i < warmup; ++i) {
        f();
    }
    cudaDeviceSynchronize();

    auto samples = std::vector<float>(trials);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    for (size_t t = 0; t < trials; ++t) {
        cudaEventRecord(a);
        f();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&samples[t], a, b);
    }

    cudaEventDestroy(a);
    cudaEventDestroy(b);

    std::nth_element(samples.begin(), samples.begin() + trials / 2, samples.end());
    return samples[trials / 2];
}

template <typename Cpu, typename Gpu>
void bench(
    const char* name, 
    std::initializer_list<const std::vector<float>*> ins, 
    size_t n_out,
    Cpu cpu,
    Gpu gpu,
    double bytes = 0,
    double flops = 0,
    bool check = true
) {
    // 准备输入
    std::vector<const float*> host_in{}, device_in{};
    std::vector<float*> device_in_{};
    for (auto* v : ins) {
        host_in.push_back(v->data());

        // 拷贝到 gpu
        float* p = nullptr;
        cudaMalloc(&p, v->size() * sizeof(float));
        cudaMemcpy(p, v->data(), v->size() * sizeof(float), cudaMemcpyHostToDevice);
        
        device_in.push_back(p);
        device_in_.push_back(p);
    }

    // 准备输出
    std::vector<float> ref(n_out), got(n_out);
    float* d_out = nullptr;
    cudaMalloc(&d_out, n_out * sizeof(float));

    if (check) {
        cpu(host_in.data(), ref.data());
        gpu(device_in.data(), d_out);
        // 将 gpu 结果 copy 到 got 
        cudaMemcpy(got.data(), d_out, n_out * sizeof(float), cudaMemcpyDeviceToHost);

        // 比较 got / ref
        for (size_t i = 0; i < n_out; ++i) {
            if (std::fabs(got[i] - ref[i]) > 1e-4f) {
                printf("FAIL %s  [%zu]  %g vs %g\n", name, i, got[i], ref[i]);
                std::exit(1);
            }
        }
        printf("OK  %s\n", name);
    } else {
        gpu(device_in.data(), d_out);
        printf("RUN %s\n", name);
    }

    if (bytes > 0 || flops > 0) {
        float ms = time_ms_median([&] () { gpu(device_in.data(), d_out); });
        printf("    %.3f ms (median)", ms);
        if (flops > 0)
            printf("  %.2f TFLOPS", flops / (ms * 1e9));
        if (bytes > 0)
            printf("  %.0f GB/s", bytes / (ms * 1e6));
        printf("\n");
    }

    for (float* p : device_in_) {
        cudaFree(p);
    }
    cudaFree(d_out);
}
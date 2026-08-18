#include <cuda_runtime.h>

#include <cstdio>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t error = (call);                                      \
        if (error != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                  \
                         cudaGetErrorString(error));                           \
            return 2;                                                          \
        }                                                                      \
    } while (0)

__global__ void write_value(int* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *output = 42;
    }
}

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 1) {
        std::fprintf(stderr, "No CUDA device is visible.\n");
        return 3;
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    int* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_output, sizeof(int)));
    write_value<<<1, 1>>>(device_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int host_output = 0;
    CUDA_CHECK(cudaMemcpy(&host_output, device_output, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_output));

    if (host_output != 42) {
        std::fprintf(stderr, "Unexpected kernel result: %d\n", host_output);
        return 4;
    }

    std::printf("PASS: device=%s compute=%d.%d value=%d\n", properties.name,
                properties.major, properties.minor, host_output);
    return 0;
}

#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        output[index] = input[index] / (1.0f + __expf(-input[index]));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

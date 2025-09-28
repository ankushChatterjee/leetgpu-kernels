#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    const int index1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int index2 = index1 + halfN;

    if (index1 < halfN && index2 < N) {
        const int index1Silu = input[index1] / (1.0f + __expf(-input[index1]));
        output[index1] = input[index2] * index1Silu;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
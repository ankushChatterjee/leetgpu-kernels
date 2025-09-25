#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__constant__ float shKernel[2048];

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size, int output_size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tileSize = blockDim.x + kernel_size - 1;
        
    extern __shared__ float shInput[];
    for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
        int inputIndex = blockIdx.x * blockDim.x + i;
        if (inputIndex < input_size) {
            shInput[i] = input[inputIndex];
        }
    }
    __syncthreads();

    
    if (index < output_size) {
        float value = 0.0f;
        for (int i = 0; i < kernel_size; i++) {
            value += shInput[threadIdx.x + i] * shKernel[i];
        }
        output[index] = value;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpyToSymbol(shKernel, kernel, kernel_size * sizeof(float));
    size_t sharedSize = (threadsPerBlock + kernel_size - 1) * sizeof(float);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(input, kernel, output, input_size, kernel_size, output_size);
    cudaDeviceSynchronize();
}
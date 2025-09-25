#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    const int xIn = blockIdx.x * TILE_DIM + threadIdx.x;
    const int yIn = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    for (int i=0; i < TILE_DIM; i+=BLOCK_ROWS) {
        int yRead = yIn + i;
        if (xIn < cols && yRead < rows) {
            tile[threadIdx.y + i][threadIdx.x] = input[yRead * cols + xIn];
        }
    }
    __syncthreads();
    
    const int xOut = blockIdx.y * TILE_DIM + threadIdx.x;
    const int yOut = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int i=0; i < TILE_DIM; i+=BLOCK_ROWS) {
        int yWrite = yOut + i;
        if (xOut < rows && yWrite < cols) {
            output[yWrite * rows + xOut] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE_DIM, BLOCK_ROWS);
    dim3 blocksPerGrid((cols + TILE_DIM - 1) / TILE_DIM,
                       (rows + TILE_DIM - 1) / TILE_DIM);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
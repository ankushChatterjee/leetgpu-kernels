#include <cuda_runtime.h>

#define COARSE_FACTOR 8

#define TILES_A_ROWS 64
#define TILES_A_COLS 8

#define TILES_B_COLS 64

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int tx = threadIdx.x;  // Thread x-coordinate within block
    
    // Get block indices within the grid
    const int bx = blockIdx.x;   // Block x-coordinate within grid
    const int by = blockIdx.y;   // Block y-coordinate within grid

    const int aTy = tx / TILES_A_COLS;
    const int aTx = tx % TILES_A_COLS;

    const int bTy = tx / TILES_B_COLS;
    const int bTx = tx % TILES_B_COLS;

    const int row = TILES_A_ROWS * by + COARSE_FACTOR*(tx/TILES_B_COLS);
    const int col = TILES_B_COLS * bx + (tx%TILES_B_COLS);

    const int tiles = ceil((float)N/TILES_A_COLS);

    __shared__ float shA[TILES_A_ROWS][TILES_A_COLS];
    __shared__ float shB[TILES_A_COLS][TILES_B_COLS];

    float value[COARSE_FACTOR] = {0.0f};
    for (int tile = 0; tile < tiles; tile++) {
        int aRow = by * TILES_A_ROWS + aTy;
        int aCol = tile * TILES_A_COLS + aTx;

        if (aRow < M && aCol < N) {
            shA[aTy][aTx] = A[aRow * N + aCol];
        } else {
            shA[aTy][aTx] = 0.0f;
        }

        int bRow = tile * TILES_A_COLS + bTy;
        int bCol = bx * TILES_B_COLS + bTx;
        
        if (bRow < N && bCol < K) {
            shB[bTy][bTx] = B[bRow * K + bCol];
        } else {
            shB[bTy][bTx] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < TILES_A_COLS; k++) {
            float bValRegister = shB[k][bTx];

            for (int c = 0; c < COARSE_FACTOR; c++) {
                value[c] += shA[bTy * COARSE_FACTOR + c][k] * bValRegister;
            }
        }

        __syncthreads();
    }
    
    for (int c = 0; c < COARSE_FACTOR; c++) {
        if (row + c < M && col < K) {
            C[(row + c) * K + col] = value[c];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILES_A_ROWS * TILES_B_COLS / COARSE_FACTOR);
    dim3 blocksPerGrid(ceil(K/(float)TILES_B_COLS), ceil(M/(float)TILES_A_ROWS));
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

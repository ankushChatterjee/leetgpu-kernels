#define COARSE_FACTOR_X 8
#define COARSE_FACTOR_Y 8

#define TILES_A_ROWS 128
#define TILES_A_COLS 16

#define TILES_B_COLS 128

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int threadPerBlocks = TILES_A_ROWS * TILES_B_COLS / COARSE_FACTOR_X / COARSE_FACTOR_Y;

    const int tx = threadIdx.x;  // Thread x-coordinate within block

    const int bx = blockIdx.x;   // Block x-coordinate within grid
    const int by = blockIdx.y;   // Block y-coordinate within grid

    const int aTy = tx / TILES_A_COLS;
    const int aTx = tx % TILES_A_COLS;
    const int strideA = threadPerBlocks / TILES_A_COLS;

    const int bTy = tx / TILES_B_COLS;
    const int bTx = tx % TILES_B_COLS;
    const int strideB = threadPerBlocks / TILES_B_COLS;

    const int tileRow = COARSE_FACTOR_Y * (tx / (TILES_B_COLS / COARSE_FACTOR_X));
    const int tileCol = COARSE_FACTOR_X * (tx % (TILES_B_COLS / COARSE_FACTOR_X));

    __shared__ float shA[TILES_A_ROWS][TILES_A_COLS];
    __shared__ float shB[TILES_A_COLS][TILES_B_COLS];

    float values[COARSE_FACTOR_Y][COARSE_FACTOR_X] = {0.0f};
    float registerA[COARSE_FACTOR_Y] = {0.0f};
    float registerB[COARSE_FACTOR_X] = {0.0f};

    for (int tile = 0; tile < ceil((float)N/TILES_A_COLS); tile++) {
        for (int load_offset = 0; load_offset < TILES_A_ROWS; load_offset+=strideA) {
            int aRow = by * TILES_A_ROWS + aTy + load_offset;
            int aCol = tile * TILES_A_COLS + aTx;
            if (aRow < M && aCol < N) {
                shA[aTy + load_offset][aTx] = A[aRow * N + aCol];
            } else {
                shA[aTy + load_offset][aTx] = 0.0f;
            }
        }
        
        for (int load_offset = 0; load_offset < TILES_A_COLS; load_offset+=strideB) {
            int bRow = tile * TILES_A_COLS + bTy + load_offset;
            int bCol = bx * TILES_B_COLS + bTx;
            if (bRow < N && bCol < K) {
                shB[bTy + load_offset][bTx] = B[bRow * K + bCol];
            } else {
                shB[bTy + load_offset][bTx] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < TILES_A_COLS; k++) {
            for (int i = 0; i < COARSE_FACTOR_Y; i++) {
                registerA[i] = shA[tileRow + i][k];
            }

            for (int i = 0; i < COARSE_FACTOR_X; i++) {
                registerB[i] = shB[k][tileCol + i];
            }

            for (int i = 0; i < COARSE_FACTOR_Y;i++) {
                for (int j = 0; j < COARSE_FACTOR_X;j++) {
                    values[j][i] += registerA[i] * registerB[j];
                }
            }
        }

        __syncthreads();

    }

    for (int i = 0; i < COARSE_FACTOR_Y; i++) {
        for (int j = 0; j < COARSE_FACTOR_X; j++) {
            const int cRow = by * TILES_A_ROWS + tileRow + i;
            const int cCol = bx * TILES_B_COLS + tileCol + j;
            if (cRow < M && cCol < K) {
                C[cRow * K + cCol] = values[j][i];
            }
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILES_A_ROWS * TILES_B_COLS / COARSE_FACTOR_X / COARSE_FACTOR_Y);
    dim3 blocksPerGrid(ceil(K/(float)TILES_B_COLS), ceil(M/(float)TILES_A_ROWS));
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

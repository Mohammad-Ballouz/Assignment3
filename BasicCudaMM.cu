#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void MatrixMultiplication(float* A, float* B, float* C, int height, int width, int x)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < x) {
        float Pvalue = 0;
        for (int k = 0; k < width; k++) {
            Pvalue += A[row * width + k] * B[k * x + col];
        }
        C[row * x + col] = Pvalue;
    }
}

void print_matrix(float* P, int x, int y) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            printf("%f ", P[i * y + j]);
        }
        printf("\n");
    }
}

int main() {
    int A = 1024;
    int B = 512; 
    int C = 2048;  
    float *w1 = (float*)malloc(A * B * sizeof(float));
    float *w2 = (float*)malloc(B * C * sizeof(float));
    float *w3 = (float*)malloc(A * C * sizeof(float));

    
    srand(time(NULL));
    for (int i = 0; i < A * B; i++) {
        w1[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < B * C; i++) {
        w2[i] = rand() / (float)RAND_MAX;
    }

   
    float *M, *N, *P;
    cudaMalloc((void**)&M, A * B * sizeof(float));
    cudaMalloc((void**)&N, B * C * sizeof(float));
    cudaMalloc((void**)&P, A * C * sizeof(float));

    
    cudaMemcpy(M, w1, A * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N, w2, B * C * sizeof(float), cudaMemcpyHostToDevice);

    
    int size = 16; 
    dim3 threadsPerBlock(size, size);
    dim3 numBlocks(ceil(C / (float)size), ceil(A / (float)size));
    
    cudaEvent_t start, end;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    MatrixMultiplication<<<numBlocks, threadsPerBlock>>>(M, N, P, A, B, C);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

  
    cudaMemcpy(w3, P, A * C * sizeof(float), cudaMemcpyDeviceToHost);

  
    printf("Elapsed time: %f ms\n", elapsed_time);

    
    free(w1);
    free(w2);
    free(w3);
    cudaFree(M);
    cudaFree(N);
    cudaFree(P);

    return 0;
}

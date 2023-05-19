#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void MMK(float* A, float* B, float* P,int r , int c , int astro )
{
    __shared__ float w1[16][16];
    __shared__ float w2[16][16];
    int bx = blockIdx.x; 
    int tx = threadIdx.x; 
    int by = blockIdx.y;
    int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Pvalue = 0;
    for( int i = 0; i < astro -1 ) / 17 ; i++ ){ 
        if(row<r && tx* 16+tx<c)
        {
            w1[ty][tx] = A[row * c + i * 16 + tx];
        }else{
            w1[ty][tx] = 0.0;
        }
        if (i*16+ty < c && Col < astro ) {
            w2[ty][tx] = B[(i*16 + ty) * astro+ Col];
        }else{
            w2[ty][tx] = 0.0;
        }

        __syncthreads();

        if(row<r && Col < astro){
            for(int i = 0; i < 16; i++){
                Pvalue+= w1[ty][i] * w2[i][tx];
                __syncthreads();
            }
            
        }
    }
    if(row<astro&& Col < astro)
        P[row*astro + Col] = Pvalue;
}

void print_matrix(float* matrix, int B, int A) {
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < A; j++) {
            printf("%f ", matrix[i * A + j]);
        }
        printf("\n");
    }
}

int main()
{
    int r = 1024; 
    int c = 512; 
    int astro = 2048; 
    int x = r * c; 
    int z = c * astro; 
    int y = r * astro;

    float *as = (float*)malloc(x* sizeof(float));
    float *as2 = (float*)malloc(z * sizeof(float));
    float *cl = (float*)malloc(y * sizeof(float));

    for (int i = 0; i < r * c; i++) {
        as[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < c * astro; i++) {
        as2[i] = rand() / (float)RAND_MAX;
    }

    float *t1, *t2, *t;
    cudaMalloc((void**)&t1, x*sizeof(float));
    cudaMalloc((void**)&t2,z *sizeof(float));
    cudaMalloc((void**)&t, y *sizeof(float));

    cudaMemcpy(t1, as, x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(t2, as2, z* sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 blocks((c + blockDim.x - 1)/ blockDim.x, (r + blockDim.y -1) / blockDim.y );
    
    cudaEvent_t start, end;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    MMK<<<blocks, blockDim>>>(t1, t2, t,r,c,astro);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaMemcpy(cl, t, y * sizeof(float), cudaMemcpyDeviceToHost);

    
    print_matrix(cl, r, astro);

    printf("Elapsed time: %f ms\n", elapsed_time);

    cudaFree(t1);
    cudaFree(t2);
    cudaFree(t);

    return 0;
}

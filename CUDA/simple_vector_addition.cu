//Author: Ryan Silverberg
//Source: https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/
//Learned: Memory
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define N 100

__global__ void vector_add(float *out, float *a, float *b, int n){
    for(int i = 0; i < n; i++){
        out[i] = a[i]+b[i];
    }
}

int main(){
    float *a, *b, *out;
    float *d_a;

    a = (float *)malloc(sizeof(float)*N);

    cudaMalloc((void**)&d_a, sizeof(float)*N);
    //allocates memory on the device (GPU) for a

    cudaMemcpy(d_a,a,sizeof(float)*N,cudaMemcpyHostToDevice);
    //transfers data from Host (CPU) to device (GPU) of size d_a along with a

    vector_add<<<1,1>>>(out, a, b, N);
    cudaFree(d_a);
    free(a);
    return 0;
}
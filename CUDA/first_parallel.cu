//Author: Ryan Silverberg
//Source: https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial02/


/**
 * CUDA uses a operator <<<>>> which is called a thread block.
 * This tells the GPU how many threads to use as it is much better
 * at paralle processing. The Kernel can launch multiple thread blocks
 * these are organized into a "grid" structure.
 * 
 * The structure is <<<M,T>>>.
 * This indicates that the kernel launches with a grid of M thread blocks
 * Each thread block has T parallel threads.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

/** Tasks:
 * 1. Copy vector_add.cu to vector_add_thread.cu
 * 2. Parallelize vector_add() using a thread block with 256 threads.
*/

//Using simple_vector_addition.cu to explain this
#define N 1000000000

//change this to work in strides, so each thread works simultaneously
__global__ void vector_add(float *out, float *a, float *b, int n){
    int index = threadIdx.x;
    int stride = blockDim.x;
    for(int i = index; i < n; i += stride){
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

    vector_add<<<1,256>>>(out, a, b, N);
    cudaFree(d_a);
    free(a);
    return 0;
} //Soln is under file vector_add_thread.cu

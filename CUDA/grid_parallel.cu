//Author: Ryan Silverberg
//Source: https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial02/


/**
 * this program now will utlize multiple thread blocks, each containing 255 threads.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

/** Tasks:
1. Copy vector_add.cu to vector_add_grid.cu
2. Parallelize vector_add() using multiple thread blocks.
3. Handle case when N is an arbitrary number.
HINT: Add a condition to check that the thread work within the acceptable array index range.
5. Compile and profile the program
*/

//NOT DONE

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

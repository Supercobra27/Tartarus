
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>

/**
 * @author Ryan Silverberg
 * @student_id: 20342023
*/

#define MEMCHECK(call) if(!call) exit(EXIT_FAILURE);

typedef float* mat_t;

using namespace std;
using namespace std::chrono;

mat_t cpuMatrix;

// One Block, Many Threads
__global__ void matmulGPU(mat_t N, mat_t M, mat_t P, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float Pv = 0;
        for (int k = 0; k < n; ++k) {
            Pv += M[row * n + k] * N[k * n + col];
        }
        P[row * n + col] = Pv;
    }
}

__global__ void matmulGPU_1B(mat_t M, mat_t N, mat_t P, int n) {
    for (int i = 0; i < n * n; i++) {
        P[i] = 0.0f;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                P[i * n + j] += M[i * n + k] * N[k * n + j];
            }
        }
    }
}

void matmulCPU(mat_t M, mat_t N, mat_t P, int n) {
    for (int i = 0; i < n * n; i++) {
        P[i] = 0.0f;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                P[i * n + j] += M[i * n + k] * N[k * n + j];
            }
        }
    }
}

// make this single precision
void initMatrix(mat_t m, int n) {
    srand(374);
    for (int i = 0; i < n * n; i++) {
        m[i] = (float)(rand()) / (float)(rand());
    }
}

void printMatrix(mat_t m, int n) {
    for (int i = 0; i < n * n; i++) {
        printf("%.2f ", m[i]);
        if ((i % n) == n - 1) printf("\n");
    }
}

cudaError_t compareMatrices(mat_t CPU, mat_t GPU, int n, float error) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(CPU[i] - GPU[i]) > error) {
            printf("Mismatch at element %d: CPU=%f, GPU=%f\n", i, CPU[i], GPU[i]);
            return cudaErrorInvalidValue;
        }
    }
    return cudaSuccess;
}

#define TOLERANCE 1e-6

typedef struct _transfer {
    float toCPU;
    float toGPU;
}transfer;

transfer transfer_time(int n) {

    size_t size = n * n * sizeof(float); // Size of matrix in bytes
    cudaEvent_t toGPUstart, toGPUend; // Host -> Device
    cudaEvent_t toCPUstart, toCPUend; // Device -> Host

    mat_t M, N; // CPU
    mat_t gM, gN; // GPU

    cudaEventCreate(&toGPUstart);
    cudaEventCreate(&toGPUend);
    cudaEventCreate(&toCPUstart);
    cudaEventCreate(&toCPUend);

    cudaDeviceSynchronize();

    cudaMallocHost((void**)&M, size);
    cudaMallocHost((void**)&N, size);
    cudaMalloc((void**)&gM, size);
    cudaMalloc((void**)&gN, size);

    initMatrix(M, n);
    initMatrix(N, n);

    transfer ret;

    // Host --> Device

    cudaEventRecord(toGPUstart, 0);
    cudaMemcpy(gM, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gN, N, size, cudaMemcpyHostToDevice);
    cudaEventRecord(toGPUend, 0);
    cudaEventSynchronize(toGPUend);
    cudaEventElapsedTime(&ret.toGPU, toGPUstart, toGPUend);

    // Device --> Host

    cudaEventRecord(toCPUstart, 0);
    cudaMemcpy(M, gM, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(N, gN, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(toCPUend, 0);
    cudaEventSynchronize(toCPUend);
    cudaEventElapsedTime(&ret.toCPU, toCPUstart, toCPUend);


    cudaFree(gM);
    cudaFree(gN);
    cudaFreeHost(M);
    cudaFreeHost(N);

    cudaEventDestroy(toGPUstart);
    cudaEventDestroy(toGPUend);
    cudaEventDestroy(toCPUstart);
    cudaEventDestroy(toCPUend);

    cudaDeviceReset();

    return ret;
}

float single_thread_time(int n) {
    float ret;
    size_t size = n * n * sizeof(float); // Size of matrix in bytes
    cudaEvent_t toGPUstart, toGPUend; // for Block Calculations

    mat_t M, N, P; // CPU
    mat_t gM, gN, gP; // GPU

    cudaEventCreate(&toGPUstart);
    cudaEventCreate(&toGPUend);

    cudaDeviceSynchronize();

    cudaMallocHost((void**)&M, size);
    cudaMallocHost((void**)&N, size);
    cudaMallocHost((void**)&P, size);
    cudaMalloc((void**)&gM, size);
    cudaMalloc((void**)&gN, size);
    cudaMalloc((void**)&gP, size);

    initMatrix(M, n);
    initMatrix(N, n);

    // Host --> Device

    cudaMemcpy(gM, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gN, N, size, cudaMemcpyHostToDevice);

    int numBlocks = n / n;
    if (n % n) numBlocks++;

    dim3 dimGrid(1);
    dim3 dimBlock(1);

    cudaEventRecord(toGPUstart, 0);
    matmulGPU_1B << <dimGrid, dimBlock >> > (gM, gN, gP, n);
    cudaEventRecord(toGPUend, 0);
    cudaEventSynchronize(toGPUend);
    cudaEventElapsedTime(&ret, toGPUstart, toGPUend);

    // Device --> Host

    cudaMemcpy(P, gP, size, cudaMemcpyDeviceToHost);

    if (!compareMatrices(cpuMatrix, P, n, TOLERANCE)) printf("Test 1 PASSED\n");

    cudaFree(gM);
    cudaFree(gN);
    cudaFree(gP);
    cudaFreeHost(M);
    cudaFreeHost(N);
    cudaFreeHost(P);

    cudaEventDestroy(toGPUstart);
    cudaEventDestroy(toGPUend);

    cudaDeviceReset();

    return ret;
}

int cpu_matmul_time(int n) {

    size_t size = n * n * sizeof(float);
    cpuMatrix = new float[n * n];

    mat_t M, N, P;

    cudaMallocHost((void**)&M, size);
    cudaMallocHost((void**)&N, size);
    cudaMallocHost((void**)&P, size);

    initMatrix(M, n);
    initMatrix(N, n);


    auto cpu_start = high_resolution_clock::now();
    matmulCPU(M, N, P, n);
    auto cpu_end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(cpu_end - cpu_start);

    cpuMatrix = P;

    cudaFreeHost(M);
    cudaFreeHost(N);
    cudaFreeHost(P);

    return duration.count();
}

float block_change_time(int n, int b) {

    float ret;
    size_t size = n * n * sizeof(float); // Size of matrix in bytes
    cudaEvent_t toGPUstart, toGPUend; // for Block Calculations

    mat_t M, N, P; // CPU
    mat_t gM, gN, gP; // GPU

    cudaEventCreate(&toGPUstart);
    cudaEventCreate(&toGPUend);

    cudaDeviceSynchronize();

    cudaMallocHost((void**)&M, size);
    cudaMallocHost((void**)&N, size);
    cudaMallocHost((void**)&P, size);
    cudaMalloc((void**)&gM, size);
    cudaMalloc((void**)&gN, size);
    cudaMalloc((void**)&gP, size);

    initMatrix(M, n);
    initMatrix(N, n);

    // Host --> Device

    cudaMemcpy(gM, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gN, N, size, cudaMemcpyHostToDevice);

    int numBlocks = n / b;
    if (n % b) numBlocks++;

    dim3 dimGrid(numBlocks, numBlocks);
    dim3 dimBlock(b, b);

    cudaEventRecord(toGPUstart, 0);
    matmulGPU << <dimGrid, dimBlock >> > (gM, gN, gP, n);
    cudaEventRecord(toGPUend, 0);
    cudaEventSynchronize(toGPUend);
    cudaEventElapsedTime(&ret, toGPUstart, toGPUend);

    // Device --> Host

    cudaMemcpy(P, gP, size, cudaMemcpyDeviceToHost);

    cudaFree(gM);
    cudaFree(gN);
    cudaFree(gP);
    cudaFreeHost(M);
    cudaFreeHost(N);
    cudaFreeHost(P);

    cudaEventDestroy(toGPUstart);
    cudaEventDestroy(toGPUend);

    cudaDeviceReset();

    return ret;

}


typedef struct _timing {
    int width;
    float toCPUtime;
    float toGPUtime;
    float blkGPUtime[5];
    float singleGPUtime;
    int mulCPUtime;

}timing;

timing run_mp(int n) {
    // Part 1: Find Transfer Times
    timing trial_out;

    trial_out.width = n;

    transfer t = transfer_time(n);

    printf("Transfer to GPU took %.2f ms\n", t.toGPU);
    printf("Transfer to CPU took %.2f ms\n", t.toCPU);

    trial_out.toGPUtime = t.toGPU;
    trial_out.toCPUtime = t.toCPU;

    // Part 2: Find Single Threaded Times
    if (!(n == 2048 || n == 4096)) {
        trial_out.mulCPUtime = cpu_matmul_time(n);
        trial_out.singleGPUtime = single_thread_time(n);

        printf("GPU-single-threaded took %.2f ms\n", trial_out.singleGPUtime);
        printf("CPU matrix calculation was %d ms\n", trial_out.mulCPUtime);
    }

    // Part 3: Find Block Execution Times
    int blksiz[5] = { 2, 4, 8, 16, 32 };
    for (int b = 0; b < 5; b++) {
        trial_out.blkGPUtime[b] = block_change_time(n, blksiz[b]);
        printf("Calculation with Block Size %d took %.2f ms\n", blksiz[b], trial_out.blkGPUtime[b]);
    }

    return trial_out;

}
#define NUM_TRIALS 5
int main() {
    int widths[5] = { 256, 512, 1024, 2048, 4096 };
    std::ofstream csvfile("data.csv");
    cpuMatrix = new float;

    timing times;
    for (int t = 0; t < NUM_TRIALS; t++) {
        for (int w = 0; w < 5; w++) {
            times = run_mp(widths[w]);
            csvfile << times.width << ',' << times.toGPUtime << ',' << times.toCPUtime << ',' << times.mulCPUtime << ',' << times.singleGPUtime <<
                times.blkGPUtime[0] << ',' <<
                times.blkGPUtime[1] << ',' <<
                times.blkGPUtime[2] << ',' <<
                times.blkGPUtime[3] << ',' <<
                times.blkGPUtime[4] << endl;
        }
    }

    delete cpuMatrix;
    csvfile.close();

    return 0;
}

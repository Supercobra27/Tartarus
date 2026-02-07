
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <vector>

/**
 * @author Ryan Silverberg
 * @student_id: 20342023
*/

#define GB (1024*1024*1024)
#define KB (1024)

void queryDevice(std::vector<cudaDeviceProp>& props) {

    int nd = 0;

    cudaGetDeviceCount(&nd);

    for (int d = 0; d < nd; d++)
    {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, d);
        props.push_back(dp);

        int coresPerSM;
        switch (dp.major) {
        case 2:  coresPerSM = (dp.minor == 1) ? 48 : 32; break;
        case 3:  coresPerSM = 192; break;
        case 5:  coresPerSM = 128; break;
        case 6:  coresPerSM = (dp.minor == 0) ? 64 : 128; break;
        case 7:  coresPerSM = 64; break;
        case 8:  coresPerSM = 128; break;
        default: coresPerSM = 64; break;
        }

        printf("Major: %d\t Minor: %d\n", dp.major, dp.minor);
        printf("GPU Name: %s\n", dp.name);
        printf("Clock Rate: %d\n", dp.clockRate);
        printf("Streaming Multiprocessors: %d\n", dp.multiProcessorCount);
        printf("CUDA Cores: %d\n", (coresPerSM * dp.multiProcessorCount)); // to fix
        printf("Warp Size: %d\n", dp.warpSize);
        printf("Global Memory: %zu GB\n", dp.totalGlobalMem / GB);

        printf("Constant Memory: %zu KB\n", dp.totalConstMem / KB);
        printf("Memory/Block: %zu KB\n", dp.sharedMemPerBlock / KB);
        printf("Regs/Block: %d\n", dp.regsPerBlock);
        printf("Threads/Block: %d\n", dp.maxThreadsPerBlock);
        printf("MaxDim/Block: %d %d %d\n", dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
        printf("Max Grid Dim: %d %d %d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
    }

}

int main()
{
    cudaError_t cudaStatus;
    std::vector<cudaDeviceProp> deviceProperties;

    queryDevice(deviceProperties);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    printf("Devices Queried: %zu", deviceProperties.size());

    return 0;
}
#include "DSZCudaUtility.h"
#include <stdlib.h>
#include <stdio.h>

void checkError(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		printf("Error (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

}

void queryDevices()
{
	int count;
	cudaError_t err;
	err = cudaGetDeviceCount(&count);
	checkError(err);

	printf("Number of CUDA-capable devices = %d\n", count);

	for (int i = 0; i < count; ++i)
	{
		cudaDeviceProp prop;
		err = cudaGetDeviceProperties(&prop, i);

		printf("**** Device %d : %s ****\n", i, prop.name);
		printf("Clock : %f GHz\n", (float)prop.clockRate/(1024*1024));
		printf("Global memory size : %4.2f GB\n", (float)prop.totalGlobalMem/(1024*1024*1024));
		printf("Constant memory size : %4.2f KB\n", (float)prop.totalConstMem / (1024));
		printf("Shared memory per block : %4.2f KB \n", (float)prop.sharedMemPerBlock/1024);
		printf("Registers per block : %d\n", prop.regsPerBlock);
		printf("Multiprocessors count %d\n", prop.multiProcessorCount);
		printf("Maximum threads per block %d\n", prop.maxThreadsPerBlock);
		printf("Maximum block dimension (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Maximum grid dimension (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Compute capability %d.%d\n", prop.major, prop.minor);
		printf("Device overlap supported : %s\n", prop.deviceOverlap != 0 ? "Yes" : "No");
	}
}

dszCudaMesureInfo_t startMesuring()
{
	dszCudaMesureInfo_t mesureInfo;
	cudaEventCreate(&mesureInfo.start);
	cudaEventCreate(&mesureInfo.stop);
	cudaEventRecord(mesureInfo.start, 0);

	return mesureInfo;
}

float stopMesuring(dszCudaMesureInfo_t mesureInfo)
{
	cudaEventRecord(mesureInfo.stop);
	cudaEventSynchronize(mesureInfo.stop);
	float ret;
	cudaEventElapsedTime(&ret, mesureInfo.start, mesureInfo.stop);
	
	cudaEventDestroy(mesureInfo.start);
	cudaEventDestroy(mesureInfo.stop);
	
	return ret;
}
#include "PrefixSum.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "DSZCudaUtility.h"

#define SIZE 100*SECTION_SIZE
float x[SIZE], cpu_y[SIZE];
float eff_gpu_y[SIZE];



void compare(float *a, float *b, unsigned int size)
{
	FILE *logFile;
	errno_t err = fopen_s(&logFile, "log.txt", "w");
	if (err != 0)
	{
		printf("Unable to create log file!\n");
		return;
	}
	for (unsigned int i = 0; i < size; ++i)
	{
		if (fabs(a[i] - b[i]) > 1e-2)
			fprintf(logFile, "i = %d, CPU = %f, GPU = %f ERROR\n", i, a[i], b[i]);
		else
			fprintf(logFile, "i = %d, CPU = %f, GPU = %f SUCCESS\n", i, a[i], b[i]);
	}

	fclose(logFile);
}

void prefixSum_test()
{
	// **** Initialization of input ****
	for (unsigned int i = 0; i < SIZE; ++i)
	{
		x[i] = ((float)rand())/RAND_MAX;
		//float v = (float)i * 0.01f;
		//x[i] = i;
	}

	// **** Sequential scan ****
	sequential_scan(x, cpu_y, SIZE);

	// **** Work efficient scan ****
	dszCudaMesureInfo_t mesureInfo = startMesuring();
	work_efficient_scan(x, eff_gpu_y, SIZE, SUM);
	float elapsed = stopMesuring(mesureInfo);
	printf("Efficient prefix scan time = %fms\n", elapsed);
	compare(cpu_y, eff_gpu_y, SIZE);

	// **** Work inefficient scan ****
	mesureInfo = startMesuring();
	work_inefficient_scan(x, eff_gpu_y, SIZE);
	elapsed = stopMesuring(mesureInfo);
	printf("Inefficient prefix scan time = %fms\n", elapsed);
	compare(cpu_y, eff_gpu_y, SIZE);
}
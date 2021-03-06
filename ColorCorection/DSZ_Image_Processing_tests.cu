#include "bmp_parser.h"
#include "DSZ_Image_Processing.h"
#include "DSZCudaUtility.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "PrefixSum.h"

float mask[MASK_WIDTH*MASK_WIDTH] =
{
	0, 0, 0,
	0, 1, 0,
	0, 0, 0
};

void convolution_test()
{
	unsigned int width, height;
	Pixel_t* inputPicture = loadPicture("input.bmp", &width, &height);
	size_t size = width * height * sizeof(Pixel_t);
	Pixel_t* outputPicture = (Pixel_t*)malloc(size);
	
	tiled_convolution2D(inputPicture, outputPicture, width, height, mask);
	storePicture("outputTiled.bmp", outputPicture, width, height);

	memset(outputPicture, 0, size);
	simple_convolution2D(inputPicture, outputPicture, width, height, mask);
	storePicture("outputSimple.bmp", outputPicture, width, height);

	free(inputPicture);
	free(outputPicture);
}

void histogram_test()
{
	unsigned int width, height;
	dszCudaMesureInfo_t info = startMesuring();
	Pixel_t* inputPicture = loadPicture("input.bmp", &width, &height);
	size_t size = width * height * sizeof(Pixel_t);
	Pixel_t* outputPicture = (Pixel_t*)malloc(size);

	greyscale(inputPicture, outputPicture, width, height);

	unsigned int histogram[256];
	compute_Image_Histogram(inputPicture, width, height, histogram);

	float probHistogram[256], cumulHistogram[256];
	prob(histogram, probHistogram, 256, width*height);

	work_efficient_scan(probHistogram, cumulHistogram, 256, SUM);
	float min = reduction_op(cumulHistogram, 256, MIN);

	equalize(inputPicture, width, height, cumulHistogram, min, outputPicture);
	float elapsed = stopMesuring(info);
	printf("Elapsed time = %fms\n", elapsed);
	storePicture("outputColorCorection.bmp", outputPicture, width, height);

	free(inputPicture);
	free(outputPicture);
}
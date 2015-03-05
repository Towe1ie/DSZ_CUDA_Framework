#ifndef _DSZ_IMAGE_PROCESSING_H_
#define _DSZ_IMAGE_PROCESSING_H_

#include <cuda_runtime.h>
#include "bmp_parser.h"

#define MASK_WIDTH 3

void simple_convolution2D(Pixel_t *inputPicture, Pixel_t *outPicture, unsigned int width, unsigned int height, float *mask);
void tiled_convolution2D(Pixel_t *inputPicture, Pixel_t *outputPicture, unsigned int width, unsigned int height, const float* __restrict__ mask);
void greyscale(Pixel_t *inputPicture, Pixel_t *outputPicture, unsigned int width, unsigned int height);
void compute_Image_Histogram(Pixel_t *inputPicture, unsigned int width, unsigned int height, unsigned int *histogram);
void equalize(const Pixel_t *inputPicture, unsigned int width, unsigned int height, const float *cdf, unsigned int cdf_min, Pixel_t *outPicture);

void convolution_test();
void histogram_test();
#endif
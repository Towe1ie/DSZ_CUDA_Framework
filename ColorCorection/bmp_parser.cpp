#include "bmp_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Pixel_t *loadPicture(char* fname, unsigned int *out_width, unsigned int *out_height)
{
	FILE* file = 0;
	errno_t err = fopen_s(&file, fname, "rb");
	if (err != 0)
	{
		printf("Error while opening file %s\n", fname);
		return 0;
	}

	BMP_Header_t header;
	BMP_InfoHeader_t infoHeader;
	fread(&header, 1, sizeof(BMP_Header_t), file);
	fread(&infoHeader, 1, sizeof(BMP_InfoHeader_t), file);

	unsigned int width = infoHeader.width, height = infoHeader.height;
	*out_height = height;
	*out_width = width;
	Pixel_t *rawData = (Pixel_t*)malloc(height * width * sizeof(Pixel_t));
	Pixel_t *temp = rawData;
	unsigned int padd = width - (width / 4) * 4;

	fseek(file, header.offset, SEEK_SET);

	unsigned int i;

	for (i = 0; i < height; ++i)
	{
		size_t totalRead = fread(temp, sizeof(Pixel_t), width, file);
		int error = ferror(file);
		int eof = feof(file);
		temp += width;
		fseek(file, padd, SEEK_CUR);
	}

	fclose(file);

	return rawData;
}

void storePicture(char *fname, Pixel_t *pixelArray, unsigned int width, unsigned int height)
{
	BMP_Header_t header;
	BMP_InfoHeader_t infoHeader;
	header.type = 0x4d42;
	header.offset = 0x36;
	header.size = 0x36;
	infoHeader.bits = 24;
	infoHeader.compression = 0;
	infoHeader.height = height;
	infoHeader.width = width;
	infoHeader.importantcolours = 0;
	infoHeader.ncolours = 0;
	infoHeader.planes = 1;
	infoHeader.size = 40;


	unsigned int padd = width - (width / 4) * 4;

	FILE *writeFile = 0;
	errno_t err2 = fopen_s(&writeFile, fname, "wb");
	fwrite(&header, sizeof(BMP_Header_t), 1, writeFile);
	fwrite(&infoHeader, sizeof(BMP_InfoHeader_t), 1, writeFile);
	Byte_t *fillBuff = (Byte_t*)malloc(padd);
	memset(fillBuff, 0, padd);
	Pixel_t *temp = pixelArray;

	for (unsigned int i = 0; i < height; ++i)
	{
		fwrite(temp, sizeof(Pixel_t), width, writeFile);
		temp += width;
		header.size += width + padd;
		infoHeader.imagesize += width + padd;
		fwrite(fillBuff, 1, padd, writeFile);
	}

	fclose(writeFile);
}

Pixel_t* allocSameSize(char *fname, unsigned int *out_width, unsigned int *out_height)
{
	FILE* file = 0;
	errno_t err = fopen_s(&file, fname, "rb");

	BMP_Header_t header;
	BMP_InfoHeader_t infoHeader;
	fread(&header, 1, sizeof(BMP_Header_t), file);
	fread(&infoHeader, 1, sizeof(BMP_InfoHeader_t), file);

	unsigned int width = infoHeader.width, height = infoHeader.height;
	*out_height = height;
	*out_width = width;
	Pixel_t *rawData = (Pixel_t*)malloc(height * width * sizeof(Pixel_t));

	return rawData;
}


#ifndef _BMP_PARSER_H_
#define _BMP_PARSER_H_

typedef unsigned char Byte_t;

typedef struct
{
	Byte_t b, g, r;
} Pixel_t;

#pragma pack(push, 1)
typedef struct
{
	unsigned short type;					/* Magic identifier            */
	unsigned int size;						/* File size in bytes          */
	unsigned short reserved1, reserved2;
	unsigned int offset;					/* Offset to image data, bytes */
} BMP_Header_t;


typedef struct
{
	unsigned int size;               /* Header size in bytes      */
	unsigned int width, height;      /* Width and height of image */
	unsigned short int planes;       /* Number of colour planes   */
	unsigned short int bits;         /* Bits per pixel            */
	unsigned int compression;        /* Compression type          */
	unsigned int imagesize;          /* Image size in bytes       */
	int xresolution, yresolution;    /* Pixels per meter          */
	unsigned int ncolours;           /* Number of colours         */
	unsigned int importantcolours;   /* Important colours         */
} BMP_InfoHeader_t;
#pragma pack(pop)

Pixel_t *loadPicture(char* fname, unsigned int *out_width, unsigned int *out_height);
void storePicture(char *fname, Pixel_t *pixelArray, unsigned int width, unsigned int height);

#endif
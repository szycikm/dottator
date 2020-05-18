#include "defines.h"

#define getRelativeLuminance(pixel) 0.2126*pixel.r + 0.7152*pixel.g + 0.0722*pixel.b

// left half of circle + fill
__device__ void putPixelLeft(uchar* imgOut, dim_t dim, uint xc, uint x, uint y)
{
	if (y >= dim.h) return;
	uint slackY = y * dim.w;

	for (int i = x; i < xc; i++)
	{
		if (i < dim.w)
			imgOut[slackY + i] = 255;
	}
}

// right side of the circle + fill
// notice the one extra pixel on the left - this is to fill the middle
__device__ void putPixelRight(uchar* imgOut, dim_t dim, uint xc, uint x, uint y)
{
	if (y >= dim.h) return;
	uint slackY = y * dim.w;

	for (int i = x; i >= xc; i--)
	{
		if (i < dim.w)
			imgOut[slackY + i] = 255;
	}
}

__device__ void drawCirclePoint(uchar* imgOut, dim_t dim, uint xc, uint yc, uint x, uint y)
{
	putPixelLeft(imgOut, dim, xc, xc-x, yc+y);
	putPixelLeft(imgOut, dim, xc, xc-x, yc-y);
	putPixelLeft(imgOut, dim, xc, xc-y, yc+x);
	putPixelLeft(imgOut, dim, xc, xc-y, yc-x);

	putPixelRight(imgOut, dim, xc, xc+x, yc+y);
	putPixelRight(imgOut, dim, xc, xc+x, yc-y);
	putPixelRight(imgOut, dim, xc, xc+y, yc+x);
	putPixelRight(imgOut, dim, xc, xc+y, yc-x);
}

__device__ void circleBres(uchar* imgOut, dim_t dim, uint xc, uint yc, uint r)
{
	uint x = 0;
	uint y = r;
	int d = 3 - 2 * r;

	// middle line (horizontal)
	putPixelLeft(imgOut, dim, xc, xc-y, yc+x);
	putPixelRight(imgOut, dim, xc, xc+y, yc-x);

	while (y >= x)
	{
		x++;

		if (d > 0)
		{
			y--;
			d = d + 4 * (x - y) + 10;
		}
		else
			d = d + 4 * x + 6;

		drawCirclePoint(imgOut, dim, xc, yc, x, y);
	}
}

// performed by each thread
__global__ void dev_makeDots(uint frameWidth, uint framesW, dim_t dim, float dotScaleFactor, pixel_t* imgIn, uchar* imgOut)
{
	uint frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
	uint offsetPxX = frameWidth * (frameIdx % framesW);
	uint offsetPxY = frameWidth * (frameIdx / framesW);

	// calculate luminance avg for all pixels in frame
	uchar avg;
	uint processedCnt = 1;
	for (uint y = 0; y < frameWidth; y++)
	{
		uint realY = offsetPxY + y;
		if (realY >= dim.h) continue;

		for (uint x = 0; x < frameWidth; x++)
		{
			uint realX = offsetPxX + x;
			if (realX >= dim.w) continue;

			uint pxIdx = realY * dim.w + realX;

			// iterative average
			if (processedCnt == 1)
			{
				avg = getRelativeLuminance(imgIn[pxIdx]);
			}
			else
			{
				avg = (avg * processedCnt + getRelativeLuminance(imgIn[pxIdx])) / (processedCnt + 1);
			}
			processedCnt++;
		}
	}

	uint r = avg * frameWidth / 512 * dotScaleFactor;

	if (r > 0)
		circleBres(imgOut, dim, offsetPxX + frameWidth/2, offsetPxY + frameWidth/2, r);
}

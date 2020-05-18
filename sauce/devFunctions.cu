#include "defines.h"

#define getRelativeLuminance(pixel) 0.2126*pixel.r + 0.7152*pixel.g + 0.0722*pixel.b

// left half of circle + fill
__device__ void putPixelLeft(uchar* imgOut, uint imgH, uint imgW, uint xc, uint x, uint y)
{
	if (y >= imgH) return;
	uint slackY = y * imgW;

	for (int i = x; i < xc; i++)
	{
		if (i < imgW)
			imgOut[slackY + i] = 255;
	}
}

// right side of the circle + fill
// notice the one extra pixel on the left - this is to fill the middle
__device__ void putPixelRight(uchar* imgOut, uint imgH, uint imgW, uint xc, uint x, uint y)
{
	if (y >= imgH) return;
	uint slackY = y * imgW;

	for (int i = x; i >= xc; i--)
	{
		if (i < imgW)
			imgOut[slackY + i] = 255;
	}
}

__device__ void drawCirclePoint(uchar* imgOut, uint imgH, uint imgW, uint xc, uint yc, uint x, uint y)
{
	putPixelLeft(imgOut, imgH, imgW, xc, xc-x, yc+y);
	putPixelLeft(imgOut, imgH, imgW, xc, xc-x, yc-y);
	putPixelLeft(imgOut, imgH, imgW, xc, xc-y, yc+x);
	putPixelLeft(imgOut, imgH, imgW, xc, xc-y, yc-x);

	putPixelRight(imgOut, imgH, imgW, xc, xc+x, yc+y);
	putPixelRight(imgOut, imgH, imgW, xc, xc+x, yc-y);
	putPixelRight(imgOut, imgH, imgW, xc, xc+y, yc+x);
	putPixelRight(imgOut, imgH, imgW, xc, xc+y, yc-x);
}

__device__ void circleBres(uchar* imgOut, uint imgH, uint imgW, uint xc, uint yc, uint r)
{
	uint x = 0;
	uint y = r;
	int d = 3 - 2 * r;

	// middle line (horizontal)
	putPixelLeft(imgOut, imgH, imgW, xc, xc-y, yc+x);
	putPixelRight(imgOut, imgH, imgW, xc, xc+y, yc-x);

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

		drawCirclePoint(imgOut, imgH, imgW, xc, yc, x, y);
	}
}

// performed by each thread
__global__ void dev_makeDots(uint frameWidth, uint framesW, uint imgW, uint imgH, float dotScaleFactor, pixel_t* imgIn, uchar* imgOut)
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
		if (realY >= imgH) continue;

		for (uint x = 0; x < frameWidth; x++)
		{
			uint realX = offsetPxX + x;
			if (realX >= imgW) continue;

			uint pxIdx = realY * imgW + realX;

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
		circleBres(imgOut, imgH, imgW, offsetPxX + frameWidth/2, offsetPxY + frameWidth/2, r);
}

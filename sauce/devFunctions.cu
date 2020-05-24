#include "defines.h"

#define getRelativeLuminance(pixel) 0.2126*pixel.r + 0.7152*pixel.g + 0.0722*pixel.b

// left half of circle + fill
__device__ void putPixelLeft(uchar* imgOut, uint xc, uint x, uint y, uint pixelsWSharedCnt)
{
	uint slackY = y * pixelsWSharedCnt;
	for (int i = x; i < xc; i++)
	{
		imgOut[slackY + i] = 255;
	}
}

// right side of the circle + fill
// notice the one extra pixel on the left - this is to fill the middle
__device__ void putPixelRight(uchar* imgOut, uint xc, uint x, uint y, uint pixelsWSharedCnt)
{
	uint slackY = y * pixelsWSharedCnt;
	for (int i = x; i >= xc; i--)
	{
		imgOut[slackY + i] = 255;
	}
}

// performed by each thread
__global__ void dev_makeDots(uint framesPerThread, uint frameWidth, uint framesW, dim_t dim, float dotScaleFactor, pixel_t* imgIn, uchar* imgOut)
{
	uint frameIdxBase = framesPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
	uint relFrameIdxBase = framesPerThread * threadIdx.x;
	uint yc = frameWidth/2;
	uint pixelsWSharedCnt = framesPerThread * frameWidth * blockDim.x;

	// TODO put variable declarations here
	extern __shared__ uchar s_imgOut[];

	// init shared memory with black
	for (int i = 0; i < pixelsWSharedCnt * frameWidth; i++)
		s_imgOut[i] = 0;

	for (int f = 0; f < framesPerThread; f++)
	{
		uint offsetPxX = frameWidth * ((frameIdxBase + f) % framesW);
		uint offsetPxY = frameWidth * ((frameIdxBase + f) / framesW);

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

		uint relOffsetPxX = frameWidth * (relFrameIdxBase + f);

		uint r = avg * frameWidth / 512 * dotScaleFactor;

		if (r > 0)
		{
			// draw circle bres in shared memry

			uint x = 0;
			uint y = r;
			int d = 3 - 2 * r;
			uint xc = relOffsetPxX + frameWidth/2;

			// middle line (horizontal)
			putPixelLeft(s_imgOut, xc, xc-y, yc+x, pixelsWSharedCnt);
			putPixelRight(s_imgOut, xc, xc+y, yc-x, pixelsWSharedCnt);

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

				putPixelLeft(s_imgOut, xc, xc-x, yc+y, pixelsWSharedCnt);
				putPixelLeft(s_imgOut, xc, xc-x, yc-y, pixelsWSharedCnt);
				putPixelLeft(s_imgOut, xc, xc-y, yc+x, pixelsWSharedCnt);
				putPixelLeft(s_imgOut, xc, xc-y, yc-x, pixelsWSharedCnt);

				putPixelRight(s_imgOut, xc, xc+x, yc+y, pixelsWSharedCnt);
				putPixelRight(s_imgOut, xc, xc+x, yc-y, pixelsWSharedCnt);
				putPixelRight(s_imgOut, xc, xc+y, yc+x, pixelsWSharedCnt);
				putPixelRight(s_imgOut, xc, xc+y, yc-x, pixelsWSharedCnt);
			}

			// copy from shared to global memry
			for (uint y = 0; y < frameWidth; y++)
			{
				uint realY = offsetPxY + y;
				if (realY >= dim.h) continue;

				for (uint x = 0; x < frameWidth; x++)
				{
					uint realX = offsetPxX + x;
					if (realX >= dim.w) continue;

					uint pxIdx = realY * dim.w + realX;
					uint relPxIdx = y * pixelsWSharedCnt + relOffsetPxX + x;

					imgOut[pxIdx] = s_imgOut[relPxIdx];
				}
			}
		}
	}
}

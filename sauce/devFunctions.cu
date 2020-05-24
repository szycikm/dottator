#include "defines.h"

#define getRelativeLuminance(pixel) 0.2126*pixel.r + 0.7152*pixel.g + 0.0722*pixel.b

// performed by each thread
__global__ void dev_makeDots(uint framesPerThread, uint frameWidth, uint framesW, dim_t dim, float dotScaleFactor, pixel_t* imgIn, uchar* imgOut)
{
	uint frameIdxBase = framesPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
	uint slackY;

#ifdef SHAREDED
	uint pixelsWSharedCnt = framesPerThread * frameWidth * blockDim.x;
	uint relFrameIdxBase = framesPerThread * threadIdx.x;
	uint yc = frameWidth/2;
	extern __shared__ uchar s_imgOut[]; // one row of dots (not like in global memry)

	// init shared memory with black
	for (int i = 0; i < pixelsWSharedCnt * frameWidth; i++)
		s_imgOut[i] = 0;
#endif

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

		uint r = avg * frameWidth / 512 * dotScaleFactor;

		if (r > 0)
		{
			// draw circle bres
#ifdef SHAREDED
			// draw the circle in shared memry

			uint relOffsetPxX = frameWidth * (relFrameIdxBase + f);
			uint x = 0;
			uint y = r;
			int d = 3 - 2 * r;
			uint xc = relOffsetPxX + frameWidth/2;

			// middle line (horizontal)
			slackY = yc * pixelsWSharedCnt;
			for (int i = xc-y; i <= xc+y; i++)
			{
				s_imgOut[slackY + i] = 255;
			}

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

				slackY = (yc+y) * pixelsWSharedCnt;
				for (int i = xc-x; i <= xc+x; i++)
				{
					s_imgOut[slackY + i] = 255;
				}

				slackY = (yc-y) * pixelsWSharedCnt;
				for (int i = xc-x; i <= xc+x; i++)
				{
					s_imgOut[slackY + i] = 255;
				}

				slackY = (yc+x) * pixelsWSharedCnt;
				for (int i = xc-y; i <= xc+y; i++)
				{
					s_imgOut[slackY + i] = 255;
				}

				slackY = (yc-x) * pixelsWSharedCnt;
				for (int i = xc-y; i <= xc+y; i++)
				{
					s_imgOut[slackY + i] = 255;
				}
			}

			// __syncthreads() not necessary because each thread has its own data

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
#else
			uint x = 0;
			uint y = r;
			int d = 3 - 2 * r;
			uint xc = offsetPxX + frameWidth/2;
			uint yc = offsetPxY + frameWidth/2;

			// middle line (horizontal)
			if (yc+x < dim.h)
			{
				slackY = (yc+x) * dim.w;

				for (int i = xc-y; i <= xc+y; i++)
				{
					if (i < dim.w)
						imgOut[slackY + i] = 255;
				}
			}

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

				if (yc+y < dim.h)
				{
					slackY = (yc+y) * dim.w;

					for (int i = xc-x; i <= xc+x; i++)
					{
						if (i < dim.w)
							imgOut[slackY + i] = 255;
					}
				}

				if (yc-y < dim.h)
				{
					slackY = (yc-y) * dim.w;

					for (int i = xc-x; i <= xc+x; i++)
					{
						if (i < dim.w)
							imgOut[slackY + i] = 255;
					}
				}

				if (yc+x < dim.h)
				{
					slackY = (yc+x) * dim.w;

					for (int i = xc-y; i <= xc+y; i++)
					{
						if (i < dim.w)
							imgOut[slackY + i] = 255;
					}
				}

				if (yc-x < dim.h)
				{
					slackY = (yc-x) * dim.w;

					for (int i = xc-y; i <= xc+y; i++)
					{
						if (i < dim.w)
							imgOut[slackY + i] = 255;
					}
				}
			}
#endif
		}
	}
}

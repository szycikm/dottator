#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include "dottator.h"

#define getRelativeLuminance(pixel) 0.2126*pixel.r + 0.7152*pixel.g + 0.0722*pixel.b

inline bool fileExists(const char* name)
{
	std::ifstream f(name);
	return f.good();
}

/*
    *
  *--
 *---
*----
*----
 *---
  *--
    *
*/
__device__ void putPixelLeft(uchar* imgOut, uint imgH, uint imgW, uint xc, uint x, uint y)
{
	if (y >= imgH) return;
	uint slackY = y * imgW;

	for (int i = x; i < xc; i++)
	{
		if (i >= imgW) break;

		imgOut[slackY + i] = 0;
	}
}

/*
notice the one extra pixel on the left - this is to fill the middle
-*
---*
----*
-----*
-----*
-----*
----*
---*
-*
*/
__device__ void putPixelRight(uchar* imgOut, uint imgH, uint imgW, uint xc, uint x, uint y)
{
	if (y >= imgH) return;
	uint slackY = y * imgW;

	for (int i = x; i >= xc; i--)
	{
		if (i >= imgW) break;

		imgOut[slackY + i] = 0;
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
	drawCirclePoint(imgOut, imgH, imgW, xc, yc, x, y);
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

__device__ void drawWholeSquare(uchar* imgOut, uint imgH, uint imgW, uint frameWidth, uint offsetPxX, uint offsetPxY, uchar value)
{
	for (uint y = 0; y < frameWidth; y++)
	{
		uint realY = offsetPxY + y;
		if (realY >= imgH) continue;

		for (uint x = 0; x < frameWidth; x++)
		{
			uint realX = offsetPxX + x;
			if (realX >= imgW) continue;

			imgOut[realY * imgW + realX] = value;
		}
	}
}

// performed by each thread
__global__ void dev_makeDots(uint frameWidth, uint imgW, uint imgH, float dotScaleFactor, pixel_t* imgIn, uchar* imgOut)
{
	uint offsetPxX = frameWidth * (blockIdx.x * THREADS_DIM + threadIdx.x);
	uint offsetPxY = frameWidth * (blockIdx.y * THREADS_DIM + threadIdx.y);

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

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		printf("Too few arguments\narg1: input filename\narg2: frame width (px)\narg3: dot scaling factor (default=1.0)\n");
		return 1;
	}

	// parse params
	char* inputFilename = argv[1];

	if(!fileExists(inputFilename))
	{
		printf("File doesn't exist\n");
		return 2;
	}

#ifndef DEBUG
	char suffix[] = "_out.jpg";
#else
	char suffix[] = "_out.png";
#endif
	char* outputFilename = new char[strlen(inputFilename) + strlen(suffix) + 1];
	strcpy(outputFilename, inputFilename);
	strcat(outputFilename, suffix);

	uint frameWidth = atoi(argv[2]);

	float dotScaleFactor = 1.0;
	if(argc >= 4)
		dotScaleFactor = atof(argv[3]);

#ifdef DEBUG
	printf("Input file:\t\t%s\nOutput file:\t\t%s\nFrame width:\t\t%dpx\nDot scaling factor:\t%f\n", inputFilename, outputFilename, frameWidth, dotScaleFactor);
#endif

	// load opencv image and convert it to array of pixels
	cv::Mat cvInImg = cv::imread(inputFilename);
	uint imgW = cvInImg.cols;
	uint imgH = cvInImg.rows;
	uint pixelCnt = imgW * imgH;
	cv::Mat cvOutImg(imgH, imgW, CV_8U);

	pixel_t* hostInPixels = new pixel_t[pixelCnt * sizeof(pixel_t)];
	uchar* hostOutPixels = new uchar[pixelCnt * sizeof(pixel_t)];

	for(int y = 0; y < imgH; y++)
	{
		for(int x = 0; x < imgW; x++)
		{
			cv::Vec3b intensity = cvInImg.at<cv::Vec3b>(y, x);

			hostInPixels[y * imgW + x] = {
				intensity.val[2],
				intensity.val[1],
				intensity.val[0]
			};
		}
	}

	// calculate number of frames and blocks
	uint imgDimFramesW = imgW/frameWidth;
	if (imgW % frameWidth != 0) imgDimFramesW++;

	uint imgDimFramesH = imgH/frameWidth;
	if (imgH % frameWidth != 0) imgDimFramesH++;

	uint blocksW = imgDimFramesW/THREADS_DIM;
	if (imgDimFramesW % THREADS_DIM != 0 || blocksW <= 0) blocksW++;

	uint blocksH = imgDimFramesH/THREADS_DIM;
	if (imgDimFramesH % THREADS_DIM != 0 || blocksH <= 0) blocksH++;

#ifdef DEBUG
	printf("imgW:\t\t\t%d\nimgH:\t\t\t%d\nimgDimFramesW:\t\t%d\nimgDimFramesH:\t\t%d\nblocksW:\t\t%d\nblocksH:\t\t%d\n", imgW, imgH, imgDimFramesW, imgDimFramesH, blocksW, blocksH);
#endif

	// copy memory to device
	pixel_t* devInPixels;
	cudaMalloc((void**)&devInPixels, pixelCnt * sizeof(pixel_t));
	cudaMemcpy(devInPixels, hostInPixels, pixelCnt * sizeof(pixel_t), cudaMemcpyHostToDevice);

	uchar* devOutPixels;
	cudaMalloc((void**)&devOutPixels, pixelCnt * sizeof(uchar));
	cudaMemset(devOutPixels, 255, pixelCnt * sizeof(uchar));

	// run kernel
	dim3 blocksPerGrid(blocksW, blocksH);
	dim3 threadsPerBlock(THREADS_DIM, THREADS_DIM);

#ifdef DEBUG
	long endTime;
	struct timeval timecheck;
	gettimeofday(&timecheck, NULL);
	long startTime = (long)timecheck.tv_sec + (long)timecheck.tv_usec;
#endif

	dev_makeDots<<<blocksPerGrid, threadsPerBlock>>>(frameWidth, imgW, imgH, dotScaleFactor, devInPixels, devOutPixels);
	cudaError err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		printf("Uh-oh, %s\n", cudaGetErrorString(err));
		goto freemem;
	}

#ifdef DEBUG
	gettimeofday(&timecheck, NULL);
	endTime = (long)timecheck.tv_sec + (long)timecheck.tv_usec;
	printf("Kernel execution took %ldus\n", endTime - startTime);
#endif

	// copy results from device
	cudaMemcpy(hostOutPixels, devOutPixels, pixelCnt * sizeof(uchar), cudaMemcpyDeviceToHost);

	// write results to output image
	for(int y = 0; y < imgH; y++)
	{
		for(int x = 0; x < imgW; x++)
		{
			cvOutImg.at<uchar>(y, x) = hostOutPixels[y * imgW + x];
		}
	}

	cv::imwrite(outputFilename, cvOutImg);

freemem:
	cudaFree(devInPixels);
	cudaFree(devOutPixels);

	free(hostOutPixels);
	free(hostInPixels);
	free(outputFilename);

	return 0;
}

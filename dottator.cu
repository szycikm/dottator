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

// performed by each thread
__global__ void dev_makeDots(uint frameWidth, uint imgW, uint imgH, pixel_t* imgIn, uchar* imgOut)
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

	for (uint y = 0; y < frameWidth; y++)
	{
		uint realY = offsetPxY + y;
		if (realY >= imgH) continue;

		for (uint x = 0; x < frameWidth; x++)
		{
			uint realX = offsetPxX + x;
			if (realX >= imgW) continue;

			imgOut[realY * imgW + realX] = avg;
		}
	}
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
	cv::Mat cvImg = cv::imread(inputFilename);
	uint imgW = cvImg.cols;
	uint imgH = cvImg.rows;
	uint pixelCnt = imgW * imgH;

	pixel_t* hostInPixels = new pixel_t[pixelCnt * sizeof(pixel_t)];
	uchar* hostOutPixels = new uchar[pixelCnt * sizeof(pixel_t)];

	for(int y = 0; y < imgH; y++)
	{
		for(int x = 0; x < imgW; x++)
		{
			cv::Vec3b intensity = cvImg.at<cv::Vec3b>(y, x);

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
	if (blocksW % THREADS_DIM != 0 || blocksW <= 0) blocksW++;

	uint blocksH = imgDimFramesH/THREADS_DIM;
	if (blocksH % THREADS_DIM != 0 || blocksH <= 0) blocksH++;

#ifdef DEBUG
	printf("imgW=%d\nimgH=%d\nimgDimFramesW=%d\nimgDimFramesH=%d\nblocksW=%d\nblocksH=%d\n", imgW, imgH, imgDimFramesW, imgDimFramesH, blocksW, blocksH);
#endif

	// copy memory to device
	pixel_t* devInPixels;
	cudaMalloc((void**)&devInPixels, pixelCnt * sizeof(pixel_t));
	cudaMemcpy(devInPixels, hostInPixels, pixelCnt * sizeof(pixel_t), cudaMemcpyHostToDevice);

	uchar* devOutPixels;
	cudaMalloc((void**)&devOutPixels, pixelCnt * sizeof(uchar));

	// run kernel
	dim3 blocksPerGrid(blocksW, blocksH);
	dim3 threadsPerBlock(THREADS_DIM, THREADS_DIM);

	struct timeval timecheck;
	gettimeofday(&timecheck, NULL);
	long startTime = (long)timecheck.tv_sec + (long)timecheck.tv_usec;

	dev_makeDots<<<blocksPerGrid, threadsPerBlock>>>(frameWidth, imgW, imgH, devInPixels, devOutPixels);

	gettimeofday(&timecheck, NULL);
	long endTime = (long)timecheck.tv_sec + (long)timecheck.tv_usec;
#ifdef DEBUG
	printf("Kernel execution took %ldus\n", endTime - startTime);
#endif

	// copy results from device
	cudaMemcpy(hostOutPixels, devOutPixels, pixelCnt * sizeof(uchar), cudaMemcpyDeviceToHost);

	// write results to output image
	cv::Mat cvOutImg(imgH, imgW, CV_8U);

	for(int y = 0; y < imgH; y++)
	{
		for(int x = 0; x < imgW; x++)
		{
			cvOutImg.at<uchar>(y, x) = hostOutPixels[y * imgW + x];
		}
	}

	cv::imwrite(outputFilename, cvOutImg);

	cudaFree(devInPixels);
	cudaFree(devOutPixels);

	free(hostOutPixels);
	free(hostInPixels);
	free(outputFilename);

	return 0;
}

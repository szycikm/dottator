#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include "defines.h"
#include "devFunctions.hpp"

inline bool fileExists(const char* name)
{
	std::ifstream f(name);
	return f.good();
}

int main(int argc, char *argv[])
{
#ifdef DEBUG
	long startTime, endTime, startTimeKernel, endTimeKernel;
	struct timeval timecheck;
	gettimeofday(&timecheck, NULL);
	startTime = (long)timecheck.tv_sec * 1000000LL + (long)timecheck.tv_usec;
#endif

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

	char suffix[] = "_out.png";
	char* outputFilename = new char[strlen(inputFilename) + strlen(suffix) + 1];
	strcpy(outputFilename, inputFilename);
	strcat(outputFilename, suffix);

	uint frameWidth = atoi(argv[2]);

	float dotScaleFactor = 1.0;
	if(argc >= 4)
		dotScaleFactor = atof(argv[3]);

	debug_printf("Input file:\t\t%s\nOutput file:\t\t%s\nFrame width:\t\t%dpx\nDot scaling factor:\t%f\n", inputFilename, outputFilename, frameWidth, dotScaleFactor);

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
	uint framesW = imgW/frameWidth;
	if (imgW % frameWidth != 0) framesW++;

	uint framesH = imgH/frameWidth;
	if (imgH % frameWidth != 0) framesH++;

	uint blocksW = framesW/THREADS_DIM;
	if (framesW % THREADS_DIM != 0 || blocksW <= 0) blocksW++;

	uint blocksH = framesH/THREADS_DIM;
	if (framesH % THREADS_DIM != 0 || blocksH <= 0) blocksH++;

	debug_printf("imgW:\t\t\t%d\nimgH:\t\t\t%d\nframesW:\t\t%d\nframesH:\t\t%d\nblocksW:\t\t%d\nblocksH:\t\t%d\n", imgW, imgH, framesW, framesH, blocksW, blocksH);

	// copy memory to device
	pixel_t* devInPixels;
	cudaMalloc((void**)&devInPixels, pixelCnt * sizeof(pixel_t));
	cudaMemcpy(devInPixels, hostInPixels, pixelCnt * sizeof(pixel_t), cudaMemcpyHostToDevice);

	uchar* devOutPixels;
	cudaMalloc((void**)&devOutPixels, pixelCnt * sizeof(uchar));
	cudaMemset(devOutPixels, 0, pixelCnt * sizeof(uchar));

	// run kernel
	dim3 blocksPerGrid(blocksW, blocksH);
	dim3 threadsPerBlock(THREADS_DIM, THREADS_DIM);

#ifdef DEBUG
	gettimeofday(&timecheck, NULL);
	startTimeKernel = (long)timecheck.tv_sec * 1000000LL + (long)timecheck.tv_usec;
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
	endTimeKernel = (long)timecheck.tv_sec * 1000000LL + (long)timecheck.tv_usec;
	printf("Kernel execution took %ldus\n", endTimeKernel - startTimeKernel);
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

#ifdef DEBUG
	gettimeofday(&timecheck, NULL);
	endTime = (long)timecheck.tv_sec * 1000000LL + (long)timecheck.tv_usec;
	printf("Total execution took %ldus\n", endTime - startTime);
#endif

	return 0;
}

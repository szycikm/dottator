#include <iostream>
#include <opencv2/opencv.hpp>
#include "dottator.h"

#define getRelativeLuminance(pixel) 0.2126*pixel.r + 0.7152*pixel.g + 0.0722*pixel.b;

inline bool fileExists(const char* name)
{
	std::ifstream f(name);
	return f.good();
}

// performed by each thread
__global__ void dev_makeDots(uint frameWidth, uint imgW, pixel_t* imgIn, uchar* imgOut)
{
	for (uint y = 0; y < frameWidth; y++)
	{
		for (uint x = 0; x < frameWidth; x++)
		{
			uint realX = frameWidth * (blockIdx.x * THREADS_DIM + threadIdx.x) + x;
			uint realY = frameWidth * (blockIdx.y * THREADS_DIM + threadIdx.y) + y;
			imgOut[realY * imgW + realX] = getRelativeLuminance(imgIn[realY * imgW + realX]);
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

	char suffix[] = "_out.jpg";
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

	// calculate number of frames
	uint imgDimFramesW = imgW/frameWidth;
	if (imgW % frameWidth != 0) imgDimFramesW++;

	uint imgDimFramesH = imgH/frameWidth;
	if (imgH % frameWidth != 0) imgDimFramesH++;

#ifdef DEBUG
	printf("imgDimFramesW=%d\nimgDimFramesH=%d\n", imgDimFramesW, imgDimFramesH);
#endif

	// copy memory to device
	pixel_t* devInPixels;
	cudaMalloc((void**)&devInPixels, pixelCnt * sizeof(pixel_t));
	cudaMemcpy(devInPixels, hostInPixels, pixelCnt * sizeof(pixel_t), cudaMemcpyHostToDevice);

	uchar* devOutPixels;
	cudaMalloc((void**)&devOutPixels, pixelCnt * sizeof(uchar));

	// run kernel
	dim3 blocksPerGrid(imgDimFramesW/THREADS_DIM, imgDimFramesH/THREADS_DIM);
	dim3 threadsPerBlock(THREADS_DIM, THREADS_DIM);
	dev_makeDots<<<blocksPerGrid, threadsPerBlock>>>(frameWidth, imgW, devInPixels, devOutPixels);

	// copy results from device
	cudaMemcpy(hostOutPixels, devOutPixels, pixelCnt * sizeof(uchar), cudaMemcpyDeviceToHost);

	// write results to output image
	for(int y = 0; y < imgH; y++)
	{
		for(int x = 0; x < imgW; x++)
		{
			uchar currPx = hostOutPixels[y * imgW + x];

			cv::Vec3b intensity(0, currPx, 0); // TODO somehow write only gray channel
			cvImg.at<cv::Vec3b>(y, x) = intensity;
		}
	}

	cv::imwrite(outputFilename, cvImg);

	cudaFree(devInPixels);
	cudaFree(devOutPixels);

	free(hostOutPixels);
	free(hostInPixels);
	free(outputFilename);

	return 0;
}

#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include "defines.h"
#include "utils.h"
#include "cvConvert.h"
#include "devFunctions.hpp"

int main(int argc, char *argv[])
{
	uint frameWidth = 25, threadsPerBlock = 32, imgW, imgH, pixelCnt, framesW, framesH, framesCnt, blocksCnt;
	float dotScaleFactor = 1.0;
	char* inputFilename;
	char* outputFilename;
	char suffix[] = "_out.png";
	cv::Mat cvInImg;
	cv::Mat* cvOutImg;
	pixel_t* hostInPixels;
	pixel_t* devInPixels;
	uchar* hostOutPixels;
	uchar* devOutPixels;
	cudaError err;

#ifdef DEBUG
	long startTime, endTime, startTimeKernel, endTimeKernel;
	struct timeval timecheck;
	gettimeofday(&timecheck, NULL);
	startTime = (long)timecheck.tv_sec * 1000000LL + (long)timecheck.tv_usec;
#endif

	if (argc < 2)
	{
		printf("Too few arguments\narg1: input filename\narg2: frame width (px) (default=25)\narg3: threads/block (default=32)\narg4: dot scaling factor (default=1.0)\n");
		return 1;
	}

	// parse params
	inputFilename = argv[1];

	if(!fileExists(inputFilename))
	{
		printf("File doesn't exist\n");
		return 2;
	}

	outputFilename = (char*)malloc(strlen(inputFilename) + strlen(suffix) + 1);
	if (outputFilename == NULL)
	{
		printf("Can't allocate memory\n");
		goto unroll_outputFilename;
	}

	strcpy(outputFilename, inputFilename);
	strcat(outputFilename, suffix);

	if (argc >= 3)
		frameWidth = atoi(argv[2]);

	if(argc >= 4)
		threadsPerBlock = atoi(argv[3]);

	if(argc >= 5)
		dotScaleFactor = atof(argv[4]);

	debug_printf("Input file:\t\t%s\nOutput file:\t\t%s\nFrame width:\t\t%dpx\nThreads/block:\t\t%d\nDot scaling factor:\t%f\n",
		inputFilename, outputFilename, frameWidth, threadsPerBlock, dotScaleFactor);

	// load opencv image and convert it to array of pixels
	cvInImg = cv::imread(inputFilename);
	imgW = cvInImg.cols;
	imgH = cvInImg.rows;
	pixelCnt = imgW * imgH;
	cvOutImg = new cv::Mat(imgH, imgW, CV_8U);

	hostInPixels = (pixel_t*)malloc(pixelCnt * sizeof(pixel_t));
	if (hostInPixels == NULL)
	{
		printf("Can't allocate memory\n");
		goto unroll_hostInPixels;
	}

	hostOutPixels = (uchar*)malloc(pixelCnt * sizeof(pixel_t));
	if (hostOutPixels == NULL)
	{
		printf("Can't allocate memory\n");
		goto unroll_hostOutPixels;
	}

	cvToRawImg(&cvInImg, hostInPixels, imgH, imgW);

	// calculate number of frames and blocks

	framesW = imgW/frameWidth;
	if (imgW % frameWidth != 0) framesW++;

	framesH = imgH/frameWidth;
	if (imgH % frameWidth != 0) framesH++;

	framesCnt = framesW * framesH;

	blocksCnt = framesCnt/threadsPerBlock;
	if (framesCnt % threadsPerBlock != 0 || blocksCnt <= 0) blocksCnt++;

	debug_printf("imgW:\t\t\t%d\nimgH:\t\t\t%d\nframesW:\t\t%d\nframesH:\t\t%d\nframesCnt:\t\t%d\nblocksCnt:\t\t%d\n",
		imgW, imgH, framesW, framesH, framesCnt, blocksCnt);

	// copy memory to device
	if (cudaMalloc((void**)&devInPixels, pixelCnt * sizeof(pixel_t)) != cudaSuccess)
	{
		printf("Can't allocate GPU memory\n");
		goto unroll_devInPixels;
	}

	cudaMemcpy(devInPixels, hostInPixels, pixelCnt * sizeof(pixel_t), cudaMemcpyHostToDevice);

	if (cudaMalloc((void**)&devOutPixels, pixelCnt * sizeof(uchar)) != cudaSuccess)
	{
		printf("Can't allocate GPU memory\n");
		goto unroll_devOutPixels;
	}
	cudaMemset(devOutPixels, 0, pixelCnt * sizeof(uchar)); // set all pixels to black

#ifdef DEBUG
	gettimeofday(&timecheck, NULL);
	startTimeKernel = (long)timecheck.tv_sec * 1000000LL + (long)timecheck.tv_usec;
#endif

	dev_makeDots<<<blocksCnt, threadsPerBlock>>>(frameWidth, framesW, imgW, imgH, dotScaleFactor, devInPixels, devOutPixels);
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		printf("Uh-oh, %s\n", cudaGetErrorString(err));
		goto unroll_cudaerror;
	}

#ifdef DEBUG
	gettimeofday(&timecheck, NULL);
	endTimeKernel = (long)timecheck.tv_sec * 1000000LL + (long)timecheck.tv_usec;
	printf("Kernel execution took\t%ldus\n", endTimeKernel - startTimeKernel);
#endif

	// copy results from device
	cudaMemcpy(hostOutPixels, devOutPixels, pixelCnt * sizeof(uchar), cudaMemcpyDeviceToHost);

	rawToCvImg(hostOutPixels, cvOutImg, imgH, imgW);

	cv::imwrite(outputFilename, *cvOutImg);

unroll_cudaerror:
	cudaFree(devOutPixels);
unroll_devOutPixels:
	cudaFree(devInPixels);
unroll_devInPixels:
	free(hostOutPixels);
unroll_hostOutPixels:
	free(hostInPixels);
unroll_hostInPixels:
	delete cvOutImg;
	free(outputFilename);
unroll_outputFilename:

#ifdef DEBUG
	gettimeofday(&timecheck, NULL);
	endTime = (long)timecheck.tv_sec * 1000000LL + (long)timecheck.tv_usec;
	printf("Total execution took:\t%ldus\n", endTime - startTime);
#endif

	return 0;
}

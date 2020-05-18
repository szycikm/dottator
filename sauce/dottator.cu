#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include "defines.h"
#include "utils.h"
#include "cvConvert.h"
#include "devFunctions.hpp"

#define HELPSTRING "Usage:\narg1:\t\t\tinput filename [required]\n-h, --help\t\tprint this help and exit\n-f, --framewidth\tframe width (px) [default=25]\n-b, --threadsperblock\tthreads/block [default=32]\n-t, --framesperthread\tframes/thread [default=1]\n-s, --scale\t\tdot scaling factor [default=1.0]\n"
#define OUT_SUFFIX "_out.png"

int main(int argc, char* argv[])
{
	uint frameWidth = 25, threadsPerBlock = 32, framesPerThread = 1, pixelCnt, framesW, framesH, framesCnt, blocksCnt;
	float dotScaleFactor = 1.0;
	char* inputFilename;
	char* outputFilename;
	dim_t dim;
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
		printf("Too few arguments\n\n");
		printf(HELPSTRING);
		return 0;
	}

	// parse optional params
	for (int i = 2; i < argc; i+=2)
	{
		std::string arg = argv[i];
		if (arg == "-h" || arg == "--help")
		{
			printf(HELPSTRING);
			return 0;
		}
		if (arg == "-f" || arg == "--framewidth")
			frameWidth = atoi(argv[i+1]);
		if (arg == "-b" || arg == "--threadsperblock")
			threadsPerBlock = atoi(argv[i+1]);
		if (arg == "-t" || arg == "--framesperthread")
			framesPerThread = atoi(argv[i+1]);
		if (arg == "-s" || arg == "--scale")
			dotScaleFactor = atof(argv[i+1]);
	}

	inputFilename = argv[1];

	// in/out files
	if(!fileExists(inputFilename))
	{
		printf("Input file doesn't exist\n");
		return 0;
	}

	outputFilename = (char*)malloc(strlen(inputFilename) + strlen(OUT_SUFFIX) + 1);
	if (outputFilename == NULL)
	{
		printf("Can't allocate memory\n");
		goto unroll_outputFilename;
	}

	strcpy(outputFilename, inputFilename);
	strcat(outputFilename, OUT_SUFFIX);

	// print summary
	debug_printf("Input file:\t\t%s\nOutput file:\t\t%s\nFrame width:\t\t%dpx\nThreads/block:\t\t%d\nFrames/thread:\t\t%d\nDot scaling factor:\t%f\n",
		inputFilename, outputFilename, frameWidth, threadsPerBlock, framesPerThread, dotScaleFactor);

	// load opencv image and convert it to array of pixels
	cvInImg = cv::imread(inputFilename);
	dim.w = cvInImg.cols;
	dim.h = cvInImg.rows;
	pixelCnt = dim.h * dim.w;
	cvOutImg = new cv::Mat(dim.h, dim.w, CV_8U);

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

	cvToRawImg(&cvInImg, hostInPixels, dim);

	// calculate number of frames and blocks

	framesW = dim.w/frameWidth;
	if (dim.w % frameWidth != 0) framesW++;

	framesH = dim.h/frameWidth;
	if (dim.h % frameWidth != 0) framesH++;

	framesCnt = framesW * framesH;

	blocksCnt = framesCnt/threadsPerBlock;
	if (framesCnt % threadsPerBlock != 0 || blocksCnt <= 0) blocksCnt++;

	debug_printf("img width:\t\t%d\nimg height:\t\t%d\nframesW:\t\t%d\nframesH:\t\t%d\nframesCnt:\t\t%d\nblocksCnt:\t\t%d\n",
		dim.w, dim.h, framesW, framesH, framesCnt, blocksCnt);

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

	dev_makeDots<<<blocksCnt, threadsPerBlock>>>(frameWidth, framesW, dim, dotScaleFactor, devInPixels, devOutPixels);
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

	rawToCvImg(hostOutPixels, cvOutImg, dim);

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

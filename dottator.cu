#include <iostream>
#include <opencv2/opencv.hpp>

typedef struct
{
	uchar r;
	uchar g;
	uchar b;
} pixel_t;

#define getRelativeLuminance(pixel) 0.2126*pixel.r + 0.7152*pixel.g + 0.0722*pixel.b;

inline bool fileExists(const char* name)
{
	std::ifstream f(name);
	return f.good();
}

// performed by each thread woah
__global__ void dev_makeDots(pixel_t* imgIn, uchar* imgOut)
{
	int i = blockIdx.x;
	imgOut[i] = (imgIn[i].b + imgIn[i].g + imgIn[i].r) / 3;
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

	int frameWidth = atoi(argv[2]);

	float dotScaleFactor = 1.0;
	if(argc >= 4)
		dotScaleFactor = atof(argv[3]);

#ifdef DEBUG
	printf("Input file:\t\t%s\nOutput file:\t\t%s\nFrame width:\t\t%dpx\nDot scaling factor:\t%f\n", inputFilename, outputFilename, frameWidth, dotScaleFactor);
#endif

	// load opencv image and convert it to array of pixel_t
	cv::Mat cvImg = cv::imread(inputFilename);
	uint pixelCnt = cvImg.rows * cvImg.cols;

	pixel_t* hostInPixels = new pixel_t[pixelCnt * sizeof(pixel_t)];
	uchar* hostOutPixels = new uchar[pixelCnt * sizeof(uchar)];

	for(int y = 0; y < cvImg.rows; y++)
	{
		for(int x = 0; x < cvImg.cols; x++)
		{
			cv::Vec3b intensity = cvImg.at<cv::Vec3b>(y, x);

			hostInPixels[y * cvImg.cols + x] = {
				intensity.val[2],
				intensity.val[1],
				intensity.val[0]
			};
		}
	}

	pixel_t* devInPixels;
	cudaMalloc((void**)&devInPixels, pixelCnt * sizeof(pixel_t));
	cudaMemcpy(devInPixels, hostInPixels, pixelCnt * sizeof(pixel_t), cudaMemcpyHostToDevice);

	uchar* devOutPixels;
	cudaMalloc((void**)&devOutPixels, pixelCnt * sizeof(uchar));

	// <<<blocks cnt per grid, threads cnt per block>>
	dev_makeDots<<<pixelCnt,1>>>(devInPixels, devOutPixels);

	cudaMemcpy(hostOutPixels, devOutPixels, pixelCnt * sizeof(uchar), cudaMemcpyDeviceToHost);

	for(int y = 0; y < cvImg.rows; y++)
	{
		for(int x = 0; x < cvImg.cols; x++)
		{
			uchar currPx = hostOutPixels[y * cvImg.cols + x];

#ifdef DEBUG
			printf("(%d,%d) = %d\n", x, y, currPx);
#endif

			cv::Vec3b intensity(currPx, currPx, currPx); // TODO somehow write only gray channel
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

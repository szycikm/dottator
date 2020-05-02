#include <iostream>
#include <opencv2/opencv.hpp>

#define getRelativeLuminance(pixel) 0.2126*pixel.r + 0.7152*pixel.g + 0.0722*pixel.b;

inline bool fileExists(const char* name)
{
	std::ifstream f(name);
	return f.good();
}

typedef struct
{
	uchar r;
	uchar g;
	uchar b;
} pixel_t;

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

	pixel_t* inputImgPixels = new pixel_t[cvImg.rows * cvImg.cols * sizeof(pixel_t)];

	for(int y = 0; y < cvImg.rows; y++)
	{
		for(int x = 0; x < cvImg.cols; x++)
		{
			cv::Vec3b intensity = cvImg.at<cv::Vec3b>(y, x);

			inputImgPixels[y * cvImg.cols + x] = {
				intensity.val[2],
				intensity.val[1],
				intensity.val[0]
			};
		}
	}

	/*for(int y = 0; y < cvImg.rows; y++)
	{
		for(int x = 0; x < cvImg.cols; x++)
		{
			pixel_t* currPx = &inputImgPixels[y * cvImg.cols + x];
			currPx->g = 255;
		}
	}*/

	for(int y = 0; y < cvImg.rows; y++)
	{
		for(int x = 0; x < cvImg.cols; x++)
		{
			pixel_t* currPx = &inputImgPixels[y * cvImg.cols + x];
			cv::Vec3b intensity(currPx->b, currPx->g, currPx->r);
			cvImg.at<cv::Vec3b>(y, x) = intensity;
		}
	}

	cv::imwrite(outputFilename, cvImg);

	free(inputImgPixels);
	free(outputFilename);

	return 0;
}

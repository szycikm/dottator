#include <opencv2/opencv.hpp>
#include "defines.h"

void cvToRawImg(cv::Mat* cvInImg, pixel_t* hostInPixels, uint imgH, uint imgW)
{
	for(int y = 0; y < imgH; y++)
	{
		for(int x = 0; x < imgW; x++)
		{
			cv::Vec3b intensity = cvInImg->at<cv::Vec3b>(y, x);

			hostInPixels[y * imgW + x] = {
				intensity.val[2],
				intensity.val[1],
				intensity.val[0]
			};
		}
	}
}

void rawToCvImg(uchar* hostOutPixels, cv::Mat* cvOutImg, uint imgH, uint imgW)
{
	for(int y = 0; y < imgH; y++)
	{
		for(int x = 0; x < imgW; x++)
		{
			cvOutImg->at<uchar>(y, x) = hostOutPixels[y * imgW + x];
		}
	}
}

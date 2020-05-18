#include <opencv2/opencv.hpp>
#include "defines.h"

void cvToRawImg(cv::Mat* cvInImg, pixel_t* hostInPixels, dim_t dim)
{
	for(int y = 0; y < dim.h; y++)
	{
		for(int x = 0; x < dim.w; x++)
		{
			cv::Vec3b intensity = cvInImg->at<cv::Vec3b>(y, x);

			hostInPixels[y * dim.w + x] = {
				intensity.val[2],
				intensity.val[1],
				intensity.val[0]
			};
		}
	}
}

void rawToCvImg(uchar* hostOutPixels, cv::Mat* cvOutImg, dim_t dim)
{
	for(int y = 0; y < dim.h; y++)
	{
		for(int x = 0; x < dim.w; x++)
		{
			cvOutImg->at<uchar>(y, x) = hostOutPixels[y * dim.w + x];
		}
	}
}

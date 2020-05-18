#include <opencv2/opencv.hpp>
#include "defines.h"

void cvToRawImg(cv::Mat* cvInImg, pixel_t* hostInPixels, dim_t imgDim);
void rawToCvImg(uchar* hostOutPixels, cv::Mat* cvOutImg, dim_t imgDim);

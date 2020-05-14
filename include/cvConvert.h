#include <opencv2/opencv.hpp>
#include "defines.h"

void cvToRawImg(cv::Mat* cvInImg, pixel_t* hostInPixels, uint imgH, uint imgW);
void rawToCvImg(uchar* hostOutPixels, cv::Mat* cvOutImg, uint imgH, uint imgW);

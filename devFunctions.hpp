#include "defines.h"

__global__ void dev_makeDots(uint frameWidth, uint imgW, uint imgH, float dotScaleFactor, pixel_t* imgIn, uchar* imgOut);

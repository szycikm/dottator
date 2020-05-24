#include "defines.h"

__global__ void dev_makeDots(uint framesPerThread, uint frameWidth, uint framesW, dim_t dim, float dotScaleFactor, pixel_t* imgIn, uchar* imgOut);

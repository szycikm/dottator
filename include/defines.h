#ifndef uchar
#define uchar unsigned char
#endif

#ifndef PIXEL_T
#define PIXEL_T
typedef struct
{
	uchar r;
	uchar g;
	uchar b;
} pixel_t;
#endif

#ifndef DIM_T
#define DIM_T
typedef struct
{
	uint h;
	uint w;
} dim_t;
#endif

#ifndef debug_printf
#ifdef DEBUG
#define debug_printf(a, ...) printf(a, ##__VA_ARGS__)
#else
#define debug_printf(a, ...)
#endif
#endif

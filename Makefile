LDFLAGS= -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS=-DDEBUG
else
	CFLAGS=
endif

all: dottator

dottator.o: dottator.cu
	nvcc -o dottator.o -c dottator.cu $(CFLAGS)

dottator: dottator.o
	nvcc $(LDFLAGS) -o dottator dottator.o

run: dottator
	./dottator IMG_20200414_145436.jpg 10 1.1

runsmall: dottator
	./dottator colors4x3.jpg 2 0.7

clean:
	rm -f dottator dottator.o

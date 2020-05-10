LDFLAGS= -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS=-DDEBUG
else
	CFLAGS=
endif

OBJECTS := dottator.o devFunctions.o

all: dottator

dottator.o: dottator.cu
	nvcc $(CFLAGS) -o $@ -c $<

devFunctions.o: devFunctions.cu
	nvcc $(CFLAGS) -o $@ -c $<

dottator: $(OBJECTS)
	nvcc $(LDFLAGS) -o $@ $+

clean:
	rm -f dottator $(OBJECTS)

LDFLAGS= -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
CFLAGS=
OBJECTS := dottator.o devFunctions.o

all: dottator

dottator.o: dottator.cu
	nvcc $(CFLAGS) -o $@ -c $<

devFunctions.o: devFunctions.cu
	nvcc $(CFLAGS) -o $@ -c $<

dottator: $(OBJECTS)
	nvcc $(LDFLAGS) -o $@ $+

debug: CFLAGS=-DDEBUG
debug: dottator

clean:
	rm -f dottator $(OBJECTS)

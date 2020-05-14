LDFLAGS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
CFLAGS = -Iinclude
OBJECTS = dottator.o utils.o cvConvert.o devFunctions.o

all: dottator

dottator.o: dottator.cu
	nvcc $(CFLAGS) -o $@ -c $<

utils.o: utils.cpp
	gcc $(CFLAGS) -o $@ -c $<

cvConvert.o: cvConvert.cpp
	gcc $(CFLAGS) -o $@ -c $<

devFunctions.o: devFunctions.cu
	nvcc $(CFLAGS) -o $@ -c $<

dottator: $(OBJECTS)
	nvcc $(LDFLAGS) -o $@ $+

debug: CFLAGS+=-DDEBUG
debug: dottator

clean:
	rm -f dottator $(OBJECTS)

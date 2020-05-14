LDFLAGS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
CFLAGS = -Iinclude
NVOBJECTS = devFunctions.o dottator.o
GCCOBJECTS = utils.o cvConvert.o

all: dottator

%.o: sauce/%.cu
	nvcc $(CFLAGS) -o $@ -c $<

%.o: sauce/%.cpp
	gcc $(CFLAGS) -o $@ -c $<

dottator: $(GCCOBJECTS) $(NVOBJECTS)
	nvcc $(LDFLAGS) -o $@ $+

debug: CFLAGS+=-DDEBUG
debug: dottator

clean:
	rm -f $(NVOBJECTS) $(GCCOBJECTS) dottator

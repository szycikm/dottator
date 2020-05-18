# dottator
Halftone filter for images.

Converts images to something like [this](https://en.wikipedia.org/wiki/Halftone) (see images).

This is my first CUDA project, and also the first project where I used makefile.

# building
Normal app:
```
$ cd dottator
$ make
```

Debug build (outputs some info when running app):
```
$ cd dottator
$ make clean debug
```

# usage
```
arg1                  input filename [required]
-h, --help            print this help and exit
-f, --framewidth      frame width (px) [default=25]
-b, --threadsperblock threads/block [default=32]
-t, --framesperthread frames/thread [default=1]
-s, --scale           dot scaling factor [default=1.0]
```

Example:
```
$ ./dottator input.jpg -f 20 -b 64 -t 2 -s 1.3
```

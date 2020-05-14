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

# running
```
./dottator [input filename] [frame width in px] [dot scaling factor (optional)]
```

Example:
```
$ ./dottator input.jpg 20
$ ./dottator input.png 35 1.4
```

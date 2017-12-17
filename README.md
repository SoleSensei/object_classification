# Object Classification
Train classificator on different images features and predict image class.

[Full documantation](https://solesensei.github.io/object_classification/)

## Learning feautures
[Histogram of oriented gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)

    1) grayscale
    2) sobel filter
    3) histogram of pixels brightness 
    4) normalization and concatenation 

[Local Binary Patterns](https://en.wikipedia.org/wiki/Local_binary_patterns)

Color features

## using SSE
Used SSE technology for Sobel filers and calculation gradients absolutes

    Naive implementation (no SSE) : Time [0.614] seconds
    SSE implementation : Time [0.561] seconds

    Ubuntu 16.04
    System: Intel Core i5-6200U 2.30GHz, 8Gb

## Build

    make all
    make clean

### Launch from ./build/bin
train classificator

    ./task2 -d ../../data/multiclass/train_labels.txt -m model.txt --train

classification

    ./task2 -d ../../data/multiclass/test_labels.txt -m model.txt -l predictions.txt --predict

check precisiong

    ./compare.py data/multiclass/test_labels.txt build/bin/predictions.txt

### Test script 
Build and launch
    
    bash run.sh

Tested with Google Test. More info [./tests/readme.md](./tests/readme.md)

    bash runTest.sh

[Documentation](https://solesensei.github.io/object_classification/) with Doxygen

    bash runDocs.sh

Clean project. Back to original

    bash runClean.sh

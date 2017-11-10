#!/bin/bash

# without SSE

cd ./build/bin


# train classificator
./task2 -d ../../data/multiclass/train_labels.txt -m model.txt --train

# test classificator
./task2 -d ../../data/multiclass/test_labels.txt -m model.txt -l predict.txt --predict

cd ../..

#compare result
echo 
python compare.py ./data/multiclass/test_labels.txt ./build/bin/predict.txt 
echo
# using SSE

cd ./build/bin


# train classificator
./task2 -d ../../data/multiclass/train_labels.txt -m model.txt --train --sse

# test classificator
./task2 -d ../../data/multiclass/test_labels.txt -m model.txt -l predict.txt --predict --sse

cd ../..

#compare result
echo
python compare.py ./data/multiclass/test_labels.txt ./build/bin/predict.txt 
echo
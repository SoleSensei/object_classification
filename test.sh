#!/bin/bash


cd ./build/bin

#train classificator
./task2 -d ../../data/multiclass/train_labels.txt -m model.txt --train

#test classificator
./task2 -d ../../data/multiclass/test_labels.txt -m model.txt -l predict.txt --predict

cd ../..

#compare result
python compare.py ./data/multiclass/test_labels.txt ./build/bin/predict.txt 
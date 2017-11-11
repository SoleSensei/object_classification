#!/bin/bash
# sudo bash runTest.sh # to begin tests

# make project 
make

# create build directory to make tests
cd ./tests
mkdir build

# copy Make Lists for cmake
cp ../bridge/CMakeLists.txt ./build/CMakeLists.txt
echo

# make tests
echo Make Tests
cd build 
cmake CMakeLists.txt
make
echo 

# run tests
echo Running Tests 
./runTests



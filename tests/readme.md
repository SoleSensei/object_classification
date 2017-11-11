Google Test C++ test framework

1. Download google.test archive https://github.com/google/googletest, unzip to ${prj} folder googletest

2. Run next commands
Install the gtest development package and cmake: 

    sudo apt-get install libgtest-dev
    sudo apt-get install cmake

3. Build google test first
go to google test dir
    
    cd  ${prj}/googletest
    
    mkdir mybuild
    cd mybuild
    sudo cmake ../CMakeLists.txt
    sudo make
    # copy or symlink libgtest.a and libgtest_main.a to your /usr/lib folder
    sudo cp *.a /usr/lib

3. Make and run tests
Look runTest.sh to more information

    cd ${prj}
    bash runTest.sh
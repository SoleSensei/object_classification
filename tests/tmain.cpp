/* Googletest main */
#include <gtest/gtest.h>
#include <iostream>

#include "matrix.h"
#include "classifier.h" 
#include "linear.h"
#include "argvparser.h"
#include "trainer.hpp"
#include "trainer.cpp"
#include "EasyBMP.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

#define TEST_IMAGE "../../data/Lenna.bmp"

class TestGradientAbsolut : public ::testing::Test {
protected:
	Image sobX;
	Image sobY;
	void SetUp(){
		cerr << "SET UP TEST" << endl;
    	BMP* src = new BMP();
    	src->ReadFromFile(TEST_IMAGE);
		Image pic = grayScale(src);
		sobX = sobelX(pic);
		sobY = sobelY(pic);
	}
	void TearDown()
	{}
};

TEST_F(TestGradientAbsolut, test) {

	Image res = gradAbs(sobX, sobY);	
	Image res_SSE = gradAbs_SSE(sobX, sobY);
	ASSERT_EQ(res.n_rows, res_SSE.n_rows) << "fatal error | different number of rows";
	ASSERT_EQ(res.n_cols, res_SSE.n_cols) << "fatal error | different number of cols";
		//compare values differents
	float error = imgDif(res, res_SSE);
	ASSERT_FLOAT_EQ(0, error) << "max difference: " << error;
}

 
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
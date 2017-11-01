#pragma once
#include <vector>
#include <cmath>


#include "matrix.h"
#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

#include <tuple>
using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::tuple;
using std::make_tuple;
//define some types
typedef Matrix<double> Image;
typedef Matrix<tuple<uint,uint,uint>> Image3;
typedef std::vector<double> Histype;

//Image ptr + class label 
typedef std::vector<pair<BMP*, int> > TDataSet;
//Image filename + class label
typedef std::vector<pair<string, int> > TFileList;
//feautures vector + class label
typedef std::vector<pair<Histype, int> > TFeatures;

#define PI 3.1415926535897932384626433832795028841971
#define EPS 0.00001

class unaryOp
{
    Matrix<double> kernel;
public:
    const uint radius;
    unaryOp(const Matrix<double>& ker): kernel(ker), radius((ker.n_rows-1)/2) {}
    Image operator()(const Image& neighbourhood) const;

};

Image grayScale (BMP* src);

//split pic into [parts_num] pieces
void splitInto(Image pic, Image* cells, const int parts_num);
void splitInto3(Image3 pic, Image3* cells, const int parts_num);
vector<int> to_vector(Image matrix);
Image3 to_image3(BMP* src);

Histype calc_hog(Image gabs, Image gdir);  
Histype calc_lbl(Image pic);  
Histype calc_color(Image3 pic);

Image custom(Image src, Image kernel);

Image sobelX(Image src);
Image sobelY(Image src);

Image gradAbs(Image src);
Image gradDir(Image src);
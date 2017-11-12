#pragma once
#include <vector>
#include <cmath>
#include <malloc.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <pmmintrin.h>

#include "Environment.h"
#include "Timer.h"

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
typedef Matrix<float> Image;
typedef Matrix<tuple<uint,uint,uint>> Image3;
typedef std::vector<float> Histype;

///Image ptr + class label 
typedef std::vector<pair<BMP*, int> > TDataSet;
///Image filename + class label
typedef std::vector<pair<string, int> > TFileList;
///feautures vector + class label
typedef std::vector<pair<Histype, int> > TFeatures;

#define PI 3.1415926535897932384626433832795028841971
#define EPS 0.00001


/**
@file trainer.hpp
*/

/**@function grayScale
 * realizes naive approach to convert BMP (RGBA) image to grayscale
 * @param src is pointer to BMP source image
 * @return Converted grayscale one channel image
 */
Image grayScale (BMP* src);

/**@function splitInto
 * splits image into several pieces
 * @param[in] pic is input one channel image
 * @param[in] parts_num is number of parts that image will be splitted
 * @param[out] cells is pointers to splitted images
 */
void splitInto(Image pic, Image* cells, const int parts_num);

/**@function splitInto3
 * splits image into several pieces, does the same as splitInto() but with 3 channels image
 * @param[in] pic is input rgb image
 * @param[in] parts_num is number of parts that image will be splitted
 * @param[out] cells is pointers to splitted images
 */
void splitInto3(Image3 pic, Image3* cells, const int parts_num);

/**@function to_vector
 * is convert function from matrix to vector.
 * @code 
 * [a, b, c]
 * [d, e, f]   ->  [a, b, c, d, e, f, g, h, i]
 * [g, h, i]
 * @endcode
 * @param matrix - image matrix representation
 * @return vector<int> - converted one row representation
 */
vector<int> to_vector(Image matrix);

/**@function to_image3
 * similar grayScale() function. Converts BMP (RGBA) image to 3 channel Image
 * @param src is pointer to BMP source image
 * @return Converted three channels image. Look representation of @link Image3
 */
Image3 to_image3(BMP* src);

/**@function calc_hog
 * calculates [histogram of oriented gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) and normalize it
 * @param gabs is calculated matrix of gradient absolutes
 * @param gdir is calculated matrix of gradient directions
 * @return Look Histype typedef. HOG - vector of floats
 */
Histype calc_hog(Image gabs, Image gdir); 

/**@function calc_lbl
 * calculates [local binary patterns](https://en.wikipedia.org/wiki/Local_binary_patterns) and normalize it to more predict precision. 
 * @param pic is image of floats
 * @return Look Histype typedef. HOG - vector of floats
 */
Histype calc_lbl(Image pic);  

/**@function calc_lbl
 * calculates color feautures and normalize it to more predict precision. 
 * @param pic is three channels image
 * @return Look Histype typedef. HOG - vector of floats
 */
Histype calc_color(Image3 pic);

/**@function custom
 * applies 'kernel' to source image 
 * @param src is source image of floats
 * @param kernel is matrix 3x3. look sobelX() | sobelY()
 * @return Matrix of floats
 */
Image custom(Image src, Image kernel);

/**@function custom_SSE
 * applies 'kernel' to source image. Does the same as custom(), but using SSE
 * @param src is source image of floats
 * @param kernel is matrix 3x3. look sobelX() | sobelY()
 * @return Matrix of floats
 */
Image custom_SSE(Image src, Image kernel);

/**@function sobelX
 * stores horisontal sobel operator to apply to image
 * @param src is source image, that will be processing 
 * @return Matrix of floats
 */
Image sobelX(Image src);

/**@function sobelX
 * stores vertical sobel operator to apply to image
 * @param src is source image, that will be processing 
 * @return Matrix of floats
 */
Image sobelY(Image src);

/**@function sobelX_SSE
 * does the same as sobelX(), but using SSE
 */
Image sobelX_SSE(Image src);

/**@function sobelY_SSE
 * does the same as sobelY(), but using SSE
 */
Image sobelY_SSE(Image src);

/**@function gradAbs
 * calculates gradients absolutes
 * @param sobelx is calculated with sobelX() image of floats
 * @param sobelY is calculated with sobelY() image of floats
 * @return Matrix of calculated gradient absolutes
 */
Image gradAbs(Image sobelx, Image sobely);

/**@function gradAbs_SSE
 * calculates gradients absolutes, does the same as gradAbs(), but using SSE
 * @param hor is calculated with sobelX_SSE() image of floats
 * @param ver is calculated with sobelY_SSE() image of floats
 * @return Matrix of calculated gradient absolutes
 */
Image gradAbs_SSE(Image hor, Image ver);

/**@function gradDir
 * calculates gradients directions
 * @param sobelx is calculated with sobelX() image of floats
 * @param sobelY is calculated with sobelY() image of floats
 * @return Matrix of calculated gradient directions
 */
Image gradDir(Image sobelx, Image sobely);

/**@function imgDif
 * compares two images on difference
 * @param img1 is first image of floats to compare
 * @param img2 is second image of floats to compare
 * @return Maximum difference between two elemets of matricies
 */
float imgDif(Image img1, Image img2);
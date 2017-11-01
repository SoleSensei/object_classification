#include <assert.h>
#include "trainer.hpp"

#include "matrix.h"
#include "classifier.h" 
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

Image3 to_image3(BMP* pic){

    auto width = pic->TellWidth();
    auto height = pic->TellHeight();
    Image3 res = Image3(width, height);

    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j){
            RGBApixel pix = pic->GetPixel(i,j);
            res(i,j) = make_tuple(pix.Red, pix.Green, pix.Blue);
    }
    
    return res;
}

Image grayScale (BMP* pic){

    auto width = pic->TellWidth();
    auto height = pic->TellHeight();
    Image res = Image(width, height);
    
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
        {
            RGBApixel pix = pic->GetPixel(i,j);
            res(i,j) = 0.299 * pix.Red + 0.587 * pix.Green + 0.114 * pix.Blue;
        }

    return res;
}

void splitInto(Image pic, Image* cells, const int parts){

    int part = sqrt(parts);
    uint lenX = pic.n_rows / part;
    uint lenY = pic.n_cols / part;
    for(int i = 0; i < parts; ++i){
        auto fromX = (i%part) * lenX;
        auto fromY = (i/part) * lenY;
        cells[i]=pic.submatrix(fromX, fromY, lenX, lenY);
    }   
}

void splitInto3(Image3 pic, Image3* cells, const int parts){
    
        int part = sqrt(parts);
        uint lenX = pic.n_rows / part;
        uint lenY = pic.n_cols / part;
        for(int i = 0; i < parts; ++i){
            auto fromX = (i%part) * lenX;
            auto fromY = (i/part) * lenY;
            cells[i]=pic.submatrix(fromX, fromY, lenX, lenY);
        }
    }

vector<int> to_vector(Image matrix){
    //only for matrix 3x3
    const int size = matrix.n_rows * matrix.n_cols - 1; //1 - one cell in the middle of matrix
    vector<int> vector(size, 0);

    uint k = 0;
    for (uint j = 0; j < matrix.n_cols; ++j) //first row (->)
        vector[k++] = matrix(0,j);
    for (uint i = 1; i < matrix.n_rows; ++i) //last col (\/)
        vector[k++] = matrix(i,2);
    for (int j = 1; j >= 0; --j) //last row (<-)
        vector[k++] = matrix(2,j);
    vector[k] = matrix(1,0); //first col (^)

    return vector; 
}

Histype calc_hog(Image gabs, Image gdir){

    const int size = 32;
    Histype hist(size, 0.0);
    for (uint i = 0; i < gdir.n_rows; ++i){
        for(uint j = 0; j < gdir.n_cols; ++j){
            uint k = uint(PI + gdir(i,j) / (2*PI) * size) % size; 
            hist[k] += gabs(i,j);
        }
    }

        //normalization
    double sum2 = 0.0;
    for(uint i = 0; i < size; ++i)
        sum2 += hist[i] * hist[i];
    if(sum2 < EPS)
       return hist;
    double sum = sqrt(sum2);
    for(uint i = 0; i < size; ++i)
       hist[i] = hist[i] / sum;

    return hist;
}

Histype calc_lbl(Image pic){

    Image neighbor = {{1, 1, 1},
                      {1, 0, 1},
                      {1, 1, 1}}; 
    Histype hist(256, 0.0);
    uint radius = neighbor.n_rows/2;
    uint nrows = pic.n_rows + 2*radius;
    uint ncols = pic.n_cols + 2*radius;
    Image tmp(nrows, ncols);
    //mirror borders
    for(uint i = 0; i < tmp.n_rows; ++i)
        for(uint j = 0; j < tmp.n_cols; ++j){
        auto x = i >= pic.n_rows + radius ? 2*(pic.n_rows-1)+radius - i : abs(radius - i);  
        auto y = j >= pic.n_cols + radius ? 2*(pic.n_cols-1)+radius - j : abs(radius - j);  
        tmp(i,j) = pic(x,y);
    }
    for(uint i = radius; i < pic.n_rows-radius; ++i)
        for(uint j = radius; j < pic.n_cols-radius; ++j){
            Image mask = tmp.submatrix(i-radius, j-radius, neighbor.n_rows, neighbor.n_cols);

            for(uint x = 0; x < neighbor.n_rows; ++x)
                for(uint y = 0; y < neighbor.n_cols; ++y){
                    if( mask(x, y) < mask(radius,radius) )
                        neighbor(x,y) = 0;
                    }
            auto vec = to_vector(neighbor);
            uint value = 0;
            for(uint k = 0; k < vec.size(); ++k)
                value = 2 * value + vec[k];
            hist[value]+=1.0;
        }
        
        //normalization
    const int size = 256;
    double sum2 = 0.0;
    for(uint i = 0; i < size; ++i)
        sum2 += hist[i] * hist[i];
    if(sum2 < EPS)
       return hist;
    double sum = sqrt(sum2);
    for(uint i = 0; i < size; ++i)
       hist[i] = hist[i] / sum;
    
    return hist;
}
Histype calc_color(Image3 pic){
    const int size = pic.n_cols * pic.n_rows;
    Histype hist(3, 0.0); // rgb
    
    int r_cl,g_cl,b_cl;
    vector<int> rHist(256, 0);
    vector<int> gHist(256, 0);
    vector<int> bHist(256, 0);
    for (uint i = 0; i < pic.n_rows; ++i)
        for(uint j = 0; j < pic.n_cols; ++j){
            std::tie(r_cl,g_cl,b_cl) = pic(i,j);
            rHist[r_cl]++;
            gHist[g_cl]++;
            bHist[b_cl]++;
        }

    int sum_r = 0, sum_g = 0, sum_b = 0;
    int r = 0, g = 0, b = 0; //median indexes
    for (int i = 0; i < 256; ++i){ //find median
        sum_r += rHist[i];
        sum_g += gHist[i];
        sum_b += bHist[i];
        if (sum_r > size/2){ //found
            r = i;
            sum_r = 0;
        }
        if (sum_g > size/2){
            g = i;
            sum_g = 0;
        }
        if (sum_b > size/2){
            b = i;
            sum_b = 0;
        }
    }
    hist[0] = r;
    hist[1] = g;
    hist[2] = b;
        //normalization
    double sum2  = r*r + g*g + b*b;
    if (sum2 < EPS) 
        return hist;
    double sum = sqrt(sum2);
    for(uint i = 0; i < 3; ++i)
        hist[i] = hist[i] / sum;

    return hist;
}
Image gradAbs(Image src){

	Image hor = sobelX(src);
    Image ver = sobelY(src);
    
    Image res(src.n_rows, src.n_cols);
	for(uint i = 0; i < hor.n_rows; ++i)
        for(uint j = 0; j < ver.n_cols; ++j)
            res(i,j) = sqrt(hor(i,j)*hor(i,j) + ver(i,j)*ver(i,j));
    
    return res;
}

Image gradDir(Image src){

	Image hor = sobelX(src);
	Image ver = sobelY(src);
	
	Image res(src.n_rows, src.n_cols);

	for(uint i = 0; i < hor.n_rows; ++i)
		for(uint j = 0; j < ver.n_cols; ++j)
			res(i,j) = std::atan2(ver(i,j), hor(i,j));

    return res;
}

Image sobelX(Image src) {

    Image kernel = {{-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}};
    return custom(src, kernel);
}

Image sobelY(Image src) {

    Image kernel = {{ 1,  2,  1},
                    { 0,  0,  0},
                    {-1, -2, -1}};
    return custom(src, kernel);
}

Image custom(Image src, Image ker){

    assert(ker.n_rows == ker.n_cols);
    uint radius = ker.n_rows/2;
    uint nrows = src.n_rows + 2*radius;
    uint ncols = src.n_cols + 2*radius;
    Image res(src.n_rows, src.n_cols);
    Image mask(ker.n_rows, ker.n_cols);
    Image tmp(nrows, ncols);

    //mirror borders
    for(uint i = 0; i < tmp.n_rows; ++i)
        for(uint j = 0; j < tmp.n_cols; ++j){
            auto x = i >= src.n_rows + radius ? 2*(src.n_rows-1)+radius - i : abs(radius - i);  
            auto y = j >= src.n_cols + radius ? 2*(src.n_cols-1)+radius - j : abs(radius - j);  
            tmp(i,j) = src(x,y);
        }

    for(uint i = radius; i < src.n_rows + radius; ++i)
        for(uint j = radius; j < src.n_cols + radius; ++j){
            mask = tmp.submatrix(i-radius, j-radius, mask.n_rows, mask.n_cols);
            
            double value = 0; 
            for(uint x = 0; x < mask.n_rows; ++x)
                for(uint y = 0; y < mask.n_cols; ++y)
                    value += mask(x,y)*ker(x,y);

            res(i-radius,j-radius) = value;
        }
    
    assert (res.n_rows == src.n_rows && res.n_cols == src.n_cols);
    return res;

}
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

Image grayScale (BMP* img){

    auto width = img->TellWidth();
    auto height = img->TellHeight();
    Image res = Image(width, height);
    
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
        {
            RGBApixel pix = img->GetPixel(i,j);
            res(i,j) = 0.299 * pix.Red + 0.587 * pix.Green + 0.114 * pix.Blue;
        }

    return res;

}

void splitInto(Image pic, Image* pieces, const int parts){

    int part = sqrt(parts);
    for(int i = 0; i < parts; ++i)
        pieces[i]=pic.submatrix((i%part)*pic.n_rows/part,(i/part)*pic.n_cols/part,pic.n_rows/part,pic.n_cols/part);
}

Histype calc_histo(Image gabs, Image gdir){

    const int size = 32;
    Histype hist(size, 0.0);
    for (uint i = 0; i < gdir.n_rows; ++i){
        for(uint j = 0; j < gdir.n_cols; ++j){
            uint k = uint((PI + gdir(i,j)/ (2*PI) * size)) % size; 
            hist[k] += gabs(i,j);
        }
    }


    //normalization
    double sum2=0.0;
    for(uint i =0; i < size; ++i)
        sum2 += hist[i] * hist[i];
    if(sum2 < EPS)
       return hist;
    double sum = sqrt(sum2);
    for(uint i = 0; i < size; ++i)
       hist[i] = hist[i] / sum;

    return hist;
}

Image gradAbs(Image src){

	Image width = sobelX(src);
    Image height = sobelY(src);
    
    Image res(src.n_rows, src.n_cols);
	for(uint i = 0; i < width.n_rows; ++i)
        for(uint j = 0; j < height.n_cols; ++j)
            res(i,j)=sqrt(width(i,j)*width(i,j) + height(i,j)*height(i,j));
    
    return res;
}

Image gradDir(Image src){

	Image width = sobelX(src);
	Image height = sobelY(src);
	
	Image res(src.n_rows, src.n_cols);

	for(uint i=0; i < width.n_rows; ++i)
		for(uint j=0; j < height.n_cols; ++j)
			res(i,j)=std::atan2(height(i,j), width(i,j));

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

    uint rrows = src.n_rows - ker.n_rows + 1;
    uint rcols = src.n_cols - ker.n_cols + 1;
    Image res(rrows, rcols);
    Image mask(ker.n_rows, ker.n_cols);


    for(uint i = 0; i < rrows; ++i)
        for(uint j = 0; j < rcols; ++j){
            mask = src.submatrix(i, j, mask.n_rows, mask.n_cols);
            
            double value = 0; 
            for(uint x = 0; x < mask.n_rows; ++x)
                for(uint y = 0; y < mask.n_cols; ++y)
                    value += mask(x,y)*ker(x,y);
            
            res(i,j) = value;
        }
        
    return res;
}
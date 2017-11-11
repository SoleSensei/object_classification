#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <stdint.h>
 

#include "classifier.h" 
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

#include "trainer.hpp"

using CommandLineProcessing::ArgvParser;


/**
@mainpage Objects classification using SSE. 
<p>Tested with GTest.
<p>Documented with Doxygen.
<p>on GitHub: [https://github.com/SoleSensei/object_classification]
@author This project was created by Sole Sensei.
*/

/**
@file main.cpp
*/

/**
@function LoadFileList
loads list of files and its labels from 'data_file' and stores it in 'file_list' 
@param data_file is string stores paths to BMP images and theirs labels
@param file_list is vector of <string pathname, int label>*/
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

/**
@function LoadImages
loads images by list of files 'file_list' and stores them in 'data_set' 
@param file_list is vector of <string pathname, int label>
@param data_set is vector of <BMP* image, int label> */
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            /// Create image
        BMP* image = new BMP();
            /// Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            /// Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

/**
@function SavePredictions 
saves result of prediction to file 
@param file_list is vector of <string pathname, int label>
@param labels is vector of <int label> 
@param prediction_file is output txt file to store classificator prediction*/
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

/**
@function ExtractFeatures
extracts features from dataset. Here all images from 'data_set' are processing
to build histograms 'features' in order to train model and predict classes
@param data_set is vector of <BMP* image, int label> 
@param features is vector of <Histype histogram, int labels> where Histype is vector <floats>
@param isSSE is boolean value shows type of next processing */
void ExtractFeatures(const TDataSet& data_set, TFeatures* features, bool isSSE) {

    Timer t;
    t.start();
    if (!isSSE){ //processing without SSE
        t.check("Naive implementation starts (no SSE)");
    for (vector <pair<BMP*,int>>::const_iterator ds = data_set.begin(); ds < data_set.end(); ++ds){
        BMP* src_image = (*ds).first;
        int label = (*ds).second;
        
            //BMP to image
        Image image = grayScale(src_image);
        Image3 image3 = to_image3(src_image);
        
            //splitting
        const int parts_num = 81; 
        Image cell[parts_num];
        Image3 color_cell[parts_num]; 
        splitInto(image, cell, parts_num); //image splitted on parts_num cells
        splitInto3(image3, color_cell, parts_num); //image3 splitted on parts_num color_cells
        
        Histype histo; //vector of histograms
        for(int i = 0; i < parts_num; ++i){
                //calculating sobel
            Image hor = sobelX(cell[i]);
            Image ver = sobelY(cell[i]);
                //calculating absolut and direction of gradient
            auto gabs = gradAbs(hor, ver);
            auto gdir = gradDir(hor, ver);
                //calculation HOG, LBL, color feautures         
            auto HOG = calc_hog(gabs, gdir); //one cell histogram calculation
            auto LBL = calc_lbl(cell[i]);
            auto COL = calc_color(color_cell[i]);
                //concatenation with other histograms
            histo.insert(histo.end(), HOG.begin(), HOG.end()); 
            histo.insert(histo.end(), LBL.begin(), LBL.end());
            histo.insert(histo.end(), COL.begin(), COL.end());
        }
        
        features->push_back(make_pair(histo, label));
                
    }
        t.check("Naive implementation complete (no SSE)");
    
    } else { 
    /*       using SSE in processing    */
        t.check("SSE implementation starts");    
    for (vector <pair<BMP*,int>>::const_iterator ds = data_set.begin(); ds < data_set.end(); ++ds){
        BMP* src_image = (*ds).first;
        int label = (*ds).second;

            //BMP to image
        Image image = grayScale(src_image);
        Image3 image3 = to_image3(src_image);
    
            //splitting
        const int parts_num = 81; 
        Image cell[parts_num];
        Image3 color_cell[parts_num]; 
        splitInto(image, cell, parts_num); //image splitted on parts_num cells
        splitInto3(image3, color_cell, parts_num); //image3 splitted on parts_num color_cells

        Histype histo; //vector of histograms
        for(int i = 0; i < parts_num; ++i){
            //calculating sobel
            Image hor = sobelX_SSE(cell[i]);
            Image ver = sobelY_SSE(cell[i]);
                //calculating absolut and direction of gradient
            auto gabs = gradAbs_SSE(hor, ver);
            auto gdir = gradDir(hor, ver);
                //calculation HOG, LBL, color feautures         
            auto HOG = calc_hog(gabs, gdir); //one cell histogram calculation
            auto LBL = calc_lbl(cell[i]);
            auto COL = calc_color(color_cell[i]);
                //concatenation with other histograms
            histo.insert(histo.end(), HOG.begin(), HOG.end()); 
            histo.insert(histo.end(), LBL.begin(), LBL.end());
            histo.insert(histo.end(), COL.begin(), COL.end());
        }
        
        features->push_back(make_pair(histo, label));

    }
        t.check("SSE implementation complete");    
    }
        t.stop();
}

/**
@function ClearDataset
clears dataset structure 
@param data_set is vector of <BMP* image, int label> */
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

/**
@function TrainClassifier
trains SVM classifier using data from 'data_file' and saves trained model to 'model_file'
@param data_set is vector of <BMP* image, int label>
@param model_file is output txt model file stores info about classificator traing
@param isSSE is boolean value shows type of next processing */
void TrainClassifier(const string& data_file, const string& model_file, bool isSSE) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features, isSSE);

        // You can change parameters of classifier here
    params.C = 0.1;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

/**
@function PredictData
predicts data from 'data_file' using model from 'model_file' and saves predictions to 'prediction_file'
@param data_set is vector of <BMP* image, int label>
@param model_file is output txt model file stores info about classificator traing
@param isSSE is boolean value shows type of next processing */
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file,
                 bool isSSE ) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features, isSSE);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2017.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
    cmd.defineOption("SSE", "Use SSE in classificator functions");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");
    cmd.defineOptionAlternative("SSE", "sse");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");
    bool isSSE = cmd.foundOption("sse");
        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file, isSSE);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file, isSSE);
    }
}
#ifndef ARCFACE_H
#define ARCFACE_H

#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <type_traits>
#include "mxnet/c_api.h"
#include "mxnet/tuple.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "mxnet-cpp/initializer.h"
// #include <experimental/filesystem>
#include <opencv2/opencv.hpp>

using namespace mxnet::cpp;
using namespace std;
using namespace cv;

class ArcFace
{
public:
    ArcFace(const Shape &input_shape = Shape(1, 3, 112, 112));
    ~ArcFace();
    inline bool FileExists(const std::string &name)
    {
        std::ifstream fhandle(name.c_str());
        return fhandle.good();
    }
    void initialize(std::string &param_file);
    void SplitParamMap(const std::map<std::string, NDArray> &paramMap,
                       std::map<std::string, NDArray> *argParamInTargetContext,
                       std::map<std::string, NDArray> *auxParamInTargetContext,
                       Context targetContext);
    std::vector<float> GetEmbedding(cv::Mat img);
    NDArray GetEmbeddingFromFile(const std::string &image_file);
    int GetDataLayerType();

private:
    void LoadModel(const std::string &model_json_file);
    void LoadParameters(const std::string &model_parameters_file);
    NDArray LoadInputImage(const std::string &image_file, const bool &flip);
    NDArray Preproecess(cv::Mat img, const bool &flip);
    Shape input_shape_;
    Symbol net;
    Executor *executor_;
    std::map<std::string, NDArray> args_map_;
    std::map<std::string, NDArray> aux_map_;
    Context global_ctx_ = Context::cpu();
};

#endif // ARCFACE_H
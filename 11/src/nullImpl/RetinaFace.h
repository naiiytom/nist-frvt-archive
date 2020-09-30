#ifndef RETINAFACE_H
#define RETINAFACE_H

#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <type_traits>
#include <opencv2/opencv.hpp>
#include "mxnet/c_api.h"
#include "mxnet/tuple.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "mxnet-cpp/initializer.h"
// #include <caffe/caffe.hpp>
// #include "tensorrt/trtretinafacenet.h"

using namespace cv;
using namespace std;
using namespace mxnet::cpp;

struct anchor_win
{
    float x_ctr;
    float y_ctr;
    float w;
    float h;
};

struct anchor_box
{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct FacePts
{
    float x[5];
    float y[5];
};

struct FaceDetectInfo
{
    float score;
    anchor_box rect;
    FacePts pts;
};

struct anchor_cfg
{
public:
    int STRIDE;
    vector<int> SCALES;
    int BASE_SIZE;
    vector<float> RATIOS;
    int ALLOWED_BORDER;

    anchor_cfg()
    {
        STRIDE = 0;
        SCALES.clear();
        BASE_SIZE = 0;
        RATIOS.clear();
        ALLOWED_BORDER = 0;
    }
};

class RetinaFace
{
public:
    RetinaFace(string network = "net3", float nms = 0.4,
               const Shape &input_shape = Shape(1, 3, 480, 640),
               const std::string &data_layer_type = "float32");
    ~RetinaFace();

    //     void detectBatchImages(vector<cv::Mat> imgs, float threshold=0.5);
    void initialize(string &param_file);
    std::vector<FaceDetectInfo> detect(Mat img, float threshold = 0.5, float scale = 1.0);

private:
    //     vector<FaceDetectInfo> postProcess(int inputW, int inputH, float threshold);
    void LoadModel(const std::string &model_json_file);
    void LoadParameters(const std::string &model_parameters_file);

    anchor_box bbox_pred(anchor_box anchor, cv::Vec4f regress);
    vector<anchor_box> bbox_pred(vector<anchor_box> anchors, vector<cv::Vec4f> regress);
    vector<FacePts> landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts);
    FacePts landmark_pred(anchor_box anchor, FacePts facePt);
    static bool CompareBBox(const FaceDetectInfo &a, const FaceDetectInfo &b);
    std::vector<FaceDetectInfo> nms(std::vector<FaceDetectInfo> &bboxes, float threshold);
    void SplitParamMap(const std::map<std::string, NDArray> &paramMap,
                       std::map<std::string, NDArray> *argParamInTargetContext,
                       std::map<std::string, NDArray> *auxParamInTargetContext,
                       Context targetContext);

    inline bool FileExists(const std::string &name)
    {
        std::ifstream fhandle(name.c_str());
        return fhandle.good();
    }

    int GetDataLayerType();

private:
    //     boost::shared_ptr<Net<float> > Net_;
    Symbol Net_;
    Executor *executor_;
    std::string data_layer_type_;
    std::map<std::string, NDArray> args_map_;
    std::map<std::string, NDArray> aux_map_;

    //     TrtRetinaFaceNet *trtNet;

    float pixel_means[3] = {0.0, 0.0, 0.0};
    float pixel_stds[3] = {1.0, 1.0, 1.0};
    float pixel_scale = 1.0;

    Context global_ctx_ = Context::cpu();
    string network;
    float decay4;
    float nms_threshold;
    Shape input_shape_;
    bool vote;
    bool nocrop;

    vector<float> _ratio;
    vector<anchor_cfg> cfg;

    vector<int> _feat_stride_fpn;
    //每一层fpn的anchor形状
    map<string, vector<anchor_box>> _anchors_fpn;
    //每一层所有点的anchor
    map<string, vector<anchor_box>> _anchors;
    //每一层fpn有几种形状的anchor
    //也就是ratio个数乘以scales个数
    map<string, int> _num_anchors;
};

#endif // RETINAFACE_H
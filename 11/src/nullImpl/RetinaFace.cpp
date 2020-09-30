#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <type_traits>
#include <opencv2/opencv.hpp>
#include <typeinfo>
#include "mxnet/c_api.h"
#include "mxnet/tuple.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "mxnet-cpp/initializer.h"

#include "RetinaFace.h"
// #include <cuda_runtime_api.h>

using namespace mxnet::cpp;
using namespace cv;

//processing
anchor_win _whctrs(anchor_box anchor)
{
    //Return width, height, x center, and y center for an anchor (window).
    anchor_win win;
    win.w = anchor.x2 - anchor.x1 + 1;
    win.h = anchor.y2 - anchor.y1 + 1;
    win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
    win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

    return win;
}

anchor_box _mkanchors(anchor_win win)
{
    //Given a vector of widths (ws) and heights (hs) around a center
    //(x_ctr, y_ctr), output a set of anchors (windows).
    anchor_box anchor;
    anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
    anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
    anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
    anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

    return anchor;
}

vector<anchor_box> _ratio_enum(anchor_box anchor, vector<float> ratios)
{
    //Enumerate a set of anchors for each aspect ratio wrt an anchor.
    vector<anchor_box> anchors;
    for (size_t i = 0; i < ratios.size(); i++)
    {
        anchor_win win = _whctrs(anchor);
        float size = win.w * win.h;
        float scale = size / ratios[i];

        win.w = std::round(sqrt(scale));
        win.h = std::round(win.w * ratios[i]);

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

vector<anchor_box> _scale_enum(anchor_box anchor, vector<int> scales)
{
    //Enumerate a set of anchors for each scale wrt an anchor.
    vector<anchor_box> anchors;
    for (size_t i = 0; i < scales.size(); i++)
    {
        anchor_win win = _whctrs(anchor);

        win.w = win.w * scales[i];
        win.h = win.h * scales[i];

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

vector<anchor_box> generate_anchors(int base_size = 16, vector<float> ratios = {0.5, 1, 2},
                                    vector<int> scales = {8, 64}, int stride = 16, bool dense_anchor = false)
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    anchor_box base_anchor;
    base_anchor.x1 = 0;
    base_anchor.y1 = 0;
    base_anchor.x2 = base_size - 1;
    base_anchor.y2 = base_size - 1;

    vector<anchor_box> ratio_anchors;
    ratio_anchors = _ratio_enum(base_anchor, ratios);

    vector<anchor_box> anchors;
    for (size_t i = 0; i < ratio_anchors.size(); i++)
    {
        vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }

    if (dense_anchor)
    {
        assert(stride % 2 == 0);
        vector<anchor_box> anchors2 = anchors;
        for (size_t i = 0; i < anchors2.size(); i++)
        {
            anchors2[i].x1 += stride / 2;
            anchors2[i].y1 += stride / 2;
            anchors2[i].x2 += stride / 2;
            anchors2[i].y2 += stride / 2;
        }
        anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
    }

    return anchors;
}

vector<vector<anchor_box>> generate_anchors_fpn(bool dense_anchor = false, vector<anchor_cfg> cfg = {})
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    vector<vector<anchor_box>> anchors;
    for (size_t i = 0; i < cfg.size(); i++)
    {
        //stride从小到大[32 16 8]
        anchor_cfg tmp = cfg[i];
        int bs = tmp.BASE_SIZE;
        vector<float> ratios = tmp.RATIOS;
        vector<int> scales = tmp.SCALES;
        int stride = tmp.STRIDE;

        vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
        anchors.push_back(r);
    }

    return anchors;
}

vector<anchor_box> anchors_plane(int height, int width, int stride, vector<anchor_box> base_anchors)
{
    /*
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: a base set of anchors
    */

    vector<anchor_box> all_anchors;
    for (size_t k = 0; k < base_anchors.size(); k++)
    {
        for (int ih = 0; ih < height; ih++)
        {
            int sh = ih * stride;
            for (int iw = 0; iw < width; iw++)
            {
                int sw = iw * stride;

                anchor_box tmp;
                tmp.x1 = base_anchors[k].x1 + sw;
                tmp.y1 = base_anchors[k].y1 + sh;
                tmp.x2 = base_anchors[k].x2 + sw;
                tmp.y2 = base_anchors[k].y2 + sh;
                all_anchors.push_back(tmp);
            }
        }
    }

    return all_anchors;
}

void clip_boxes(vector<anchor_box> &boxes, int width, int height)
{
    //Clip boxes to image boundaries.
    for (size_t i = 0; i < boxes.size(); i++)
    {
        if (boxes[i].x1 < 0)
        {
            boxes[i].x1 = 0;
        }
        if (boxes[i].y1 < 0)
        {
            boxes[i].y1 = 0;
        }
        if (boxes[i].x2 > width - 1)
        {
            boxes[i].x2 = width - 1;
        }
        if (boxes[i].y2 > height - 1)
        {
            boxes[i].y2 = height - 1;
        }
        //        boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
        //        boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
        //        boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
        //        boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);
    }
}

void clip_boxes(anchor_box &box, int width, int height)
{
    //Clip boxes to image boundaries.
    if (box.x1 < 0)
    {
        box.x1 = 0;
    }
    if (box.y1 < 0)
    {
        box.y1 = 0;
    }
    if (box.x2 > width - 1)
    {
        box.x2 = width - 1;
    }
    if (box.y2 > height - 1)
    {
        box.y2 = height - 1;
    }
    //    boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
    //    boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
    //    boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
    //    boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);
}

enum TypeFlag
{
    kFloat32 = 0,
    kFloat64 = 1,
    kFloat16 = 2,
    kUint8 = 3,
    kInt32 = 4,
    kInt8 = 5,
    kInt64 = 6,
};

//######################################################################
//retinaface
//######################################################################

RetinaFace::RetinaFace(string network, float nms, const Shape &input_shape,
                       const std::string &data_layer_type)
    : input_shape_(input_shape),
      network(network),
      nms_threshold(nms),
      data_layer_type_(data_layer_type)
{
}

void RetinaFace::initialize(string &param_file)
{
    //主干网络选择
    int fmc = 3;

    if (network == "ssh" || network == "vgg")
    {
        pixel_means[0] = 103.939;
        pixel_means[1] = 116.779;
        pixel_means[2] = 123.68;
    }
    else if (network == "net3")
    {
        _ratio = {1.0};
    }
    else if (network == "net3a")
    {
        _ratio = {1.0, 1.5};
    }
    else if (network == "net6")
    { //like pyramidbox or s3fd
        fmc = 6;
    }
    else if (network == "net5")
    { //retinaface
        fmc = 5;
    }
    else if (network == "net5a")
    {
        fmc = 5;
        _ratio = {1.0, 1.5};
    }

    else if (network == "net4")
    {
        fmc = 4;
    }
    else if (network == "net5a")
    {
        fmc = 4;
        _ratio = {1.0, 1.5};
    }
    else
    {
        // std::cout << "network setting error" << network << std::endl;
    }

    //anchor配置
    if (fmc == 3)
    {
        _feat_stride_fpn = {32, 16, 8};
        anchor_cfg tmp;
        tmp.SCALES = {32, 16};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 32;
        cfg.push_back(tmp);

        tmp.SCALES = {8, 4};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 16;
        cfg.push_back(tmp);

        tmp.SCALES = {2, 1};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 8;
        cfg.push_back(tmp);
    }
    else
    {
        // std::cout << "please reconfig anchor_cfg" << network << std::endl;
    }

    std::string token = param_file.substr(0, param_file.rfind("-"));
    LoadModel(token + "-symbol.json");
    LoadParameters(param_file);

    bool dense_anchor = false;
    vector<vector<anchor_box>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
    for (size_t i = 0; i < anchors_fpn.size(); i++)
    {
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        _anchors_fpn[key] = anchors_fpn[i];
        _num_anchors[key] = anchors_fpn[i].size();
    }

    int dtype = GetDataLayerType();
    if (dtype == -1)
    {
        throw std::runtime_error("Unsupported data layer type...");
    }

    args_map_["data"] = NDArray(input_shape_, global_ctx_, false, dtype);
    //     Shape label_shape(input_shape_[0]);
    //     args_map_["softmax_label"] = NDArray(label_shape, global_ctx_, false);
    std::vector<NDArray> arg_arrays;
    std::vector<NDArray> grad_arrays;
    std::vector<OpReqType> grad_reqs;
    std::vector<NDArray> aux_arrays;

    // infer and create ndarrays according to the given input ndarrays.
    Net_.InferExecutorArrays(global_ctx_, &arg_arrays, &grad_arrays, &grad_reqs,
                             &aux_arrays, args_map_, std::map<std::string, NDArray>(),
                             std::map<std::string, OpReqType>(), aux_map_);
    for (auto &i : grad_reqs)
        i = OpReqType::kNullOp;

    // Create an executor after binding the model to input parameters.
    executor_ = new Executor(Net_, global_ctx_, arg_arrays, grad_arrays, grad_reqs, aux_arrays);
}

int RetinaFace::GetDataLayerType()
{
    int ret_type = -1;
    if (data_layer_type_ == "float32")
    {
        ret_type = kFloat32;
    }
    else if (data_layer_type_ == "int8")
    {
        ret_type = kInt8;
    }
    else if (data_layer_type_ == "uint8")
    {
        ret_type = kUint8;
    }
    else
    {
        // LG << "Unsupported data layer type " << data_layer_type_ << "..." << "Please use one of {float32, int8, uint8}";
    }
    return ret_type;
}

void RetinaFace::LoadModel(const std::string &model_json_file)
{

    if (!FileExists(model_json_file))
    {
        // LG << "Model file " << model_json_file << " does not exist";
        throw std::runtime_error("Model file does not exist");
    }
    // LG << "Loading the model from " << model_json_file << std::endl;
    Net_ = Symbol::Load(model_json_file);
}

void RetinaFace::LoadParameters(const std::string &model_parameters_file)
{

    if (!FileExists(model_parameters_file))
    {
        // LG << "Parameter file " << model_parameters_file << " does not exist";
        throw std::runtime_error("Model parameters does not exist");
    }
    // LG << "Loading the model parameters from " << model_parameters_file << std::endl;
    std::map<std::string, NDArray> parameters;
    NDArray::Load(model_parameters_file, 0, &parameters);
    SplitParamMap(parameters, &args_map_, &aux_map_, global_ctx_);

    /*WaitAll is need when we copy data between GPU and the main memory*/
    NDArray::WaitAll();
}

/*
 * The following function split loaded param map into arg parm
 *   and aux param with target context
 */
void RetinaFace::SplitParamMap(const std::map<std::string, NDArray> &paramMap,
                               std::map<std::string, NDArray> *argParamInTargetContext,
                               std::map<std::string, NDArray> *auxParamInTargetContext,
                               Context targetContext)
{

    for (const auto &pair : paramMap)
    {
        std::string type = pair.first.substr(0, 4);
        std::string name = pair.first.substr(4);
        if (type == "arg:")
        {
            (*argParamInTargetContext)[name] = pair.second.Copy(targetContext);
        }
        else if (type == "aux:")
        {
            (*auxParamInTargetContext)[name] = pair.second.Copy(targetContext);
        }
    }
}

RetinaFace::~RetinaFace()
{
    // #ifdef USE_TENSORRT
    //     delete trtNet;
    //     free(cpuBuffers);
    // #endif

    if (executor_)
    {
        delete executor_;
    }

    MXNotifyShutdown();
}

vector<anchor_box> RetinaFace::bbox_pred(std::vector<anchor_box> anchors, std::vector<cv::Vec4f> regress)
{
    //"""
    //  Transform the set of class-agnostic boxes into class-specific boxes
    //  by applying the predicted offsets (box_deltas)
    //  :param boxes: !important [N 4]
    //  :param box_deltas: [N, 4 * num_classes]
    //  :return: [N 4 * num_classes]
    //  """

    std::vector<anchor_box> rects(anchors.size());
    for (size_t i = 0; i < anchors.size(); i++)
    {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        float pred_ctr_x = regress[i][0] * width + ctr_x;
        float pred_ctr_y = regress[i][1] * height + ctr_y;
        float pred_w = exp(regress[i][2]) * width;
        float pred_h = exp(regress[i][3]) * height;

        rects[i].x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
        rects[i].y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
        rects[i].x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
        rects[i].y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
    }

    return rects;
}

anchor_box RetinaFace::bbox_pred(anchor_box anchor, cv::Vec4f regress)
{
    anchor_box rect;

    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = exp(regress[2]) * width;
    float pred_h = exp(regress[3]) * height;

    rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    return rect;
}

vector<FacePts> RetinaFace::landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts)
{
    vector<FacePts> pts(anchors.size());
    for (size_t i = 0; i < anchors.size(); i++)
    {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        for (size_t j = 0; j < 5; j++)
        {
            pts[i].x[j] = facePts[i].x[j] * width + ctr_x;
            pts[i].y[j] = facePts[i].y[j] * height + ctr_y;
        }
    }

    return pts;
}

FacePts RetinaFace::landmark_pred(anchor_box anchor, FacePts facePt)
{
    FacePts pt;
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    for (size_t j = 0; j < 5; j++)
    {
        pt.x[j] = facePt.x[j] * width + ctr_x;
        pt.y[j] = facePt.y[j] * height + ctr_y;
    }

    return pt;
}

bool RetinaFace::CompareBBox(const FaceDetectInfo &a, const FaceDetectInfo &b)
{
    return a.score > b.score;
}

std::vector<FaceDetectInfo> RetinaFace::nms(std::vector<FaceDetectInfo> &bboxes, float threshold)
{
    std::vector<FaceDetectInfo> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged)
    {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        //如果全部执行完则返回
        if (select_idx == num_bbox)
        {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        anchor_box select_bbox = bboxes[select_idx].rect;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++)
        {
            if (mask_merged[i] == 1)
                continue;

            anchor_box &bbox_i = bboxes[i].rect;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1; //<- float 型不加1
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;

            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold)
            {
                mask_merged[i] = 1;
            }
        }
    }

    return bboxes_nms;
}

std::vector<FaceDetectInfo> RetinaFace::detect(Mat img, float threshold, float scale)
{

    //     executor_->~Executor();

    std::vector<FaceDetectInfo> faceInfo = {};

    if (img.empty())
    {
        return faceInfo;
    }

    cv::Mat im = img.clone();

    // std::cout << "Image type: " << im.type() << std::endl;

    //if (scale != 1.0) {
    //    cv::resize(im, im, cv::Size(), scale, scale, CV_INTER_LINEAR);
    //}

    // Resizing image to specifice 640x480 px
    cv::resize(im, im, cv::Size(480, 640), 0, 0, CV_INTER_LINEAR);

    // im.convertTo(im, CV_32FC3);
    cv::cvtColor(im, im, CV_BGR2RGB);

    // debugging img size
    // std::cout << "img height: " << im.size().height << "\n";
    // std::cout << "img width: " << im.size().width << "\n";
    //
    std::vector<index_t> image_dimensions{1, 3, static_cast<uint16_t>(im.size().height), static_cast<uint16_t>(im.size().width)};
    Shape image_shape(image_dimensions);

    std::vector<float> array;
    array.reserve(im.total() * im.elemSize());

    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < im.size().height; ++i)
        {
            for (int j = 0; j < im.size().width; ++j)
            {

                int imDataIndex = (i * im.size().width + j) * 3 + (2 - c);

                if (imDataIndex < im.total() * im.elemSize())
                {
                    array.push_back(static_cast<float>(im.data[imDataIndex]));
                }
            }
        }
    }

    int dtype = GetDataLayerType();
    NDArray data = NDArray(image_shape, global_ctx_, false);

    // std::cout << "Input shape: " << image_shape << std::endl;

    args_map_["data"] = NDArray(image_shape, global_ctx_, false, dtype);

    Executor *executor;
    executor = Net_.SimpleBind(global_ctx_, args_map_, std::map<std::string, NDArray>(),
                               std::map<std::string, OpReqType>(), aux_map_);

    data.SyncCopyFromCPU(array.data(), image_shape.Size());
    NDArray::WaitAll();

    // LG << "Running the forward pass on model to predict the image";

    data.CopyTo(&(executor->arg_dict()["data"]));
    NDArray::WaitAll();

    // Run the forward pass.
    executor->Forward(false);

    //     // The output is available in executor->outputs.
    auto outputs = executor->outputs;
    NDArray::WaitAll();

    int idx_ = 0;
    int idx = 0;

    for (int s : _feat_stride_fpn)
    {

        // for use landmark
        idx = idx_ * 3;

        auto output = executor->outputs[idx].Copy(global_ctx_);
        NDArray::WaitAll();

        auto output_shape = output.GetShape();
        int output_1d_shape_ = 1;

        for (int i = 0; i < output_shape.size(); i++)
        {
            output_1d_shape_ *= output_shape.at(i);
        }

        Shape output_1d_shape(output_1d_shape_);

        auto scores_ = output.Reshape(output_1d_shape);
        std::vector<float> scores;
        string key = "stride" + std::to_string(s);

        std::vector<mx_uint> scores_shape = output_shape;

        for (int i = 0; i < scores_shape.at(0); i++)
        {
            for (int j = _num_anchors[key]; j < scores_shape.at(1); j++)
            {
                for (int k = 0; k < scores_shape.at(2); k++)
                {
                    for (int l = 0; l < scores_shape.at(3); l++)
                    {

                        int index = i * scores_shape.at(1) * scores_shape.at(2) * scores_shape.at(3) +
                                    j * scores_shape.at(2) * scores_shape.at(3) + k * scores_shape.at(3) + l;

                        scores.push_back(static_cast<float>(scores_.At(index)));
                    }
                }
            }
        }

        idx += 1;

        auto bbox_deltas_ = executor->outputs[idx].Copy(global_ctx_);
        NDArray::WaitAll();
        std::vector<float> bbox_deltas(bbox_deltas_.Size());
        bbox_deltas_.SyncCopyToCPU(bbox_deltas.data(), bbox_deltas_.Size());
        NDArray::WaitAll();

        idx += 1;

        auto landmark_deltas_ = executor->outputs[idx].Copy(global_ctx_);
        NDArray::WaitAll();
        std::vector<float> landmark_deltas(landmark_deltas_.Size());
        landmark_deltas_.SyncCopyToCPU(landmark_deltas.data(), landmark_deltas_.Size());
        NDArray::WaitAll();

        int height = bbox_deltas_.GetShape().at(2);
        int width = bbox_deltas_.GetShape().at(3);

        int A = _num_anchors[key];
        int K = height * width;

        std::vector<anchor_box> anchors = anchors_plane(height, width, s, _anchors_fpn[key]);

        for (size_t num = 0; num < A; num++)
        {
            for (size_t j = 0; j < K; j++)
            {

                float conf = scores[j + K * num];

                //                 std::cout << conf << std::endl;

                if (conf <= threshold)
                {
                    continue;
                }

                cv::Vec4f regress;
                float dx = bbox_deltas[j + K * (0 + num * 4)];
                float dy = bbox_deltas[j + K * (1 + num * 4)];
                float dw = bbox_deltas[j + K * (2 + num * 4)];
                float dh = bbox_deltas[j + K * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);

                //                 //回归人脸框
                anchor_box rect = bbox_pred(anchors[j + K * num], regress);

                //越界处理
                clip_boxes(rect, im.size().width, im.size().height);

                FacePts pts;
                for (size_t k = 0; k < 5; k++)
                {
                    pts.x[k] = landmark_deltas[j + K * (num * 10 + k * 2)];
                    pts.y[k] = landmark_deltas[j + K * (num * 10 + k * 2 + 1)];
                }
                //回归人脸关键点
                FacePts landmarks = landmark_pred(anchors[j + K * num], pts);

                FaceDetectInfo tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                faceInfo.push_back(tmp);

                //                 std::cout << rect << std::endl;
            }
        }

        bbox_deltas.clear();
        scores.clear();
        scores_shape.clear();
        landmark_deltas.clear();
        anchors.clear();

        idx_++;
    }

    faceInfo = nms(faceInfo, nms_threshold);

    // Clear memory
    array.clear();
    im.release();
    delete executor;

    // for (auto tmpp : faceInfo)
    // {
    //     std::cout << tmpp.score << " scores!!!!" << std::endl;
    //     std::cout << tmpp.rect.x1 << ", " << tmpp.rect.y1 << " | " << tmpp.rect.x2 << ", " << tmpp.rect.y2 << std::endl;
    //     for (int i = 0; i < 5; i++)
    //     {
    //         std::cout << tmpp.pts.x[i] << ", " << tmpp.pts.y[i] << " | ";
    //     }
    //     std::cout << "" << std::endl;
    // }

    return faceInfo;
}

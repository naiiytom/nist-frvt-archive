#ifndef FACEALIGN_H
#define FACEALIGN_H

#include <opencv2/opencv.hpp>

#include "RetinaFace.h"

cv::Mat norm_crop(Mat img, FacePts landmark, cv::Size imageSize = cv::Size(112, 112));

cv::Mat norm_crop_temp(Mat img, anchor_box bbox, cv::Size imageSize = cv::Size(112, 112));

#endif
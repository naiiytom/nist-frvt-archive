#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"

#include "RetinaFace.h"

float v1[5][2] = {
    {30.2946f + 8.0, 51.6963f},
    {65.5318f + 8.0, 51.5014f},
    {48.0252f + 8.0, 71.7366f},
    {33.5493f + 8.0, 92.3655f},
    {62.7299f + 8.0, 92.2041f}};

cv::Mat norm_crop(Mat img, FacePts landmark, cv::Size imageSize)
{

    cv::Mat src(5, 2, CV_32FC1, v1);
    memcpy(src.data, v1, 2 * 5 * sizeof(float));

    float v2[5][2];

    for (int i = 0; i < 5; i++)
    {
        v2[i][0] = landmark.x[i];
        v2[i][1] = landmark.y[i];
    }

    cv::Mat dst(5, 2, CV_32FC1, v2);
    memcpy(dst.data, v2, 2 * 5 * sizeof(float));

    cv::Mat M = cv::estimateRigidTransform(dst, src, true);

    cv::Mat img_aligned;
    cv::warpAffine(img, img_aligned, M, imageSize);

    return img_aligned;
}

cv::Mat norm_crop_temp(Mat img, anchor_box bbox, cv::Size imageSize)
{

    cv::Rect myROI(static_cast<int>(bbox.x1), static_cast<int>(bbox.y1), static_cast<int>(bbox.x2 - bbox.x1), static_cast<int>(bbox.y2 - bbox.y1));
    cv::Mat croppedImage = img(myROI);
    cv::resize(croppedImage, croppedImage, imageSize, 1.0, 1.0, CV_INTER_LINEAR);

    return croppedImage;
}
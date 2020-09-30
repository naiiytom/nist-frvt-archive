#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <type_traits>
// #include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "FacePreprocess.h"

// namespace fs = std::experimental::filesystem;
using namespace std;
using namespace cv;

int main()
{
  std::string img_file;
  img_file = "./donald_trump_1.jpg";
  float v1[5][2] = {
      {30.2946f + 8.0, 51.6963f},
      {65.5318f + 8.0, 51.5014f},
      {48.0252f + 8.0, 71.7366f},
      {33.5493f + 8.0, 92.3655f},
      {62.7299f + 8.0, 92.2041f}};

  cv::Mat src(5, 2, CV_32FC1, v1);
  memcpy(src.data, v1, 2 * 5 * sizeof(float));

  float v2[5][2] = {
      {112.87324f, 142.33607f},
      {293.43738f, 146.33543f},
      {196.80606f, 244.43976f},
      {121.66194f, 322.76303f},
      {278.65118f, 326.35095f}};

  cv::Mat dst(5, 2, CV_32FC1, v2);
  memcpy(dst.data, v2, 2 * 5 * sizeof(float));

  // cout << dst;

  // cv::Mat M = FacePreprocess::similarTransform(src, dst);
  cv::Mat M = cv::estimateRigidTransform(dst, src, true);
  // cout << M;

  Mat img = cv::imread(img_file);
  Mat img_aligned;
  warpAffine(img, img_aligned, M, cv::Size(112, 112));

  imwrite("test.jpg", img_aligned);
}
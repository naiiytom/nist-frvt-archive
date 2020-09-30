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
#include <opencv2/opencv.hpp>

#include "ArcFace.h"

using namespace mxnet::cpp;
using namespace std;
using namespace cv;

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

int ArcFace::GetDataLayerType()
{
  int ret_type = kFloat32;
  return ret_type;
}

void ArcFace::LoadModel(const std::string &model_json_file)
{
  if (!FileExists(model_json_file))
  {
    // LG << "Model file " << model_json_file << " does not exist";
    throw std::runtime_error("Model file does not exist");
  }
  // LG << "Loading the model from " << model_json_file << std::endl;
  net = Symbol::Load(model_json_file);
}

/*
 * The following function loads the model parameters.
 */
void ArcFace::LoadParameters(const std::string &model_parameters_file)
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
void ArcFace::SplitParamMap(const std::map<std::string, NDArray> &paramMap,
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

NDArray ArcFace::LoadInputImage(const std::string &image_file,
                                const bool &flip)
{
  if (!FileExists(image_file))
  {
    // LG << "Image file " << image_file << " does not exist";
    throw std::runtime_error("Image file does not exist");
  }
  // LG << "Loading the image " << image_file << std::endl;
  cv::Mat mat = cv::imread(image_file);
  return Preproecess(mat, flip);
}

NDArray ArcFace::Preproecess(cv::Mat img, const bool &flip)
{

  cv::Mat im;

  if (flip == true)
  {
    cv::flip(img, im, 1);
  }
  else
  {
    im = img.clone();
  }

  cv::cvtColor(im, im, CV_BGR2RGB);

  /*resize pictures to (112, 112) according to the pretrained model*/
  // LG << "Loading the image " << input_shape_ << std::endl;
  int height = input_shape_[2];
  int width = input_shape_[3];
  int channels = input_shape_[1];
  cv::resize(im, im, cv::Size(height, width));

  std::vector<float> array;

  for (int c = 0; c < 3; ++c)
  {
    for (int i = 0; i < im.size().height; ++i)
    {
      for (int j = 0; j < im.size().width; ++j)
      {
        //                 array.push_back(static_cast<float>(mat.data[(i * mat.size().width + j) * 3 + (2-c)]));
        array.push_back(static_cast<float>(im.data[(i * im.size().width + j) * 3 + c]));
      }
    }
  }

  NDArray image_data = NDArray(input_shape_, global_ctx_, false);
  image_data.SyncCopyFromCPU(array.data(), input_shape_.Size());
  NDArray::WaitAll();

  // Clear memory
  array.clear();
  im.release();

  return image_data;
}

std::vector<float> ArcFace::GetEmbedding(Mat img)
{

  //     executor_->~Executor();

  // Load the input image
  NDArray image_data_1 = Preproecess(img, false);

  // LG << "Running the forward pass on model to predict the image" << std::endl;

  image_data_1.CopyTo(&(executor_->arg_dict()["data"]));
  NDArray::WaitAll();

  // Run the forward pass.
  /*bind the executor*/
  executor_->Forward(false);

  // The output is available in executor->outputs.
  auto array_1 = executor_->outputs[0].Copy(global_ctx_);

  NDArray::WaitAll();

  auto embedding_ = array_1;

  std::vector<float> embedding(embedding_.Size());
  embedding_.SyncCopyToCPU(embedding.data(), embedding_.Size());
  NDArray::WaitAll();

  // Clear memory
  //     arg_arrays.clear();
  //     grad_arrays.clear();
  //     grad_reqs.clear();
  //     aux_arrays.clear();
  //     delete executor_;

  return embedding;
}

NDArray ArcFace::GetEmbeddingFromFile(const std::string &image_file)
{
  // Load the input image
  NDArray image_data_1 = LoadInputImage(image_file, false);
  //   NDArray image_data_2 = LoadInputImage(image_file, false);

  // image_data_1.Slice(0, 1) /= 255.;
  // image_data_2.Slice(0, 1) /= 255.;

  // LG << "Running the forward pass on model to predict the image" << std::endl;

  image_data_1.CopyTo(&(executor_->arg_dict()["data"]));
  NDArray::WaitAll();

  // Run the forward pass.
  /*bind the executor*/
  executor_->Forward(false);

  // The output is available in executor->outputs.
  auto array_1 = executor_->outputs[0].Copy(global_ctx_);

  NDArray::WaitAll();

  //   // Create an executor after binding the model to input parameters.
  //   executor_ = new Executor(net, global_ctx_, arg_arrays, grad_arrays, grad_reqs, aux_arrays);

  //   image_data_2.CopyTo(&(executor_->arg_dict()["data"]));
  //   NDArray::WaitAll();

  //   // Run the forward pass.
  //   /*bind the executor*/
  //   executor_->Forward(false);

  //   // The output is available in executor->outputs.
  //   auto array_2 = executor_-> outputs[0].Copy(global_ctx_);
  //   NDArray::WaitAll();

  // //   std::cout << array_1 << std::endl;

  //   auto embedding = array_1 + array_2;
  auto embedding = array_1;
  return embedding;
}

ArcFace::ArcFace(const Shape &input_shape) : input_shape_(input_shape)
{
}

void ArcFace::initialize(std::string &param_file)
{

  std::string token = param_file.substr(0, param_file.rfind("-"));

  // Load the model
  LoadModel(token + "-symbol.json");

  // Load the model parameters.
  LoadParameters(param_file);

  args_map_["data"] = NDArray(input_shape_, global_ctx_, false, kFloat32);

  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<OpReqType> grad_reqs;
  std::vector<NDArray> aux_arrays;

  // infer and create ndarrays according to the given input ndarrays.
  net.InferExecutorArrays(global_ctx_, &arg_arrays, &grad_arrays, &grad_reqs,
                          &aux_arrays, args_map_, std::map<std::string, NDArray>(),
                          std::map<std::string, OpReqType>(), aux_map_);

  for (auto &i : grad_reqs)
    i = OpReqType::kNullOp;

  // Create an executor after binding the model to input parameters.
  executor_ = new Executor(net, global_ctx_, arg_arrays, grad_arrays, grad_reqs, aux_arrays);
}

ArcFace::~ArcFace()
{

  if (executor_)
  {
    delete executor_;
  }

  MXNotifyShutdown();
}

/*
 * Convert the input string of number into the vector.
 */
template <typename T>
std::vector<T> createVectorFromString(const std::string &input_string)
{
  std::vector<T> dst_vec;
  char *p_next;
  T elem;
  bool bFloat = std::is_same<T, float>::value;
  if (!bFloat)
  {
    elem = strtol(input_string.c_str(), &p_next, 10);
  }
  else
  {
    elem = strtof(input_string.c_str(), &p_next);
  }

  dst_vec.push_back(elem);
  while (*p_next)
  {
    if (!bFloat)
    {
      elem = strtol(p_next, &p_next, 10);
    }
    else
    {
      elem = strtof(p_next, &p_next);
    }
    dst_vec.push_back(elem);
  }
  return dst_vec;
}

double dotProduct(NDArray A, NDArray B, int Vector_Length)
{
  A = A.Reshape({1, 512});
  B = B.Reshape({1, 512});
  double product = 0;
  for (int i = 0; i < 512; ++i)
  {
    product += +A.At(0, i) * B.At(0, i);
  }
  return product;
}

double cosineSimilarity(NDArray A, NDArray B, int Vector_Length)
{
  double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
  A = A.Reshape({1, 512});
  B = B.Reshape({1, 512});
  for (int i = 0; i < 512; ++i)
  {
    dot += A.At(0, i) * B.At(0, i);
    denom_a += A.At(0, i) * A.At(0, i);
    denom_b += B.At(0, i) * B.At(0, i);
  }
  return dot / (sqrt(denom_a) * sqrt(denom_b));
}

// int main() {
//   std::string model_file_json;
//   std::string model_file_params;
//   std::string img_file_path_1;
//   std::string img_file_path_2;
//   int batch_size = 1;

//   model_file_json = "./model/model-symbol.json";
//   model_file_params = "./model/model-0000.params";
//   img_file_path_1 = "./donald_trump_1.jpg";
//   img_file_path_2 = "./unknow_person.jpg";

//   std::vector<index_t> input_dimensions = createVectorFromString<index_t>("3 112 112");
//   input_dimensions.insert(input_dimensions.begin(), batch_size);
//   Shape input_data_shape(input_dimensions);

//   ArcFace arcface(model_file_json, model_file_params, input_data_shape);
//   NDArray embedding1 = arcface.GetEmbedding(img_file_path_1);
//   NDArray embedding2 = arcface.GetEmbedding(img_file_path_2);

//   double dot = dotProduct(embedding1, embedding2, 512);
//   double sim = cosineSimilarity(embedding1, embedding2, 512);
// //   LG << embedding1 << std::endl;
// //   LG << embedding2 << std::endl;
// //   LG << sim << std::endl;
// }
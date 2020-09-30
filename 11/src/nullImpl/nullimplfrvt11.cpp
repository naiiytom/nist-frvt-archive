/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility  whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#include <cstring>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <exception>

#include <string>
#include <iterator>

#include "nullimplfrvt11.h"
#include "RetinaFace.h"
#include "FaceAlign.h"

using namespace std;
using namespace FRVT;
using namespace FRVT_11;

double cosineSimilarity(float *A, float *B, unsigned int Vector_Length, unsigned int start)
{
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (unsigned int i = start; i < start + Vector_Length; ++i)
    {
        dot += A[i] * B[i];
        denom_a += A[i] * A[i];
        denom_b += B[i] * B[i];
    }

    double similarity = dot / (sqrt(denom_a) * sqrt(denom_b));
    if (similarity < 0)
    {
        similarity = 0;
    }

    return similarity;
}

NullImplFRVT11::NullImplFRVT11() {}

NullImplFRVT11::~NullImplFRVT11() {}

ReturnStatus
NullImplFRVT11::initialize(const std::string &configDir)
{

    string detector_file_params = configDir + "/model/retinaface_r50_v1/R50-0000.params";
    string recognitor_file_params = configDir + "/model/arcface_r100_v1/model-0000.params";

    detector.initialize(detector_file_params);
    recognitor.initialize(recognitor_file_params);

    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::createTemplate(
    const Multiface &faces,
    TemplateRole role,
    std::vector<uint8_t> &templ,
    std::vector<EyePair> &eyeCoordinates)
{

    std::vector<float> fv;
    std::vector<FaceDetectInfo> detectionInfo;
    //     int numberOfFV = 0;

    // std::cout << "Number faces: " << faces.size() << std::endl;

    for (unsigned int i = 0; i < faces.size(); i++)
    {

        cv::Mat faceMat;
        std::vector<FaceDetectInfo> detectionInfo_;
        bool canDectect = true;

        if (faces[i].depth == 8)
        {

            // std::cout << "Depth: " << std::to_string(faces[i].depth) << std::endl;

            faceMat = cv::Mat(faces[i].height, faces[i].width, CV_8UC1, faces[i].data.get());
            cv::cvtColor(faceMat, faceMat, CV_GRAY2BGR);
        }
        else if (faces[i].depth == 24)
        {

            // std::cout << "Depth: " << std::to_string(faces[i].depth) << std::endl;

            faceMat = cv::Mat(faces[i].height, faces[i].width, CV_8UC3, faces[i].data.get());
            cv::cvtColor(faceMat, faceMat, CV_RGB2BGR);
        }

        // std::cout << "Face image size: " << faceMat.size() << std::endl;

        if (faces[i].depth == 8 || faces[i].depth == 24)
        {
            detectionInfo_ = detector.detect(faceMat, 0.4);
        }

        if (detectionInfo_.empty())
        {

            // std::cout << "Warning: can't detect image" << std::endl;

            canDectect = false;

            anchor_box anchor;
            anchor.x1 = 0;
            anchor.y1 = 0;
            anchor.x2 = faces[i].width - 1;
            anchor.y2 = faces[i].height - 1;

            FacePts landmarks;
            for (size_t j = 0; j < 5; j++)
            {
                landmarks.x[j] = j;
                landmarks.y[j] = j;
            }

            FaceDetectInfo tmp;
            tmp.score = 0.0;
            tmp.rect = anchor;
            tmp.pts = landmarks;
            detectionInfo_.push_back(tmp);
        }

        detectionInfo.push_back(detectionInfo_[0]);

        for (int j = 0; j < detectionInfo_.size(); j++)
        {

            cv::Mat cropImage;

            if (canDectect)
            {
                cropImage = norm_crop(faceMat, detectionInfo_[j].pts);
            }
            else
            {
                cropImage = faceMat.clone();
            }
            //             cv::Mat cropImage = norm_crop_temp(faceMat, detectionInfo_[j].rect);

            std::vector<float> currentFV = recognitor.GetEmbedding(cropImage);

            fv.reserve(fv.size() + std::distance(currentFV.begin(), currentFV.end()));
            fv.insert(fv.end(), currentFV.begin(), currentFV.end());

            cropImage.release();

            //             numberOfFV += 1;
            break;
        }

        faceMat.release();
        detectionInfo_.clear();
    }

    // std::cout << "FV size: " << fv.size() << std::endl;

    /* Note: example code, potentially not portable across machines. */
    //     std::vector<float> fv = {1.0, 2.0, 8.88, 765.88989};
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(fv.data());
    int dataSize = sizeof(float) * fv.size();
    templ.resize(dataSize);
    memcpy(templ.data(), bytes, dataSize);

    for (unsigned int i = 0; i < detectionInfo.size(); i++)
    {
        FacePts landmarks = detectionInfo[i].pts;
        eyeCoordinates.push_back(EyePair(true, true, static_cast<int>(landmarks.x[0]),
                                         static_cast<int>(landmarks.y[0]),
                                         static_cast<int>(landmarks.x[1]),
                                         static_cast<int>(landmarks.y[1])));
    }

    // std::cout << "Template size: " << templ.size() << std::endl;

    // Clear memory
    fv.clear();
    detectionInfo.clear();

    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::matchTemplates(
    const std::vector<uint8_t> &verifTemplate,
    const std::vector<uint8_t> &enrollTemplate,
    double &similarity)
{

    // std::cout << "Verify Template size: " << verifTemplate.size() << std::endl;
    // std::cout << "Enrollment Template size: " << enrollTemplate.size() << std::endl;

    float *enrollVector = (float *)enrollTemplate.data();
    float *verifVector = (float *)verifTemplate.data();

    std::vector<double> similaritys;
    double threshold = 0.3;
    int valid = 0;
    double sumValidSimilarity = 0.0;
    double sumInvalidSimilarity = 0.0;

    int countTemplate = enrollTemplate.size() / (this->featureVectorSize * sizeof(float));
    similaritys.reserve(countTemplate);

    // std::cout << "count Template: " << countTemplate << std::endl;
    // std::cout << "Feature size: " << this->featureVectorSize << std::endl;

    for (int i = 0; i < countTemplate; i++)
    {

        double currentSimilarity = cosineSimilarity(enrollVector, verifVector, this->featureVectorSize, this->featureVectorSize * i);

        similaritys.push_back(currentSimilarity);

        if (currentSimilarity > threshold)
        {
            valid += 1;
            sumValidSimilarity += currentSimilarity;
        }
        else
        {
            sumInvalidSimilarity += currentSimilarity;
        }
    }

    if (countTemplate == 1)
    {

        similarity = similaritys[0];
    }
    else if (countTemplate > 1 && countTemplate < 5)
    {

        if (valid >= countTemplate * 0.6)
        {
            similarity = sumValidSimilarity / valid;
        }
        else if (countTemplate == 3 && countTemplate == 4)
        {
            similarity = sumInvalidSimilarity / (countTemplate - valid);
        }
        else
        {
            similarity = (sumValidSimilarity + sumInvalidSimilarity) / 2.0;
        }
    }
    else
    {

        if (valid >= countTemplate * 0.8)
        {
            similarity = sumValidSimilarity / valid;
        }
        else
        {
            similarity = sumInvalidSimilarity / (countTemplate - valid);
        }
    }

    //     for (unsigned int i=0; i<this->featureVectorSize; i++) {
    //         std::cout << enrollVector[i] << " ";
    //     }
    //     std::cout << std::endl;

    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<NullImplFRVT11>();
}

#include <opencv2/opencv.hpp>

int main() {
    
    cv::Mat faceMat = cv::imread("issue2.jpg", CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(faceMat, faceMat, CV_BGR2RGB);
    cv::cvtColor(faceMat, faceMat, CV_RGB2BGR);
    cv::cvtColor(faceMat, faceMat, CV_BGR2RGB);
    cv::imwrite( "issue2-out.jpg", faceMat );
    
    std::cout << faceMat.size() << std::endl;
    
    return 0;
}
#include "img_process.hpp"


void img_enhance(cv::Mat& img_left, cv::Mat& img_right, cv::Mat& img_left_scale, cv::Mat& img_right_scale, cv::Size size_scale){
    // img_left， img_right enhance
    cv::Mat tmp;
    cv::resize(img_left, tmp, size_scale);
    cv::cvtColor(tmp, img_left_scale, cv::COLOR_BGR2GRAY);
    cv::resize(img_right, tmp, size_scale);
    cv::cvtColor(tmp, img_left_scale, cv::COLOR_BGR2GRAY);
    return ;
}
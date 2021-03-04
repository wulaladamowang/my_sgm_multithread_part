#ifndef __IMG_PROCESS_H__
#define __IMG_PROCESS_H__
#include <opencv2/opencv.hpp>
void img_enhance(cv::Mat& img_left, cv::Mat& img_right, cv::Mat& img_left_scale, cv::Mat& img_right_scale, cv::Size size_scale);

#endif

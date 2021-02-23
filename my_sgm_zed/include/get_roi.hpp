/*
 * @Author: your name
 * @Date: 2021-02-08 17:31:48
 * @LastEditTime: 2021-02-23 21:50:01
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /my_sgm_zed_multithread_v2/my_sgm_zed/include/get_roi.hpp
 */
#ifndef __GET_MARKER_ROI_H__
#define __GET_MARKER_ROI_H__


#include <vector> 
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
int relativeDis(cv::Vec4f line_para, std::vector<cv::Point2f> point);
void get_roi(cv::Mat& image, cv::Mat& mask, bool& has_roi, std::vector<int>& rect_roi, std::vector<cv::Point2f>& marker_position) ;

#endif
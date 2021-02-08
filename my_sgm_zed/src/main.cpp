#include "my_camera.hpp"
#include "get_point_cloud.hpp"
#include "get_roi.hpp"
#include "get_disparity.hpp"
#include "little_tips.hpp"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <string>
#include <iostream>

#define WINDOW "原图"
cv::Point previousPoint;
bool isdispaly = false;
void On_mouse(int event, int x, int y, int flags, void*)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		previousPoint = cv::Point(x, y);
        isdispaly = true;
	}
}

void getPosition(cv::Mat& mask, cv::Mat& disparity, std::vector<int>& rect_roi, std::vector<cv::Point>& roi_points, 
                std::vector<cv::Point3f>& position_points, const float scale, bool& has_roi){   //23 01对应长边， 03 12对应于短边，贴码导致
    if(!has_roi)
        return;
    int x1 = (roi_points[0].x + roi_points[3].x)/4;
    int y1 = (roi_points[0].y + roi_points[3].y)/4;
    int x2 = (roi_points[1].x + roi_points[2].x)/4;
    int y2 = (roi_points[1].y + roi_points[2].y)/4;
    int middlex = (rect_roi[0] + rect_roi[2])/4;
    int middley = (rect_roi[1] + rect_roi[3])/4;
    position_points.clear();
    
    float sum = 0;
    int num = 0;
    float x_sum = 0;
    float y_sum = 0;
    float standrad_depth = 0;
    for (int m = -8; m < 9; m++){
        for (int n = -8; n < 9; n++){
            float depth = 62.196873360591700*1.398428736568180e+03/(disparity.ptr<float>(middley+m)[middlex+n]/scale);
            if (0<depth && depth<10000)
            {
                sum += depth;
                x_sum += depth * (middlex+n) * 2 / 1.398428736568180e+03;
                y_sum += depth * (middley+m) * 2 / 1.398428736568180e+03;
                num++;
            }   
        }
    }
    standrad_depth = sum/num;
    position_points[2].z = sum/num;
    position_points[2].x = x_sum / num;
    position_points[2].y = y_sum / num;

    sum = 0;
    num = 0;
    x_sum = 0;
    y_sum = 0;
    for (int m = -15; m < 16; m++){
        for (int n = -15; n < 16; n++){
            float depth = 62.196873360591700*1.398428736568180e+03/(disparity.ptr<float>(y1+m)[x1+n]/scale);
            if (standrad_depth-1000<depth && depth<standrad_depth+1000)
            {
                sum += depth;
                x_sum += depth * (x1+n) * 2 / 1.398428736568180e+03;
                y_sum += depth * (y1+m) * 2 / 1.398428736568180e+03;
                num++;
            }   
        }
    }
    position_points[0].z = sum/num;
    position_points[0].x = x_sum/num;
    position_points[0].y = y_sum/num;

    sum = 0;
    num = 0;
    x_sum = 0;
    y_sum = 0;
    for (int m = -15; m < 16; m++){
        for (int n = -15; n < 16; n++){
            float depth = 62.196873360591700*1.398428736568180e+03/(disparity.ptr<float>(y2+m)[x2+n]/scale);
            if (standrad_depth-1000<depth && depth<standrad_depth+1000)
            {
                sum += depth;
                x_sum += depth * (x2+n) * 2 / 1.398428736568180e+03;
                y_sum += depth * (y2+m) * 2 / 1.398428736568180e+03;
                num++;
            }   
        }
    }
    position_points[1].z = sum/num;
    position_points[1].x = x_sum/num;
    position_points[1].y = y_sum/num;

    float vect_1_x = position_points[0].x - position_points[1].x;
    float vect_1_y = position_points[0].y - position_points[1].y;
    float vect_1_z = position_points[0].z - position_points[1].z;
    float vect_1_x_standrad = vect_1_x / sqrt(vect_1_x*vect_1_x + vect_1_y*vect_1_y + vect_1_z*vect_1_z);
    float vect_1_y_standrad = vect_1_y / sqrt(vect_1_x*vect_1_x + vect_1_y*vect_1_y + vect_1_z*vect_1_z);
    float vect_1_z_standrad = vect_1_z / sqrt(vect_1_x*vect_1_x + vect_1_y*vect_1_y + vect_1_z*vect_1_z);

    float vect_2_x = position_points[0].x - position_points[2].x;
    float vect_2_y = position_points[0].y - position_points[2].y;
    float vect_2_z = position_points[0].z - position_points[2].z;
    float vect_2_x_standrad = vect_2_x / sqrt(vect_2_x*vect_2_x + vect_2_y*vect_2_y + vect_2_z*vect_2_z);
    float vect_2_y_standrad = vect_2_y / sqrt(vect_2_x*vect_2_x + vect_2_y*vect_2_y + vect_2_z*vect_2_z);
    float vect_2_z_standrad = vect_2_z / sqrt(vect_2_x*vect_2_x + vect_2_y*vect_2_y + vect_2_z*vect_2_z);

    float vect_3_x = position_points[2].x - position_points[1].x;
    float vect_3_y = position_points[2].y - position_points[1].y;
    float vect_3_z = position_points[2].z - position_points[1].z;
    float vect_3_x_standrad = vect_3_x / sqrt(vect_3_x*vect_3_x + vect_3_y*vect_3_y + vect_3_z*vect_3_z);
    float vect_3_y_standrad = vect_3_y / sqrt(vect_3_x*vect_3_x + vect_3_y*vect_3_y + vect_3_z*vect_3_z);
    float vect_3_z_standrad = vect_3_z / sqrt(vect_3_x*vect_3_x + vect_3_y*vect_3_y + vect_3_z*vect_3_z);

    float vect_x = vect_1_x_standrad + vect_2_x_standrad + vect_3_x_standrad;
    float vect_y = vect_1_y_standrad + vect_2_y_standrad + vect_3_y_standrad;
    float vect_z = vect_1_z_standrad + vect_2_z_standrad + vect_3_z_standrad;
    float vect_x_standrad = vect_x / sqrt(vect_x*vect_x + vect_y*vect_y + vect_z*vect_z);
    float vect_y_standrad = vect_y / sqrt(vect_x*vect_x + vect_y*vect_y + vect_z*vect_z);
    float vect_z_standrad = vect_z / sqrt(vect_x*vect_x + vect_y*vect_y + vect_z*vect_z);

    std::cout << "X: " << position_points[2].x << " Y: " << position_points[2].y << " Z: " << position_points[2].z << "\n"
              << "Vect_x: " << vect_x_standrad << " Vect_y: " << vect_y_standrad << " Vect_z: " << vect_z_standrad << std::endl;
}

std::mutex lock_roi;
bool not_roi = true;// 未处理过roi?
std::condition_variable con_roi;

void showImg(cv::Mat& img_zed_left_scale, cv::Mat& disparity_mask, cv::Mat& disparity_8u, bool& run){
    while (run){
        cv::setMouseCallback(WINDOW, On_mouse, 0);
        cv::imshow(WINDOW, img_zed_left_scale);
        cv::imshow("mask scale", disparity_mask);
        cv::imshow("disparity_8u", disparity_8u);
        const char key = cv::waitKey(10);
        if (key == 27)
            break;
    }
    run = false;
}

void GetMaskRoi(cv::Mat& img_left, cv::Size& siz_scale, cv::Mat& mask, cv::Mat& mask_scale, bool& has_roi, std::vector<int>& rect_roi, std::vector<cv::Point>& roi_points, bool& run){
    cv::Mat img_detect(img_left.size(), img_left.type());
    while (run){
        std::unique_lock<std::mutex> lck_r(lock_roi);
        con_roi.wait(lck_r, []{return not_roi;});
        img_detect = img_left.clone();
        get_roi(std::ref(img_left), std::ref(mask), std::ref(has_roi), std::ref(rect_roi), std::ref(roi_points));
        cv::resize(mask, mask_scale, siz_scale, cv::INTER_LINEAR);
        not_roi = false;
        con_roi.notify_one();
    }    
}



int main(int argc, char** argv){

// zed相机的初始化
	sl::Camera zed;
	sl::InitParameters initParameters;
	initParameters.camera_resolution = sl::RESOLUTION::HD1080;
    if (argc >= 5) initParameters.input.setFromSVOFile(argv[4]);
	sl::ERROR_CODE err = zed.open(initParameters);
	if (err != sl::ERROR_CODE::SUCCESS) {
		std::cout << toString(err) << std::endl;
		zed.close();
		return 1;
	}
    sl::RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = sl::SENSING_MODE::LAST;

//图像变量的获取及初始化
    const int width = static_cast<int>(zed.getCameraInformation().camera_resolution.width);
	const int height = static_cast<int>(zed.getCameraInformation().camera_resolution.height);
    sl::Mat zed_image_l(zed.getCameraInformation().camera_resolution, sl::MAT_TYPE::U8_C1);
	sl::Mat zed_image_r(zed.getCameraInformation().camera_resolution, sl::MAT_TYPE::U8_C1);//相机原格式获得图像
    cv::Mat img_zed_left = slMat2cvMat(zed_image_l);
    cv::Mat img_zed_right = slMat2cvMat(zed_image_r);//sl::Mat 到opencv 格式图像的转换
    sl::Mat point_cloud;
    cv::Mat img_zed_left_remap(img_zed_left.size(), img_zed_left.type());// 存储校正畸变后的图像
    cv::Mat img_zed_right_remap(img_zed_left.size(), img_zed_left.type());

// 相机内外参数的读取, 注意相机内外参数要与使用的相机型号或拍摄视频的相机型号相一致
    std::string in = "/home/wang/code/c++Code/my_sgm_zed_server/canshu/intrinsics.yml";
    cv::FileStorage fs(in, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", argv[3]);
        return -1;
    }
    cv::Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;
    std::string out = "/home/wang/code/c++Code/my_sgm_zed_server/canshu/extrinsics.yml";
    fs.open(out, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", argv[4]);
        return -1;
    }
    cv::Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;
    cv::Mat Q;
    cv::Size img_size = img_zed_right.size();
    cv::stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, img_size, nullptr, nullptr );
    cv::Mat map11, map12, map21, map22;//校正参数
    cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

//立体匹配参数设定
    const float scale = argc >= 2 ? atof(argv[1]) : 0.5;//图像缩放比例,默认为变为原来的0.5倍，图像大小对sgm算法影响较大
    const int disp_size = (argc >= 3) ? std::stoi(argv[2]) : 128;//默认的disparity size, 可选择64,128,256
    const bool subpixel = (argc >= 4) ? std::stoi(argv[3]) != 0 : true;//是否使用subpixel

    cv::Size siz_scale = cv::Size(width*scale, height*scale);// 对于原图进行缩放尺寸，缩放后的尺寸大小
    cv::Mat img_zed_left_scale(siz_scale, img_zed_left.type());
    cv::Mat img_zed_right_scale(siz_scale, img_zed_left.type());// 缩放之后的图像, 在校正之后的图像基础上进行缩放
    cv::Mat disparity(siz_scale, CV_16S);//存储sgm算法获得的视差图
    cv::Mat disparity_8u(siz_scale, CV_8U), disparity_32f(siz_scale, CV_32F), disparity_mask(siz_scale, CV_8U);// 8u转化便于显示的灰度图, 32f实际视差图,带小数点, disparity_mask保留mask区域的视差
    const int input_depth = img_zed_left_scale.type() == CV_8U ? 8 : 16;
    const int output_depth = 16;
    const sgm::StereoSGM::Parameters params{10, 120, 0.95f, subpixel};
    sgm::StereoSGM sgm(width*scale, height*scale, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA, params);

// 记录时间
    long long ts0 =getCurrentTime();
    long long ts1 =getCurrentTime();
    long long ts2 = getCurrentTime();
    long long ts3 =getCurrentTime();
    long long ts4 = getCurrentTime();
    long long ts5 =getCurrentTime();
    long long ts6 = getCurrentTime();

//获取目标区域函数参数设定
    bool has_roi = false;// 是否检测到roi
    cv::Mat mask(img_size, CV_8U);// 检测到的roi, 用mask 标记出来
    cv::Mat mask_scale(siz_scale, CV_8U);// 对Mask进行缩放之后的大小
    std::vector<int> rect_roi;// 存储包围目标区域的最小正矩形的四个像素坐标，原图大小
    std::vector<cv::Point> roi_points;
    std::vector<cv::Point3f> position_points;
    position_points.reserve(3);
    rect_roi.reserve(4);
    roi_points.reserve(4);

    bool run = true;
    
	
    std::thread DispImg(showImg, std::ref(img_zed_left_scale), std::ref(disparity_mask), std::ref(disparity_8u), std::ref(run));
    std::thread GetRoi(GetMaskRoi, std::ref(img_zed_left_remap), std::ref(siz_scale), std::ref(mask), std::ref(mask_scale), std::ref(has_roi), std::ref(rect_roi), std::ref(roi_points), std::ref(run));
    // cv::namedWindow(WINDOW);
    //cv::setMouseCallback(WINDOW, On_mouse, 0);
    
    while(run){

        if (zed.grab(runtime_parameters) == sl::ERROR_CODE::SUCCESS) {
        // 获取并校正图像，对校正之后的图像进行缩放
            
            zed.retrieveImage(zed_image_l, sl::VIEW::LEFT_UNRECTIFIED_GRAY, sl::MEM::CPU);
            zed.retrieveImage(zed_image_r, sl::VIEW::RIGHT_UNRECTIFIED_GRAY, sl::MEM::CPU);
            //zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::CPU);
            if (img_zed_left_scale.empty() || img_zed_right_scale.empty())
                continue;
            std::unique_lock<std::mutex> lck_r(lock_roi);// 些许不妥，速度快
            con_roi.wait(lck_r, []{return !not_roi;});
            getPosition(std::ref(mask), std::ref(disparity_32f), std::ref(rect_roi), std::ref(roi_points), std::ref(position_points), std::ref(scale), std::ref(has_roi));
            cv::remap(img_zed_left, img_zed_left_remap, map11, map12, cv::INTER_LINEAR);
            cv::remap(img_zed_right, img_zed_right_remap, map21, map22, cv::INTER_LINEAR);

            cv::resize(img_zed_left_remap, img_zed_left_scale, siz_scale, cv::INTER_LINEAR);
            cv::resize(img_zed_right_remap, img_zed_right_scale, siz_scale, cv::INTER_LINEAR);
        // 图像匹配获得视差图
            get_disparity(std::ref(sgm), std::ref(img_zed_left_scale), std::ref(img_zed_right_scale), std::ref(disparity), disp_size, subpixel);
            disparity.convertTo(disparity_32f, CV_32F, subpixel ? 1. / sgm::StereoSGM::SUBPIXEL_SCALE : 1);
            
            
            // sl::float4 point3d;
            // point_cloud.getValue(previousPoint.x/scale, previousPoint.y/scale , &point3d);
            // std::cout << "Point X: " << previousPoint.x << " Y: " << previousPoint.y << "  Disparity: " << 62.196873360591700*1.398428736568180e+03/(disparity_32f.ptr<float>(previousPoint.y)[previousPoint.x]/scale) << " depth: " << point3d.z << std::endl;
            // cv::waitKey(0);
            not_roi = true;
            con_roi.notify_one();

            ts5 = getCurrentTime();
            std::cout << "total: " << (ts5-ts6) << "毫秒" << std::endl;
            std::cout << "-------------" << std::endl;
        // 展示图像（非必要）
            disparity_32f.convertTo(disparity_8u, CV_8U, 255. / 64);
            disparity_mask = disparity_8u.mul(mask_scale);// 只保留目标检测区域的视差
            ts6 = getCurrentTime();
        }
    }
    DispImg.join();
    GetRoi.join();
    //zed.close();// 增加此条语句会造成 段错误， 已在自带zed例子上进行验证，程序自带bug
    std::cout << "finish ---------------- " << std::endl;
    return 0;
}



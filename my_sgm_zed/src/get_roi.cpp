#include "get_roi.hpp"

int relativeDis(cv::Vec4f line_para, std::vector<cv::Point2f> point) {
    double A = line_para[1]/line_para[0];
    double B = -1;
    double C = line_para[3]*(1-line_para[1]/line_para[0]);
    double min = 0.0;
    int index = -1;
    for(int i=0;i<point.size();i++){
        double dis = A*point[i].x+B*point[i].y+C;
        if(dis<0)
            dis = -dis;
        if(-1==index || dis<min)
        {
            min = dis;
            index = i;
        }
    }
    return index;
};
/**
 * @description: 在图像中进行目标识别 
 * @param 
 * {
 * image : 原始图像
 * mask : 识别目标区域的掩码
 * has_roi : 是否识别到目标区域
 * rect_roi : 包围目标区域的最小正矩形, minx miny maxx maxy
 * roi_points : 目标区域的四个角的坐标
 * }
 * @return {*}
 */
void get_roi(cv::Mat& image, cv::Mat& mask, bool& has_roi, std::vector<int>& rect_roi, std::vector<cv::Point2f>& marker_position) {

    static std::vector<int > pre_rect_roi = {0, 0, 1, 1};// 初始化用于记录上次检测到目标的位置, 正矩形
    static cv::Size mask_size = image.size();
    static const cv::Ptr<cv::aruco::Dictionary> c_dictionary = cv::aruco::getPredefinedDictionary(
        cv::aruco::DICT_4X4_50);//DICT_6X6_1000

    static std::vector<std::vector<cv::Point2f>> marker_corners;
    static std::vector<int> marker_ids;
    
    bool is_full = false;
    int min_x, min_y, max_x, max_y;
// 将上次目标位置扩大范围后作为目标检测的初始位置，若未检测到目标区域，则再进行全局搜索
    int pre_roi_min_x, pre_roi_min_y, pre_roi_max_x, pre_roi_max_y;   
    pre_roi_min_x = (pre_rect_roi[0] - 150) < 0 ? 0 : (pre_rect_roi[0] - 150);
    pre_roi_max_x = (pre_rect_roi[2] + 150) > image.cols-1 ? image.cols-1 : (pre_rect_roi[2] + 150);
    pre_roi_min_y = (pre_rect_roi[1] - 150) < 0 ? 0 : (pre_rect_roi[1] - 150);
    pre_roi_max_y = (pre_rect_roi[3] + 150) > image.rows-1 ? image.rows-1 : (pre_rect_roi[3] + 150);
   // 检测部分区域是否有marker， 可以增加检测速度 
    cv::Mat img_roi = image(cv::Rect(cv::Point(pre_roi_min_x, pre_roi_min_y), cv::Point(pre_roi_max_x, pre_roi_max_y)));
    marker_corners.clear(); marker_ids.clear();
    cv::aruco::detectMarkers(img_roi, c_dictionary, marker_corners, marker_ids);
    ///获得检测的面积最大的aruco marker序号,每个ID都有一个（若检测到）
    int marker_number = marker_ids.size();
    if (1 > marker_number)
    {
        std::cout << "full detect ------" << std::endl;
        cv::aruco::detectMarkers(image, c_dictionary, marker_corners, marker_ids);
        marker_number = marker_ids.size();
        is_full = true;
    }
    if(0 < marker_number)
    {
        int buff_id[6] = {-1, -1, -1, -1, -1, -1};//若存在相应的aruco marker id, 则记录其序号
        double buff_id_length[6] = {0.0, 0, 0, 0, 0, 0};//若存在相应的aruco marker， 则记录其在当前时刻周长的最大值
    // 保存每个id的周长最大值的序号
        for(int i=0;i<marker_number;i++){
            double cur_marker_len = cv::arcLength(marker_corners[i], true);
            if(1 == marker_ids[i]){
                if(cur_marker_len > buff_id_length[1]){
                    buff_id_length[1] = cur_marker_len;
                    buff_id[1] = i;
                }
            }else if(2 == marker_ids[i]){
                if(cur_marker_len > buff_id_length[2]){
                    buff_id_length[2] = cur_marker_len;
                    buff_id[2] = i;
                }
            }else if(3 == marker_ids[i]){
                if(cur_marker_len > buff_id_length[3]){
                    buff_id_length[3] = cur_marker_len;
                    buff_id[3] = i;
                }
            }else if(4 == marker_ids[i]){
                if(cur_marker_len > buff_id_length[4]){
                    buff_id_length[4] = cur_marker_len;
                    buff_id[4] = i;
                }
           }else if(5 == marker_ids[i]){
               if(cur_marker_len > buff_id_length[5]){
                   buff_id_length[5] = cur_marker_len;
                   buff_id[5] = i;
               }
            } else
                continue;
        }

        std::vector<cv::Point2f> compute_line;///用于选中ID的线，每个存储的为选中的ID的中点坐标
        std::vector<int> no_id;///存储被选中的ID的id号
        for(int m=5;m>0;m--){
            if(-1!=buff_id[m]){
                compute_line.emplace_back(cv::Point2f((marker_corners[buff_id[m]][1].x + marker_corners[buff_id[m]][2].x + marker_corners[buff_id[m]][3].x + marker_corners[buff_id[m]][0].x)/4,
                                                      (marker_corners[buff_id[m]][1].y + marker_corners[buff_id[m]][2].y + marker_corners[buff_id[m]][3].y + marker_corners[buff_id[m]][0].y)/4));
                no_id.push_back(m);
            }
        }
        int center_x ;
        int center_y ;
        int index = 0;///记录compute_line中距离拟合线最近的点的序号
        if(no_id.size()<3){
        }else{
            cv::Vec4f line_para;
            cv::fitLine(compute_line, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
            index = relativeDis(line_para, compute_line);// 返回距离线最近的id
        }
        // index 记录了选取的id 的序号， 每个id对应物体的相对位置可知，根据序号进行比例扩大
        int i = no_id[index];
        // marker_position 记录了距离拟合直线最近的marker的坐标
        marker_position.clear();
        marker_position.reserve(4);
        cv::Point2f point0, point1, point2, point3;
        point0.x = marker_corners[buff_id[i]][0].x + (is_full ? 0 : pre_roi_min_x );
        point0.y = marker_corners[buff_id[i]][0].y + (is_full ? 0 : pre_roi_min_y );
        point1.x = marker_corners[buff_id[i]][1].x + (is_full ? 0 : pre_roi_min_x );
        point1.y = marker_corners[buff_id[i]][1].y + (is_full ? 0 : pre_roi_min_y );
        point2.x = marker_corners[buff_id[i]][2].x + (is_full ? 0 : pre_roi_min_x );
        point2.y = marker_corners[buff_id[i]][2].y + (is_full ? 0 : pre_roi_min_y );
        point3.x = marker_corners[buff_id[i]][3].x + (is_full ? 0 : pre_roi_min_x );
        point3.y = marker_corners[buff_id[i]][3].y + (is_full ? 0 : pre_roi_min_y );
        marker_position = {point0, point1, point2, point3};
        

        std::vector<cv::Point > roi_position;//用于记录roi四个角点的位置, 后续做掩码图
        roi_position.reserve(4);
        cv::Point roi_p0, roi_p1, roi_p2, roi_p3;//用来包裹整个目标圆柱
        ///轴向比例扩大，该参数通过过分支choose coefficient 获得，与目标检测物与贴码位置有关
        float axial_coefficient = 0;//目标长度为marker边长的倍数
        float axial_position_coefficient = 0.0;//marker的ID不同，则则其位于目标的位置不同，距离ID1的上端的位置分数比例
        float radial_coefficient = 0;//目标横向（径向）为marker边长的倍数
        switch (no_id[index]) {
            case 5 : axial_coefficient = 12 ; axial_position_coefficient = 0.13 ; radial_coefficient = 1.4  ;break;
            case 4 : axial_coefficient = 13 ; axial_position_coefficient = 0.32 ; radial_coefficient = 1.9  ;break;
            case 3 : axial_coefficient = 14 ; axial_position_coefficient = 0.50 ; radial_coefficient = 2.2 ;break;
            case 2 : axial_coefficient = 16 ; axial_position_coefficient = 0.67 ; radial_coefficient = 2.5 ;break;
            case 1 : axial_coefficient = 17 ; axial_position_coefficient = 0.84 ; radial_coefficient = 3.1 ;break;
        }

        ///纵向比例扩大

        roi_p0.x = axial_position_coefficient*axial_coefficient*(marker_corners[buff_id[i]][1].x - marker_corners[buff_id[i]][0].x)+marker_corners[buff_id[i]][0].x;
        roi_p0.y = axial_position_coefficient*axial_coefficient*(marker_corners[buff_id[i]][1].y - marker_corners[buff_id[i]][0].y)+marker_corners[buff_id[i]][0].y;

        roi_p1.x = (1-axial_position_coefficient)*axial_coefficient*(marker_corners[buff_id[i]][0].x - marker_corners[buff_id[i]][1].x)+marker_corners[buff_id[i]][1].x;
        roi_p1.y = (1-axial_position_coefficient)*axial_coefficient*(marker_corners[buff_id[i]][0].y - marker_corners[buff_id[i]][1].y)+marker_corners[buff_id[i]][1].y;

        roi_p2.x = (1-axial_position_coefficient)*axial_coefficient*(marker_corners[buff_id[i]][3].x - marker_corners[buff_id[i]][2].x)+marker_corners[buff_id[i]][2].x;
        roi_p2.y = (1-axial_position_coefficient)*axial_coefficient*(marker_corners[buff_id[i]][3].y - marker_corners[buff_id[i]][2].y)+marker_corners[buff_id[i]][2].y;

        roi_p3.x = axial_position_coefficient*axial_coefficient*(marker_corners[buff_id[i]][2].x - marker_corners[buff_id[i]][3].x)+marker_corners[buff_id[i]][3].x;
        roi_p3.y = axial_position_coefficient*axial_coefficient*(marker_corners[buff_id[i]][2].y - marker_corners[buff_id[i]][3].y)+marker_corners[buff_id[i]][3].y;
        ///径向比例扩大

        roi_p0.x = radial_coefficient*(marker_corners[buff_id[i]][0].x - marker_corners[buff_id[i]][3].x)+roi_p0.x;
        roi_p0.y = radial_coefficient*(marker_corners[buff_id[i]][0].y - marker_corners[buff_id[i]][3].y)+roi_p0.y;

        roi_p1.x = radial_coefficient*(marker_corners[buff_id[i]][1].x - marker_corners[buff_id[i]][2].x)+roi_p1.x;
        roi_p1.y = radial_coefficient*(marker_corners[buff_id[i]][1].y - marker_corners[buff_id[i]][2].y)+roi_p1.y;

        roi_p2.x = radial_coefficient*(marker_corners[buff_id[i]][2].x - marker_corners[buff_id[i]][1].x)+roi_p2.x;
        roi_p2.y = radial_coefficient*(marker_corners[buff_id[i]][2].y - marker_corners[buff_id[i]][1].y)+roi_p2.y;

        roi_p3.x = radial_coefficient*(marker_corners[buff_id[i]][3].x - marker_corners[buff_id[i]][0].x)+roi_p3.x;
        roi_p3.y = radial_coefficient*(marker_corners[buff_id[i]][3].y - marker_corners[buff_id[i]][0].y)+roi_p3.y;
        // 若选取目标区域不为原图， 其在原图的位置需要进行修正, 坐标有可能在图像之外
        roi_p0.x = roi_p0.x + (is_full ? 0 : pre_roi_min_x );
        roi_p0.y = roi_p0.y + (is_full ? 0 : pre_roi_min_y );
        roi_p1.x = roi_p1.x + (is_full ? 0 : pre_roi_min_x );
        roi_p1.y = roi_p1.y + (is_full ? 0 : pre_roi_min_y );
        roi_p2.x = roi_p2.x + (is_full ? 0 : pre_roi_min_x );
        roi_p2.y = roi_p2.y + (is_full ? 0 : pre_roi_min_y );
        roi_p3.x = roi_p3.x + (is_full ? 0 : pre_roi_min_x ); 
        roi_p3.y = roi_p3.y + (is_full ? 0 : pre_roi_min_y );

        mask.setTo(0);
        roi_position.push_back(roi_p0);
        roi_position.push_back(roi_p1);
        roi_position.push_back(roi_p2);
        roi_position.push_back(roi_p3);
        min_x = std::min({roi_p0.x, roi_p1.x, roi_p2.x, roi_p3.x});
        min_y = std::min({roi_p0.y, roi_p1.y, roi_p2.y, roi_p3.y});
        max_x = std::max({roi_p0.x, roi_p1.x, roi_p2.x, roi_p3.x});
        max_y = std::max({roi_p0.y, roi_p1.y, roi_p2.y, roi_p3.y});
        min_x = min_x < 0 ? 0 : min_x;
        max_x = max_x > mask_size.width-1 ? mask_size.width - 1 : max_x;
        min_y = min_y < 0 ? 0 : min_y;
        max_y = max_y > mask_size.height-1 ? mask_size.height - 1 : max_y;
    
        rect_roi[0] = min_x;
        rect_roi[1] = min_y;
        rect_roi[2] = max_x;
        rect_roi[3] = max_y;
        pre_rect_roi = {min_x, min_y, max_x, max_y};

        std::vector<std::vector<cv::Point>> contours;
        contours.push_back(roi_position);
        cv::fillPoly(mask, contours, 1);
        has_roi = true;
    }else{
        mask.setTo(0);
        std::cout << "No aruco marker detected " << "\n";
        has_roi = false;
    }
}

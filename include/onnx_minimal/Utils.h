#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

namespace onnx_minimal {

    struct args_t {
        std::string img_path;
        std::string model_path;
    };

    struct bbox_t {
        int x;
        int y;
        int w;
        int h;
        float score;
        int class_id;
    };

    class Utils {

        public:
            static args_t parse_args(int argc, char* argv[]);
            static std::vector<onnx_minimal::bbox_t> nms_iou(float* boxes, float* scores, int num_boxes, int num_classes, float iou_threshold, float score_threshold);
            static cv::Mat z_score_normalize(cv::Mat img);
        
    };
} // namespace onnx_minimal

#endif // UTILS_H_
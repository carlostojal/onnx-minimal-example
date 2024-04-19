#include <onnx_minimal/Utils.h>

namespace onnx_minimal {

    args_t Utils::parse_args(int argc, char* argv[]) {
        args_t args;
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <path_to_image> <path_to_model>" << std::endl;
            throw std::invalid_argument("Invalid number of arguments");
        }
        args.img_path = argv[1];
        args.model_path = argv[2];
        return args;
    }

    std::vector<bbox_t> Utils::nms_iou(float* boxes, float* scores, int num_boxes, int num_classes, float iou_threshold, float score_threshold) {

        // create a vector of bbox_t objects
        std::vector<bbox_t> bboxes;
        for (int i = 0; i < num_boxes; i++) {
            bbox_t bbox;
            bbox.x = boxes[i * 4];
            bbox.y = boxes[i * 4 + 1];
            bbox.w = boxes[i * 4 + 2];
            bbox.h = boxes[i * 4 + 3];
            // find the class with the highest score
            float max_score = 0.0;
            int class_id = 0;
            for (int j = 0; j < num_classes; j++) {
                if (scores[i * num_classes + j] > max_score) {
                    max_score = scores[i * num_classes + j];
                    class_id = j;
                }
            }
            bbox.score = max_score;
            bbox.class_id = class_id;
            bboxes.push_back(bbox);
        }

        // sort the bboxes by score
        std::sort(bboxes.begin(), bboxes.end(), [](const bbox_t& a, const bbox_t& b) {
            return a.score > b.score;
        });

        // apply non-maximum suppression
        std::vector<bbox_t> nms_bboxes;
        for (int i = 0; i < bboxes.size(); i++) {
            bool keep = true;
            // check negative coordinates
            if (bboxes[i].w < 0 || bboxes[i].h < 0 || bboxes[i].x < 0 || bboxes[i].y < 0) {
                continue;
            }
            // check score threshold
            if (bboxes[i].score < score_threshold) {
                continue;
            }
            for (int j = 0; j < nms_bboxes.size(); j++) {
                if (keep) {
                    float overlap = std::min(bboxes[i].x + bboxes[i].w, nms_bboxes[j].x + nms_bboxes[j].w) - std::max(bboxes[i].x, nms_bboxes[j].x);
                    overlap *= std::min(bboxes[i].y + bboxes[i].h, nms_bboxes[j].y + nms_bboxes[j].h) - std::max(bboxes[i].y, nms_bboxes[j].y);
                    float iou = overlap / (bboxes[i].w * bboxes[i].h + nms_bboxes[j].w * nms_bboxes[j].h - overlap);
                    if (iou > iou_threshold) {
                        keep = false;
                    }
                }
            }
            if (keep) {
                nms_bboxes.push_back(bboxes[i]);
            }
        }

        return nms_bboxes;
    }

    cv::Mat Utils::z_score_normalize(cv::Mat img) {

        // split the channels, one per matrix
        cv::Mat img_channels[3];
        cv::split(img, img_channels);

        // normalize each channel
        for(short i = 0; i < 3; i++) {
            // get the mean and std deviation values
            cv::Scalar mean, std_dev;
            cv::meanStdDev(img_channels[i], mean, std_dev);
            float mean_val = mean.val[0];
            float std_dev_val = std_dev.val[0];
            // apply z-score normalization
            img_channels[i] = (img_channels[i] - mean_val) / std_dev_val;
        }

        // merge the channels back
        cv::Mat img_normalized;
        cv::merge(img_channels, 3, img_normalized);

        return img_normalized;
    }

    cv::Mat Utils::pad_to_shape(cv::Mat img, int target_width, int target_height) {

        // get the current image size
        int img_width = img.cols;
        int img_height = img.rows;

        // calculate the padding values
        int pad_left = (target_width - img_width) / 2;
        int pad_right = target_width - img_width - pad_left;
        int pad_top = (target_height - img_height) / 2;
        int pad_bottom = target_height - img_height - pad_top;

        std::cout << "Pad left: " << pad_left << ", pad right: " << pad_right << ", pad top: " << pad_top << ", pad bottom: " << pad_bottom << std::endl;

        // apply the padding
        cv::Mat img_padded;
        cv::copyMakeBorder(img, img_padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        return img_padded;
    }

} // namespace onnx_minimal
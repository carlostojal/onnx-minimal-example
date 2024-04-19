#include <iostream>
#include <onnx_minimal/Utils.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

using namespace onnx_minimal;

int main(int argc, char* argv[]) {

	// parse arguments
	args_t args = Utils::parse_args(argc, argv);

	// create the session
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_minimal");
	Ort::SessionOptions session_options;
	Ort::Session* session = new Ort::Session(env, args.model_path.c_str(), session_options);

	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	Ort::AllocatorWithDefaultOptions allocator;

	// print the input and output node counts
	size_t num_input_nodes = session->GetInputCount();
	size_t num_output_nodes = session->GetOutputCount();
	std::cout << "Number of input nodes: " << num_input_nodes << std::endl;
	std::cout << "Number of output nodes: " << num_output_nodes << std::endl;

	// get the image tensor shape
	std::cout << "Image tensor shape: ";
	Ort::TypeInfo image_type_info = session->GetInputTypeInfo(0);
	auto image_tensor_info = image_type_info.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> image_shape = image_tensor_info.GetShape();
	std::cout << "(";
	for (size_t i = 0; i < image_shape.size(); i++) {
		std::cout << image_shape[i] << " ";
	}
	std::cout << ")" << std::endl;

	// get the probability tensor shape and name
	std::cout << "Probability tensor shape: ";
	Ort::TypeInfo prob_type_info = session->GetOutputTypeInfo(0);
	auto prob_tensor_info = prob_type_info.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> prob_shape = prob_tensor_info.GetShape();
	std::cout << "(";
	for (size_t i = 0; i < prob_shape.size(); i++) {
		std::cout << prob_shape[i] << " ";
	}
	std::cout << ")" << std::endl;

	// get the bounding box tensor shape
	std::cout << "Bounding box tensor shape: ";
	Ort::TypeInfo bbox_type_info = session->GetOutputTypeInfo(1);
	auto bbox_tensor_info = bbox_type_info.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> bbox_shape = bbox_tensor_info.GetShape();
	std::cout << "(";
	for (size_t i = 0; i < bbox_shape.size(); i++) {
		std::cout << bbox_shape[i] << " ";
	}
	std::cout << ")" << std::endl;

	// load the input image
	cv::Mat img = cv::imread(args.img_path, cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cerr << "Could not read the image: " << args.img_path << std::endl;
		return 1;
	}
	// convert the image from BGR to RGB
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	// preprocess the input image
	cv::Mat img_resized;
	cv::resize(img, img_resized, cv::Size(416, 416));
	img_resized.convertTo(img_resized, CV_32FC3);
	// normalize the image using z-score normalization
	img_resized = Utils::z_score_normalize(img_resized);
	// pad the image to the target shape
	img_resized = Utils::pad_to_shape(img_resized, image_shape[3], image_shape[2]);

	// convert the input image () to a tensor
	std::vector<float> input_tensor_values(img_resized.rows * img_resized.cols * img_resized.channels());
	memcpy(input_tensor_values.data(), img_resized.data, input_tensor_values.size() * sizeof(float));
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), image_shape.data(), image_shape.size());

	// run the inference (names from ONNX introspection in Netron)
	std::vector<const char*> input_node_names = { "images" };
	std::vector<const char*> output_node_names = { "output", "1074" };

	// start measuring time
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<Ort::Value> output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);

	// stop measuring time
	auto end = std::chrono::high_resolution_clock::now();

	// print the inference time
	std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

	// extract the probability tensor
	float* prob_tensor_values = output_tensors[0].GetTensorMutableData<float>();

	// extract the bounding box tensor
	float* bbox_tensor_values = output_tensors[1].GetTensorMutableData<float>();

	// print the probability tensor values
	for(int i = 0; i < prob_shape[1]; i++) {
		for(int j = 0; j < prob_shape[2]; j++) {
			std::cout << prob_tensor_values[i * prob_shape[2] + j] << " ";
		}
		std::cout << std::endl;
	}

	// perform non-maximum suppression
	std::vector<bbox_t> bboxes = Utils::nms_iou(bbox_tensor_values, prob_tensor_values, bbox_shape[1], prob_shape[2], 0.5, 0.5);

	for(bbox_t bbox : bboxes) {
		std::cout << "Class ID: " << bbox.class_id << ", Score: " << bbox.score << ", Bounding box: (" << bbox.x << ", " << bbox.y << ", " << bbox.w << ", " << bbox.h << ")" << std::endl;
	}

	// draw the bounding boxes
	for (bbox_t bbox : bboxes) {
		cv::rectangle(img, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x + bbox.w, bbox.y + bbox.h), cv::Scalar(0, 255, 0), 2);
		cv::putText(img, std::to_string(bbox.class_id), cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
	}

	// save the output image
	cv::imwrite("output.jpg", img);
	std::cout << "Output image saved as output.jpg" << std::endl;

	// release the session
	delete session;

	return 0;
}

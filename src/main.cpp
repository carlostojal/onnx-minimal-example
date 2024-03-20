#include <iostream>
#include <onnx_minimal/Utils.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

using namespace onnx_minimal;

int main(int argc, char* argv[]) {

	// parse arguments
	args_t args = Utils::parse_args(argc, argv);

	// create the session
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_minimal");
	Ort::SessionOptions session_options;
	Ort::Session session(env, args.model_path.c_str(), session_options);

	// print the input and output node counts
	size_t num_input_nodes = session.GetInputCount();
	size_t num_output_nodes = session.GetOutputCount();
	std::cout << "Number of input nodes: " << num_input_nodes << std::endl;
	std::cout << "Number of output nodes: " << num_output_nodes << std::endl;

	// get the image tensor shape
	std::cout << "Image tensor shape: ";
	Ort::TypeInfo image_type_info = session.GetInputTypeInfo(0);
	auto image_tensor_info = image_type_info.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> image_shape = image_tensor_info.GetShape();
	std::cout << "(";
	for (size_t i = 0; i < image_shape.size(); i++) {
		std::cout << image_shape[i] << " ";
	}
	std::cout << ")" << std::endl;

	// get the probability tensor shape and name
	std::cout << "Probability tensor shape: ";
	Ort::TypeInfo prob_type_info = session.GetOutputTypeInfo(0);
	auto prob_tensor_info = prob_type_info.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> prob_shape = prob_tensor_info.GetShape();
	std::cout << "(";
	for (size_t i = 0; i < prob_shape.size(); i++) {
		std::cout << prob_shape[i] << " ";
	}
	std::cout << ")" << std::endl;

	// get the bounding box tensor shape
	std::cout << "Bounding box tensor shape: ";
	Ort::TypeInfo bbox_type_info = session.GetOutputTypeInfo(1);
	auto bbox_tensor_info = bbox_type_info.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> bbox_shape = bbox_tensor_info.GetShape();
	std::cout << "(";
	for (size_t i = 0; i < bbox_shape.size(); i++) {
		std::cout << bbox_shape[i] << " ";
	}
	std::cout << ")" << std::endl;

	/*
	// load the input image
	cv::Mat img = cv::imread(args.img_path, cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cerr << "Could not read the image: " << args.img_path << std::endl;
		return 1;
	}

	// preprocess the input image
	cv::Mat img_resized;
	cv::resize(img, img_resized, cv::Size(image_shape[2], image_shape[3]));

	// convert the input image to a tensor
	std::vector<float> input_tensor_values;
	for (int i = 0; i < input_shape[2]; i++) {
		for (int j = 0; j < input_shape[3]; j++) {
			cv::Vec3b pixel = img_resized.at<cv::Vec3b>(i, j);
			input_tensor_values.push_back(pixel[0]);
			input_tensor_values.push_back(pixel[1]);
			input_tensor_values.push_back(pixel[2]);
		}
	}
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(env, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());
	*/

	return 0;
}

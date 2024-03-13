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

	// print the input and output node shapes
	std::cout << "Input node shapes:" << std::endl;
	for (size_t i = 0; i < num_input_nodes; i++) {
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> input_shape = tensor_info.GetShape();
		std::cout << "Input node " << i << " shape: ";
		for (size_t j = 0; j < input_shape.size(); j++) {
			std::cout << input_shape[j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "Output node shapes:" << std::endl;
	for (size_t i = 0; i < num_output_nodes; i++) {
		Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> output_shape = tensor_info.GetShape();
		std::cout << "Output node " << i << " shape: ";
		for (size_t j = 0; j < output_shape.size(); j++) {
			std::cout << output_shape[j] << " ";
		}
		std::cout << std::endl;
	}

	// print the input and output node types
	std::cout << "Input node types:" << std::endl;
	for (size_t i = 0; i < num_input_nodes; i++) {
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType input_type = tensor_info.GetElementType();
		std::cout << "Input node " << i << " type: " << input_type << std::endl;
	}

	std::cout << "Output node types:" << std::endl;
	for (size_t i = 0; i < num_output_nodes; i++) {
		Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType output_type = tensor_info.GetElementType();
		std::cout << "Output node " << i << " type: " << output_type << std::endl;
	}

	// TODO: create the input and output tensors

	// TODO: load the input image into the input tensor

	return 0;
}

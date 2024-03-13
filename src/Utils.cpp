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

} // namespace onnx_minimal
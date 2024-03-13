#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>

namespace onnx_minimal {

    struct args_t {
        std::string img_path;
        std::string model_path;
    };

    class Utils {

        public:
            static args_t parse_args(int argc, char* argv[]);
        
    };
} // namespace onnx_minimal

#endif // UTILS_H_
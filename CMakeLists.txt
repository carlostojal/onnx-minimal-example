cmake_minimum_required(VERSION 3.25)
project(onnx-minimal-example LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

# declare the ONNX include directory and libraries
set(ONNX_INCLUDE_DIR /tmp/onnx/onnxruntime/include)
set(ONNX_LIBRARIES /tmp/onnx/onnxruntime/build/RelWithDebInfo/libonnxruntime.so
    /tmp/onnx/onnxruntime/build/RelWithDebInfo/libonnxruntime_providers_cuda.so
    /tmp/onnx/onnxruntime/build/RelWithDebInfo/libonnxruntime_providers_shared.so)

# find the OpenCV library
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${ONNX_INCLUDE_DIR})

add_executable(onnx_minimal_example src/main.cpp src/Utils.cpp)
target_link_libraries(onnx_minimal_example ${OpenCV_LIBS} ${ONNX_LIBRARIES})

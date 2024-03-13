cmake_minimum_required(VERSION 3.25)
project(onnx-minimal-example LANGUAGES CXX)

include_directories(include)

add_executable(onnx_minimal_example src/main.cpp)


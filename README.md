
# onnx-minimal-example

ONNX minimal example / test on an object detection neural network.

## Requirements

For bare metal:
- C++ compiler
- CMake
- ONNX Runtime
- OpenCV

or, for docker:
- Docker Engine :)

## Building
### Bare metal - TODO
- Create a subdirectory in this directory named ```build``` and ```cd``` there.
- Run the command ```cmake ..```.
- Run the command ```make``` to finaly compile the source code into a binary.

### Docker - TODO
- Run the command ```docker build -t onnx_minimal_example .```.

## Running
### Bare metal
- Run the command ```./onnx_minimal_example```.

### Docker - TODO
- Run the command ```docker run --rm onnx_minimal_example```.


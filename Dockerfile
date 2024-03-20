FROM carlostojal/onnx-opencv:latest

# copy the files
COPY . /app

# set the working directory
WORKDIR /app

# build the app
RUN mkdir build
WORKDIR /app/build
RUN cmake ..
RUN make -j4

# run the app
CMD ["./onnx_minimal_example", "path/to/image.jpg", "../model/damoyolo_fsoco.onnx"]

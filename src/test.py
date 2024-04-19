import onnxruntime
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--img", type=str, required=True)
args = parser.parse_args()

session = onnxruntime.InferenceSession(args.model)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

print("Input name: ", input_name)
print("Input shape: ", input_shape)
print("Input type: ", input_type)

img = cv2.imread(args.img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640,640))
img = img.astype(np.float32)
img /= 255.0
img = np.expand_dims(img, 0)
img = np.moveaxis(img, 3, -3)

print(img)

outputs = session.run(None, {"images": img})

outputs[0] *= 1000
print(outputs[0])


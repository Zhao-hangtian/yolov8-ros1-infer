'''
Author: Zhao Hangtian iamzhaohangtian@gmail.com
Date: 2023-11-22 05:40:58
LastEditors: Zhao Hangtian iamzhaohangtian@gmail.com
LastEditTime: 2024-06-23 14:12:54
Description: 

Copyright (c) 2023 by Zhao Hangtian, All Rights Reserved. 
'''
from ultralytics import YOLO

model_path = '/home/nv/zht_ws/yolov8n.pt'
# "yolov8n.pt"

# Load a YOLOv8n PyTorch model
model = YOLO(model_path)

# Export the model
model.export(format="engine", batch=4)  # creates 'yolov8n.engine'

# Load the exported TensorRT model
trt_model = YOLO(model_path.replace('.pt', '.engine'))

# Run inference
results = trt_model("/home/nv/zht_ws/src/py_yolov8/src/imgs/oak_ffc_4p_image_CAM_C_compressed/2023-11-22-05-39-36-722241.jpg")

print(results)
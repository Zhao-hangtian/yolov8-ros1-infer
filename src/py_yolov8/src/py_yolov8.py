#!/usr/bin/env python3
'''
Author: Zhao Hangtian jp-vip@qq.com
Date: 2024-05-25 15:25:43
LastEditors: Zhao Hangtian iamzhaohangtian@gmail.com
LastEditTime: 2024-06-23 14:02:38
Description: 动态支持多路图像topic的目标检测, 兼容 Image / CompressedImage 消息类型, 推理模型解耦, 可选择任意模型.

Copyright (c) 2024 by Zhao Hangtian, All Rights Reserved. 
'''


import os
import rospy
import sys
from sensor_msgs.msg import Image, CompressedImage
from custom_msgs.msg import BoundingBoxArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import numpy as np
import time
from std_msgs.msg import Header

model = None
model_path =None
# Create a CvBridge object
bridge = CvBridge()

# Image_OR_CompressedImage = None

# Buffer to store images
image_buffer = []

# Dictionary to store publishers for each topic
publishers = {}

time_sum = 0

# coco_person_only = False

def callback(data, params):
    topic, coco_person_only=params
    try:
        if topic.endswith('compressed'):
            np_arr = np.frombuffer(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            timestamp = data.header.stamp
        else:
            # Convert ROS Image message to OpenCV image
            cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        # time_error = rospy.Time.now() - data.header.stamp
        # print('time from publish to receive:%f'%time_error.to_sec())
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    # Add the image to the buffer
    image_buffer.append((cv_image, topic, timestamp))

    # If we have received images from all topics, process the batch
    if len(image_buffer) == len(image_topics):
        process_batch(coco_person_only)

def compress_image(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(encimg, 1)
    return compressed_image

def process_batch(coco_person_only=False):
    global image_buffer
    global time_sum, cnt

    # Extract images from the buffer
    images = [img for img, _, _ in image_buffer]
    
    start_time = time.time()
    # Perform batch inference
    results = model(images, verbose=False)  
    end_time = time.time()
    total_time = end_time - start_time
    
    # rospy.loginfo("Detection time: %f", total_time)
    

    # Iterate over each image and its corresponding results
    for (cv_image, topic, timestamp), result in zip(image_buffer, results):
        
        # result exam & format:
        # bbox_xyxy: array([[     474.73,      729.63,      787.31,      799.78]], dtype=float32)
        # bbox_cls: array([          0], dtype=float32)
        # bbox_conf: array([    0.38333], dtype=float32)
        
        # bbox_xyxy = result.boxes.xyxy.cpu().numpy().flatten()
        # bbox_cls = result.boxes.cls.cpu().numpy().flatten()
        # bbox_conf = result.boxes.conf.cpu().numpy().flatten()
        
        bbox_xyxy_all = result.boxes.xyxy.cpu().numpy()
        bbox_cls_all = result.boxes.cls.cpu().numpy()
        bbox_conf_all = result.boxes.conf.cpu().numpy()
        
        if coco_person_only:
            # print('enable coco_person_only')
            mask_cls = bbox_cls_all == 0
            bbox_xyxy_all = bbox_xyxy_all[mask_cls]
            bbox_cls_all = bbox_cls_all[mask_cls]
            bbox_conf_all = bbox_conf_all[mask_cls]
            
        
        # 使用布尔索引筛选出满足条件的框
        mask_conf = bbox_conf_all > bbox_conf_thre
        bbox_xyxy = bbox_xyxy_all[mask_conf]
        bbox_cls = bbox_cls_all[mask_conf]
        bbox_conf = bbox_conf_all[mask_conf]
        
        # Annotate the image with bounding boxes
        if not coco_person_only:
            annotated_frame = result.plot() # contains bbox conf < conf_thre
        else:
            # 读取原始图片
            image = result.orig_img
            print(bbox_xyxy, bbox_conf, bbox_cls)
            for box, score, cls in zip(bbox_xyxy, bbox_conf, bbox_cls):
                x1, y1, x2, y2 = map(int, box)
                # 绘制矩形框，颜色为红色，线条宽度为2
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 构建标注文本
                label = f"Class: {int(cls)}, Conf: {score:.2f}"
                
                # 计算文本大小
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # 在检测框上方绘制文本背景
                cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 0, 255), -1)
                
                # 绘制文本
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            annotated_frame = image
        
        # 应用掩码过滤结果
        bbox_xyxy = bbox_xyxy.flatten()
        bbox_cls = bbox_cls.flatten()
        bbox_conf = bbox_conf.flatten()

        # Log the bounding box information
        if len(bbox_conf) != 0:
            rospy.loginfo("Topic: %s, Bounding boxes: %s", topic, bbox_xyxy)
            rospy.loginfo("Topic: %s, Bounding classes: %s", topic, bbox_cls)
            rospy.loginfo("Topic: %s, Bounding confidences: %s", topic, bbox_conf)

        header = Header()
        header.stamp = rospy.Time.now()
        # header.stamp = timestamp # Consistent timestamp, same as original img
        # print('Publisher error time%f'%(rospy.Time.now()-timestamp).to_sec())
        # Publish bounding box information
        bbox_msg = BoundingBoxArray()
        bbox_msg.header = header
        bbox_msg.bbox_xyxy = bbox_xyxy
        bbox_msg.bbox_cls = bbox_cls
        bbox_msg.bbox_conf = bbox_conf
        publishers[topic]['bbox'].publish(bbox_msg)


        # Publish the annotated image
        try:
            if publish_raw:
                annotated_image_msg = bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
                annotated_image_msg.header = header
                publishers[topic]['image'].publish(annotated_image_msg)
            # if publish_compressed:
            #     start_time = time.time()
            #     # compressed_image = compress_image(annotated_frame)
            #     # compressed_image_msg = bridge.cv2_to_imgmsg(compressed_image, "bgr8")
            #     compressed_image_msg = bridge.cv2_to_compressed_imgmsg(annotated_frame, dst_format='jpg')
            #     compressed_image_msg.header = header
            #     publishers[topic]['compressed_image'].publish(compressed_image_msg)
            #     end_time = time.time()
            #     total_time = end_time - start_time
            #     rospy.loginfo("Compress time: %f", total_time)
        except CvBridgeError as e:
            rospy.logerr(e)

    # Clear the buffer
    image_buffer = []

def image_listener(image_topics, coco_person_only=False):
    global publishers

    # Initialize subscribers and publishers
    for topic in image_topics:
        
        Image_OR_CompressedImage = Image if not topic.endswith('compressed') else CompressedImage
        print(f'topic {topic} 使用的是 {"Image" if not topic.endswith("compressed") else "CompressedImage"} 消息类型! 如无网络传输需求(无跨机器通信)建议使用Image以减轻CPU负担')
        
        rospy.Subscriber(topic, Image_OR_CompressedImage, callback, callback_args=(topic, coco_person_only))
        
        # Generate output topic names
        base_topic_name = topic.split('/')[-2]  # Adjust this based on your topic structure
        annotated_image_topic = f'/detection/{base_topic_name}/annotated_image'
        bbox_info_topic = f'/detection/{base_topic_name}/bbox_info'
        
        # Create publishers for each topic
        publishers[topic] = {
            'image': rospy.Publisher(annotated_image_topic, Image, queue_size=10),
            'bbox': rospy.Publisher(bbox_info_topic, BoundingBoxArray, queue_size=10)
        }

    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('image_listener', anonymous=True)
    # if len(sys.argv) < 5:
    #     print("Usage: rosrun py_yolov8 py_yolov8.py <model_path> <image_topic1> [<image_topic2> ...]")
    #     print("Example: rosrun py_yolov8 py_yolov8.py /home/nv/zht_ws/best.pt /robot/image_CAM_A/compressed /robot/image_CAM_B/compressed")
    #     sys.exit(1)
        
    # model_path = sys.argv[1]
    model_path = rospy.get_param('~model_path')
    
    print(f'使用模型: {model_path}')
    
    if not os.path.exists(model_path):
        print(f'{model_path} 文件不存在,请检查!')
        exit(-1)

    # image_topics = sys.argv[2:]
    image_topics = rospy.get_param('~image_topics')

    print('监听并处理的topic(s):')
    for topic in image_topics:
        print(topic)

    bbox_conf_thre = rospy.get_param('~bbox_conf_thre', 0.7)
    print('Bbox threshold:', bbox_conf_thre)
    # publish_compressed = rospy.get_param('~compress_img', True)
    publish_raw = rospy.get_param('~raw_img', True)
        
    print("初始化,模型加载中...")
    # Initialize the YOLOv8 model
    # COCO类别筛选
    coco_person_only = False
    if model_path.split('/')[-1].split('.')[0] == 'yolov8n':
        coco_person_only = True
        print(f'使用ultralytics的{model_path.split("/")[-1]}模型!仅导出person类别!(id=1)')
        
    model = YOLO(model_path)
    print("模型加载完毕")
    
    image_listener(image_topics, coco_person_only)

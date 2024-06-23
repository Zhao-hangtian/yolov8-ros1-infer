#!/usr/bin/env python3
from datetime import datetime
import os
import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

# Create a CvBridge object
bridge = CvBridge()
# Dictionary to store the last save time for each topic
last_save_time = {}
# Directory to save images
current_dir = os.path.dirname(os.path.realpath(__file__))

def callback(data, topic):
    try:
        np_arr = np.frombuffer(data.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        current_time = rospy.Time.now()
        
        # Save the image if the save frequency is reached
        if topic not in last_save_time or (current_time - last_save_time[topic]).to_sec() >= (1/save_frequency):
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            save_dir = os.path.join(current_dir, "imgs", topic.strip("/").replace("/", "_"))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            target_width, target_height = 640, 480  # Set resolution
            resized_image = cv_image
            if (cv_image.shape[1] > target_width) or (cv_image.shape[0] > target_height):
                resized_image = cv2.resize(cv_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            compress_quality = 50 # Set compression quality, current config lead to final size about 25KB
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_quality]
            result, encimg = cv2.imencode('.jpg', resized_image, encode_param)
            if result:
                file_path = os.path.join(save_dir, f"{timestamp}.jpg")
                with open(file_path, 'wb') as f:
                    f.write(encimg)
                rospy.loginfo(f"Compressed and saved image to {file_path}")
            else:
                rospy.logwarn("Image compression failed.")
            
            last_save_time[topic] = current_time       
    except CvBridgeError as e:
        rospy.logerr(e)
        return


def image_listener(image_topics):
    # Initialize subscribers
    for topic in image_topics:
        rospy.Subscriber(topic, CompressedImage, callback, callback_args=topic)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('image_saver', anonymous=True)
    
    # Get the parameters
    image_topics = rospy.get_param('~image_topics')
    save_frequency = rospy.get_param('~save_frequency', 1.0)
    save_img = rospy.get_param('~save_img', False)
    print('Save frequency:', save_frequency)
    print('Save frequency:', save_img)
    print('监听并处理的topic(s):')
    for topic in image_topics:
        print(topic)
    
    image_listener(image_topics)

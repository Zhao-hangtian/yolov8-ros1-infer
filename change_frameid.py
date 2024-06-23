#!/usr/bin/env python
import rosbag
from sensor_msgs.msg import PointCloud2
in_name = '2023-11-22-06-03-52.bag'
out_name = '2023-11-22-06-03-52_map.bag'
with rosbag.Bag(out_name, 'w') as outbag:
    for topic, msg, t in rosbag.Bag(in_name).read_messages():
        if topic == "/cloud_registered_body":
            # print("Type: ", type(msg))
            # print('topic good')
            if isinstance(msg, msg.__class__):
                msg.header.frame_id = "map"
                print('success!')
        outbag.write(topic, msg, t)


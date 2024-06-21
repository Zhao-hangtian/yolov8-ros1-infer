#!/usr/bin/env python3

from collections import deque
import copy
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from custom_msgs.msg import BoundingBoxArray, TargetCoordinate
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans
import std_msgs.msg

class TargetDistanceCalculator:
    def __init__(self):
        # Get parameters from the ROS parameter server
        self.publish_raw = rospy.get_param('~raw_img', True)
        self.dist_thre_add_point = rospy.get_param('~dist_thre_add_point', 0.1)
        self.dist_thre_final = rospy.get_param('~dist_thre_final', 0.1)
    
        self.bridge = CvBridge()
        
        # Define the camera matrix and distortion coefficients
        self.camera_matrix = np.array([[895.79468, 0.0, 674.04929], 
                                       [0.0, 896.49722, 392.10665], 
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0.031106, -0.056139, 0.000962, -0.001041, 0.000000])
        
        # Define the rotation matrices for each camera
        # NOTE: different type of drones may have different rotation matrices and translation vectors!
        self.rotation_matrix = {
            'CAM_A' : np.array([[0,  0, -1],
                                [1,  0,  0],
                                [0, -1,  0]]),
            'CAM_B' : np.array([[-1, 0,  0],
                                [0,  0, -1],
                                [0,  -1, 0]]),
            'CAM_C' : np.array([[0,   0, 1],
                                [-1,  0, 0],
                                [0,  -1, 0]]),
            'CAM_D' : np.array([[1,   0, 0],
                                [0,   0, 1],
                                [0,  -1, 0]]),
        }

        length = 0.01 # length of the arm
        height = 0.02
        self.translation_vector = {
            'CAM_A' : np.array([-length,  0, -height]),
            'CAM_B' : np.array([0,  -length,  -height]),
            'CAM_C' : np.array([length, 0,  -height]),
            'CAM_D' : np.array([0, length, -height]),
        }
        
        # Set the maximal accepted time difference between computed coordinates and (lidar and odom)
        self.diff = 0.5
        
        # Initialize the container of images with coordinates computed
        self.image_with_coord_msgs = {
            'CAM_A': [],
            'CAM_B': [],
            'CAM_C': [],
            'CAM_D': [],
        }
        self.accumulated_points = []
        self.received_bbox = False
        self.received_img = False

        #Subscribe to image and bounding box information of four cameras
        self.images = {}
        self.image_subs = {
            'CAM_A': rospy.Subscriber('/detection/image_CAM_A/annotated_image', Image, self.image_callback, 'CAM_A'),
            'CAM_B': rospy.Subscriber('/detection/image_CAM_B/annotated_image', Image, self.image_callback, 'CAM_B'),
            'CAM_C': rospy.Subscriber('/detection/image_CAM_C/annotated_image', Image, self.image_callback, 'CAM_C'),
            'CAM_D': rospy.Subscriber('/detection/image_CAM_D/annotated_image', Image, self.image_callback, 'CAM_D')
        }
        self.bboxes = {}
        self.bbox_subs = {
            'CAM_A': rospy.Subscriber('/detection/image_CAM_A/bbox_info', BoundingBoxArray, self.bbox_callback, 'CAM_A'),
            'CAM_B': rospy.Subscriber('/detection/image_CAM_B/bbox_info', BoundingBoxArray, self.bbox_callback, 'CAM_B'),
            'CAM_C': rospy.Subscriber('/detection/image_CAM_C/bbox_info', BoundingBoxArray, self.bbox_callback, 'CAM_C'),
            'CAM_D': rospy.Subscriber('/detection/image_CAM_D/bbox_info', BoundingBoxArray, self.bbox_callback, 'CAM_D')
        }

        # Subscribe to lidar point cloud data
        self.got_lidar = False
        self.point_cloud_data = None
        self.point_cloud_cache = deque(maxlen=20)     
        self.lidar_sub = rospy.Subscriber('/cloud_registered_body', PointCloud2, self.lidar_callback)

        # Subscribe to odometry
        self.position = None
        self.orientation = None
        self.odometry_cache = deque(maxlen=20)  
        # self.odometry = rospy.Subscriber('/Odometry_map', Odometry, self.odometry_callback)
        self.odometry = rospy.Subscriber('/ekf_quat/ekf_odom', Odometry, self.odometry_callback)

        # Publisher for target coordinates and images in body frame
        self.rate = 20 # Hz
        self.last_published_time = rospy.Time.now()
        self.target_pubs_coord = {
            'CAM_A': rospy.Publisher('/detection/image_CAM_A/target_coordinates', TargetCoordinate, queue_size=10),
            'CAM_B': rospy.Publisher('/detection/image_CAM_B/target_coordinates', TargetCoordinate, queue_size=10),
            'CAM_C': rospy.Publisher('/detection/image_CAM_C/target_coordinates', TargetCoordinate, queue_size=10),
            'CAM_D': rospy.Publisher('/detection/image_CAM_D/target_coordinates', TargetCoordinate, queue_size=10),
        }
        self.target_pubs_img_with_coord = {
            'CAM_A': rospy.Publisher('/detection/image_CAM_A/target_img_with_coord', Image, queue_size=10),
            'CAM_B': rospy.Publisher('/detection/image_CAM_B/target_img_with_coord', Image, queue_size=10),
            'CAM_C': rospy.Publisher('/detection/image_CAM_C/target_img_with_coord', Image, queue_size=10),
            'CAM_D': rospy.Publisher('/detection/image_CAM_D/target_img_with_coord', Image, queue_size=10),
        }

        # Publisher for target coordinates in world frame
        self.target_pubs_coord_world = [
            rospy.Publisher('/detection/target_coordinates/drone', PointCloud2, queue_size=1000),
            rospy.Publisher('/detection/target_coordinates/people', PointCloud2, queue_size=1000),
            rospy.Publisher('/detection/target_coordinates/box', PointCloud2, queue_size=1000),
        ]
        
        # Publisher for target bearing
        self.target_pub_bearing_camera = rospy.Publisher('/detection/target_bearing', Point, queue_size=10)

    def image_callback(self, msg, camera):
        # This callback function processes image data
        self.images[camera] = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.received_img = True
        # time_error = rospy.Time.now() - msg.header.stamp
        # print('time from publish to receive:%f'%time_error.to_sec())

    def bbox_callback(self, msg, camera):
        # This callback function processes bounding box information
        self.bboxes[camera] = msg
        bbox_xyxy = msg.bbox_xyxy
       
        if len(bbox_xyxy) != 0:
            self.received_bbox = True

    def lidar_callback(self, msg):
        # This callback function processes lidar point cloud data
        # self.point_cloud_data = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        data = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        stamp = msg.header.stamp
        self.point_cloud_cache.append({'stamp': stamp, 'data': data})
        self.got_lidar = True

    def odometry_callback(self, msg):
        # This callback function processes odometry data
        # self.position = msg.pose.pose.position
        # self.orientation = msg.pose.pose.orientation
        odometry_data = msg.pose.pose
        stamp = msg.header.stamp
        self.odometry_cache.append({'stamp': stamp, 'data': odometry_data})

    def publish_image_with_coordinates(self, camera):
        if self.received_img == False:
            return
        
        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.last_published_time).to_sec()

        if elapsed_time < 1.0 / self.rate:
            return  # Not enough time has passed, return without publishing
        
        image = self.images[camera].copy()

        for target_msg, bbox_center in self.image_with_coord_msgs[camera]:
            # Put the depth and 3D coordinates
            depth_and_position = "Depth: %.2f\n3D Position: (%.2f, %.2f, %.2f)" % (target_msg.depth, target_msg.target_x, target_msg.target_y, target_msg.target_z)
            depth, position = depth_and_position.split('\n')
            cv2.putText(image, depth, (int(bbox_center[0]), int(bbox_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(image, position, (int(bbox_center[0]), int(bbox_center[1] + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(image, depth_and_position, (int(bbox_center[0]), int(bbox_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if self.publish_raw:
            original_image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.target_pubs_img_with_coord[camera].publish(original_image_msg)

        self.received_img = False
        self.last_published_time = current_time
    
    # Transform given point from body frame to world frame
    def transform_point(self, point): 
        # Extract the current position and orientation of the drone
        position  = self.position
        orientation = self.orientation
        rot_matrix = tf_trans.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
        # Extract the 3x3 rotation matrix from the 4x4 transformation matrix
        rot_matrix = rot_matrix[:3, :3]  # Take the top-left 3x3 submatrix
        position_array = np.array([position.x, position.y, position.z])
        # Using the inverse of the rotation matrix: transform the point to the world frame
        transformed_point = np.dot(point, np.linalg.inv(rot_matrix)) + position_array
        # Return the transformed point in the world frame
        return transformed_point
    
    def publish_target_coordinates_world_frame(self, point, target_class):
        # Add the new point to the accumulated points list
        transformed_point = self.transform_point(point) # Transform to world frame
        self.accumulated_points.clear()
        self.accumulated_points.append(transformed_point) # Only publish one point

        # Create a PointCloud2 message
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]

        # Convert the accumulated points list to a numpy array
        points_array = np.array(self.accumulated_points)

        cloud = pc2.create_cloud(header, fields, points_array)

        self.target_pubs_coord_world[int(target_class)].publish(cloud)
        return

    def publish_target_bearing(self, bearing):
        # Create a Point message to publish the target bearing in body frame
        bearing_point = Point()
        bearing_point.x = bearing[0]
        bearing_point.y = bearing[1]
        bearing_point.z = bearing[2]
        self.target_pub_bearing_camera.publish(bearing_point)

    def check_point_validation(self, camera, point):
        # Validate the point based on the camera orientation and point coordinates
        # NOTE: This function should be adjusted based on the camera orientation.
        if(camera == 'CAM_A' and point[0] < 0):
            return True
        elif(camera == 'CAM_B' and point[1] < 0):
            return True
        elif(camera == 'CAM_C' and point[0] > 0):
            return True
        elif(camera == 'CAM_D' and point[1] > 0):
            return True
        else:
            return False
    
    def select_on_stamp(self, bbox_time, diff):
        # Find the closest lidar and odometry data based on the bounding box timestamp
        lidar_idx, lidar_time, lidar_diff = self.find_closest_diff(self.point_cloud_cache, bbox_time)
        odom_idx, odom_time, odom_diff = self.find_closest_diff(self.odometry_cache, bbox_time)

        if lidar_idx is not None:
            self.point_cloud_data = self.point_cloud_cache[lidar_idx]['data']
        if odom_idx is not None:
            self.position = self.odometry_cache[odom_idx]['data'].position
            self.orientation = self.odometry_cache[odom_idx]['data'].orientation

        print(f"Lidar index: {lidar_idx}, time difference: {lidar_diff}")
        if lidar_diff > diff:
            rospy.loginfo('Lidar diff bigger than given!')
        print(f"Odometry index: {odom_idx}, time difference: {odom_diff}")
        if odom_diff > diff:
            rospy.loginfo('Odom diff bigger than given!')

        return lidar_time, lidar_idx, odom_time, odom_idx

    def find_closest_diff(self, data_list, target_time):
        data_list = data_list.copy()
        closest_idx = None
        closest_time = None
        min_diff = float('inf')

        # Iterate over the data list to find the closest timestamp to the0 target time
        for i, data in enumerate(data_list):
            time_diff = abs(data['stamp'] - target_time).to_sec()
            if time_diff < min_diff:
                min_diff = time_diff
                closest_idx = i
                closest_time = data['stamp']

        return closest_idx, closest_time, min_diff

        
    def calculate_depth_and_position(self):
        """
            Determine 3D positions and depths of targets from camera bounding boxes using LiDAR data:
            convert coordinates, find closest LiDAR points, and publish target positions and depths.
        """
        bboxes_copy = copy.deepcopy(self.bboxes)
        for camera, bboxes in bboxes_copy.items():
            self.image_with_coord_msgs[camera].clear()

            bbox_time = bboxes.header.stamp
            bbox_class = bboxes.bbox_cls
            bbox_xyxy = bboxes.bbox_xyxy
            num_bboxes = len(bbox_xyxy) // 4  # Each bounding box has 4 coordinates

            for i in range(num_bboxes):
                x1 = bbox_xyxy[4 * i]
                y1 = bbox_xyxy[4 * i + 1]
                x2 = bbox_xyxy[4 * i + 2]
                y2 = bbox_xyxy[4 * i + 3]
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                # Convert image coordinates to camera coordinates
                center_x_camera = (center_x - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
                center_y_camera = (center_y - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
                
                # Compute the expression of the line in body coordinates
                rotation_matrix = self.rotation_matrix[camera]
                translation_vector = self.translation_vector[camera]
                camera_origin_point = translation_vector
                end_point_camera = np.array([center_x_camera, center_y_camera, 1.0])
                end_point_world = np.dot(rotation_matrix, end_point_camera) + translation_vector # camera to body frame
                bearing_camera = (end_point_world - camera_origin_point)
                bearing_camera /= np.linalg.norm(bearing_camera)                
                line = [camera_origin_point, bearing_camera]

                self.publish_target_bearing(bearing_camera)
                
                if self.got_lidar is False:
                    print('Not get lidar data!')
                    continue # No need to compute the depth and position if there is no lidar data
                
                def distance_to_line(point, line):
                    # Calculate the distance from a point to a line
                    A, direction_vector = line
                    B = A + direction_vector
                    # Compute distance = |P-A| x |B-A| / |B-A|
                    cross_product = np.cross(point - A, B - A)
                    distance = np.linalg.norm(cross_product) / np.linalg.norm(B - A)
                    return distance
                
                # Find the closest point in the lidar data to the line
                closest_point = None
                min_distance = float('inf')
                closest_point_list = []

                # Select corresponding point cloud data according to Bbox stamp
                self.select_on_stamp(bbox_time, self.diff)

                # Iterate over each point in the lidar data
                for point in self.point_cloud_data:
                    # Validate the point based on the camera type and point coordinates
                    if self.check_point_validation(camera, point) == False:
                        continue
                    point = np.array(point)
                    distance = distance_to_line(point, line)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = point
                    if distance <= self.dist_thre_add_point:
                        closest_point_list.append(point)

                closest_list_len = len(closest_point_list)
                
                rospy.loginfo('closest_list_len:%d'%(closest_list_len))
                rospy.loginfo('min_distance:%f'%(min_distance))
                
                if closest_list_len != 0 and min_distance <= self.dist_thre_final:
                    # Calculate depth and 3D position
                    sum = [0, 0, 0]
                    for p in closest_point_list:
                        sum[0] += p[0]
                        sum[1] += p[1]
                        sum[2] += p[2]
                    closest_point[0] = sum[0]/closest_list_len
                    closest_point[1] = sum[1]/closest_list_len
                    closest_point[2] = sum[2]/closest_list_len        

                    depth = np.sqrt(closest_point[0]**2 + closest_point[1]**2 + closest_point[2]**2)
        
                    # print("Camera: %s, Depth: %.2f m, 3D Position in World: (%.2f, %.2f, %.2f), Min dist:%.2f" %
                        #   (camera, depth, *closest_point, min_distance))
                    
                    # Publish the target coordinates and imgs
                    target_msg = TargetCoordinate()
                    target_msg.target_class = bbox_class[i]
                    target_msg.target_x = closest_point[0]
                    target_msg.target_y = closest_point[1]
                    target_msg.target_z = closest_point[2]
                    target_msg.depth = depth
                    self.image_with_coord_msgs[camera].append((target_msg, (center_x, center_y)))

                    # Publish the target coordinates in world frame
                    self.publish_target_coordinates_world_frame(closest_point, bbox_class[i])
            
            # Publish the image with coordinates for the current camera
            self.publish_image_with_coordinates(camera)
        self.bboxes.clear()


if __name__ == '__main__':
    rospy.init_node('target_distance_calculator')
    calculator = TargetDistanceCalculator()
    
    while not rospy.is_shutdown():
        if calculator.received_bbox:
            calculator.calculate_depth_and_position()
            calculator.received_bbox = False
        pass
#!/usr/bin/env python
import threading
import rospy
import math
import tf
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image, PointCloud2
import numpy as np
from tf.transformations import euler_from_quaternion
import tf2_ros
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
from tf2_geometry_msgs import PointStamped
import skimage
import message_filters
from skimage.morphology import skeletonize
import cv2, cv_bridge
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud
import ros_numpy
from geometry_msgs.msg import Point32

class Identifier():
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer(cache_time=rospy.Duration(60.))
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.ground_truth_position = Pose()
        self.ground_truth_twist = Twist()
        self.noisy_position = Pose()
        self.noisy_twist = Twist()
        self.robot_frame = "mobile_base"
        self.model_states_sub = rospy.Subscriber('gazebo/model_states',ModelStates,self.model_states_callback)
        self.camera_info_sub = rospy.Subscriber('usb_cam/rgb/camera_info',CameraInfo,self.camera_info_callback)
        self.camera_info = CameraInfo()
        self.depth_camera_info_sub = rospy.Subscriber('usb_cam/depth/camera_info',CameraInfo,self.depth_camera_info_callback)
        self.depth_camera_info = CameraInfo()
        # self.green_track_sub = rospy.Subscriber('usb_cam/rgb/image_raw',Image,self.green_track_callback)
        self.img_sub = message_filters.Subscriber('usb_cam/rgb/image_raw', Image)
        self.pcl_sub = message_filters.Subscriber('usb_cam/depth/points', PointCloud2)
        self.measurements = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.pcl_sub], queue_size=5, slop=0.1)
        self.measurements.registerCallback(self.measurement_callback)
        self.closest_goal_pub = rospy.Publisher('camera_detection',PointCloud,queue_size=10)


        self.R = np.eye(2)
        self.t = np.zeros((2,1))
        self.bridge = cv_bridge.CvBridge()
        self.success = False
        self.current_segment_lock = threading.Lock()

        while not self.success:
          try:
            trans = self.tfBuffer.lookup_transform('base_link', 'mono_camera_link', rospy.Time(0))
            theta = euler_from_quaternion([trans.transform.rotation.x,
                                         trans.transform.rotation.y,
                                         trans.transform.rotation.z,
                                         trans.transform.rotation.w])[2]
          except:
            pass
          else:
              self.R = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]])
              self.t = np.array([[trans.transform.translation.x],
                             [trans.transform.translation.y]])
              self.t0 = np.array([[trans.transform.translation.x],
                                [trans.transform.translation.y]])
              self.success = True



    def model_states_callback(self,states_msg):
        frame_list = states_msg.name
        for index in range(len(frame_list)):
            if frame_list[index] == self.robot_frame:
                model_index = index
                self.ground_truth_position = states_msg.pose[model_index]
                self.ground_truth_twist = states_msg.twist[model_index]


    def measurement_callback(self,img_msg,pcl_msg):
        self.current_segment_lock.acquire()
        # camera_p = self.camera_info.P
        # print("self.camera_info.height",self.camera_info.height,"self.camera_info.width",self.camera_info.width)
        # camera_dp = self.depth_camera_info.P
        # print("self.depth_camera_info.height",self.depth_camera_info.height,"self.depth_camera_info.width",self.depth_camera_info.width)

        try:
            trans = self.tfBuffer.lookup_transform('base_link', 'mono_camera_link', rospy.Time(0))
            theta = euler_from_quaternion([trans.transform.rotation.x,
                                         trans.transform.rotation.y,
                                         trans.transform.rotation.z,
                                         trans.transform.rotation.w])[2]
            self.R = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]])
            self.t = np.array([[trans.transform.translation.x],
                             [trans.transform.translation.y]])

        except:
            print("failed to find available tf")
        try:
            pc = ros_numpy.numpify(pcl_msg)
            points=np.zeros((pc.shape[0],pc.shape[1],3))
            points[:,:,0]=pc['x']
            points[:,:,1]=pc['y']
            points[:,:,2]=pc['z']
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([30, 150, 150])
            upper_green = np.array([80, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            imask = mask_green > 0
            inds = np.argwhere(imask.flatten())
            # print("skeleton_start","rostime",rospy.Time.now().to_sec())
            skeleton = skeletonize(imask)
            [convert_pixel_row,convert_pixel_col]= np.where(skeleton)
            points2 = PointCloud()
            if len(convert_pixel_row):
                depth_points = []
                depth_points_robot = []
                for skel_index in range(0,len(convert_pixel_row)):
                    temp_depth_points = points[convert_pixel_row[skel_index],convert_pixel_col[skel_index]]
                    depth_points.append(temp_depth_points)
                    depth_points_robot.append(np.matmul(self.R.T, [[temp_depth_points[0]],[temp_depth_points[1]]]) - self.t)
                points2.header.stamp = pcl_msg.header.stamp
                points2.header.frame_id = "front_camera"
                for one_depth_point in depth_points_robot:
                    points2.points.append(Point32(one_depth_point[0],one_depth_point[1],0))
                self.closest_goal_pub.publish(points2)
            else:
                current_min_points = []




        except cv_bridge.CvBridgeError as e:
            print(e)

        self.current_segment_lock.release()


    def camera_info_callback(self,msg):
        self.camera_info = msg

    def depth_camera_info_callback(self,msg):
        self.depth_camera_info = msg

    # def from_index_to_camera_coordinate(self,index,camera_info):
    #     u0 = camera_info.P[2]
    #     v0 = camera_info.P[6]
    #     cam_P = np.reshape(camera_info.P,(3,4))
    #     cam_P_33 = np.array(cam_P)[0:3,0:3]
    #     index_in_camera_coordinate = []
    #     index_in_world_coordinate = []
    #     inv_cam_P_33 = np.linalg.inv(cam_P_33)
    #     for one_index in index:
    #         row = np.floor(one_index/self.cam_resolution_x)
    #         col = np.floor(one_index - row * self.cam_resolution_x)
    #         cam_x = np.floor(col - u0)
    #         cam_y = np.floor(row - v0)
    #         index_in_camera_coordinate.append([cam_x,cam_y])
    #         world_coordinate_point = np.matmul(inv_cam_P_33,np.transpose([cam_x - cam_P[0,3],cam_y - cam_P[1,3],1-cam_P[2,3]]))
    #         index_in_world_coordinate.append(np.transpose(world_coordinate_point))
    #     return index_in_world_coordinate


if __name__ == '__main__':
  rospy.init_node('tf_and_track_detection')
  identifier = Identifier()
  rospy.spin()

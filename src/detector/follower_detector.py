#!/usr/bin/env python
# BEGIN ALL
import rospy
import cv2, cv_bridge
import numpy as np
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
from sensor_msgs.msg import Image, PointCloud2
import message_filters 
from sensor_msgs import point_cloud2
from nav_msgs.msg import Odometry
from followbot.msg import MGSMeasurement, MGSMeasurements, MGSCommand

class Detector:
  def __init__(self):
    # ros
    self.bridge = cv_bridge.CvBridge()
    self.img_sub = message_filters.Subscriber('camera/rgb/image_raw', Image)
    self.pcl_sub = message_filters.Subscriber('camera/depth/points', PointCloud2)   
    self.odom_sub = message_filters.Subscriber('odom', Odometry)   
    self.measurements = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.pcl_sub, self.odom_sub], queue_size=5, slop=0.1)
    self.measurements.registerCallback(self.measurement_callback)
    self.command_sub = rospy.Subscriber('mgs_command', MGSCommand, self.mgs_command_callback)
    self.measurements_pub = rospy.Publisher('mgs', MGSMeasurements, queue_size=10)
    self.detection_img_pub = rospy.Publisher('detection_image', Image, queue_size=10)
    self.track_width = rospy.get_param('~track_width', 0.083) # 0.08216275
    self.follow_left = False
  

  def mgs_command_callback(self, msg):
    if msg.command == MGSCommand.FOLLOW_LEFT:
      self.follow_left = True
    elif msg.command == MGSCommand.FOLLOW_RIGHT:
      self.follow_left = False
    else:
      rospy.error('Command not defined')
    

  def measurement_callback(self, img_msg, pcl_msg, odom_msg):
    # Initalize measurements:
    detections_msg = MGSMeasurements()
    detections_msg.header.stamp = rospy.Time.now()
    detections_msg.header.frame_id = pcl_msg.header.frame_id
    detections_msg.odometry = odom_msg
    detections_msg.measurements = []

    # read point cloud data
    cloud_pts = pointcloud2_to_xyz_array(pcl_msg)
    # height is 0.19444602727890015

    # read image:
    try:
      cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
      hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    except cv_bridge.CvBridgeError as e:
      print(e)
    
    # green:
    lower_green = np.array([30, 150, 150])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # blue:
    lower_blue = np.array([ 90, 150, 150])
    upper_blue = np.array([150, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # red:
    lower_red = np.array([ 0, 150, 150])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
      # also need to add in line crossing (if width if very large?)
    
    # find track
    inds = np.argwhere(mask_green.flatten())
    if not inds.size == 0:
      # make measurement
      m_track = MGSMeasurement()
      m_track.type = MGSMeasurement.TRACK
      # get point
      if self.follow_left:
        m_track.position = -np.asscalar(cloud_pts[inds[0],0] + self.track_width / 2.) - 0.0125
      else:
        m_track.position = -np.asscalar(cloud_pts[inds[-1],0] - self.track_width / 2.) - 0.0125
      detections_msg.measurements.append(m_track)
      # draw point:
      if self.detection_img_pub.get_num_connections() > 0:
        M_tape = cv2.moments(mask_green)
        cx = int(M_tape['m10']/M_tape['m00'])
        cy = int(M_tape['m01']/M_tape['m00'])
        cv2.circle(cv_image, (cx, cy), 20, (0,0,255), -1)

    # find blue marker
    inds = np.argwhere(mask_blue.flatten())
    if not inds.size == 0:
      # make measurement
      m_marker = MGSMeasurement()
      m_marker.type = MGSMeasurement.MARKER
      # get point
      ind = int(np.median(inds))
      m_marker.position = -np.asscalar(cloud_pts[ind,0] + self.track_width / 2.) - 0.0125
      detections_msg.measurements.append(m_marker)
      # draw point:
      if self.detection_img_pub.get_num_connections() > 0:
        M_left = cv2.moments(mask_blue)
        # draw point:
        cx = int(M_left['m10']/M_left['m00'])
        cy = int(M_left['m01']/M_left['m00'])
        cv2.circle(cv_image, (cx, cy), 10, (120,255,120), -1)

    # find red marker
    inds = np.argwhere(mask_red.flatten())
    if not inds.size == 0:
      # make measurement
      m_marker = MGSMeasurement()
      m_marker.type = MGSMeasurement.MARKER
      # get point
      ind = int(np.median(inds))
      m_marker.position = -np.asscalar(cloud_pts[ind,0] + self.track_width / 2.) - 0.0125
      detections_msg.measurements.append(m_marker)
      # draw point:
      if self.detection_img_pub.get_num_connections() > 0:
        M_right = cv2.moments(mask_red)
        # draw point:
        cx = int(M_right['m10']/M_right['m00'])
        cy = int(M_right['m01']/M_right['m00'])
        cv2.circle(cv_image, (cx, cy), 10, (120,255,120), -1)

    # publish the sphd measurements:
    self.measurements_pub.publish(detections_msg)

    # get the detection image:
    if self.detection_img_pub.get_num_connections() > 0:
      detection_img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
      detection_img_msg.header.stamp = detections_msg.header.stamp
      detection_img_msg.header.frame_id = 'camera_depth_optical_frame'
      try:
        self.detection_img_pub.publish(detection_img_msg)
      except cv_bridge.CvBridgeError as e:
        print(e)


if __name__ == '__main__':
  rospy.init_node('follower_detector')
  detector = Detector()
  rospy.spin()
# END ALL

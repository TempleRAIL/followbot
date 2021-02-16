#!/usr/bin/env python
# BEGIN ALL
import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image, PointCloud2
import message_filters 
from sensor_msgs import point_cloud2
from nav_msgs.msg import Odometry
from followbot.msg import Measurement, Measurements

class Detector:
  def __init__(self):
    # ros
    self.bridge = cv_bridge.CvBridge()
    self.img_sub = message_filters.Subscriber('camera/rgb/image_raw', Image)
    self.pcl_sub = message_filters.Subscriber('camera/depth/points', PointCloud2)   
    self.odom_sub = message_filters.Subscriber('odom', Odometry)   
    self.measurements = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.pcl_sub, self.odom_sub], queue_size=5, slop=0.1)
    self.measurements.registerCallback(self.measurement_callback)
    self.measurements_pub = rospy.Publisher('detection_measurements', Measurements, queue_size=10)
    self.detection_img_pub = rospy.Publisher('detection_image', Image, queue_size=10)

  def measurement_callback(self, img_msg, pcl_msg, odom_msg):
    # Initalize measurements:
    detections_msg = Measurements()
    detections_msg.header.stamp = rospy.Time.now()
    detections_msg.header.frame_id = 'base_footprint'
    detections_msg.odometry = odom_msg
    detections_msg.measurements = []

    # read point cloud data
    cloud_points = list(point_cloud2.read_points(pcl_msg, skip_nans=True, field_names = ("x", "y", "z")))

    # read image:
    try:
      cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
      hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    except cv_bridge.CvBridgeError as e:
      print(e)
    # green:
    lower_green = numpy.array([ 30,  150,  150])
    upper_green = numpy.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # blue:
    lower_blue = numpy.array([ 90,  150,  150])
    upper_blue = numpy.array([150, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # red:
    lower_red = numpy.array([ 0,  150,  150])
    upper_red = numpy.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    
    # search area: 
    h, w, d = cv_image.shape
    search_top = 1*h/5
    search_bot = search_top + 80
    # green: tape detect 
    mask_green[0:search_top, 0:w] = 0
    mask_green[search_bot:h, 0:w] = 0
    M_tape = cv2.moments(mask_green)
    # blue: left marker 
    mask_blue[0:search_top, 0:w] = 0
    mask_blue[search_bot:h, 0:w] = 0
    M_left = cv2.moments(mask_blue)
    # red: right marker 
    mask_red[0:search_top, 0:w] = 0
    mask_red[search_bot:h, 0:w] = 0
    M_right = cv2.moments(mask_red)

    # tape detect 
    if M_tape['m00'] > 0:
      # make measurement
      m_track = Measurement()
      m_track.header = detections_msg.header
      m_track.type = Measurement.TRACK
      # draw point:
      cx = int(M_tape['m10']/M_tape['m00'])
      cy = int(M_tape['m01']/M_tape['m00'])
      cv2.circle(cv_image, (cx, cy), 20, (0,0,255), -1)
      # get position from point cloud
      idx = cx + cy*w
      '''
      # turtlbot2:
      self.x = -cloud_points[idx][1] + 0.18
      self.y = -cloud_points[idx][0] - 0.0125
      self.z = -cloud_points[idx][2] + 0.19 + 0.0102
      '''
      # fred robot:
      # self.x = -cloud_points[idx][1] + 0.72
      # self.y = -cloud_points[idx][0] - 0.0125
      # self.z = -cloud_points[idx][2] + 0.167
      m_track.position = -cloud_points[idx][0] - 0.0125
      # rospy.logdebug("Detect tape: (" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")")     
      detections_msg.measurements.append(m_track)

    # left marker: 
    if M_left['m00'] > 0:
      # make measurement
      m_left = Measurement()
      m_left.header = detections_msg.header
      m_left.type = Measurement.LEFT
      # draw point:
      cx = int(M_left['m10']/M_left['m00'])
      cy = int(M_left['m01']/M_left['m00'])
      cv2.circle(cv_image, (cx, cy), 10, (120,255,120), -1)
      # get position from point cloud
      idx = cx + cy*w
      m_left.position = -cloud_points[idx][0] - 0.0125
      rospy.logdebug("Detect left marker")
      detections_msg.measurements.append(m_left)

    # right marker: 
    if M_right['m00'] > 0:
      # make measurement
      m_right = Measurement()
      m_right.header = detections_msg.header
      m_right.type = Measurement.RIGHT
      # draw point:
      cx = int(M_right['m10']/M_right['m00'])
      cy = int(M_right['m01']/M_right['m00'])
      cv2.circle(cv_image, (cx, cy), 10, (120,255,120), -1)
      # get position from point cloud
      idx = cx + cy*w
      m_right.position = -cloud_points[idx][0] - 0.0125
      rospy.logdebug("Detect right marker")
      detections_msg.measurements.append(m_right)

    # publish the sphd measurements:
    self.measurements_pub.publish(detections_msg)

    # get the detection image:
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

#!/usr/bin/env python
# BEGIN ALL
import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class Follower:
  def __init__(self):
    self.bridge = cv_bridge.CvBridge()
    self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
    self.cmd_vel_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
    self.twist = Twist()
  def image_callback(self, msg):
    image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = numpy.array([ 30,  150,  150])
    upper_green = numpy.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    h, w, d = image.shape
    search_top = 1*h/5
    search_bot = search_top + 80
    mask_green[0:search_top, 0:w] = 0
    mask_green[search_bot:h, 0:w] = 0
    M = cv2.moments(mask_green)
    if M['m00'] > 0:
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
      cv2.circle(image, (cx, cy), 20, (0,0,255), -1)
      # BEGIN CONTROL
      err = cx - w/2
      self.twist.linear.x = 0.2
      self.twist.angular.z = -float(err) / 100
      self.cmd_vel_pub.publish(self.twist)
      # END CONTROL
    cv2.imshow("window", image)
    cv2.waitKey(1)

if __name__ == '__main__':
  rospy.init_node('follower_controller')
  follower = Follower()
  rospy.spin()
# END ALL
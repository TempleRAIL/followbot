#!/usr/bin/env python
# BEGIN ALL
import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image

class Follower:
  def __init__(self):
    self.bridge = cv_bridge.CvBridge()
    #cv2.namedWindow("window", 1)
    self.image_sub = rospy.Subscriber('camera/rgb/image_raw', 
                                      Image, self.image_callback)
  def image_callback(self, msg):
    # BEGIN BRIDGE
    image = self.bridge.imgmsg_to_cv2(msg)
    # END BRIDGE
    # BEGIN HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # END HSV
    # BEGIN FILTER
    # green
    lower_green = numpy.array([ 30,  150,  150])
    upper_green = numpy.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # blue
    lower_blue = numpy.array([ 90,  150,  150])
    upper_blue = numpy.array([150, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # red
    lower_red = numpy.array([ 0,  150,  150])
    upper_red = numpy.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    '''
    # yellow
    lower_yellow = numpy.array([ 10,  150,  150])
    upper_yellow = numpy.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    '''
    # END FILTER
    masked = cv2.bitwise_and(image, image, mask=mask_blue)
    cv2.imshow("window", mask_green ) 
    cv2.waitKey(1)
    cv2.imshow("window1", mask_blue ) 
    cv2.waitKey(1)
    cv2.imshow("window2", mask_red ) 
    cv2.waitKey(1)
    # cv2.imshow("window3", mask_yellow ) 
    # cv2.waitKey(1)

rospy.init_node('follower')
follower = Follower()
rospy.spin()
# END ALL

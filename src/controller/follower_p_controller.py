#!/usr/bin/env python
# BEGIN ALL
import threading
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from followbot.msg import MGSMeasurements, MGSMeasurement, MGSMarker

class Follower:
  def __init__(self):
    self.measurement_sub = rospy.Subscriber('mgs', MGSMeasurements, self.mgs_callback)
    self.mgs_marker_sub = rospy.Subscriber('mgs_marker', MGSMarker, self.marker_callback)
    self.cmd_vel_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
    self.twist = Twist()
    self.twist_lock = threading.Lock()
    self.p = rospy.get_param('~p', 7.0) # proportional controller constant
    self.v = rospy.get_param('~v', 0.2) # nominal velocity
  

  def marker_callback(self, msg):
    self.twist_lock.acquire()
    try:
      if msg.type == MGSMarker.STOP:
        self.twist.linear.x = 0.
      elif msg.type == MGSMarker.START:
        self.twist.linear.x = self.v
    finally:
      self.twist_lock.release()

    
  def mgs_callback(self, msg):
    # get tape position
    pos = None
    #pos_list = []
    for m in msg.measurements:
      if m.type == MGSMeasurement.TRACK:
        pos = m.position

    '''
      pos_list.append(m.position)
    if(len(pos_list) != 0):
      pos = np.sum(pos_list)
    else:
      pos = None
    '''
    # set forward velocity
    self.twist_lock.acquire()
    try:
      if not pos is None:
        # self.twist.angular.z = -float(err) / 100 # turtlebot
        self.twist.angular.z = self.p * pos
      self.cmd_vel_pub.publish(self.twist)
    finally:
      self.twist_lock.release()
    # END CONTROL


if __name__ == '__main__':
  rospy.init_node('follower_controller')
  follower = Follower()
  rospy.spin()
# END ALL

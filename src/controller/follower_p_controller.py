#!/usr/bin/env python
# BEGIN ALL
import threading

import rospy
from geometry_msgs.msg import Twist
from followbot.msg import MGSMeasurements, MGSMeasurement, MGSCommand

class Follower:
  def __init__(self):
    self.measurement_sub = rospy.Subscriber('mgs', MGSMeasurements, self.mgs_callback)
    self.mgs_command_sub = rospy.Subscriber('mgs_command', MGSCommand, self.command_callback)
    self.cmd_vel_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
    self.twist = Twist()
    self.twist_lock = threading.Lock()
    self.p = rospy.get_param('~p', 7.0) # proportional controller constant
    self.v = rospy.get_param('~v', 0.2) # nominal velocity
  

  def command_callback(self, msg):
    self.twist_lock.acquire()
    try:
      if msg.command == MGSCommand.STOP:
        self.twist.linear.x = 0.
      elif msg.command == MGSCommand.START:
        self.twist.linear.x = self.v
    finally:
      self.twist_lock.release()

    
  def mgs_callback(self, msg):
    # get tape position
    pos = None
    for m in msg.measurements:
      if m.type == MGSMeasurement.TRACK:
        pos = m.position
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

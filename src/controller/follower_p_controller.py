#!/usr/bin/env python
# BEGIN ALL
import rospy
from geometry_msgs.msg import Twist
from followbot.msg import MGSMeasurements, MGSMeasurement

class Follower:
  def __init__(self):
    self.measurement_sub = rospy.Subscriber('detection_measurements', MGSMeasurements, self.callback)
    self.cmd_vel_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
    self.twist = Twist()
    self.omega_prev = 0.0
    self.p = rospy.get_param('p', 7.0)
    
  def callback(self, msg):
    # get tape position
    pos = None
    for m in msg.measurements:
      if m.type == MGSMeasurement.TRACK:
        pos = m.position
    
    # set forward velocity
    self.twist.linear.x = 0.2
    '''
    # turtlbot:
    self.twist.angular.z = -float(err) / 100
    '''
    # fred robot:
    if not pos is None:
      self.twist.angular.z = self.p * pos
    self.cmd_vel_pub.publish(self.twist)
    # END CONTROL

if __name__ == '__main__':
  rospy.init_node('follower_controller')
  follower = Follower()
  rospy.spin()
# END ALL

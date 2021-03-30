#!/usr/bin/env python
# BEGIN ALL
import threading
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from followbot.msg import MGSMeasurements, MGSMeasurement, MGSMarker
from tf.transformations import euler_from_quaternion

class Follower:
  def __init__(self):
    self.measurement_sub = rospy.Subscriber('mgs', MGSMeasurements, self.mgs_callback)
    self.mgs_marker_sub = rospy.Subscriber('mgs_marker', MGSMarker, self.marker_callback)
    self.cmd_vel_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
    self.twist = Twist()
    self.twist_lock = threading.Lock()
    self.scale_factor = 0.795 # from true world to simulation
    self.p = self.scale_factor*rospy.get_param('~p', 7.0) # proportional controller constant
    self.v1 = self.scale_factor*rospy.get_param('~v', 0.666) # nominal velocity (1.49 MPH)
    self.v2 = self.scale_factor*rospy.get_param('~v', 0.782) # nominal velocity (1.75 MPH)
    self.v3 = self.scale_factor*rospy.get_param('~v', 0.849) # nominal velocity (1.90 MPH)
    self.v4 = self.scale_factor*rospy.get_param('~v', 0.939) # nominal velocity (2.10 MPH)
    self.v_turn = self.scale_factor*rospy.get_param('~v', 0.425) # nominal velocity (0.95 MPH)
    self.v_laser = self.scale_factor*rospy.get_param('~v', 0.308) # nominal velocity (0.69 MPH)
    self.v_stop = self.scale_factor*rospy.get_param('~v', 0.125) # nominal velocity (0.28 MPH)
    self.pose_history = []
    self.curve_flag = 0 # 0 for no curve, 1 for curve
    self.hypothesis_radius = self.scale_factor * 1


  def marker_callback(self, msg):
    self.twist_lock.acquire()
    try:
      if msg.command == MGSMarker.STOP:
        self.twist.linear.x = 0.
      elif msg.command == MGSMarker.START:
        self.twist.linear.x = self.v1
      elif msg.command == MGSMarker.SPEED_1:
        self.twist.linear.x = self.v1
      elif msg.command == MGSMarker.SPEED_2:
        self.twist.linear.x = self.v2
      elif msg.command == MGSMarker.SPEED_3:
        self.twist.linear.x = self.v3
      elif msg.command == MGSMarker.SPEED_4:
        self.twist.linear.x = self.v4
    finally:
      self.twist_lock.release()


  def mgs_callback(self, msg):
    # get tape position
    pos = None
    #pos_list = []
    if len(self.pose_history) >= 15:
      self.pose_history.pop(0)
      self.pose_history.append(msg)
    else:
      self.pose_history.append(msg)
    if len(self.pose_history) > 1:

      total_turning_time = self.pose_history[-1].header.stamp.secs - self.pose_history[0].header.stamp.secs + \
        (self.pose_history[-1].header.stamp.nsecs - self.pose_history[0].header.stamp.nsecs)*0.000000001
      theta_1 = euler_from_quaternion([self.pose_history[0].odometry.pose.pose.orientation.x,
                                     self.pose_history[0].odometry.pose.pose.orientation.y,
                                     self.pose_history[0].odometry.pose.pose.orientation.z,
                                     self.pose_history[0].odometry.pose.pose.orientation.w])[2]

      theta_2 = euler_from_quaternion([self.pose_history[-1].odometry.pose.pose.orientation.x,
                                     self.pose_history[-1].odometry.pose.pose.orientation.y,
                                     self.pose_history[-1].odometry.pose.pose.orientation.z,
                                     self.pose_history[-1].odometry.pose.pose.orientation.w])[2]
      expected_turning_angle = float(self.v_turn * total_turning_time/float(self.hypothesis_radius))
      total_turning_angle = np.abs(theta_1-theta_2)

      if total_turning_angle >= 0.8 * expected_turning_angle:
        self.twist.linear.x = self.v_turn
        self.curve_flag = 1

      if total_turning_angle <= 0.4 * expected_turning_angle and self.curve_flag==1:
        self.curve_flag = 0
        self.twist.linear.x = self.v1
      # print("total_turning_angle",total_turning_angle,"expected_turning_angle",expected_turning_angle,"v_x",self.twist.linear.x)
    pos_list = []

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

#!/usr/bin/env python
# BEGIN ALL
import functools
import sys
import numpy as np

import rospy
import rosparam
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Int8
from followbot.msg import MGSMeasurement, MGSMeasurements, MGSMarker, MGS_and_Control
import math
from nav_msgs.msg import Odometry
import message_filters
class Identifier:

  def __init__(self):
    # ros
    self.mgs_sub = rospy.Subscriber('mgs', MGSMeasurements, self.mgs_callback)
    self.battery_sub = rospy.Subscriber('battery', BatteryState, self.battery_callback)
    self.multi_send_sub = rospy.Subscriber('multi_send', Int8, self.multi_send_callback)
    self.marker_pub = rospy.Publisher('mgs_marker', MGSMarker, queue_size=10)
    self.temp_mgs_control = MGS_and_Control()
    self.odom_sub = rospy.Subscriber('odom',Odometry,self.odom_callback)
    # self.marker_delay_pub = rospy.Publisher('delay_mgs_marker', MGSMarker, queue_size=10)
    # self.marker_delay_sub = message_filters.Subscriber('mgs_marker',MGSMarker,self.marker_delay_sub)
    # self.odom_sub = message_filters.Subscriber('odom', Odometry)
    # self.measurements = message_filters.ApproximateTimeSynchronizer([self.marker_delay_sub, self.odom_sub], queue_size=5, slop=0.5)
    # self.measurements.registerCallback(self.marker_delay_callback)
    if rospy.has_param('~filename'):
      filename = rospy.get_param('~filename')
      self.marker_types = rosparam.load_file(filename)
      self.marker_types = self.marker_types[0][0]['marker_types']
    else:
      rospy.logerr('Marker parameter file must exist')
      sys.exit(1)
    self.history = []
    self.min_count = rospy.get_param('~min_count', 15)
    self.battery_state = BatteryState()
    self.battery_state.percentage = 1.0
    self.multi_send_val = -1
    self.scale_factor_1 = 0.795 # from true world to simulation
    self.scale_factor_2 = 1.2
    self.marker1_length = self.scale_factor_1*float(12/39.37)
    self.marker2_length = self.scale_factor_2*float(3/39.37)
    self.seperated_space_min_p = 0.7
    self.seperated_space_max_p = 1.6
    self.last_marker_type = 0 # num 1 for 12 inch, num 2 for 3 inch, default is 12 inch
    self.first_marker_label = 1 # check if the current marker is the start of a series, 1 for current is the first marker
    self.current_marker_type = 1 # num 1 for 12 inch, num 2 for 3 inch, default is 12 inch
    self.label_list = [] # record label list, up to 16
    self.label_pose_list = [] # robot position in global frame when detected the markers

    # 1 MPH = 0.447 m/s = 0.447*39.37 inch/s = 17.59839 inch/s


  def odom_callback(self,odom_msg):
    if hasattr(self.temp_mgs_control.measurements,'header'):
      marker_odom = self.temp_mgs_control.measurements.odometry
      # print("odom_time",rospy.get_rostime(),"self.temp_mgs_control.mgsmarker.header",self.temp_mgs_control.measurements.header)
      # print("marker_odom",marker_odom)
      marker_odom_x = marker_odom.pose.pose.position.x
      marker_odom_y = marker_odom.pose.pose.position.y
      current_odom_x = odom_msg.pose.pose.position.x
      current_odom_y = odom_msg.pose.pose.position.y

      dist = math.sqrt((float(marker_odom_x) - float(current_odom_x))**2+(float(marker_odom_y) - float(current_odom_y))**2)

      if not self.first_marker_label:
        last_type = self.temp_mgs_control.markers.type
        rospy.sleep(0.5)
        if self.temp_mgs_control.markers.type == last_type:
          if self.last_marker_type == 1:
            if dist >= 2.4 * self.marker1_length:
              self.marker_pub.publish(self.temp_mgs_control.markers)
              self.last_marker_type = 0
              rospy.loginfo('Found marker: {},70'.format(self.temp_mgs_control.markers.type))
              print("self.first_marker_label",self.first_marker_label,"self.last_marker_type",self.last_marker_type)
              self.temp_mgs_control = MGS_and_Control()
              # print("self.temp_mgs_control.markers_1",self.temp_mgs_control.markers)
          elif self.last_marker_type == 2:
            if dist >= 2.4 * self.marker2_length:
              self.marker_pub.publish(self.temp_mgs_control.markers)
              self.last_marker_type = 0
              rospy.loginfo('Found marker: {},76'.format(self.temp_mgs_control.markers.type))
              print("self.first_marker_label",self.first_marker_label,"self.last_marker_type",self.last_marker_type)
              self.temp_mgs_control = MGS_and_Control()
              # print("self.temp_mgs_control.markers_2",self.temp_mgs_control.markers)
        #   else:
        #     rospy.sleep(0.1)
        else:
          rospy.sleep(0.5)
      else:
        rospy.sleep(1)
        if self.first_marker_label:
          if self.last_marker_type == 1:
            self.marker_pub.publish(self.temp_mgs_control.markers)
            self.last_marker_type = 0
            rospy.loginfo('Found marker: {},90'.format(self.temp_mgs_control.markers.type))
            print("self.first_marker_label",self.first_marker_label,"self.last_marker_type",self.last_marker_type)
            self.temp_mgs_control = MGS_and_Control()
          else:
            print("first_marker_label 1, last_marker_type = 0")
        else:
          print("first_marker_label 0")


  def mgs_callback(self, msg):
    # TODO can add in delay to avoid compute marker early if missed detections in one frame
    if any([m.type == MGSMeasurement.MARKER for m in msg.measurements]):
      # add marker to history if detected
      # print("mgs_time",rospy.get_rostime())
      self.history.append(msg)
    else:
      # compute marker type if no marker present and have a history
      if len(self.history) > 0:
        self._compute_marker_type()

  

  def battery_callback(self, msg):
    self.battery_state = msg
  

  def multi_send_callback(self, msg):
    self.multi_send_val = msg.data
    

  # def marker_delay_callback(self,mgs_msg,odom_msg):
  #   last_mgs_time = mgs_msg.header
  #
  #   now = rospy.get_rostime()


  def _compute_marker_type(self):
    # convert to numpy array
    layout = np.zeros((len(self.history)+2, 2), dtype=int) # pad with initial and final row of zeros
    layout_pose = np.zeros((len(self.history)+2, 2), dtype=float)

    for i, m in enumerate(self.history):
      layout[i+1,:] = self._get_layout(m)
      layout_pose[i+1,:] = self._get_layout_pose(m)
    # only keep rows repeated at least min_count times
    diff = np.sum(np.abs(np.diff(layout,axis=0)), axis=1) # compute differences between rows
    index_diff = np.argwhere(diff) # indices where there are differences
    marker_layout = [] # final layout
    pose_marker_layout = []

    for j in range(len(index_diff)-1):
      if index_diff[j+1] - index_diff[j] > self.min_count:
        marker_layout.extend( layout[index_diff[j]+1,:].flatten().tolist())
        pose_marker_layout.extend( layout_pose[index_diff[j]+1].flatten().tolist())
        pose_marker_layout.extend( layout_pose[index_diff[j+1]].flatten().tolist())

    if marker_layout:
      if len(self.label_list) < 10:
        self.label_list.append(marker_layout)
        self.label_pose_list.append(pose_marker_layout)
      else:
        temp = self.label_list[len(self.label_list) - 9:len(self.label_list)]
        self.label_list = temp
        self.label_list.append(marker_layout)
        temp_last_label_pose_list = self.label_pose_list[len(self.label_pose_list) - 9:len(self.label_pose_list)]
        self.label_pose_list = temp_last_label_pose_list
        self.label_pose_list.append(pose_marker_layout)

    self.pattern_recognition(self.label_list,self.label_pose_list)
    # print("self.label_list,97",self.label_list)
    # for ii in self.label_list:
    #   marker_layout.extend(ii)



    # TODO can recurse on this simplification to account for spurious misses within a strip
    #   -need to merge two sections after filtering out abberant readings within
    # check if marker matches any in library
    found = False
    marker_layout = []

    if self.label_list:
      if len(self.label_list) == 1:
        marker_layout = self.label_list[0]
      else:
        for i,m in enumerate(self.label_list):
          if i < len(self.label_list)-1:
            marker_layout.extend(m)
            marker_layout.extend([0,0])
          else:
            marker_layout.extend(m)
    # print("marker_layout,118",marker_layout)
    for mt in self.marker_types:
      if self._check_list_equality(mt['layout'], marker_layout):
        # print("marker_layout",marker_layout)
        # print("self._check_list_equality(mt['layout'], marker_layout)",self._check_list_equality(mt['layout'], marker_layout))
        found = True
        # rospy.loginfo('Found marker: {}'.format(mt['type']))
        c = MGSMarker()
        # Basic Markers
        if self.last_marker_type == 1:
          if mt['type'] == 'bear_left':
            c.type = MGSMarker.BEAR_LEFT
            if not self.multi_send_val == -1:
              c.command = MGSMarker.BEAR_LEFT
            else:
              c.command = MGSMarker.BEAR_RIGHT
          elif mt['type'] == 'bear_right':
            c.type = MGSMarker.BEAR_RIGHT
            if not self.multi_send_val == -1:
              c.command = MGSMarker.BEAR_RIGHT
            else:
              c.command = MGSMarker.BEAR_LEFT
          elif mt['type'] == 'stop':
            c.type = MGSMarker.STOP
            c.command = MGSMarker.STOP
          else:
            rospy.logwarn('Type not defined,200')
        elif self.last_marker_type == 2:
        # Speed Hyper Markers
          if mt['type'] == 'speed_1':
            c.type = MGSMarker.SPEED_1
            c.command = MGSMarker.SPEED_1
          elif mt['type'] == 'speed_2':
            c.type = MGSMarker.SPEED_2
            c.command = MGSMarker.SPEED_2
          elif mt['type'] == 'speed_3':
            c.type = MGSMarker.SPEED_3
            c.command = MGSMarker.SPEED_3
          elif mt['type'] == 'speed_4':
            c.type = MGSMarker.SPEED_4
            c.command = MGSMarker.SPEED_4
          # Battery Charge Hyper Markers
          elif mt['type'] == 'battery_charge_left':
            c.type = MGSMarker.BATTERY_CHARGE_LEFT
            if self.battery_state.percentage > 0.3:
              c.command = MGSMarker.BEAR_RIGHT
            else:
              c.command = MGSMarker.BEAR_LEFT
          elif mt['type'] == 'battery_charge_right':
            c.type = MGSMarker.BATTERY_CHARGE_RIGHT
            if self.battery_state.percentage > 0.3:
              c.command = MGSMarker.BEAR_LEFT
            else:
              c.command = MGSMarker.BEAR_RIGHT
          # Multi Send Markers
          elif mt['type'] == 'multi_send_1_left':
            c.type = MGSMarker.MULTI_SEND_1_LEFT
            if self.multi_send_val == 1:
              c.command = MGSMarker.BEAR_LEFT
            else:
              c.command = MGSMarker.BEAR_RIGHT
          elif mt['type'] == 'multi_send_2_left':
            c.type = MGSMarker.MULTI_SEND_2_LEFT
            if self.multi_send_val == 2:
              c.command = MGSMarker.BEAR_LEFT
            else:
              c.command = MGSMarker.BEAR_RIGHT
          elif mt['type'] == 'multi_send_3_left':
            c.type = MGSMarker.MULTI_SEND_3_LEFT
            if self.multi_send_val == 3:
              c.command = MGSMarker.BEAR_LEFT
            else:
              c.command = MGSMarker.BEAR_RIGHT
          elif mt['type'] == 'multi_send_4_left':
            c.type = MGSMarker.MULTI_SEND_4_LEFT
            if self.multi_send_val == 4:
              c.command = MGSMarker.BEAR_LEFT
            else:
              c.command = MGSMarker.BEAR_RIGHT
          elif mt['type'] == 'multi_send_5_left':
            c.type = MGSMarker.MULTI_SEND_5_LEFT
            if self.multi_send_val == 5:
              c.command = MGSMarker.BEAR_LEFT
            else:
              c.command = MGSMarker.BEAR_RIGHT
          elif mt['type'] == 'multi_send_6_left':
            c.type = MGSMarker.MULTI_SEND_6_LEFT
            if self.multi_send_val == 6:
              c.command = MGSMarker.BEAR_LEFT
            else:
              c.command = MGSMarker.BEAR_RIGHT
          elif mt['type'] == 'multi_send_7_left':
            c.type = MGSMarker.MULTI_SEND_7_LEFT
            if self.multi_send_val == 7:
              c.command = MGSMarker.BEAR_LEFT
            else:
              c.command = MGSMarker.BEAR_RIGHT
          elif mt['type'] == 'multi_send_8_left':
            c.type = MGSMarker.MULTI_SEND_8_LEFT
            if self.multi_send_val == 8:
              c.command = MGSMarker.BEAR_LEFT
            else:
              c.command = MGSMarker.BEAR_RIGHT
          elif mt['type'] == 'multi_send_1_right':
            c.type = MGSMarker.MULTI_SEND_1_RIGHT
            if self.multi_send_val == 1:
              c.command = MGSMarker.BEAR_RIGHT
            else:
              c.command = MGSMarker.BEAR_LEFT
          elif mt['type'] == 'multi_send_2_right':
            c.type = MGSMarker.MULTI_SEND_2_RIGHT
            if self.multi_send_val == 2:
              c.command = MGSMarker.BEAR_RIGHT
            else:
              c.command = MGSMarker.BEAR_LEFT
          elif mt['type'] == 'multi_send_3_right':
            c.type = MGSMarker.MULTI_SEND_3_RIGHT
            if self.multi_send_val == 3:
              c.command = MGSMarker.BEAR_RIGHT
            else:
              c.command = MGSMarker.BEAR_LEFT
          elif mt['type'] == 'multi_send_4_right':
            c.type = MGSMarker.MULTI_SEND_4_RIGHT
            if self.multi_send_val == 4:
              c.command = MGSMarker.BEAR_RIGHT
            else:
              c.command = MGSMarker.BEAR_LEFT
          elif mt['type'] == 'multi_send_5_right':
            c.type = MGSMarker.MULTI_SEND_5_RIGHT
            if self.multi_send_val == 5:
              c.command = MGSMarker.BEAR_RIGHT
            else:
              c.command = MGSMarker.BEAR_LEFT
          elif mt['type'] == 'multi_send_6_right':
            c.type = MGSMarker.MULTI_SEND_6_RIGHT
            if self.multi_send_val == 6:
              c.command = MGSMarker.BEAR_RIGHT
            else:
              c.command = MGSMarker.BEAR_LEFT
          elif mt['type'] == 'multi_send_7_right':
            c.type = MGSMarker.MULTI_SEND_7_RIGHT
            if self.multi_send_val == 7:
              c.command = MGSMarker.BEAR_RIGHT
            else:
              c.command = MGSMarker.BEAR_LEFT
          elif mt['type'] == 'multi_send_8_right':
            c.type = MGSMarker.MULTI_SEND_8_RIGHT
            if self.multi_send_val == 8:
              c.command = MGSMarker.BEAR_RIGHT
            else:
              c.command = MGSMarker.BEAR_LEFT
          else:
            rospy.logwarn('Type not defined,326')
        # self.marker_pub.publish(c)
        self.temp_mgs_control.markers = c
        self.temp_mgs_control.measurements = self.history[-1]
        break
    if marker_layout and not found:
      rospy.logwarn('Marker layout not defined: {}'.format(marker_layout))
    # reset history
    self.history = []


  def _get_layout(self, msg):
    # note: right is negative
    # get positions
    track = None # track position
    markers = [] # marker position(s)
    for m in msg.measurements:
      if m.type == MGSMeasurement.TRACK:
        track = m.position
      else:
        markers.append(m.position)

    # get layout
    layout = [0, 0]
    for m in markers:
      if m < track:
        layout[1] = 1
      else:
        layout[0] = 1
    
    return layout


  def _get_layout_pose(self, msg):
    layout_position = [msg.odometry.pose.pose.position.x,msg.odometry.pose.pose.position.y]
    # print("layout_position",layout_position)
    return layout_position


  def _check_list_equality(self, list1, list2):
    # https://www.geeksforgeeks.org/python-check-if-two-lists-are-identical/
    return functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, list1, list2), True)

  def pattern_recognition(self, label_list, label_pose_list):
    # print("label_pose_list",label_pose_list)
    dist_list = np.zeros(len(label_pose_list))
    first_points_list = []
    second_points_list = []
    center_points_list = []
    seg_length_list = []
    temp_label_list = []
    temp_label_pose_list = []
    for i, m in enumerate(label_pose_list):
      # print("i,m",i,m)
      temp_edge_points = m
      temp_edge_point_1 = np.array(temp_edge_points[0:2])
      temp_edge_point_2 = np.array(temp_edge_points[2:4])
      first_points_list.append(temp_edge_point_1)
      second_points_list.append(temp_edge_point_2)
      center_point = (temp_edge_point_1 + temp_edge_point_2)/2.0
      center_points_list.append(center_point)
      temp_seg_length = math.sqrt((float(temp_edge_points[0]) - float(temp_edge_points[2]))**2+\
                           (float(temp_edge_points[1]) - float(temp_edge_points[3]))**2)
      seg_length_list.append(temp_seg_length)
      if i>=1:
        dist_list[i] = math.sqrt((float(center_points_list[i][0]) - float(center_points_list[i-1][0]))**2+\
                                   (float(center_points_list[i][1]) - float(center_points_list[i-1][1]))**2)
    if len(label_list) > 1:
      current_seg_length = seg_length_list[-1]
      different_size_list = []
      for i, m in enumerate(seg_length_list):
        if m < 0.7 * current_seg_length or m >= 1.4 * current_seg_length:
          different_size_list.append(i)
      # print("current_seg_length",current_seg_length,"seg_length_list",seg_length_list)
      # print("different_size_list",different_size_list)
      if different_size_list:
        valid_index = range(np.max(different_size_list)+1,len(seg_length_list))
      else:
        valid_index = range(0,len(seg_length_list))


      # print("valid_index",valid_index)
      # print("self.label_list_1",self.label_list)
      # print("self.label_list[valid_index[0]]",self.label_list[int(valid_index[0])])
      if len(valid_index) == 1:
        self.first_marker_label = 1
        print("421,self.first_marker_label",self.first_marker_label)
      else:
        self.first_marker_label = 0
        print("424,self.first_marker_label",self.first_marker_label)
      for i,m in enumerate(valid_index):
        temp_label_list.append(self.label_list[int(m)])
        temp_label_pose_list.append(self.label_pose_list[int(m)])
      self.label_list = temp_label_list
      self.label_pose_list = temp_label_pose_list
      # print("self.label_list_2",self.label_list)
      # print("self.label_pose_list",self.label_pose_list)
      if np.abs(current_seg_length-self.marker2_length) <= np.abs(current_seg_length-self.marker1_length):
        marker_length = self.marker2_length
        self.last_marker_type = 2
      else:
        marker_length = self.marker1_length
        self.last_marker_type = 1
      # print("marker_length",marker_length)
      temp_dist_list = []
      for i,m in enumerate(valid_index):
        if i!=0:
          temp_dist_list.append(dist_list[int(m)])
      # print("dist_list",dist_list,"temp_dist_list",temp_dist_list)
      dist_list_1 = np.array(temp_dist_list) - np.array(marker_length)*2*(np.array(self.seperated_space_min_p))
      dist_list_2 = np.array(temp_dist_list) - np.array(marker_length)*2*(np.array(self.seperated_space_max_p))

      # print("dist_list_1",dist_list_1,"dist_list_2",dist_list_2)
      p1 = np.argwhere(dist_list_1 > 0)
      p2 = np.argwhere(dist_list_2 < 0)
      final_p = np.intersect1d(p1,p2)
      consecutive_index = np.zeros(len(valid_index))
      consecutive_index[final_p] = 1
      # print("consecutive_index",consecutive_index)
      ls = [i for i, e in enumerate(consecutive_index) if e != 0] # find non-zero term
      ls2 = [i for i, e in enumerate(consecutive_index) if e == 0] #
      if ls2:
        ls2.pop(0)

      if ls2:
        if max(ls2) == len(label_pose_list)-1:
          self.label_list = [self.label_list[-1]]
          self.label_pose_list = [self.label_pose_list[-1]]
        else:
          self.label_list = [self.label_list[ls > max(ls2)]]
          self.label_pose_list = [self.label_pose_list[ls > max(ls2)]]
      else:
        self.label_list = self.label_list
        self.label_pose_list = self.label_pose_list
    else:
      self.first_marker_label = 1
      print("471,self.first_marker_label",self.first_marker_label)

if __name__ == '__main__':
  rospy.init_node('marker_identifier')
  identifier = Identifier()
  rospy.spin()
# END ALL

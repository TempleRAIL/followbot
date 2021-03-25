#!/usr/bin/env python
# BEGIN ALL
import functools
import sys
import numpy as np

import rospy
import rosparam
from std_msgs.msg import String
from followbot.msg import MGSMeasurement, MGSMeasurements, MGSMarker


class Identifier:

  def __init__(self):
    # ros
    self.mgs_sub = rospy.Subscriber('mgs', MGSMeasurements, self.mgs_callback)
    self.marker_pub = rospy.Publisher('mgs_marker', MGSMarker, queue_size=10)
    if rospy.has_param('~filename'):
      filename = rospy.get_param('~filename')
      self.marker_types = rosparam.load_file(filename)
      self.marker_types = self.marker_types[0][0]['marker_types']
    else:
      rospy.logerr('Marker parameter file must exist')
      sys.exit(1)
    self.history = []
    self.min_count = rospy.get_param('~min_count', 50)


  def mgs_callback(self, msg):
    # TODO can add in delay to avoid compute marker early if missed detections in one frame
    if any([m.type == MGSMeasurement.MARKER for m in msg.measurements]):
      # add marker to history if detected
      self.history.append(msg)
    else:
      # compute marker type if no marker present and have a history
      if len(self.history) > 0:
        self._compute_marker_type()
    

  def _compute_marker_type(self):
    # convert to numpy array
    layout = np.zeros((len(self.history)+2, 2), dtype=int) # pad with initial and final row of zeros
    for i, m in enumerate(self.history):
      layout[i+1,:] = self._get_layout(m)
    # only keep rows repeated at least min_count times
    diff = np.sum(np.abs(np.diff(layout,axis=0)), axis=1) # compute differences between rows
    index_diff = np.argwhere(diff) # indices where there are differences
    marker_layout = [] # final layout
    for j in range(len(index_diff)-1):
      if index_diff[j+1] - index_diff[j] > self.min_count:
        marker_layout.extend( layout[index_diff[j]+1,:].flatten().tolist() )
    # TODO can recurse on this simplification to account for spurious misses within a strip
    #   -need to merge two sections after filtering out abberant readings within
    # check if marker matches any in library
    found = False
    for mt in self.marker_types:
      if self._check_list_equality(mt['layout'], marker_layout):
        found = True
        rospy.loginfo('Found marker: {}'.format(mt['command']))
        c = MGSCommand()
        # Basic Markers
        if mt['command'] == 'bear_left':
          c.command = MGSCommand.BEAR_LEFT
        elif mt['command'] == 'bear_right':
          c.command = MGSCommand.BEAR_RIGHT
        elif mt['command'] == 'stop':
          c.command = MGSCommand.STOP
        # Speed Hyper Markers
        elif mt['command'] == 'speed_1':
          c.command = MGSCommand.SPEED_1
        elif mt['command'] == 'speed_2':
          c.command = MGSCommand.speed_2
        elif mt['command'] == 'speed_3':
          c.command = MGSCommand.speed_3
        elif mt['command'] == 'speed_4':
          c.command = MGSCommand.speed_4
        # Battery Charge Hyper Markers
        elif mt['command'] == 'battery_charge_left':
          c.command = MGSCommand.BATTERY_CHARGE_LEFT
        elif mt['command'] == 'battery_charge_right':
          c.command = MGSCommand.BATTERY_CHARGE_RIGHT
        # Multi Send Markers
        elif mt['command'] == 'multi_send_1_left':
          c.command = MGSCommand.MULTI_SEND_1_LEFT
        elif mt['command'] == 'multi_send_2_left':
          c.command = MGSCommand.MULTI_SEND_2_LEFT
        elif mt['command'] == 'multi_send_3_left':
          c.command = MGSCommand.MULTI_SEND_3_LEFT
        elif mt['command'] == 'multi_send_4_left':
          c.command = MGSCommand.MULTI_SEND_4_LEFT
        elif mt['command'] == 'multi_send_5_left':
          c.command = MGSCommand.MULTI_SEND_5_LEFT
        elif mt['command'] == 'multi_send_6_left':
          c.command = MGSCommand.MULTI_SEND_6_LEFT
        elif mt['command'] == 'multi_send_7_left':
          c.command = MGSCommand.MULTI_SEND_7_LEFT
        elif mt['command'] == 'multi_send_8_left':
          c.command = MGSCommand.MULTI_SEND_8_LEFT
        elif mt['command'] == 'multi_send_1_right':
          c.command = MGSCommand.MULTI_SEND_1_RIGHT
        elif mt['command'] == 'multi_send_2_right':
          c.command = MGSCommand.MULTI_SEND_2_RIGHT
        elif mt['command'] == 'multi_send_3_right':
          c.command = MGSCommand.MULTI_SEND_3_RIGHT
        elif mt['command'] == 'multi_send_4_right':
          c.command = MGSCommand.MULTI_SEND_4_RIGHT
        elif mt['command'] == 'multi_send_5_right':
          c.command = MGSCommand.MULTI_SEND_5_RIGHT
        elif mt['command'] == 'multi_send_6_right':
          c.command = MGSCommand.MULTI_SEND_6_RIGHT
        elif mt['command'] == 'multi_send_7_right':
          c.command = MGSCommand.MULTI_SEND_7_RIGHT
        elif mt['command'] == 'multi_send_8_right':
          c.command = MGSCommand.MULTI_SEND_8_RIGHT
        else:
          rospy.logwarn('Type not defined')
        self.marker_pub.publish(c)
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
  

  def _check_list_equality(self, list1, list2):
    # https://www.geeksforgeeks.org/python-check-if-two-lists-are-identical/
    return functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, list1, list2), True)


if __name__ == '__main__':
  rospy.init_node('marker_identifier')
  identifier = Identifier()
  rospy.spin()
# END ALL

#!/usr/bin/env python
# BEGIN ALL
import rospy
import rosparam
import sys
import functools
from nav_msgs.msg import Odometry
from followbot.msg import MGSMeasurement, MGSMeasurements


class Identifier:

  def __init__(self):
    # ros
    self.mgs_sub = rospy.Subscriber('mgs', MGSMeasurements, self.mgs_callback)
    # self.marker_pub = rospy.Publisher('mgs_marker', MGSMarker, queue_size=10)
    if rospy.has_param('~filename'):
      filename = rospy.get_param('~filename')
      self.marker_types = rosparam.load_file(filename)
      self.marker_types = self.marker_types[0][0]['marker_types']
    else:
      rospy.logerr('Marker parameter file must exist')
      sys.exit(1)
    self.history = []


  def mgs_callback(self, msg):
    if any([m.type == MGSMeasurement.MARKER for m in msg.measurements]):
      # add marker to history if detected
      self.history.append(msg)
    else:
      # compute marker type if no marker present and have a history
      if len(self.history) > 0:
        self._compute_marker_type()
    

  def _compute_marker_type(self):
    rospy.loginfo('Identifying marker type')
    marker_layout = []
    prev_layout = None
    for m in self.history:
      layout = self._get_layout(m)
      if prev_layout is None:
        # add first marker
        marker_layout.extend(layout)
      else:
        # check if layout changed
        if not self._check_list_equality(prev_layout, layout):
          marker_layout.extend(layout)
      prev_layout = layout
    print(marker_layout)
    # TODO  fix issue where slight offsets causing wrong marker types
    #       perhaps check how many instances of a layout there are to ignore very short ones and/or skip small gaps

    found = False
    for mt in self.marker_types:
      if self._check_list_equality(mt['layout'], marker_layout):
        print(mt['command'])
        found = True
        break
    if not found:
      print('marker type not defined')

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

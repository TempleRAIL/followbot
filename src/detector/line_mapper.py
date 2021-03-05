#!/usr/bin/env python
# BEGIN ALL
import copy
# import matplotlib.pyplot as plt
import numpy as np
import threading

from scipy.interpolate import CubicSpline
from tf.transformations import euler_from_quaternion

import rospy
import tf2_ros
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import Point
from std_msgs.msg import String, MultiArrayDimension
from followbot.msg import MGSMeasurement, MGSMeasurements, MGSMarker, PathSegment



class LineMapper:

  def __init__(self):
    # ros
    self.tfBuffer = tf2_ros.Buffer(cache_time=rospy.Duration(60.))
    self.listener = tf2_ros.TransformListener(self.tfBuffer)
    self.mgs_sub = rospy.Subscriber('mgs', MGSMeasurements, self.mgs_callback)
    self.marker_sub = rospy.Subscriber('mgs_marker', MGSMarker, self.marker_callback)
    self.segment_pub = rospy.Publisher('path_segments', PathSegment, queue_size=10)
    # parameters
    self.step_size = rospy.get_param('~step_size', 0.05)
    # data about path
    self.current_segment = []
    self.current_segment_lock = threading.Lock()
    self.marker_prev = MGSMarker()
    self.marker_prev.type = MGSMarker.NONE
    # transformation to most recent marker location
    self.R = np.eye(2)
    self.t = np.zeros((2,1))
    # get offset from robot to camera frame
    success = False
    while not success:
      try:
        trans = self.tfBuffer.lookup_transform('base_footprint', 'camera_depth_optical_frame', rospy.Time())
      except:
        pass
      else:
        self.t0 = np.array([[trans.transform.translation.x], 
                            [trans.transform.translation.y]])
        success = True


  def mgs_callback(self, msg):
    if any([m.type == MGSMeasurement.TRACK for m in msg.measurements]):
      # add msg to current segment if track detected
      self.current_segment_lock.acquire()
      try:
        self.current_segment.append(msg)
      finally:
        self.current_segment_lock.release()


  def marker_callback(self, msg):
    # Fit path using cubic splines, parameterized by distance traveled
    # copy points from current segment into local variable
    self.current_segment_lock.acquire()
    try:
      # Put current segment point into odom frame
      pts = self._msgs_to_odom_frame(self.current_segment)
      # Re-zero data at current location
      trans = self.tfBuffer.lookup_transform('odom', 'base_footprint', rospy.Time())
      theta = euler_from_quaternion([trans.transform.rotation.x, 
                                     trans.transform.rotation.y, 
                                     trans.transform.rotation.z, 
                                     trans.transform.rotation.w])[2]
      self.R = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
      self.t = np.array([[trans.transform.translation.x], 
                         [trans.transform.translation.y]])
      # Reset current segment
      self.current_segment = []
    finally:
      self.current_segment_lock.release()
    
    # Calculate cumulative distance traveled
    # TODO subsample (e.g. change in distance of at least a certain amount)
    s = np.cumsum(np.hstack(([0], np.sqrt(np.sum(np.diff(pts,axis=1)**2, axis=0)))))
    # Remove any points that do not make progress
    idx = np.argwhere(np.diff(s)==0.)
    s = np.delete(s, idx)
    pts = np.delete(pts, idx, axis=1)
    spl = CubicSpline(s, pts, axis=1)

    # Interpolate by a fixed distance to limit size of message
    n_steps = np.ceil(s[-1] / self.step_size)
    s = np.linspace(0, s[-1], n_steps)
    pts = spl(s)
    print(s)
    print(pts)

    # Publish segment data
    p = PathSegment()
    p.marker_start = self.marker_prev
    p.marker_end = msg
    p.breakpoints = spl.x
    p.coefficients.layout.dim.append(MultiArrayDimension("dimension",  spl.c.shape[0], pts.size))
    p.coefficients.layout.dim.append(MultiArrayDimension("point",      spl.c.shape[1], spl.c.shape[1]*spl.c.shape[2]))
    p.coefficients.layout.dim.append(MultiArrayDimension("coordinate", spl.c.shape[2], spl.c.shape[2]))
    p.coefficients.data = spl.c.flatten()
    self.segment_pub.publish(p)

    self.marker_prev = p.marker_end
    
    # plt.plot(pts[0,:], pts[1,:], 'ro', ms=5)
    # plt.plot(spl(s)[0,:], spl(s)[1,:], 'b', lw=3)
    # # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()


  def _msgs_to_odom_frame(self, msgs):
    # put all points into odom frame
    pts = np.zeros((2, len(msgs)))
    for i, m in enumerate(msgs):
      for meas in m.measurements:
        if meas.type == MGSMeasurement.TRACK:
          pt = Point(0., meas.position, 0.)
          break
      new_pt = self.tfBuffer.transform(PointStamped(m.header, pt), 'odom')
      pts[:,i] = [new_pt.point.x, new_pt.point.y]
    return np.matmul(self.R.T, pts - self.t) - self.t0


if __name__ == '__main__':
  rospy.init_node('line_mapper')
  mapper = LineMapper()
  rospy.spin()
# END ALL

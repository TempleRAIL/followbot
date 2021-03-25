#!/usr/bin/env python
# BEGIN ALL
import rospy
from followbot.msg import MGSMarker


class AutoRestart:

  def __init__(self):
    # ros
    self.marker_sub = rospy.Subscriber('mgs_marker', MGSMarker, self.callback)
    self.marker_pub = rospy.Publisher('mgs_marker', MGSMarker, queue_size=10)
    self.delay = rospy.get_param('~delay', 2.0)
    self.start_timer()


  def callback(self, msg):
    if msg.command == MGSMarker.STOP:
      self.start_timer()


  def timer_callback(self, event):
    rospy.loginfo('Starting robot')
    c = MGSMarker()
    c.command = MGSMarker.START
    self.marker_pub.publish(c)
  

  def start_timer(self):
    self.timer = rospy.Timer(rospy.Duration(self.delay), self.timer_callback, oneshot=True)


if __name__ == '__main__':
  rospy.init_node('auto_restart')
  ar = AutoRestart()
  rospy.spin()
# END ALL

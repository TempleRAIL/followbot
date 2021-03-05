#!/usr/bin/env python
# BEGIN ALL
import rospy
from followbot.msg import MGSCommand


class AutoRestart:

  def __init__(self):
    # ros
    self.command_sub = rospy.Subscriber('mgs_command', MGSCommand, self.callback)
    self.command_pub = rospy.Publisher('mgs_command', MGSCommand, queue_size=10)
    self.delay = rospy.get_param('~delay', 2.0)
    self.start_timer()


  def callback(self, msg):
    if msg.command == MGSCommand.STOP:
      self.start_timer()


  def timer_callback(self, event):
    rospy.loginfo('Starting robot')
    c = MGSCommand()
    c.command = MGSCommand.START
    self.command_pub.publish(c)
  

  def start_timer(self):
    self.timer = rospy.Timer(rospy.Duration(self.delay), self.timer_callback, oneshot=True)


if __name__ == '__main__':
  rospy.init_node('auto_restart')
  ar = AutoRestart()
  rospy.spin()
# END ALL

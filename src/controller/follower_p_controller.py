#!/usr/bin/env python
# BEGIN ALL
import threading
import numpy as np
import rospy
from geometry_msgs.msg import Twist,Pose
from followbot.msg import MGSMeasurements, MGSMeasurement, MGSMarker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import PointCloud
from nav_msgs.msg import Path
import tf2_ros
from scipy import stats
import std_msgs.msg
import math
import tf

class Follower:
    def __init__(self):
        self.measurement_sub = rospy.Subscriber('mgs', MGSMeasurements, self.mgs_callback)
        self.mgs_marker_sub = rospy.Subscriber('mgs_marker', MGSMarker, self.marker_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
        self.closest_goal_sub = rospy.Subscriber('camera_detection',PointCloud,self.goal_callback)
        self.twist = Twist()
        self.twist_lock = threading.Lock()
        self.tfBuffer = tf2_ros.Buffer(cache_time=rospy.Duration(60.))
        self.scale_factor = 0.795 # from true world to simulation
        self.p = self.scale_factor*rospy.get_param('~p', 7.0) # proportional controller constant
        self.v1 = self.scale_factor*rospy.get_param('~v', 0.666) # nominal velocity (1.49 MPH)
        self.v2 = self.scale_factor*rospy.get_param('~v', 0.782) # nominal velocity (1.75 MPH)
        self.v3 = self.scale_factor*rospy.get_param('~v', 0.849) # nominal velocity (1.90 MPH)
        self.v4 = self.scale_factor*rospy.get_param('~v', 0.939) # nominal velocity (2.10 MPH)
        self.v_turn = self.scale_factor*rospy.get_param('~v', 0.425) # nominal velocity (0.95 MPH)
        self.v_laser = self.scale_factor*rospy.get_param('~v', 0.308) # nominal velocity (0.69 MPH)
        self.v_stop = self.scale_factor*rospy.get_param('~v', 0.125) # nominal velocity (0.28 MPH)
        self.goal_margin = rospy.get_param('~goal_margin', 3.0)
        self.pose_history = []
        self.curve_flag = 0 # 0 for no curve, 1 for curve
        self.hypothesis_radius = self.scale_factor * 1
        self.wheel_radius = self.scale_factor * 0.127 # 5 inches how to check???
        self.wheel_base = self.scale_factor * 1 # width is 40 inches
        self.max_wheel_w = float(self.v4/self.wheel_radius)
        self.max_turning_omega = 1.5 * self.v_turn/self.hypothesis_radius
        self.curve_fitting_distance_threshold = self.scale_factor * 0.5 # suppose the turning radius = 1m
        self.upper_turning_threshold = 0.6 * self.v_turn/self.hypothesis_radius
        self.recover_turning_omega = 0.3 * self.v_turn/self.hypothesis_radius
        self.command = -1
        self.path = Path()
        self.last_command_header = std_msgs.msg.Header()
        # self.turning_distance_threshold = 0.3
        # self.temp_turning_position = []
        try:
            trans2 = self.tfBuffer.lookup_transform('base_link', 'camera_rgb_frame',rospy.Time(0))
            print("trans2")
        except:
            print("nothing")
        try:
            trans = self.tfBuffer.lookup_transform('base_link', 'mono_camera_link',rospy.Time(0))
            theta = euler_from_quaternion([trans.transform.rotation.x,
                                       trans.transform.rotation.y,
                                       trans.transform.rotation.z,
                                       trans.transform.rotation.w])[2]
            self.R = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
            self.t = np.array([[trans.transform.translation.x],
                           [trans.transform.translation.y]])
        except:
            pass
        else:
            self.R = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
            self.t = np.array([[trans.transform.translation.x],
                           [trans.transform.translation.y]])

    def marker_callback(self, msg):
        self.twist_lock.acquire()
        try:
          if msg.command == MGSMarker.STOP:
            # self.twist.linear.x = 0.
            self.command = 0
          elif msg.command == MGSMarker.START:
            # self.twist.linear.x = self.v1
            self.command = 1
          elif msg.command == MGSMarker.SPEED_1:
            # self.twist.linear.x = self.v1
            self.command = 1
          elif msg.command == MGSMarker.SPEED_2:
            # self.twist.linear.x = self.v2
            self.command = 2
          elif msg.command == MGSMarker.SPEED_3:
            # self.twist.linear.x = self.v3
            self.command = 3
          elif msg.command == MGSMarker.SPEED_4:
            # self.twist.linear.x = self.v4
            self.command = 4
        finally:
            self.twist_lock.release()


    def mgs_callback(self, msg):
        # get tape position
        pos = None
        try:
            trans = self.tfBuffer.lookup_transform('base_link', 'mono_camera_link',rospy.Time(0))


            theta = euler_from_quaternion([trans.transform.rotation.x,
                                       trans.transform.rotation.y,
                                       trans.transform.rotation.z,
                                       trans.transform.rotation.w])[2]
            self.R = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
            self.t = np.array([[trans.transform.translation.x],
                           [trans.transform.translation.y]])
        except:
            pass
        else:
            self.R = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
            self.t = np.array([[trans.transform.translation.x],
                           [trans.transform.translation.y]])
        if msg.GAP_FLAG ==0:
        # pos_list = []
            if len(self.pose_history) >= 15:
              self.pose_history.pop(0)
              self.pose_history.append(msg)
            else:
              self.pose_history.append(msg)

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
                    self.twist.angular.z = self.p * pos
                    if abs(self.twist.angular.z) >= self.max_turning_omega:
                      if self.twist.angular.z > 0:
                        self.twist.angular.z = self.max_turning_omega
                        self.twist.linear.x = self.v_turn
                      else:
                        self.twist.angular.z = -self.max_turning_omega
                        self.twist.linear.x = self.v_turn
                      # print("case 1","self.twist.angular.z",self.twist.angular.z,"self.twist.linear.x",self.twist.linear.x)
                    elif abs(self.twist.angular.z) >= self.upper_turning_threshold and abs(self.twist.angular.z) < self.max_turning_omega:
                      self.twist.angular.z = self.p * pos
                      self.twist.linear.x = self.v_turn
                      # print("case 2","self.twist.angular.z",self.twist.angular.z,"self.twist.linear.x",self.twist.linear.x)
                    else:
                      if abs(self.twist.angular.z) < self.recover_turning_omega:
                          if self.command == 0:
                            self.twist.linear.x = 0
                          elif self.command == 1:
                            self.twist.linear.x = self.v1
                          elif self.command == 2:
                            self.twist.linear.x = self.v2
                          elif self.command == 3:
                            self.twist.linear.x = self.v3
                          elif self.command == 4:
                            self.twist.linear.x = self.v4
                          else:
                            self.twist.linear.x = self.v1
                            self.twist.angular.z = self.p * pos
                      else:
                          if self.command == 0:
                            self.twist.linear.x = 0
                          else:
                            self.twist.linear.x = self.v_turn
                            self.twist.angular.z = self.p * pos
                      # print("case 3","self.twist.angular.z",self.twist.angular.z,"self.twist.linear.x",self.twist.linear.x)
                    # print("self.twist.linear.x",self.twist.linear.x,"self.twist.angular.z",self.twist.angular.z)
                self.cmd_vel_pub.publish(self.twist)
            finally:
                self.twist_lock.release()

        else:
            # have a double check with the current magnetic detection
            if self.path.header.stamp.to_sec():
                print("self.path.header.stamp.to_sec()",self.path.header.stamp.to_sec())
                [cmd_v, cmd_w] = self.calculate_velocity()
                self.twist.linear.x = cmd_v
                self.twist.angular.z = cmd_w
                self.cmd_vel_pub.publish(self.twist)
            else:
                [cmd_v, cmd_w] = [0,0]


    def calculate_velocity(self):
        goal_position = self.path.poses[0].position
        goal_pose_theta = euler_from_quaternion(self.path.poses[0].orientation)[2]
        # calculate the radius of curvature
        # R = np.dot(goal_position, goal_position) / (2. * goal_position[1])
        if goal_pose_theta > 0:
            R = np.dot(goal_position, goal_position) / (goal_pose_theta/(2*np.pi))
            v_cmd = self.v_turn
            w_cmd = R * 2 * goal_pose_theta / v_cmd

        elif goal_pose_theta < 0:
            R = np.dot(goal_position, goal_position) / ((2*np.pi + goal_pose_theta)/(2*np.pi))
            v_cmd = self.v_turn
            w_cmd = R * 2 * goal_pose_theta / v_cmd

        else:
            R = 0
            v_cmd = self.v_turn
            w_cmd = 0

        # ensure velocity obeys the speed constraints
        r = self.wheel_radius
        L = self.wheel_base
        u = max(v_cmd / r + L * w_cmd / (2. * r) * np.array([-1, 1]))
        if u > self.v1:
            v_cmd = 0
            w_cmd = self.v_turn / L * np.sign(w_cmd)

        return (v_cmd, w_cmd)

    def goal_callback(self,msg):
        points = msg.points
        # try:


        if len(points)<=100:
            current_min_distance = np.Inf
            current_min_points = []
            curve_fitting_list = []
            for one_depth_point in points:
                temp_dist = sum((one_depth_point.x - self.t[0])**2 + (one_depth_point.y - self.t[1])**2)
                if temp_dist < current_min_distance:
                    current_min_distance = temp_dist
                    current_min_points = [one_depth_point.x,one_depth_point.y]
            for one_depth_point in points:
                if (one_depth_point.x - current_min_points[0])**2 + (one_depth_point.y - current_min_points[1])**2 < self.curve_fitting_distance_threshold**2:
                    curve_fitting_list.append([one_depth_point.x,one_depth_point.y])
        else:
            depth_points_robot = points[0:len(points):int(np.floor((len(points)-1)/99))]
            current_min_distance = np.Inf
            current_min_points = []
            curve_fitting_list = []
            for one_depth_point in depth_points_robot:
                print("one_depth_point",one_depth_point)
                temp_dist = sum((one_depth_point.x - self.t[0])**2 + (one_depth_point.y - self.t[1])**2)
                if temp_dist < current_min_distance:
                    current_min_distance = temp_dist
                    current_min_points = [one_depth_point.x,one_depth_point.y]
            for one_depth_point in depth_points_robot:
                if (one_depth_point.x - current_min_points[0])**2 + (one_depth_point.y - current_min_points[1])**2 < self.curve_fitting_distance_threshold**2:
                    curve_fitting_list.append([one_depth_point.x,one_depth_point.y])
        # px = np.array(curve_fitting_list)[:,0]
        # py = np.array(curve_fitting_list)[:,1]
        # reg = LinearRegression().fit(np.reshape(px,[len(px),1]), np.reshape(py,[len(py),1]))
        # k = reg.coef_[0][0]
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(curve_fitting_list)[:,0],np.array(curve_fitting_list)[:,1])
            if slope >= 0:
                if current_min_points[0]>=0:
                    theta = np.arctan(slope)
                else:
                    theta = np.arctan(slope) + np.pi
            else:
                if current_min_points[0]>=0:
                    theta = np.arctan(slope)
                else:
                    theta = np.arctan(slope)+ np.pi

        except: # k = Inf
            if current_min_points[1]>=0:
                theta = np.pi/2
            else:
                theta = -np.pi/2

        temp_pose = Pose()
        temp_pose.position = [current_min_points[0],current_min_points[1],0]
        temp_pose.orientation = tf.transformations.quaternion_from_euler(0, 0, theta)
        self.path.header = msg.header
        self.path.poses = temp_pose
        # except:
        #     print("failed to find available tf")



if __name__ == '__main__':
    rospy.init_node('follower_controller')
    follower = Follower()
    rospy.spin()
    # END ALL

#!/usr/bin/env python
# BEGIN ALL
import threading
import numpy as np
import rospy
from geometry_msgs.msg import Twist,Pose
from followbot.msg import MGSMeasurements, MGSMeasurement, MGSMarker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import PointCloud
from nav_msgs.msg import Path,Odometry
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
        self.tflistener = tf.TransformListener()
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
        self.last_odom = Odometry()
        self.fixed_theta = 0
        # self.turning_distance_threshold = 0.3
        # self.temp_turning_position = []

        self.success = False
        while not self.success:
            try:

                (trans,rot) = self.tflistener.lookupTransform('/base_link', '/mono_camera_link', rospy.Time(0))
                temp_quaternion = (rot[0],rot[1],rot[2],rot[3])
                self.fixed_theta = euler_from_quaternion(temp_quaternion)[2]
                self.R = np.array([[np.cos(self.fixed_theta), -np.sin(self.fixed_theta)],
                               [np.sin(self.fixed_theta),  np.cos(self.fixed_theta)]])
                self.t = np.array([[trans[0]],
                               [trans[1]]])
            except:
                pass
            else:
                self.R = np.array([[np.cos(self.fixed_theta), -np.sin(self.fixed_theta)],
                               [np.sin(self.fixed_theta),  np.cos(self.fixed_theta)]])
                self.t = np.array([[trans[0]],
                               [trans[1]]])
                self.t0 = np.array([[trans[0]],
                               [trans[1]]])
                self.success = True

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

        if msg.GAP_FLAG ==0:

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
                # print("self.path.header.stamp.to_sec()",self.path.header.stamp.to_sec())
                last_odom = msg.last_odometry
                current_odom = msg.odometry
                [cmd_v, cmd_w] = self.calculate_velocity(last_odom,current_odom)
                self.twist.linear.x = cmd_v
                self.twist.angular.z = cmd_w
                self.cmd_vel_pub.publish(self.twist)
            else:
                [cmd_v, cmd_w] = [0,0]


    def calculate_velocity(self,last_odom,current_odom):

        last_position = last_odom.pose.pose.position
        last_orientation = last_odom.pose.pose.orientation
        current_position = current_odom.pose.pose.position
        current_orientation = current_odom.pose.pose.orientation
        trans_l2c = np.array([current_odom.pose.pose.position.x - last_odom.pose.pose.position.x,current_odom.pose.pose.position.y - last_odom.pose.pose.position.y])
        theta_last = euler_from_quaternion([last_orientation.x,last_orientation.y,last_orientation.z,last_orientation.w])[2]
        theta_current = euler_from_quaternion([current_orientation.x,current_orientation.y,current_orientation.z,current_orientation.w])[2]
        dtheta = theta_current - theta_last
        print("self.path.poses.position",self.path.poses.position)
        if np.abs(dtheta)<0.1 and trans_l2c.dot(trans_l2c.T) < 1:
            # no transformation/rotation between last and current base_link in world frame
            goal_position = np.array([self.path.poses.position[0],self.path.poses.position[1]])
            goal_pose_theta = euler_from_quaternion(self.path.poses.orientation)[2]
        else:
            goal_position = np.array([self.path.poses.position[0],self.path.poses.position[1]])
            goal_position = np.matmul(np.array([[np.cos(dtheta),-np.sin(dtheta)],[np.sin(dtheta), np.cos(dtheta)]]), [goal_position[0] - trans_l2c[0],goal_position[1] - trans_l2c[1]])
            goal_pose_theta = euler_from_quaternion(self.path.poses.orientation)[2] + dtheta
        # calculate the radius of curvature
        # R = np.dot(goal_position, goal_position) / (2. * goal_position[1])
        # print("goal_pose_theta",goal_pose_theta)
        # transfer the goal into
        try:
            goal_position = [np.matmul(self.R, [goal_position[0],goal_position[1]])[0]+self.t[0] , np.matmul(self.R, [goal_position[0],goal_position[1]])[1]+self.t[1]]
            goal_pose_theta = goal_pose_theta + self.fixed_theta
        except:
            print("self.R",self.R,"self.t",self.t,"goal_position",goal_position)
        print("goal_position_1",goal_position)
        if 0 < goal_pose_theta < np.pi/2 or goal_pose_theta > 3 * np.pi/2 :
            if goal_pose_theta < 0:
                goal_pose_theta = np.pi + goal_pose_theta
            print("case_1")
            R = (goal_position[0]* goal_position[0]+goal_position[1]* goal_position[1]) / (goal_pose_theta/(2*np.pi))
            v_cmd = self.v_turn
            w_cmd = goal_pose_theta / ((goal_position[0]* goal_position[0]+goal_position[1]* goal_position[1])/v_cmd)

        elif np.pi/2 < goal_pose_theta < 3* np.pi/2:
            R = (goal_position[0]* goal_position[0]+goal_position[1]* goal_position[1]) / ((2*np.pi + goal_pose_theta)/(2*np.pi))
            v_cmd = self.v_turn
            w_cmd = R * 2 * goal_pose_theta / v_cmd
            print("case_2")
        else:
            R = 0
            v_cmd = self.v_turn
            w_cmd = 0
            print("case_3")
        # ensure velocity obeys the speed constraints
        r = self.wheel_radius
        L = self.wheel_base
        print("v_cmd",v_cmd,"w_cmd",w_cmd,"goal_pose_theta",goal_pose_theta,"goal_position",goal_position)
        u = max(v_cmd + L * w_cmd * 1,v_cmd - L * w_cmd * 1)
        # print("u",u,self.v4)
        if u > self.v4:
            v_cmd = 0
            w_cmd = self.v_turn / L * np.sign(w_cmd)
            # print("w_cmd")
        return (v_cmd, w_cmd)

    def goal_callback(self,msg):

        points = msg.points
        # try:
        if len(points)<=100:
            current_min_distance = np.Inf
            current_min_points = []
            curve_fitting_list = []
            current_max_points = []
            current_max_distance = 0
            for one_depth_point in points:
                temp_dist = one_depth_point.x**2 + one_depth_point.y **2
                if temp_dist < current_min_distance:
                    current_min_distance = temp_dist
                    current_min_points = [one_depth_point.x,one_depth_point.y]
            for one_depth_point in points:
                temp_dist = (one_depth_point.x - current_min_points[0])**2 + (one_depth_point.y - current_min_points[1])**2
                if  temp_dist < self.curve_fitting_distance_threshold**2:
                    if temp_dist > current_max_distance:
                        current_max_distance = temp_dist
                        current_max_points = [one_depth_point.x,one_depth_point.y]
                    curve_fitting_list.append([one_depth_point.x,one_depth_point.y])

        else:
            depth_points_robot = points[0:len(points):int(np.floor((len(points)-1)/99))]
            current_min_distance = np.Inf
            current_min_points = []
            current_max_points = []
            current_max_distance = 0
            curve_fitting_list = []
            for one_depth_point in depth_points_robot:
                # print("one_depth_point",one_depth_point)
                temp_dist = one_depth_point.x **2 + one_depth_point.y **2
                if temp_dist < current_min_distance:
                    current_min_distance = temp_dist
                    current_min_points = [one_depth_point.x,one_depth_point.y]
            for one_depth_point in depth_points_robot:
                temp_dist = (one_depth_point.x - current_min_points[0])**2 + (one_depth_point.y - current_min_points[1])**2
                if  temp_dist < self.curve_fitting_distance_threshold**2:
                    if temp_dist > current_max_distance:
                        current_max_distance = temp_dist
                        current_max_points = [one_depth_point.x,one_depth_point.y]
                    curve_fitting_list.append([one_depth_point.x,one_depth_point.y])

        # px = np.array(curve_fitting_list)[:,0]
        # py = np.array(curve_fitting_list)[:,1]
        # reg = LinearRegression().fit(np.reshape(px,[len(px),1]), np.reshape(py,[len(py),1]))
        # k = reg.coef_[0][0]
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(curve_fitting_list)[:,0],np.array(curve_fitting_list)[:,1])
            temp_vec = [current_max_points[0] - current_max_points[0],current_max_points[1] - current_max_points[1]]
            vec_pos = [1,slope*1]
            vec_neg = [-1, -slope*1]
            case_1 = np.dot(temp_vec, vec_pos)
            case_2 = np.dot(temp_vec, vec_neg)
            if case_1 >= 0:
                if slope>=0:
                    theta = np.arctan(slope)
                else:
                    theta = np.arctan(slope)+np.pi
            else:
                if slope>=0:
                    theta = np.arctan(slope)+ np.pi
                else:
                    theta = np.arctan(slope)+ 2*np.pi
            print("theta",theta,"slope",slope)
                    # print("case4")
        except: # k = Inf
            if current_min_points[0]>=0:
                theta = np.pi/2
                # print("case5")
            else:
                theta = 3* np.pi/2
                # print("case6")
        temp_pose = Pose()
        temp_pose.position = [current_min_points[0],current_min_points[1],0]
        print("temp_pose.position",temp_pose.position,"theta",theta )
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

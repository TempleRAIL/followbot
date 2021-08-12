#!/usr/bin/env python
# BEGIN ALL
from __future__ import division
import functools
import sys
import numpy as np

import rospy
import rosparam
import cv2, cv_bridge
from sensor_msgs.msg import BatteryState,Image
from std_msgs.msg import Int8,ColorRGBA
from followbot.msg import MGSMeasurement, MGSMeasurements, MGSMarker
import math
import message_filters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line,hough_line, hough_line_peaks
from scipy.spatial import distance
import decimal
from nav_msgs.msg import Odometry,Path
from tf.transformations import euler_from_quaternion,quaternion_from_euler
import tf2_ros
from geometry_msgs.msg import Twist, Pose, PoseStamped, Point
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import Marker,MarkerArray
# from roboteq_motor_controller_driver.msg import channel_values
from scipy.optimize import fsolve
import tf
from vectors import *

class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return math.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint y1)
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with
        min_distance_to_merge = 30
        min_angle_to_merge = 30
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])
        # special case
        if (len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical

        if 45 < orientation and orientation < 135:
            # sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            # sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines, img):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        for line_i in [l for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation and  orientation < 135:
                lines_y.append(line_i)
            else:
                lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_x, lines_y]:
            if len(i) > 0:
                groups = self.merge_lines_pipeline_2(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_lines_segments1(group))
                merged_lines_all.extend(merged_lines)
        return merged_lines_all


class Controller:
    def __init__(self):
        self.twist = Twist()
        self.lookahead = rospy.get_param('~lookahead',1)
        self.rate = rospy.get_param('~rate',10)
        self.goal_margin = rospy.get_param('~goal_margin',0.1)

        self.wheel_base = rospy.get_param('~wheel_base',0.8)
        self.wheel_radius = rospy.get_param('~wheel_radius',0.2)
        self.v_max = rospy.get_param('~v_max',0.1)
        self.w_max = rospy.get_param('~w_max',0.8)
        self.distance_mag = 20.8 * 0.0254
        self.scale_factor = 0.5 # from true world to simulation
        self.p = self.scale_factor*rospy.get_param('~p', -7.0) # proportional controller constant
        self.v1 = self.scale_factor * rospy.get_param('~v', 0.666)  # nominal velocity (1.49 MPH)
        self.v2 = self.scale_factor * rospy.get_param('~v', 0.782)  # nominal velocity (1.75 MPH)
        self.v3 = self.scale_factor * rospy.get_param('~v', 0.849)  # nominal velocity (1.90 MPH)
        self.v4 = self.scale_factor * rospy.get_param('~v', 0.939)  # nominal velocity (2.10 MPH)
        self.v_turn = self.scale_factor*rospy.get_param('~v', 0.425)
        self.max_turning_omega = 1.5 * self.v_turn/1
        self.current_v = []
        self.current_theta = []
        self.last_goal = []


    def find_goal(self, current_p, current_vel, final_p, dist_min,final_theta):
        # x0, y0 = np.array(current_p)
        x0, y0 = np.array([0,0])
        vx0, vy0 = np.array(current_vel)  # vx = vel, vy = 0
        xf, yf = np.array(final_p)
        vxf, vyf = np.array([self.v1*np.cos(final_theta),self.v1*np.sin(final_theta)])
        if np.sqrt(xf ** 2 + yf ** 2) >= dist_min:
            total_expected_time = 1.2 * np.sqrt(xf**2+yf**2)/self.v1
            x_params = self.cubic_poly_motion_planning(x0, vx0, xf, vxf,total_expected_time)
            y_params = self.cubic_poly_motion_planning(y0, vy0, yf, vyf,total_expected_time)
            min_circle_time = self.convolve_and_sum(x_params, y_params, dist_min)
            # print("min_circle_time",min_circle_time)
            if min_circle_time == -1:
                min_circle_time = 1.2 * dist_min/self.v1
            x_params = np.reshape(x_params,4)
            y_params = np.reshape(y_params,4)
            goal_x = np.matmul(x_params , np.array([1, min_circle_time, min_circle_time ** 2, min_circle_time ** 3]))
            goal_y = np.matmul(y_params , np.array([1, min_circle_time, min_circle_time ** 2, min_circle_time ** 3]))
            time_segment_list = np.linspace(0, total_expected_time, 10+1)[1:]
            midway_x_list = []
            midway_y_list = []
            for time_segment in time_segment_list:
                goal_x_seg = np.matmul(x_params , np.array([1, time_segment, time_segment ** 2, time_segment ** 3]))
                goal_y_seg = np.matmul(y_params , np.array([1, time_segment, time_segment ** 2, time_segment ** 3]))
                midway_x_list.append(goal_x_seg)
                midway_y_list.append(goal_y_seg)
            # print("x_params",x_params,"y_params",y_params)
            vx = np.matmul(x_params , np.array([0, 1, 2 * min_circle_time, 3 * min_circle_time ** 2]))
            vy = np.matmul(y_params , np.array([0, 1, 2 * min_circle_time, 3 * min_circle_time ** 2]))
            if vx <= 0.01:
                theta = np.pi/2
            else:
                theta = np.arctan(vy/vx)
            case = 2

        else:
            goal_x = xf
            goal_y = yf
            goal = [goal_x,goal_y]
            R = np.dot(goal, goal) / (2. * goal[1])
            midway_x_list = []
            midway_y_list = []
            if np.isinf(R):
                arc_theta_list =  np.linspace(0, np.sign(R)*0.01, 10+1)[1:]
                for temp_theta in arc_theta_list:
                    goal_x_seg = R * np.cos(temp_theta)
                    goal_y_seg = R * np.sin(temp_theta)
                    midway_x_list.append(goal_x_seg)
                    midway_y_list.append(goal_y_seg)
                case = 3
            else:
                if R > 0:
                    arc_theta = np.pi/2 - np.arctan(goal_x/(R - goal_y))
                    arc_theta_list =  np.linspace(0, arc_theta, 10+1)[1:]
                    for temp_theta in arc_theta_list:
                        goal_x_seg = R * np.sin(temp_theta)
                        goal_y_seg = R * (1 - np.cos(temp_theta))
                        midway_x_list.append(goal_x_seg)
                        midway_y_list.append(goal_y_seg)
                elif R < 0:
                    arc_theta = np.pi/2 - np.arctan(np.abs(goal_x)/(np.abs(R) - np.abs(goal_y)))
                    arc_theta_list =  np.linspace(0, arc_theta, 10+1)[1:]
                    for temp_theta in arc_theta_list:
                        goal_x_seg = np.abs(R) * np.sin(temp_theta)
                        goal_y_seg = R * (1 - np.cos(temp_theta))
                        midway_x_list.append(goal_x_seg)
                        midway_y_list.append(goal_y_seg)
                case = 3
        [v, w, R] = self.calculate_velocity([goal_x, goal_y])  # to be changed with proportional control
        print("v, w,previous",v, w)
        # [update_theta,update_R] = fsolve(self.radius_solver,np.array([np.pi - 2 * np.arctan(np.abs(goal_x)/(np.abs(R)) - np.abs(goal_y)), np.abs(R)]),args = (self.distance_mag,goal_x,goal_y))
        # if np.sign(R):
        #     update_theta,update_R = np.abs(update_theta),np.abs(update_R)
        # else:
        #     update_theta,update_R = np.abs(update_theta),-np.abs(update_R)
        # [v, w, R] = self.calculate_velocity([update_R*np.sin(np.pi - update_theta), update_R*(1 + np.cos(np.pi-update_theta))])  # to be changed with proportional control
        new_goal = [0,0]
        # new_goal = [update_R*np.sin(np.pi - update_theta), update_R*(1 + np.cos(np.pi-update_theta))]
        print("[goal_x, goal_y]",[goal_x, goal_y],"v_cub,w_cub",v,w)

        # [v, w] = self.calculate_velocity(final_p)
        self.twist.linear.x = v
        self.twist.angular.z = w
        print("final_p",final_p,"245,v,w",v,w)
        return [self.twist,midway_x_list,midway_y_list,R,new_goal,case]

    def cubic_poly_motion_planning(self, currentp, current_vel, finalp, final_vel,total_expected_time):
        p0 = np.array([currentp])
        v0 = np.array([current_vel])  # vx = vel, vy = 0
        pf = np.array([finalp])
        vf = np.array([final_vel])
        dt = total_expected_time
        a0 = p0
        a1 = v0
        M = np.array([[dt ** 2, dt ** 3], [2 * dt, 3 * dt ** 2]])
        a2, a3 =np.matmul(np.linalg.inv(M) , np.array([pf - a0 - dt * a1, vf - a1]))#here is a problem
        # print("dt",dt,"a0",a0,"a1",a1,"a2",a2,"a3",a3)
        return [list(a0), list(a1), list(a2), list(a3)]

    def convolve_and_sum(self, x_a, y_a, dist_min):
        x_a = np.reshape(x_a,4)
        y_a = np.reshape(y_a,4)
        x_a_convolve = np.convolve(x_a, x_a)
        y_a_convolve = np.convolve(y_a, y_a)
        # print("x_a_convolve",x_a_convolve,"y_a_convolve",y_a_convolve)
        poly_param = x_a_convolve + y_a_convolve
        poly_param[0] = poly_param[0] - dist_min**2
        # print("poly_param",poly_param)
        [low_bound,up_bound] = self.init_binary_search(poly_param)
        while np.abs(low_bound - up_bound) > 0.1:
            [low_bound,up_bound] = self.binary_search(poly_param,low_bound,up_bound)
            # print("np.abs(low_bound - up_bound)",np.abs(low_bound - up_bound),low_bound,up_bound)
            if up_bound == -1:
                break
            elif np.abs(low_bound - up_bound) < 0.1:
                break
        if up_bound == -1:
            return -1
        else:
            return low_bound + (up_bound - low_bound)/2.0

    def init_binary_search(self,poly_param):
        # negative = -1, positive = 1
        poly_sign_dict = dict()
        for t in range(20):
            t_list = np.array([1,t,t**2,t**3,t**4,t**5,t**6])
            if np.dot(t_list,poly_param) > 0.01:
                poly_sign_dict[t] = 1
            elif np.dot(t_list,poly_param) < -0.01:
                poly_sign_dict[t] = -1
            else:
                poly_sign_dict[t] = 0

        p0 = poly_sign_dict[0]
        j = 0
        for i in poly_sign_dict:
            if i!=0:
                if poly_sign_dict[i] != p0:
                    j = i
                    break
        if j == 0:
            return [0,-1]
        else:
            return [0, j]

    def binary_search(self, poly_param, low_bound, up_bound):
        t = (low_bound + up_bound) / 2.0
        t0 = low_bound
        t1 = up_bound
        # print("t0,t,t1",t0,t,t1)
        low_value = np.dot(np.array([1,t0,t0**2,t0**3,t0**4,t0**5,t0**6]),poly_param)
        high_value = np.dot(np.array([1,t1,t1**2,t1**3,t1**4,t1**5,t1**6]),poly_param)
        mid_value = np.dot(np.array([1,t,t**2,t**3,t**4,t**5,t**6]),poly_param)
        if low_value * mid_value <= 0:
            return [t0, t]
        else:
            if high_value * mid_value <= 0:
                return [t, t1]
            else:
                return [t0,-1]

    def calculate_velocity(self, goal):
        # calculate the radius of curvature
        R = np.dot(goal, goal) / (2. * goal[1])
        v_cmd = w_cmd = 0.
        print("R",R)
        if np.abs(R) < 0.01:
            v_cmd = 0.
            w_cmd = self.w_max / np.sign(R)
        elif np.isinf(R):
            v_cmd = self.v_max
            w_cmd = 0.0
        else:
            v_cmd = np.sign(goal[0]) * 1
            w_cmd = v_cmd / R

        r = self.wheel_radius
        L = self.wheel_base
        u = v_cmd / r + L * w_cmd / (2. * r) * np.array([-1, 1])

        u_limit = min(self.v_max, self.w_max * L) / r
        u = u * u_limit / max(abs(u[0]), abs(u[1]))

        v = r / 2. * (u[0] + u[1])
        w = r / L * (u[1] - u[0])
        return (v, w, R)

    def straight_line_follower(self,tf_gap,tf_now):
        tf_gap_theta = euler_from_quaternion([tf_gap.transform.rotation.x,
                                              tf_gap.transform.rotation.y,
                                              tf_gap.transform.rotation.z,
                                              tf_gap.transform.rotation.w])[2]
        tf_gap_position = [tf_gap.transform.translation.x, tf_gap.transform.translation.y]
        tf_gap_position2 = [tf_gap.transform.translation.x + np.cos(tf_gap_theta), tf_gap.transform.translation.y + np.sin(tf_gap_theta)]
        tf_now_theta = euler_from_quaternion([tf_now.transform.rotation.x,
                                              tf_now.transform.rotation.y,
                                              tf_now.transform.rotation.z,
                                              tf_now.transform.rotation.w])[2]

        tf_now_position = [tf_now.transform.translation.x, tf_now.transform.translation.y]
        a = HoughBundler()
        dist = a.DistancePointLine(tf_now_position, [tf_gap_position[0], tf_gap_position[1], tf_gap_position2[0], tf_gap_position2[1]])
        self.twist.angular.z = self.p * dist
        vx = self.v1
        self.twist.linear.x = vx

    def cmd_vel_tf(self,tf_info,cmd_vel):
        print("tf_info")

    def radius_solver(self,x,*arg):
        goal_x = arg[1]
        goal_y = arg[2]
        l = arg[0]
        return [np.sin(x[0]) * x[1] + np.sin(x[0]) * l - goal_x, (1-np.cos(x[0]))*x[1] + np.cos(x[0]) * l - goal_y]



class markerGen():
    def __init__(self,current_position,goal_position,current_vel,goal_vel_theta,current_cmd_vel,midway_x_list,midway_y_list,R,new_goal,color_case):
        self.markerPointPub = rospy.Publisher("/markerPointPub", Marker, queue_size=10)
        self.markerPub = rospy.Publisher("/markerPub", MarkerArray, queue_size=10)
        self.robotGoalMarker = MarkerArray()
        self.robotPoseMarker = MarkerArray()
        self.robotPoseMarker_path = MarkerArray()
        self.goal_marker = MarkerArray()
        self.markerID = 0
        self.colorID = 0
        self.colorDict = {
            0:[0,0,0],
            1:[25,100,255],
            2:[0,255,0],
            3:[0,0,255],
            4:[255,255,0],
            5:[0,255,255],
            6:[255,0,255],
            7:[128,128,128],
            8:[0,0,128],
            9:[0,128,128],
            10:[128,0,128],
            11:[0,128,0]
        }
        self.markerID = 0

        marker1 = Marker()
        marker1.frame_locked = True
        marker1.ns = "Strip"
        marker1.header.frame_id = "/magnetic"
        marker1.id = self.markerID
        self.markerID+=1
        marker1.header.stamp = rospy.Time.now()
        marker1.type = Marker.LINE_STRIP
        marker1.scale.x = 0.05
        marker1Points = []
        marker1Colors = []
        marker1.lifetime = rospy.Duration(0.4)
        [colorR,colorG,colorB] = self.colorDict[color_case]
        # print("colorR,colorG,colorB",colorR,colorG,colorB)
        for num in range(0,len(midway_x_list)):
            pt = Point()
            pt.x = midway_x_list[num]
            pt.y = midway_y_list[num]
            pt.z = 0
            color = ColorRGBA()
            color.r = colorR/255
            color.g = colorG/255
            color.b = colorB/255
            color.a = 1
            marker1Points.append(pt)
            marker1Colors.append(color)
        marker1.points = marker1Points
        marker1.colors = marker1Colors
        self.robotPoseMarker_path.markers.append(marker1)
        for i in range(2):
            if i == 0:
                theta = 0
                position = current_position
                color_num = 2
            else:
                theta = goal_vel_theta
                position = goal_position
                color_num = 3
            robotPose_marker = Marker()
            robotPose_marker.header.frame_id = "/magnetic"
            robotPose_marker.type = Marker.ARROW
            robotPose_marker.ns = "Arrows"
            robotPose_marker.id = self.markerID
            self.markerID+=1
            # robotPose_marker.action = marker.ADD
            robotPose_marker.scale.x = 0.5
            robotPose_marker.scale.y = 0.05
            robotPose_marker.scale.z = 0.05
            robotPose_marker.color.a = 1.0
            robotPose_marker.color.r = self.colorDict[color_num][0]
            robotPose_marker.color.g = self.colorDict[color_num][1]
            robotPose_marker.color.b = self.colorDict[color_num][2]
            robotPose_marker.pose.orientation.w = 1.0
            robotPose_marker.pose.position.x = position[0]
            robotPose_marker.pose.position.y = position[1]
            robotPose_marker.pose.position.z = 0
            [x,y,z,w] = quaternion_from_euler(0,0,theta)
            robotPose_marker.pose.orientation.x = x
            robotPose_marker.pose.orientation.y = y
            robotPose_marker.pose.orientation.z = z
            robotPose_marker.pose.orientation.w = w
            self.robotPoseMarker.markers.append(robotPose_marker)

        goal_marker = Marker()
        goal_marker.header.frame_id = "/magnetic"
        goal_marker.type = Marker.SPHERE
        goal_marker.id = self.markerID
        self.markerID+=1
        goal_marker.scale.x = 1
        goal_marker.scale.y = 1
        goal_marker.scale.z = 1
        [x,y,z,w] = quaternion_from_euler(0,0,0)
        goal_marker.pose.orientation.x = x
        goal_marker.pose.orientation.y = y
        goal_marker.pose.orientation.z = z
        goal_marker.pose.orientation.w = w
        goal_marker.pose.position.x = new_goal[0]
        goal_marker.pose.position.y = new_goal[1]
        goal_marker.pose.position.z = 0
        goal_marker.color.r = self.colorDict[7][0]
        goal_marker.color.g = self.colorDict[7][1]
        goal_marker.color.b = self.colorDict[7][2]
        goal_marker.action = goal_marker.ADD
        self.goal_marker.markers.append(goal_marker)

        self.markerPub.publish(self.robotPoseMarker_path)
        self.markerPub.publish(self.robotPoseMarker)
        self.markerPub.publish(self.goal_marker)
        # self.markerPub.publish(self.robotGoalMarker)



class Detector:
    def __init__(self):
        self.twist = Twist()
        self.bridge = cv_bridge.CvBridge()
        self.detection_image_pub = rospy.Publisher("/detection_result_usage",Image,queue_size=5)
        # self.lidar_front_detection_sub = rospy.Subscriber('lidar_front',channel_values,self.lidar_front_callback,queue_size = 5)
        # self.lidar_back_detection_sub = rospy.Subscriber('lidar_back',channel_values,self.lidar_back_callback,queue_size = 5)
        # self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)
        # T265 parameters
        self.PPX1 = 419.467010498047
        self.PPY1 = 386.97509765625
        self.Fx1 = 286.221588134766
        self.Fy1 = 287.480102539062
        self.K1 = np.array([[self.Fx1, 0., self.PPX1],
                            [0., self.Fy1, self.PPY1],
                            [0., 0., 1.]])
        self.D1 = np.array([-0.0043481751345098, 0.037125650793314, -0.0355393998324871, 0.00577297387644649])
        self.Knew1 = self.K1.copy()
        self.Knew1[(0, 1), (0, 1)] = 1 * self.Knew1[(0, 1), (0, 1)]
        self.R = np.eye(3)
        self.t = np.array([0.15, -0.03, 0.15])
        self.detection_pub = rospy.Publisher("/detected_line", PoseStamped, queue_size=5)
        self.mgs = 1
        self.last_trans = []
        self.last_theta = []
        self.mag_to_cam = [2*0.0254,-3*0.0254,0]
        self.br = tf.TransformBroadcaster()
        self.br.sendTransform((-6*0.0254, 3*0.0254, 0),tf.transformations.quaternion_from_euler(0, 0, 0),rospy.Time.now(),"magnetic","camera_pose_frame")
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        rate = rospy.Rate(10.0)
        self.current_v = []
        self.current_theta = 0
        self.vx_max = 0.1
        self.az_max = 0.85
        self.lidar_flag = 1
        self.twist = Twist()
        self.bridge = cv_bridge.CvBridge()
        self.img_sub_1 = message_filters.Subscriber('camera/fisheye1/image_raw', Image)
        self.img_sub_2 = message_filters.Subscriber('camera/fisheye2/image_raw', Image)
        # self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)
        self.measurements = message_filters.ApproximateTimeSynchronizer([self.img_sub_1, self.img_sub_2], queue_size=15, slop=0.1)
        self.measurements.registerCallback(self.measurements_callback)
        self.odom_sub = rospy.Subscriber('camera/odom/sample', Odometry, self.odom_callback,queue_size=5)
        self.gap_vel_pub = rospy.Publisher('/gap_cmd_vel', Twist, queue_size=5)

        while not rospy.is_shutdown():
            self.br.sendTransform((-6*0.0254, 3*0.0254, 0),tf.transformations.quaternion_from_euler(0, 0, 0),rospy.Time.now(),"magnetic","camera_pose_frame")
            if self.mgs == 1:
                try:
                    self.trans = tfBuffer.lookup_transform('camera_odom_frame', 'camera_pose_frame', rospy.Time())
                    self.theta = euler_from_quaternion([self.trans.transform.rotation.x,
                                                        self.trans.transform.rotation.y,
                                                        self.trans.transform.rotation.z,
                                                        self.trans.transform.rotation.w])[2]
                    self.last_trans = []
                    self.last_theta = []
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    continue
            else:
                try:
                    self.trans = tfBuffer.lookup_transform('camera_odom_frame', 'camera_pose_frame', rospy.Time())
                    self.theta = euler_from_quaternion([self.trans.transform.rotation.x,
                                                        self.trans.transform.rotation.y,
                                                        self.trans.transform.rotation.z,
                                                        self.trans.transform.rotation.w])[2]
                    if not self.last_trans:
                        self.last_trans = self.trans
                        self.last_theta = self.theta
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    continue

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        theta = msg.twist.twist.angular.z
        self.current_v = [vx, vy]
        self.current_theta = theta

    # def lidar_front_callback(self,msg):
    #     if any(msg.value):
    #         self.twist.angular.z = 0
    #         self.twist.angular.y = 0
    #         self.twist.angular.x = 0
    #         self.twist.linear.x = 0
    #         self.twist.linear.y = 0
    #         self.twist.linear.z = 0
    #         self.lidar_flag = 0
    #     else:
    #         self.lidar_flag = 1
    #
    # def lidar_back_callback(self,msg):
    #     if any(msg.value):
    #         self.twist.angular.z = 0
    #         self.twist.angular.y = 0
    #         self.twist.angular.x = 0
    #         self.twist.linear.x = 0
    #         self.twist.linear.y = 0
    #         self.twist.linear.z = 0
    #         self.lidar_flag = 0
    #     else:
    #         self.lidar_flag = 1

    def measurements_callback(self, img1, img2):
        cv_image1 = self.bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        cv_image2 = self.bridge.imgmsg_to_cv2(img2, desired_encoding='bgr8')
        img_undistorted = cv2.fisheye.undistortImage(cv_image1, self.K1, D=self.D1, Knew=self.Knew1)
        img_undistorted = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(img_undistorted,(7,7),0)
        (thresh, im_bw2) = cv2.threshold(img_undistorted, 0, 32,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel1 = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((3, 3), np.uint8)
        num_column = np.shape(im_bw2)[1]
        im_bw2_left = im_bw2[:,0:np.int(np.floor(num_column/4))]
        im_bw2_mid = im_bw2[:,(np.int(np.floor(num_column/4))+1):np.int(np.floor(3 * num_column/4)) -1]
        im_bw2_right = im_bw2[:, np.int(np.floor(3 * num_column/4)):num_column]
        print("shape im_bw2",np.shape(im_bw2),np.floor(np.shape(im_bw2)[1]/4))
        im_bw2_left = cv2.dilate(im_bw2_left, kernel1, cv2.BORDER_REFLECT)
        im_bw2_mid = cv2.dilate(im_bw2_mid, kernel2, cv2.BORDER_REFLECT)
        im_bw2_right = cv2.dilate(im_bw2_right, kernel1, cv2.BORDER_REFLECT)
        im_bw2 = np.hstack((im_bw2_left,im_bw2_mid,im_bw2_right))
        # im_bw2 = cv2.dilate(im_bw2, kernel1, cv2.BORDER_REFLECT)
        im_bw2[0:int(self.PPY1)+25,:] = 0
        edges = cv2.Canny(im_bw2, 0,127) # with the black&white image, any value seems to be fine
        # edges[:,0:100] = 0
        # edges[:,-100:-1] = 0
        # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        # h, theta, d = hough_line(edges, theta=tested_angles)
        prob_hough_lines = probabilistic_hough_line(edges, threshold = 12, line_length=25, line_gap=10)
        new_lines = []
        for line in prob_hough_lines:
            p0, p1 = line
            new_lines.append((p0[0], p0[1], p1[0], p1[1]))

        a = HoughBundler()
        foo = a.process_lines(new_lines, edges)
        [goal,theta_lines] = self.XY_2_XYZ_Goal(foo, img_undistorted.shape)
        # target_pose = PoseStamped()
        # target_pose.pose.position.x = goal[0]
        # target_pose.pose.position.y = goal[1]
        # target_pose.pose.position.z = goal[2]
        # # target_pose.pose.orientation.x = quaternion_from_euler(0, 0, yaw)
        print("goal", goal)

        controller_object = Controller()
        if any(goal):
            current_p = [self.trans.transform.translation.x, self.trans.transform.translation.y]
            # target_v = [self.current_v[0] * np.cos(goal[3]), self.current_v[0] * np.sin(goal[3])]
            [result,midway_x_list,midway_y_list,R,new_goal,color_case] = controller_object.find_goal(current_p, self.current_v, [goal[0], goal[1]], 0.1, goal[3])

        else:
            result = controller_object.straight_line_follower(self.last_trans, self.trans)
            midway_x_list = []
            midway_y_list = []
            new_goal = [0.1,0,-0.15]
            R = []
            color_case = 4

        self.twist = result

        current_v = [self.current_v[0],self.current_v[1],self.current_theta]
        goal_vel_theta = goal[3]
        # rviz_sim = markerGen([0,0], [goal[0],goal[1]], current_v,goal_vel_theta,result,midway_x_list,midway_y_list,R,new_goal,color_case)


        if np.abs(self.twist.linear.x) <= 0.001:
            self.twist.linear.x = 0
            print("case_0",self.twist.linear.x,self.twist.angular.z)
        else:
            self.twist.linear.x = self.twist.linear.x + 0.35 * np.sign(self.twist.linear.x)
            print("case_-1",self.twist.linear.x,self.twist.angular.z)

        if self.lidar_flag == 0:
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            print("case_1",self.twist.linear.x,self.twist.angular.z)

        if np.abs(self.twist.linear.x) > self.vx_max or np.abs(self.twist.angular.z) > self.az_max:
            print("case_2",self.twist.linear.x,self.twist.angular.z)
            propotion1 = np.abs(self.twist.linear.x/self.vx_max) + 0.01
            propotion2 = np.abs(self.twist.angular.z/self.az_max) + 0.01
            propotion = max(propotion1,propotion2)
            if propotion > 0.01:
                self.twist.linear.x = self.twist.linear.x / propotion
                self.twist.angular.z = self.twist.angular.z / propotion
            else:
                self.twist.linear.x = 0
                self.twist.angular.z = 0

        if np.abs(self.twist.linear.x) > self.vx_max or np.abs(self.twist.angular.z) > self.az_max:
            print("case_3",self.twist.linear.x,self.twist.angular.z)
            self.twist.linear.x = 0
            self.twist.linear.y = 0
            self.twist.linear.z = 0
            self.twist.angular.x = 0
            self.twist.angular.y = 0
            self.twist.angular.z = 0
            print("error")

        self.gap_vel_pub.publish(self.twist)

        # img = img_undistorted
        # for line in lines2:
        #     p0, p1 = line
        #     pts = np.array([[p0[0], p1[0]],[p0[1], p1[1]]])
        #     pts = pts.reshape((-1, 1, 2))
        #     img = cv2.line(img, p0, p1, [255, 255, 255], 3)
        # image_message = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
        # self.detection_image_pub.publish(image_message)


        # if np.abs(goal_vel_theta) > 1:
        # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 20),
        #                          sharex=True, sharey=True)
        # ax = axes.ravel()
        # ax[0].imshow(cv_image1, cmap='gray', vmin=0, vmax=255)
        # ax[0].axis('off')
        # ax[0].set_title('fisheye', fontsize=20)
        # ax[0].imshow(img_undistorted, cmap='gray', vmin=0, vmax=255)
        # ax[0].axis('off')
        # ax[0].set_title('img_undistorted', fontsize=20)
        #
        # ax[1].imshow(im_bw2 , cmap='gray', vmin=0, vmax=255)
        # ax[1].axis('off')
        # ax[1].set_title('gaussianBlur_img ', fontsize=20)
        #
        # ax[5].imshow(edges , cmap='gray')
        # for line in lines2:
        #     p0, p1 = line
        #     ax[5].plot((p0[0], p1[0]), (p0[1], p1[1]))
        #     ax[5].plot([p0[0], p1[0]], [p0[1], p1[1]], 'bo')
        #
        # ax[5].scatter(np.array([goal[7][0],goal[8][0]]),np.array([goal[7][1],goal[8][1]]),s=100,c='r', marker ="+")
        # ax[5].set_xlim((0, img_undistorted.shape[1]))
        # ax[5].set_ylim((img_undistorted.shape[0], 0))
        # ax[5].set_title('Merged Probabilistic Hough')
        #
        #
        # ax[2].imshow(edges, cmap='gray', vmin=0, vmax=255)
        # ax[2].axis('off')
        # ax[2].set_title('edges', fontsize=20)
        #
        # ax[3].imshow(edges * 0, cmap='gray')
        # for line in prob_hough_lines:
        #     p0, p1 = line
        #     ax[3].plot((p0[0], p1[0]), (p0[1], p1[1]))
        # ax[3].set_xlim((0, img_undistorted.shape[1]))
        # ax[3].set_ylim((img_undistorted.shape[0], 0))
        # ax[3].set_title('Probabilistic Hough')
        #
        # ax[4].imshow(edges , cmap='gray')
        # for line in foo:
        #     p0, p1 = line
        #     ax[4].plot((p0[0], p1[0]), (p0[1], p1[1]))
        #     ax[4].plot([p0[0], p1[0]], [p0[1], p1[1]], 'bo')
        #
        # ax[4].scatter(np.array([goal[7][0],goal[8][0]]),np.array([goal[7][1],goal[8][1]]),s=100,c='r', marker ="+")
        # ax[4].set_xlim((0, img_undistorted.shape[1]))
        # ax[4].set_ylim((img_undistorted.shape[0], 0))
        # ax[4].set_title('Merged Probabilistic Hough')
        #
        # fig.tight_layout()
        # plt.show()
        # # cv2.imshow("Image window", skeleton)
        # # cv2.waitKey(0)

    def theta_filter(self, points_list):
        points_1 = []
        points_2 = []
        temp_lines = []

        for line in points_list:
            print("line",line)
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            if np.abs((x1-x2)/(y1-y2)) > 0.001:
                k = np.abs((x1-x2)/(y1-y2))
            else:
                k = np.Inf
            case = (np.abs(y1) <= 0.4 or np.abs(y2) <= 0.4)
            if k >= 1 and case:
                temp_lines.append(line)
                points_1.append([line[0],line[1]])
                points_2.append([line[2],line[3]])
        points_1 = np.array(points_1)
        points_2 = np.array(points_2)
        return [temp_lines,points_1,points_2]

    def img2ground(self,XY_goal):
        Y = 150
        Z = (self.Fy1 * Y) / (XY_goal[1] - self.PPY1)
        X = -(XY_goal[0] - self.PPX1) / (self.Fx1 / Z)
        X,Y,Z = Z,-X,-Y
        return X/ 1000.0,Y/1000.0,Z/1000.0

    def camera_tf2_mag(self,x,y,z):
        x = x + 2 * 0.0254
        y = y - 3 * 0.0254
        return [x,y,z]

    def lineseg_dists(self,p, a, b):
        """Cartesian distance from point to line segment

        Edited to support arguments as series, from:
        https://stackoverflow.com/a/54442561/11208892

        Args:
            - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
            - a: np.array of shape (x, 2)
            - b: np.array of shape (x, 2)
        """
        # normalized tangent vectors
        d_ba = b - a
        d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                               .reshape(-1, 1)))
        # signed parallel distance components
        # rowwise dot products of 2D vectors
        s = np.multiply(a - p, d).sum(axis=1)
        t = np.multiply(p - b, d).sum(axis=1)
        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(s))])
        # perpendicular distance component
        # rowwise cross products of 2D vectors
        d_pa = p - a
        c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

        return np.hypot(h, c)

    def XY_2_XYZ_Goal(self, lines, shape):
        height, width = shape
        points_list2 = []
        print("inputlines",lines)
        imagelines = lines
        for line in lines:
            temp_X1,temp_Y1,temp_Z1 = self.img2ground(line[0])
            temp_X1,temp_Y1,temp_Z1 = self.camera_tf2_mag(temp_X1,temp_Y1,temp_Z1)
            temp_X2,temp_Y2,temp_Z2 = self.img2ground(line[1])
            temp_X2,temp_Y2,temp_Z2 = self.camera_tf2_mag(temp_X2,temp_Y2,temp_Z2)
            points_list2.append((temp_X1,temp_Y1,temp_X2,temp_Y2))
        lines,points_a,points_b = self.theta_filter(points_list2)

        if len(lines) == 0:
            print("no detection")
            result = np.array([0.1, 0, -0.15, 0, 0.1, 0, -0.15, [0.1, 0], [0.1, 0]])
        else:
            if len(lines) > 0:
                distance_list = self.lineseg_dists(np.array([0,0]),points_a,points_b)
                index2_1 = np.where(distance_list == distance_list.min())
                X = points_a[np.int(index2_1[0])][0]
                Y = points_a[np.int(index2_1[0])][1]
                Z = -0.15
                X2 = points_b[np.int(index2_1[0])][0]
                Y2 = points_b[np.int(index2_1[0])][1]
                Z2 = -0.15
                if np.abs(X2-X) < 0.01:
                    theta = 0
                else:
                    theta = np.arctan((X2-X)/(Y2-Y))
                result = np.array([ X,Y,Z,theta,X2,Y2,Z2,points_a[np.int(index2_1[0])],points_b[np.int(index2_1[0])]])
            else:
                print("no lines")
                result = np.array([0.1, 0, -0.15, 0, 0.1, 0, -0.15, [0.1, 0], [0.1, 0]])
        return [result,lines]


if __name__ == '__main__':
    rospy.init_node('line_detector')
    detector = Detector()
    rospy.spin()

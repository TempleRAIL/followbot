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
from scipy.spatial import distance
import decimal
from nav_msgs.msg import Odometry,Path
from tf.transformations import euler_from_quaternion,quaternion_from_euler
import tf2_ros
from geometry_msgs.msg import Twist, PoseStamped, Point
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import Marker,MarkerArray
from roboteq_motor_controller_driver.msg import channel_values
from vectors import *
import time


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

        if 0 < orientation and orientation < 180:
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
            if 0 < orientation and orientation < 180:
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
        self.rate = 10
        self.goal_margin = 0.1
        self.wheel_base = 0.8
        self.wheel_radius = 0.2
        self.v_max = 0.2
        self.w_max = 0.8
        self.cam_to_center = [-26*0.0254,2*0.0254]
        self.distance_mag = 20.8 * 0.0254
        self.p = -7.0 # proportional controller constant
        self.v0 = 0.1
        self.v1 = 0.5*0.666 # nominal velocity (1.49 MPH)
        self.v2 = 0.5*0.782 # nominal velocity (1.75 MPH)
        self.v3 = 0.5*0.849  # nominal velocity (1.90 MPH)
        self.v4 = 0.5*0.939  # nominal velocity (2.10 MPH)
        self.v_turn = 0.5 * 0.425
        self.max_turning_omega = 1.5 * self.v_turn/1
        self.current_v = 0
        self.current_theta = 0
        self.last_goal = []


    def find_goal(self, current_p, current_vel, final_p, dist_min,final_theta):

        # x0, y0 = np.array(current_p)
        x0, y0 = np.array([0,0])
        vx0 = current_vel  # vx = vel, vy = 0
        vy0 = 0
        xf, yf = np.array(final_p)
        vxf, vyf = np.array([self.v1*np.cos(final_theta),self.v1*np.sin(final_theta)])
        if np.sqrt(xf ** 2 + yf ** 2) >= dist_min:
            start = time.time()
            total_expected_time = 1.2 * np.sqrt(xf**2+yf**2)/self.v1
            x_params = self.cubic_poly_motion_planning(x0, vx0, xf, vxf,total_expected_time)
            y_params = self.cubic_poly_motion_planning(y0, vy0, yf, vyf,total_expected_time)
            time_601 = (time.time()-start)*1000
            start = time.time()
            min_circle_time = self.convolve_and_sum(x_params, y_params, dist_min)
            time_602 = (time.time() - start) * 1000
            start = time.time()
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
            time_603 = (time.time() - start) * 1000

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
        start = time.time()
        [v, w, R] = self.calculate_velocity([goal_x, goal_y])  # to be changed with proportional control

        time_604 = (time.time() - start) * 1000
        print("v, w,previous",v, w)
        # [update_theta,update_R] = fsolve(self.radius_solver,np.array([np.pi - 2 * np.arctan(np.abs(goal_x)/(np.abs(R)) - np.abs(goal_y)), np.abs(R)]),args = (self.distance_mag,goal_x,goal_y))
        # if np.sign(R):
        #     update_theta,update_R = np.abs(update_theta),np.abs(update_R)
        # else:
        #     update_theta,update_R = np.abs(update_theta),-np.abs(update_R)
        # [v, w, R] = self.calculate_velocity([update_R*np.sin(np.pi - update_theta), update_R*(1 + np.cos(np.pi-update_theta))])  # to be changed with proportional control
        new_goal = [goal_x, goal_y]
        # new_goal = [update_R*np.sin(np.pi - update_theta), update_R*(1 + np.cos(np.pi-update_theta))]
        print("[goal_x, goal_y]",[goal_x, goal_y],"v_cub,w_cub",v,w)
        # [v, w] = self.calculate_velocity(final_p)
        if np.abs(v) <= self.v_max/2 and np.abs(w) <= self.w_max/2 and np.abs(v) >=0.001 and np.abs(w) >= 0.001:
            p1 = v/self.v_max * 2
            p2 = w/self.w_max * 2
            ppp = max(np.abs(p1),np.abs(p2))
            v = v/ppp
            w = w/ppp

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
        a2, a3 =np.matmul(np.linalg.pinv(M) , np.array([pf - a0 - dt * a1, vf - a1]))#here is a problem
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
        while np.abs(low_bound - up_bound) > 0.4:
            [low_bound,up_bound] = self.binary_search(poly_param,low_bound,up_bound)
            # print("np.abs(low_bound - up_bound)",np.abs(low_bound - up_bound),low_bound,up_bound)
            if up_bound == -1:
                break
            elif np.abs(low_bound - up_bound) < 0.4:
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
        if np.abs(goal[1]) <= 0.00001:
            R = np.Inf
        else:
            R = np.dot(goal, goal) / (2. * goal[1])
        v_cmd = w_cmd = 0.
        print("R",R)
        if np.abs(R) < 0.01:
            v_cmd = 0.
            w_cmd = self.w_max / np.sign(R)
        elif np.isinf(R):
            v_cmd = self.v0
            w_cmd = 0.0
        else:
            v_cmd = np.sign(goal[0]) * 1
            w_cmd = v_cmd / R

        r = self.wheel_radius
        L = self.wheel_base
        u = v_cmd / r + L * w_cmd / (2. * r) * np.array([-1, 1])

        u_limit = min(self.v0, self.w_max * L) / r
        u = u * u_limit / max(abs(u[0]), abs(u[1]))

        v = r / 2. * (u[0] + u[1])
        w = r / L * (u[1] - u[0])
        return (v, w, R)

    def straight_line_follower(self,tf_gap,tf_now,vel):
        tf_gap_theta = euler_from_quaternion([tf_gap.transform.rotation.x,
                                              tf_gap.transform.rotation.y,
                                              tf_gap.transform.rotation.z,
                                              tf_gap.transform.rotation.w])[2]
        tf_gap_position = [tf_gap.transform.translation.x + self.cam_to_center[0], tf_gap.transform.translation.y + self.cam_to_center[1]]
        tf_now_theta = euler_from_quaternion([tf_now.transform.rotation.x,
                                              tf_now.transform.rotation.y,
                                              tf_now.transform.rotation.z,
                                              tf_now.transform.rotation.w])[2]
        tf_now_position = [tf_now.transform.translation.x + self.cam_to_center[0] * np.cos(tf_gap_theta - tf_now_theta),
                           tf_now.transform.translation.y + self.cam_to_center[1]* np.sin(tf_gap_theta - tf_now_theta)]
        temp_distance = np.sqrt((tf_now_position[0] - tf_gap_position[0])**2 + (tf_now_position[1] - tf_gap_position[1])**2)
        print("temp_distance",temp_distance)
        # tf_goal_position = [tf_gap_position[0] + (1+temp_distance)*np.cos(tf_gap_theta) - tf_now_position[0] ,
        #                     tf_gap_position[1]  + (1+temp_distance)*np.sin(tf_gap_theta) - tf_now_position[1]]
        # final_p = tf_goal_position
        # final_theta = tf_gap_theta - tf_now_theta
        # current_p = [0,0]
        # current_vel = vel[0]
        # dist_min = 0.1
        # [result,midway_x_list,midway_y_list,R,new_goal,color_case] = self.find_goal(current_p, current_vel, final_p, dist_min,final_theta)

        dtheta = tf_gap_theta - tf_now_theta
        if temp_distance >=2:
            v = self.v0
            w = dtheta * 0.25
            self.twist.linear.x = v
            self.twist.angular.z = w
            flag = 0
        else:
            v = self.v2
            w = dtheta * 0.25
            self.twist.linear.x = v
            self.twist.angular.z = w
            flag = 1
        return [self.twist, flag]

    def no_goal_no_flag(self):
        self.twist.angular.z = 0
        self.twist.linear.x = self.v0
        return self.twist

    def cmd_vel_tf(self,tf_info,cmd_vel):
        print("tf_info")

    def radius_solver(self,x,*arg):
        goal_x = arg[1]
        goal_y = arg[2]
        l = arg[0]
        return [np.sin(x[0]) * x[1] + np.sin(x[0]) * l - goal_x, (1-np.cos(x[0]))*x[1] + np.cos(x[0]) * l - goal_y]



class markerGen():
    def __init__(self,current_position,goal_position,goal_vel_theta,midway_x_list,midway_y_list,new_goal,color_case):
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
        marker1.header.frame_id = "/camera_odom_frame"
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
            pt = Point(midway_x_list[num],midway_y_list[num],0)
            pt.x = midway_x_list[num]
            pt.y = midway_y_list[num]
            pt.z = 0
            color = ColorRGBA()
            color.r = colorR/255.0
            color.g = colorG/255.0
            color.b = colorB/255.0
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
            robotPose_marker.header.frame_id = "/camera_odom_frame"
            robotPose_marker.type = Marker.ARROW
            robotPose_marker.ns = "Arrows"
            robotPose_marker.id = self.markerID
            self.markerID+=1
            # robotPose_marker.action = marker.ADD
            if i == 0:
                robotPose_marker.scale.x = 0.05
            else:
                robotPose_marker.scale.x = 0.50
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
        goal_marker.header.frame_id = "/camera_odom_frame"
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
        self.markerPub.publish(self.robotGoalMarker)



class Detector:
    def __init__(self):
        self.twist = Twist()
        self.bridge = cv_bridge.CvBridge()
        self.detection_image_pub = rospy.Publisher("/detection_result_usage",Image,queue_size=5)
        self.detection_image_pub2 = rospy.Publisher("/detection_result_usage2",Image,queue_size=5)
        self.straight_line_patrol_flag = 0
        self.mgs = 1
        self.current_v = 0
        self.current_w = 0
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)
        self.lidar_front_detection_sub = rospy.Subscriber('lidar_front',channel_values,self.lidar_front_callback,queue_size = 5)
        self.lidar_back_detection_sub = rospy.Subscriber('lidar_back',channel_values,self.lidar_back_callback,queue_size = 5)
        self.straight_line_patrol_sub = rospy.Subscriber('mag_marker_detect',channel_values,self.patrol_callback,queue_size = 5)
        self.mag_detection_sub = rospy.Subscriber('mag_track_detect',channel_values,self.mag_detection_callback,queue_size = 5)

        # T265 parameters
        self.map_1 = []
        self.map_2 = []
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
        self.look_forward = 0.1
        self.detection_pub = rospy.Publisher("/detected_line", PoseStamped, queue_size=5)

        self.last_trans = []
        self.last_theta = []
        self.mag_to_cam = [5 * 0.0254, -2 * 0.0254, 0]
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        rate = rospy.Rate(10.0)

        self.vx_max = 0.5
        self.az_max = 0.6
        self.lidar_flag = 1
        self.twist = Twist()
        self.bridge = cv_bridge.CvBridge()
        self.img_sub = rospy.Subscriber("camera/fisheye1/image_raw", Image, self.measurements_callback,queue_size=5)
        self.odom_sub = rospy.Subscriber('camera/odom/sample', Odometry, self.odom_callback,queue_size=5)
        self.gap_vel_pub = rospy.Publisher('/gap_cmd_vel', Twist, queue_size=5)

        while not rospy.is_shutdown():
            if self.mgs == 1:
                try:
                    self.trans = tfBuffer.lookup_transform('camera_odom_frame', 'camera_pose_frame', rospy.Time())
                    self.theta = euler_from_quaternion([self.trans.transform.rotation.x,
                                                        self.trans.transform.rotation.y,
                                                        self.trans.transform.rotation.z,
                                                        self.trans.transform.rotation.w])[2]
                    self.last_trans = self.trans
                    self.last_theta = self.theta
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    continue

            else:
                try:
                    self.trans = tfBuffer.lookup_transform('camera_odom_frame', 'camera_pose_frame', rospy.Time())
                    self.theta = euler_from_quaternion([self.trans.transform.rotation.x,
                                                        self.trans.transform.rotation.y,
                                                        self.trans.transform.rotation.z,
                                                        self.trans.transform.rotation.w])[2]
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    continue

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        theta = msg.twist.twist.angular.z
        self.current_v = vx
        self.current_w = theta

    def patrol_callback(self,msg):
        if msg.value[1] == 1:
            self.straight_line_patrol_flag = 1

    def mag_detection_callback(self,msg):
        if msg.value[0] == 1 or msg.value[1] == 1:
            self.mgs = 1
        else:
            self.mgs = 0

    def lidar_front_callback(self,msg):
        if any(msg.value):
            self.twist.angular.z = 0
            self.twist.angular.y = 0
            self.twist.angular.x = 0
            self.twist.linear.x = 0
            self.twist.linear.y = 0
            self.twist.linear.z = 0
            self.lidar_flag = 0
        else:
            self.lidar_flag = 1

    def lidar_back_callback(self,msg):
        if any(msg.value):
            self.twist.angular.z = 0
            self.twist.angular.y = 0
            self.twist.angular.x = 0
            self.twist.linear.x = 0
            self.twist.linear.y = 0
            self.twist.linear.z = 0
            self.lidar_flag = 0
        else:
            self.lidar_flag = 1

    def measurements_callback(self, img1):
        start = time.time()
        cv_image1 = self.bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        # cv2.initUndistortRectifyMap(self.K1, self.D1, rH, self.K1, imsize, cv.CV_16SC2)
        # http://blog.nishihara.me/opencv/2015/09/03/how-to-improve-opencv-performance-on-lens-undistortion-from-a-video-feed/
        if len(self.map_1) and len(self.map_2):
            img_undistorted = cv2.remap(cv_image1, self.map_1, self.map_2, interpolation=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT)
        else:
            self.map_1, self.map_2 = cv2.fisheye.initUndistortRectifyMap(self.K1, self.D1, np.eye(3), self.Knew1, (848,800), cv2.CV_32FC1)
            img_undistorted = cv2.remap(cv_image1, self.map_1, self.map_2, interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
        # img_undistorted = cv2.fisheye.undistortImage(cv_image1, self.K1, D=self.D1, Knew=self.Knew1)
        (height, width,z) = img_undistorted.shape
        print("img_undistorted.shape",img_undistorted.shape)
        img_undistorted2 = cv2.pyrDown(img_undistorted)
        time_1 = (time.time()-start)*1000
        start = time.time()
        img_undistorted2 = cv2.cvtColor(img_undistorted2, cv2.COLOR_BGR2GRAY)
        time_0 = (time.time()-start)*1000
        start = time.time()
        (thresh, im_bw2) = cv2.threshold(img_undistorted2, 0, 32,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        time_2 = (time.time()-start)*1000
        start = time.time()
        im_bw2[0:int(self.PPY1/2)+32,:] = 1
        kernel1 = np.ones((3, 3), np.uint8)
        im_bw2 = cv2.dilate(im_bw2, kernel1, cv2.BORDER_REFLECT)
        edges = cv2.Canny(im_bw2, 0,127) # with the black&white image, any value seems to be fine
        time_3 = (time.time()-start)*1000
        start = time.time()
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, None, 20, 10)
        new_lines = []
        for line in linesP:
            new_lines.append((line[0][0]*2,line[0][1]*2,line[0][2]*2,line[0][3]*2))
        if len(new_lines):
            a = HoughBundler()
            foo = a.process_lines(new_lines, edges)
            print("foo",foo)
            time_4 = (time.time()-start)*1000
            start = time.time()
            [goal,imagelines] = self.XY_2_XYZ_Goal(foo, img_undistorted2.shape)

            time_5 = (time.time()-start)*1000
            start = time.time()
            controller_object = Controller()
            time_6_0 = (time.time()-start)*1000
            start = time.time()
            false_goal = [goal[4],goal[5],goal[6]]
            print("false_goal",false_goal,"self.straight_line_patrol_flag",self.straight_line_patrol_flag)
            if false_goal != [0.1, 0, -0.15] and self.straight_line_patrol_flag == 0:
                current_p = [0, 0]
                [result, midway_x_list, midway_y_list, R, new_goal, color_case] = controller_object.find_goal(current_p, self.current_v, [goal[4], goal[5]], self.look_forward, goal[3])
                time_6 = (time.time() - start) * 1000
                goal_vel_theta = goal[3]
                color_case = 6
                print("time_0,time_1,time_2,time_3,time_4,time_5,time_6_0,time_6_1", time_0, time_1, time_2, time_3, time_4, time_5,time_6_0, time_6)
            elif false_goal == [0.1, 0, -0.15] and self.straight_line_patrol_flag == 1:
                [result, return_flag] = controller_object.straight_line_follower(self.last_trans, self.trans,[self.current_v,self.current_w])
                midway_x_list = np.linspace(0,2,11)
                midway_y_list = np.linspace(0,0,11)
                new_goal = [2,0]
                goal_vel_theta = 0
                self.straight_line_patrol_flag = return_flag
                time_6 = (time.time() - start) * 1000
                color_case = 1
                print("time_0,time_1,time_2,time_3,time_4,time_5,time_6_0,time_6_2", time_0, time_1, time_2, time_3, time_4, time_5,time_6_0, time_6)
            else:
                result = controller_object.no_goal_no_flag()
                midway_x_list = np.linspace(0,0.5,11)
                midway_y_list = np.linspace(0,0,11)
                new_goal = [0.5,0]
                goal_vel_theta = 0
                time_6 = (time.time() - start) * 1000
                color_case = 2
                print("time_0,time_1,time_2,time_3,time_4,time_5,time_6_0,time_6_3", time_0, time_1, time_2, time_3, time_4, time_5,time_6_0, time_6)
            print("\n")
            img = img_undistorted
            # image_message = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
            # self.detection_image_pub.publish(image_message)
            if imagelines:
                p0, p1 = imagelines
                pts = np.array([[p0[0], p1[0]],[p0[1], p1[1]]])
                pts = pts.reshape((-1, 1, 2))
                img = cv2.line(img, p0, p1, [255, 255, 255], 3)
                image_message = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
                self.detection_image_pub.publish(image_message)
            else:
                image_message = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
                self.detection_image_pub.publish(image_message)

            image_message_2 = self.bridge.cv2_to_imgmsg(edges, encoding="passthrough")
            self.detection_image_pub2.publish(image_message_2)
        else:
            result = Twist()
            result.angular.z = 0
            vx = 0.06
            result.linear.x = vx
            midway_x_list = np.linspace(0,0.5,11)
            midway_y_list = np.linspace(0,0,11)
            new_goal = [0.5,0]
            goal_vel_theta = 0
            color_case = 2
        self.twist = result


        # print("[goal[0],goal[1]]",[goal[0],goal[1]],"current_v",current_v,"goal_vel_theta",goal_vel_theta,"result",result,"midway_x_list",midway_x_list,"midway_y_list",midway_y_list,"R",R,"new_goal",new_goal)
        rviz_sim = markerGen([0,0],new_goal,goal_vel_theta,midway_x_list,midway_y_list,new_goal,color_case)

        if np.abs(self.twist.linear.x) <= 0.001:
            self.twist.linear.x = 0
            print("case_0",self.twist.linear.x,self.twist.angular.z)
        else:
            self.twist.linear.x = self.twist.linear.x
            print("case_-1",self.twist.linear.x,self.twist.angular.z)

        if self.lidar_flag == 0:
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            print("case_1",self.twist.linear.x,self.twist.angular.z)

        if np.abs(self.twist.linear.x) > self.vx_max or np.abs(self.twist.angular.z) > self.az_max:
            propotion1 = np.abs(self.twist.linear.x/self.vx_max) + 0.01
            propotion2 = np.abs(self.twist.angular.z/self.az_max) + 0.01
            propotion = max(propotion1,propotion2)
            if propotion > 0.01:
                self.twist.linear.x = self.twist.linear.x / propotion
                self.twist.angular.z = self.twist.angular.z / propotion
            else:
                self.twist.linear.x = 0
                self.twist.angular.z = 0
            print("case_2", self.twist.linear.x, self.twist.angular.z)

        if np.abs(self.twist.linear.x) > self.vx_max or np.abs(self.twist.angular.z) > self.az_max:
            print("case_3",self.twist.linear.x, self.twist.angular.z)
            self.twist.linear.x = 0
            self.twist.linear.y = 0
            self.twist.linear.z = 0
            self.twist.angular.x = 0
            self.twist.angular.y = 0
            self.twist.angular.z = 0
            print("error")

        self.gap_vel_pub.publish(self.twist)




    def theta_filter(self, points_list,imagelines):
        points_1 = []
        points_2 = []
        temp_lines = []
        temp_imagelines = []
        count = 0
        for line in points_list:
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            if np.abs(x1 - x2) > 0.001:
                k = np.abs((y1-y2)/(x1-x2))
            else:
                k = np.Inf
            case_1 = (np.abs(y1 - self.mag_to_cam[1]) <= 0.35 or np.abs(y2 - self.mag_to_cam[1]) <= 0.35)
            case_2 = np.sqrt((y1-y2)**2 + (x1 - x2)**2) >=0.1
            case_3 = (np.min([np.abs(x1),np.abs(x2)]) < 0.2) and ((np.abs(y1 - self.mag_to_cam[1]) <= 0.6 or np.abs(y2 - self.mag_to_cam[1]) <= 0.6))
            case = case_1 or case_3
            if k <= 0.8 and case and case_2:
                temp_lines.append(line)
                points_1.append([line[0],line[1]])
                points_2.append([line[2],line[3]])
                temp_imagelines.append(imagelines[count])
            count = count + 1
        points_1 = np.array(points_1)
        points_2 = np.array(points_2)
        return [temp_lines, points_1, points_2,temp_imagelines]

    def img2ground(self,XY_goal):
        Y = 150
        Z = (self.Fy1 * Y) / (XY_goal[1] - self.PPY1)
        X = -(XY_goal[0] - self.PPX1) / (self.Fx1 / Z)
        X,Y,Z = Z, X,-Y
        return X/ 1000.0,Y/1000.0,Z/1000.0

    def camera_tf2_mag(self,x,y,z):
        x = x + self.mag_to_cam[0]
        y = y + self.mag_to_cam[1]
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
        for line in lines:
            temp_X1,temp_Y1,temp_Z1 = self.img2ground(line[0])
            temp_X1,temp_Y1,temp_Z1 = self.camera_tf2_mag(temp_X1,temp_Y1,temp_Z1)
            temp_X2,temp_Y2,temp_Z2 = self.img2ground(line[1])
            temp_X2,temp_Y2,temp_Z2 = self.camera_tf2_mag(temp_X2,temp_Y2,temp_Z2)
            points_list2.append((temp_X1,temp_Y1,temp_X2,temp_Y2))
        print("points_list2",points_list2)

        lines,points_a,points_b,imagelines = self.theta_filter(points_list2,lines)

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
                [p1x, p1y] = [X, Y]
                [p2x, p2y] = [X2, Y2]
                t_list = np.linspace(0,1,21)
                # p2min = t*(x,y) + (1-t)*(x,y) >= min_dist
                minval_search_list = np.abs(np.hypot(t_list * (p1x - p2x) + p2x, t_list * (p1y - p2y) + p2y) ** 2 - distance_list.min() ** 2)
                temp_index = np.where(minval_search_list == minval_search_list.min())
                t_value = np.int(t_list[np.int(temp_index[0][0])])
                imagelines = imagelines[np.int(index2_1[0])]
                px = t_value * p1x + (1 - t_value) * p2x
                py = t_value * p1y + (1 - t_value) * p2y
                pz = Z
                if np.abs(X2-X) < 0.01:
                    theta = 0
                else:
                    theta = np.arctan((Y-Y2)/np.abs(X-X2))
                if np.sqrt(px**2 + py ** 2) >=3:
                    px = 0.1
                    py = 0
                    pz = -0.15
                    imagelines = []
                result = np.array([0.1, 0, -0.15,theta,px,py,pz,points_a[np.int(index2_1[0])],points_b[np.int(index2_1[0])]])
            else:
                print("no lines")
                result = np.array([0.1, 0, -0.15, 0, 0.1, 0, -0.15, [0.1, 0], [0.1, 0]])
        return [result,imagelines]


if __name__ == '__main__':
    rospy.init_node('line_detector')
    detector = Detector()
    rospy.spin()

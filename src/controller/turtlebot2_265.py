#!/usr/bin/env python
# BEGIN ALL
import functools
import sys
import numpy as np

import rospy
import rosparam
import cv2, cv_bridge
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Int8
from followbot.msg import MGSMeasurement, MGSMeasurements, MGSMarker
import math
from nav_msgs.msg import Odometry
import message_filters
from sensor_msgs.msg import Image, PointCloud2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line
from scipy.spatial import distance
import decimal
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion,quaternion_from_euler
from geometry_msgs.msg import Twist, Pose
import tf2_ros
from geometry_msgs.msg import PoseStamped

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
            # // to an endpoint
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
        min_distance_to_merge = 15
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
        if 45 < orientation < 135:
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
            if 45 < orientation < 135:
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
        self.mgs = 1
        self.lookahead = rospy.get_param('~lookahead')
        self.rate = rospy.get_param('~rate')
        self.goal_margin = rospy.get_param('~goal_margin')

        self.wheel_base = rospy.get_param('~wheel_base')
        self.wheel_radius = rospy.get_param('~wheel_radius')
        self.v_max = rospy.get_param('~v_max')
        self.w_max = rospy.get_param('~w_max')
        self.v1 = self.scale_factor * rospy.get_param('~v', 0.666)  # nominal velocity (1.49 MPH)
        self.v2 = self.scale_factor * rospy.get_param('~v', 0.782)  # nominal velocity (1.75 MPH)
        self.v3 = self.scale_factor * rospy.get_param('~v', 0.849)  # nominal velocity (1.90 MPH)
        self.v4 = self.scale_factor * rospy.get_param('~v', 0.939)  # nominal velocity (2.10 MPH)

        self.current_v = []
        self.current_theta = []

        self.gap_start_tf = []
        rospy.init_node('tf2_turtle_listener')
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        rate = rospy.Rate(10.0)
        rospy.Subscriber("/camera/odom/sample", Odometry, self.odom_callback)
        while not rospy.is_shutdown():
            if self.mgs == 1:
                try:
                    trans = tfBuffer.lookup_transform('camera_odom_frame', 'camera_pose_frame', rospy.Time())
                    self.gap_start_tf = trans

                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    continue


            else:
                try:
                    trans = tfBuffer.lookup_transform('camera_odom_frame', 'camera_pose_frame', rospy.Time())
                    self.straight_line_follower(self.gap_start_tf, trans)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    continue

    def odom_callback(self,msg):

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        theta = msg.twist.twist.angular.z
        self.current_v = [vx,vy]
        self.current_theta = theta


    def find_goal(self, current_p, current_vel, final_p, final_vel, avg_vel, dist_min):
        x0, y0 = np.array(current_p)
        vx0, vy0 = np.array(current_vel)  # vx = vel, vy = 0
        xf, yf = np.array(final_p)
        vxf, vyf = np.array(final_vel)
        if xf ^ 2 + yf ^ 2 >= dist_min:
            x_params = self.cubic_poly_motion_planning(x0, vx0, xf, vxf)
            y_params = self.cubic_poly_motion_planning(y0, vy0, yf, vyf)
            expected_time = self.convolve_and_sum(x_params, y_params, dist_min)
            goal_x = x_params * np.array([1, expected_time, expected_time ^ 2, expected_time ^ 3])
            goal_y = y_params * np.array([1, expected_time, expected_time ^ 2, expected_time ^ 3])
            vx = x_params * np.array([0, 1, 2 * expected_time, 3 * expected_time ^ 2])
            vy = y_params * np.array([0, 1, 2 * expected_time, 3 * expected_time ^ 2])
            if vx > 0:
                theta = np.arctan(vy / vx)
            elif vx == 0:
                theta = np.pi / 2
            else:
                if vy >= 0:
                    theta = np.pi - np.arctan(vy / np.abs(vx))
                else:
                    theta = np.pi + np.arctan(np.abs(vy) / np.abs(vx))
        else:
            goal_x = xf
            goal_y = yf
            if vyf > 0:
                theta = np.arctan(vxf / vyf)
            elif vyf == 0:
                theta = np.pi / 2
            else:
                if vyf >= 0:
                    theta = np.pi - np.arctan(vyf / np.abs(vxf))
                else:
                    theta = np.pi + np.arctan(np.abs(vyf) / np.abs(vxf))
        goal_pose = [goal_x, goal_y, theta]
        v, w = self.pure_persuit(goal_pose)  # to be changed with proportional control
        self.twist.linear.x = v
        self.twist.angular.z = w
        return self.twist

    def cubic_poly_motion_planning(self, currentp, current_vel, finalp, final_vel, avg_vel):
        p0 = np.array(currentp)
        v0 = np.array(current_vel)  # vx = vel, vy = 0
        pf = np.array(finalp)
        vf = np.array(final_vel)
        temp_vec = np.array(finalp - currentp)
        dt = np.dot(temp_vec, temp_vec) / avg_vel
        a0 = p0
        a1 = v0
        M = np.array([[dt ^ 2, dt ^ 3], [2 * dt, 3 * dt ^ 2]])
        a2, a3 = np.inv(M) * np.array([[pf - a0 - dt * a1], [vf - a1]])
        return [a0, a1, a2, a3]

    def convolve_and_sum(self, x_a, y_a, dist_min):
        x_a_convolve = np.convolve(x_a, x_a)
        y_a_convolve = np.convolve(y_a, y_a)
        poly_param = x_a_convolve + y_a_convolve
        poly_param[-1] = poly_param[-1] - dist_min
        approximate_time = np.roots(poly_param)
        temp = []
        for item in approximate_time:
            if item > 0:
                temp.append(item)
        return np.min(temp)

    def pure_persuit(self, goal_pose):
        # pure_persuit is under robot frame.
        [gx, gy, gtheta] = goal_pose
        dist = np.sqrt(gx ** 2 + gy ** 2)
        fai = np.arcsin(gy / dist)
        case_label = 0
        if fai > 0:
            if gtheta > fai and gtheta <= np.pi / 2 - fai:
                R = (dist / 2) / np.sin(fai / 2)
                v_cmd = self.v_turn
                w_cmd = v_cmd / R
                case_label = 1
            elif gtheta >= np.pi / 2 - fai and gtheta <= np.pi / 2 + fai:
                v_cmd = 0
                w_cmd = self.max_turning_omega
                case_label = 2
            else:
                R = dist / 2
                v_cmd = self.v_turn
                w_cmd = 0.5 * fai / (dist / self.v_turn)
                case_label = 3
        else:
            if gtheta >= 3 * np.pi / 2 and gtheta < 2 * np.pi - np.abs(fai):
                R = (dist / 2) / np.abs(np.sin(fai / 2))
                v_cmd = self.v_turn
                w_cmd = -v_cmd / R
                case_label = 4
            elif gtheta > 3 * np.pi / 2 - np.abs(fai) and gtheta <= 3 * np.pi / 2:
                v_cmd = 0
                w_cmd = -self.max_turning_omega
                case_label = 5
            else:
                R = dist / 2
                v_cmd = self.v_turn
                w_cmd = 0.5 * fai / (dist / self.v_turn)
                case_label = 6
        print("gx,gy", [gx, gy], "fai", fai, "gtheta", gtheta, "case_label", case_label)
        L = self.wheel_base
        u = max(v_cmd + L * w_cmd * 1, v_cmd - L * w_cmd * 1)
        if u > self.v4:
            v_cmd = v_cmd * (self.v4 / u)
            w_cmd = w_cmd * (self.v4 / u)
        return (v_cmd, w_cmd, case_label)

    def straight_line_follower(self,tf_gap,tf_now):



class Detector:
    def __init__(self):
        self.twist = Twist()
        self.bridge = cv_bridge.CvBridge()
        self.img_sub_1 = message_filters.Subscriber('camera/fisheye1/image_raw', Image)
        self.img_sub_2 = message_filters.Subscriber('camera/fisheye2/image_raw', Image)
        # self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)
        self.measurements = message_filters.ApproximateTimeSynchronizer([self.img_sub_1, self.img_sub_2], queue_size=5,
                                                                        slop=0.1)
        self.measurements.registerCallback(self.measurements_callback)
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
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
        self.detection_pub = rospy.Publisher("/detected_line",PoseStamped,queue_size=1)

    def measurements_callback(self, img1, img2):
        cv_image1 = self.bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        cv_image2 = self.bridge.imgmsg_to_cv2(img2, desired_encoding='bgr8')
        img_undistorted = cv2.fisheye.undistortImage(cv_image1, self.K1, D=self.D1, Knew=self.Knew1)
        img_undistorted = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)

        # https://github.com/smidm/opencv-python-fisheye-example/blob/master/fisheye_example.py
        img_undistorted = cv2.GaussianBlur(img_undistorted, (5, 5), 1)
        # cv2.imshow("Image window", img_undistorted)
        # cv2.waitKey(0)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(img_undistorted, kernel, iterations=1)
        (thresh, im_bw) = cv2.threshold(erosion, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        edges = cv2.Canny(erosion, 128, 250)
        temp_lines = probabilistic_hough_line(edges, threshold=20, line_length=40, line_gap=10)

        new_lines = []
        for line in temp_lines:
            p0, p1 = line
            if p0[1] > self.Fy1 and p1[1] > self.Fy1:
                new_lines.append((p0[0], p0[1], p1[0], p1[1]))

        a = HoughBundler()
        foo = a.process_lines(new_lines, edges)
        goal = self.XY_2_XYZ_Goal(foo, img_undistorted)
        target_pose = PoseStamped()
        target_pose.pose.position.x = goal[0]
        target_pose.pose.position.y = goal[1]
        target_pose.pose.position.z = goal[2]
        target_pose.pose.orientation.x = quaternion_from_euler(0, 0, yaw)

        self.detection_pub()
        print("goal", goal)
        # [vx, w] = self.goal_pursuit(goal)
        # self.twist.linear.x = vx
        # self.twist.angular.z = w
        # self.cmd_vel_pub.publish(self.twist)


        # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 20),
        #                          sharex=True, sharey=True)
        # ax = axes.ravel()
        #
        # ax[0].imshow(cv_image1, cmap='gray', vmin=0, vmax=255)
        # ax[0].axis('off')
        # ax[0].set_title('fisheye', fontsize=20)
        #
        # ax[1].imshow(img_undistorted, cmap='gray', vmin=0, vmax=255)
        # ax[1].axis('off')
        # ax[1].set_title('img_undistorted', fontsize=20)
        #
        # ax[2].imshow(im_bw, cmap='gray', vmin=0, vmax=255)
        # ax[2].axis('off')
        # ax[2].set_title('erosion', fontsize=20)
        #
        # ax[3].imshow(edges, cmap='gray', vmin=0, vmax=255)
        # ax[3].axis('off')
        # ax[3].set_title('edges', fontsize=20)
        #
        # ax[3].imshow(edges, cmap='gray', vmin=0, vmax=255)
        # ax[3].axis('off')
        # ax[3].set_title('edges', fontsize=20)
        #
        # ax[4].imshow(edges * 0, cmap='gray')
        # for line in temp_lines:
        #     p0, p1 = line
        #     ax[4].plot((p0[0], p1[0]), (p0[1], p1[1]))
        # ax[4].set_xlim((0, img_undistorted.shape[1]))
        # ax[4].set_ylim((img_undistorted.shape[0], 0))
        # ax[4].set_title('Probabilistic Hough')
        #
        # ax[5].imshow(edges * 0, cmap='gray')
        # for line in foo:
        #     p0, p1 = line
        #     ax[5].plot((p0[0], p1[0]), (p0[1], p1[1]))
        #     ax[5].plot([p0[0], p1[0]], [p0[1], p1[1]], 'bo')
        # print("foo[0]", foo[0])
        # ax[5].set_xlim((0, img_undistorted.shape[1]))
        # ax[5].set_ylim((img_undistorted.shape[0], 0))
        # ax[5].set_title('Merged Probabilistic Hough')
        #
        # fig.tight_layout()
        # plt.show()
        # cv2.imshow("Image window", skeleton)
        # cv2.waitKey(0)

    def theta_filter(self, lines):
        temp_lines = []
        for line in lines:
            if line[1][0] - line[0][0] > 0.00001:
                k = (line[0][1] - line[1][1]) / (line[1][0] - line[0][0])
            else:
                k = np.Inf
            print("line", line, "k", k)
            if np.abs(k) >= 1.73:
                temp_lines.append(line)
        return temp_lines

    def XY_2_XYZ_Goal(self, lines, img_undistorted):
        height, width = img_undistorted.shape
        center_point = [(np.floor(width / 2), height - 1)]
        points_list = []
        lines = self.theta_filter(lines)
        if len(lines) == 0:
            result = np.array([0, 0.15, 1])
        else:
            for line in lines:
                points_list.append(line[0])
                points_list.append(line[1])
            DIST = distance.cdist(np.array(points_list), np.array(center_point))
            index = np.where(DIST == DIST.min())
            if np.int(index[0])%2 == 0:
                the_other_point = points_list[np.int(index[0]) + 1]
            else:
                the_other_point = points_list[np.int(index[0]) - 1]

            # print("lines", lines)
            # print("DIST",DIST,"index",index,"center_point",center_point)
            XY_goal = points_list[np.int(index[0])]
            # print("XY_goal", XY_goal)
            Y = 150
            # print("self.Fy1/Y", self.Fy1 * Y, "XY_goal[1]-self.PPY1", XY_goal[1] - self.PPY1)
            Z = (self.Fy1 * Y) / (XY_goal[1] - self.PPY1)
            # print("Z", Z)
            X = -(XY_goal[0] - self.PPX1) / (self.Fx1 / Z)
            result = np.array([X, Y, Z]) / 1000.0

            XY_goal2 = the_other_point
            Y = 150
            Z = (self.Fy1 * Y) / (XY_goal[1] - self.PPY1)
            X = -(XY_goal[0] - self.PPX1) / (self.Fx1 / Z)
            result2 = np.array([X, Y, Z]) / 1000.0



        return result


if __name__ == '__main__':
    rospy.init_node('line_detector')
    detector = Detector()
    rospy.spin()

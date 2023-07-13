#!/usr/bin/env python
# BEGIN ALL
import numpy as np
import cv2,cv_bridge
import rospy
import math
from sensor_msgs.msg import BatteryState,Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point,Pose, PoseStamped
from tf.transformations import euler_from_quaternion
import copy
import time
from roboteq_motor_controller_driver.msg import channel_values
from nav_msgs.msg import Odometry,Path

class PurePursuit:
    def __init__(self, path, look_ahead):
        self.path = path
        self.look_ahead = look_ahead

    def get_steering(self, robot_position):
        # Find the point on the path that's look_ahead distance from the robot
        for i in range(len(self.path) - 1):
            # Get the current point and the next point on the path
            current_point = self.path[i]
            next_point = self.path[i+1]

            # See if the line between current_point and next_point intersects with the look_ahead circle
            d = next_point - current_point
            f = current_point - robot_position

            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - self.look_ahead**2

            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                continue

            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2*a)
            t2 = (-b + discriminant) / (2*a)

            if 0 <= t1 <= 1:
                return current_point + d * t1
            if 0 <= t2 <= 1:
                return current_point + d * t2

        # If there's no intersection point, return the last point in the path
        return self.path[-1]


class PDController:
    def __init__(self, P, D):
        self.P = 1
        self.D = 0.5
        self.previous_error = 0

    def calculate(self, error, dt):
        derivative = (error - self.previous_error) / dt
        output = self.P * error + self.D * derivative
        self.previous_error = error
        return output


class pathPlanning():
    def __init__(self):
        self.usingTurtlebot = 1

        self.v0 = 0
        self.w0 = 0
        self.v_max = 1
        self.w_max = 0.5
        self.v = copy.deepcopy(self.v0)
        self.w = copy.deepcopy(self.w0)

        self.mag_to_cam = [5 * 0.0254, -1 * 0.0254, 0]
        self.cam_to_center = [-26 * 0.0254, -1 * 0.0254]
        self.wheel_base = 0.8
        self.wheel_radius = 0.2

        self.driftSign = 0  # 0 is off, 1 is on with flow optical flow
        self.lastMagneticDetection = 0
        self.lastZedPoseWhenLoseMagnetic = Pose()
        self.currentZedPose = Pose()
        self.controllers = {
            'orientation': PDController(1, 0.5),
            'position': PDController(1, 0.5)
        }
        self.dt = 1 # PD controller parameter
        self.zedOdomSubscriber = rospy.Subscriber('/zed/zed_node/pose', PoseStamped, self.pose_callback)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.lidar_front_detection_sub = rospy.Subscriber('lidar_front',channel_values,self.lidar_front_callback,queue_size = 5)
        self.lidar_back_detection_sub = rospy.Subscriber('lidar_back',channel_values,self.lidar_back_callback,queue_size = 5)
        self.kick_signal_sub = rospy.Subscriber("/kick_signal", channel_values, self.kick_signal_callback)
        self.straight_line_patrol_sub = rospy.Subscriber('/mag_marker_detect',channel_values,self.patrol_callback,queue_size = 5)
        self.mag_detection_sub = rospy.Subscriber('/mag_track_detect',channel_values,self.mag_detection_callback,queue_size = 5)

        if self.usingTurtlebot:
            self.odomSub = rospy.Subscriber('/zed/zed_node/odom', Odometry, self.odom_callback, queue_size=5)
        else:
            if self.front_kick_val == 1:
                assert self.back_kick_val == 0
                self.odom_B_sub = rospy.Subscriber('/zedB/zed_node_B/odom', Odometry, self.odom_callback,queue_size=5)
            if self.back_kick_val == 1:
                assert self.front_kick_val == 0
                self.odom_A_sub = rospy.Subscriber('/zedA/zed_node_A/odom', Odometry, self.odom_callback,queue_size=5)

        # while not rospy.is_shutdown():
        #     if self.mgs == 1:
        #         try:
        #             self.trans = tfBuffer.lookup_transform('camera_odom_frame', 'camera_pose_frame', rospy.Time())
        #             self.theta = euler_from_quaternion([self.trans.transform.rotation.x,
        #                                                 self.trans.transform.rotation.y,
        #                                                 self.trans.transform.rotation.z,
        #                                                 self.trans.transform.rotation.w])[2]
        #             self.last_trans = self.trans
        #             self.last_theta = self.theta
        #         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #             continue
        #
        #     else:
        #         try:
        #             self.trans = tfBuffer.lookup_transform('camera_odom_frame', 'camera_pose_frame', rospy.Time())
        #             self.theta = euler_from_quaternion([self.trans.transform.rotation.x,
        #                                                 self.trans.transform.rotation.y,
        #                                                 self.trans.transform.rotation.z,
        #                                                 self.trans.transform.rotation.w])[2]
        #         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #             continue
    def kick_front_callback(self,msg):
        if msg.value[0] == 1:
            self.front_kick_val = 1
            self.back_kick_val = 0

    def kick_back_callback(self,msg):
        if msg.value[0] == 1:
            self.front_kick_val = 0
            self.back_kick_val = 1

    def kick_signal_callback(self,msg):
        if msg.value[0] == 1:
            self.front_kick_val = 1
            self.back_kick_val = 0
        elif msg.value[1] == 1:
            self.front_kick_val = 0
            self.back_kick_val = 1
    def mag_detection_callback(self,msg):
        if msg.value[0] == 1 or msg.value[1] == 1:
            self.mgs = 1
        else:
            self.mgs = 0
    def odom_callback(self, msg):
        if self.lastMagneticDetection == 1 and self.mgs == 0:
            self.lastZedPoseWhenLoseMagnetic = msg
            self.lastMagneticDetection = 0
            self.currentZedPose = msg
    def patrol_callback(self,msg):
        if self.mgs == 0 and self.driftSign:
            yaw, desiredYaw = self.convertPose2Yaw(msg)
            control_orientation = self.controllers['orientation'].calculate(desiredYaw - yaw, self.dt)
            pStart, pRobot = self.convertPose2Translation(msg)
            distance = self.distance_point_to_line(pStart, desiredYaw, pRobot)
            control_position = self.controllers['position'].calculate(distance, self.dt)
            self.convertError2Velocity(control_orientation, control_position)

    def convertPose2Yaw(self, msgInPose):
        (_, _, yaw) = euler_from_quaternion(msgInPose.transform.rotation)
        (_, _, desiredYaw) = euler_from_quaternion(self.lastZedPoseWhenLoseMagnetic.transform.rotation)
        return yaw, desiredYaw

    def convertPose2Translation(self, msgInPose):
        x1 = self.lastZedPoseWhenLoseMagnetic.transform.translation.x
        y1 = self.lastZedPoseWhenLoseMagnetic.transform.translation.y
        z1 = self.lastZedPoseWhenLoseMagnetic.transform.translation.z
        P1 = np.array([x1, y1, z1])

        x2 = msgInPose.transform.translation.x
        y2 = msgInPose.transform.translation.y
        z2 = msgInPose.transform.translation.z
        P2 = np.array([x2, y2, z2])
        return P1, P2

    def distance_point_to_line(self, P1, yaw_angle, P2):
        # Convert yaw angle to direction vector
        v = np.array([np.cos(yaw_angle), np.sin(yaw_angle), 0])
        # Calculate the distance from P2 to the line
        distance = np.linalg.norm(np.cross(P2 - P1, v)) / np.linalg.norm(v)
        return distance

    def convertError2Velocity(self, control_orientation, control_position):
        # orientation is same between the camera and robot body

        return control_orientation, control_position
    def vAndwNormalization(self, v, w):
        self.v = v
        self.w = w
class Detector:
    def __init__(self):
        self.PPX1 = 665.465
        self.PPY1 = 371.953
        self.Fx1 = 700.819
        self.Fy1 = 700.819
        self.K1 = np.array([[self.Fx1, 0., self.PPX1],
                            [0., self.Fy1, self.PPY1],
                            [0., 0., 1.]])
        self.Knew1 = self.K1.copy()
        self.Knew1[(0, 1), (0, 1)] = 1 * self.Knew1[(0, 1), (0, 1)]
        self.R = np.eye(3)
        self.mag_to_cam = [5 * 0.0254, -1 * 0.0254, 0]

        self.lineWidth = 0.1 # only remove the detection significantly larger than the width of the magnetic tape

        self.img_A_sub = rospy.Subscriber("/zed/zed_node/left_raw/image_raw_color", Image, self.measurements_callback,
                                     queue_size=5)
        self.img_B_sub = rospy.Subscriber("/zedB/zed_node_B/left_raw/image_raw_color", Image, self.measurements_callback,
                                     queue_size=5)
        self.linePublisher = rospy.Publisher("/markerPointPub", MarkerArray, queue_size=10)
        self.detectionGoalPublisher = rospy.Publisher("/detectedSegment",Marker, queue_size=10)
    def measurements_callback(self, img1):
        image = cv_bridge.CvBridge().imgmsg_to_cv2(img1, desired_encoding='bgr8')
        # imageD0 = cv2.pyrDown(image)
        imageD = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageD[0:int(self.PPY1) + 16, :] = 1
        kernel1 = np.ones((5, 5), np.uint8)
        im_bw2 = cv2.erode(imageD, kernel1)
        thresh = cv2.adaptiveThreshold(im_bw2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        min_size = 25  # minimum size of the component (in pixels)
        max_size = 6000  # maximum size of the component (in pixels)
        y_threshold_min = 100  # Adjust this value to your needs
        y_threshold_max = 1000  # Adjust this value to your needs

        # Create a new image to store the filtered components
        filtered_img = np.zeros_like(labels)
        # Loop through the connected components
        for i in range(1, num_labels):
            # If the size of the connected component is within the desired range
            if min_size <= stats[i, cv2.CC_STAT_AREA] <= max_size and stats[i, cv2.CC_STAT_WIDTH] < stats[i, cv2.CC_STAT_HEIGHT]:
                # Add this component to the filtered image
                filtered_img[labels == i] = 255
                left = stats[i, cv2.CC_STAT_LEFT]
                top = stats[i, cv2.CC_STAT_TOP]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                # The area (in pixels)
                area = stats[i, cv2.CC_STAT_AREA]

        filtered_img_uint8 = filtered_img.astype(np.uint8)
        filtered_color = cv2.cvtColor(filtered_img_uint8, cv2.COLOR_GRAY2BGR)
        filtered_color = cv2.resize(filtered_color, (image.shape[1], image.shape[0]))
        blended = cv2.addWeighted(image, 0.5, filtered_color, 0.5, 0)

        self.showInMovedWindow('video_image2', blended, 600, 800)
        cv2.waitKey(3)

    def convertPattern2Point(self,i, labels_i, stats, centroids):
        # label_i is the binary image
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        # The centroid
        centroidX, centroidY = centroids[i]
        centroidWidth = self.find_balance_point(labels_i[centroidY])
        lowQuarterY = np.round(centroidY - height / 4)
        upQuarterY = np.round(centroidY + height / 4)

        # Estimate the line width at lower-quarter, middle and upper quarter
        lowQuarterWidth = np.sum(labels_i[lowQuarterY])
        upQuarterWidth = np.sum(labels_i[upQuarterY])
        lowQuarterX = self.find_balance_point(labels_i[lowQuarterY])
        upQuarterX = self.find_balance_point(labels_i[upQuarterY])
        [filterSign, pLowLeft, pLowRight, pCenLeft, pCenRight, pUpLeft, pUpRight] = self.XY2XYZ(lowQuarterX,
            lowQuarterY, lowQuarterWidth, upQuarterX, upQuarterY, upQuarterWidth, centroidX, centroidY, centroidWidth)


    def find_balance_point(self, arr):
        # Calculate the cumulative sums from left to right and from right to left
        cumsum_left_to_right = np.cumsum(arr)
        cumsum_right_to_left = np.cumsum(arr[::-1])[::-1]
        # Find the first index where the two cumulative sums are equal
        balance_points = np.where(cumsum_left_to_right == cumsum_right_to_left)[0]
        # If there is at least one balance point, return the first one; otherwise, return None
        return balance_points[0] if balance_points.size > 0 else None

    def XY2XYZ(self, lowQuarterX, lowQuarterY, lowQuarterWidth, upQuarterX, upQuarterY, upQuarterWidth,
               centroidX, centroidY, centroidWidth):
        # convert pixel to fred local coordinate
        xLowLeft, yLowLeft, zLowLeft = self.img2ground([lowQuarterX-lowQuarterWidth/2,lowQuarterY])
        xLowRight, yLowRight, zLowRight = self.img2ground([lowQuarterX + lowQuarterWidth / 2, lowQuarterY])
        xCenLeft, yCenLeft, zCenLeft = self.img2ground([centroidX - centroidWidth / 2, centroidY])
        xCenRight, yCenRight, zCenRight = self.img2ground([centroidX + centroidWidth / 2, centroidY])
        xUpLeft, yUpLeft, zUpLeft = self.img2ground([upQuarterX - upQuarterWidth / 2, upQuarterY])
        xUpRight, yUpRight, zUpRight = self.img2ground([upQuarterX + upQuarterWidth / 2, upQuarterY])
        pLowLeft = np.array([xLowLeft, yLowLeft, zLowLeft])
        pLowRight = np.array([xLowRight, yLowRight, zLowRight])
        pCenLeft = np.array([xCenLeft, yCenLeft, zCenLeft])
        pCenRight = np.array([xCenRight, yCenRight, zCenRight])
        pUpLeft = np.array([xUpLeft, yUpLeft, zUpLeft])
        pUpRight = np.array([xUpRight, yUpRight, zUpRight])
        pNull = np.array([None,None,None])

        distanceLow = np.linalg.norm(pLowLeft - pLowRight)
        distanceCen = np.linalg.norm(pCenLeft - pCenRight)
        distanceUp = np.linalg.norm(pUpLeft - pUpRight)

        if self.line_width_filter(distanceLow, distanceCen, distanceUp):
            return [True, pLowLeft, pLowRight, pCenLeft, pCenRight, pUpLeft, pUpRight]
        else:
            return [False,pNull, pNull, pNull, pNull, pNull, pNull]
    def line_width_filter(self,distanceLow, distanceCen, distanceUp):
        if distanceLow <= self.lineWidth and distanceCen <= self.lineWidth and  distanceUp <= self.lineWidth:
            return True
        else:
            return False
    def img2ground(self, XY_goal):
        Y = 150
        Z = (self.Fy1 * Y) / (XY_goal[1] - self.PPY1)
        X = -(XY_goal[0] - self.PPX1) / (self.Fx1 / Z)
        X, Y, Z = Z, X, -Y
        return X / 1000.0, Y / 1000.0, Z / 1000.0


    def detectionGoal(self,points):
        # points is a 3-by-2 np.array(), edge points and middle points
        markerArray = MarkerArray()
        markerArray.header.stamp = rospy.get_rostime()
        for i in range(points):
            marker = Marker()
            marker.header.frame_id = "/fred_local"
            marker.id = i
            marker.type = marker.POINTS
            marker.action = marker.ADD

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0

            pi = Point(); pi.x = points[i][0]; pi.y = points[i][1]
            markerArray.append(pi)
        self.linePublisher.publish(markerArray)
    def showInMovedWindow(self,winname, img, x, y):
        cv2.namedWindow(winname)  # Create a named window
        cv2.moveWindow(winname, x, y)  # Move it to (x,y)
        cv2.imshow(winname, img)
def measurements_callback(img1):
    cv_image1 = cv_bridge.CvBridge().imgmsg_to_cv2(img1, desired_encoding='bgr8')
    img_undistorted2 = cv2.pyrDown(cv_image1)
    cv2.imshow('image', img_undistorted2)
    cv2.waitKey(0)

if __name__ == '__main__':
    rospy.init_node('line_detector')
    detector = Detector()
    rospy.spin()

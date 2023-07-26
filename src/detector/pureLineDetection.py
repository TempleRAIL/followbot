#!/usr/bin/env python
# BEGIN ALL
import numpy as np
import cv2,cv_bridge
import rospy
import math
from sensor_msgs.msg import BatteryState,Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
import copy
import time
# from roboteq_motor_controller_driver.msg import channel_values
from nav_msgs.msg import Odometry,Path
from image_geometry import PinholeCameraModel

class Detector:
    def __init__(self,usingTurtlebot):
        self.ratioChangeDistance = 1 #Tune this parameter to change the exact distance self.img2ground(XY_goal)

        self.Fx1 = 667.2269897460938
        self.Fy1 = 667.2269897460938
        self.PPX1 = 610.2620239257812
        self.PPY1 = 365.79400634765625
        self.K1 = np.array([[self.Fx1, 0., self.PPX1],
                            [0., self.Fy1, self.PPY1],
                            [0., 0., 1.]])
        self.Knew1 = self.K1.copy()
        self.Knew1[(0, 1), (0, 1)] = 1 * self.Knew1[(0, 1), (0, 1)]
        self.R = np.eye(3)
        self.plane_threshold = 0.6
        self.minDepth = 0.4
        self.focal_length_x = 667.1478881835938
        self.focal_length_y = 667.1478881835938
        self.optical_center_x = 610.6473388671875
        self.optical_center_y = 365.609619140625
        self.depthK = np.array([[self.focal_length_x, 0., self.optical_center_x],
                            [0., self.focal_length_y, self.optical_center_y],
                            [0., 0., 1.]])
        self.depthMask = np.full((720, 1280), False)
        self.depth_sub = rospy.Subscriber('/zed/zed_node/depth/depth_registered', Image, self.depth_callback,
                                        queue_size=5)
        self.lineWidth = 0.15 # only remove the detection significantly larger than the width of the magnetic tape
        if usingTurtlebot:
            self.mag_to_cam = [0, 0, 0]
            self.plane_params = np.array([-1, -1, 1, -0.23])
            self.img_sub = rospy.Subscriber('/zed/zed_node/left/image_rect_color', Image, self.measurements_callback,
                                            queue_size=5)
        else:
            # FRED case
            self.mag_to_cam = [5 * 0.0254, -1 * 0.0254, 0]
            self.plane_params = np.array([0, 0, 1, -0.23]) # height value is same with turtlebot
            self.img_A_sub = rospy.Subscriber("/zed/zed_node/left/image_rect_color", Image, self.measurements_callback,
                                         queue_size=5)
            self.img_B_sub = rospy.Subscriber("/zed/zed_node/left/image_rect_color", Image, self.measurements_callback,
                                         queue_size=5)
        self.linePublisher = rospy.Publisher("/markerPointPub", MarkerArray, queue_size=10)
        self.detectionGoalPublisher = rospy.Publisher("/detectedSegment",Marker, queue_size=10)
    def measurements_callback(self, img1):
        image = cv_bridge.CvBridge().imgmsg_to_cv2(img1, desired_encoding='bgr8')
        # imageD0 = cv2.pyrDown(image)
        imageD = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageD[0:int(self.PPY1) + 16, :] = 1
        kernel1 = np.ones((3, 3), np.uint8)
        im_bw2 = cv2.erode(imageD, kernel1)
        thresh = cv2.adaptiveThreshold(im_bw2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh[int(self.PPY1)+16: int(self.PPY1)+16*10: 24, :] = 0
        thresh[int(self.PPY1) + 16 * 10:int(self.PPY1) + 16 * 21:48, :] = 0
        # cv2.imshow('Thin Parts', thresh)
        # cv2.waitKey(1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        min_size = 20  # minimum size of the component (in pixels)
        min_size2 = 40
        max_size = 2000  # maximum size of the component (in pixels)
        max_size2 = 12000
        # Create a new image to store the filtered components
        filtered_img = np.zeros_like(labels)
        # Loop through the connected components
        potentialLabelList = []
        for i in range(1, num_labels):
            # If the size of the connected component is within the desired range
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # If the magnetic tape is in parallelogram, need a correction ratio for width/height in threshold
            ratio = area/(height * width)
            if min_size <= area <= max_size and width*ratio < 1.2 * height:
                # Add this component to the filtered image
                filtered_img[labels == i] = 255
                potentialLabelList.append([i])

        filtered_img_uint8 = filtered_img.astype(np.uint8)
        kernel2 = np.ones((5, 5), np.uint8)
        final_image = cv2.dilate(filtered_img_uint8, kernel2,iterations=2)
        kernel3 = np.array([[1],[1],[1],[1],[1]], dtype=np.uint8)
        final_image = cv2.dilate(final_image, kernel3, iterations=2)

        # cv2.imshow('Thin Parts', final_image)
        # cv2.waitKey(1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_image, 4, cv2.CV_32S)
        filtered_color = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)
        filtered_color = cv2.resize(filtered_color, (image.shape[1], image.shape[0]))
        blended = cv2.addWeighted(image, 0.5, filtered_color, 0.5, 0)
        # self.showInMovedWindow('video_image2', blended, 600, 800)
        # cv2.waitKey(1)
        scoreArea = np.zeros(len(range(num_labels)))
        scoreLengthWidthRatio = np.zeros(len(range(num_labels)))
        filtered_img2 = np.zeros_like(labels)
        for i in range(1,num_labels):
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            ratio = area / (height * width)
            depthMaskArea = np.sum(labels[self.depthMask] == i)
            if min_size2 <= area <= max_size2 and 2*width*ratio < height:
                # Add this component to the filtered image
                # filtered_img2[labels == i] = 255
                temp_result = self.convertPattern2Point(i, labels, stats, centroids)
                if temp_result[0]:
                    # filtered_img2[labels == i)] = 255
                    scoreArea[i] = depthMaskArea

                    pLowLeft = temp_result[1]
                    pUpLeft = temp_result[5]
                    pLowRight = temp_result[2]
                    pUpRight = temp_result[6]
                    length = np.linalg.norm(pLowLeft-pUpLeft) + np.linalg.norm(pLowRight-pUpRight)
                    scoreLengthWidthRatio[i] = length
                # else:
                #     left = stats[i,cv2.CC_STAT_LEFT]
                #     top = stats[i,cv2.CC_STAT_TOP]
                #     # cv2.rectangle(filtered_img2, (left,top-height), (left + width, top), (0, 255, 0), 2)
                #     # filtered_img2[top:top+ height, left:left + width] = 127
                #     print("[left ,top], [left + width, top+ height]",[left,top], [left + width,top+ height])
                #     print(self.img2ground([left, top]),self.img2ground([left + width, top+height]))
        mostProbablePatternIndex = np.argmax(0.5*scoreArea/np.sum(scoreArea)+0.5*scoreLengthWidthRatio/np.sum(scoreLengthWidthRatio))
        # print(score, mostProbablePatternIndex, "mostProbablePatternIndex")
        if mostProbablePatternIndex != 0:
            goal_position_list = self.convertPattern2Point(mostProbablePatternIndex, labels, stats, centroids)
            filtered_img2[labels == mostProbablePatternIndex] = 255
            goal_position = (goal_position_list[1] + goal_position_list[2])/2
            print("goal_position",goal_position)
        else:
            goal_position = [0.1, 0, 0, 0]
        filtered_img_uint8 = filtered_img2.astype(np.uint8)
        final_image = filtered_img_uint8
        filtered_color = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)
        filtered_color = cv2.resize(filtered_color, (image.shape[1], image.shape[0]))
        blended = cv2.addWeighted(image, 0.5, filtered_color, 0.5, 0)

        self.showInMovedWindow('video_image2', blended, 600, 800)
        cv2.waitKey(1)
        return goal_position
    def convertPattern2Point(self, i, labels, stats, centroids):
        # label_i is the binary image
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        # The centroid
        centroidX, centroidY = centroids[i]
        coords = np.argwhere(labels)
        medianY = centroidY
        minMedianX, maxMedianX = self.findEdgePoints(coords, medianY,left,top,width,height)
        upQuarterY = int(centroidY - height / 4)
        minUpQuarterX, maxUpQuarterX = self.findEdgePoints(coords, upQuarterY, left,top,width,height)
        lowQuarterY = int(centroidY + height / 4)
        minLowQuarterX, maxLowQuarterX = self.findEdgePoints(coords, lowQuarterY,left,top,width,height)

        [filterSign, pLowLeft, pLowRight, pCenLeft, pCenRight, pUpLeft, pUpRight] = self.XY2XYZ(minMedianX,
            maxMedianX, medianY, minUpQuarterX, maxUpQuarterX, upQuarterY, minLowQuarterX, maxLowQuarterX, lowQuarterY)
        if filterSign:
            # print("pLowLeft, pLowRight, pCenLeft, pCenRight, pUpLeft, pUpRight")
            # print(pLowLeft, pLowRight, pCenLeft, pCenRight, pUpLeft, pUpRight)
            # print([minMedianX,maxMedianX, medianY], [minLowQuarterX, maxLowQuarterX, lowQuarterY],
            #       [minUpQuarterX, maxUpQuarterX, upQuarterY])
            return [True,pLowLeft, pLowRight, pCenLeft, pCenRight, pUpLeft, pUpRight ]
        else:
            return [False,pLowLeft, pLowRight, pCenLeft, pCenRight, pUpLeft, pUpRight]

    def findEdgePoints(self, coords, median_y, left, top, width, height):
        median_y = int(round(median_y))
        median_row_coords = coords[coords[:, 0] == median_y]
        min_x = np.min(median_row_coords[:, 1])
        max_x = np.max(median_row_coords[:, 1])
        if min_x < left:
            min_x = left
        if max_x > left + width:
            max_x = left + width
        return min_x, max_x
    def XY2XYZ(self, minMedianX,maxMedianX, medianY, minLowQuarterX, maxLowQuarterX, lowQuarterY, minUpQuarterX,
               maxUpQuarterX, upQuarterY):
        # convert pixel to fred local coordinate
        xLowLeft, yLowLeft, zLowLeft = self.img2ground([minLowQuarterX, lowQuarterY])
        xLowRight, yLowRight, zLowRight = self.img2ground([maxLowQuarterX, lowQuarterY])
        xCenLeft, yCenLeft, zCenLeft = self.img2ground([minMedianX, medianY])
        xCenRight, yCenRight, zCenRight = self.img2ground([maxMedianX, medianY])
        xUpLeft, yUpLeft, zUpLeft = self.img2ground([minUpQuarterX, upQuarterY])
        xUpRight, yUpRight, zUpRight = self.img2ground([maxUpQuarterX, upQuarterY])
        pLowLeft = np.array([xLowLeft, yLowLeft, zLowLeft])
        pLowRight = np.array([xLowRight, yLowRight, zLowRight])
        pCenLeft = np.array([xCenLeft, yCenLeft, zCenLeft])
        pCenRight = np.array([xCenRight, yCenRight, zCenRight])
        pUpLeft = np.array([xUpLeft, yUpLeft, zUpLeft])
        pUpRight = np.array([xUpRight, yUpRight, zUpRight])
        pNull = np.array([None, None, None])

        distanceLow = np.linalg.norm(pLowLeft - pLowRight)
        distanceCen = np.linalg.norm(pCenLeft - pCenRight)
        distanceUp = np.linalg.norm(pUpLeft - pUpRight)

        if self.line_width_filter(distanceLow, distanceCen, distanceUp):
            return [True, pLowLeft, pLowRight, pCenLeft, pCenRight, pUpLeft, pUpRight]
        else:
            return [False, pNull, pNull, pNull, pNull, pNull, pNull]
    def line_width_filter(self,distanceLow, distanceCen, distanceUp):
        if np.mean([distanceLow, distanceCen, distanceUp]) <= self.lineWidth:
            return True
        else:
            return False
    def img2ground(self, XY_goal):
        Y = 1000*np.abs(self.plane_params[-1])
        Z = (self.Fy1 * Y) / (XY_goal[1] - self.PPY1)
        X = -(XY_goal[0] - self.PPX1) / (self.Fx1 / Z)
        X, Y, Z = self.ratioChangeDistance * Z, self.ratioChangeDistance * X, -Y
        return X / 1000.0, Y / 1000.0, Z / 1000.0
    def distance_between_lines(self, line1, line2):
        # Calculate the distance between two lines
        x_diff = line1[0] - line2[0]
        y_diff = line1[1] - line2[1]
        return np.sqrt(x_diff ** 2 + y_diff ** 2)

    def depth_callback(self, data):
        # depth_image = cv_bridge.CvBridge().imgmsg_to_cv2(data).copy()  # convert ROS Image message to OpenCV image
        depth_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, desired_encoding='passthrough').copy()

        # Handle depth_image's NaN, inf, and neginf values
        depth_image[np.isnan(depth_image)] = self.plane_params[-1]**2
        depth_image[np.isinf(depth_image)] = self.plane_params[-1]**2
        depth_image[np.isneginf(depth_image)] = self.plane_params[-1]**2

        # Compute filtered_img
        filtered_img = depth_image ** 2  # Square each pixel value
        nan_indices = np.argwhere(np.isnan(filtered_img))
        inf_indices = np.argwhere(np.isinf(filtered_img))
        neginf_indices = np.argwhere(np.isneginf(filtered_img))

        # Display filtered_img
        filtered_img_uint8 = filtered_img.astype(np.uint8)
        # cv2.imshow('Thin Parts', filtered_img_uint8)
        # cv2.waitKey(1)

        # Compute expected distance to ground
        height, width = depth_image.shape
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        angle = np.arctan2((v - self.optical_center_y), self.focal_length_y)
        depth = self.plane_params[-1] / (np.tan(angle)+0.00001)
        expectedDistanceImage = depth/1000  # Squaring each term
        # print("expectedDistanceImage[720][640]", expectedDistanceImage[719][640],"X",X[719][640],"Z",Z[719][640])
        # Compute non_ground_mask
        is_ground = np.abs(expectedDistanceImage - filtered_img) > self.plane_threshold
        non_ground_mask = np.logical_not(is_ground)
        # Convert 2D indices to 1D and set to True in non_ground_mask
        non_ground_mask[nan_indices[:, 0], nan_indices[:, 1]] = True
        non_ground_mask[inf_indices[:, 0], inf_indices[:, 1]] = True
        non_ground_mask[neginf_indices[:, 0], neginf_indices[:, 1]] = True
        self.depthMask = non_ground_mask
        non_ground_mask_uint8 = (non_ground_mask * 255).astype(np.uint8)
        # cv2.imshow('Filtered Depth Image', non_ground_mask_uint8)
        # cv2.waitKey(1)
    def showInMovedWindow(self,winname, img, x, y):
        cv2.namedWindow(winname)  # Create a named window
        cv2.moveWindow(winname, x, y)  # Move it to (x,y)
        cv2.imshow(winname, img)

if __name__ == '__main__':
    rospy.init_node('line_detector')
    detector = Detector(1)
    rospy.spin()
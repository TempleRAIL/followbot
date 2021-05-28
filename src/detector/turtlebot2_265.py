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
from followbot.msg import MGSMeasurement, MGSMeasurements, MGSMarker, MGS_and_Control
import math
from nav_msgs.msg import Odometry
import message_filters
from sensor_msgs.msg import Image, PointCloud2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line

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
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
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
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        if 45 < orientation < 135:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
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

class Detector:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.img_sub_1 = message_filters.Subscriber('camera/fisheye1/image_raw', Image)
        self.img_sub_2 = message_filters.Subscriber('camera/fisheye2/image_raw', Image)
        # self.odom_sub = message_filters.Subscriber('odom', Odometry)
        self.measurements = message_filters.ApproximateTimeSynchronizer([self.img_sub_1,self.img_sub_2], queue_size=5, slop=0.1)
        self.measurements.registerCallback(self.measurements_callback)
        # T265 parameters
        self.PPX1 = 419.467010498047
        self.PPY1 = 386.97509765625
        self.Fx1 = 286.221588134766
        self.Fy1 = 287.480102539062
        self.K1 = np.array([[self.Fx1, 0.  , self.PPX1],
                            [0.  , self.Fy1, self.PPY1],
                            [    0.  ,     0.  ,     1.  ]])
        self.D1 = np.array([-0.0043481751345098, 0.037125650793314, -0.0355393998324871, 0.00577297387644649])
        self.Knew1 = self.K1.copy()
        self.Knew1[(0, 1), (0, 1)] = 1 * self.Knew1[(0, 1), (0, 1)]
        self.R = np.eye(3)
        self.t = np.array([0.15,-0.03,0.15])

    def measurements_callback(self,img1,img2):
        cv_image1 = self.bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        cv_image2 = self.bridge.imgmsg_to_cv2(img2, desired_encoding='bgr8')
        img_undistorted = cv2.fisheye.undistortImage(cv_image1, self.K1, D=self.D1, Knew=self.Knew1)
        img_undistorted = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)

        # https://github.com/smidm/opencv-python-fisheye-example/blob/master/fisheye_example.py
        img_undistorted = cv2.GaussianBlur(img_undistorted, (5,5), 1)
        # cv2.imshow("Image window", img_undistorted)
        # cv2.waitKey(0)
        kernel = np.ones((9,9),np.uint8)
        erosion = cv2.erode(img_undistorted,kernel,iterations = 1)

        (thresh, im_bw) = cv2.threshold(erosion, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        edges = cv2.Canny(erosion,175,250)
        temp_lines = probabilistic_hough_line(edges, threshold=20, line_length=40,line_gap=10)

        new_lines = []
        for line in temp_lines:
            p0, p1 = line
            if p0[1] > self.Fy1 and p1[1] > self.Fy1:
                new_lines.append((p0[0],p0[1],p1[0],p1[1]))
        # we can set ROI to decrease computation work load
        print("temp_lines",temp_lines)
        print("new_lines",new_lines)
        a = HoughBundler()
        foo = a.process_lines(new_lines, edges)
        print("merged_lines_all",foo)









        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 20),
                         sharex=True, sharey=True)




        ax = axes.ravel()

        ax[0].imshow(cv_image1,cmap='gray', vmin=0, vmax=255)
        ax[0].axis('off')
        ax[0].set_title('fisheye', fontsize=20)

        ax[1].imshow(img_undistorted,cmap='gray', vmin=0, vmax=255)
        ax[1].axis('off')
        ax[1].set_title('img_undistorted', fontsize=20)

        ax[2].imshow(im_bw,cmap='gray', vmin=0, vmax=255)
        ax[2].axis('off')
        ax[2].set_title('erosion', fontsize=20)

        ax[3].imshow(edges,cmap='gray', vmin=0, vmax=255)
        ax[3].axis('off')
        ax[3].set_title('edges', fontsize=20)

        ax[3].imshow(edges,cmap='gray', vmin=0, vmax=255)
        ax[3].axis('off')
        ax[3].set_title('edges', fontsize=20)

        ax[4].imshow(edges * 0,cmap='gray')
        for line in temp_lines:
            p0, p1 = line
            ax[4].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[4].set_xlim((0, img_undistorted.shape[1]))
        ax[4].set_ylim((img_undistorted.shape[0], 0))
        ax[4].set_title('Probabilistic Hough')

        ax[5].imshow(edges * 0,cmap='gray')
        for line in foo:
            p0, p1 = line
            ax[5].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[5].set_xlim((0, img_undistorted.shape[1]))
        ax[5].set_ylim((img_undistorted.shape[0], 0))
        ax[5].set_title('Merged Probabilistic Hough')

        fig.tight_layout()
        plt.show()
        # cv2.imshow("Image window", skeleton)
        # cv2.waitKey(0)
if __name__ == '__main__':
    rospy.init_node('line_detector')
    detector = Detector()
    rospy.spin()
    # END ALL

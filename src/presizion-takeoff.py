#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import imutils
import socket
import time
import cv2
import numpy as np
import math
import random

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Range
from sensor_msgs.msg import CameraInfo
import std_msgs.msg
import geometry_msgs.msg

from clover import srv
from std_srvs.srv import Trigger
from mavros_msgs.srv import SetMode

from pymavlink import mavutil
from mavros import mavlink
from mavros_msgs.msg import Mavlink

# Default method
tracking_method = 'tracker'
# ROS param defines which method we will use: object tracking or features
if rospy.has_param('/presizion_takeoff_method'):
    tracking_method = rospy.get_param('/presizion_takeoff_method')
    if tracking_method not in ['tracker', 'features']:
        tracking_method = 'tracker'

# Pipeline to send video frames to QGC
stream = cv2.VideoWriter(
    "appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc speed-preset=ultrafast tune=zerolatency ! h264parse ! rtph264pay ! udpsink host=127.0.0.1 port=5600 sync=false", cv2.CAP_GSTREAMER, 0, 60.0, (640, 480))
if not stream.isOpened():
    print("Stream not opened")

# initialize OpenCV's CSRT tracker
tracker = cv2.TrackerCSRT_create()

rospy.init_node('presizion_takeoff')
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
land = rospy.ServiceProxy('land', Trigger)
set_mode = rospy.ServiceProxy('mavros/set_mode', SetMode)

bridge = CvBridge()

tracker_initialized = False

# Get camera params
camera_info = rospy.wait_for_message('main_camera/camera_info', CameraInfo)

# Central point of the camera (intrinsics)
central_point_x = camera_info.K[2]
central_point_y = camera_info.K[5]

# Focal length in pixels
focal_length = camera_info.K[0]

# Feature-based tracking presets
previous_descriptors = None
previous_keypoints = None
summary_keypoint_offset_x = 0
summary_keypoint_offset_y = 0
summary_keypoint_offset_meters_x = 0
summary_keypoint_offset_meters_y = 0
previous_rangefinder = 0

previous_tracker_landing_point = (0, 0)

# Running object tracker
def run_tracker(x, y):
    global tracker_initialized, cv_image, tracker
    r = 15
    tracker_bbox = (x - r, y - r, r * 2, r * 2)

    if tracker_initialized:
        del tracker
        tracker = cv2.TrackerCSRT_create()

        # Initialize tracker with first frame and bounding box
    ok = tracker.init(cv_image, tracker_bbox)
    tracker_initialized = True

# Translate pixel coordinates into real using rangefinder data and similar triangles


def calculate_real_coordinates(px_x, px_y, rangefinder_z):
    global central_point_x, central_point_y, focal_length
    real_x = ((float(px_x) - central_point_x) /
              focal_length) * rangefinder_z
    real_y = ((float(px_y) - central_point_y) /
              focal_length) * rangefinder_z
    return (real_x, real_y)

# Calculation of start point coordinates (tracker method)


def calculate_start_point(tracking_point_x, tracking_point_y, rangefinder_data):
    real_x, real_y = calculate_real_coordinates(
        tracking_point_x, tracking_point_y, rangefinder_data.range)

    dst_to_start_point = math.hypot(abs(real_x), abs(real_y))

    # Drawing line with distance in meters from current image center to start point
    cv2.arrowedLine(cv_image, (camera_info.width / 2, camera_info.height / 2),
                    (tracking_point_x, tracking_point_y), (255, 0, 0), 2)
    cv2.putText(cv_image, str(dst_to_start_point), (camera_info.width / 2,
                                                    camera_info.height / 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    publish_start_point(real_x, real_y, rangefinder_data.range)


def publish_start_point(real_x, real_y, rangefinder_z):
    # Publishing topic to draw start point in rviz
    msg = geometry_msgs.msg.PointStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "base_link"

    msg.point.x = -real_y
    msg.point.y = -real_x
    msg.point.z = -rangefinder_z

    # print("Start point coordinates - x: ", real_x, " y: ", real_y,
    #       " z: ", rangefinder_data.range)
    start_point_pub.publish(msg)


# It is for simple timer functionality
last_time = rospy.get_time()


def image_callback(frame):

    # check to see if we have reached the end of the stream
    if frame is None:
        return

    global cv_image, tracker_initialized, last_time, tracker
    cv_image = bridge.imgmsg_to_cv2(frame, desired_encoding='passthrough')

    if tracking_method == 'tracker' and tracker_initialized:
        (ok, bbox) = tracker.update(cv_image)

        # When lost target
        if not ok:
            print("Tracker reported failure")
            del tracker
            tracker = cv2.TrackerCSRT_create()
            tracker_initialized = False

        (x, y, w, h) = [int(v) for v in bbox]

        landing_point_x = x + w/2
        landing_point_y = y + h/2

        # if rospy.get_time() - last_time > 1: # and telemetry.armed:
        rangefinder_data = rospy.wait_for_message(
            'rangefinder/range', Range)

        
        # last_time = rospy.get_time()

        global previous_descriptors, summary_keypoint_offset_x, summary_keypoint_offset_y, previous_keypoints, \
                previous_rangefinder, summary_keypoint_offset_meters_x, summary_keypoint_offset_meters_y, previous_tracker_landing_point

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints with ORB
        kp = orb.detect(cv_image, None)

        # compute the descriptors with ORB
        kp, current_descriptors = orb.compute(cv_image, kp)

        # current_rangefinder = rospy.wait_for_message(
        #     'rangefinder/range', Range)

        current_rangefinder = rangefinder_data

        tracker_rect_color = (0, 255, 0)

        drift_detected = False

        if previous_descriptors is not None:
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors.
            matches = bf.match(current_descriptors, previous_descriptors)

            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)

            cv_image = cv2.drawKeypoints(cv_image, kp, None, color=(
                0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

            if len(matches) > 0:
                matched_kp = []

                # Average offset of currently matched keypoints in pixels
                avg_offset_px_x = 0
                avg_offset_px_y = 0

                # Average offset of currently matched keypoints in meters
                avg_offset_meters_x = 0
                avg_offset_meters_y = 0

                best_matches_num = 10

                # How many keypoints detected tracker's drift
                drift_detection_counter = 0
                drift_distance_threshold = 0.1

                for m in matches[:best_matches_num]:
                    matched_kp.append(kp[m.queryIdx])

                    cur_kp_dst_to_landing_point =  math.hypot(kp[m.queryIdx].pt[0] - landing_point_x, kp[m.queryIdx].pt[1] - landing_point_y)
                    prev_kp_dst_to_landing_point = math.hypot(previous_keypoints[m.trainIdx].pt[0] - previous_tracker_landing_point[0], previous_keypoints[m.trainIdx].pt[1] - previous_tracker_landing_point[1])

                    if abs(cur_kp_dst_to_landing_point - prev_kp_dst_to_landing_point) / prev_kp_dst_to_landing_point > drift_distance_threshold:
                        drift_detection_counter += 1

                if drift_detection_counter >= 5:
                    print('Drift detected!')
                    tracker_rect_color = (0, 0, 255)

                    drift_detected = True

                    centroid_x_sum = 0
                    centroid_y_sum = 0

                    m_cnt = 0

                    # Calculating true tracker position based on previous frame keypoints
                    for m in matches[:best_matches_num]:
                        m_cnt += 1
                        prev_x_to_lp_offset = previous_keypoints[m.trainIdx].pt[0] - previous_tracker_landing_point[0]
                        cur_x_coordinate = kp[m.queryIdx].pt[0] - prev_x_to_lp_offset
                        centroid_x_sum += cur_x_coordinate

                        prev_y_to_lp_offset = previous_keypoints[m.trainIdx].pt[1] - previous_tracker_landing_point[1]
                        cur_y_coordinate = kp[m.queryIdx].pt[1] - prev_y_to_lp_offset
                        centroid_y_sum += cur_y_coordinate

                    new_landing_point_x = int(centroid_x_sum / m_cnt)
                    new_landing_point_y = int(centroid_y_sum / m_cnt)

                    print("Restarting tracker at: ", new_landing_point_x, ", ", new_landing_point_y)

                    # Run tracker at a detected point
                    run_tracker(new_landing_point_x, new_landing_point_y)


                # print(summary_keypoint_offset_x, summary_keypoint_offset_y)

                cv_image = cv2.drawKeypoints(cv_image, matched_kp, None, color=(
                    0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        if not drift_detected:
            previous_keypoints = kp
            previous_descriptors = current_descriptors
            previous_rangefinder = current_rangefinder
            previous_tracker_landing_point = (landing_point_x, landing_point_y)
        else:
            previous_descriptors = None

        cv2.circle(cv_image, (landing_point_x, landing_point_y),
                   1, (0, 0, 255), 1)
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), tracker_rect_color, 2)

        calculate_start_point(
            landing_point_x, landing_point_y, rangefinder_data)

    # show the output frame
    cv2.imshow("Frame", cv_image)

    # Sending frame to QGC
    stream.write(cv_image)

    key = cv2.waitKey(1) & 0xFF


image_sub = rospy.Subscriber('main_camera/image_raw', Image, image_callback)
start_point_pub = rospy.Publisher(
    'start_point_position', geometry_msgs.msg.PointStamped, queue_size=1)

z = 0
z_direction = 1
x_direction = 1
y_direction = 1

# Startup initialization
navigate(x=0, y=0, z=0.5,
         yaw=float('nan'), speed=1, frame_id='map', auto_arm=True)
rospy.sleep(10)

run_tracker(camera_info.width / 2, camera_info.height / 2)

while not rospy.is_shutdown():
    if z < 2:
        z_direction = 1
    elif z > 10:
        z_direction = -1

    # if z > 1 and not tracker_initialized:
    #     run_tracker(camera_info.width/2, camera_info.height / 2)

    x_direction = x_direction * -1
    x = random.random() * x_direction

    y_direction = y_direction * -1
    y = random.random() * y_direction

    speed = 1 #random.random()

    z = z + 1 * z_direction

    print("Navigate to x: ", x, " y: ", y,
          " z: ", z, "speed: ", speed)

    # Moving continuously up and down with random horizontal offset
    navigate(x=x, y=y, z=z,
             yaw=float('nan'), speed=speed, frame_id='map', auto_arm=True)

    rospy.sleep(3)

udp_video_socket.close()

# close all windows
cv2.destroyAllWindows()

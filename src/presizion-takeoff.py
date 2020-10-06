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

# Central point of the camera
central_point_x = camera_info.K[2]
central_point_y = camera_info.K[5]

# Focal length in pixels
focal_length = camera_info.K[0]


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


# Calculation of start point coordinates
def calculate_start_point(tracking_point_x, tracking_point_y, rangefinder_data):
    real_x = ((float(tracking_point_x) - central_point_x) / \
              focal_length) * rangefinder_data.range
    real_y = ((float(tracking_point_y) - central_point_y) / \
              focal_length) * rangefinder_data.range

    dst_to_start_point = math.hypot(abs(real_x), abs(real_y))

    cv2.putText(cv_image, str(dst_to_start_point), (camera_info.width / 2, camera_info.height / 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    msg = geometry_msgs.msg.PointStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "base_link"

    msg.point.x = -real_y
    msg.point.y = -real_x
    msg.point.z = -rangefinder_data.range

    # print("Start point coordinates - x: ", real_x, " y: ", real_y,
    #       " z: ", rangefinder_data.range)
    start_point_pub.publish(msg)


# It is for simple timer functionality
last_time = rospy.get_time()


def image_callback(frame):

    # check to see if we have reached the end of the stream
    if frame is None:
        return

    global cv_image
    cv_image = bridge.imgmsg_to_cv2(frame, desired_encoding='passthrough')

    global tracker_initialized
    global last_time

    global tracker

    if tracker_initialized:
        (ok, bbox) = tracker.update(cv_image)

        # HOLD mode when lost target
        if not ok:
            print("Tracker reported failure")
            set_mode(custom_mode='HOLD')
            del tracker
            tracker = cv2.TrackerCSRT_create()
            tracker_initialized = False

        (x, y, w, h) = [int(v) for v in bbox]

        landing_point_x = x + w/2
        landing_point_y = y + h/2

        cv2.circle(cv_image, (landing_point_x, landing_point_y),
                   1, (0, 0, 255), 1)
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.line(cv_image, (camera_info.width / 2, camera_info.height / 2), (landing_point_x, landing_point_y), (255, 0, 0), 2)

        # telemetry = get_telemetry()

        # if rospy.get_time() - last_time > 1: # and telemetry.armed:
        rangefinder_data = rospy.wait_for_message(
            'rangefinder/range', Range)

        calculate_start_point(landing_point_x, landing_point_y, rangefinder_data)
            # last_time = rospy.get_time()

    # show the output frame
    cv2.imshow("Frame", cv_image)

    # Sending frame to QGC
    stream.write(cv_image)

    key = cv2.waitKey(1) & 0xFF


image_sub = rospy.Subscriber('main_camera/image_raw', Image, image_callback)
start_point_pub = rospy.Publisher('start_point_position', geometry_msgs.msg.PointStamped, queue_size=1)

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

    speed = random.random()

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

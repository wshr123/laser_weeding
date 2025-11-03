#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import os


def make_camera_info_from_intrinsics(intr, frame_id):
    info = CameraInfo()
    info.width  = intr.width
    info.height = intr.height
    info.distortion_model = "plumb_bob"
    # rs intrinsics: fx, fy, ppx, ppy, coeffs[5]
    info.K = [intr.fx, 0.0, intr.ppx,
              0.0, intr.fy, intr.ppy,
              0.0, 0.0, 1.0]
    info.D = list(intr.coeffs)
    info.R = [1,0,0, 0,1,0, 0,0,1]
    info.P = [intr.fx, 0.0, intr.ppx, 0.0,
              0.0, intr.fy, intr.ppy, 0.0,
              0.0, 0.0, 1.0, 0.0]
    info.header.frame_id = frame_id
    return info

def main():
    rospy.init_node("rs_minimal_publisher", anonymous=False)
    color_w = rospy.get_param("~color_width", 1280)
    color_h = rospy.get_param("~color_height", 720)
    depth_w = rospy.get_param("~depth_width", 1280)
    depth_h = rospy.get_param("~depth_height", 720)
    fps     = rospy.get_param("~fps", 30)
    align_to_color = rospy.get_param("~align_depth", True)
    camera_ns = rospy.get_param("~camera_ns", "camera")

    frame_id_link   = f"{camera_ns}_link"
    frame_id_color  = f"{camera_ns}_color_optical_frame"
    frame_id_depth  = f"{camera_ns}_depth_optical_frame"
    frame_id_aligned_depth = frame_id_color  # 对齐到彩色坐标系

    # ROS pubs
    color_pub = rospy.Publisher(f"/{camera_ns}/color/image_raw", Image, queue_size=1)
    depth_pub = rospy.Publisher(f"/{camera_ns}/aligned_depth_to_color/image_raw", Image, queue_size=1)
    color_info_pub = rospy.Publisher(f"/{camera_ns}/color/camera_info", CameraInfo, queue_size=1)
    depth_info_pub = rospy.Publisher(f"/{camera_ns}/aligned_depth_to_color/camera_info", CameraInfo, queue_size=1)

    bridge = CvBridge()

    # librealsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)

    profile = pipeline.start(config)

    #todo
    # dev = profile.get_device()
    # for s in dev.sensors:
    #     if s.supports(rs.option.frames_queue_size):
    #         s.set_option(rs.option.frames_queue_size, 2)  # 彩色+深度推荐 2

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()

    color_intr = color_stream.get_intrinsics()
    depth_intr = depth_stream.get_intrinsics()

    color_info_msg  = make_camera_info_from_intrinsics(color_intr, frame_id_color)
    aligned_info_msg = make_camera_info_from_intrinsics(color_intr, frame_id_aligned_depth)  # 对齐后使用彩色内参

    align = rs.align(rs.stream.color) if align_to_color else None

    rospy.loginfo("RealSense minimal publisher started.")
    rate = rospy.Rate(fps)

    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()  # 若已对齐，则是对齐到彩色坐标系

            if not color_frame or not depth_frame:
                continue

            # 转 numpy
            color_img = np.asanyarray(color_frame.get_data())  # HxWx3, BGR
            depth_img = np.asanyarray(depth_frame.get_data())  # HxW, uint16, 单位：毫米（z16）

            # ROS header（用同一时间戳保持同步）
            stamp = rospy.Time.now()
            header = Header(stamp=stamp, frame_id=frame_id_color)

            # 彩色图
            color_msg = bridge.cv2_to_imgmsg(color_img, encoding="bgr8")
            color_msg.header = header
            color_info = color_info_msg
            color_info.header.stamp = stamp

            color_pub.publish(color_msg)
            color_info_pub.publish(color_info)

            # 深度图（对齐到彩色）
            depth_header = Header(stamp=stamp, frame_id=frame_id_aligned_depth)
            depth_msg = bridge.cv2_to_imgmsg(depth_img, encoding="16UC1")
            depth_msg.header = depth_header

            aligned_info = aligned_info_msg
            aligned_info.header.stamp = stamp

            depth_pub.publish(depth_msg)
            depth_info_pub.publish(aligned_info)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        pipeline.stop()
        rospy.loginfo("RealSense minimal publisher stopped.")

if __name__ == "__main__":
    main()

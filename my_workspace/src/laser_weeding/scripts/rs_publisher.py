#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

# 预先定义常用变量，避免在循环中重复创建
DISTORTION_MODEL = "plumb_bob"

def make_camera_info_from_intrinsics(intr, frame_id, stamp):
    info = CameraInfo()
    info.header.stamp = stamp
    info.header.frame_id = frame_id
    info.width  = intr.width
    info.height = intr.height
    info.distortion_model = DISTORTION_MODEL
    # rs intrinsics: fx, fy, ppx, ppy, coeffs[5]
    info.K = [intr.fx, 0.0, intr.ppx,
              0.0, intr.fy, intr.ppy,
              0.0, 0.0, 1.0]
    info.D = list(intr.coeffs)
    info.R = [1,0,0, 0,1,0, 0,0,1]
    info.P = [intr.fx, 0.0, intr.ppx, 0.0,
              0.0, intr.fy, intr.ppy, 0.0,
              0.0, 0.0, 1.0, 0.0]
    return info

def main():
    rospy.init_node("rs_minimal_publisher", anonymous=False)

    # 参数获取
    color_w = rospy.get_param("~color_width", 1280)  # 建议尝试 848 或 640 以降低延迟
    color_h = rospy.get_param("~color_height", 720)  # 建议尝试 480
    depth_w = rospy.get_param("~depth_width", 1280)
    depth_h = rospy.get_param("~depth_height", 720)
    fps     = rospy.get_param("~fps", 30)
    align_to_color = rospy.get_param("~align_depth", True)
    camera_ns = rospy.get_param("~camera_ns", "camera")

    frame_id_color  = f"{camera_ns}_color_optical_frame"
    frame_id_depth  = f"{camera_ns}_depth_optical_frame"  # 未对齐时的深度系
    frame_id_aligned = frame_id_color  # 对齐后深度图和彩色图共用坐标系

    # ROS Publishers
    # queue_size=1 是关键，防止ROS内部积压旧消息
    pub_color = rospy.Publisher(f"/{camera_ns}/color/image_raw", Image, queue_size=1)
    pub_depth = rospy.Publisher(f"/{camera_ns}/aligned_depth_to_color/image_raw", Image, queue_size=1)
    pub_info_c = rospy.Publisher(f"/{camera_ns}/color/camera_info", CameraInfo, queue_size=1)
    pub_info_d = rospy.Publisher(f"/{camera_ns}/aligned_depth_to_color/camera_info", CameraInfo, queue_size=1)

    bridge = CvBridge()

    # Pipeline 配置
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)

    # ★★★ 优化 1: 启动并立即配置传感器参数 ★★★
    profile = pipeline.start(config)
    dev = profile.get_device()

    # 获取深度传感器
    depth_sensor = dev.first_depth_sensor()
    color_sensor = dev.first_color_sensor()

    # 设置特定参数以降低延迟
    if depth_sensor.supports(rs.option.frames_queue_size):
        # 设为 1 或 0，只保留最新帧，丢弃旧帧
        depth_sensor.set_option(rs.option.frames_queue_size, 1)
    if color_sensor.supports(rs.option.frames_queue_size):
        color_sensor.set_option(rs.option.frames_queue_size, 1)

    # 禁用自动曝光优先级（重要！防止低光下降帧率）
    if color_sensor.supports(rs.option.auto_exposure_priority):
        color_sensor.set_option(rs.option.auto_exposure_priority, 0.0)

    # 获取内参
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()

    # 预先创建 CameraInfo 模板（除了 header 之外内容不变）
    # 注意：对齐深度图使用彩色相机的内参
    # 这里我们只构建一次数据，循环中只更新 header
    static_color_info_msg = make_camera_info_from_intrinsics(color_intr, frame_id_color, rospy.Time(0))
    static_aligned_info_msg = make_camera_info_from_intrinsics(color_intr, frame_id_aligned, rospy.Time(0))

    align = rs.align(rs.stream.color) if align_to_color else None

    rospy.loginfo(f"RealSense started. Latency optimization enabled. Queue size: 1")

    # ★★★ 优化 2: 去掉 rospy.Rate ★★★
    # 纯靠 wait_for_frames 驱动
    try:
        while not rospy.is_shutdown():
            # 等待帧（阻塞）
            frames = pipeline.wait_for_frames()

            # ★★★ 优化 3: 时间戳处理 ★★★
            # 尽量使用系统当前时间作为 stamp，或者使用 get_timestamp() 换算
            # 为了 minimize latency 造成的 drift，这里取当前时间作为"送达ROS的时间"
            # 如果需要极高精度同步，应该用 frames.get_timestamp() 并与 rospy.Time 进行校准
            now = rospy.Time.now()

            # 对齐处理 (这是 CPU 密集型操作，是延迟的主要来源之一)
            if align:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 转换为 numpy
            # np.asanyarray 避免了不必要的内存拷贝 (如果数据本身就是 contiguous 的)
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

            # 构建 Header
            header_color = Header(stamp=now, frame_id=frame_id_color)
            header_depth = Header(stamp=now, frame_id=frame_id_aligned)

            # 构造消息
            msg_color = bridge.cv2_to_imgmsg(color_img, encoding="bgr8")
            msg_color.header = header_color

            msg_depth = bridge.cv2_to_imgmsg(depth_img, encoding="16UC1")
            msg_depth.header = header_depth

            # 更新 CameraInfo header
            static_color_info_msg.header = header_color
            static_aligned_info_msg.header = header_depth

            # 发布
            pub_color.publish(msg_color)
            pub_info_c.publish(static_color_info_msg)

            pub_depth.publish(msg_depth)
            pub_info_d.publish(static_aligned_info_msg)

    except rospy.ROSInterruptException:
        pass
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()

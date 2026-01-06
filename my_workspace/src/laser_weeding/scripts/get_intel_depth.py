#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class DepthReaderNode(object):
    def __init__(self):
        rospy.init_node("depth_reader_node", anonymous=True)

        # 参数：话题名称，可根据你的启动配置调整
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/aligned_depth_to_color/camera_info")
        self.depth_units = rospy.get_param("~depth_units", 0.001)  # RealSense 通常为毫米，0.001 转为米

        self.bridge = CvBridge()
        self.camera_info = None
        self.depth_image = None

        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.depth_cb, queue_size=1)

        # 可选：显示窗口与鼠标回调
        self.window_name = "Depth (meters)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def camera_info_cb(self, msg):
        self.camera_info = msg

    def depth_cb(self, msg):
        try:
            # RealSense 深度图通常是16UC1
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if depth_raw.dtype != np.uint16:
                rospy.logwarn_throttle(5.0, "Unexpected depth dtype: %s" % depth_raw.dtype)
            # 转换为米
            self.depth_image = depth_raw.astype(np.float32) * self.depth_units

            # 可视化（伪彩色）
            disp = np.nan_to_num(self.depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            # 限制显示范围以更好对比，可根据场景调整
            max_range = rospy.get_param("~max_display_range", 4.0)  # 4米
            disp_clipped = np.clip(disp, 0, max_range)
            disp_norm = cv2.normalize(disp_clipped, None, 0, 255, cv2.NORM_MINMAX)
            disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_JET)

            # 显示
            cv2.imshow(self.window_name, disp_color)
            cv2.waitKey(1)

            # 示例：打印中心点深度（1Hz）
            h, w = self.depth_image.shape
            center_depth = float(self.depth_image[h//2, w//2])
            rospy.loginfo_throttle(1.0, "Center depth: %.3f m" % center_depth)

        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", str(e))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.depth_image is not None:
            if 0 <= y < self.depth_image.shape[0] and 0 <= x < self.depth_image.shape[1]:
                d = float(self.depth_image[y, x])
                rospy.loginfo("Depth at (%d, %d): %.3f m" % (x, y, d))

    def spin(self):
        rospy.loginfo("DepthReaderNode started. Subscribing to: %s", self.depth_topic)
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = DepthReaderNode()
    node.spin()
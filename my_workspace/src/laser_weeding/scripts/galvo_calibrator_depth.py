#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import cv2
import numpy as np
import rospy
import time
import yaml
import os
import math
from std_msgs.msg import String, Int32MultiArray, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import json
import threading
import signal
import sys
from scipy.spatial.transform import Rotation

# 自有模块
from coordinate_transform import CameraGalvoTransform, resolve_config_path
from send_to_teensy import XY2_100Controller


class CircleTrack:
    """单个圆圈的追踪状态类"""
    _id_counter = 0

    def __init__(self, center, radius, area, circularity, solidity):
        self.id = CircleTrack._id_counter
        CircleTrack._id_counter += 1
        self.center = center  # (x, y)
        self.radius = radius
        # 追踪状态
        self.hits = 1  # 连续命中次数
        self.misses = 0  # 丢失帧数
        self.active = False  # 是否确认为有效目标 (hits >= 3)
        # 记录属性用于调试或平滑
        self.area = area
        self.circularity = circularity
        self.solidity = solidity

    def update(self, new_center, new_radius):
        """
        步骤5: EMA权重更新 (0.75旧 + 0.25新)
        """
        alpha = 0.25  # 新值的权重
        # 更新位置
        self.center = (
            self.center[0] * (1 - alpha) + new_center[0] * alpha,
            self.center[1] * (1 - alpha) + new_center[1] * alpha
        )
        # 更新半径
        self.radius = self.radius * (1 - alpha) + new_radius * alpha
        self.hits += 1
        self.misses = 0
        # 连续命中阈值：3次
        if self.hits >= 3:
            self.active = True

    def mark_missed(self):
        self.misses += 1


class ManualGalvoCalibrationNode:
    """手动振镜校准节点 - 计算角度偏移 bias,支持深度"""

    def __init__(self):
        rospy.init_node('manual_galvo_calibration_node', anonymous=True)
        rospy.loginfo("Starting Manual Galvo Calibration Node...")

        self.bridge = CvBridge()

        # ========== 参数 ==========
        self.load_parameters()

        self.galvo_limits = ((-32767, 32767), (-32767, 32767))
        self.galvo_min = -32767
        self.galvo_max = 32767

        # 轴适配
        # self.swap_axes = rospy.get_param('~swap_axes', False)
        # self.invert_x  = rospy.get_param('~invert_x', False)
        # self.invert_y  = rospy.get_param('~invert_y', False)
        # rospy.loginfo(f"[Axis Adapt] swap_axes={self.swap_axes}, invert_x={self.invert_x}, invert_y={self.invert_y}")

        # 仅用于显示的“视觉适配”开关
        # self.visual_swap_axes = rospy.get_param('~visual_swap_axes', False)
        # self.visual_invert_x  = rospy.get_param('~visual_invert_x', False)
        # self.visual_invert_y  = rospy.get_param('~visual_invert_y', False)
        # rospy.loginfo(f"[Visual Overlay] swap={self.visual_swap_axes}, invx={self.visual_invert_x}, invy={self.visual_invert_y}")

        # ========== 坐标变换 ==========
        try:
            self.coordinate_transform = CameraGalvoTransform(
                config_file=self.transform_config_file,
                use_3d_transform=True
            )
            self.coordinate_transform.set_active_galvo_profile(self.galvo_index)
            rospy.loginfo("3D coordinate transformer initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize coordinate transformer: {e}")
            raise

        self._init_galvo_display_state()

        try:
            limits = self.coordinate_transform.get_code_limits(self.galvo_index)
            self.galvo_limits = limits
            (x_min, x_max), (y_min, y_max) = limits
            self.galvo_min = min(x_min, y_min)
            self.galvo_max = max(x_max, y_max)
            rospy.loginfo(
                f"Galvo {self.galvo_index} limits set to X[{x_min}, {x_max}] Y[{y_min}, {y_max}]"
            )
        except Exception as exc:
            rospy.logwarn(f"Failed to read galvo limits from config: {exc}")

        # ========== 振镜控制 ==========
        try:
            self.galvo_controller = XY2_100Controller(
                port=self.serial_port,
                baudrate=self.serial_baudrate,
                galvo_count=self.galvo_index + 1
            )
            try:
                self.galvo_controller.select_galvo(self.galvo_index)
            except Exception as exc:
                rospy.logwarn(f"Failed to select galvo {self.galvo_index}: {exc}")
            rospy.loginfo("Galvo controller initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize galvo controller: {e}")
            raise

        try:
            (x_min, x_max), (y_min, y_max) = self.galvo_limits
            self.galvo_controller.configure_limits(
                self.galvo_index, x_min, x_max, y_min, y_max
            )
        except Exception as exc:
            rospy.logwarn(f"Failed to push galvo limits to controller: {exc}")

        # ========== 状态 ==========
        self.calibration_state = "IDLE"  # IDLE, CENTERING, SELECTING, MANUAL_AIMING
        self.calibration_data = []
        self.current_target_index = -1
        self.current_target = None

        self.detected_circles = []
        self.selected_targets = []

        self.current_image = None
        self.first_frame_synced = False  # 首帧同步宽高

        # 黑色圆检测参数
        self.min_circle_radius = 10
        self.max_circle_radius = 150
        
        # 黑色圆圈检测方法选择: 'hsv' 或 'adaptive_threshold'
        # 手动修改此处来选择检测方法
        self.detection_method = 'adaptive_threshold'  # 可选: 'hsv' 或 'adaptive_threshold'

        # 圆圈跟踪池（使用CircleTrack类）
        self.tracks = []  # List[CircleTrack]

        # 多批次标定数据
        self.calibration_batches = []           # 所有批次数据
        self.current_batch_index = -1           # 当前批次索引
        self.current_batch_description = ""     # 当前批次描述

        # 振镜位置
        self.current_galvo_pos = [0, 0]
        self.target_galvo_pos  = [0, 0]
        self.manual_galvo_pos  = [0, 0]
        self.image_center_galvo_pos = [0, 0]
        self.laser_on = False

        # 手动控制
        self.manual_step = 500
        self.fine_step   = 100
        self.is_fine_mode = False

        self.position_lock = threading.Lock()
        self.running = True

        self.old_terminal_settings = None
        self.terminal_modified = False

        # 键盘绑定
        self.key_handlers = {
            # 移动控制
            'w': self.move_up,
            's': self.move_down,
            'a': self.move_left,
            'd': self.move_right,

            # 步进与模式控制
            'q': self.increase_step,
            'e': self.decrease_step,
            'f': self.toggle_fine_mode,

            # 激光与标定流程控制
            'l': self.toggle_laser,
            'r': self.record_calibration_point,
            'n': self.next_target,
            'p': self.previous_target,
            'c': self.save_calibration,
            ' ': self.center_to_auto_position,

            # === 全局控制 ===
            'b': self.start_calibration,      # 开始标定/新批次
            'm': self.complete_current_batch, # 完成当前批次
            'i': self.init_galvo_center,      # 回到图像中心
            'k': self.stop_calibration,       # 停止标定
            'x': self.reset_calibration,      # 重置标定

            # 其他功能键
            'h': self.move_to_home,
            'o': self.test_four_corners,
        }

        # 信号
        signal.signal(signal.SIGINT,  self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # ROS I/O
        self.setup_ros_interface()

        # 控制线程
        self.galvo_thread = threading.Thread(target=self.galvo_control_loop, daemon=True)
        self.galvo_thread.start()

        # 键盘监听
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        rospy.loginfo("Manual Galvo Calibration Node initialized successfully!")
        rospy.Timer(rospy.Duration(2.0), self.auto_init_galvo_center, oneshot=True)

        # ========== 深度 ==========
        self.use_depth = rospy.get_param('~use_depth', True)
        self.depth_image = None
        self.depth_image_encoding = None
        self.depth_image_lock = threading.Lock()

        if self.use_depth:
            # depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
            self.depth_sub = rospy.Subscriber(
                "/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback, queue_size=1, buff_size=2 ** 24
            )
            # rospy.loginfo(f"Subscribed to depth topic: {depth_topic}")
            # 将 query 函数交给几何模块
            self.coordinate_transform.set_depth_query(self.depth_query_func)
        else:
            rospy.loginfo("Depth disabled by parameter ~use_depth=false")

    def signal_handler(self, signum, frame):
        rospy.loginfo("Received shutdown signal, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        self.running = False
        if hasattr(self, 'laser_on'):
            self.set_laser(False)
        self.restore_terminal()
        if hasattr(self, 'galvo_controller') and self.galvo_controller:
            try:
                self.galvo_controller.close()
            except Exception:
                pass
        rospy.loginfo("Cleanup completed")

    def load_parameters(self):
        self.serial_port = rospy.get_param('~serial_port', '/dev/ttyACM0')
        self.serial_baudrate = rospy.get_param('~serial_baudrate', 115200)

        config_param = rospy.get_param('~transform_config_file', 'cam_params.yaml')
        self.transform_config_file = resolve_config_path(config_param)

        self.galvo_index = int(rospy.get_param('~galvo_index', 0))
        self.galvo_name = rospy.get_param('~galvo_name', f'galvo_{self.galvo_index}')

        self.image_width  = rospy.get_param('~image_width', 640)
        self.image_height = rospy.get_param('~image_height', 480)

        self.calibration_result_file = rospy.get_param('~calibration_result_file', 'manual_galvo_calibration.yaml')

    def setup_ros_interface(self):
        # image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.image_sub = rospy.Subscriber( '/camera/color/image_raw', Image, self.image_callback, queue_size=1)
        self.command_sub = rospy.Subscriber('/manual_calibration_command', String, self.command_callback, queue_size=1)

        self.status_pub = rospy.Publisher('/manual_calibration_status', String, queue_size=1)
        self.result_img_pub = rospy.Publisher('/manual_calibration_image', Image, queue_size=1)
        self.galvo_pub = rospy.Publisher('/galvo_xy', Int32MultiArray, queue_size=1)
        self.laser_pub = rospy.Publisher('/laser_control', Bool, queue_size=1)

        self.status_timer = rospy.Timer(rospy.Duration(0.5), self.publish_status)

    # ===================== 回调 =====================
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if not self.first_frame_synced:
                h, w = cv_image.shape[:2]
                if (w != self.image_width) or (h != self.image_height):
                    rospy.logwarn(f"[Image size sync] param ({self.image_width}x{self.image_height}) "
                                  f"!= msg ({w}x{h}), using msg size.")
                    self.image_width, self.image_height = w, h
                self.first_frame_synced = True

            self.current_image = cv_image
            self.detect_black_circles(cv_image)
            result_image = self.draw_calibration_info(cv_image)

            try:
                result_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                self.result_img_pub.publish(result_msg)
            except CvBridgeError as e:
                rospy.logwarn(f"Failed to publish result image: {e}")
        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")

    def depth_image_callback(self, msg):
        """深度图回调：缓存为米"""
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if msg.encoding == '16UC1':
                depth_m = depth_cv.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                depth_m = depth_cv.astype(np.float32)
            else:
                rospy.logwarn_throttle(5.0, f"[depth] unsupported encoding: {msg.encoding}")
                return

            with self.depth_image_lock:
                self.depth_image = depth_m
                self.depth_image_encoding = msg.encoding

        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[depth] failed to convert: {e}")

    # ===================== 控制环 =====================
    def galvo_control_loop(self):
        rate = rospy.Rate(200)  # 200Hz
        while self.running and not rospy.is_shutdown():
            try:
                with self.position_lock:
                    target_pos_logical = self.target_galvo_pos.copy()

                target_pos_logical = self.clamp_galvo_position(target_pos_logical[0], target_pos_logical[1])

                # 发送到硬件前：逻辑 -> 硬件
                hx, hy = self._to_hw_axes(target_pos_logical[0], target_pos_logical[1])

                if self.galvo_controller:
                    self.galvo_controller.move_to_position(hx, hy, galvo_index=self.galvo_index)

                self.current_galvo_pos = target_pos_logical  # 记录当前“逻辑”位置
                self._set_display_code(self.galvo_index, hx, hy)

                galvo_msg = Int32MultiArray()
                galvo_msg.data = [int(target_pos_logical[0]), int(target_pos_logical[1]), 1 if self.laser_on else 0]
                self.galvo_pub.publish(galvo_msg)

                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Galvo control error: {e}")
                time.sleep(0.01)

    def set_laser(self, enable):
        if not hasattr(self, 'laser_on') or self.laser_on != enable:
            self.laser_on = enable
            laser_msg = Bool()
            laser_msg.data = enable
            self.laser_pub.publish(laser_msg)
            if self.galvo_controller:
                try:
                    self.galvo_controller.send_command("LASER:ON" if enable else "LASER:OFF")
                except Exception as e:
                    rospy.logerr(f"Failed to control laser: {e}")
            rospy.loginfo(f"Laser: {'ON' if enable else 'OFF'}")

    # ===================== 命令流 =====================
    def command_callback(self, msg):
        """处理来自GUI的命令"""
        command = msg.data.strip().lower()
        rospy.loginfo(f"Received command: {command}")
        
        # 映射GUI发送的单字符命令到相应的处理方法
        if command == 'b':
            self.start_calibration()
        elif command == 'r' or command == 'record':  # 支持 'r' 和 'record'
            self.record_calibration_point()
        elif command == 'n':
            self.next_target()
        elif command == 'p':
            self.previous_target()
        elif command == 'm':
            self.complete_current_batch()
        elif command == 'c' or command == 'save':
            self.save_calibration()
        elif command == 'x' or command == 'reset':
            self.reset_calibration()
        elif command == 'k' or command == 'stop':
            self.stop_calibration()
        elif command == 'w':
            self.move_up()
        elif command == 's':
            self.move_down()
        elif command == 'a':
            self.move_left()
        elif command == 'd':
            self.move_right()
        elif command == ' ' or command == 'space':
            self.center_to_auto_position()
        elif command == 'f':
            self.toggle_fine_mode()
        elif command == 'l':
            self.toggle_laser()
        elif command == 'i':
            self.init_galvo_center()
        elif command == 'h':
            self.move_to_home()
        elif command == 'o':
            self.test_four_corners()
        # 兼容旧版本的字符串命令
        elif command == 'start':
            self.start_calibration()
        elif command == 'center':
            self.init_galvo_center()
        else:
            rospy.logwarn(f"Unknown command: {command}")

    # ===================== 初始化与测试 =====================
    def auto_init_galvo_center(self, event):
        rospy.loginfo("Auto-initializing galvo to image center...")
        self.init_galvo_center()

    def init_galvo_center(self):
        self.calibration_state = "CENTERING"
        self.image_center_galvo_pos = [0, 0]
        with self.position_lock:
            self.target_galvo_pos = [0, 0]
            self.manual_galvo_pos = [0, 0]
        rospy.Timer(rospy.Duration(1.0), lambda e: setattr(self, 'calibration_state', 'IDLE'), oneshot=True)

    def test_four_corners(self):
        rospy.loginfo("Testing four corners mapping...")
        corners = [
            (50, 50, "Top-Left"),
            (self.image_width - 50, 50, "Top-Right"),
            (self.image_width - 50, self.image_height - 50, "Bottom-Right"),
            (50, self.image_height - 50, "Bottom-Left"),
        ]
        for px, py, label in corners:
            code_hw = self.coordinate_transform.pixel_to_galvo_code(
                px, py, self.image_width, self.image_height, galvo_index=self.galvo_index
            )
            rospy.loginfo(f"{label}: Pixel({px}, {py}) -> Galvo(HW){code_hw}")

    # ===================== 基础工具 =====================
    def clamp_galvo_position(self, x, y):
        (x_min, x_max), (y_min, y_max) = self.galvo_limits
        x = max(x_min, min(x_max, int(x)))
        y = max(y_min, min(y_max, int(y)))
        return [x, y]


    def depth_query_func(self, u, v):
        """
        查询像素(u,v)深度（米）
        使用邻域中值法提高鲁棒性，避免在黑色圆圈边缘失效
        """
        with self.depth_image_lock:
            if self.depth_image is None:
                return None
            H, W = self.depth_image.shape[:2]
            if u < 0 or v < 0 or u >= W or v >= H:
                return None

            # 使用邻域中值法（3x3窗口）
            radius = 1  # 邻域半径（3x3窗口）
            y_int = int(round(v))
            x_int = int(round(u))

            # 边界检查
            y_min = max(0, y_int - radius)
            y_max = min(H, y_int + radius + 1)
            x_min = max(0, x_int - radius)
            x_max = min(W, x_int + radius + 1)

            # 提取邻域
            neighborhood = self.depth_image[y_min:y_max, x_min:x_max]

            # 过滤有效深度值（有限且>0）
            valid_depths = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0)]

            if len(valid_depths) == 0:
                return None

            # 使用中值（比均值更鲁棒，不受异常值影响）
            z = float(np.median(valid_depths))

            if not np.isfinite(z) or z <= 0:
                return None
            return z

    # ===================== 目标检测 =====================
    def detect_black_circles(self, image):
        """
        检测黑色圆圈（统一入口，根据参数选择检测方法）
        """
        if self.detection_method == 'hsv':
            final_detections = self._detect_with_hsv(image)
        elif self.detection_method == 'adaptive_threshold':
            final_detections = self._detect_with_adaptive_threshold(image)
        else:
            rospy.logerr(f"Unknown detection method: {self.detection_method}")
            final_detections = []

        # 时间一致性跟踪（两种方法共用）
        self.update_tracks(final_detections)

        # 输出激活的tracks（hits >= 3 且 misses < 5）
        stable = []
        for track in self.tracks:
            if track.active and track.misses < 5:
                stable.append({
                    'id': track.id,
                    'center': (int(track.center[0]), int(track.center[1])),
                    'radius': int(track.radius),
                    'area': track.area,
                    'confidence': 1.0 if track.active else 0.5
                })

        # 排序（按面积降序）
        stable.sort(key=lambda x: x['area'], reverse=True)
        self.detected_circles = stable[:52]

    def _detect_with_hsv(self, image):
        """
        使用HSV颜色空间检测黑色圆圈
        步骤1: HSV + V通道法
        步骤2: 形态学去噪
        步骤3: 形状过滤
        步骤4: NMS
        """
        # -------------------------------------------------
        # 步骤1: HSV + V通道法
        # -------------------------------------------------
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # V(亮度) < 80, S(饱和度) < 100
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 100, 80])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # -------------------------------------------------
        # 步骤2: 形态学去噪 (开运算 -> 闭运算)
        # -------------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # -------------------------------------------------
        # 步骤3: 形状过滤
        # -------------------------------------------------
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            if not (self.min_circle_radius <= radius <= self.max_circle_radius):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.6:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.65:
                continue

            if len(cnt) < 5:
                continue

            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            MA, ma = axes
            if ma > 0:
                aspect_ratio = MA / ma
                if aspect_ratio < 0.65:
                    continue
            else:
                continue

            score = 0.5 * area + 0.3 * circularity * 1000 + 0.2 * solidity * 1000

            candidates.append({
                'center': (int(x), int(y)),
                'radius': radius,
                'area': area,
                'circularity': circularity,
                'solidity': solidity,
                'score': score
            })

        # -------------------------------------------------
        # 步骤4: NMS
        # -------------------------------------------------
        candidates.sort(key=lambda x: x['score'], reverse=True)
        final_detections = []
        nms_threshold = 35

        while len(candidates) > 0:
            current = candidates.pop(0)
            final_detections.append(current)

            remaining = []
            for other in candidates:
                dist = math.sqrt((current['center'][0] - other['center'][0]) ** 2 +
                                 (current['center'][1] - other['center'][1]) ** 2)
                if dist >= nms_threshold:
                    remaining.append(other)
            candidates = remaining

        return final_detections

    def _detect_with_adaptive_threshold(self, image):
        """
        使用自适应阈值检测黑色圆圈（更适合反光情况）
        步骤1: 预处理（灰度 + 自适应阈值）
        步骤2: 形态学去噪
        步骤3: 形状过滤（基于凸包）
        步骤4: NMS
        """
        # -------------------------------------------------
        # 步骤1: 预处理 (灰度 + 自适应阈值)
        # -------------------------------------------------
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值参数：
        # 25: blockSize (邻域大小)，必须是奇数
        # 10: C (常数)，调大更严格，调小更宽松
        mask = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            25, 
            10
        )

        # -------------------------------------------------
        # 步骤2: 形态学去噪 (针对反光斑点修复)
        # -------------------------------------------------
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # -------------------------------------------------
        # 步骤3: 形状过滤 (基于凸包)
        # -------------------------------------------------
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            ((x, y), radius) = cv2.minEnclosingCircle(hull)

            if not (self.min_circle_radius <= radius <= self.max_circle_radius):
                continue

            hull_perimeter = cv2.arcLength(hull, True)
            if hull_perimeter == 0:
                continue

            hull_circularity = 4 * np.pi * hull_area / (hull_perimeter * hull_perimeter)
            if hull_circularity < 0.6:
                continue

            solidity = area / hull_area
            if solidity < 0.5:
                continue

            if len(cnt) > 5:
                ellipse = cv2.fitEllipse(cnt)
                (center, axes, angle) = ellipse
                MA, ma = axes
                if ma > 0 and (MA / ma) < 0.5:
                    continue

            score = 0.5 * hull_area + 0.3 * hull_circularity * 1000 + 0.2 * solidity * 1000

            candidates.append({
                'center': (int(x), int(y)),
                'radius': radius,
                'area': hull_area,
                'circularity': hull_circularity,
                'solidity': solidity,
                'score': score
            })

        # -------------------------------------------------
        # 步骤4: NMS
        # -------------------------------------------------
        candidates.sort(key=lambda x: x['score'], reverse=True)
        final_detections = []
        nms_threshold = 35

        while len(candidates) > 0:
            current = candidates.pop(0)
            final_detections.append(current)

            remaining = []
            for other in candidates:
                dist = math.sqrt((current['center'][0] - other['center'][0]) ** 2 +
                                 (current['center'][1] - other['center'][1]) ** 2)
                if dist >= nms_threshold:
                    remaining.append(other)
            candidates = remaining

        return final_detections

    def update_tracks(self, detections):
        """
        步骤5: 时间一致性跟踪
        使用贪婪匹配算法
        """
        match_dist_threshold = 40  # 帧间匹配距离

        matched_track_indices = set()
        matched_detection_indices = set()

        # 尝试将检测结果匹配到现有的轨迹
        for det_idx, det in enumerate(detections):
            best_dist = float('inf')
            best_track_idx = -1
            cx, cy = det['center']

            for trk_idx, track in enumerate(self.tracks):
                if trk_idx in matched_track_indices:
                    continue
                dist = math.sqrt((cx - track.center[0]) ** 2 + (cy - track.center[1]) ** 2)
                if dist < match_dist_threshold and dist < best_dist:
                    best_dist = dist
                    best_track_idx = trk_idx

            if best_track_idx != -1:
                # 匹配成功：更新轨迹
                self.tracks[best_track_idx].update(det['center'], det['radius'])
                matched_track_indices.add(best_track_idx)
                matched_detection_indices.add(det_idx)

        # 处理未匹配的检测 -> 新建轨迹
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detection_indices:
                new_track = CircleTrack(
                    det['center'],
                    det['radius'],
                    det['area'],
                    det['circularity'],
                    det['solidity']
                )
                self.tracks.append(new_track)

        # 处理未匹配的轨迹 -> 增加丢失计数
        # ID生命周期管理：超过10帧未见才删除
        max_misses_for_deletion = 10
        active_tracks = []

        for trk_idx, track in enumerate(self.tracks):
            if trk_idx not in matched_track_indices:
                track.mark_missed()
            if track.misses <= max_misses_for_deletion:
                active_tracks.append(track)

        self.tracks = active_tracks



    def start_calibration(self):
        """开始标定或开始新批次"""
        if len(self.detected_circles) < 2:
            rospy.logwarn("需要至少检测到 2 个黑色圆圈才能开始标定")
            return

        # 开始新批次
        self.current_batch_index += 1
        self.calibration_state = "SELECTING"
        self.selected_targets = self.detected_circles.copy()
        self.current_target_index = 0
        self.calibration_data = []  # 当前批次的数据

        total_points = sum(len(batch['points']) for batch in self.calibration_batches)
        rospy.loginfo(f"========================================")
        rospy.loginfo(f"开始第 {self.current_batch_index + 1} 批次标定")
        rospy.loginfo(f"检测到 {len(self.selected_targets)} 个黑色圆圈")
        rospy.loginfo(f"已累积 {len(self.calibration_batches)} 批次，共 {total_points} 个点")
        rospy.loginfo(f"========================================")

        self.next_target()

    def next_target(self):
        if self.calibration_state not in ["SELECTING", "MANUAL_AIMING"]:
            rospy.logwarn("Not in calibration mode")
            return
        if self.current_target_index >= len(self.selected_targets):
            rospy.loginfo("All targets completed. You can save calibration results with 'C' key.")
            return

        self.current_target = self.selected_targets[self.current_target_index]
        self.calibration_state = "MANUAL_AIMING"

        pixel_x, pixel_y = self.current_target['center']
        code_hw = self.coordinate_transform.pixel_to_galvo_code(
            pixel_x, pixel_y, self.image_width, self.image_height, galvo_index=self.galvo_index
        )

        if code_hw:
            # lx, ly = self._from_hw_axes(int(code_hw[0]), int(code_hw[1]))
            lx,ly = code_hw[0],code_hw[1]
            auto_galvo_pos = self.clamp_galvo_position(lx, ly)
        else:
            auto_galvo_pos = [0, 0]

        with self.position_lock:
            self.target_galvo_pos = auto_galvo_pos.copy()
            self.manual_galvo_pos = auto_galvo_pos.copy()

        rospy.loginfo("Use keyboard to manually adjust laser position, then press 'R' to record")

    def previous_target(self):
        if self.current_target_index > 0:
            self.current_target_index -= 1
            self.next_target()
        else:
            rospy.loginfo("Already at first target")

    def complete_current_batch(self):
        """完成当前批次，准备开始新批次"""
        if len(self.calibration_data) == 0:
            rospy.logwarn("当前批次没有标定数据，无法完成批次")
            return

        # 保存当前批次数据
        batch_data = {
            'batch_index': self.current_batch_index,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'description': self.current_batch_description or f"批次{self.current_batch_index + 1}",
            'points': self.calibration_data.copy()
        }
        self.calibration_batches.append(batch_data)

        # 重置状态
        self.calibration_state = "IDLE"
        self.current_target_index = -1
        self.current_target = None
        self.selected_targets = []
        self.set_laser(False)

        total_points = sum(len(batch['points']) for batch in self.calibration_batches)

        rospy.loginfo(f"========================================")
        rospy.loginfo(f"批次 {self.current_batch_index + 1} 已完成！")
        rospy.loginfo(f"本批次标定点数: {len(self.calibration_data)}")
        rospy.loginfo(f"累积批次数: {len(self.calibration_batches)}")
        rospy.loginfo(f"累积总点数: {total_points}")
        rospy.loginfo(f"========================================")
        rospy.loginfo(f"请移动标定板到新位置，然后按 'B' 开始新批次")
        rospy.loginfo(f"或按 'C' 使用所有数据计算并保存最终标定结果")

        self.calibration_data = []

    # ===================== 键盘控制（逻辑坐标） =====================
    def keyboard_listener(self):
        import tty, termios, select
        try:
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            self.terminal_modified = True
            tty.setraw(sys.stdin.fileno())
            rospy.loginfo("Keyboard listener started. Press 'H' for help.")
            while self.running and not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    self.handle_keyboard_input(key)
        except Exception as e:
            rospy.logerr(f"Keyboard listener error: {e}")
        finally:
            self.restore_terminal()

    def restore_terminal(self):
        if self.terminal_modified and self.old_terminal_settings:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
                self.terminal_modified = False
            except Exception as e:
                rospy.logwarn(f"Failed to restore terminal settings: {e}")

    def handle_keyboard_input(self, key):
        if key in self.key_handlers:
            self.key_handlers[key]()
        elif key == '\x03':  # Ctrl+C
            rospy.loginfo("Ctrl+C pressed, shutting down...")
            self.cleanup()
            rospy.signal_shutdown("User requested shutdown")
        elif key == '?' or ord(key) == 63:
            self.print_help()
        else:
            rospy.logdebug(f"Unknown key: {repr(key)}")

    def print_help(self):
        rospy.loginfo("========== 键盘帮助 ==========")
        rospy.loginfo("【移动控制】")
        rospy.loginfo("  WASD: 上下左右移动 | H: 回原点(0,0)")
        rospy.loginfo("  QE: 增/减步长 | F: 精细模式切换")
        rospy.loginfo("")
        rospy.loginfo("【标定流程】")
        rospy.loginfo("  B: 开始标定/新批次 | M: 完成当前批次")
        rospy.loginfo("  R: 记录当前点 | N/P: 下/上一个目标")
        rospy.loginfo("  C: 保存最终结果 | X: 重置所有数据")
        rospy.loginfo("  SPACE: 自动对准当前目标")
        rospy.loginfo("")
        rospy.loginfo("【其他功能】")
        rospy.loginfo("  L: 激光开关 | I: 初始化到中心")
        rospy.loginfo("  K: 停止流程 | O: 测试四角映射")
        rospy.loginfo("  ?: 显示帮助 | Ctrl+C: 退出程序")
        rospy.loginfo("==============================")

    # 原始 WASD（逻辑坐标）
    def move_up(self):
        step = self.fine_step if self.is_fine_mode else self.manual_step
        with self.position_lock:
            new_pos = self.clamp_galvo_position(self.manual_galvo_pos[0], self.manual_galvo_pos[1] + step)
            self.manual_galvo_pos = new_pos; self.target_galvo_pos = new_pos.copy()

    def move_down(self):
        step = self.fine_step if self.is_fine_mode else self.manual_step
        with self.position_lock:
            new_pos = self.clamp_galvo_position(self.manual_galvo_pos[0], self.manual_galvo_pos[1] - step)
            self.manual_galvo_pos = new_pos; self.target_galvo_pos = new_pos.copy()

    def move_left(self):
        step = self.fine_step if self.is_fine_mode else self.manual_step
        with self.position_lock:
            new_pos = self.clamp_galvo_position(self.manual_galvo_pos[0] - step, self.manual_galvo_pos[1])
            self.manual_galvo_pos = new_pos; self.target_galvo_pos = new_pos.copy()

    def move_right(self):
        step = self.fine_step if self.is_fine_mode else self.manual_step
        with self.position_lock:
            new_pos = self.clamp_galvo_position(self.manual_galvo_pos[0] + step, self.manual_galvo_pos[1])
            self.manual_galvo_pos = new_pos; self.target_galvo_pos = new_pos.copy()

    def move_to_home(self):
        with self.position_lock:
            self.manual_galvo_pos = [0, 0]; self.target_galvo_pos = [0, 0]
        rospy.loginfo("Moved to galvo HOME position (0, 0)")

    def move_to_image_center(self):
        with self.position_lock:
            self.manual_galvo_pos = self.image_center_galvo_pos.copy()
            self.target_galvo_pos = self.image_center_galvo_pos.copy()
        rospy.loginfo(f"Moved to image center position ({self.image_center_galvo_pos[0]}, {self.image_center_galvo_pos[1]})")

    def increase_step(self):
        if self.is_fine_mode:
            self.fine_step = min(self.fine_step + 50, 1000)
            rospy.loginfo(f"Fine step increased to: {self.fine_step}")
        else:
            self.manual_step = min(self.manual_step + 100, 2000)
            rospy.loginfo(f"Manual step increased to: {self.manual_step}")

    def decrease_step(self):
        if self.is_fine_mode:
            self.fine_step = max(self.fine_step - 50, 10)
            rospy.loginfo(f"Fine step decreased to: {self.fine_step}")
        else:
            self.manual_step = max(self.manual_step - 100, 50)
            rospy.loginfo(f"Manual step decreased to: {self.manual_step}")

    def toggle_fine_mode(self):
        self.is_fine_mode = not self.is_fine_mode
        mode = "FINE" if self.is_fine_mode else "NORMAL"
        step = self.fine_step if self.is_fine_mode else self.manual_step
        rospy.loginfo(f"Mode: {mode}, Step: {step}")

    def center_to_auto_position(self):
        if self.current_target:
            pixel_x, pixel_y = self.current_target['center']
            code_hw = self.coordinate_transform.pixel_to_galvo_code(
                pixel_x, pixel_y, self.image_width, self.image_height, galvo_index=self.galvo_index
            )
            if code_hw:
                lx, ly = self._from_hw_axes(int(code_hw[0]), int(code_hw[1]))
                auto_galvo_pos = self.clamp_galvo_position(lx, ly)
            else:
                auto_galvo_pos = [0, 0]
            with self.position_lock:
                self.manual_galvo_pos = auto_galvo_pos.copy()
                self.target_galvo_pos = auto_galvo_pos.copy()
            rospy.loginfo(f"Centered to auto position: ({auto_galvo_pos[0]}, {auto_galvo_pos[1]})")
        else:
            self.move_to_image_center()

    # ===================== 记录/存储 =====================
    def toggle_laser(self):
        self.set_laser(not self.laser_on)

    def record_calibration_point(self):
        """
        记录一个标定数据点。
        获取当前目标的像素位置、深度，以及手动微调后的振镜码值，
        并将这些原始数据保存到 self.calibration_data 列表中，以备后续进行三维计算。
        """
        if self.calibration_state != "MANUAL_AIMING" or not self.current_target:
            rospy.logwarn("不在手动瞄准模式，或没有当前目标，无法记录。\n")
            return

        # 1. 获取目标的像素坐标 (u, v)
        pixel_x, pixel_y = self.current_target['center']


        # 2. 查询并验证该像素点的深度值 (z)
        depth_z = self.depth_query_func(pixel_x, pixel_y)
        if depth_z is None or depth_z <= 0:  # 深度值必须是有效的正数
            rospy.logwarn(f"无法在目标点 ({pixel_x}, {pixel_y}) 获取有效深度，该点已跳过。\n")
            return

        # 3. 获取手动校准后的振镜逻辑码值 (gx, gy)
        manual_galvo_x, manual_galvo_y = self.manual_galvo_pos

        # 4. 将所有必需的原始数据打包成一个字典
        calibration_point = {
            'target_index': self.current_target_index,
            'pixel_position': [int(pixel_x), int(pixel_y)],
            'depth_meters': float(depth_z),
            'manual_galvo_position_logical': [int(manual_galvo_x), int(manual_galvo_y)],
            'timestamp': time.time()
        }

        # 5. 将数据点添加到列表中
        self.calibration_data.append(calibration_point)
        rospy.loginfo(f"成功记录第 {self.current_target_index + 1} 个标定点。\n")

        # 6. 自动前进到下一个目标点
        self.current_target_index += 1
        rospy.Timer(rospy.Duration(0.5), lambda e: self.next_target(), oneshot=True)

    def save_calibration(self):
        """
        计算并保存最终的三维标定结果（外参）。
        使用所有批次累积的数据点。
        """
        # 如果当前批次有未保存的数据，先完成当前批次
        if len(self.calibration_data) > 0:
            rospy.loginfo("检测到当前批次有未保存数据，自动完成当前批次...")
            self.complete_current_batch()

        # 汇总所有批次的数据
        accumulated_data = []
        for batch in self.calibration_batches:
            accumulated_data.extend(batch['points'])

        if len(accumulated_data) < 3:
            rospy.logwarn(f"三维标定至少需要3个标定点，当前只有 {len(accumulated_data)} 个。\n")
            rospy.logwarn(f"已完成批次: {len(self.calibration_batches)}，请继续标定。\n")
            return

        rospy.loginfo("========================================")
        rospy.loginfo("开始计算三维刚体变换...")
        rospy.loginfo(f"总批次数: {len(self.calibration_batches)}")
        rospy.loginfo(f"总标定点数: {len(accumulated_data)}")
        rospy.loginfo("========================================\n")

        points_camera = []
        points_galvo = []
        # 遍历所有批次的所有数据点
        # 1.从像素坐标到振镜坐标(程序识别的)
        # 2.手动控制的galvo code到振镜坐标
        # 3.找出两个振镜坐标之间的变换关系
        for point in accumulated_data:
            u, v = point['pixel_position']
            z = point['depth_meters']
            gx, gy = point['manual_galvo_position_logical']

            p_cam = self.coordinate_transform.pixel_depth_to_point_galvo(u, v, z)    #from pixel to galvo
            z = float(p_cam[2]/1000)
            print("p_cam", p_cam)
            p_galvo = self._manual_code_to_galvo_frame_mm(gx, gy, z)
            print("p_galvo", p_galvo)
            if p_cam is not None and p_galvo is not None:
                points_camera.append(p_cam)
                points_galvo.append(p_galvo)
            else:
                rospy.logwarn(f"跳过标定点 {point['target_index']}，因为坐标转换失败。\n")

        if len(points_camera) < 3:
            rospy.logerr(f"有效标定点不足3个({len(points_camera)}个)，无法进行三维标定。\n")
            return

        points_camera_np = np.array(points_camera, dtype=np.float64)
        points_galvo_np = np.array(points_galvo, dtype=np.float64)

        # (调试可选) 保存点云到文件进行可视化检查
        # np.savetxt("points_camera.txt", points_camera_np, fmt='%.4f')
        # np.savetxt("points_galvo.txt", points_galvo_np, fmt='%.4f')

        try:
            R, t = self.find_rigid_transform_3d(points_camera_np, points_galvo_np)
            rospy.loginfo("三维变换计算成功。\n")
            rospy.loginfo(f"新的旋转矩阵 R (相机->振镜):\n{np.round(R, 4)}\n")
            rospy.loginfo(f"新的平移向量 t (相机->振镜) [mm]:\n{np.round(t, 4)}\n")
        except Exception as e:
            rospy.logerr(f"计算三维变换时发生错误: {e}\n")
            return

        transformed_points = (R @ points_camera_np.T + t).T
        residuals = points_galvo_np - transformed_points
        per_point_error = np.linalg.norm(residuals, axis=1)
        axis_rmse = np.sqrt(np.mean(residuals ** 2, axis=0))
        rmse_total = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
        max_error = float(np.max(per_point_error))
        mean_error = float(np.mean(per_point_error))

        rospy.loginfo(
            "标定残差统计 (mm): "
            f"RMSE_total={rmse_total:.3f}, max={max_error:.3f}, mean={mean_error:.3f}, "
            f"axis_rmse=[{axis_rmse[0]:.3f}, {axis_rmse[1]:.3f}, {axis_rmse[2]:.3f}]\n"
        )

        # 计算每个批次的残差
        batch_statistics = []
        point_idx = 0
        for batch in self.calibration_batches:
            batch_point_count = len(batch['points'])
            batch_errors = per_point_error[point_idx:point_idx + batch_point_count]
            batch_rmse = float(np.sqrt(np.mean(batch_errors ** 2)))
            batch_statistics.append({
                'batch_index': batch['batch_index'],
                'description': batch.get('description', ''),
                'points_count': batch_point_count,
                'rmse_mm': batch_rmse
            })
            point_idx += batch_point_count
            rospy.loginfo(f"批次 {batch['batch_index'] + 1}: {batch_point_count} 点, RMSE={batch_rmse:.3f}mm")

        calibration_info = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_points_used': len(points_camera),
            'method': '3D_rigid_body_transform_SVD_multi_batch',
            'total_batches': len(self.calibration_batches),
            'batches': batch_statistics,
            'validation': {
                'rmse_total_mm': rmse_total,
                'max_error_mm': max_error,
                'mean_error_mm': mean_error,
                'rmse_axis_mm': [float(axis_rmse[0]), float(axis_rmse[1]), float(axis_rmse[2])],
                'samples': len(points_camera)
            }
        }
        extrinsics = {
            'description': '新的相机外参: 从相机坐标系到振镜坐标系的变换 (Pg = R * Pc + t)',
            't_gc_mm': t.flatten().tolist(),
            'R_gc': R.tolist(),
            'q_gc_xyzw': Rotation.from_matrix(R).as_quat().tolist()
        }

        validation_metrics = {
            'rmse_axis_mm': [float(axis_rmse[0]), float(axis_rmse[1]), float(axis_rmse[2])],
            'rmse_total_mm': rmse_total,
            'max_error_mm': max_error,
            'mean_error_mm': mean_error,
            'per_point_error_mm': per_point_error.tolist(),
            'residuals_mm': residuals.tolist(),
            'samples': len(points_camera)
        }

        galvo_entry = {
            'id': self.galvo_index,
            'name': self.galvo_name,
            'calibration_info': calibration_info,
            'refined_extrinsics': extrinsics,
            'active_extrinsics': 'refined',
            'validation': validation_metrics,
            'code_limits': {
                'x': [int(self.galvo_limits[0][0]), int(self.galvo_limits[0][1])],
                'y': [int(self.galvo_limits[1][0]), int(self.galvo_limits[1][1])]
            },
            'galvo_range': [int(self.galvo_min), int(self.galvo_max)]
        }

        try:
            existing_data = {}
            if os.path.exists(self.calibration_result_file):
                with open(self.calibration_result_file, 'r') as f:
                    existing_data = yaml.safe_load(f) or {}

            galvos = existing_data.get('galvos', [])
            updated = False
            for entry in galvos:
                if entry.get('id') == self.galvo_index or entry.get('name') == self.galvo_name:
                    entry.update(galvo_entry)
                    updated = True
                    break

            if not updated:
                galvos.append(galvo_entry)

            existing_data['galvos'] = galvos
            existing_data['last_updated'] = calibration_info['timestamp']

            with open(self.calibration_result_file, 'w') as f:
                yaml.dump(existing_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            rospy.loginfo(f"新的三维标定结果已成功保存至: {self.calibration_result_file}\n")
        except Exception as e:
            rospy.logerr(f"保存标定文件失败: {e}\n")

    def _unproject_to_camera_frame_mm(self, u, v, z_m):
        """辅助方法：将像素和深度（米）反投影为相机坐标系下的三维点（毫米）。"""
        try:
            K = self.coordinate_transform.K
            x_c = (u - K[0, 2]) / K[0, 0]* z_m
            y_c = (v - K[1, 2]) / K[1, 1]* z_m
            z_c = z_m
            return np.array([x_c * 1000.0, y_c * 1000.0, z_c * 1000.0])
        except Exception as e:
            rospy.logwarn(f"反投影计算失败: {e}\n")
            return None

    def _manual_code_to_galvo_frame_mm(self, gx, gy, z_m):
        """辅助方法：将振镜码值和深度（米）转换为振镜坐标系下的三维点（毫米）。"""
        try:
            theta_x, theta_y = self.coordinate_transform.codes_to_angles(gx, gy)
            # print("theta_x, theta_y ",theta_x, theta_y)
            z_mm = z_m * 1000.0
            point_g = self.coordinate_transform.galvo_angles_to_point_depth_cam(theta_x, theta_y, z_mm)
            # print("point g,",point_g)
            return point_g
        except Exception as e:
            rospy.logwarn(f"从振镜码值计算三维点失败: {e}\n")
            return None

    def find_rigid_transform_3d(self, points_A, points_B):
        """使用SVD算法计算从点云A到点云B的三维刚体变换（旋转R和平移t）。"""
        if points_A.shape != points_B.shape:
            raise ValueError("输入点云的维度必须相同\n")
        if points_A.shape[0] < 3:
            raise ValueError("至少需要3个点来计算变换\n")

        centroid_A = np.mean(points_A, axis=0)
        centroid_B = np.mean(points_B, axis=0)
        A_centered = points_A - centroid_A
        B_centered = points_B - centroid_B
        H = A_centered.T @ B_centered
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T
        R = V @ U.T

        if np.linalg.det(R) < 0:
            rospy.logdebug("检测到反射，正在进行修正...\n")
            V[:, -1] *= -1
            R = V @ U.T

        t = centroid_B.T - R @ centroid_A.T
        return R, t.reshape(3, 1)


    def reset_calibration(self):
        """重置所有标定数据（包括所有批次）"""
        self.calibration_state = "IDLE"
        self.calibration_data = []
        self.calibration_batches = []
        self.current_batch_index = -1
        self.current_target_index = -1
        self.current_target = None
        self.selected_targets = []
        self.set_laser(False)
        self.init_galvo_center()
        rospy.loginfo("========================================")
        rospy.loginfo("已重置所有标定数据（包括所有批次）")
        rospy.loginfo("========================================")

    def stop_calibration(self):
        self.calibration_state = "IDLE"
        self.set_laser(False)
        rospy.loginfo("Manual calibration stopped")


    # ===================== 轴适配=====================
    def _to_hw_axes(self, x, y):
        lx, ly = int(x), int(y)
        # if self.swap_axes:
        #     lx, ly = ly, lx
        # if self.invert_x:
        #     lx = -lx
        # if self.invert_y:
        #     ly = -ly
        return lx, ly

    def _from_hw_axes(self, x, y):
        hx, hy = int(x), int(y)
        # if self.invert_x:
        #     hx = -hx
        # if self.invert_y:
        #     hy = -hy
        # if self.swap_axes:
        #     hx, hy = hy, hx
        return hx, hy

    # ===================== 叠加显示（几何一律用"硬件"码值） =====================
    def draw_calibration_info(self, image):
        result = image.copy()

        # 图像中心十字线
        cx = int(self.image_width / 2)
        cy = int(self.image_height / 2)
        cv2.line(result, (cx - 20, cy), (cx + 20, cy), (128, 128, 128), 1)
        cv2.line(result, (cx, cy - 20), (cx, cy + 20), (128, 128, 128), 1)
        cv2.putText(result, "IMG CENTER", (cx + 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # 左上角显示多批次信息
        y_offset = 30
        total_points = sum(len(batch['points']) for batch in self.calibration_batches)
        info_texts = [
            f"Batch: {self.current_batch_index + 1} | Total: {len(self.calibration_batches)}",
            f"Current: {len(self.calibration_data)} pts | Accum: {total_points} pts",
            f"Target: {self.current_target_index + 1}/{len(self.selected_targets)}" if self.selected_targets else "No targets",
            f"State: {self.calibration_state}"
        ]
        for text in info_texts:
            cv2.putText(result, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 25

        # 黑色圆圈目标（带ID和置信度）
        # 如果正在进行标定，优先显示 selected_targets（固定位置）
        if self.calibration_state in ["SELECTING", "MANUAL_AIMING"] and len(self.selected_targets) > 0:
            # 绘制标定目标（来自 selected_targets，固定位置）
            for i, target in enumerate(self.selected_targets):
                center = target['center']
                radius = target.get('radius', 20)  # 使用保存的半径或默认值
                circle_id = target.get('id', i)

                is_current = (self.current_target and
                             i == self.current_target_index)
                is_calibrated = any(
                    abs(center[0] - cp['pixel_position'][0]) < 10 and
                    abs(center[1] - cp['pixel_position'][1]) < 10
                    for cp in self.calibration_data
                )

                # 颜色：当前目标=红色，已标定=黄色，未标定=绿色
                if is_current:
                    color = (0, 0, 255); thickness = 3
                elif is_calibrated:
                    color = (0, 255, 255); thickness = 2
                else:
                    color = (0, 255, 0); thickness = 2

                cv2.circle(result, (int(center[0]), int(center[1])), int(radius), color, thickness)
                cv2.circle(result, (int(center[0]), int(center[1])), 3, color, -1)

                # 标签：显示ID和状态
                label = f"T{i+1}"
                if is_current:
                    label += " [CURRENT]"
                elif is_calibrated:
                    label += " [DONE]"

                cv2.putText(result, label, (int(center[0] + radius + 5), int(center[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        else:
            # 非标定状态，显示实时检测的圆圈
            for i, circle in enumerate(self.detected_circles):
                center = circle['center']
                radius = circle['radius']
                circle_id = circle.get('id', -1)
                confidence = circle.get('confidence', 1.0)

                is_calibrated = any(
                    abs(center[0] - cp['pixel_position'][0]) < 10 and
                    abs(center[1] - cp['pixel_position'][1]) < 10
                    for cp in self.calibration_data
                )

                # 颜色：已标定=黄色，未标定=绿色
                if is_calibrated:
                    color = (0, 255, 255); thickness = 2
                else:
                    color = (0, 255, 0); thickness = 2

                cv2.circle(result, (int(center[0]), int(center[1])), int(radius), color, thickness)
                cv2.circle(result, (int(center[0]), int(center[1])), 3, color, -1)

                # 标签：显示ID和状态
                label = f"ID{circle_id}"
                if is_calibrated:
                    label += " [DONE]"

                cv2.putText(result, label, (int(center[0] + radius + 5), int(center[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(result, f"Conf:{confidence:.2f}", (int(center[0] + radius + 5), int(center[1] + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        result = self._draw_all_galvo_markers(result)

        # AUTO 位置
        if self.current_target:
            try:
                px, py = self.current_target['center']
                code_hw = self.coordinate_transform.pixel_to_galvo_code(
                    px, py, self.image_width, self.image_height, galvo_index=self.galvo_index
                )
                if code_hw:
                    self.coordinate_transform.set_active_galvo_profile(self.galvo_index)
                    ax, ay = self.coordinate_transform.galvo_code_to_pixel_3d(int(code_hw[0]), int(code_hw[1]),
                                                                           self.image_width, self.image_height)
                    ax, ay = int(round(ax)), int(round(ay))
                    if 0 <= ax < self.image_width and 0 <= ay < self.image_height:
                        cv2.circle(result, (ax, ay), 8, (255, 0, 255), 2)
                        cv2.putText(result, "AUTO", (ax + 15, ay + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            except Exception:
                pass

        return result

    def _init_galvo_display_state(self):
        try:
            self.galvo_count = self.coordinate_transform.get_galvo_profile_count()
        except Exception:
            self.galvo_count = 1

        self.galvo_names = []
        self.galvo_display_codes = []

        for idx in range(self.galvo_count):
            metadata = self._fetch_galvo_metadata(idx)
            self.galvo_names.append(metadata.get('name', f'galvo_{idx}'))

            offset = metadata.get('code_offset', [0.0, 0.0])
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                offset = [0.0, 0.0]

            hw_x, hw_y = self._to_hw_axes(offset[0], offset[1])
            self.galvo_display_codes.append([int(hw_x), int(hw_y)])

        if self.galvo_index >= self.galvo_count:
            rospy.logwarn(
                f"Configured galvo_index={self.galvo_index} exceeds available galvos ({self.galvo_count});"
                " clamping to last profile."
            )
            self.galvo_index = max(0, self.galvo_count - 1)

        if 0 <= self.galvo_index < len(self.galvo_names):
            self.galvo_name = self.galvo_names[self.galvo_index]

    def _fetch_galvo_metadata(self, index):
        try:
            return self.coordinate_transform.get_profile_metadata(index)
        except Exception:
            return {}

    def _set_display_code(self, index, x_code, y_code):
        if 0 <= index < len(self.galvo_display_codes):
            self.galvo_display_codes[index][0] = int(x_code)
            self.galvo_display_codes[index][1] = int(y_code)

    def _get_display_code(self, index):
        if 0 <= index < len(self.galvo_display_codes):
            return self.galvo_display_codes[index]
        return None

    def _draw_all_galvo_markers(self, image):
        if not hasattr(self, 'galvo_count') or self.galvo_count <= 0:
            return image

        result = image
        try:
            active_idx = self.coordinate_transform.active_profile_index
        except AttributeError:
            active_idx = self.galvo_index

        try:
            for idx in range(self.galvo_count):
                codes = self._get_display_code(idx)
                if not codes:
                    continue

                try:
                    self.coordinate_transform.set_active_galvo_profile(idx)
                except Exception:
                    continue

                galvo_pixel = self.coordinate_transform.galvo_code_to_pixel(
                    int(codes[0]), int(codes[1]), self.image_width, self.image_height
                )

                if galvo_pixel is None:
                    continue

                x, y = int(round(galvo_pixel[0])), int(round(galvo_pixel[1]))
                if not (0 <= x < self.image_width and 0 <= y < self.image_height):
                    continue

                is_active = (idx == self.galvo_index)
                if is_active:
                    color = (0, 0, 255) if self.laser_on else (0, 165, 255)
                else:
                    color = (255, 255, 0)

                cv2.line(result, (x - 16, y), (x + 16, y), color, 2)
                cv2.line(result, (x, y - 16), (x, y + 16), color, 2)
                cv2.circle(result, (x, y), 8, color, 2)

                label = self.galvo_names[idx] if idx < len(self.galvo_names) else f'galvo_{idx}'
                if is_active:
                    label += " [ACTIVE]"
                    if self.laser_on:
                        label += " LASER"

                cv2.putText(result, label, (x + 20, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(result, f"CODE({codes[0]},{codes[1]})", (x + 20, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        finally:
            try:
                self.coordinate_transform.set_active_galvo_profile(active_idx)
            except Exception:
                pass

        return result

    # ===================== 状态发布 =====================
    def publish_status(self, event):
        try:
            total_points = sum(len(batch['points']) for batch in self.calibration_batches)

            status_info = {
                'state': self.calibration_state,
                'detected_circles': len(self.detected_circles),
                'current_batch_index': self.current_batch_index,
                'total_batches': len(self.calibration_batches),
                'current_batch_points': len(self.calibration_data),
                'accumulated_points': total_points,
                'current_target_index': self.current_target_index,
                'total_targets': len(self.selected_targets),
                'current_target': self.current_target['center'] if self.current_target else None,
                'galvo_index': self.galvo_index,
                'galvo_name': self.galvo_name,
                'galvo_position_logical': self.current_galvo_pos,
                'manual_galvo_position_logical': self.manual_galvo_pos,
                'image_center_galvo_pos_logical': self.image_center_galvo_pos,
                'code_limits': {
                    'x': list(self.galvo_limits[0]),
                    'y': list(self.galvo_limits[1])
                },
                'galvo_range': [self.galvo_min, self.galvo_max],
                'laser_on': self.laser_on,
                'step_size': self.fine_step if self.is_fine_mode else self.manual_step,
                'fine_mode': self.is_fine_mode,
            }
            status_msg = String()
            status_msg.data = json.dumps(status_info)
            self.status_pub.publish(status_msg)
        except Exception as e:
            rospy.logdebug(f"Failed to publish status: {e}")

    # ===================== 析构 =====================
    def __del__(self):
        self.cleanup()


def main():
    try:
        node = ManualGalvoCalibrationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Manual calibration node interrupted")
    except Exception as e:
        rospy.logerr(f"Manual calibration node error: {e}")
    finally:
        if 'node' in locals():
            node.cleanup()


if __name__ == '__main__':
    main()

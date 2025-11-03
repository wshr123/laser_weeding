#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import traceback
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray, Float32MultiArray, String, Bool
from detector import WeedDetector
import json
import time
from collections import deque
from enum import Enum
import threading
from typing import Dict, List, Tuple

# 导入振镜控制器和新的坐标变换模块
from send_to_teensy import XY2_100Controller
from coordinate_transform import CameraGalvoTransform, resolve_config_path


class SystemState(Enum):
    """系统状态枚举"""
    IDLE = "IDLE"      # 空闲，等待目标
    TRACKING = "TRACKING"  # 跟踪目标
    FIRING = "FIRING"  # 激光照射中


class LaserWeedingNode:
    def __init__(self):
        try:
            rospy.init_node('laser_weeding_node', anonymous=True)
            rospy.loginfo("=" * 50)
            rospy.loginfo("starting laser weeding node...")
            rospy.loginfo("=" * 50)

            # 基础组件
            self.bridge = CvBridge()

            # ========== 坐标变换模式选择 ==========
            use_3d_transform = rospy.get_param('~use_3d_transform', True)
            config_param = rospy.get_param('~transform_config_file', None)
            config_file = resolve_config_path(config_param)

            try:
                self.coordinate_transform = CameraGalvoTransform(
                    config_file=config_file,
                    use_3d_transform=use_3d_transform
                )
                rospy.loginfo(f"using transform config: {self.coordinate_transform.config_file_path}")
            except Exception as e:
                rospy.logerr(f"failed initialize coordinate transform: {e}")
                sys.exit(1)

            # ========== 参数加载 ==========
            # 模型参数
            self.model_path = rospy.get_param('~model_path', '')
            self.model_type = rospy.get_param('~model_type', 'yolov11')
            self.device = rospy.get_param('~device', '0')
            self.weed_class_id = rospy.get_param('~weed_class_id', 0)
            self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.3)

            # 预测参数
            self.total_delay = rospy.get_param('~total_delay', 0.08)
            self.prediction_time = rospy.get_param('~prediction_time', 0.15)
            self.use_kalman = rospy.get_param('~use_kalman', True)
            self.max_prediction_distance = rospy.get_param('~max_prediction_distance', 300)

            # 激光控制参数
            self.aiming_time = rospy.get_param('~aiming_time', 0.1)
            self.laser_time = rospy.get_param('~laser_time', 0.2)

            # 振镜参数
            self.serial_port = rospy.get_param('~serial_port', '/dev/ttyACM0')
            self.serial_baudrate = rospy.get_param('~serial_baudrate', 115200)
            self.min_move_step = rospy.get_param('~min_move_step', 2)
            profile_count = max(1, self.coordinate_transform.get_galvo_profile_count())
            self.requested_galvo_count = int(rospy.get_param('~galvo_count', profile_count))
            self.galvo_split_axis = rospy.get_param('~galvo_split_axis', 'vertical').lower()
            self.galvo_split_ratio = float(rospy.get_param('~galvo_split_ratio', 0.5))
            self.galvo_overlap_px = int(rospy.get_param('~galvo_overlap_px', 40))

            # 图像参数
            self.image_width = rospy.get_param('~image_width', 640)
            self.image_height = rospy.get_param('~image_height', 480)

            # 目标管理参数
            self.target_timeout = rospy.get_param('~target_timeout', 0.5)
            self.min_stable_frames = rospy.get_param('~min_stable_frames', 2)

            # ========== 振镜控制器初始化 ==========
            try:
                self.galvo_controller = XY2_100Controller(
                    port=self.serial_port,
                    baudrate=self.serial_baudrate,
                    galvo_count=max(1, self.requested_galvo_count)
                )
                rospy.loginfo(f"initializing galvo controller : {self.serial_port}")
            except Exception as e:
                rospy.logwarn(f": initializing galvo controller {e}")
                self.galvo_controller = None

            controller_count = getattr(
                self.galvo_controller,
                'galvo_count',
                max(1, self.requested_galvo_count)
            )
            requested_count = max(1, self.requested_galvo_count)

            if controller_count < profile_count:
                rospy.logwarn(
                    f"galvo controller supports {controller_count} heads but {profile_count} profiles are available"
                )
            if profile_count < requested_count:
                rospy.logwarn(
                    f"only {profile_count} galvo profiles found, adjusting requested count {requested_count}"
                )

            self.galvo_count = max(1, min(controller_count, profile_count, requested_count))
            if self.galvo_controller and self.galvo_controller.galvo_count != self.galvo_count:
                rospy.loginfo(
                    f"using {self.galvo_count} galvo heads for scheduling (controller reports {controller_count})"
                )

            self.galvo_limits = []
            for idx in range(self.galvo_count):
                try:
                    limits = self.coordinate_transform.get_code_limits(idx)
                except Exception as exc:
                    rospy.logwarn(f"failed to read galvo limits for head {idx}: {exc}")
                    limits = ((-32767, 32767), (-32767, 32767))
                self.galvo_limits.append(limits)

            self._configure_controller_limits()

            # ========== 检测器初始化 ==========
            try:
                self.detector = WeedDetector(
                    model_path=self.model_path,
                    model_type=self.model_type,
                    weed_class_id=self.weed_class_id,
                    crop_class_id=1,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                    tracker_type='custom'
                )
                rospy.loginfo(f' {self.model_type.upper()} loading model')
            except Exception as e:
                rospy.logerr(f" failed loading model {e}")
                rospy.logerr(traceback.format_exc())
                sys.exit(1)

            # ========== 状态变量初始化 ==========
            # 系统状态
            self.system_state = SystemState.IDLE
            self.state_start_time = time.time()

            # 目标管理
            self.current_target = None
            self.target_queue = []            # 将按“距离中心”动态排序
            self.processed_targets = set()
            self.processing_target = None
            self.all_targets = {}

            # 位置历史（用于预测）
            self.position_history = deque(maxlen=20)
            self.last_update_time = time.time()

            # 振镜控制
            self.active_galvo_index = 0
            self.galvo_positions = [[0, 0] for _ in range(self.galvo_count)]
            self.target_galvo_positions = [[0, 0] for _ in range(self.galvo_count)]
            self.galvo_regions = []
            self._update_galvo_regions()
            self.laser_on = False
            self.tracking_active = False

            # 卡尔曼滤波器
            self.kalman_filter = None
            if self.use_kalman:
                self.init_kalman_filter()

            # 线程控制
            self.running = True
            self.position_lock = threading.Lock()

            # 性能监控
            self.frame_count = 0
            self.fps_counter = deque(maxlen=30)

            # 图像缓存
            self.current_image = None

            # 相机信息
            self.camera_info_received = False

            # ========== ROS 发布器和订阅器 ==========
            # 发布器
            self.galvo_pub = rospy.Publisher('/galvo_xy', Int32MultiArray, queue_size=1)
            self.laser_pub = rospy.Publisher('/laser_control', Bool, queue_size=1)
            self.det_img_pub = rospy.Publisher('/det_img/image_raw', Image, queue_size=1)
            self.status_pub = rospy.Publisher('/system_status', String, queue_size=1)
            self.target_pub = rospy.Publisher('/current_target', String, queue_size=1)
            self.transform_pub = rospy.Publisher('/transform_info', String, queue_size=1)

            # 订阅
            image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
            self.image_sub = rospy.Subscriber(
                image_topic,
                Image,
                self.image_callback,
                queue_size=1
            )

            # 标定和控制相关订阅器
            # self.calibration_sub = rospy.Subscriber(
            #     '/calibration_command',
            #     String,
            #     self.calibration_callback,
            #     queue_size=1
            # )

            # 深度图订阅  注入 depth_query_func
            self.depth_image = None
            self.depth_image_encoding = None
            self.depth_image_lock = threading.Lock()
            depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
            self.depth_sub = rospy.Subscriber(
                depth_topic,
                Image,
                self.depth_image_callback,
                queue_size=1,
                buff_size=2 ** 24
            )
            rospy.loginfo(f"subscribed to depth topic: {depth_topic}")
            # 设置 query 函数给变换器
            self.coordinate_transform.set_depth_query(self.depth_query_func)

            # ========== 启动控制线程 ==========
            self.galvo_thread = threading.Thread(target=self.galvo_control_loop)
            self.galvo_thread.daemon = True
            self.galvo_thread.start()

            # 主控制定时器
            self.control_timer = rospy.Timer(
                rospy.Duration(0.005),  # 200 Hz
                self.control_loop
            )

            # 状态发布定时器
            self.status_timer = rospy.Timer(
                rospy.Duration(0.5),  # 2 Hz
                self.publish_status
            )

            rospy.loginfo("=" * 50)
            rospy.loginfo("finish loading laser weeding node!")
            rospy.loginfo("=" * 50)

        except Exception as e:
            rospy.logerr(f"failed initializing laser weeding node : {e}")
            rospy.logerr(traceback.format_exc())
            sys.exit(1)

    def image_callback(self, msg):
        try:
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image

            # 更新图像尺寸（如果变化）
            h, w = cv_image.shape[:2]
            if w != self.image_width or h != self.image_height:
                self.image_width = w
                self.image_height = h
                rospy.loginfo(f"update image size: {w}x{h}")
                self._update_galvo_regions()

            # FPS计算
            current_time = time.time()
            self.fps_counter.append(current_time)
            self.frame_count += 1

            # 检测和跟踪
            result_image, detections = self.detector.detect_and_track_weeds(cv_image)

            # 更新目标
            self.update_targets(detections, current_time)

            # 绘制信息
            result_image = self.draw_info(result_image)

            # 发布检测结果图像
            if self.frame_count % 1 == 0:
                try:
                    det_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                    self.det_img_pub.publish(det_msg)
                except CvBridgeError as e:
                    rospy.logerr(f"det_img publish failed: {e}")

        except Exception as e:
            rospy.logerr(f"image callback error: {e}")

    def control_loop(self, event):
        try:
            current_time = time.time()
            # self.update_position_history([640,360], current_time)  #for test
            if self.system_state == SystemState.IDLE:
                # 进入空闲状态时，重新挑选队列：只保留“在扫描范围内”的目标，并按“距离中心”排序
                candidates = []
                cx = self.image_width / 2.0
                cy = self.image_height / 2.0

                for tid, info in self.all_targets.items():
                    if tid in self.processed_targets:
                        continue
                    if 'center' not in info:
                        continue
                    # 目标超时丢弃
                    timeout = self.target_timeout
                    if current_time - info.get('last_seen', current_time) > timeout:
                        continue
                    # 必须稳定到一定帧数
                    if info.get('stable_frames', 0) < self.min_stable_frames:
                        continue
                    # 在扫描范围内才作为候选
                    if not info.get('in_range', False):
                        continue

                    dist = np.hypot(info['center'][0] - cx, info['center'][1] - cy)
                    candidates.append((dist, tid))

                # 中心优先：按距离从近到远排序
                candidates.sort(key=lambda x: x[0])
                self.target_queue = [tid for _, tid in candidates]

                # 取最近的一个作为当前处理对象
                if self.target_queue:
                    target_id = self.target_queue.pop(0)
                    if target_id in self.all_targets and target_id not in self.processed_targets:
                        target_info = self.all_targets[target_id]
                        galvo_index = target_info.get('galvo_index', 0)
                        self.current_target = {
                            'id': target_id,
                            'start_time': current_time,
                            'galvo_index': galvo_index
                        }
                        self.processing_target = target_id
                        self.position_history.clear()

                        self.active_galvo_index = galvo_index
                        if self.galvo_controller:
                            try:
                                self.galvo_controller.select_galvo(galvo_index)
                            except Exception as exc:
                                rospy.logdebug(f"failed to select galvo {galvo_index}: {exc}")

                        with self.position_lock:
                            self.target_galvo_positions[galvo_index] = self.galvo_positions[galvo_index][:]

                        if self.use_kalman and self.kalman_filter:
                            center = target_info['center']
                            self.kalman_filter.statePre = np.array(
                                [[center[0]], [center[1]], [0], [0]],
                                dtype=np.float32
                            )

                        self.change_state(SystemState.TRACKING)

            elif self.system_state == SystemState.TRACKING:
                if self.current_target:
                    target_id = self.current_target['id']

                    if target_id in self.processed_targets:
                        self.current_target = None
                        self.processing_target = None
                        self.change_state(SystemState.IDLE)
                        return

                    if target_id in self.all_targets:
                        target_info = self.all_targets[target_id]
                        galvo_index = target_info.get('galvo_index', self.active_galvo_index)
                        self.update_position_history(target_info['center'], current_time, galvo_index)

                        elapsed = current_time - self.current_target['start_time']
                        if elapsed >= self.aiming_time:
                            self.change_state(SystemState.FIRING)
                    else:
                        self.current_target = None
                        self.processing_target = None
                        self.change_state(SystemState.IDLE)

            elif self.system_state == SystemState.FIRING:
                if self.current_target:
                    target_id = self.current_target['id']

                    if target_id in self.all_targets:
                        target_info = self.all_targets[target_id]
                        galvo_index = target_info.get('galvo_index', self.active_galvo_index)
                        self.update_position_history(target_info['center'], current_time, galvo_index)

                    elapsed = current_time - self.state_start_time
                    if elapsed >= self.laser_time:
                        self.processed_targets.add(target_id)
                        if target_id in self.all_targets:
                            self.all_targets[target_id]['processed'] = True

                        self.current_target = None
                        self.processing_target = None
                        self.change_state(SystemState.IDLE)

        except Exception as e:
            rospy.logerr(f"control loop failed: {e}")

    def depth_image_callback(self, msg):
        """深度图回调：自动缓存为 numpy 格式 米"""
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

    def depth_query_func(self, u, v):
        """查询像素(u,v)的深度，单位：米。用于 coordinate_transform 中"""
        with self.depth_image_lock:
            if self.depth_image is None:
                return None
            H, W = self.depth_image.shape[:2]
            if u < 0 or v < 0 or u >= W or v >= H:
                return None
            z = float(self.depth_image[int(round(v)), int(round(u))])
            if not np.isfinite(z) or z <= 0:
                return None
            return z

    def _update_galvo_regions(self):
        if self.galvo_count <= 1:
            self.galvo_regions = [{
                'u_min': 0,
                'u_max': self.image_width,
                'v_min': 0,
                'v_max': self.image_height
            }]
            return

        if self.galvo_count > 2:
            rospy.logwarn_once("Current implementation supports at most two galvo regions; extra heads will share the last region")

        width = max(1, int(self.image_width))
        height = max(1, int(self.image_height))
        overlap = max(0, int(self.galvo_overlap_px))

        if self.galvo_split_axis == 'horizontal':
            split = int(round(height * self.galvo_split_ratio))
            split = max(0, min(height, split))
            top_max = min(height, split + overlap // 2)
            bottom_min = max(0, split - overlap // 2)
            self.galvo_regions = [
                {'u_min': 0, 'u_max': width, 'v_min': 0, 'v_max': max(0, top_max)},
                {'u_min': 0, 'u_max': width, 'v_min': min(height, bottom_min), 'v_max': height}
            ]
        else:
            split = int(round(width * self.galvo_split_ratio))
            split = max(0, min(width, split))
            left_max = min(width, split + overlap // 2)
            right_min = max(0, split - overlap // 2)
            self.galvo_regions = [
                {'u_min': 0, 'u_max': max(0, left_max), 'v_min': 0, 'v_max': height},
                {'u_min': min(width, right_min), 'u_max': width, 'v_min': 0, 'v_max': height}
            ]

    def _configure_controller_limits(self):
        if not self.galvo_controller or not self.galvo_controller.is_connected():
            return

        for idx, limits in enumerate(self.galvo_limits):
            if not limits:
                continue
            (x_min, x_max), (y_min, y_max) = limits
            try:
                self.galvo_controller.configure_limits(
                    idx,
                    int(x_min),
                    int(x_max),
                    int(y_min),
                    int(y_max)
                )
            except Exception as exc:
                rospy.logwarn(f"failed to push galvo limits for head {idx}: {exc}")

    def get_galvo_limits(self, galvo_index: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if 0 <= galvo_index < len(self.galvo_limits):
            return self.galvo_limits[galvo_index]
        return ((-32767, 32767), (-32767, 32767))

    def is_within_galvo_limits(self, galvo_index: int, x: float, y: float) -> bool:
        (x_min, x_max), (y_min, y_max) = self.get_galvo_limits(galvo_index)
        return x_min <= x <= x_max and y_min <= y <= y_max

    def clamp_to_galvo_limits(self, galvo_index: int, x: float, y: float) -> Tuple[int, int]:
        (x_min, x_max), (y_min, y_max) = self.get_galvo_limits(galvo_index)
        clamped_x = int(max(x_min, min(x_max, int(round(x)))))
        clamped_y = int(max(y_min, min(y_max, int(round(y)))))
        return clamped_x, clamped_y

    def select_galvo_for_pixel(self, u: float, v: float) -> int:
        if not self.galvo_regions:
            return 0

        candidates = []
        fallback = []
        for idx, region in enumerate(self.galvo_regions):
            in_region = (
                region['u_min'] <= u <= region['u_max'] and
                region['v_min'] <= v <= region['v_max']
            )
            center_u = (region['u_min'] + region['u_max']) / 2.0
            center_v = (region['v_min'] + region['v_max']) / 2.0
            center_dist = (u - center_u) ** 2 + (v - center_v) ** 2

            if in_region:
                candidates.append((center_dist, idx))
                continue

            du = 0.0
            if u < region['u_min']:
                du = region['u_min'] - u
            elif u > region['u_max']:
                du = u - region['u_max']

            dv = 0.0
            if v < region['v_min']:
                dv = region['v_min'] - v
            elif v > region['v_max']:
                dv = v - region['v_max']

            fallback.append((du * du + dv * dv, idx, center_dist))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]

        if fallback:
            fallback.sort(key=lambda item: (item[0], item[2]))
            return fallback[0][1]

        return 0

    def in_galvo_scan_range(self, u, v):
        """判断像素点是否在振镜扫描范围内：用几何模型映射到码值并检查范围"""
        try:
            galvo_index = self.select_galvo_for_pixel(float(u), float(v))
            code = self.coordinate_transform.pixel_to_galvo_code(
                float(u), float(v), self.image_width, self.image_height,
                galvo_index=galvo_index
            )
            if code is None:
                return False
            x, y = int(code[0]), int(code[1])
            return self.is_within_galvo_limits(galvo_index, x, y)
        except Exception:
            return False

    def init_kalman_filter(self):
        """初始化简单的2D卡尔曼滤波器（位置+速度）"""
        self.kalman_filter = cv2.KalmanFilter(4, 2)

        dt = 0.033
        self.kalman_filter.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kalman_filter.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kalman_filter.errorCovPost = np.eye(4, dtype=np.float32) * 100

    def update_targets(self, detections, current_time):
        """更新目标信息（不入队，在 control_loop 中统一排序队列）"""
        current_frame_ids = set()

        for track_id, bbox, confidence in detections:
            if confidence < self.confidence_threshold:
                continue

            x, y, w, h = bbox
            cx = x + w / 2.0
            cy = y + h / 2.0
            current_frame_ids.add(track_id)

            if track_id not in self.all_targets:
                self.all_targets[track_id] = {
                    'first_seen': current_time,
                    'stable_frames': 0,
                    'processed': False
                }

            # 已处理的目标更新基础信息后继续标记 processed
            galvo_index = self.select_galvo_for_pixel(cx, cy)
            in_range = self.in_galvo_scan_range(cx, cy)
            self.all_targets[track_id].update({
                'bbox': bbox,
                'center': [cx, cy],
                'confidence': confidence,
                'last_seen': current_time,
                'galvo_index': galvo_index,
                'in_range': in_range
            })

            if track_id in self.processed_targets:
                self.all_targets[track_id]['processed'] = True
            else:
                # 仅未处理目标累积稳定帧数
                self.all_targets[track_id]['stable_frames'] = self.all_targets[track_id].get('stable_frames', 0) + 1

        # 清理超时目标
        to_remove = []
        for tid, tinfo in list(self.all_targets.items()):
            if tid not in current_frame_ids:
                timeout = self.target_timeout * (3 if tid in self.processed_targets else 1)
                if current_time - tinfo.get('last_seen', current_time) > timeout:
                    to_remove.append(tid)

        for tid in to_remove:
            if tid in self.all_targets:
                del self.all_targets[tid]
            if self.current_target and self.current_target.get('id') == tid:
                rospy.logwarn(f"current {tid} lost")
                self.current_target = None
                self.processing_target = None
                self.change_state(SystemState.IDLE)
            if tid in self.target_queue:
                try:
                    self.target_queue.remove(tid)
                except ValueError:
                    pass

    def update_position_history(self, position, timestamp, galvo_index):
        """更新位置历史并计算预测位置"""
        self.position_history.append({
            'position': position,
            'time': timestamp,
            'galvo': galvo_index
        })

        predicted_pos = self.predict_position(self.prediction_time, galvo_index)

        if predicted_pos:
            current_pos = position
            distance = np.hypot(predicted_pos[0] - current_pos[0],
                                predicted_pos[1] - current_pos[1])

            if distance > self.max_prediction_distance:
                predicted_pos = current_pos

            galvo_result = self.coordinate_transform.pixel_to_galvo_code(
                predicted_pos[0], predicted_pos[1],
                self.image_width, self.image_height,
                galvo_index=galvo_index
            )

            if galvo_result:
                galvo_x, galvo_y = self.clamp_to_galvo_limits(galvo_index, galvo_result[0], galvo_result[1])
                with self.position_lock:
                    self.target_galvo_positions[galvo_index] = [galvo_x, galvo_y]
                    self.active_galvo_index = galvo_index

    def predict_position(self, dt, galvo_index):
        """预测未来位置"""
        relevant_history = [
            entry for entry in self.position_history if entry.get('galvo') == galvo_index
        ]

        if len(relevant_history) < 2:
            return relevant_history[-1]['position'] if relevant_history else None

        if self.use_kalman and self.kalman_filter:
            try:
                current_pos = relevant_history[-1]['position']
                measurement = np.array([[current_pos[0]], [current_pos[1]]], dtype=np.float32)

                self.kalman_filter.correct(measurement)
                self.kalman_filter.predict()

                state = self.kalman_filter.statePost
                pred_x = state[0, 0] + state[2, 0] * dt
                pred_y = state[1, 0] + state[3, 0] * dt

                return [float(pred_x), float(pred_y)]
            except Exception as e:
                rospy.logdebug(f"kalman predict failed: {e}")

        if len(relevant_history) >= 2:
            p1 = relevant_history[-2]
            p2 = relevant_history[-1]

            time_diff = p2['time'] - p1['time']
            if time_diff > 0:
                vx = (p2['position'][0] - p1['position'][0]) / time_diff
                vy = (p2['position'][1] - p1['position'][1]) / time_diff

                pred_x = p2['position'][0] + vx * dt
                pred_y = p2['position'][1] + vy * dt

                return [pred_x, pred_y]

        return relevant_history[-1]['position']

    def galvo_control_loop(self):
        """振镜控制线程（高频率）"""
        rate = rospy.Rate(500)  # 500Hz

        while self.running and not rospy.is_shutdown():
            try:
                with self.position_lock:
                    target_positions = [pos[:] for pos in self.target_galvo_positions]

                for idx, target_pos in enumerate(target_positions):
                    current_pos = self.galvo_positions[idx]
                    dx = target_pos[0] - current_pos[0]
                    dy = target_pos[1] - current_pos[1]
                    distance = np.hypot(dx, dy)

                    cmd_x, cmd_y = self.clamp_to_galvo_limits(idx, target_pos[0], target_pos[1])
                    if distance > self.min_move_step and self.galvo_controller:
                        self.galvo_controller.move_to_position(
                            cmd_x,
                            cmd_y,
                            galvo_index=idx
                        )

                    self.galvo_positions[idx] = [cmd_x, cmd_y]

                    galvo_msg = Int32MultiArray()
                    galvo_msg.data = [
                        cmd_x,
                        cmd_y,
                        1 if (self.laser_on and idx == self.active_galvo_index) else 0,
                        idx
                    ]
                    self.galvo_pub.publish(galvo_msg)

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"galvo control error: {e}")
                time.sleep(0.001)

    def change_state(self, new_state):
        """改变系统状态"""
        old_state = self.system_state
        self.system_state = new_state
        self.state_start_time = time.time()

        if new_state == SystemState.IDLE:
            self.tracking_active = False
            self.set_laser(False)
        elif new_state == SystemState.TRACKING:
            self.tracking_active = True
            self.set_laser(False)
        elif new_state == SystemState.FIRING:
            self.tracking_active = True
            self.set_laser(True)

    def set_laser(self, enable):
        """控制激光"""
        if self.laser_on != enable:
            self.laser_on = enable

            laser_msg = Bool()
            laser_msg.data = enable
            self.laser_pub.publish(laser_msg)

            if self.galvo_controller:
                try:
                    if enable:
                        self.galvo_controller.select_galvo(self.active_galvo_index)
                        self.galvo_controller.send_command("LASER:ON")
                    else:
                        self.galvo_controller.send_command("LASER:OFF")
                except Exception as e:
                    rospy.logerr(f"failed control laser: {e}")

    def draw_info(self, image):
        """在图像上绘制信息"""
        result = image.copy()

        # 绘制所有目标
        for track_id, target_info in self.all_targets.items():
            if 'bbox' not in target_info:
                continue

            bbox = target_info['bbox']
            center = target_info['center']
            x, y, w, h = bbox

            if track_id in self.processed_targets or target_info.get('processed', False):
                color = (128, 128, 128)  # 灰色：已处理
                label = f"ID:{track_id} [completed]"
                thickness = 1
            elif self.current_target and self.current_target['id'] == track_id:
                if self.system_state == SystemState.FIRING:
                    color = (0, 0, 255)  # 红色：激光照射
                    label = f"ID:{track_id} [laser]"
                else:
                    color = (0, 255, 255)  # 黄色：跟踪中
                    label = f"ID:{track_id} [tracking]"
                thickness = 2
            else:
                color = (0, 255, 0)  # 绿色：检测到
                label = f"ID:{track_id}"
                thickness = 1

            # 绘制检测框
            cv2.rectangle(result, (int(x), int(y)),
                          (int(x + w), int(y + h)), color, thickness)
            cv2.circle(result, (int(center[0]), int(center[1])), 3, color, -1)
            cv2.putText(result, label, (int(x), int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # # 当前目标的预测瞄准点
            # if track_id == (self.current_target['id'] if self.current_target else None):
            #     predicted_pos = self.predict_position(self.prediction_time)
            #     if predicted_pos:
            #         cv2.circle(result, (int(predicted_pos[0]), int(predicted_pos[1])),
            #                    6, (255, 0, 255), 2)
            #         cv2.putText(result, "PRED", (int(predicted_pos[0] + 10), int(predicted_pos[1])),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # 绘制振镜激光实际瞄准位置
        try:    #最开始是因为没有读到深度,用的是给定的深度值,所以可视化中瞄准的不准确
            active_pos = self.galvo_positions[self.active_galvo_index]
            self.coordinate_transform.set_active_galvo_profile(self.active_galvo_index)
            galvo_pixel = self.coordinate_transform.galvo_code_to_pixel(
                active_pos[0], active_pos[1],
                self.image_width, self.image_height
            )

            if galvo_pixel is not None:
                color = (0, 0, 255) if self.laser_on else (255, 255, 0)
                xg = int(round(galvo_pixel[0]))
                yg = int(round(galvo_pixel[1]))

                cv2.line(result, (xg - 15, yg), (xg + 15, yg), color, 3)
                cv2.line(result, (xg, yg - 15), (xg, yg + 15), color, 3)
                cv2.circle(result, (xg, yg), 10, color, 2)

                laser_status = "LASER ON" if self.laser_on else "AIM"
                cv2.putText(result, laser_status, (xg + 20, yg - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                coord_text = f"({active_pos[0]:.0f},{active_pos[1]:.0f})"
                pixel_text = f"{xg:.0f},{yg:.0f}"
                cv2.putText(result, coord_text, (xg + 20, yg + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(result, pixel_text, (xg + 20, yg + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                cv2.putText(result, "GALVO POS UNKNOWN", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        except Exception as e:
            rospy.logdebug(f"Failed to draw galvo position: {e}")
            cv2.putText(result, "GALVO DISPLAY ERROR", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status_text = f"State: {self.system_state.value}"
        cv2.putText(result, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result

    def publish_status(self, event):
        """发布系统状态"""
        try:
            fps = 0
            if len(self.fps_counter) > 1:
                fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])

            transform_info = self.coordinate_transform.get_transform_info()

            status_info = {
                'state': self.system_state.value,
                'current_target': self.current_target['id'] if self.current_target else None,
                'processing_target': self.processing_target,
                'queue_size': len(self.target_queue),
                'processed_count': len(self.processed_targets),
                'processed_targets': list(self.processed_targets),
                'total_targets': len(self.all_targets),
                'laser_on': self.laser_on,
                'fps': round(fps, 1),
                'galvo_position': self.galvo_positions[self.active_galvo_index],
                'galvo_positions': self.galvo_positions,
                'active_galvo': self.active_galvo_index,
                'galvo_limits': [
                    {
                        'x': [int(lim[0][0]), int(lim[0][1])],
                        'y': [int(lim[1][0]), int(lim[1][1])]
                    }
                    for lim in self.galvo_limits
                ],
                'galvo_regions': self.galvo_regions,
                'prediction_time_ms': self.prediction_time * 1000,
                'total_delay_ms': self.total_delay * 1000,
                'transform_info': transform_info,
                'camera_info_received': self.camera_info_received
            }

            status_msg = String()
            status_msg.data = json.dumps(status_info)
            self.status_pub.publish(status_msg)

            transform_msg = String()
            transform_msg.data = json.dumps(transform_info)
            self.transform_pub.publish(transform_msg)

            if self.current_target and self.current_target['id'] in self.all_targets:
                target_info = self.all_targets[self.current_target['id']]
                assigned_galvo = target_info.get('galvo_index', self.active_galvo_index)
                target_data = {
                    'id': self.current_target['id'],
                    'center': target_info['center'],
                    'confidence': target_info['confidence'],
                    'galvo_index': assigned_galvo,
                    'predicted_position': self.predict_position(self.prediction_time, assigned_galvo)
                }
                target_msg = String()
                target_msg.data = json.dumps(target_data)
                self.target_pub.publish(target_msg)

        except Exception as e:
            rospy.logerr(f"failed publish status: {e}")

    def __del__(self):
        """析构函数"""
        self.running = False

        self.set_laser(False)

        if hasattr(self, 'galvo_controller') and self.galvo_controller:
            try:
                self.galvo_controller.close()
            except:
                pass

        rospy.loginfo("close laser weeding node")


def main():
    """主函数"""
    try:
        node = LaserWeedingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ros interrupt")
    except Exception as e:
        rospy.logerr(f"ros error: {e}")
        rospy.logerr(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import traceback
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import serial
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Float32, String, Bool
from detector import WeedDetector
import json
import time
from collections import defaultdict, deque
from enum import Enum
import threading

# 导入振镜控制器
from one_object_to_teensy import XY2_100Controller


class SystemState(Enum):
    """系统状态枚举"""
    IDLE = "IDLE"  # 空闲，等待目标
    TRACKING = "TRACKING"  # 跟踪目标
    AIMING = "AIMING"  # 瞄准中（跟踪但未开激光）
    FIRING = "FIRING"  # 激光照射中
    COMPLETED = "COMPLETED"  # 当前目标完成


class PredictiveLaserWeedingNode:
    def __init__(self):
        try:
            rospy.init_node('laser_weeding_node', anonymous=True)
            rospy.loginfo("=" * 50)
            rospy.loginfo("Starting Laser Weeding Node...")
            rospy.loginfo("=" * 50)

            # Parameters
            self.bridge = CvBridge()

            # 振镜控制器初始化
            serial_port_name = rospy.get_param('~serial_port', '/dev/ttyACM0')
            serial_baudrate = rospy.get_param('~serial_baudrate', 115200)

            try:
                self.galvo_controller = XY2_100Controller(
                    port=serial_port_name,
                    baudrate=serial_baudrate
                )
                rospy.loginfo(f"Galvo controller initialized successfully")
            except Exception as e:
                rospy.logwarn(f"Failed to initialize galvo controller: {e}")
                rospy.logwarn("Continuing without galvo connection...")
                self.galvo_controller = None

            # 时间同步和延迟补偿参数
            self.total_system_delay = rospy.get_param('~total_system_delay', 0.08)  # 总系统延迟80ms
            self.camera_delay = rospy.get_param('~camera_delay', 0.033)  # 相机延迟33ms (30fps)
            self.processing_delay = rospy.get_param('~processing_delay', 0.02)  # 处理延迟20ms
            self.galvo_delay = rospy.get_param('~galvo_delay', 0.01)  # 振镜响应延迟10ms
            self.communication_delay = rospy.get_param('~communication_delay', 0.005)  # 通信延迟5ms

            # 预测参数
            self.prediction_time = self.total_system_delay + rospy.get_param('~extra_prediction', 0.02)  # 额外预测20ms
            # self.prediction_time = 0.02
            self.max_prediction_distance = rospy.get_param('~max_prediction_distance', 100)  # 最大预测距离限制

            # 激光控制参数
            self.aiming_duration = rospy.get_param('~aiming_duration', 0.5)  # 瞄准时长
            self.laser_duration = rospy.get_param('~laser_duration', 0.3)  # 激光照射时长

            # 目标跟踪相关参数
            self.current_target_id = None  # 当前处理的目标ID
            self.all_detected_targets = {}  # 存储所有检测到的目标信息
            self.processed_targets = set()  # 已处理的目标集合
            self.target_queue = []  # 目标队列
            self.min_confidence = rospy.get_param('~min_confidence', 0.3)
            self.target_stable_frames = rospy.get_param('~target_stable_frames', 2)
            self.terget_timeout = rospy.get_param('~terget_timeout', 0.5)   #清除消失目标时间

            # 系统状态
            self.system_state = SystemState.IDLE
            self.state_start_time = None

            # 高级运动预测
            self.position_history = deque(maxlen=10)  # 存储更多历史位置
            self.velocity_history = deque(maxlen=5)  # 速度历史
            self.acceleration_history = deque(maxlen=3)  # 加速度历史

            # 卡尔曼滤波器参数（用于平滑预测）
            self.use_kalman_filter = rospy.get_param('~use_kalman_filter', True)
            self.kalman_filter = None

            # 振镜控制
            self.tracking_active = False
            self.last_galvo_position = [0, 0]
            self.target_galvo_position = [0, 0]

            # 激光状态
            self.laser_on = False

            # 多线程控制
            self.control_thread_active = True
            self.position_lock = threading.Lock()

            # 性能监控
            self.frame_timestamps = deque(maxlen=30)
            self.galvo_timestamps = deque(maxlen=30)
            self.prediction_errors = deque(maxlen=20)

            # Detection model initialization
            self.weed_class_id = rospy.get_param('~weed_class_id', 0)
            self.crop_class_id = rospy.get_param('~crop_class_id', 1)
            self.model_path = rospy.get_param('~model_path', '')
            self.model_type = rospy.get_param('~model_type', 'yolov8')
            self.device = rospy.get_param('~device', '0')
            self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.3)

            # 跟踪器配置
            self.tracker_type = rospy.get_param('~tracker_type', 'custom')
            self.tracker_config = rospy.get_param('~tracker_config', None)

            try:
                self.detector = WeedDetector(
                    model_path=self.model_path,
                    model_type=self.model_type,
                    weed_class_id=self.weed_class_id,
                    crop_class_id=self.crop_class_id,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                    tracker_type=self.tracker_type,
                    tracker_config=self.tracker_config
                )
                rospy.loginfo(f'{self.model_type.upper()} detection model loaded successfully')
            except Exception as e:
                rospy.logerr(f"Failed to load detection model: {e}")
                rospy.logerr(traceback.format_exc())
                sys.exit(1)

            # Calibration parameters
            self.image_width = rospy.get_param('~image_width', 640)
            self.image_height = rospy.get_param('~image_height', 480)

            # 初始化图像变量
            self.left_img = None
            self.frame_count = 0

            # Publishers
            self.xy_pub = rospy.Publisher('/galvo_xy', Int32MultiArray, queue_size=1)
            self.laser_pub = rospy.Publisher('/laser_control', Bool, queue_size=1)
            self.det_img_pub = rospy.Publisher('/det_img/image_raw', Image, queue_size=1)
            self.target_pub = rospy.Publisher('/current_target', String, queue_size=1)
            self.prediction_pub = rospy.Publisher('/prediction_info', String, queue_size=1)
            self.status_pub = rospy.Publisher('/system_status', String, queue_size=1)

            # Subscribers
            image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
            rospy.loginfo(f"Subscribing to image topic: {image_topic}")

            self.left_img_sub = rospy.Subscriber(
                image_topic,
                Image,
                self.left_img_cb,
                queue_size=1
            )

            self.image_sub = rospy.Subscriber(
                image_topic,
                Image,
                self.image_callback,
                queue_size=1
            )

            # 启动独立的振镜控制线程
            self.galvo_control_thread = threading.Thread(target=self.galvo_control_loop)
            self.galvo_control_thread.daemon = True
            self.galvo_control_thread.start()

            # 主控制循环定时器
            self.control_timer = rospy.Timer(
                rospy.Duration(0.02),  # 50Hz
                self.control_loop_callback
            )

            # 性能监控定时器
            self.performance_timer = rospy.Timer(
                rospy.Duration(2.0),
                self.performance_callback
            )

            rospy.loginfo("=" * 50)
            rospy.loginfo("Laser Weeding Node Initialized!")
            rospy.loginfo("=" * 50)

        except Exception as e:
            rospy.logerr(f"Failed to initialize PredictiveLaserWeedingNode: {e}")
            rospy.logerr(traceback.format_exc())
            sys.exit(1)

    def initialize_kalman_filter(self, initial_position):
        """初始化卡尔曼滤波器"""
        try:
            # 状态向量: [x, y, vx, vy, ax, ay]
            self.kalman_filter = cv2.KalmanFilter(6, 2)

            # 测量矩阵 (我们只能观测位置)
            self.kalman_filter.measurementMatrix = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ], dtype=np.float32)

            # 状态转移矩阵 (匀加速运动模型)
            dt = 0.033  # 假设30fps
            self.kalman_filter.transitionMatrix = np.array([
                [1, 0, dt, 0, 0.5 * dt * dt, 0],
                [0, 1, 0, dt, 0, 0.5 * dt * dt],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)

            # 过程噪声协方差
            self.kalman_filter.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1

            # 测量噪声协方差
            self.kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

            # 初始状态
            self.kalman_filter.statePre = np.array([
                initial_position[0], initial_position[1], 0, 0, 0, 0
            ], dtype=np.float32)

            # 初始协方差
            self.kalman_filter.errorCovPre = np.eye(6, dtype=np.float32) * 1000

            rospy.loginfo("Kalman filter initialized")

        except Exception as e:
            rospy.logwarn(f"Failed to initialize Kalman filter: {e}")
            self.kalman_filter = None

    def left_img_cb(self, msg):
        """图像回调"""
        try:
            self.left_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")

    def image_callback(self, data):
        """主图像处理回调"""
        if self.left_img is None:
            return

        try:
            current_time = time.time()
            self.frame_count += 1
            self.frame_timestamps.append(current_time)

            cv_image = self.left_img

            # 检测杂草
            result_image, detection_results = self.detector.detect_and_track_weeds(cv_image)

            # 更新目标信息
            self.update_all_targets(detection_results, current_time)

            # 绘制预测信息
            result_image = self.draw_prediction_info(result_image)

            # 发布图像
            if self.frame_count % 2 == 0:
                try:
                    ros_det_image = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                    self.det_img_pub.publish(ros_det_image)
                except CvBridgeError as e:
                    rospy.logerr(f"Failed to convert detection image: {e}")

            # 发布目标信息
            self.publish_target_info()

        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")

    def update_all_targets(self, detection_results, current_time):
        current_target_ids = set()

        # 更新检测到的目标
        for track_id, bbox, confidence in detection_results:
            if confidence >= self.min_confidence:
                x, y, w, h = bbox
                centroid = [x + w * 0.5, y + h * 0.5]

                # 更新目标信息
                self.all_detected_targets[track_id] = {
                    'track_id': track_id,
                    'bbox': bbox,
                    'centroid': centroid,
                    'confidence': confidence,
                    'timestamp': current_time,
                    'last_seen': current_time
                }

                current_target_ids.add(track_id)

                # 如果是新目标且未处理过，加入队列
                if (track_id not in self.processed_targets and
                        track_id not in self.target_queue and
                        track_id != self.current_target_id):  # 确保当前正在处理的目标不会被重复加入
                    self.target_queue.append(track_id)
                    rospy.loginfo(f"New target added to queue: ID {track_id}")

        # 清理长时间未见的目标
        timeout = self.terget_timeout
        to_remove = []
        for track_id, target_info in self.all_detected_targets.items():
            if track_id not in current_target_ids:
                if current_time - target_info['last_seen'] > timeout:
                    to_remove.append(track_id)

        for track_id in to_remove:
            if track_id in self.all_detected_targets:
                del self.all_detected_targets[track_id]
            if track_id in self.target_queue:
                self.target_queue.remove(track_id)
            # 如果是当前目标，清除当前目标
            if track_id == self.current_target_id:
                rospy.logwarn(f"Current target ID {track_id} lost, resetting to IDLE")
                self.current_target_id = None
                self.change_state(SystemState.IDLE)
            rospy.loginfo(f"Target ID {track_id} removed due to timeout")
        return

    def get_current_target_info(self):
        """获取当前目标的完整信息"""
        if self.current_target_id is None:
            return None

        # 从all_detected_targets中获取最新信息
        if self.current_target_id in self.all_detected_targets:
            return self.all_detected_targets[self.current_target_id]

        return None

    def control_loop_callback(self, event):
        """主控制循环"""
        try:
            current_time = rospy.Time.now()

            if self.system_state == SystemState.IDLE:
                # 空闲状态，检查是否有待处理目标
                if self.target_queue:
                    self.target_queue.sort()
                    target_id = self.target_queue.pop(0)

                    # 检查目标是否仍然存在
                    if target_id in self.all_detected_targets:
                        self.current_target_id = target_id
                        self.change_state(SystemState.TRACKING)
                        rospy.loginfo(f"Start processing target ID: {target_id}")
                    else:
                        rospy.logwarn(f"Target ID {target_id} no longer exists, skipping")

            elif self.system_state == SystemState.TRACKING:
                # 跟踪状态
                current_target_info = self.get_current_target_info()
                if current_target_info:
                    # 检查跟踪时间，进入瞄准状态
                    elapsed_time = (current_time - self.state_start_time).to_sec()
                    if elapsed_time > 0.2:  # 跟踪0.2秒后开始瞄准
                        self.change_state(SystemState.AIMING)
                else:
                    # 目标丢失，返回空闲
                    rospy.logwarn(f"Lost target during tracking: ID {self.current_target_id}")
                    self.current_target_id = None
                    self.change_state(SystemState.IDLE)

            elif self.system_state == SystemState.AIMING:
                # 瞄准阶段
                current_target_info = self.get_current_target_info()
                if current_target_info:
                    if (current_time - self.state_start_time).to_sec() >= self.aiming_duration:
                        self.change_state(SystemState.FIRING)
                else:
                    rospy.logwarn(f"Lost target during aiming: ID {self.current_target_id}")
                    self.current_target_id = None
                    self.change_state(SystemState.IDLE)

            elif self.system_state == SystemState.FIRING:
                # 激光照射阶段
                current_target_info = self.get_current_target_info()
                if current_target_info:
                    if (current_time - self.state_start_time).to_sec() >= self.laser_duration:
                        self.change_state(SystemState.COMPLETED)
                else:
                    # 即使目标丢失，也要完成激光照射周期
                    if (current_time - self.state_start_time).to_sec() >= self.laser_duration:
                        self.change_state(SystemState.COMPLETED)

            elif self.system_state == SystemState.COMPLETED:
                # 完成状态
                if self.current_target_id is not None:
                    self.processed_targets.add(self.current_target_id)
                    rospy.loginfo(f"Target ID {self.current_target_id} processing completed")
                    self.current_target_id = None

                    # 清理历史数据
                    self.position_history.clear()
                    self.velocity_history.clear()
                    self.acceleration_history.clear()
                    self.kalman_filter = None

                self.change_state(SystemState.IDLE)

        except Exception as e:
            rospy.logerr(f"Control loop error: {e}")

    def change_state(self, new_state):
        """状态转换"""
        old_state = self.system_state
        self.system_state = new_state
        self.state_start_time = rospy.Time.now()

        rospy.loginfo(f"State change: {old_state.value} -> {new_state.value}")

        # 根据状态控制激光（只在FIRING状态开启激光）
        if new_state == SystemState.FIRING:
            self.set_laser(True)
        elif old_state == SystemState.FIRING and new_state != SystemState.FIRING:
            self.set_laser(False)

        # 根据状态控制跟踪
        if new_state in [SystemState.TRACKING, SystemState.AIMING, SystemState.FIRING]:
            with self.position_lock:
                self.tracking_active = True
        else:
            with self.position_lock:
                self.tracking_active = False

    def set_laser(self, enable):
        """控制激光开关"""
        # 防止重复设置
        if self.laser_on == enable:
            return

        self.laser_on = enable

        # 发布激光控制消息
        laser_msg = Bool()
        laser_msg.data = enable
        self.laser_pub.publish(laser_msg)

        # 控制实际硬件
        if self.galvo_controller:
            try:
                if enable:
                    self.galvo_controller.send_command("LASER:ON")
                    rospy.loginfo(f"LASER ON for target ID: {self.current_target_id}")
                else:
                    self.galvo_controller.send_command("LASER:OFF")
                    rospy.loginfo(f"LASER OFF for target ID: {self.current_target_id}")
            except Exception as e:
                rospy.logerr(f"Failed to control laser: {e}")

    def update_target_with_prediction(self, target_info, current_time):
        """更新目标并计算预测位置"""
        if not target_info or 'centroid' not in target_info:
            return

        # 添加到位置历史
        position_data = {
            'position': target_info['centroid'],
            'timestamp': current_time,
            'camera_capture_time': current_time - self.camera_delay  # 估计实际拍摄时间
        }
        self.position_history.append(position_data)

        # 计算运动参数
        self.calculate_motion_parameters()

        # 更新卡尔曼滤波器
        if self.use_kalman_filter:
            self.update_kalman_filter(target_info['centroid'])

        # 计算预测位置
        predicted_position = self.predict_future_position(current_time + self.prediction_time)

        if predicted_position:
            # 转换为振镜坐标
            galvo_x, galvo_y = self.pixel_to_galvo(predicted_position[0], predicted_position[1])

            # 限制预测距离
            current_galvo = self.pixel_to_galvo(target_info['centroid'][0], target_info['centroid'][1])
            prediction_distance = np.sqrt((galvo_x - current_galvo[0]) ** 2 + (galvo_y - current_galvo[1]) ** 2)

            if prediction_distance > self.max_prediction_distance:
                # 限制预测距离
                scale = self.max_prediction_distance / prediction_distance
                galvo_x = current_galvo[0] + (galvo_x - current_galvo[0]) * scale
                galvo_y = current_galvo[1] + (galvo_y - current_galvo[1]) * scale

            # 更新目标振镜位置
            with self.position_lock:
                self.target_galvo_position = [galvo_x, galvo_y]

    def calculate_motion_parameters(self):
        """计算运动参数（速度、加速度）"""
        if len(self.position_history) < 2:
            return

        # 计算速度
        recent_positions = list(self.position_history)[-3:]  # 使用最近3个位置

        if len(recent_positions) >= 2:
            velocities = []
            for i in range(1, len(recent_positions)):
                dt = recent_positions[i]['timestamp'] - recent_positions[i - 1]['timestamp']
                if dt > 0:
                    dx = recent_positions[i]['position'][0] - recent_positions[i - 1]['position'][0]
                    dy = recent_positions[i]['position'][1] - recent_positions[i - 1]['position'][1]
                    velocities.append([dx / dt, dy / dt])

            if velocities:
                # 平均速度
                avg_velocity = np.mean(velocities, axis=0)
                self.velocity_history.append({
                    'velocity': avg_velocity,
                    'timestamp': recent_positions[-1]['timestamp']
                })

        # 计算加速度
        if len(self.velocity_history) >= 2:
            recent_velocities = list(self.velocity_history)[-2:]
            dt = recent_velocities[1]['timestamp'] - recent_velocities[0]['timestamp']
            if dt > 0:
                dv = recent_velocities[1]['velocity'] - recent_velocities[0]['velocity']
                acceleration = dv / dt
                self.acceleration_history.append({
                    'acceleration': acceleration,
                    'timestamp': recent_velocities[1]['timestamp']
                })

    def update_kalman_filter(self, observed_position):
        """更新卡尔曼滤波器"""
        if self.kalman_filter is None:
            self.initialize_kalman_filter(observed_position)
            return

        try:
            # 预测步骤
            predicted = self.kalman_filter.predict()

            # 更新步骤
            measurement = np.array([[observed_position[0]], [observed_position[1]]], dtype=np.float32)
            self.kalman_filter.correct(measurement)

        except Exception as e:
            rospy.logwarn(f"Kalman filter update failed: {e}")

    def predict_future_position(self, target_time):
        """预测未来位置"""
        if not self.position_history:
            return None

        current_time = time.time()
        prediction_dt = target_time - current_time

        if self.use_kalman_filter and self.kalman_filter is not None:
            # 使用卡尔曼滤波器预测
            return self.predict_with_kalman(prediction_dt)
        else:
            # 使用运动学模型预测
            return self.predict_with_kinematics(prediction_dt)

    def predict_with_kalman(self, dt):
        """使用卡尔曼滤波器预测"""
        if self.kalman_filter is None:
            return None

        try:
            # 创建预测用的转移矩阵
            prediction_matrix = np.array([
                [1, 0, dt, 0, 0.5 * dt * dt, 0],
                [0, 1, 0, dt, 0, 0.5 * dt * dt],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)

            # 获取当前状态
            current_state = self.kalman_filter.statePost.copy()

            # 预测未来状态
            future_state = prediction_matrix.dot(current_state)

            return [float(future_state[0]), float(future_state[1])]

        except Exception as e:
            rospy.logwarn(f"Kalman prediction failed: {e}")
            return None

    def predict_with_kinematics(self, dt):
        """使用运动学模型预测"""
        if len(self.position_history) < 2:
            return self.position_history[-1]['position'] if self.position_history else None

        # 获取最新位置
        latest = self.position_history[-1]
        current_pos = latest['position']

        # 估计速度
        velocity = [0, 0]
        if len(self.velocity_history) > 0:
            velocity = self.velocity_history[-1]['velocity']

        # 估计加速度
        acceleration = [0, 0]
        if len(self.acceleration_history) > 0:
            acceleration = self.acceleration_history[-1]['acceleration']

        # 运动学预测: s = s0 + v*t + 0.5*a*t^2
        predicted_x = current_pos[0] + velocity[0] * dt + 0.5 * acceleration[0] * dt * dt
        predicted_y = current_pos[1] + velocity[1] * dt + 0.5 * acceleration[1] * dt * dt

        return [predicted_x, predicted_y]

    def galvo_control_loop(self):
        """独立的振镜控制循环"""
        rate = rospy.Rate(1000)  # 1000Hz 高频控制

        while self.control_thread_active and not rospy.is_shutdown():
            try:
                # 只有在跟踪状态时才进行预测和控制
                if self.tracking_active:
                    # 获取当前目标的完整信息
                    current_target_info = self.get_current_target_info()

                    if current_target_info and 'centroid' in current_target_info:
                        current_time = time.time()
                        # 对当前目标进行预测和跟踪
                        self.update_target_with_prediction(current_target_info, current_time)

                        with self.position_lock:
                            if self.target_galvo_position:
                                target_x, target_y = self.target_galvo_position

                                # 检查是否需要移动
                                movement = np.sqrt((target_x - self.last_galvo_position[0]) ** 2 +
                                                   (target_y - self.last_galvo_position[1]) ** 2)

                                if movement > 5:  # 最小移动阈值
                                    # 发送到振镜
                                    if self.galvo_controller:
                                        self.galvo_controller.move_to_position(target_x, target_y)

                                    self.last_galvo_position = [target_x, target_y]
                                    self.galvo_timestamps.append(time.time())

                                    # 发布位置
                                    xy_msg = Int32MultiArray()
                                    xy_msg.data = [target_x, target_y, 1 if self.laser_on else 0]
                                    self.xy_pub.publish(xy_msg)

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"Galvo control loop error: {e}")
                time.sleep(0.001)

    def performance_callback(self, event):
        """性能监控回调"""
        if len(self.frame_timestamps) > 1:
            frame_intervals = np.diff(self.frame_timestamps)
            avg_fps = 1.0 / np.mean(frame_intervals) if len(frame_intervals) > 0 else 0

            total_delay = self.total_system_delay * 1000
            prediction_time = self.prediction_time * 1000

            status_msg = (f"Performance: FPS={avg_fps:.1f}, "
                          f"State={self.system_state.value}, "
                          f"CurrentID={self.current_target_id}, "
                          f"Queue={len(self.target_queue)}, "
                          f"Processed={len(self.processed_targets)}, "
                          f"Targets={len(self.all_detected_targets)}, "
                          f"Laser={'ON' if self.laser_on else 'OFF'}")

            rospy.loginfo(status_msg)

            # 发布预测信息
            current_target_info = self.get_current_target_info()
            if current_target_info:
                self.publish_prediction_info()

    def publish_prediction_info(self):
        """发布预测信息"""
        current_target_info = self.get_current_target_info()
        if not current_target_info or 'centroid' not in current_target_info:
            return

        # 计算当前预测位置
        predicted_pos = self.predict_future_position(time.time() + self.prediction_time)

        info = {
            'current_position': current_target_info['centroid'],
            'predicted_position': predicted_pos,
            'prediction_time_ms': self.prediction_time * 1000,
            'system_delay_ms': self.total_system_delay * 1000,
            'galvo_position': self.last_galvo_position,
            'velocity': self.velocity_history[-1]['velocity'].tolist() if self.velocity_history else [0, 0],
            'acceleration': self.acceleration_history[-1]['acceleration'].tolist() if self.acceleration_history else [0,
                                                                                                                      0]
        }

        msg = String()
        msg.data = json.dumps(info)
        self.prediction_pub.publish(msg)

    def draw_prediction_info(self, image):
        """绘制预测信息"""
        result_image = image.copy()

        # 绘制所有检测到的目标
        for track_id, target_info in self.all_detected_targets.items():
            bbox = target_info['bbox']
            centroid = target_info['centroid']
            confidence = target_info['confidence']

            x, y, w, h = bbox

            # 选择颜色和标签
            color = (0, 255, 0)  # 默认绿色
            label = f"ID:{track_id}"
            thickness = 1

            if track_id in self.processed_targets:
                color = (128, 128, 128)  # 灰色 - 已处理
                label += " [DONE]"
            elif track_id == self.current_target_id:
                if self.system_state == SystemState.FIRING:
                    color = (0, 0, 255)  # 红色 - 激光照射中
                    label += " [LASER]"
                elif self.system_state == SystemState.AIMING:
                    color = (0, 255, 255)  # 黄色 - 瞄准中
                    label += " [AIMING]"
                elif self.system_state == SystemState.TRACKING:
                    color = (0, 255, 255)  # 黄色 - 跟踪中
                    label += " [TRACKING]"
                thickness = 2
            elif track_id in self.target_queue:
                color = (255, 255, 0)  # 青色 - 队列中
                label += " [QUEUE]"
                thickness = 2

            # 绘制边界框
            cv2.rectangle(result_image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)

            # 绘制中心点
            cv2.circle(result_image, (int(centroid[0]), int(centroid[1])), 3, color, -1)

            # 绘制标签
            cv2.putText(result_image, label, (int(x), int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 绘制当前目标的预测信息
        current_target_info = self.get_current_target_info()
        if current_target_info and 'centroid' in current_target_info:
            current_pos = current_target_info['centroid']

            # 绘制预测位置
            predicted_pos = self.predict_future_position(time.time() + self.prediction_time)
            # if predicted_pos:
            #     cv2.circle(result_image, (int(predicted_pos[0]), int(predicted_pos[1])), 5, (0, 0, 255), -1)
            #
            #     # 绘制预测轨迹
            #     cv2.arrowedLine(result_image,
            #                     (int(current_pos[0]), int(current_pos[1])),
            #                     (int(predicted_pos[0]), int(predicted_pos[1])),
            #                     (255, 0, 255), 2)
            #
            #     # 显示预测时间
            #     cv2.putText(result_image, f"Pred: {self.prediction_time * 1000:.0f}ms",
            #                 (int(predicted_pos[0] + 10), int(predicted_pos[1])),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            # 绘制振镜位置
            if self.tracking_active:
                galvo_pixel = self.galvo_to_pixel(self.last_galvo_position[0], self.last_galvo_position[1])

                # 根据激光状态选择颜色
                if self.laser_on:
                    galvo_color = (0, 0, 255)  # 红色
                    # 绘制激光效果
                    for radius in [15, 25, 35]:
                        alpha = 1.0 - (radius / 35.0) * 0.6
                        overlay = result_image.copy()
                        cv2.circle(overlay, (int(galvo_pixel[0]), int(galvo_pixel[1])), radius, galvo_color, 1)
                        # cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)
                else:
                    galvo_color = (255, 255, 0)  # 黄色

                cv2.circle(result_image, (int(galvo_pixel[0]), int(galvo_pixel[1])), 8, galvo_color, 1)
                cv2.putText(result_image, "GALVO",
                            (int(galvo_pixel[0] + 10), int(galvo_pixel[1] + 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, galvo_color, 1)
                # 绘制十字线
                cv2.line(result_image,
                         (int(galvo_pixel[0] - 15), int(galvo_pixel[1])),
                         (int(galvo_pixel[0] + 15), int(galvo_pixel[1])),
                         galvo_color, 1)
                cv2.line(result_image,
                         (int(galvo_pixel[0]), int(galvo_pixel[1] - 15)),
                         (int(galvo_pixel[0]), int(galvo_pixel[1] + 15)),
                         galvo_color, 1)

            # 显示速度信息
            if self.velocity_history:
                velocity = self.velocity_history[-1]['velocity']
                speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
                cv2.putText(result_image, f"Speed: {speed:.1f}px/s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 绘制系统状态信息
        info_lines = [
            f"Frame: {self.frame_count}",
            f"State: {self.system_state.value}",
            f"Current: {self.current_target_id if self.current_target_id else 'None'}",
            f"Queue: {len(self.target_queue)}",
            f"Processed: {len(self.processed_targets)}",
            f"Targets: {len(self.all_detected_targets)}",
            f"Laser: {'ON' if self.laser_on else 'OFF'}",
            f"Pred: {self.prediction_time * 1000:.0f}ms"
        ]

        # # 绘制状态信息背景
        # font_scale = 0.4
        # line_height = 18
        # box_width = 180
        # box_height = len(info_lines) * line_height + 10
        #
        # if self.laser_on:
        #     bg_color = (0, 0, 80)  # 深红背景
        #     border_color = (0, 0, 255)  # 红色边框
        # else:
        #     bg_color = (40, 40, 40)  # 深灰背景
        #     border_color = (200, 200, 200)  # 浅灰边框
        #
        # cv2.rectangle(result_image, (5, 5), (box_width, box_height), bg_color, -1)
        # cv2.rectangle(result_image, (5, 5), (box_width, box_height), border_color, 1)
        #
        # # 绘制状态信息文字
        # for i, line in enumerate(info_lines):
        #     y_pos = 20 + i * line_height
        #     cv2.putText(result_image, line, (10, y_pos),
        #                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        return result_image

    def pixel_to_galvo(self, pixel_x, pixel_y):
        """像素到振镜坐标转换"""
        if self.galvo_controller:
            return self.galvo_controller.pixel_to_galvo(
                pixel_x, pixel_y,
                self.image_width, self.image_height
            )
        else:
            x = int((pixel_x / self.image_width - 0.5) * 60000)
            y = int((pixel_y / self.image_height - 0.5) * 60000)
            return max(-30000, min(30000, x)), max(-30000, min(30000, y))

    def galvo_to_pixel(self, galvo_x, galvo_y):
        """振镜到像素坐标转换"""
        pixel_x = (galvo_x / 60000 + 0.5) * self.image_width
        pixel_y = (galvo_y / 60000 + 0.5) * self.image_height
        return [pixel_x, pixel_y]

    def publish_target_info(self):
        """发布目标信息"""
        current_target_info = self.get_current_target_info()
        if current_target_info:
            target_info = {
                'track_id': current_target_info.get('track_id', -1),
                'confidence': current_target_info.get('confidence', 0.0),
                'current_position': current_target_info.get('centroid', [0, 0]),
                'galvo_position': self.last_galvo_position,
                'tracking_active': self.tracking_active,
                'prediction_active': self.use_kalman_filter,
                'system_state': self.system_state.value,
                'laser_on': self.laser_on
            }
            target_msg = String()
            target_msg.data = json.dumps(target_info)
            self.target_pub.publish(target_msg)

        # 发布系统状态
        status_info = {
            'state': self.system_state.value,
            'current_target_id': self.current_target_id,
            'queue_size': len(self.target_queue),
            'processed_count': len(self.processed_targets),
            'total_targets': len(self.all_detected_targets),
            'laser_on': self.laser_on,
            'tracking_active': self.tracking_active,
            'prediction_time_ms': self.prediction_time * 1000,
            'system_delay_ms': self.total_system_delay * 1000
        }
        status_msg = String()
        status_msg.data = json.dumps(status_info)
        self.status_pub.publish(status_msg)

    def __del__(self):
        """析构函数"""
        self.control_thread_active = False
        if hasattr(self, 'galvo_controller') and self.galvo_controller:
            try:
                self.set_laser(False)
                self.galvo_controller.close()
            except:
                pass


def main():
    """主函数"""
    try:
        rospy.loginfo("Starting Predictive Laser Weeding Node...")
        node = PredictiveLaserWeedingNode()
        rospy.loginfo("Node created, entering spin...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received, shutting down...")
    except Exception as e:
        rospy.logerr(f"Fatal error in main: {e}")
        rospy.logerr(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
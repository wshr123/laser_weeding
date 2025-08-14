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
from std_msgs.msg import Int32MultiArray, Float32MultiArray, String, Bool
from detector import WeedDetector
import json
import time
from collections import deque
from enum import Enum
import threading

# 导入振镜控制器
from one_object_to_teensy import XY2_100Controller


class SystemState(Enum):
    """系统状态枚举"""
    IDLE = "IDLE"  # 空闲，等待目标
    TRACKING = "TRACKING"  # 跟踪目标
    FIRING = "FIRING"  # 激光照射中


class LaserWeedingNode:
    def __init__(self):
        try:
            rospy.init_node('laser_weeding_node', anonymous=True)
            rospy.loginfo("=" * 50)
            rospy.loginfo("Starting Simplified Laser Weeding Node...")
            rospy.loginfo("=" * 50)

            # 基础组件
            self.bridge = CvBridge()

            # ========== 参数加载 ==========
            # 模型参数
            self.model_path = rospy.get_param('~model_path', '')
            self.model_type = rospy.get_param('~model_type', 'yolov11')
            self.device = rospy.get_param('~device', '0')
            self.weed_class_id = rospy.get_param('~weed_class_id', 0)
            self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.3)

            # 预测参数（简化版）
            self.total_delay = rospy.get_param('~total_delay', 0.08)  # 总系统延迟
            self.prediction_time = rospy.get_param('~prediction_time', 0.15)  # 预测时间
            self.use_simple_kalman = rospy.get_param('~use_kalman', False)  # 是否使用简单卡尔曼
            self.max_prediction_distance = rospy.get_param('~max_prediction_distance', 300)  # 最大预测距离(像素)

            # 激光控制参数
            self.aiming_time = rospy.get_param('~aiming_time', 0.1)  # 瞄准时间
            self.laser_time = rospy.get_param('~laser_time', 0.2)  # 激光持续时间

            # 振镜参数
            self.serial_port = rospy.get_param('~serial_port', '/dev/ttyACM0')
            self.serial_baudrate = rospy.get_param('~serial_baudrate', 115200)
            self.min_move_step = rospy.get_param('~min_move_step', 2)  # 最小移动步长

            # 图像参数
            self.image_width = rospy.get_param('~image_width', 640)
            self.image_height = rospy.get_param('~image_height', 480)

            # 目标管理参数
            self.target_timeout = rospy.get_param('~target_timeout', 0.5)  # 目标丢失超时
            self.min_stable_frames = rospy.get_param('~min_stable_frames', 2)  # 最小稳定帧数

            # ========== 振镜控制器初始化 ==========
            try:
                self.galvo_controller = XY2_100Controller(
                    port=self.serial_port,
                    baudrate=self.serial_baudrate
                )
                rospy.loginfo(f"Galvo controller initialized on {self.serial_port}")
            except Exception as e:
                rospy.logwarn(f"Failed to initialize galvo controller: {e}")
                self.galvo_controller = None

            # ========== 检测器初始化 ==========
            try:
                self.detector = WeedDetector(
                    model_path=self.model_path,
                    model_type=self.model_type,
                    weed_class_id=self.weed_class_id,
                    crop_class_id=1,  # 默认作物ID
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                    tracker_type='custom'
                )
                rospy.loginfo(f'{self.model_type.upper()} model loaded successfully')
            except Exception as e:
                rospy.logerr(f"Failed to load detection model: {e}")
                rospy.logerr(traceback.format_exc())
                sys.exit(1)

            # ========== 状态变量初始化 ==========
            # 系统状态
            self.system_state = SystemState.IDLE
            self.state_start_time = time.time()

            # 目标管理 - 修复：添加更严格的已处理目标管理
            self.current_target = None  # 当前跟踪的目标
            self.target_queue = []  # 待处理目标队列
            self.processed_targets = set()  # 已处理目标集合（永久记录）
            self.processing_target = None  # 正在处理的目标ID（防止重入）
            self.all_targets = {}  # 所有检测到的目标

            # 位置历史（用于预测）
            self.position_history = deque(maxlen=20)
            self.last_update_time = time.time()

            # 振镜控制
            self.galvo_position = [0, 0]  # 当前振镜位置
            self.target_galvo_position = [0, 0]  # 目标振镜位置
            self.laser_on = False
            self.tracking_active = False

            # 简单卡尔曼滤波器
            self.kalman_filter = None
            if self.use_simple_kalman:
                self.init_kalman_filter()

            # 线程控制
            self.running = True
            self.position_lock = threading.Lock()

            # 性能监控
            self.frame_count = 0
            self.fps_counter = deque(maxlen=30)

            # 图像缓存
            self.current_image = None

            # ========== ROS 发布器和订阅器 ==========
            # 发布器
            self.galvo_pub = rospy.Publisher('/galvo_xy', Int32MultiArray, queue_size=1)
            self.laser_pub = rospy.Publisher('/laser_control', Bool, queue_size=1)
            self.det_img_pub = rospy.Publisher('/det_img/image_raw', Image, queue_size=1)
            self.status_pub = rospy.Publisher('/system_status', String, queue_size=1)
            self.target_pub = rospy.Publisher('/current_target', String, queue_size=1)

            # 订阅器
            image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
            self.image_sub = rospy.Subscriber(
                image_topic,
                Image,
                self.image_callback,
                queue_size=1
            )

            # ========== 启动控制线程 ==========
            self.galvo_thread = threading.Thread(target=self.galvo_control_loop)
            self.galvo_thread.daemon = True
            self.galvo_thread.start()

            # 主控制定时器
            self.control_timer = rospy.Timer(
                rospy.Duration(0.01),  # 100Hz
                self.control_loop
            )

            # 状态发布定时器
            self.status_timer = rospy.Timer(
                rospy.Duration(0.5),  # 2Hz
                self.publish_status
            )

            rospy.loginfo("=" * 50)
            rospy.loginfo("Simplified Laser Weeding Node Ready!")
            rospy.loginfo(f"Total delay: {self.total_delay * 1000:.1f}ms")
            rospy.loginfo(f"Prediction time: {self.prediction_time * 1000:.1f}ms")
            rospy.loginfo("=" * 50)

        except Exception as e:
            rospy.logerr(f"Failed to initialize node: {e}")
            rospy.logerr(traceback.format_exc())
            sys.exit(1)

    def image_callback(self, msg):
        """图像回调函数"""
        try:
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image

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

            # 发布检测结果图像（降低频率）
            if self.frame_count % 3 == 0:  # 每3帧发布一次
                try:
                    det_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                    self.det_img_pub.publish(det_msg)
                except CvBridgeError as e:
                    rospy.logerr(f"Failed to publish image: {e}")

        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")

    def control_loop(self, event):
        """主控制循环（100Hz）- 修复版本"""
        try:
            current_time = time.time()

            # 状态机逻辑
            if self.system_state == SystemState.IDLE:
                # 从队列获取下一个目标
                if self.target_queue and not self.current_target:
                    # 再次过滤已处理的目标（双重保险）
                    while self.target_queue:
                        target_id = self.target_queue[0]
                        if target_id in self.processed_targets:
                            # 如果发现已处理的目标在队列中，移除它
                            self.target_queue.pop(0)
                            rospy.logwarn(f"Removed processed target {target_id} from queue")
                        else:
                            # 找到未处理的目标
                            break

                    if self.target_queue:
                        target_id = self.target_queue.pop(0)
                        if target_id in self.all_targets and target_id not in self.processed_targets:
                            self.current_target = {
                                'id': target_id,
                                'start_time': current_time
                            }
                            self.processing_target = target_id  # 标记为正在处理
                            self.position_history.clear()

                            if self.use_simple_kalman and self.kalman_filter:
                                # 重置卡尔曼滤波器
                                center = self.all_targets[target_id]['center']
                                self.kalman_filter.statePre = np.array(
                                    [[center[0]], [center[1]], [0], [0]],
                                    dtype=np.float32
                                )

                            self.change_state(SystemState.TRACKING)
                            rospy.loginfo(f"Start tracking target {target_id}")

            elif self.system_state == SystemState.TRACKING:
                if self.current_target:
                    target_id = self.current_target['id']

                    # 再次检查是否已处理（防止并发问题）
                    if target_id in self.processed_targets:
                        rospy.logwarn(f"Target {target_id} already processed, skipping")
                        self.current_target = None
                        self.processing_target = None
                        self.change_state(SystemState.IDLE)
                        return

                    if target_id in self.all_targets:
                        # 更新位置历史
                        target_info = self.all_targets[target_id]
                        self.update_position_history(target_info['center'], current_time)

                        # 检查是否达到瞄准时间
                        elapsed = current_time - self.current_target['start_time']
                        if elapsed >= self.aiming_time:
                            self.change_state(SystemState.FIRING)
                            rospy.loginfo(f"Start firing at target {target_id}")
                    else:
                        # 目标丢失
                        rospy.logwarn(f"Target {target_id} lost during tracking")
                        self.current_target = None
                        self.processing_target = None
                        self.change_state(SystemState.IDLE)

            elif self.system_state == SystemState.FIRING:
                if self.current_target:
                    target_id = self.current_target['id']

                    # 继续更新位置（如果目标还在）
                    if target_id in self.all_targets:
                        target_info = self.all_targets[target_id]
                        self.update_position_history(target_info['center'], current_time)

                    # 检查激光时间
                    elapsed = current_time - self.state_start_time
                    if elapsed >= self.laser_time:
                        # 完成处理 - 关键修复：确保目标被标记为已处理
                        self.processed_targets.add(target_id)

                        # 更新目标信息中的处理状态
                        if target_id in self.all_targets:
                            self.all_targets[target_id]['processed'] = True

                        rospy.loginfo(f"Target {target_id} processed and added to processed_targets")
                        rospy.loginfo(f"Total processed targets: {len(self.processed_targets)}")

                        # 清理当前目标
                        self.current_target = None
                        self.processing_target = None
                        self.change_state(SystemState.IDLE)

        except Exception as e:
            rospy.logerr(f"Control loop error: {e}")

    def init_kalman_filter(self):
        """初始化简单的2D卡尔曼滤波器（位置+速度）"""
        self.kalman_filter = cv2.KalmanFilter(4, 2)  # 4个状态变量，2个测量值

        # 状态转移矩阵 [x, y, vx, vy]
        dt = 0.033  # 假设30FPS
        self.kalman_filter.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 测量矩阵（只能测量位置）
        self.kalman_filter.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 过程噪声
        self.kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        # 测量噪声
        self.kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        # 初始协方差
        self.kalman_filter.errorCovPost = np.eye(4, dtype=np.float32) * 100


    def update_targets(self, detections, current_time):
        """更新目标信息 - 修复版本"""
        # 清空当前帧的目标
        current_frame_ids = set()

        for track_id, bbox, confidence in detections:
            if confidence < self.confidence_threshold:
                continue

            # 修复：首先检查是否是已处理的目标
            if track_id in self.processed_targets:
                # 已处理的目标仍然更新其位置信息（用于显示），但不会重新处理
                current_frame_ids.add(track_id)

                # 计算中心点
                x, y, w, h = bbox
                cx = x + w / 2
                cy = y + h / 2

                # 更新已处理目标的信息（仅用于显示）
                if track_id not in self.all_targets:
                    self.all_targets[track_id] = {
                        'first_seen': current_time,
                        'stable_frames': 0,
                        'processed': True  # 标记为已处理
                    }

                self.all_targets[track_id].update({
                    'bbox': bbox,
                    'center': [cx, cy],
                    'confidence': confidence,
                    'last_seen': current_time,
                    'processed': True  # 确保标记为已处理
                })

                # 重要：跳过后续处理，不加入队列
                continue

            # 对于未处理的目标，正常处理
            # 计算中心点
            x, y, w, h = bbox
            cx = x + w / 2
            cy = y + h / 2

            # 更新目标信息
            if track_id not in self.all_targets:
                self.all_targets[track_id] = {
                    'first_seen': current_time,
                    'stable_frames': 0,
                    'processed': False
                }

            self.all_targets[track_id].update({
                'bbox': bbox,
                'center': [cx, cy],
                'confidence': confidence,
                'last_seen': current_time,
                'stable_frames': self.all_targets[track_id]['stable_frames'] + 1
            })

            current_frame_ids.add(track_id)

            # 添加到队列的条件更严格
            if (track_id not in self.processed_targets and  # 未处理过
                    track_id not in self.target_queue and  # 不在队列中
                    track_id != self.processing_target and  # 不是正在处理的目标
                    (not self.current_target or self.current_target['id'] != track_id) and  # 不是当前目标
                    self.all_targets[track_id]['stable_frames'] >= self.min_stable_frames):  # 足够稳定

                self.target_queue.append(track_id)
                rospy.loginfo(f"New target added to queue: ID {track_id}")

        # 清理超时目标
        to_remove = []
        for tid, tinfo in self.all_targets.items():
            if tid not in current_frame_ids:
                # 对于已处理的目标，保留更长时间（避免误删）
                timeout = self.target_timeout * 3 if tid in self.processed_targets else self.target_timeout
                if current_time - tinfo['last_seen'] > timeout:
                    to_remove.append(tid)

        for tid in to_remove:
            del self.all_targets[tid]
            if tid in self.target_queue:
                self.target_queue.remove(tid)
            if self.current_target and self.current_target['id'] == tid:
                rospy.logwarn(f"Current target {tid} lost")
                self.current_target = None
                self.processing_target = None
                self.change_state(SystemState.IDLE)


    def update_position_history(self, position, timestamp):
        """更新位置历史并计算预测位置"""
        # 添加到历史
        self.position_history.append({
            'position': position,
            'time': timestamp
        })

        # 计算预测位置
        predicted_pos = self.predict_position(self.prediction_time)

        if predicted_pos:
            # 检查预测距离是否合理
            current_pos = position
            distance = np.sqrt((predicted_pos[0] - current_pos[0]) ** 2 +
                               (predicted_pos[1] - current_pos[1]) ** 2)

            if distance > self.max_prediction_distance:
                rospy.logdebug(f"Prediction distance too large: {distance:.1f}px")
                predicted_pos = current_pos

            # 转换为振镜坐标
            galvo_x, galvo_y = self.pixel_to_galvo(predicted_pos[0], predicted_pos[1])

            with self.position_lock:
                self.target_galvo_position = [galvo_x, galvo_y]

    def predict_position(self, dt):
        """预测未来位置"""
        if len(self.position_history) < 2:
            return self.position_history[-1]['position'] if self.position_history else None

        # 使用卡尔曼滤波
        if self.use_simple_kalman and self.kalman_filter:
            try:
                # 更新卡尔曼滤波
                current_pos = self.position_history[-1]['position']
                measurement = np.array([[current_pos[0]], [current_pos[1]]], dtype=np.float32)

                self.kalman_filter.correct(measurement)
                prediction = self.kalman_filter.predict()

                # 根据dt进行外推
                state = self.kalman_filter.statePost
                pred_x = state[0, 0] + state[2, 0] * dt
                pred_y = state[1, 0] + state[3, 0] * dt

                return [float(pred_x), float(pred_y)]
            except Exception as e:
                rospy.logdebug(f"Kalman prediction failed: {e}")

        # 简单线性预测（备用方案）
        if len(self.position_history) >= 2:
            p1 = self.position_history[-2]
            p2 = self.position_history[-1]

            time_diff = p2['time'] - p1['time']
            if time_diff > 0:
                vx = (p2['position'][0] - p1['position'][0]) / time_diff
                vy = (p2['position'][1] - p1['position'][1]) / time_diff

                pred_x = p2['position'][0] + vx * dt
                pred_y = p2['position'][1] + vy * dt

                return [pred_x, pred_y]

        return self.position_history[-1]['position']

    def galvo_control_loop(self):
        """振镜控制线程（高频率）"""
        rate = rospy.Rate(500)  # 500Hz

        while self.running and not rospy.is_shutdown():
            try:
                # if self.tracking_active:
                with self.position_lock:
                    target_pos = self.target_galvo_position.copy()

                # 计算移动距离
                dx = target_pos[0] - self.galvo_position[0]
                dy = target_pos[1] - self.galvo_position[1]
                distance = np.sqrt(dx ** 2 + dy ** 2)

                # 只有超过最小步长才移动
                if distance > self.min_move_step:
                    # 发送振镜命令
                    if self.galvo_controller:
                        self.galvo_controller.move_to_position(
                            int(target_pos[0]),
                            int(target_pos[1])
                        )

                    self.galvo_position = target_pos

                    # 发布振镜位置
                    galvo_msg = Int32MultiArray()
                    galvo_msg.data = [
                        int(target_pos[0]),
                        int(target_pos[1]),
                        1 if self.laser_on else 0
                    ]
                    self.galvo_pub.publish(galvo_msg)

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"Galvo control error: {e}")
                time.sleep(0.001)

    def change_state(self, new_state):
        """改变系统状态"""
        old_state = self.system_state
        self.system_state = new_state
        self.state_start_time = time.time()

        rospy.loginfo(f"State: {old_state.value} -> {new_state.value}")

        # 根据状态控制激光和跟踪
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

            # 发布激光状态
            laser_msg = Bool()
            laser_msg.data = enable
            self.laser_pub.publish(laser_msg)

            # 控制硬件
            if self.galvo_controller:
                try:
                    if enable:
                        self.galvo_controller.send_command("LASER:ON")
                    else:
                        self.galvo_controller.send_command("LASER:OFF")
                except Exception as e:
                    rospy.logerr(f"Failed to control laser: {e}")

    def draw_info(self, image):
        """在图像上绘制信息 - 修复版本"""
        result = image.copy()

        # 绘制所有目标
        for track_id, target_info in self.all_targets.items():
            if 'bbox' not in target_info:
                continue

            bbox = target_info['bbox']
            center = target_info['center']
            x, y, w, h = bbox

            # 选择颜色 - 修复：优先检查processed_targets
            if track_id in self.processed_targets or target_info.get('processed', False):
                color = (128, 128, 128)  # 灰色：已处理
                label = f"ID:{track_id} [DONE]"
                thickness = 1
            elif self.current_target and self.current_target['id'] == track_id:
                if self.system_state == SystemState.FIRING:
                    color = (0, 0, 255)  # 红色：激光照射
                    label = f"ID:{track_id} [LASER]"
                else:
                    color = (0, 255, 255)  # 黄色：跟踪中
                    label = f"ID:{track_id} [TRACKING]"
                thickness = 2
            elif track_id in self.target_queue:
                color = (255, 255, 0)  # 青色：队列中
                label = f"ID:{track_id} [QUEUE]"
                thickness = 2
            else:
                color = (0, 255, 0)  # 绿色：检测到
                label = f"ID:{track_id}"
                thickness = 1

            # 绘制边界框
            cv2.rectangle(result, (int(x), int(y)),
                          (int(x + w), int(y + h)), color, thickness)

            # 绘制中心点
            cv2.circle(result, (int(center[0]), int(center[1])), 3, color, -1)

            # 绘制标签
            cv2.putText(result, label, (int(x), int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # # 绘制预测位置（如果正在跟踪）
        # if self.current_target and self.tracking_active:
        #     predicted_pos = self.predict_position(self.prediction_time)
        #     if predicted_pos:
        #         # 绘制预测位置
        #         cv2.circle(result,
        #                    (int(predicted_pos[0]), int(predicted_pos[1])),
        #                    8, (255, 0, 255), 2)
        #         cv2.putText(result, "PRED",
        #                     (int(predicted_pos[0] + 10), int(predicted_pos[1])),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # 绘制振镜位置
        # if self.tracking_active:
        galvo_pixel = self.galvo_to_pixel(self.galvo_position[0],
                                          self.galvo_position[1])
        color = (0, 0, 255) if self.laser_on else (255, 255, 0)

        # 绘制十字准线
        cv2.line(result,
                 (int(galvo_pixel[0] - 10), int(galvo_pixel[1])),
                 (int(galvo_pixel[0] + 10), int(galvo_pixel[1])),
                 color, 1)
        cv2.line(result,
                 (int(galvo_pixel[0]), int(galvo_pixel[1] - 10)),
                 (int(galvo_pixel[0]), int(galvo_pixel[1] + 10)),
                 color, 1)
        cv2.circle(result,
                   (int(galvo_pixel[0]), int(galvo_pixel[1])),
                   8, color, 1)
        # 显示状态信息
        status_text = f"State: {self.system_state.value}"
        cv2.putText(result, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示FPS
        if len(self.fps_counter) > 1:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(result, fps_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示队列信息 - 修复：显示正确的已处理数量
        queue_text = f"Queue: {len(self.target_queue)} | Processed: {len(self.processed_targets)}"
        cv2.putText(result, queue_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 添加调试信息
        if self.processing_target:
            debug_text = f"Processing: {self.processing_target}"
            cv2.putText(result, debug_text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        return result

    def pixel_to_galvo(self, pixel_x, pixel_y):
        """像素坐标转振镜坐标"""
        if self.galvo_controller:
            return self.galvo_controller.pixel_to_galvo(
                pixel_x, pixel_y,
                self.image_width, self.image_height
            )
        else:
            # 简单的线性映射
            galvo_x = int((pixel_x / self.image_width - 0.5) * 60000)
            galvo_y = int((pixel_y / self.image_height - 0.5) * 60000)
            return (
                max(-30000, min(30000, galvo_x)),
                max(-30000, min(30000, galvo_y))
            )

    def galvo_to_pixel(self, galvo_x, galvo_y):
        """振镜坐标转像素坐标"""
        pixel_x = (galvo_x / 60000 + 0.5) * self.image_width
        pixel_y = (galvo_y / 60000 + 0.5) * self.image_height
        return [pixel_x, pixel_y]

    def publish_status(self, event):
        """发布系统状态"""
        try:
            # 计算FPS
            fps = 0
            if len(self.fps_counter) > 1:
                fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])

            # 构建状态信息
            status_info = {
                'state': self.system_state.value,
                'current_target': self.current_target['id'] if self.current_target else None,
                'processing_target': self.processing_target,  # 添加正在处理的目标
                'queue_size': len(self.target_queue),
                'processed_count': len(self.processed_targets),
                'processed_targets': list(self.processed_targets),  # 添加已处理目标列表
                'total_targets': len(self.all_targets),
                'laser_on': self.laser_on,
                'fps': round(fps, 1),
                'galvo_position': self.galvo_position,
                'prediction_time_ms': self.prediction_time * 1000,
                'total_delay_ms': self.total_delay * 1000
            }

            # 发布状态
            status_msg = String()
            status_msg.data = json.dumps(status_info)
            self.status_pub.publish(status_msg)

            # 发布当前目标信息
            if self.current_target and self.current_target['id'] in self.all_targets:
                target_info = self.all_targets[self.current_target['id']]
                target_data = {
                    'id': self.current_target['id'],
                    'center': target_info['center'],
                    'confidence': target_info['confidence'],
                    'predicted_position': self.predict_position(self.prediction_time)
                }
                target_msg = String()
                target_msg.data = json.dumps(target_data)
                self.target_pub.publish(target_msg)

            # 打印简要状态
            rospy.loginfo(
                f"Status: {self.system_state.value} | "
                f"FPS: {fps:.1f} | "
                f"Queue: {len(self.target_queue)} | "
                f"Processed: {len(self.processed_targets)} | "
                f"Current: {self.current_target['id'] if self.current_target else 'None'} | "
                f"Laser: {'ON' if self.laser_on else 'OFF'}"
            )

        except Exception as e:
            rospy.logerr(f"Failed to publish status: {e}")

    def __del__(self):
        """析构函数"""
        self.running = False

        # 关闭激光
        self.set_laser(False)

        # 关闭振镜控制器
        if hasattr(self, 'galvo_controller') and self.galvo_controller:
            try:
                self.galvo_controller.close()
            except:
                pass

        rospy.loginfo("Laser weeding node shutdown complete")


def main():
    """主函数"""
    try:
        node = LaserWeedingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        rospy.logerr(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
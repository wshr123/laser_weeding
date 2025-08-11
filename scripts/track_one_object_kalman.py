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
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Float32, String
from detector import WeedDetector
import json
import time
from collections import defaultdict, deque
import threading
from scipy import interpolate

# 导入振镜控制器
from send_to_teensy import XY2_100Controller


class PredictiveGalvoTrackingNode:
    def __init__(self):
        try:
            rospy.init_node('predictive_galvo_tracking_node', anonymous=True)
            rospy.loginfo("=" * 50)
            rospy.loginfo("Starting Predictive Galvo Tracking Node...")
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
            self.max_prediction_distance = rospy.get_param('~max_prediction_distance', 100)  # 最大预测距离限制

            # 目标跟踪相关参数
            self.current_target = None
            self.min_confidence = rospy.get_param('~min_confidence', 0.3)
            self.target_stable_frames = rospy.get_param('~target_stable_frames', 2)

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
            self.device = rospy.get_param('~device', 'cpu')
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
            self.det_img_pub = rospy.Publisher('/det_img/image_raw', Image, queue_size=1)
            self.target_pub = rospy.Publisher('/current_target', String, queue_size=1)
            self.prediction_pub = rospy.Publisher('/prediction_info', String, queue_size=1)

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

            # 性能监控定时器
            self.performance_timer = rospy.Timer(
                rospy.Duration(2.0),
                self.performance_callback
            )

            rospy.loginfo("=" * 50)
            rospy.loginfo("Predictive Galvo Tracking Node Initialized!")
            rospy.loginfo(f"Total system delay: {self.total_system_delay * 1000:.1f}ms")
            rospy.loginfo(f"Prediction time: {self.prediction_time * 1000:.1f}ms")
            rospy.loginfo(f"Using Kalman filter: {self.use_kalman_filter}")
            rospy.loginfo("=" * 50)

        except Exception as e:
            rospy.logerr(f"Failed to initialize PredictiveGalvoTrackingNode: {e}")
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

            # 找到最高置信度目标
            highest_confidence_target = self.find_highest_confidence_weed(detection_results)

            # 更新目标和运动预测
            self.update_target_with_prediction(highest_confidence_target, current_time)

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

    def find_highest_confidence_weed(self, detection_results):
        """查找最高置信度杂草"""
        if not detection_results:
            return None

        confidences = [conf for _, _, conf in detection_results]
        if not confidences:
            return None

        max_idx = np.argmax(confidences)
        track_id, bbox, confidence = detection_results[max_idx]

        if confidence < self.min_confidence:
            return None

        x, y, w, h = bbox
        centroid = [x + w * 0.5, y + h * 0.5]

        return {
            'track_id': track_id,
            'bbox': bbox,
            'centroid': centroid,
            'confidence': confidence,
            'timestamp': time.time()
        }

    def update_target_with_prediction(self, new_target, current_time):
        """更新目标并计算预测位置"""
        if new_target is None:
            if self.current_target is not None:
                with self.position_lock:
                    self.current_target = None
                    self.tracking_active = False
            return

        # 更新当前目标
        self.current_target = new_target

        # 添加到位置历史
        position_data = {
            'position': new_target['centroid'],
            'timestamp': current_time,
            'camera_capture_time': current_time - self.camera_delay  # 估计实际拍摄时间
        }
        self.position_history.append(position_data)

        # 计算运动参数
        self.calculate_motion_parameters()

        # 更新卡尔曼滤波器
        if self.use_kalman_filter:
            self.update_kalman_filter(new_target['centroid'])

        # 计算预测位置
        predicted_position = self.predict_future_position(current_time + self.prediction_time)

        if predicted_position:
            # 转换为振镜坐标
            galvo_x, galvo_y = self.pixel_to_galvo(predicted_position[0], predicted_position[1])

            # 限制预测距离
            current_galvo = self.pixel_to_galvo(new_target['centroid'][0], new_target['centroid'][1])
            prediction_distance = np.sqrt((galvo_x - current_galvo[0]) ** 2 + (galvo_y - current_galvo[1]) ** 2)

            if prediction_distance > self.max_prediction_distance:
                # 限制预测距离
                scale = self.max_prediction_distance / prediction_distance
                galvo_x = current_galvo[0] + (galvo_x - current_galvo[0]) * scale
                galvo_y = current_galvo[1] + (galvo_y - current_galvo[1]) * scale

            # 更新目标振镜位置
            with self.position_lock:
                self.target_galvo_position = [galvo_x, galvo_y]
                self.tracking_active = True

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
                with self.position_lock:
                    if self.tracking_active and self.target_galvo_position:
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
                            xy_msg.data = [target_x, target_y, 1]
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

            rospy.loginfo(f"Performance: FPS={avg_fps:.1f}, "
                          f"System_delay={total_delay:.1f}ms, "
                          f"Prediction={prediction_time:.1f}ms, "
                          f"Tracking={'ACTIVE' if self.tracking_active else 'INACTIVE'}")

            # 发布预测信息
            if self.current_target:
                self.publish_prediction_info()

    def publish_prediction_info(self):
        """发布预测信息"""
        if not self.current_target:
            return

        # 计算当前预测位置
        predicted_pos = self.predict_future_position(time.time() + self.prediction_time)

        info = {
            'current_position': self.current_target['centroid'],
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

        if self.current_target:
            current_pos = self.current_target['centroid']

            # 绘制当前位置
            cv2.circle(result_image, (int(current_pos[0]), int(current_pos[1])), 5, (0, 255, 0), -1)

            # 绘制预测位置
            predicted_pos = self.predict_future_position(time.time() + self.prediction_time)
            if predicted_pos:
                cv2.circle(result_image, (int(predicted_pos[0]), int(predicted_pos[1])), 5, (0, 0, 255), -1)

                # 绘制预测轨迹
                cv2.arrowedLine(result_image,
                                (int(current_pos[0]), int(current_pos[1])),
                                (int(predicted_pos[0]), int(predicted_pos[1])),
                                (255, 0, 255), 2)

                # 显示预测时间
                cv2.putText(result_image, f"Pred: {self.prediction_time * 1000:.0f}ms",
                            (int(predicted_pos[0] + 10), int(predicted_pos[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # 绘制振镜位置
            if self.tracking_active:
                galvo_pixel = self.galvo_to_pixel(self.last_galvo_position[0], self.last_galvo_position[1])
                cv2.circle(result_image, (int(galvo_pixel[0]), int(galvo_pixel[1])), 8, (255, 255, 0), 2)
                cv2.putText(result_image, "GALVO",
                            (int(galvo_pixel[0] + 10), int(galvo_pixel[1] + 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # 显示速度信息
            if self.velocity_history:
                velocity = self.velocity_history[-1]['velocity']
                speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
                cv2.putText(result_image, f"Speed: {speed:.1f}px/s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
        if self.current_target:
            target_info = {
                'track_id': self.current_target['track_id'],
                'confidence': self.current_target['confidence'],
                'current_position': self.current_target['centroid'],
                'galvo_position': self.last_galvo_position,
                'tracking_active': self.tracking_active,
                'prediction_active': self.use_kalman_filter
            }
            target_msg = String()
            target_msg.data = json.dumps(target_info)
            self.target_pub.publish(target_msg)

    def __del__(self):
        """析构函数"""
        self.control_thread_active = False


def main():
    """主函数"""
    try:
        rospy.loginfo("Starting Predictive Galvo Tracking Node...")
        node = PredictiveGalvoTrackingNode()
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
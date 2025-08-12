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
from collections import OrderedDict, deque
from enum import Enum
import threading

class PredictiveKalmanFilter:
    """预测性卡尔曼滤波器"""

    def __init__(self):
        # 创建OpenCV卡尔曼滤波器
        # 状态向量：[x, y, vx, vy, ax, ay] (6维 - 位置、速度、加速度)
        # 观测向量：[x, y] (2维 - 只观测位置)
        self.kf = cv2.KalmanFilter(6, 2)

        # 时间步长
        self.dt = 0.033  # 30fps

        # 状态转移矩阵 (匀加速运动模型)
        self.update_transition_matrix(self.dt)

        # 观测矩阵 (只观测位置)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        # 过程噪声协方差矩阵
        self.update_process_noise(self.dt)

        # 观测噪声协方差矩阵
        self.kf.measurementNoiseCov = np.array([
            [2.0, 0],
            [0, 2.0]
        ], dtype=np.float32)

        # 误差协方差矩阵 - 初始不确定性
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1000

        # 初始化标志
        self.initialized = False
        self.last_update_time = None

        # 预测质量评估
        self.prediction_history = deque(maxlen=10)
        self.prediction_errors = deque(maxlen=20)

    def update_transition_matrix(self, dt):
        """更新状态转移矩阵"""
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

    def update_process_noise(self, dt):
        """更新过程噪声协方差矩阵"""
        # 适应性过程噪声
        q_pos = 1.0  # 位置噪声
        q_vel = 10.0  # 速度噪声
        q_acc = 100.0  # 加速度噪声

        self.kf.processNoiseCov = np.array([
            [q_pos * dt * dt * dt * dt / 4, 0, q_pos * dt * dt * dt / 2, 0, q_pos * dt * dt / 2, 0],
            [0, q_pos * dt * dt * dt * dt / 4, 0, q_pos * dt * dt * dt / 2, 0, q_pos * dt * dt / 2],
            [q_pos * dt * dt * dt / 2, 0, q_vel * dt * dt, 0, q_vel * dt, 0],
            [0, q_pos * dt * dt * dt / 2, 0, q_vel * dt * dt, 0, q_vel * dt],
            [q_pos * dt * dt / 2, 0, q_vel * dt, 0, q_acc, 0],
            [0, q_pos * dt * dt / 2, 0, q_vel * dt, 0, q_acc]
        ], dtype=np.float32)

    def initialize(self, x, y):
        """初始化滤波器"""
        # 初始状态：[x, y, 0, 0, 0, 0] (位置已知，速度和加速度为0)
        self.kf.statePre = np.array([x, y, 0, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([x, y, 0, 0, 0, 0], dtype=np.float32)
        self.initialized = True
        self.last_update_time = rospy.Time.now()

    def update(self, x, y):
        """更新滤波器"""
        current_time = rospy.Time.now()

        if not self.initialized:
            self.initialize(x, y)
            return x, y

        # 更新时间步长
        if self.last_update_time:
            dt = (current_time - self.last_update_time).to_sec()
            if 0.01 < dt < 0.5:  # 合理的时间间隔
                self.dt = dt
                self.update_transition_matrix(dt)
                self.update_process_noise(dt)

        self.last_update_time = current_time

        # 预测步骤
        prediction = self.kf.predict()

        # 记录预测误差（用于质量评估）
        if len(self.prediction_history) > 0:
            last_prediction = self.prediction_history[-1]
            error = np.sqrt((x - last_prediction[0]) ** 2 + (y - last_prediction[1]) ** 2)
            self.prediction_errors.append(error)

        # 更新步骤
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kf.correct(measurement)

        # 记录当前预测（用于下次误差计算）
        self.prediction_history.append([float(prediction[0]), float(prediction[1])])

        # 返回滤波后的位置
        return float(self.kf.statePost[0]), float(self.kf.statePost[1])

    def predict_future(self, time_ahead):
        """预测未来time_ahead秒后的位置"""
        if not self.initialized:
            return None, None

        # 创建预测用的状态转移矩阵
        dt = time_ahead
        prediction_matrix = np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # 使用当前状态预测未来状态
        current_state = self.kf.statePost.copy()
        future_state = prediction_matrix.dot(current_state)

        return float(future_state[0]), float(future_state[1])

    def get_motion_info(self):
        """获取运动信息"""
        if not self.initialized:
            return {
                'position': [0, 0],
                'velocity': [0, 0],
                'acceleration': [0, 0],
                'speed': 0,
                'prediction_quality': 0
            }

        state = self.kf.statePost
        velocity = [float(state[2]), float(state[3])]
        acceleration = [float(state[4]), float(state[5])]
        speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

        # 计算预测质量（基于历史误差）
        quality = 1.0
        if len(self.prediction_errors) > 0:
            avg_error = np.mean(self.prediction_errors)
            quality = max(0, 1.0 - avg_error / 50.0)  # 50像素为质量阈值

        return {
            'position': [float(state[0]), float(state[1])],
            'velocity': velocity,
            'acceleration': acceleration,
            'speed': speed,
            'prediction_quality': quality
        }
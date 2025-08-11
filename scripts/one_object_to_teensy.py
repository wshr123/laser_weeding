#!/usr/bin/env python
# -*- coding: utf-8 -*-

import serial
import struct
import time
import numpy as np
import rospy
"""
跟踪单个目标与teensy通信程序
"""

class XY2_100Controller:
    """
    XY2-100振镜控制器 - 仅位置控制版本
    """

    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None

        # 当前位置
        self.current_x = 0
        self.current_y = 0

        # 调试模式
        self.debug_mode = True

        # 初始化串口
        self.connect()

    def connect(self):
        """连接到Teensy"""
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.serial_port.flush()
            rospy.loginfo(f"Successfully connected to {self.port}")

            # 等待Teensy初始化
            time.sleep(2)

            # 初始化振镜
            self.initialize_galvo()

        except serial.SerialException as e:
            rospy.logerr(f"Failed to connect to {self.port}: {e}")
            self.serial_port = None

    def initialize_galvo(self):
        """初始化振镜到中心位置"""
        rospy.loginfo("Initializing galvo...")
        self.move_to_center()
        rospy.loginfo("Galvo initialized")

    def send_command(self, command):
        """发送命令到Teensy"""
        if self.serial_port is None:
            if self.debug_mode:
                rospy.logwarn(f"Serial not connected, command: {command}")
            return False

        try:
            command_bytes = (command + '\n').encode('utf-8')
            self.serial_port.write(command_bytes)
            self.serial_port.flush()

            if self.debug_mode:
                rospy.logdebug(f"Sent: {command}")
            return True

        except Exception as e:
            rospy.logerr(f"Failed to send command '{command}': {e}")
            return False

    def move_to_position(self, x, y):
        """移动振镜到指定位置"""
        # 处理不同的输入格式
        if isinstance(x, float):
            x = int(x)
        if isinstance(y, float):
            y = int(y)

        # 如果输入是0-65535格式，转换为有符号格式
        if x > 32767:
            x = x - 65536
        if y > 32767:
            y = y - 65536

        # 限制范围
        x = max(-30000, min(30000, int(x)))
        y = max(-30000, min(30000, int(y)))

        command = f"XY:{x},{y}"

        if self.send_command(command):
            self.current_x = x
            self.current_y = y
            if self.debug_mode:
                rospy.logdebug(f"Moved to ({x}, {y})")

    def move_to_center(self):
        """移动到中心位置"""
        self.move_to_position(0, 0)

    def pixel_to_galvo(self, pixel_x, pixel_y, image_width=640, image_height=480):
        """将像素坐标转换为振镜坐标"""
        # 归一化到0-1范围
        norm_x = pixel_x / image_width
        norm_y = pixel_y / image_height

        # 转换到有符号振镜坐标系 (-30000 to 30000)
        galvo_x = int((norm_x - 0.5) * 60000)
        galvo_y = int((norm_y - 0.5) * 60000)

        return galvo_x, galvo_y

    def get_current_position(self):
        """获取当前位置"""
        return self.current_x, self.current_y

    def close(self):
        """关闭连接"""
        if self.serial_port:
            self.move_to_center()
            time.sleep(0.5)
            self.serial_port.close()
            rospy.loginfo("Serial connection closed")
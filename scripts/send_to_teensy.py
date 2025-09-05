#!/usr/bin/env python
# -*- coding: utf-8 -*-

import serial
import struct
import time
import numpy as np
import rospy


class XY2_100Controller:
    """
    XY2-100振镜控制器
    通过Teensy 3.2实现XY2-100协议控制振镜
    """

    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        """
        初始化控制器
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None

        # XY2-100协议参数
        self.max_value = 65535
        self.center_value = 32767

        # 振镜工作范围
        self.scan_angle = 30
        # self.field_size = 100

        # 当前位置
        self.current_x = self.center_value
        self.current_y = self.center_value

        # 激光状态
        self.laser_enabled = False

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

            # 发送初始化命令
            self.initialize_galvo()

        except serial.SerialException as e:
            rospy.logerr(f"Failed to connect to {self.port}: {e}")
            self.serial_port = None

    def is_connected(self):
        """检查连接状态"""
        if self.serial_port is None:
            self.connected = False
            return False

        try:
            return self.serial_port.is_open and self.connected
        except:
            self.connected = False
            return False

    def initialize_galvo(self):
        """初始化振镜到中心位置"""
        rospy.loginfo("Initializing galvo...")
        self.move_to_center()
        self.laser_off()
        rospy.loginfo("Galvo initialized")

    def send_command(self, command):
        """发送命令到Teensy"""
        if self.serial_port is None:
            rospy.logdebug(f"Serial port not connected, command: {command}")
            return False

        try:
            command_bytes = (command + '\n').encode('utf-8')
            self.serial_port.write(command_bytes)
            rospy.logdebug(f"Sent command: {command}")
            return True

        except Exception as e:
            rospy.logerr(f"Failed to send command: {e}")
            return False

    def move_to_position(self, x, y):
        """移动振镜到指定位置 (支持0-65535和有符号格式)"""
        # 如果输入是0-65535格式，转换为有符号格式
        # if x > 32767:
        #     x = x - 65536
        # if y > 32767:
        #     y = y - 65536

        # 限制范围到±32767 teensy里规定的y是x，x是y
        x = max(-32767, min(32767, int(x)))
        y = max(-32767, min(32767, int(y)))
        # x = -x
        # print(x,y)
        command = f"XY:{x},{y}"
        # print(command)
        if self.send_command(command):
            self.current_x = y
            self.current_y = x

    def move_to_center(self):
        """移动到中心位置"""
        self.move_to_position(0, 0)

    def laser_on(self):
        """打开激光"""
        command = "LASER:ON"
        if self.send_command(command):
            self.laser_enabled = True

    def laser_off(self):
        """关闭激光"""
        command = "LASER:OFF"
        if self.send_command(command):
            self.laser_enabled = False

    # def pixel_to_galvo(self, pixel_x, pixel_y, image_width=640, image_height=480):
    #     """将像素坐标转换为振镜坐标"""
    #     # 归一化到0-1范围
    #     norm_x = pixel_x / image_width
    #     norm_y = pixel_y / image_height
    #
    #     # 转换到有符号振镜坐标系 (-30000 to 30000)
    #     galvo_x = int((norm_x - 0.5) * 65535)  # -30000 to 30000
    #     galvo_y = int((norm_y - 0.5) * 65535)  # -30000 to 30000

        # return galvo_x, galvo_y

    # def weed_elimination(self, weed_x, weed_y, duration=0.1):
    #     """消除杂草"""
    #     rospy.loginfo(f"Eliminating weed at position ({weed_x}, {weed_y})")
    #
    #     # 移动到杂草位置
    #     self.move_to_position(weed_x, weed_y)
    #
    #     # 等待振镜稳定
    #     time.sleep(0.05)
    #
    #     # 打开激光
    #     self.laser_on()
    #
    #     # 保持激光
    #     time.sleep(duration)
    #
    #     # 关闭激光
    #     self.laser_off()
    #
    #     rospy.loginfo("Weed elimination completed")

    def close(self):
        """关闭连接"""
        if self.serial_port:
            self.laser_off()
            self.move_to_center()
            self.serial_port.close()
            rospy.loginfo("Serial connection closed")
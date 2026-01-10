#!/usr/bin/env python
# -*- coding: utf-8 -*-

import serial
import time
import os
from typing import List, Optional, Tuple

import rospy


class XY2_100Controller:
    """
    XY2-100振镜控制器
    通过Teensy 3.2实现XY2-100协议控制振镜
    """

    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 115200, galvo_count: int = 1):
        """
        初始化控制器
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_port: Optional[serial.Serial] = None

        # XY2-100协议参数
        self.max_value = 65535
        self.center_value = 32767

        # 多振镜支持
        self.galvo_count = max(1, galvo_count)
        self._active_galvo = 0
        self.current_positions = [
            [self.center_value, self.center_value] for _ in range(self.galvo_count)
        ]
        self.galvo_limits: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [
            ((-32767, 32767), (-32767, 32767)) for _ in range(self.galvo_count)
        ]

        # 激光状态
        self.laser_enabled = False

        # 连接状态
        self.connected = False

        # 初始化串口
        self.connect()

    def connect(self):
        """连接到Teensy"""
        # 检查设备是否存在
        if not os.path.exists(self.port):
            rospy.logerr(f"串口设备不存在: {self.port}")
            rospy.logerr(f"请检查: 1) 设备是否连接 2) 设备路径是否正确")
            rospy.logerr(f"可用设备列表: {self._list_available_ports()}")
            self.serial_port = None
            self.connected = False
            return
        
        # 检查权限
        if not os.access(self.port, os.R_OK | os.W_OK):
            rospy.logerr(f"没有访问权限: {self.port}")
            rospy.logerr(f"请运行: sudo usermod -a -G dialout $USER")
            rospy.logerr(f"然后重新登录或运行: newgrp dialout")
            self.serial_port = None
            self.connected = False
            return
        
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            rospy.loginfo(f"Successfully connected to {self.port}")

            self._drain_startup_banner()

            # 等待Teensy初始化
            time.sleep(0.5)

            # 发送初始化命令
            self.initialize_galvo()

            self.connected = True

        except serial.SerialException as e:
            error_msg = str(e)
            if "could not open port" in error_msg.lower() or "permission denied" in error_msg.lower():
                rospy.logerr(f"无法打开串口 {self.port}: {e}")
                rospy.logerr(f"可能原因: 1) 权限不足 (运行: sudo usermod -a -G dialout $USER)")
                rospy.logerr(f"2) 设备被占用 (检查: lsof {self.port})")
            elif "device reports readiness" in error_msg.lower():
                rospy.logwarn(f"设备就绪检查失败，但继续尝试: {e}")
                # 继续尝试，有时这个警告可以忽略
                self.connected = True
            else:
                rospy.logerr(f"Failed to connect to {self.port}: {e}")
            if not self.connected:
                self.serial_port = None
        except Exception as e:
            rospy.logerr(f"Unexpected error connecting to {self.port}: {e}")
            self.serial_port = None
            self.connected = False

    def is_connected(self):
        """检查连接状态"""
        if self.serial_port is None:
            self.connected = False
            return False

        try:
            self.connected = self.serial_port.is_open
        except Exception:
            self.connected = False

        return self.connected

    def initialize_galvo(self):
        """初始化振镜到中心位置"""
        rospy.loginfo("Initializing galvo...")
        for galvo_index in range(self.galvo_count):
            self.move_to_center(galvo_index=galvo_index)
        self.laser_off()
        rospy.loginfo("Galvo initialized")

    def send_command(self, command):
        """发送命令到Teensy"""
        if self.serial_port is None:
            rospy.logdebug(f"Serial port not connected, command: {command}")
            return False

        # 检查串口是否仍然打开
        if not self.serial_port.is_open:
            rospy.logerr(f"Serial port {self.port} is not open. Attempting to reconnect...")
            self.connect()
            if not self.serial_port or not self.serial_port.is_open:
                rospy.logerr(f"Failed to reconnect to {self.port}")
                return False

        try:
            command_bytes = (command + '\n').encode('utf-8')
            self.serial_port.write(command_bytes)
            self.serial_port.flush()
            rospy.logdebug(f"Sent command: {command}")
            return True

        except serial.SerialException as e:
            error_msg = str(e)
            if "Input/output error" in error_msg or "[Errno 5]" in error_msg:
                rospy.logerr(f"串口I/O错误 (可能原因: 设备未连接/断开/权限不足): {self.port}")
                rospy.logerr(f"请检查: 1) 设备是否连接 2) 权限是否正确 (sudo usermod -a -G dialout $USER)")
                rospy.logerr(f"3) 设备是否被其他程序占用 (lsof {self.port})")
            else:
                rospy.logerr(f"串口通信错误: {e}")
            # 标记为未连接，下次尝试重连
            self.connected = False
            return False
        except Exception as e:
            rospy.logerr(f"Failed to send command: {e}")
            return False

    def select_galvo(self, galvo_index: int) -> bool:
        """选择当前活动的振镜。"""
        if galvo_index < 0 or galvo_index >= self.galvo_count:
            rospy.logwarn(f"Invalid galvo index: {galvo_index}")
            return False

        if galvo_index == self._active_galvo:
            return True

        if self.send_command(f"GALVO:{galvo_index + 1}"):
            self._active_galvo = galvo_index
            return True

        return False

    def move_to_position(self, x: int, y: int, galvo_index: Optional[int] = None):
        """移动振镜到指定位置 (支持0-65535和有符号格式)。"""
        # 保存原始 galvo_index（在 sanitize 之前）
        original_index = galvo_index
        galvo_index = self._sanitize_galvo_index(galvo_index)

        limits = self.galvo_limits[galvo_index]
        x_min, x_max = limits[0]
        y_min, y_max = limits[1]

        x = int(max(x_min, min(x_max, int(x))))
        y = int(max(y_min, min(y_max, int(y))))

        # 使用 XY{galvo_index + 1}: 格式直接指定振镜
        # Arduino 代码会从命令中提取数字并解析为 galvo_index
        # galvo_index: 0 -> XY1: -> Arduino解析为 target_galvo=0
        # galvo_index: 1 -> XY2: -> Arduino解析为 target_galvo=1
        # 注意：不要先调用 select_galvo，因为 XY1:/XY2: 格式已经明确指定了振镜
        prefix = f"XY{galvo_index + 1}"
        command = f"{prefix}:{x},{y}"

        if self.send_command(command):
            self.current_positions[galvo_index] = [x, y]
            # 更新 _active_galvo 以保持状态一致
            self._active_galvo = galvo_index
            # 调试级别输出，避免在正常运行时刷屏
            rospy.logdebug(f"[Hardware] Sent command: '{command}', galvo_index={galvo_index}, x={x}, y={y}")

    def move_to_center(self, galvo_index: Optional[int] = None):
        """移动到中心位置。"""
        galvo_index = self._sanitize_galvo_index(galvo_index)
        self.move_to_position(0, 0, galvo_index=galvo_index)

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

    def set_laser_mode(self, mode: str) -> bool:
        """设置激光模式（POINT 或 SPIRAL）。"""
        mode_upper = mode.strip().upper()
        if mode_upper not in {"POINT", "SPIRAL"}:
            rospy.logwarn(f"Unsupported laser mode: {mode}")
            return False
        return self.send_command(f"MODE:{mode_upper}")

    def configure_spiral(self, radius: int, spacing: float, dwell_us: int, angle_step: float = 0.25) -> bool:
        """配置螺旋灼烧参数。"""
        command = f"SPIRAL:CONFIG:{radius},{spacing},{dwell_us},{angle_step}"
        return self.send_command(command)

    def close(self):
        """关闭连接"""
        if self.serial_port:
            self.laser_off()
            for galvo_index in range(self.galvo_count):
                try:
                    self.move_to_center(galvo_index=galvo_index)
                except Exception:
                    pass
            self.serial_port.close()
            rospy.loginfo("Serial connection closed")

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------

    def _drain_startup_banner(self, timeout: float = 2.0) -> None:
        """读取Teensy启动打印，直到收到READY或超时。"""
        if not self.serial_port:
            return

        end_time = time.time() + max(0.0, timeout)
        try:
            old_timeout = self.serial_port.timeout
        except AttributeError:
            old_timeout = None

        try:
            self.serial_port.timeout = 0.1
            while time.time() < end_time:
                try:
                    raw = self.serial_port.readline()
                except Exception:
                    break

                if not raw:
                    continue

                line = raw.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                rospy.loginfo(f"[teensy] {line}")
                if line.upper().startswith("READY"):
                    break
        finally:
            if old_timeout is not None:
                self.serial_port.timeout = old_timeout

    def configure_limits(self, galvo_index: int, x_min: int, x_max: int, y_min: int, y_max: int) -> bool:
        """配置指定振镜的码值上下限。"""
        galvo_index = self._sanitize_galvo_index(galvo_index)

        x_min = int(x_min)
        x_max = int(x_max)
        y_min = int(y_min)
        y_max = int(y_max)

        command = f"LIMITS:{galvo_index + 1}:{x_min},{x_max},{y_min},{y_max}"
        if self.send_command(command):
            self.galvo_limits[galvo_index] = ((x_min, x_max), (y_min, y_max))
            return True

        rospy.logdebug(f"Failed to configure galvo limits for index {galvo_index}")
        return False

    def _sanitize_galvo_index(self, galvo_index: Optional[int]) -> int:
        if galvo_index is None:
            return self._active_galvo

        if galvo_index < 0 or galvo_index >= self.galvo_count:
            rospy.logwarn(f"Galvo index {galvo_index} out of range, using active galvo {self._active_galvo}")
            return self._active_galvo

        return galvo_index
    
    def _list_available_ports(self) -> List[str]:
        """列出可用的串口设备"""
        import serial.tools.list_ports
        try:
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]
        except Exception:
            # 如果无法列出，返回常见的设备路径
            common_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
            return [p for p in common_ports if os.path.exists(p)]

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
import time
import yaml
import os
from std_msgs.msg import String, Int32MultiArray, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import json
import threading
import math
import signal
import sys

# 导入您的坐标变换和振镜控制模块
from coordinate_transform import CameraGalvoTransform
from send_to_teensy import XY2_100Controller
# rostopic pub /manual_calibration_command std_msgs/String "data: 'start'"
"""
方向移动

W：Y+（向上）
S：Y−（向下）
A：X−（向左）
D：X+（向右）

Q：增大当前模式的步长（NORMAL 模式增大 manual_step；FINE 模式增大 fine_step）
E：减小步长
F：切换 FINE（精细）/ NORMAL（普通）模式

L：开关激光
R：记录当前校准点（仅在 MANUAL_AIMING 且有当前目标时有效）
空格：回到“该目标的自动计算位置”（若无当前目标，则回到图像中心对应的振镜位置）
H：回到振镜 HOME (0,0)
I：初始化振镜中心到图像中心（进入 CENTERING，随后回到 IDLE）

N：下一个目标（在 SELECTING 或 MANUAL_AIMING 时有效）
P：上一个目标
T：打印/测试中心点映射，并将振镜移动到映射结果位置
O：打印四角映射计算结果（不移动）

C：保存校准结果到文件（需要至少2个点）
X：重置校准（清空数据，关激光，并回到 CENTERING 再回到 IDLE）
"""

class ManualGalvoCalibrationNode:
    """手动振镜校准节点 - 计算角度偏移bias"""

    def __init__(self):
        rospy.init_node('manual_galvo_calibration_node', anonymous=True)
        rospy.loginfo("Starting Manual Galvo Calibration Node...")

        # 基础组件
        self.bridge = CvBridge()

        # 参数加载
        self.load_parameters()

        # 初始化坐标变换器（使用3D几何）
        try:
            self.coordinate_transform = CameraGalvoTransform(
                config_file=self.transform_config_file,
                use_3d_transform=True
            )
            rospy.loginfo("3D coordinate transformer initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize coordinate transformer: {e}")
            return

        # 初始化振镜控制器
        try:
            self.galvo_controller = XY2_100Controller(
                port=self.serial_port,
                baudrate=self.serial_baudrate
            )
            rospy.loginfo("Galvo controller initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize galvo controller: {e}")
            return

        # 校准状态
        self.calibration_state = "IDLE"  # IDLE, CENTERING, SELECTING, MANUAL_AIMING, RECORDING
        self.calibration_data = []  # 存储校准数据
        self.current_target_index = -1
        self.current_target = None

        # 检测到的目标
        self.detected_circles = []
        self.selected_targets = []  # 选择要校准的目标

        # 图像处理
        self.current_image = None

        # 绿色检测参数
        self.green_lower = np.array([40, 40, 40])  # HSV下限
        self.green_upper = np.array([80, 255, 255])  # HSV上限
        self.min_circle_radius = 15
        self.max_circle_radius = 50

        # 振镜位置控制
        self.galvo_min = -32767
        self.galvo_max = 32767
        self.current_galvo_pos = [0, 0]
        self.target_galvo_pos = [0, 0]
        self.manual_galvo_pos = [0, 0]  # 手动调整的位置
        self.image_center_galvo_pos = [0, 0]  # 图像中心对应的振镜位置（逻辑中心）
        self.laser_on = False

        # 手动控制参数
        self.manual_step = 500  # 每次按键移动的步长（码值）
        self.fine_step = 100  # 精细调整步长
        self.is_fine_mode = False

        # 线程控制
        self.position_lock = threading.Lock()
        self.running = True

        # 终端设置保存
        self.old_terminal_settings = None
        self.terminal_modified = False

        # 键盘输入处理
        self.key_handlers = {
            'w': self.move_up,
            's': self.move_down,
            'a': self.move_left,
            'd': self.move_right,
            'q': self.increase_step,
            'e': self.decrease_step,
            'f': self.toggle_fine_mode,
            'l': self.toggle_laser,
            'r': self.record_calibration_point,
            'n': self.next_target,
            'p': self.previous_target,
            'c': self.save_calibration,
            'x': self.reset_calibration,
            ' ': self.center_to_auto_position,  # 空格键
            'h': self.move_to_home,  # 回到中心位置
            'i': self.init_galvo_center,  # 初始化振镜中心
            't': self.test_coordinate_mapping,  # 测试坐标映射
            'o': self.test_four_corners,  # 测试四角
        }

        # 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # ROS 发布器和订阅器
        self.setup_ros_interface()

        # 启动振镜控制线程
        self.galvo_thread = threading.Thread(target=self.galvo_control_loop)
        self.galvo_thread.daemon = True
        self.galvo_thread.start()

        # 启动键盘监听线程
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        rospy.loginfo("Manual Galvo Calibration Node initialized successfully!")
        # 自动初始化振镜到图像中心
        rospy.Timer(rospy.Duration(2.0), self.auto_init_galvo_center, oneshot=True)

    def signal_handler(self, signum, frame):
        """信号处理器 - 确保正确清理"""
        rospy.loginfo("Received shutdown signal, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """清理资源"""
        self.running = False

        # 关闭激光
        self.set_laser(False)

        # 恢复终端设置
        self.restore_terminal()

        # 关闭振镜控制器
        if hasattr(self, 'galvo_controller') and self.galvo_controller:
            try:
                self.galvo_controller.close()
            except:
                pass

        rospy.loginfo("Cleanup completed")

    def load_parameters(self):
        """加载参数"""
        # 振镜参数
        self.serial_port = rospy.get_param('~serial_port', '/dev/ttyACM0')
        self.serial_baudrate = rospy.get_param('~serial_baudrate', 115200)

        # 坐标变换配置文件
        self.transform_config_file = rospy.get_param('~transform_config_file', 'cam_params.yaml')

        # 图像参数
        self.image_width = rospy.get_param('~image_width', 640)
        self.image_height = rospy.get_param('~image_height', 480)

        # 校准结果保存路径
        self.calibration_result_file = rospy.get_param('~calibration_result_file', 'manual_galvo_calibration.yaml')

    def setup_ros_interface(self):
        """设置ROS接口"""
        image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.image_sub = rospy.Subscriber(
            image_topic, Image, self.image_callback, queue_size=1
        )

        # 命令订阅器
        self.command_sub = rospy.Subscriber(
            '/manual_calibration_command', String, self.command_callback, queue_size=1
        )

        # 发布器
        self.status_pub = rospy.Publisher('/manual_calibration_status', String, queue_size=1)
        self.result_img_pub = rospy.Publisher('/manual_calibration_image', Image, queue_size=1)
        self.galvo_pub = rospy.Publisher('/galvo_xy', Int32MultiArray, queue_size=1)
        self.laser_pub = rospy.Publisher('/laser_control', Bool, queue_size=1)

        # 状态发布定时器
        self.status_timer = rospy.Timer(rospy.Duration(0.5), self.publish_status)

    def auto_init_galvo_center(self, event):
        """自动初始化振镜到图像中心"""
        rospy.loginfo("Auto-initializing galvo to image center...")
        self.init_galvo_center()

    def init_galvo_center(self):
        """初始化振镜中心位置到图像中心"""
        self.calibration_state = "CENTERING"

        # 图像中心现在对应振镜的(0,0)位置（控制零点）
        self.image_center_galvo_pos = [0, 0]

        # 移动振镜到中心位置(0,0)
        with self.position_lock:
            self.target_galvo_pos = [0, 0]
            self.manual_galvo_pos = [0, 0]

        # 等待1秒后切换到空闲状态
        rospy.Timer(rospy.Duration(1.0), lambda e: setattr(self, 'calibration_state', 'IDLE'), oneshot=True)

    def test_coordinate_mapping(self):
        """测试坐标映射（几何模型往返一致性）"""
        rospy.loginfo("Testing coordinate mapping...")

        # 测试图像中心
        u = self.image_width // 2
        v = self.image_height // 2

        code = self.coordinate_transform.pixel_to_galvo_code(u, v, self.image_width, self.image_height)
        rospy.loginfo(f"Image center ({u}, {v}) -> Galvo {code}")

        if code:
            uv2 = self.coordinate_transform.galvo_code_to_pixel(code[0], code[1], self.image_width, self.image_height)
            if uv2:
                du, dv = uv2[0] - u, uv2[1] - v
                rospy.loginfo(f"Roundtrip error: ({du:.2f}, {dv:.2f}) px")

        # 移动到这个位置测试
        if code:
            galvo_pos = self.clamp_galvo_position(code[0], code[1])
            with self.position_lock:
                self.target_galvo_pos = galvo_pos
                self.manual_galvo_pos = galvo_pos

    def test_four_corners(self):
        """测试四个角点（打印几何映射结果）"""
        rospy.loginfo("Testing four corners mapping...")

        corners = [
            (50, 50, "Top-Left"),
            (self.image_width - 50, 50, "Top-Right"),
            (self.image_width - 50, self.image_height - 50, "Bottom-Right"),
            (50, self.image_height - 50, "Bottom-Left"),
        ]

        for px, py, label in corners:
            code = self.coordinate_transform.pixel_to_galvo_code(px, py, self.image_width, self.image_height)
            rospy.loginfo(f"{label}: Pixel({px}, {py}) -> Galvo{code}")

    def clamp_galvo_position(self, x, y):
        """限制振镜位置在有效范围内"""
        x = max(self.galvo_min, min(self.galvo_max, int(x)))
        y = max(self.galvo_min, min(self.galvo_max, int(y)))
        return [x, y]

    def image_callback(self, msg):
        """图像回调"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image

            # 检测绿色圆形
            self.detect_green_circles(cv_image)

            # 绘制结果并发布
            result_image = self.draw_calibration_info(cv_image)

            try:
                result_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                self.result_img_pub.publish(result_msg)
            except CvBridgeError as e:
                rospy.logwarn(f"Failed to publish result image: {e}")

        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")

    def detect_green_circles(self, image):
        """检测绿色圆形"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if self.min_circle_radius <= radius <= self.max_circle_radius:
                circle_area = np.pi * radius * radius
                if area / circle_area > 0.6:
                    circles.append({
                        'center': (int(x), int(y)),
                        'radius': int(radius),
                        'area': area
                    })

        circles.sort(key=lambda x: x['area'], reverse=True)
        self.detected_circles = circles[:10]

    def command_callback(self, msg):
        """命令回调"""
        command = msg.data.strip().lower()
        rospy.loginfo(f"Received command: {command}")

        if command == 'start':
            self.start_calibration()
        elif command == 'center':
            self.init_galvo_center()
        elif command == 'stop':
            self.stop_calibration()
        elif command == 'reset':
            self.reset_calibration()
        else:
            rospy.logwarn(f"Unknown command: {command}")

    def start_calibration(self):
        """开始校准"""
        if len(self.detected_circles) < 2:
            rospy.logwarn("Need at least 2 green circles detected to start calibration")
            return

        self.calibration_state = "SELECTING"
        self.selected_targets = self.detected_circles.copy()
        self.current_target_index = 0
        self.calibration_data = []

        rospy.loginfo(f"Starting manual calibration with {len(self.selected_targets)} targets")
        self.next_target()

    def next_target(self):
        """下一个目标"""
        if self.calibration_state not in ["SELECTING", "MANUAL_AIMING"]:
            rospy.logwarn("Not in calibration mode")
            return

        if self.current_target_index >= len(self.selected_targets):
            rospy.loginfo("All targets completed. You can save calibration results with 'C' key.")
            return

        self.current_target = self.selected_targets[self.current_target_index]
        self.calibration_state = "MANUAL_AIMING"

        # 使用几何模型计算自动位置（像素→码值）
        pixel_x, pixel_y = self.current_target['center']
        code = self.coordinate_transform.pixel_to_galvo_code(
            pixel_x, pixel_y, self.image_width, self.image_height
        )

        # 限制坐标范围并设置为目标位置
        auto_galvo_pos = self.clamp_galvo_position(*(code if code else (0, 0)))
        with self.position_lock:
            self.target_galvo_pos = auto_galvo_pos.copy()
            self.manual_galvo_pos = auto_galvo_pos.copy()

        rospy.loginfo("Use keyboard to manually adjust laser position, then press 'R' to record")

    def previous_target(self):
        """上一个目标"""
        if self.current_target_index > 0:
            self.current_target_index -= 1
            self.next_target()
        else:
            rospy.loginfo("Already at first target")

    def keyboard_listener(self):
        """键盘监听线程 - 修复终端卡死问题"""
        import sys, tty, termios, select

        try:
            # 保存原始终端设置
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
        """恢复终端设置"""
        if self.terminal_modified and self.old_terminal_settings:
            try:
                import sys, termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
                self.terminal_modified = False
                rospy.logdebug("Terminal settings restored")
            except Exception as e:
                rospy.logwarn(f"Failed to restore terminal settings: {e}")

    def handle_keyboard_input(self, key):
        """处理键盘输入"""
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
        """打印帮助信息"""
        rospy.loginfo("=== KEYBOARD HELP ===")
        rospy.loginfo("WASD: Move galvo | L: Laser | R: Record | SPACE: Auto pos")
        rospy.loginfo("QE: Step size | F: Fine mode | NP: Next/Prev target")
        rospy.loginfo("C: Save | X: Reset | H: Home | I: Init center")
        rospy.loginfo("T: Test mapping | O: Corners | Ctrl+C: Exit")

    def move_up(self):
        """向上移动"""
        step = self.fine_step if self.is_fine_mode else self.manual_step
        with self.position_lock:
            new_pos = self.clamp_galvo_position(
                self.manual_galvo_pos[0],
                self.manual_galvo_pos[1] + step
            )
            self.manual_galvo_pos = new_pos
            self.target_galvo_pos = new_pos.copy()

    def move_down(self):
        """向下移动"""
        step = self.fine_step if self.is_fine_mode else self.manual_step
        with self.position_lock:
            new_pos = self.clamp_galvo_position(
                self.manual_galvo_pos[0],
                self.manual_galvo_pos[1] - step
            )
            self.manual_galvo_pos = new_pos
            self.target_galvo_pos = new_pos.copy()

    def move_left(self):
        """向左移动"""
        step = self.fine_step if self.is_fine_mode else self.manual_step
        with self.position_lock:
            new_pos = self.clamp_galvo_position(
                self.manual_galvo_pos[0] - step,
                self.manual_galvo_pos[1]
            )
            self.manual_galvo_pos = new_pos
            self.target_galvo_pos = new_pos.copy()

    def move_right(self):
        """向右移动"""
        step = self.fine_step if self.is_fine_mode else self.manual_step
        with self.position_lock:
            new_pos = self.clamp_galvo_position(
                self.manual_galvo_pos[0] + step,
                self.manual_galvo_pos[1]
            )
            self.manual_galvo_pos = new_pos
            self.target_galvo_pos = new_pos.copy()

    def move_to_home(self):
        """移动到振镜零点位置"""
        with self.position_lock:
            self.manual_galvo_pos = [0, 0]
            self.target_galvo_pos = [0, 0]
        rospy.loginfo("Moved to galvo HOME position (0, 0)")

    def move_to_image_center(self):
        """移动到图像中心对应的振镜位置"""
        with self.position_lock:
            self.manual_galvo_pos = self.image_center_galvo_pos.copy()
            self.target_galvo_pos = self.image_center_galvo_pos.copy()
        rospy.loginfo(
            f"Moved to image center position ({self.image_center_galvo_pos[0]}, {self.image_center_galvo_pos[1]})")

    def increase_step(self):
        """增加步长"""
        if self.is_fine_mode:
            self.fine_step = min(self.fine_step + 50, 1000)
            rospy.loginfo(f"Fine step increased to: {self.fine_step}")
        else:
            self.manual_step = min(self.manual_step + 100, 2000)
            rospy.loginfo(f"Manual step increased to: {self.manual_step}")

    def decrease_step(self):
        """减少步长"""
        if self.is_fine_mode:
            self.fine_step = max(self.fine_step - 50, 10)
            rospy.loginfo(f"Fine step decreased to: {self.fine_step}")
        else:
            self.manual_step = max(self.manual_step - 100, 50)
            rospy.loginfo(f"Manual step decreased to: {self.manual_step}")

    def toggle_fine_mode(self):
        """切换精细模式"""
        self.is_fine_mode = not self.is_fine_mode
        mode = "FINE" if self.is_fine_mode else "NORMAL"
        step = self.fine_step if self.is_fine_mode else self.manual_step
        rospy.loginfo(f"Mode: {mode}, Step: {step}")

    def center_to_auto_position(self):
        """回到自动计算位置"""
        if self.current_target:
            pixel_x, pixel_y = self.current_target['center']
            code = self.coordinate_transform.pixel_to_galvo_code(
                pixel_x, pixel_y, self.image_width, self.image_height
            )
            auto_galvo_pos = self.clamp_galvo_position(*(code if code else (0, 0)))
            with self.position_lock:
                self.manual_galvo_pos = auto_galvo_pos.copy()
                self.target_galvo_pos = auto_galvo_pos.copy()
            rospy.loginfo(f"Centered to auto position: ({auto_galvo_pos[0]}, {auto_galvo_pos[1]})")
        else:
            # 如果没有当前目标，移动到图像中心零点
            self.move_to_image_center()

    def toggle_laser(self):
        """切换激光状态"""
        self.set_laser(not self.laser_on)

    def record_calibration_point(self):
        """记录校准点"""
        if self.calibration_state != "MANUAL_AIMING" or not self.current_target:
            rospy.logwarn("No current target to record")
            return

        # 获取当前数据
        pixel_x, pixel_y = self.current_target['center']
        manual_galvo_x, manual_galvo_y = self.manual_galvo_pos

        # 计算图像中心偏移
        center_x = self.image_width / 2.0
        center_y = self.image_height / 2.0
        pixel_offset_x = pixel_x - center_x
        pixel_offset_y = center_y - pixel_y  # 图像Y向下，几何Y向上

        # 使用几何映射的理论码值
        theoretical_code = self.coordinate_transform.pixel_to_galvo_code(
            pixel_x, pixel_y, self.image_width, self.image_height
        )
        theoretical_galvo_x, theoretical_galvo_y = (theoretical_code if theoretical_code else (0, 0))

        # 码值偏移
        galvo_offset_x = manual_galvo_x - theoretical_galvo_x
        galvo_offset_y = manual_galvo_y - theoretical_galvo_y

        # 转换为角度偏移（度）
        angle_offset_x, angle_offset_y = self.galvo_offset_to_angle_offset(
            galvo_offset_x, galvo_offset_y
        )

        # 记录
        calibration_point = {
            'target_index': self.current_target_index,
            'pixel_position': [pixel_x, pixel_y],
            'pixel_offset_from_center': [pixel_offset_x, pixel_offset_y],
            'theoretical_galvo_position': [theoretical_galvo_x, theoretical_galvo_y],
            'manual_galvo_position': [manual_galvo_x, manual_galvo_y],
            'galvo_offset': [galvo_offset_x, galvo_offset_y],
            'angle_offset_deg': [angle_offset_x, angle_offset_y],
            'timestamp': time.time()
        }

        self.calibration_data.append(calibration_point)

        # 自动移到下一个目标
        self.current_target_index += 1
        rospy.Timer(rospy.Duration(1.0), lambda e: self.next_target(), oneshot=True)

    def galvo_offset_to_angle_offset(self, galvo_offset_x, galvo_offset_y):
        """将振镜码值偏移转换为角度偏移（度）"""
        try:
            galvo_params = self.coordinate_transform.params['galvo_params']
            scan_angle = galvo_params['scan_angle']  # 总扫描角度（度）
            max_code = galvo_params['max_code']      # 最大码值
            half_scan_angle = scan_angle / 2.0
            # 码值[-max_code, max_code] 对应角度[-half_scan_angle, half_scan_angle]
            angle_offset_x = (galvo_offset_x / max_code) * half_scan_angle
            angle_offset_y = (galvo_offset_y / max_code) * half_scan_angle
            return angle_offset_x, angle_offset_y
        except Exception as e:
            rospy.logerr(f"Failed to convert galvo offset to angle offset: {e}")
            return 0.0, 0.0

    def save_calibration(self):
        """保存校准结果"""
        if len(self.calibration_data) < 2:
            rospy.logwarn("Need at least 2 calibration points to save")
            return

        # 计算角度偏移统计
        angle_offsets = np.array([cp['angle_offset_deg'] for cp in self.calibration_data])

        bias_x = float(np.mean(angle_offsets[:, 0]))
        bias_y = float(np.mean(angle_offsets[:, 1]))
        std_x = float(np.std(angle_offsets[:, 0]))
        std_y = float(np.std(angle_offsets[:, 1]))
        max_error_x = float(np.max(np.abs(angle_offsets[:, 0])))
        max_error_y = float(np.max(np.abs(angle_offsets[:, 1])))

        calibration_result = {
            'calibration_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_points': len(self.calibration_data),
                'image_size': [self.image_width, self.image_height],
                'method': 'manual_adjustment_with_geometric_mapping',
                'image_center_galvo_pos': self.image_center_galvo_pos,
            },

            'calibration_points': self.calibration_data,

            'angle_bias': {
                'bias_x': bias_x,  # X轴零点偏移 (度)
                'bias_y': bias_y,  # Y轴零点偏移 (度)
                'std_x': std_x,
                'std_y': std_y,
                'max_error_x': max_error_x,
                'max_error_y': max_error_y,
                'rms_error': float(np.sqrt(np.mean(np.sum(angle_offsets ** 2, axis=1))))
            },

            'updated_galvo_params': {
                'scan_angle': self.coordinate_transform.params['galvo_params']['scan_angle'],
                'scale_x': self.coordinate_transform.params['galvo_params']['scale_x'],
                'scale_y': self.coordinate_transform.params['galvo_params']['scale_y'],
                'bias_x': bias_x,  # 更新的偏移值
                'bias_y': bias_y,  # 更新的偏移值
                'max_code': self.coordinate_transform.params['galvo_params']['max_code']
            }
        }

        # 保存到文件
        try:
            with open(self.calibration_result_file, 'w') as f:
                yaml.dump(calibration_result, f, default_flow_style=False)

            rospy.loginfo(f"Manual calibration saved to: {self.calibration_result_file}")
            self.generate_updated_config(calibration_result)

        except Exception as e:
            rospy.logerr(f"Failed to save calibration: {e}")

    def generate_updated_config(self, calibration_result):
        """生成更新的配置文件"""
        try:
            original_config = self.coordinate_transform.params.copy()
            # 更新galvo_params中的bias值
            original_config['galvo_params']['bias_x'] = calibration_result['angle_bias']['bias_x']
            original_config['galvo_params']['bias_y'] = calibration_result['angle_bias']['bias_y']

            # 生成更新的配置文件名
            base_name = os.path.splitext(self.calibration_result_file)[0]
            updated_config_file = f"{base_name}_updated_config.yaml"

            with open(updated_config_file, 'w') as f:
                yaml.dump(original_config, f, default_flow_style=False)

            rospy.loginfo(f"Updated configuration saved to: {updated_config_file}")
            rospy.loginfo("You can use this updated config file in your main program")

        except Exception as e:
            rospy.logerr(f"Failed to generate updated config: {e}")

    def reset_calibration(self):
        """重置校准"""
        self.calibration_state = "IDLE"
        self.calibration_data = []
        self.current_target_index = -1
        self.current_target = None
        self.set_laser(False)
        # 重新初始化到图像中心
        self.init_galvo_center()
        rospy.loginfo("Manual calibration reset")

    def stop_calibration(self):
        """停止校准"""
        self.calibration_state = "IDLE"
        self.set_laser(False)
        rospy.loginfo("Manual calibration stopped")

    def galvo_control_loop(self):
        """振镜控制线程"""
        rate = rospy.Rate(200)  # 200Hz

        while self.running and not rospy.is_shutdown():
            try:
                with self.position_lock:
                    target_pos = self.target_galvo_pos.copy()

                # 确保位置在有效范围内
                target_pos = self.clamp_galvo_position(target_pos[0], target_pos[1])

                # 移动振镜
                if self.galvo_controller:
                    self.galvo_controller.move_to_position(
                        target_pos[0], target_pos[1]
                    )

                self.current_galvo_pos = target_pos

                # 发布振镜位置
                galvo_msg = Int32MultiArray()
                galvo_msg.data = [
                    target_pos[0],
                    target_pos[1],
                    1 if self.laser_on else 0
                ]
                self.galvo_pub.publish(galvo_msg)

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"Galvo control error: {e}")
                time.sleep(0.01)

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

            status = "ON" if enable else "OFF"
            rospy.loginfo(f"Laser: {status}")

    def draw_calibration_info(self, image):
        """绘制校准信息"""
        result = image.copy()

        # 绘制图像中心十字线
        center_x = int(self.image_width / 2)
        center_y = int(self.image_height / 2)
        cv2.line(result, (center_x - 20, center_y), (center_x + 20, center_y), (128, 128, 128), 1)
        cv2.line(result, (center_x, center_y - 20), (center_x, center_y + 20), (128, 128, 128), 1)
        cv2.putText(result, "IMG CENTER", (center_x + 25, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # 绘制检测到的绿色圆形
        for i, circle in enumerate(self.detected_circles):
            center = circle['center']
            radius = circle['radius']

            is_current = (self.current_target and
                          abs(center[0] - self.current_target['center'][0]) < 5 and
                          abs(center[1] - self.current_target['center'][1]) < 5)

            is_calibrated = any(
                abs(center[0] - cp['pixel_position'][0]) < 10 and
                abs(center[1] - cp['pixel_position'][1]) < 10
                for cp in self.calibration_data
            )

            if is_current:
                color = (0, 0, 255)
                thickness = 3
            elif is_calibrated:
                color = (128, 128, 128)
                thickness = 2
            else:
                color = (0, 255, 0)
                thickness = 2

            cv2.circle(result, (int(center[0]), int(center[1])), int(radius), color, thickness)
            cv2.circle(result, (int(center[0]), int(center[1])), 3, color, -1)

            label = f"T{i + 1}"
            if is_current:
                label += " [CURRENT]"
            elif is_calibrated:
                label += " [DONE]"

            cv2.putText(result, label, (int(center[0] + 20), int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制振镜瞄准位置（用几何反投：码值→像素）
        try:
            galvo_pixel = self.coordinate_transform.galvo_code_to_pixel(
                int(self.current_galvo_pos[0]), int(self.current_galvo_pos[1]),
                self.image_width, self.image_height
            )
            if galvo_pixel is not None:
                galvo_pixel_x, galvo_pixel_y = galvo_pixel
                if (0 <= galvo_pixel_x < self.image_width and
                        0 <= galvo_pixel_y < self.image_height):
                    x = int(round(galvo_pixel_x))
                    y = int(round(galvo_pixel_y))
                    color = (0, 0, 255) if self.laser_on else (255, 255, 0)
                    cv2.line(result, (x - 20, y), (x + 20, y), color, 3)
                    cv2.line(result, (x, y - 20), (x, y + 20), color, 3)
                    cv2.circle(result, (x, y), 10, color, 3)
                    laser_text = "LASER ON" if self.laser_on else "GALVO"
                    cv2.putText(result, laser_text, (x + 25, y - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    coord_text = f"({self.current_galvo_pos[0]}, {self.current_galvo_pos[1]})"
                    cv2.putText(result, coord_text, (x + 25, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception:
            pass

        # 绘制自动计算位置（如果有当前目标）
        if self.current_target:
            try:
                pixel_x, pixel_y = self.current_target['center']
                code = self.coordinate_transform.pixel_to_galvo_code(
                    pixel_x, pixel_y, self.image_width, self.image_height
                )
                auto_clamped = self.clamp_galvo_position(*(code if code else (0, 0)))
                auto_pixel = self.coordinate_transform.galvo_code_to_pixel(
                    int(auto_clamped[0]), int(auto_clamped[1]),
                    self.image_width, self.image_height
                )
                if auto_pixel is not None:
                    ax, ay = int(round(auto_pixel[0])), int(round(auto_pixel[1]))
                    if (0 <= ax < self.image_width and 0 <= ay < self.image_height):
                        cv2.circle(result, (ax, ay), 8, (255, 0, 255), 2)
                        cv2.putText(result, "AUTO", (ax + 15, ay + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            except Exception:
                pass

        return result

    def publish_status(self, event):
        """发布状态信息"""
        try:
            status_info = {
                'state': self.calibration_state,
                'detected_circles': len(self.detected_circles),
                'calibration_points': len(self.calibration_data),
                'current_target_index': self.current_target_index,
                'current_target': self.current_target['center'] if self.current_target else None,
                'galvo_position': self.current_galvo_pos,
                'manual_galvo_position': self.manual_galvo_pos,
                'image_center_galvo_pos': self.image_center_galvo_pos,
                'galvo_range': [self.galvo_min, self.galvo_max],
                'laser_on': self.laser_on,
                'step_size': self.fine_step if self.is_fine_mode else self.manual_step,
                'fine_mode': self.is_fine_mode
            }
            status_msg = String()
            status_msg.data = json.dumps(status_info)
            self.status_pub.publish(status_msg)
        except Exception as e:
            rospy.logdebug(f"Failed to publish status: {e}")

    def __del__(self):
        """析构函数"""
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
        # 确保清理资源
        if 'node' in locals():
            node.cleanup()


if __name__ == '__main__':
    main()
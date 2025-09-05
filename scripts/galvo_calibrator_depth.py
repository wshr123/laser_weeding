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
import signal
import sys

# 自有模块
from coordinate_transform import CameraGalvoTransform
from send_to_teensy import XY2_100Controller


class ManualGalvoCalibrationNode:
    """手动振镜校准节点 - 计算角度偏移 bias，支持深度"""

    def __init__(self):
        rospy.init_node('manual_galvo_calibration_node', anonymous=True)
        rospy.loginfo("Starting Manual Galvo Calibration Node...")

        self.bridge = CvBridge()

        # ========== 参数 ==========
        self.load_parameters()

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

        # ========== 坐标变换器 ==========
        try:
            self.coordinate_transform = CameraGalvoTransform(
                config_file=self.transform_config_file,
                use_3d_transform=True
            )
            rospy.loginfo("3D coordinate transformer initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize coordinate transformer: {e}")
            raise

        # ========== 振镜控制器 ==========
        try:
            self.galvo_controller = XY2_100Controller(
                port=self.serial_port,
                baudrate=self.serial_baudrate
            )
            rospy.loginfo("Galvo controller initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize galvo controller: {e}")
            raise

        # ========== 状态 ==========
        self.calibration_state = "IDLE"  # IDLE, CENTERING, SELECTING, MANUAL_AIMING
        self.calibration_data = []
        self.current_target_index = -1
        self.current_target = None

        self.detected_circles = []
        self.selected_targets = []

        self.current_image = None
        self.first_frame_synced = False  # 首帧同步宽高

        # 绿色检测参数（Realsense 一般偏黄/绿，这个范围更宽）
        self.green_lower = np.array([30, 30, 30])
        self.green_upper = np.array([85, 255, 255])
        self.min_circle_radius = 10
        self.max_circle_radius = 100
        self._stable_pool = {}  # id -> tracked circle with EMA & hits
        self._next_circle_id = 0
        self._frame_index = 0

        # 振镜位置
        self.galvo_min = -32767
        self.galvo_max =  32767
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
            ' ': self.center_to_auto_position,
            'h': self.move_to_home,
            # 'i': self.init_galvo_center,
            # 't': self.test_coordinate_mapping,
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

    # ===================== 轴适配：逻辑 <-> 硬件 =====================
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

    # ===================== ROS 基础 =====================
    def signal_handler(self, signum, frame):
        rospy.loginfo("Received shutdown signal, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        self.running = False
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

        self.transform_config_file = rospy.get_param('~transform_config_file', 'cam_params.yaml')

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

    # def test_coordinate_mapping(self):
    #     rospy.loginfo("Testing coordinate mapping...")
    #     u = int(self.image_width // 2)
    #     v = int(self.image_height // 2)
    #
    #     code_hw = self.coordinate_transform.pixel_to_galvo_code(u, v, self.image_width, self.image_height)
    #     rospy.loginfo(f"Image center ({u}, {v}) -> Galvo(HW) {code_hw}")
    #
    #     if code_hw:
    #         uv2 = self.coordinate_transform.galvo_code_to_pixel_3d(int(code_hw[0]), int(code_hw[1]),
    #                                                             self.image_width, self.image_height)
    #         if uv2:
    #             du, dv = uv2[0] - u, uv2[1] - v
    #             rospy.loginfo(f"Roundtrip error: ({du:.2f}, {dv:.2f}) px")
    #
    #         # 把硬件码值转回逻辑坐标存入目标位置
    #         lx, ly = self._from_hw_axes(int(code_hw[0]), int(code_hw[1]))
    #         galvo_pos_logical = self.clamp_galvo_position(lx, ly)
    #         with self.position_lock:
    #             self.target_galvo_pos = galvo_pos_logical
    #             self.manual_galvo_pos = galvo_pos_logical

    def test_four_corners(self):
        rospy.loginfo("Testing four corners mapping...")
        corners = [
            (50, 50, "Top-Left"),
            (self.image_width - 50, 50, "Top-Right"),
            (self.image_width - 50, self.image_height - 50, "Bottom-Right"),
            (50, self.image_height - 50, "Bottom-Left"),
        ]
        for px, py, label in corners:
            code_hw = self.coordinate_transform.pixel_to_galvo_code(px, py, self.image_width, self.image_height)
            rospy.loginfo(f"{label}: Pixel({px}, {py}) -> Galvo(HW){code_hw}")

    # ===================== 基础工具 =====================
    def clamp_galvo_position(self, x, y):
        x = max(self.galvo_min, min(self.galvo_max, int(x)))
        y = max(self.galvo_min, min(self.galvo_max, int(y)))
        return [x, y]

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
            self.detect_green_circles(cv_image)
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

    def depth_query_func(self, u, v):
        """查询像素(u,v)深度（米）"""
        with self.depth_image_lock:
            if self.depth_image is None:
                return None
            H, W = self.depth_image.shape[:2]
            if u < 0 or v < 0 or u >= W or v >= H:
                return None

            # 直接采样（如需更稳，可改为邻域中值）
            z = float(self.depth_image[int(round(v)), int(round(u))])

            if not np.isfinite(z) or z <= 0:
                return None
            return z

    # ===================== 目标检测 =====================
    def detect_green_circles(self, image):
        H, W = image.shape[:2]

        # ---------- 1) 颜色预处理：HSV 双区间 + S/V 下限 ----------
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 绿色常见两段：低H段（~35-85）；不同相机可能偏移，留冗余
        lower1 = np.array([35, 60, 60], dtype=np.uint8)  # H,S,V 最低值提高，避免灰/暗
        upper1 = np.array([85, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower1, upper1)

        # ---------- 2) 形态学抑噪：先开再闭，核稍大 ----------
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # ---------- 3) 连通域 + 形状过滤 ----------
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 根据画面尺度自适应的面积极小值（防止远处微点）
        min_area = max(200, int(0.00015 * W * H))  # 可调：0.0001~0.0003
        # 半径门限（像素）
        min_r = max(self.min_circle_radius, 12)  # 提高下限
        max_r = self.max_circle_radius

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # 周长与圆度：圆度 = 4πA / P^2，越接近1越圆
            perim = cv2.arcLength(cnt, True)
            if perim <= 1e-3:
                continue
            circularity = 4.0 * np.pi * area / (perim * perim)
            if circularity < 0.7:  # 0.7~0.85 之间按实际调整；越高越严格
                continue

            # 最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius < min_r or radius > max_r:
                continue

            # 圆填充度（等价你原先的 area / (πr²)）
            circle_area = np.pi * radius * radius
            fill_ratio = float(area) / float(circle_area + 1e-6)
            if fill_ratio < 0.75:  # 0.6->0.75，更严格，过滤半圆/镂空边缘
                continue

            # 椭圆拟合离心率（对近似圆，轴长相近）
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (ex, ey), (MA, ma), angle = ellipse  # MA>=ma（OpenCV可能返回相反顺序，统一排序）
                MA, ma = max(MA, ma), min(MA, ma)
                ratio = ma / (MA + 1e-6)  # 越接近1越圆
                if ratio < 0.75:  # 过滤细长形
                    continue

            candidates.append({
                'center': (int(round(x)), int(round(y))),
                'radius': int(round(radius)),
                'area': float(area),
                'circularity': float(circularity),
                'fill': float(fill_ratio)
            })

        # ---------- 4) NMS：最小圆间距，抑制紧邻小圈 ----------
        # 以“综合得分”排序：更大面积、更圆、填充度高优先
        def score(c):
            return 0.6 * c['area'] + 0.3 * c['circularity'] * 10000 + 0.1 * c['fill'] * 10000

        candidates.sort(key=score, reverse=True)

        picked = []
        min_sep = 20  # 最小圆心距离（像素），可调大一点：25~35
        for c in candidates:
            ok = True
            for p in picked:
                if (c['center'][0] - p['center'][0]) ** 2 + (c['center'][1] - p['center'][1]) ** 2 < (min_sep ** 2):
                    ok = False
                    break
            if ok:
                picked.append(c)
            if len(picked) >= 32:
                break

        # ---------- 5) （可选）时间一致性：连续命中才输出 ----------
        # 需要在 __init__ 里：self._stable_pool = {}, self._frame_index = 0
        self._frame_index += 1
        new_pool = {}
        max_dist_track = 25  # 帧间匹配距离像素
        min_hits_to_show = 2  # 连续命中次数阈值（2~3更稳）
        # 先把上一帧的目标拿出来匹配
        for c in picked:
            best_id, best_d = None, max_dist_track
            for tid, t in self._stable_pool.items():
                d = np.hypot(c['center'][0] - t['center'][0], c['center'][1] - t['center'][1])
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                # 新建 ID
                self._next_circle_id += 1
                best_id = self._next_circle_id
                hits = 1
            else:
                hits = self._stable_pool[best_id]['hits'] + 1

            # 指数平滑位置（减少抖动）
            if best_id in self._stable_pool:
                px, py = self._stable_pool[best_id]['center']
                nx = int(round(0.6 * px + 0.4 * c['center'][0]))
                ny = int(round(0.6 * py + 0.4 * c['center'][1]))
            else:
                nx, ny = c['center']

            new_pool[best_id] = {
                'center': (nx, ny),
                'radius': int(
                    round(0.5 * self._stable_pool.get(best_id, {'radius': c['radius']})['radius'] + 0.5 * c['radius'])),
                'area': c['area'],
                'circularity': c['circularity'],
                'fill': c['fill'],
                'hits': hits,
                'last_seen': self._frame_index
            }

        # 清理长时间看不见的
        for tid, t in list(self._stable_pool.items()):
            if tid not in new_pool and (self._frame_index - t['last_seen']) <= 2:
                # 给 2 帧缓冲（短时丢失不立即删除）
                new_pool[tid] = t

        self._stable_pool = new_pool

        # 只有 hits 达标的才输出为 detected_circles
        stable = []
        for tid, t in self._stable_pool.items():
            if t['hits'] >= min_hits_to_show:
                stable.append({'id': tid, 'center': t['center'], 'radius': t['radius'], 'area': t['area']})

        # 排序并截断
        stable.sort(key=lambda x: x['area'], reverse=True)
        self.detected_circles = stable[:10]

        # （可选）深度一致性：若开启 use_depth，可在此对 stable 逐个圆做邻域深度方差过滤
        # 例：方差 > 0.05m^2 或 无效深度比例过高则剔除

    # ===================== 命令流 =====================
    def command_callback(self, msg):
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
            pixel_x, pixel_y, self.image_width, self.image_height
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
        rospy.loginfo("=== KEYBOARD HELP ===")
        rospy.loginfo("WASD: Move galvo | L: Laser | R: Record | SPACE: Auto pos")
        rospy.loginfo("QE: Step size | F: Fine mode | NP: Next/Prev target")
        rospy.loginfo("C: Save | X: Reset | H: Home | I: Init center")
        rospy.loginfo("T: Test mapping | O: Corners | Ctrl+C: Exit")

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
                pixel_x, pixel_y, self.image_width, self.image_height
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
        if self.calibration_state != "MANUAL_AIMING" or not self.current_target:
            rospy.logwarn("No current target to record")
            return

        pixel_x, pixel_y = self.current_target['center']
        manual_galvo_x, manual_galvo_y = self.manual_galvo_pos  # 逻辑

        center_x = self.image_width / 2.0
        center_y = self.image_height / 2.0
        pixel_offset_x = pixel_x - center_x
        pixel_offset_y = center_y - pixel_y

        # 几何映射得到code
        code_hw = self.coordinate_transform.pixel_to_galvo_code(
            pixel_x, pixel_y, self.image_width, self.image_height
        )
        if code_hw:
            th_lx, th_ly = self._from_hw_axes(int(code_hw[0]), int(code_hw[1]))  # 理论“逻辑”码值
        else:
            th_lx, th_ly = 0, 0

        # 码值偏移
        galvo_offset_x = manual_galvo_x - th_lx
        galvo_offset_y = manual_galvo_y - th_ly

        # 偏移角度
        angle_offset_x, angle_offset_y = self.galvo_offset_to_angle_offset(galvo_offset_x, galvo_offset_y)

        calibration_point = {
            'target_index': self.current_target_index,
            'pixel_position': [int(pixel_x), int(pixel_y)],
            'pixel_offset_from_center': [float(pixel_offset_x), float(pixel_offset_y)],
            'theoretical_galvo_position_logical': [int(th_lx), int(th_ly)],
            'manual_galvo_position_logical': [int(manual_galvo_x), int(manual_galvo_y)],
            'galvo_offset_logical': [int(galvo_offset_x), int(galvo_offset_y)],
            'angle_offset_deg': [float(angle_offset_x), float(angle_offset_y)],
            'timestamp': time.time()
        }

        self.calibration_data.append(calibration_point)

        self.current_target_index += 1
        rospy.Timer(rospy.Duration(1.0), lambda e: self.next_target(), oneshot=True)

    def galvo_offset_to_angle_offset(self, galvo_offset_x, galvo_offset_y):
        try:
            galvo_params = self.coordinate_transform.params['galvo_params']
            scan_angle = galvo_params['scan_angle']
            max_code = galvo_params['max_code']
            half_scan_angle = scan_angle / 2.0
            angle_offset_x = (float(galvo_offset_x) / float(max_code)) * half_scan_angle
            angle_offset_y = (float(galvo_offset_y) / float(max_code)) * half_scan_angle
            return angle_offset_x, angle_offset_y
        except Exception as e:
            rospy.logerr(f"Failed to convert galvo offset to angle offset: {e}")
            return 0.0, 0.0

    def save_calibration(self):
        if len(self.calibration_data) < 2:
            rospy.logwarn("Need at least 2 calibration points to save")
            return

        angle_offsets = np.array([cp['angle_offset_deg'] for cp in self.calibration_data], dtype=np.float64)
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
                'image_size': [int(self.image_width), int(self.image_height)],
                'method': 'manual_adjustment_with_geometric_mapping',
                'image_center_galvo_pos_logical': [int(self.image_center_galvo_pos[0]), int(self.image_center_galvo_pos[1])],
                'axis_adaptation': {
                    # 'swap_axes': bool(self.swap_axes),
                    # 'invert_x': bool(self.invert_x),
                    # 'invert_y': bool(self.invert_y),
                }
            },
            'calibration_points': self.calibration_data,
            'angle_bias': {
                'bias_x': bias_x,
                'bias_y': bias_y,
                'std_x': std_x,
                'std_y': std_y,
                'max_error_x': max_error_x,
                'max_error_y': max_error_y,
                'rms_error': float(np.sqrt(np.mean(np.sum(angle_offsets ** 2, axis=1))))
            },
            'updated_galvo_params': {
                'scan_angle': float(self.coordinate_transform.params['galvo_params']['scan_angle']),
                'scale_x': float(self.coordinate_transform.params['galvo_params']['scale_x']),
                'scale_y': float(self.coordinate_transform.params['galvo_params']['scale_y']),
                'bias_x': bias_x,
                'bias_y': bias_y,
                'max_code': int(self.coordinate_transform.params['galvo_params']['max_code'])
            }
        }

        try:
            with open(self.calibration_result_file, 'w') as f:
                yaml.dump(calibration_result, f, default_flow_style=False)
            rospy.loginfo(f"Manual calibration saved to: {self.calibration_result_file}")
            self.generate_updated_config(calibration_result)
        except Exception as e:
            rospy.logerr(f"Failed to save calibration: {e}")

    def generate_updated_config(self, calibration_result):
        try:
            original_config = self.coordinate_transform.params.copy()
            original_config['galvo_params']['bias_x'] = calibration_result['angle_bias']['bias_x']
            original_config['galvo_params']['bias_y'] = calibration_result['angle_bias']['bias_y']
            base_name = os.path.splitext(self.calibration_result_file)[0]
            updated_config_file = f"{base_name}_updated_config.yaml"
            with open(updated_config_file, 'w') as f:
                yaml.dump(original_config, f, default_flow_style=False)
            rospy.loginfo(f"Updated configuration saved to: {updated_config_file}")
        except Exception as e:
            rospy.logerr(f"Failed to generate updated config: {e}")

    def reset_calibration(self):
        self.calibration_state = "IDLE"
        self.calibration_data = []
        self.current_target_index = -1
        self.current_target = None
        self.set_laser(False)
        self.init_galvo_center()
        rospy.loginfo("Manual calibration reset")

    def stop_calibration(self):
        self.calibration_state = "IDLE"
        self.set_laser(False)
        rospy.loginfo("Manual calibration stopped")

    # ===================== 控制环（逻辑 -> 硬件 -> Teensy） =====================
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
                    self.galvo_controller.move_to_position(hx, hy)

                self.current_galvo_pos = target_pos_logical  # 记录当前“逻辑”位置

                galvo_msg = Int32MultiArray()
                galvo_msg.data = [int(target_pos_logical[0]), int(target_pos_logical[1]), 1 if self.laser_on else 0]
                self.galvo_pub.publish(galvo_msg)

                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Galvo control error: {e}")
                time.sleep(0.01)

    def set_laser(self, enable):
        if self.laser_on != enable:
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

    # ===================== 叠加显示（几何一律用“硬件”码值） =====================
    def draw_calibration_info(self, image):
        result = image.copy()

        # 图像中心十字线
        cx = int(self.image_width / 2)
        cy = int(self.image_height / 2)
        cv2.line(result, (cx - 20, cy), (cx + 20, cy), (128, 128, 128), 1)
        cv2.line(result, (cx, cy - 20), (cx, cy + 20), (128, 128, 128), 1)
        cv2.putText(result, "IMG CENTER", (cx + 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # 绿色目标
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
                color = (0, 0, 255); thickness = 3
            elif is_calibrated:
                color = (128, 128, 128); thickness = 2
            else:
                color = (0, 255, 0); thickness = 2

            cv2.circle(result, (int(center[0]), int(center[1])), int(radius), color, thickness)
            cv2.circle(result, (int(center[0]), int(center[1])), 3, color, -1)
            label = f"T{i+1}"
            if is_current: label += " [CURRENT]"
            elif is_calibrated: label += " [DONE]"
            cv2.putText(result, label, (int(center[0] + 20), int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 当前振镜位置
        try:
            hx, hy = self._to_hw_axes(self.current_galvo_pos[0], self.current_galvo_pos[1])
            galvo_pixel = self.coordinate_transform.galvo_code_to_pixel(int(hx), int(hy),
                                                                        self.image_width, self.image_height)
            # print("galvo_pixel",galvo_pixel)
            if galvo_pixel is not None:
                x, y = int(round(galvo_pixel[0])), int(round(galvo_pixel[1]))
                if 0 <= x < self.image_width and 0 <= y < self.image_height:
                    color = (0, 0, 255) if self.laser_on else (255, 255, 0)
                    cv2.line(result, (x - 20, y), (x + 20, y), color, 3)
                    cv2.line(result, (x, y - 20), (x, y + 20), color, 3)
                    cv2.circle(result, (x, y), 10, color, 3)
                    cv2.putText(result, "LASER ON" if self.laser_on else "GALVO",
                                (x + 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    coord_text = f"LOGIC({self.current_galvo_pos[0]},{self.current_galvo_pos[1]})"
                    cv2.putText(result, coord_text, (x + 25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception:
            pass

        # AUTO 位置
        if self.current_target:
            try:
                px, py = self.current_target['center']
                code_hw = self.coordinate_transform.pixel_to_galvo_code(px, py, self.image_width, self.image_height)
                if code_hw:
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

    # ===================== 状态发布 =====================
    def publish_status(self, event):
        try:
            status_info = {
                'state': self.calibration_state,
                'detected_circles': len(self.detected_circles),
                'calibration_points': len(self.calibration_data),
                'current_target_index': self.current_target_index,
                'current_target': self.current_target['center'] if self.current_target else None,
                'galvo_position_logical': self.current_galvo_pos,
                'manual_galvo_position_logical': self.manual_galvo_pos,
                'image_center_galvo_pos_logical': self.image_center_galvo_pos,
                'galvo_range': [self.galvo_min, self.galvo_max],
                'laser_on': self.laser_on,
                'step_size': self.fine_step if self.is_fine_mode else self.manual_step,
                'fine_mode': self.is_fine_mode,
                # 'axis_adaptation': {
                #     'swap_axes': self.swap_axes, 'invert_x': self.invert_x, 'invert_y': self.invert_y
                # },
                # 'visual_axis_adaptation': {
                #     'swap_axes': self.visual_swap_axes, 'invert_x': self.visual_invert_x, 'invert_y': self.visual_invert_y
                # }
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

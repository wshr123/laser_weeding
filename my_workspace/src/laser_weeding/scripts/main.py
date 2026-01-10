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
            self.use_reverse_projection = rospy.get_param('~use_reverse_projection', True)
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
            # 检测模式参数
            self.detection_mode = rospy.get_param('~detection_mode', 'bbox')
            # 跟踪器类型参数
            self.tracker_type = rospy.get_param('~tracker_type', 'custom')

            # 预测参数
            self.total_delay = rospy.get_param('~total_delay', 0.08)
            self.prediction_time = rospy.get_param('~prediction_time', 0.15)
            self.use_kalman = rospy.get_param('~use_kalman', True)
            self.max_prediction_distance = rospy.get_param('~max_prediction_distance', 300)

            # 激光控制参数
            self.aiming_time = rospy.get_param('~aiming_time', 0.1)
            self.laser_time = rospy.get_param('~laser_time', 0.2)
            # 激光模式：'point' 点射模式 或 'spiral' 螺旋线模式
            self.laser_mode = rospy.get_param('~laser_mode', 'point').lower()
            if self.laser_mode not in ['point', 'spiral']:
                rospy.logwarn(f"Invalid laser_mode '{self.laser_mode}', defaulting to 'point'")
                self.laser_mode = 'point'
            # 螺旋线参数
            self.spiral_diameter_mm = rospy.get_param('~spiral_diameter_mm', 20.0)  # 螺旋线直径（毫米），默认2cm
            self.spiral_spacing_ratio = rospy.get_param('~spiral_spacing_ratio', 0.1)  # 螺旋线间距与半径的比值
            self.spiral_point_delay_us = rospy.get_param('~spiral_point_delay_us', 1500)  # 螺旋线点间延迟（微秒）
            self.spiral_angle_step = rospy.get_param('~spiral_angle_step', 0.25)  # 螺旋线角度步进（弧度）

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

            # ========== 设置激光模式 ==========
            if self.galvo_controller:
                mode_upper = self.laser_mode.upper()
                if self.galvo_controller.set_laser_mode(mode_upper):
                    rospy.loginfo(f"laser mode set to: {mode_upper}")
                else:
                    rospy.logwarn(f"failed to set laser mode to: {mode_upper}")

            # ========== 检测器初始化 ==========
            try:
                self.detector = WeedDetector(
                    model_path=self.model_path,
                    model_type=self.model_type,
                    weed_class_id=self.weed_class_id,
                    crop_class_id=1,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                    tracker_type=self.tracker_type,
                    detection_mode=self.detection_mode
                )
                rospy.loginfo(f' {self.model_type.upper()} loading model')
            except Exception as e:
                rospy.logerr(f" failed loading model {e}")
                rospy.logerr(traceback.format_exc())
                sys.exit(1)

            # ========== 状态变量初始化 ==========
            # 系统状态（保留旧变量以兼容，后续逐步移除）
            self.system_state = SystemState.IDLE
            self.state_start_time = time.time()

            # 目标管理
            self.current_target = None
            self.target_queue = []            # 将按"距离中心"动态排序
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
            self.galvo_pixel_targets = [[None, None] for _ in range(self.galvo_count)]
            self.galvo_regions = []
            self._update_galvo_regions()
            self.laser_on = False
            self.tracking_active = False

            # ========== 双振镜独立状态管理 ==========
            # 每个振镜独立的状态列表
            self.galvo_states = [SystemState.IDLE] * self.galvo_count
            self.galvo_current_targets = [None] * self.galvo_count
            self.galvo_laser_on = [False] * self.galvo_count
            self.galvo_state_start_times = [time.time()] * self.galvo_count
            self.galvo_position_histories = [deque(maxlen=20) for _ in range(self.galvo_count)]
            self.galvo_target_queues = [[] for _ in range(self.galvo_count)]
            
            # ========== 累计统计计数器 ==========
            # 每个振镜区域的累计检测数和作业数（即使目标被删除也保持累计）
            self.galvo_cumulative_detected = [0] * self.galvo_count
            self.galvo_cumulative_processed = [0] * self.galvo_count
            # 记录已计入累计统计的目标ID（避免重复计数）
            self.counted_targets = set()

            # 卡尔曼滤波器（每个振镜独立一个）
            self.kalman_filters = [None] * self.galvo_count
            if self.use_kalman:
                self.init_kalman_filters()

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
            
            # 跟踪评估发布器（可选，用于评估工具）
            self.enable_tracking_evaluation = rospy.get_param('~enable_tracking_evaluation', False)
            if self.enable_tracking_evaluation:
                self.tracking_results_pub = rospy.Publisher('/tracking_results', String, queue_size=10)
                rospy.loginfo("Tracking evaluation enabled - results will be published to /tracking_results")

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

            # ========== ROS 话题订阅 ==========
            # 激光模式切换话题订阅（通过发布String消息切换模式）
            self.laser_mode_sub = rospy.Subscriber(
                '~set_laser_mode',
                String,
                self.laser_mode_callback,
                queue_size=1
            )
            rospy.loginfo("laser mode topic available at: ~set_laser_mode (publish 'point' or 'spiral')")

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

            # 发布跟踪结果（用于评估）
            if self.enable_tracking_evaluation:
                self._publish_tracking_results(detections, current_time)

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
        """每个振镜独立运行状态机"""
        try:
            current_time = time.time()
            
            # 对每个振镜独立执行状态机逻辑
            for galvo_idx in range(self.galvo_count):
                self._control_single_galvo(galvo_idx, current_time)
            
            # 保持旧的全局状态以兼容可视化（取第一个活跃振镜的状态）
            for idx in range(self.galvo_count):
                if self.galvo_states[idx] != SystemState.IDLE:
                    self.system_state = self.galvo_states[idx]
                    if self.galvo_current_targets[idx]:
                        self.current_target = self.galvo_current_targets[idx]
                        self.processing_target = self.galvo_current_targets[idx].get('id')
                        self.active_galvo_index = idx
                    break
            else:
                # 所有振镜都空闲
                self.system_state = SystemState.IDLE
                self.current_target = None
                self.processing_target = None

        except Exception as e:
            rospy.logerr(f"control loop failed: {e}")
    
    def _get_target_point(self, target_info):
        """获取对靶点，优先使用target_point，否则使用center"""
        if 'target_point' in target_info and target_info['target_point']:
            return target_info['target_point']
        elif 'center' in target_info:
            return target_info['center']
        else:
            return None

    def _control_single_galvo(self, galvo_idx: int, current_time: float):
        """单个振镜的控制状态机"""
        try:
            state = self.galvo_states[galvo_idx]
            
            if state == SystemState.IDLE:
                # 空闲状态：从该振镜的队列中选择目标
                if not self.galvo_target_queues[galvo_idx]:
                    return
                
                # 取队列中的第一个目标
                target_id = self.galvo_target_queues[galvo_idx].pop(0)
                
                if target_id in self.all_targets and target_id not in self.processed_targets:
                    target_info = self.all_targets[target_id]
                    
                    # 设置该振镜的当前目标
                    self.galvo_current_targets[galvo_idx] = {
                        'id': target_id,
                        'start_time': current_time,
                        'galvo_index': galvo_idx
                    }
                    
                    # 清除该振镜的位置历史
                    self.galvo_position_histories[galvo_idx].clear()
                    
                    # 初始化该振镜的卡尔曼滤波器（使用对靶点）
                    target_point = self._get_target_point(target_info)
                    if target_point:
                        self.reset_kalman_filter(galvo_idx, target_point)
                    
                    # 初始化目标位置
                    with self.position_lock:
                        self.target_galvo_positions[galvo_idx] = self.galvo_positions[galvo_idx][:]
                    
                    # 切换到跟踪状态
                    self.change_galvo_state(galvo_idx, SystemState.TRACKING)
                    rospy.loginfo(f"Galvo {galvo_idx}: 开始跟踪目标 {target_id}")
            
            elif state == SystemState.TRACKING:
                # 跟踪状态：更新位置并检查是否可以开火
                current_target = self.galvo_current_targets[galvo_idx]
                if not current_target:
                    self.change_galvo_state(galvo_idx, SystemState.IDLE)
                    return
                
                target_id = current_target['id']
                
                # 检查目标是否已处理或丢失
                if target_id in self.processed_targets:
                    self.galvo_current_targets[galvo_idx] = None
                    self.change_galvo_state(galvo_idx, SystemState.IDLE)
                    return
                
                if target_id not in self.all_targets:
                    rospy.logwarn(f"Galvo {galvo_idx}: 目标 {target_id} 丢失")
                    self.galvo_current_targets[galvo_idx] = None
                    self.change_galvo_state(galvo_idx, SystemState.IDLE)
                    return
                
                # 更新位置历史（使用对靶点）
                target_info = self.all_targets[target_id]
                target_point = self._get_target_point(target_info)
                if target_point:
                    self.update_position_history(target_point, current_time, galvo_idx)
                    # 保存最后已知的bbox和位置，用于盲打模式
                    if 'bbox' in target_info:
                        current_target['last_known_bbox'] = target_info['bbox']
                    current_target['last_known_point'] = target_point
                
                # 检查是否达到瞄准时间
                elapsed = current_time - current_target['start_time']
                if elapsed >= self.aiming_time:
                    # 初始化盲打模式标志（进入 FIRING 时默认为 False，表示还未进入盲打模式）
                    current_target['blind_mode'] = False
                    self.change_galvo_state(galvo_idx, SystemState.FIRING)
                    rospy.loginfo(f"Galvo {galvo_idx}: 开始照射目标 {target_id}")
            
            elif state == SystemState.FIRING:
                # 照射状态：持续跟踪并照射（支持盲打模式）
                current_target = self.galvo_current_targets[galvo_idx]
                if not current_target:
                    self.change_galvo_state(galvo_idx, SystemState.IDLE)
                    return
                
                target_id = current_target['id']
                
                # ==================== 盲打模式实现 ====================
                # 1. 检查是否有视觉检测结果（排除虚拟目标）
                has_visual_contact = target_id in self.all_targets and not self.all_targets[target_id].get('is_virtual', False)
                
                if has_visual_contact:
                    # A. 有检测结果：正常执行 Predict + Update
                    # 如果之前在盲打模式，现在检测框恢复，退出盲打模式
                    if current_target.get('blind_mode', False):
                        current_target['blind_mode'] = False
                        rospy.logdebug(f"Galvo {galvo_idx}: 目标 {target_id} 检测框恢复，退出盲打模式")
                    
                    target_info = self.all_targets[target_id]
                    target_point = self._get_target_point(target_info)
                    
                    if target_point:
                        # 更新位置历史（会触发 Predict + Update）
                        self.update_position_history(target_point, current_time, galvo_idx)
                        # 保存最后已知位置和bbox，以防下一帧丢失
                        current_target['last_known_point'] = target_point
                        if 'bbox' in target_info:
                            current_target['last_known_bbox'] = target_info['bbox']
                else:
                    # B. 无检测结果（盲区）：仅执行 Predict（不执行 Update）
                    # 进入或保持盲打模式
                    if not current_target.get('blind_mode', False):
                        # 首次进入盲打模式，标记状态
                        current_target['blind_mode'] = True
                        rospy.logdebug(f"Galvo {galvo_idx}: 目标 {target_id} 进入盲打模式")
                    
                    # 使用卡尔曼滤波器的纯预测模式
                    predicted_pos = self.predict_position(0.0, galvo_idx, use_measurement=False)
                    
                    # 如果预测失败，尝试使用最后已知位置
                    if not predicted_pos and 'last_known_point' in current_target:
                        predicted_pos = current_target['last_known_point']
                    
                    if predicted_pos:
                        # 关键技巧：创建/更新虚拟目标对象并注入到 all_targets
                        # 这样后续的控制和显示逻辑可以正常工作
                        virtual_target = {
                            'bbox': current_target.get('last_known_bbox', [0, 0, 20, 20]),  # 使用最后已知的bbox或默认值
                            'center': predicted_pos,
                            'target_point': predicted_pos,
                            'confidence': 0.0,  # 虚拟目标置信度为0
                            'last_seen': current_time,
                            'galvo_index': galvo_idx,
                            'in_range': True,
                            'is_virtual': True,  # 标记为虚拟目标
                            'processed': False,
                            'stable_frames': 0
                        }
                        
                        # 如果有最后已知的bbox，使用它；否则基于预测位置生成一个
                        if 'last_known_bbox' in current_target:
                            virtual_target['bbox'] = current_target['last_known_bbox']
                        else:
                            # 基于预测位置生成一个默认大小的bbox
                            x, y = predicted_pos
                            default_w, default_h = 30, 30  # 默认bbox大小
                            virtual_target['bbox'] = [x - default_w/2, y - default_h/2, default_w, default_h]
                        
                        # 将虚拟目标注入到 all_targets（仅在盲打模式时）
                        self.all_targets[target_id] = virtual_target
                        
                        # 更新目标位置（用于控制）
                        clamped_px = float(np.clip(predicted_pos[0], 0, self.image_width - 1))
                        clamped_py = float(np.clip(predicted_pos[1], 0, self.image_height - 1))
                        
                        with self.position_lock:
                            self.galvo_pixel_targets[galvo_idx] = [clamped_px, clamped_py]
                        
                        # 转换为振镜坐标
                        galvo_result = self.coordinate_transform.pixel_to_galvo_code(
                            clamped_px, clamped_py,
                            self.image_width, self.image_height,
                            galvo_index=galvo_idx
                        )
                        
                        if galvo_result:
                            galvo_x, galvo_y = self.clamp_to_galvo_limits(galvo_idx, galvo_result[0], galvo_result[1])
                            with self.position_lock:
                                self.target_galvo_positions[galvo_idx] = [galvo_x, galvo_y]
                                self.active_galvo_index = galvo_idx
                
                # ==================== 盲打模式结束 ====================
                
                # 检查是否达到照射时间
                elapsed = current_time - self.galvo_state_start_times[galvo_idx]
                if elapsed >= self.laser_time:
                    # 标记目标已处理
                    was_already_processed = target_id in self.processed_targets
                    self.processed_targets.add(target_id)
                    if target_id in self.all_targets:
                        was_already_processed = was_already_processed or self.all_targets[target_id].get('processed', False)
                        self.all_targets[target_id]['processed'] = True
                    
                    # 如果目标刚被处理（之前未处理），增加该振镜的累计作业数
                    if not was_already_processed and 0 <= galvo_idx < self.galvo_count:
                        self.galvo_cumulative_processed[galvo_idx] += 1
                    
                    # 清除盲打模式标志
                    if 'blind_mode' in current_target:
                        blind_mode_used = current_target['blind_mode']
                        if blind_mode_used:
                            rospy.loginfo(f"Galvo {galvo_idx}: 完成照射目标 {target_id} (盲打模式)")
                        else:
                            rospy.loginfo(f"Galvo {galvo_idx}: 完成照射目标 {target_id}")
                    else:
                        rospy.loginfo(f"Galvo {galvo_idx}: 完成照射目标 {target_id}")
                    
                    # 清除当前目标并返回空闲
                    self.galvo_current_targets[galvo_idx] = None
                    self.change_galvo_state(galvo_idx, SystemState.IDLE)
        
        except Exception as e:
            rospy.logerr(f"Galvo {galvo_idx} control failed: {e}")

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

    def laser_mode_callback(self, msg):
        """激光模式切换回调函数"""
        try:
            mode_str = msg.data.strip().lower()
            if mode_str not in ['point', 'spiral']:
                rospy.logwarn(f"Invalid laser mode: '{mode_str}'. Use 'point' or 'spiral'")
                return

            if self.galvo_controller:
                mode_upper = mode_str.upper()
                if self.galvo_controller.set_laser_mode(mode_upper):
                    self.laser_mode = mode_str
                    rospy.loginfo(f"Laser mode changed to: {mode_upper}")
                else:
                    rospy.logwarn(f"Failed to set laser mode to: {mode_upper}")
            else:
                rospy.logwarn("Galvo controller not available")
        except Exception as e:
            rospy.logerr(f"Error in laser_mode_callback: {e}")

    def depth_query_func(self, u, v):
        """
        查询像素(u,v)的深度，单位：米。用于 coordinate_transform 中
        使用邻域中值法提高鲁棒性，避免单个像素噪声影响
        """
        with self.depth_image_lock:
            if self.depth_image is None:
                return None
            H, W = self.depth_image.shape[:2]
            if u < 0 or v < 0 or u >= W or v >= H:
                return None

            # 使用邻域中值法（3x3窗口）提高鲁棒性
            radius = 1  # 邻域半径（3x3窗口）
            y_int = int(round(v))
            x_int = int(round(u))

            # 边界检查
            y_min = max(0, y_int - radius)
            y_max = min(H, y_int + radius + 1)
            x_min = max(0, x_int - radius)
            x_max = min(W, x_int + radius + 1)

            # 提取邻域
            neighborhood = self.depth_image[y_min:y_max, x_min:x_max]

            # 过滤有效深度值（有限且>0）
            valid_depths = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0)]

            if len(valid_depths) == 0:
                return None

            # 使用中值（比均值更鲁棒，不受异常值影响）
            z = float(np.median(valid_depths))

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

    def init_kalman_filters(self):
        """为每个振镜初始化独立的2D卡尔曼滤波器（位置+速度）"""
        dt = 0.033
        
        for idx in range(self.galvo_count):
            kf = cv2.KalmanFilter(4, 2)
            
            kf.transitionMatrix = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            
            kf.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=np.float32)
            
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
            kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
            
            self.kalman_filters[idx] = kf
        
        rospy.loginfo(f"初始化了 {self.galvo_count} 个独立的卡尔曼滤波器")
    
    def reset_kalman_filter(self, galvo_idx: int, initial_position: List[float]):
        """重置指定振镜的卡尔曼滤波器状态到初始位置"""
        if not self.use_kalman or galvo_idx >= len(self.kalman_filters):
            return
        
        kf = self.kalman_filters[galvo_idx]
        if kf is None:
            return
        
        # 重置状态：位置为初始位置，速度为0
        kf.statePre = np.array(
            [[initial_position[0]], [initial_position[1]], [0], [0]],
            dtype=np.float32
        )
        kf.statePost = np.array(
            [[initial_position[0]], [initial_position[1]], [0], [0]],
            dtype=np.float32
        )
        # 重置误差协方差
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 100

    def update_targets(self, detections, current_time):
        """更新目标信息并按区域分配到各振镜队列"""
        current_frame_ids = set()

        for detection in detections:
            # 支持新格式 (track_id, bbox, conf, target_point) 和旧格式 (track_id, bbox, conf)
            if len(detection) == 4:
                track_id, bbox, confidence, target_point = detection
            elif len(detection) == 3:
                track_id, bbox, confidence = detection
                # 旧格式：使用bbox中心作为对靶点
                x, y, w, h = bbox
                target_point = [x + w / 2.0, y + h / 2.0]
            else:
                rospy.logwarn(f"Unexpected detection format: {detection}")
                continue

            if confidence < self.confidence_threshold:
                continue

            x, y, w, h = bbox
            cx = x + w / 2.0
            cy = y + h / 2.0
            current_frame_ids.add(track_id)

            # 使用target_point进行区域分配和范围检查（如果存在）
            aim_point = target_point if target_point else [cx, cy]
            galvo_index = self.select_galvo_for_pixel(aim_point[0], aim_point[1])
            in_range = self.in_galvo_scan_range(aim_point[0], aim_point[1])
            
            # 判断是否是新目标（首次检测到）
            is_new_target = track_id not in self.all_targets
            if is_new_target:
                self.all_targets[track_id] = {
                    'first_seen': current_time,
                    'stable_frames': 0,
                    'processed': False
                }
                # 新目标：增加对应振镜的累计检测数
                if 0 <= galvo_index < self.galvo_count:
                    self.galvo_cumulative_detected[galvo_index] += 1
                    self.counted_targets.add(track_id)
            
            # 检查目标是否正在 FIRING 状态且是虚拟目标，且处于盲打模式
            # 只有在盲打模式时，才阻止覆盖虚拟目标（因为此时应该使用预测）
            # 如果检测框恢复（不在盲打模式），应该允许覆盖虚拟目标
            is_firing_blind_mode = False
            if track_id in self.all_targets:
                existing_target = self.all_targets[track_id]
                if existing_target.get('is_virtual', False):
                    # 检查是否有振镜正在 FIRING 这个目标，且处于盲打模式
                    for idx in range(self.galvo_count):
                        if (self.galvo_states[idx] == SystemState.FIRING and
                            self.galvo_current_targets[idx] and
                            self.galvo_current_targets[idx].get('id') == track_id):
                            # 检查是否在盲打模式
                            if self.galvo_current_targets[idx].get('blind_mode', False):
                                is_firing_blind_mode = True
                                break
            
            # 如果目标正在 FIRING 且处于盲打模式，不覆盖虚拟目标（保持预测模式）
            # 否则（检测框恢复），允许覆盖虚拟目标，使用真实检测
            if not is_firing_blind_mode:
                self.all_targets[track_id].update({
                    'bbox': bbox,
                    'center': [cx, cy],  # bbox中心（用于跟踪）
                    'target_point': target_point,  # 对靶点（用于激光瞄准）
                    'confidence': confidence,
                    'last_seen': current_time,
                    'galvo_index': galvo_index,
                    'in_range': in_range,
                    'is_virtual': False  # 清除虚拟标记（如果有真实检测）
                })

            # 检查目标是否被处理
            if track_id in self.processed_targets:
                self.all_targets[track_id]['processed'] = True
            else:
                # 仅未处理目标累积稳定帧数
                self.all_targets[track_id]['stable_frames'] = self.all_targets[track_id].get('stable_frames', 0) + 1

        # 更新各振镜的目标队列
        self._update_galvo_queues(current_time)

        # 清理超时目标（但保护 FIRING 状态中的虚拟目标）
        to_remove = []
        for tid, tinfo in list(self.all_targets.items()):
            if tid not in current_frame_ids:
                # 检查目标是否正在 FIRING 状态
                is_firing = False
                for idx in range(self.galvo_count):
                    if (self.galvo_states[idx] == SystemState.FIRING and
                        self.galvo_current_targets[idx] and
                        self.galvo_current_targets[idx].get('id') == tid):
                        is_firing = True
                        break
                
                # 如果目标正在 FIRING，不删除（即使超时也要等到 FIRING 结束）
                if is_firing:
                    continue
                
                timeout = self.target_timeout * (3 if tid in self.processed_targets else 1)
                if current_time - tinfo.get('last_seen', current_time) > timeout:
                    to_remove.append(tid)

        for tid in to_remove:
            if tid in self.all_targets:
                del self.all_targets[tid]
            # 从各振镜的当前目标和队列中移除
            for idx in range(self.galvo_count):
                if self.galvo_current_targets[idx] and self.galvo_current_targets[idx].get('id') == tid:
                    rospy.logwarn(f"Galvo {idx}: 目标 {tid} 丢失")
                    self.galvo_current_targets[idx] = None
                    self.change_galvo_state(idx, SystemState.IDLE)
                if tid in self.galvo_target_queues[idx]:
                    try:
                        self.galvo_target_queues[idx].remove(tid)
                    except ValueError:
                        pass
            # 保留旧变量以兼容
            if self.current_target and self.current_target.get('id') == tid:
                self.current_target = None
                self.processing_target = None
            if tid in self.target_queue:
                try:
                    self.target_queue.remove(tid)
                except ValueError:
                    pass
    
    def _update_galvo_queues(self, current_time):
        """根据当前所有目标，为每个振镜构建优先级队列"""
        # 清空旧队列
        for idx in range(self.galvo_count):
            self.galvo_target_queues[idx].clear()
        
        # 按振镜区域分组候选目标
        galvo_candidates = [[] for _ in range(self.galvo_count)]
        
        for tid, info in self.all_targets.items():
            # 跳过已处理的
            if tid in self.processed_targets:
                continue
            # 跳过正在被其他振镜处理的
            if any(ct and ct.get('id') == tid for ct in self.galvo_current_targets):
                continue
            # 必须有中心点
            if 'center' not in info:
                continue
            # 目标超时丢弃
            if current_time - info.get('last_seen', current_time) > self.target_timeout:
                continue
            # 必须稳定到一定帧数
            if info.get('stable_frames', 0) < self.min_stable_frames:
                continue
            # 在扫描范围内才作为候选
            if not info.get('in_range', False):
                continue
            
            galvo_idx = info.get('galvo_index', 0)
            
            # 计算到该区域中心的距离（区域内优先中心）
            if galvo_idx < len(self.galvo_regions):
                region = self.galvo_regions[galvo_idx]
                center_u = (region['u_min'] + region['u_max']) / 2.0
                center_v = (region['v_min'] + region['v_max']) / 2.0
            else:
                center_u = self.image_width / 2.0
                center_v = self.image_height / 2.0
            
            # 使用对靶点计算距离
            target_point = self._get_target_point(info)
            if target_point:
                cx, cy = target_point
                dist = np.hypot(cx - center_u, cy - center_v)
                galvo_candidates[galvo_idx].append((dist, tid))
        
        # 对每个振镜的候选目标按距离排序
        for idx in range(self.galvo_count):
            galvo_candidates[idx].sort(key=lambda x: x[0])
            self.galvo_target_queues[idx] = [tid for _, tid in galvo_candidates[idx]]

    def update_position_history(self, position, timestamp, galvo_index):
        """更新位置历史并计算预测位置（使用对应振镜的历史）"""
        # 使用对应振镜的位置历史
        if galvo_index < len(self.galvo_position_histories):
            self.galvo_position_histories[galvo_index].append({
                'position': position,
                'time': timestamp
            })
        else:
            # 兼容旧代码
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

            if predicted_pos:
                clamped_px = float(np.clip(predicted_pos[0], 0, self.image_width - 1))
                clamped_py = float(np.clip(predicted_pos[1], 0, self.image_height - 1))
                with self.position_lock:
                    self.galvo_pixel_targets[galvo_index] = [clamped_px, clamped_py]

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

    def predict_position(self, dt, galvo_index, use_measurement=True):
        """
        预测未来位置（使用对应振镜的卡尔曼滤波器和位置历史）
        
        Args:
            dt: 预测时间间隔（秒）
            galvo_index: 振镜索引
            use_measurement: 是否使用测量值进行校正（True=Predict+Update, False=Predict Only）
        
        Returns:
            预测位置 [x, y] 或 None
        """
        # 使用对应振镜的位置历史
        if galvo_index < len(self.galvo_position_histories):
            relevant_history = list(self.galvo_position_histories[galvo_index])
        else:
            # 兼容旧代码：从全局历史中筛选
            relevant_history = [
                entry for entry in self.position_history if entry.get('galvo') == galvo_index
            ]

        if len(relevant_history) < 1:
            return None
        
        if len(relevant_history) < 2:
            return relevant_history[-1]['position'] if relevant_history else None

        # 使用对应振镜的卡尔曼滤波器
        if self.use_kalman and galvo_index < len(self.kalman_filters):
            kf = self.kalman_filters[galvo_index]
            if kf is not None:
                try:
                    if use_measurement:
                        # 标准模式：Predict + Update（有测量值）
                        current_pos = relevant_history[-1]['position']
                        measurement = np.array([[current_pos[0]], [current_pos[1]]], dtype=np.float32)
                        kf.correct(measurement)
                        kf.predict()
                        state = kf.statePost
                    else:
                        # 盲打模式：仅 Predict（无测量值，纯预测）
                        kf.predict()
                        state = kf.statePre  # 预测后的状态在 statePre 中
                    
                    pred_x = state[0, 0] + state[2, 0] * dt
                    pred_y = state[1, 0] + state[3, 0] * dt

                    return [float(pred_x), float(pred_y)]
                except Exception as e:
                    rospy.logdebug(f"Galvo {galvo_index} kalman predict failed: {e}")

        # 回退到简单速度预测
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

        return relevant_history[-1]['position'] if relevant_history else None

    def galvo_control_loop(self):
        """振镜控制线程（高频率），每个振镜独立发送命令"""
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
                    
                    # 只有当距离超过阈值时才发送移动命令
                    if distance > self.min_move_step and self.galvo_controller:
                        self.galvo_controller.move_to_position(
                            cmd_x,
                            cmd_y,
                            galvo_index=idx
                        )

                    self.galvo_positions[idx] = [cmd_x, cmd_y]

                    # 发布该振镜的位置和激光状态
                    galvo_msg = Int32MultiArray()
                    galvo_msg.data = [
                        cmd_x,
                        cmd_y,
                        1 if self.galvo_laser_on[idx] else 0,
                        idx
                    ]
                    self.galvo_pub.publish(galvo_msg)

                rate.sleep()

            except Exception as e:
                rospy.logerr(f"galvo control error: {e}")
                time.sleep(0.001)

    def change_galvo_state(self, galvo_idx, new_state):
        """改变指定振镜的状态"""
        self.galvo_states[galvo_idx] = new_state
        self.galvo_state_start_times[galvo_idx] = time.time()

        if new_state == SystemState.IDLE:
            self.set_galvo_laser(galvo_idx, False)
        elif new_state == SystemState.TRACKING:
            self.set_galvo_laser(galvo_idx, False)
        elif new_state == SystemState.FIRING:
            self.set_galvo_laser(galvo_idx, True)

    def change_state(self, new_state):
        """改变系统状态（保留用于兼容，后续移除）"""
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

    def calculate_spiral_radius_from_bbox(self, bbox: List[float], galvo_idx: int = 0) -> int:
        """
        根据bounding box计算螺旋线半径（振镜代码）
        螺旋线半径 = bounding box短边长的一半
        
        参数:
            bbox: bounding box [x, y, w, h]（像素）
            galvo_idx: 振镜索引
            
        返回:
            螺旋线半径（振镜代码）
        """
        if bbox is None or len(bbox) < 4:
            # 如果bbox无效，使用默认值
            rospy.logwarn("Invalid bbox for spiral calculation, using default radius")
            return 6000
        
        try:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # 计算短边长的一半（像素）
            short_side = min(w, h)
            radius_px = short_side / 2.0
            
            # 将像素半径转换为振镜代码
            # 使用坐标变换：将像素坐标转换为振镜代码
            # 计算bbox中心点和半径对应的两个点（中心点和中心+半径点）
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            
            # 计算半径对应的点（在短边方向上）
            if w <= h:
                # 宽度更短，在x方向扩展
                radius_point_x = center_x + radius_px
                radius_point_y = center_y
            else:
                # 高度更短，在y方向扩展
                radius_point_x = center_x
                radius_point_y = center_y + radius_px
            
            # 将中心点和半径点转换为振镜代码
            center_code = self.coordinate_transform.pixel_to_galvo_code(
                center_x, center_y,
                self.image_width, self.image_height,
                galvo_index=galvo_idx
            )
            radius_code = self.coordinate_transform.pixel_to_galvo_code(
                radius_point_x, radius_point_y,
                self.image_width, self.image_height,
                galvo_index=galvo_idx
            )
            
            if center_code is None or radius_code is None:
                rospy.logwarn("Failed to convert pixel to galvo code for spiral radius, using default")
                return 6000
            
            # 计算振镜代码中的半径（欧氏距离）
            code_radius = int(np.hypot(
                radius_code[0] - center_code[0],
                radius_code[1] - center_code[1]
            ))
            
            # 限制范围，确保合理（最小100，最大20000）
            code_radius = max(100, min(code_radius, 20000))
            
            return code_radius
            
        except Exception as e:
            rospy.logwarn(f"Error calculating spiral radius from bbox: {e}, using default")
            return 6000

    def configure_spiral_for_target(self, galvo_idx: int, target_id: int = None):
        """
        根据目标的bounding box配置螺旋参数
        螺旋线半径 = bounding box短边长的一半
        
        参数:
            galvo_idx: 振镜索引
            target_id: 目标ID（可选，如果不提供则从galvo_current_targets获取）
        """
        if self.laser_mode != 'spiral' or not self.galvo_controller:
            return
        
        try:
            # 获取目标信息
            target_info = None
            if target_id is not None and target_id in self.all_targets:
                target_info = self.all_targets[target_id]
            elif self.galvo_current_targets[galvo_idx]:
                current_target = self.galvo_current_targets[galvo_idx]
                target_id = current_target.get('id')
                if target_id and target_id in self.all_targets:
                    target_info = self.all_targets[target_id]
            
            # 获取bbox
            bbox = None
            if target_info and 'bbox' in target_info:
                bbox = target_info['bbox']
            elif self.galvo_current_targets[galvo_idx]:
                # 尝试从last_known_bbox获取
                bbox = self.galvo_current_targets[galvo_idx].get('last_known_bbox')
            
            if bbox is None or len(bbox) < 4:
                rospy.logwarn(f"Galvo {galvo_idx}: 无法获取目标bbox，使用默认螺旋参数")
                return
            
            # 计算螺旋半径（基于bbox短边长的一半）
            spiral_radius = self.calculate_spiral_radius_from_bbox(bbox, galvo_idx)
            
            # 计算螺旋间距（基于半径的百分比）
            spiral_spacing = spiral_radius * self.spiral_spacing_ratio
            
            # 配置螺旋参数
            if self.galvo_controller.select_galvo(galvo_idx):
                self.galvo_controller.configure_spiral(
                    radius=spiral_radius,
                    spacing=spiral_spacing,
                    dwell_us=self.spiral_point_delay_us,
                    angle_step=self.spiral_angle_step
                )
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                short_side = min(w, h)
                rospy.logdebug(f"Galvo {galvo_idx}: 配置螺旋参数 - 半径={spiral_radius}, 间距={spiral_spacing:.1f}, bbox短边={short_side:.1f}px")
        
        except Exception as e:
            rospy.logwarn(f"Failed to configure spiral for galvo {galvo_idx}: {e}")

    def set_galvo_laser(self, galvo_idx, enable):
        """控制指定振镜的激光（独立控制，互不干扰）"""
        if self.galvo_laser_on[galvo_idx] != enable:
            self.galvo_laser_on[galvo_idx] = enable

            if self.galvo_controller:
                try:
                    # 选择对应的振镜并控制其激光
                    if self.galvo_controller.select_galvo(galvo_idx):
                        if enable:
                            # 如果是螺旋模式，先根据bbox配置螺旋参数
                            if self.laser_mode == 'spiral':
                                self.configure_spiral_for_target(galvo_idx)
                            
                            self.galvo_controller.send_command("LASER:ON")
                            rospy.logdebug(f"Galvo {galvo_idx}: 激光开启 (模式: {self.laser_mode})")
                        else:
                            self.galvo_controller.send_command("LASER:OFF")
                            rospy.logdebug(f"Galvo {galvo_idx}: 激光关闭")
                except Exception as e:
                    rospy.logerr(f"failed control galvo {galvo_idx} laser: {e}")
            
            # 同步到全局激光状态（兼容旧代码，取任一振镜激光状态）
            self.laser_on = any(self.galvo_laser_on)

    def set_laser(self, enable):
        """控制激光（保留用于兼容，后续移除）"""
        if not hasattr(self, 'laser_on'):
            self.laser_on = False
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
        """在图像上绘制信息（支持双振镜独立显示）"""
        result = image.copy()

        # 绘制所有目标
        for track_id, target_info in self.all_targets.items():
            if 'bbox' not in target_info:
                continue

            bbox = target_info['bbox']
            # 使用对靶点进行绘制和判断
            target_point = self._get_target_point(target_info)
            center = target_point if target_point else target_info.get('center', [0, 0])
            x, y, w, h = bbox

            # 判断目标当前状态
            is_processed = track_id in self.processed_targets or target_info.get('processed', False)
            active_galvo_idx = None
            current_state = None
            
            # 检查是否正在被某个振镜处理
            for idx in range(self.galvo_count):
                if self.galvo_current_targets[idx] and self.galvo_current_targets[idx].get('id') == track_id:
                    active_galvo_idx = idx
                    current_state = self.galvo_states[idx]
                    break
            
            # 检查是否为虚拟目标（盲打模式）
            is_virtual = target_info.get('is_virtual', False)
            
            # 根据状态设置颜色和标签
            if is_processed:
                color = (128, 128, 128)  # 灰色：已处理
                label = f"ID:{track_id} [completed]"
                thickness = 1
            elif active_galvo_idx is not None:
                if current_state == SystemState.FIRING:
                    if is_virtual:
                        # 虚拟目标（盲打模式）：使用虚线框或特殊颜色
                        color = (255, 0, 255)  # 品红色：盲打模式
                        label = f"ID:{track_id} [G{active_galvo_idx} BLIND]"
                    else:
                        color = (0, 0, 255)  # 红色：激光照射（有检测）
                        label = f"ID:{track_id} [G{active_galvo_idx} laser]"
                else:
                    color = (0, 255, 255)  # 黄色：跟踪中
                    label = f"ID:{track_id} [G{active_galvo_idx} tracking]"
                thickness = 2
            else:
                color = (0, 255, 0)  # 绿色：检测到
                galvo_idx = target_info.get('galvo_index', 0)
                label = f"ID:{track_id} [G{galvo_idx}]"
                thickness = 1

            # 绘制检测框
            if is_virtual:
                # 虚拟目标：使用虚线框（通过绘制多个小线段模拟虚线）
                line_length = 5
                gap_length = 3
                # 上边
                for px in range(int(x), int(x + w), line_length + gap_length):
                    cv2.line(result, (px, int(y)), (min(px + line_length, int(x + w)), int(y)), color, thickness)
                # 下边
                for px in range(int(x), int(x + w), line_length + gap_length):
                    cv2.line(result, (px, int(y + h)), (min(px + line_length, int(x + w)), int(y + h)), color, thickness)
                # 左边
                for py in range(int(y), int(y + h), line_length + gap_length):
                    cv2.line(result, (int(x), py), (int(x), min(py + line_length, int(y + h))), color, thickness)
                # 右边
                for py in range(int(y), int(y + h), line_length + gap_length):
                    cv2.line(result, (int(x + w), py), (int(x + w), min(py + line_length, int(y + h))), color, thickness)
            else:
                # 真实目标：使用实线框
                cv2.rectangle(result, (int(x), int(y)),
                            (int(x + w), int(y + h)), color, thickness)
            cv2.circle(result, (int(center[0]), int(center[1])), 3, color, -1)
            cv2.putText(result, label, (int(x), int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制所有振镜的激光瞄准位置
        legend_entries = []
        palette = [
            (255, 170, 0),
            (0, 170, 255),
            (200, 200, 200),
            (255, 0, 255)
        ]
        try:
            for idx, active_pos in enumerate(self.galvo_positions):
                if not isinstance(active_pos, (list, tuple)) or len(active_pos) < 2:
                    continue

                galvo_pixel = None

                if self.coordinate_transform and self.use_reverse_projection:
                    try:
                        galvo_pixel = self.coordinate_transform.galvo_code_to_pixel(
                            active_pos[0], active_pos[1],
                            self.image_width, self.image_height,
                            galvo_index=idx
                        )
                    except Exception as exc:
                        rospy.logdebug(f"Failed reverse transform for galvo {idx}: {exc}")
                        galvo_pixel = None

                if galvo_pixel is None:
                    with self.position_lock:
                        pixel_target = self.galvo_pixel_targets[idx][:] if self.galvo_pixel_targets[idx] else None
                    if pixel_target and pixel_target[0] is not None and pixel_target[1] is not None:
                        galvo_pixel = pixel_target

                if galvo_pixel is None:
                    legend_entries.append((f"G{idx}: 坐标未知", (0, 0, 255)))
                    continue

                xg = int(round(galvo_pixel[0]))
                yg = int(round(galvo_pixel[1]))

                # 根据该振镜的状态设置颜色和标签
                galvo_state = self.galvo_states[idx]
                if galvo_state == SystemState.FIRING and self.galvo_laser_on[idx]:
                    color = (0, 0, 255)
                    status = "LASER"
                elif galvo_state == SystemState.TRACKING:
                    color = (0, 255, 255)
                    status = "AIM"
                else:
                    color = palette[idx % len(palette)]
                    status = "IDLE"

                is_active = galvo_state != SystemState.IDLE
                cross_half = 15 if is_active else 12
                thickness = 3 if is_active else 2
                cv2.line(result, (xg - cross_half, yg), (xg + cross_half, yg), color, thickness)
                cv2.line(result, (xg, yg - cross_half), (xg, yg + cross_half), color, thickness)
                cv2.circle(result, (xg, yg), cross_half - 5, color, 2)
                cv2.putText(result, f"G{idx}", (xg + 12, yg - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 显示目标ID（如果正在处理）
                target_id_str = ""
                if self.galvo_current_targets[idx]:
                    target_id_str = f" T:{self.galvo_current_targets[idx].get('id', '?')}"
                
                legend_entries.append((
                    f"G{idx} {status}: code({active_pos[0]:.0f},{active_pos[1]:.0f}) pix({xg},{yg}){target_id_str}",
                    color
                ))

            if legend_entries:
                text_y = 150
                for text, color in legend_entries:
                    cv2.putText(result, text, (10, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    text_y += 24
            else:
                cv2.putText(result, "GALVO POS UNKNOWN", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        except Exception as e:
            rospy.logdebug(f"Failed to draw galvo position: {e}")
            cv2.putText(result, "GALVO DISPLAY ERROR", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示系统整体状态（统计各振镜状态）
        states_summary = f"Galvo States: "
        for idx in range(self.galvo_count):
            states_summary += f"G{idx}:{self.galvo_states[idx].value} "
        
        cv2.putText(result, states_summary, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示队列信息
        queue_info = f"Queues: "
        for idx in range(self.galvo_count):
            queue_info += f"G{idx}:{len(self.galvo_target_queues[idx])} "
        cv2.putText(result, queue_info, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result

    def publish_status(self, event):
        """发布系统状态（包含双振镜独立状态）"""
        try:
            fps = 0
            if len(self.fps_counter) > 1:
                fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])

            transform_info = self.coordinate_transform.get_transform_info()

            # 构建各振镜的状态信息，并统计每个振镜的检测数和作业数（使用累计值）
            galvo_states_info = []
            # 为每个振镜使用累计统计数据（累加的总和）
            for idx in range(self.galvo_count):
                # 使用累计检测数和累计作业数（即使目标被删除也保持累计）
                detected_count = self.galvo_cumulative_detected[idx]
                processed_count = self.galvo_cumulative_processed[idx]
                
                galvo_info = {
                    'index': idx,
                    'state': self.galvo_states[idx].value,
                    'laser_on': self.galvo_laser_on[idx],
                    'current_target': self.galvo_current_targets[idx].get('id') if self.galvo_current_targets[idx] else None,
                    'queue_size': len(self.galvo_target_queues[idx]),
                    'position': self.galvo_positions[idx],
                    'detected_count': detected_count,
                    'processed_count': processed_count
                }
                galvo_states_info.append(galvo_info)

            # 构建各振镜的统计数据（兼容 GUI 的解析格式）
            galvo_stats = {}
            for idx in range(self.galvo_count):
                if idx < len(galvo_states_info):
                    galvo_info = galvo_states_info[idx]
                    galvo_stats[f'galvo_{idx}'] = {
                        'detected_count': galvo_info.get('detected_count', 0),
                        'processed_count': galvo_info.get('processed_count', 0)
                    }
                    # 同时提供扁平化格式（galvo_0_detected_count）
                    galvo_stats[f'galvo_{idx}_detected_count'] = galvo_info.get('detected_count', 0)
                    galvo_stats[f'galvo_{idx}_processed_count'] = galvo_info.get('processed_count', 0)
            
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
                # 新增：各振镜的独立状态
                'galvo_states': galvo_states_info,
                # 新增：各振镜的统计数据（兼容 GUI 解析）
                **galvo_stats,  # 展开 galvo_stats 字典
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

            # 发布当前目标信息（兼容旧代码）
            if self.current_target and self.current_target['id'] in self.all_targets:
                target_info = self.all_targets[self.current_target['id']]
                assigned_galvo = target_info.get('galvo_index', self.active_galvo_index)
                target_data = {
                    'id': self.current_target['id'],
                    'center': target_info.get('center', [0, 0]),
                    'target_point': target_info.get('target_point', target_info.get('center', [0, 0])),
                    'confidence': target_info['confidence'],
                    'galvo_index': assigned_galvo,
                    'predicted_position': self.predict_position(self.prediction_time, assigned_galvo)
                }
                target_msg = String()
                target_msg.data = json.dumps(target_data)
                self.target_pub.publish(target_msg)

        except Exception as e:
            rospy.logerr(f"failed publish status: {e}")

    def _publish_tracking_results(self, detections, timestamp):
        """发布跟踪结果用于评估"""
        try:
            # 构建跟踪结果数据
            detections_data = []
            for det in detections:
                # 支持新格式 (track_id, bbox, conf, target_point) 和旧格式 (track_id, bbox, conf)
                if len(det) == 4:
                    track_id, bbox, conf, target_point = det
                elif len(det) == 3:
                    track_id, bbox, conf = det
                    target_point = None
                else:
                    continue
                
                detections_data.append({
                    'track_id': int(track_id),
                    'bbox': [float(x) for x in bbox],
                    'confidence': float(conf),
                    'target_point': [float(x) for x in target_point] if target_point else None
                })
            
            tracking_data = {
                'frame_id': self.frame_count,
                'timestamp': timestamp,
                'detections': detections_data
            }
            
            msg = String()
            msg.data = json.dumps(tracking_data)
            self.tracking_results_pub.publish(msg)
            
        except Exception as e:
            rospy.logdebug(f"Error publishing tracking results: {e}")

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
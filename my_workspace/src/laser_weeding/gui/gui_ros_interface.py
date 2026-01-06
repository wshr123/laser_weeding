#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROS 通信接口模块
使用 QThread 实现非阻塞的 ROS 通信
"""

import rospy
import sys
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QObject
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, Bool, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import subprocess
import os
import signal
import threading


class ROSInterfaceThread(QThread):
    """ROS 通信线程，负责订阅和发布 ROS 话题"""
    
    # 信号定义
    image_received = pyqtSignal(np.ndarray)  # 检测结果图像
    raw_image_received = pyqtSignal(np.ndarray)  # 原始 RGB 图像
    calib_image_received = pyqtSignal(np.ndarray)  # 标定辅助图像
    calib_status_updated = pyqtSignal(dict)  # 标定系统状态
    statistics_updated = pyqtSignal(dict)  # 统计信息更新
    status_message = pyqtSignal(str, str)  # 状态消息 (message, level)
    launch_status_changed = pyqtSignal(bool)  # Launch 启动状态变化
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 注意：CvBridge 应该在 run() 方法中创建，避免线程问题
        self.bridge = None
        # QMutex 可以在主线程创建，但为了安全，我们在 run() 中创建
        self.mutex = None
        
        # ROS 节点状态
        self.ros_initialized = False
        self.running = False
        
        # 订阅器
        self.det_img_sub = None
        self.raw_img_sub = None
        self.calib_img_sub = None
        self.calib_status_sub = None
        self.status_sub = None
        self.target_sub = None
        
        # 发布器
        self.galvo_pub = None
        self.laser_pub = None
        self.calib_cmd_pub = None
        
        # 统计数据（支持双振镜分别统计）
        self.stats = {
            'detected_count': 0,  # 全局检测数量（向后兼容）
            'processed_count': 0,  # 全局作业数量（向后兼容）
            'galvo_0_detected_count': 0,  # 振镜0检测数量
            'galvo_0_processed_count': 0,  # 振镜0作业数量
            'galvo_1_detected_count': 0,  # 振镜1检测数量
            'galvo_1_processed_count': 0,  # 振镜1作业数量
            'fps': 0.0,
            'current_target': None,
            'system_status': 'IDLE'
        }
        
        # FPS 计算
        self.frame_times = []
        self.last_frame_time = None
        
    def initialize_ros(self):
        """初始化 ROS 节点（在线程中调用）"""
        try:
            # 在线程中初始化 ROS，避免线程问题
            import rospy
            if not rospy.get_node_uri():
                rospy.init_node('laser_weeding_gui', anonymous=True, disable_signals=True)
            self.ros_initialized = True
            self.status_message.emit("ROS 节点初始化成功", "info")
            return True
        except Exception as e:
            self.status_message.emit(f"ROS 初始化失败: {str(e)}", "error")
            return False
    
    def setup_subscribers(self):
        """设置订阅器"""
        if not self.ros_initialized:
            return False
            
        try:
            # 检查话题是否存在
            available_topics = [topic[0] for topic in rospy.get_published_topics()]
            print(f"DEBUG: 可用的话题: {available_topics}")
            
            # 订阅检测结果图像（如果存在，否则等待自动连接）
            if '/det_img/image_raw' in available_topics:
                print("DEBUG: 订阅 /det_img/image_raw")
                self.det_img_sub = rospy.Subscriber(
                    '/det_img/image_raw',
                    Image,
                    self.det_image_callback,
                    queue_size=1
                )
            
            # 订阅原始 RGB 图像
            raw_topic = '/camera/color/image_raw'
            if raw_topic in available_topics:
                print(f"DEBUG: 订阅 {raw_topic}")
                self.raw_img_sub = rospy.Subscriber(
                    raw_topic,
                    Image,
                    self.raw_image_callback,
                    queue_size=1
                )
            
            # 订阅系统状态（如果存在，否则等待自动连接）
            if '/system_status' in available_topics:
                print("DEBUG: 订阅 /system_status")
                self.status_sub = rospy.Subscriber(
                    '/system_status',
                    String,
                    self.status_callback,
                    queue_size=5
                )
            else:
                print("DEBUG: /system_status 话题不存在，将在话题出现时自动连接")
            
            # 订阅当前目标（如果存在，否则等待自动连接）
            if '/current_target' in available_topics:
                print("DEBUG: 订阅 /current_target")
                self.target_sub = rospy.Subscriber(
                    '/current_target',
                    String,
                    self.target_callback,
                    queue_size=5
                )
            else:
                print("DEBUG: /current_target 话题不存在，将在话题出现时自动连接")
            
            self.status_message.emit("ROS 订阅器设置完成（将自动连接话题）", "info")
            return True
        except Exception as e:
            error_msg = f"设置订阅器失败: {str(e)}"
            print(f"DEBUG: {error_msg}")
            import traceback
            traceback.print_exc()
            self.status_message.emit(error_msg, "error")
            return False
    
    def setup_publishers(self):
        """设置发布器"""
        if not self.ros_initialized:
            return False
            
        try:
            self.galvo_pub = rospy.Publisher('/galvo_xy', Int32MultiArray, queue_size=1)
            self.laser_pub = rospy.Publisher('/laser_control', Bool, queue_size=1)
            self.calib_cmd_pub = rospy.Publisher('/manual_calibration_command', String, queue_size=1)
            self.status_message.emit("ROS 发布器设置完成", "info")
            return True
        except Exception as e:
            self.status_message.emit(f"设置发布器失败: {str(e)}", "error")
            return False
    
    def calib_image_callback(self, msg):
        """标定图像回调"""
        if self.bridge is None: return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.calib_image_received.emit(cv_image)
        except Exception as e:
            print(f"DEBUG: 标定图像处理错误: {e}")

    def calib_status_callback(self, msg):
        """标定状态回调"""
        try:
            status_data = json.loads(msg.data)
            self.calib_status_updated.emit(status_data)
        except Exception as e:
            print(f"DEBUG: 标定状态处理错误: {e}")

    def send_calib_command(self, cmd):
        """发送标定指令"""
        if self.calib_cmd_pub:
            self.calib_cmd_pub.publish(String(data=cmd))
            return True
        return False

    def det_image_callback(self, msg):
        """检测结果图像回调"""
        if self.bridge is None:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_received.emit(cv_image)
        except Exception as e:
            print(f"DEBUG: 图像处理错误: {e}")

    def raw_image_callback(self, msg):
        """原始 RGB 图像回调"""
        if self.bridge is None:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.raw_image_received.emit(cv_image)
        except Exception as e:
            print(f"DEBUG: 原始图像处理错误: {e}")
    
    def status_callback(self, msg):
        """系统状态回调"""
        if self.mutex is None:
            return
        try:
            status_data = json.loads(msg.data)
            # print(f"DEBUG: 收到状态数据: {status_data}")  # 可以取消注释来查看状态数据
            with QMutexLocker(self.mutex):
                # 全局检测数量（优先使用 total_targets）
                if 'total_targets' in status_data:
                    self.stats['detected_count'] = int(status_data['total_targets'])
                    self.stats['total_targets'] = int(status_data['total_targets'])  # 同时保存为 total_targets
                elif 'detected_count' in status_data:
                    self.stats['detected_count'] = int(status_data['detected_count'])
                
                # 全局作业数量
                if 'processed_count' in status_data:
                    self.stats['processed_count'] = int(status_data['processed_count'])
                
                # 优先使用扁平化格式（galvo_X_detected_count）
                for galvo_idx in range(2):
                    detected_key = f'galvo_{galvo_idx}_detected_count'
                    processed_key = f'galvo_{galvo_idx}_processed_count'
                    if detected_key in status_data:
                        self.stats[detected_key] = int(status_data[detected_key])
                    if processed_key in status_data:
                        self.stats[processed_key] = int(status_data[processed_key])
                
                # 兼容嵌套格式（galvo_0 对象）
                for galvo_idx in range(2):
                    galvo_key = f'galvo_{galvo_idx}'
                    if galvo_key in status_data and isinstance(status_data[galvo_key], dict):
                        galvo_data = status_data[galvo_key]
                        if 'detected_count' in galvo_data:
                            self.stats[f'galvo_{galvo_idx}_detected_count'] = int(galvo_data['detected_count'])
                        if 'processed_count' in galvo_data:
                            self.stats[f'galvo_{galvo_idx}_processed_count'] = int(galvo_data['processed_count'])
                
                # 兼容 galvo_states 数组格式
                if 'galvo_states' in status_data and isinstance(status_data['galvo_states'], list):
                    for galvo_info in status_data['galvo_states']:
                        if isinstance(galvo_info, dict):
                            galvo_idx = galvo_info.get('index', -1)
                            if 0 <= galvo_idx < 2:
                                if 'detected_count' in galvo_info:
                                    self.stats[f'galvo_{galvo_idx}_detected_count'] = int(galvo_info['detected_count'])
                                if 'processed_count' in galvo_info:
                                    self.stats[f'galvo_{galvo_idx}_processed_count'] = int(galvo_info['processed_count'])
                
                # 系统状态（使用 state 或 system_status）
                if 'state' in status_data:
                    self.stats['system_status'] = status_data['state']
                elif 'system_status' in status_data:
                    self.stats['system_status'] = status_data['system_status']
                
                # FPS（如果状态信息中包含）
                if 'fps' in status_data:
                    self.stats['fps'] = float(status_data['fps'])
            
            self.statistics_updated.emit(self.stats.copy())
        except json.JSONDecodeError as e:
            # 如果不是 JSON，直接作为字符串处理
            print(f"DEBUG: JSON 解析失败: {e}, 原始数据: {msg.data[:200]}")
            with QMutexLocker(self.mutex):
                self.stats['system_status'] = msg.data
            self.statistics_updated.emit(self.stats.copy())
        except Exception as e:
            print(f"DEBUG: 状态解析错误: {e}")
            import traceback
            traceback.print_exc()
            self.status_message.emit(f"状态解析错误: {str(e)}", "warning")
    
    def target_callback(self, msg):
        """当前目标回调"""
        if self.mutex is None:
            return
        try:
            target_data = json.loads(msg.data)
            with QMutexLocker(self.mutex):
                self.stats['current_target'] = target_data
            self.statistics_updated.emit(self.stats.copy())
        except:
            with QMutexLocker(self.mutex):
                self.stats['current_target'] = msg.data
            self.statistics_updated.emit(self.stats.copy())
    
    def publish_galvo_command(self, x, y, galvo_index=0):
        """发布振镜控制命令"""
        if self.galvo_pub is None:
            return False
        
        try:
            msg = Int32MultiArray()
            # 根据协议，可能需要包含振镜索引
            if galvo_index == 0:
                msg.data = [int(x), int(y)]
            else:
                msg.data = [galvo_index, int(x), int(y)]
            self.galvo_pub.publish(msg)
            return True
        except Exception as e:
            self.status_message.emit(f"发布振镜命令失败: {str(e)}", "error")
            return False
    
    def publish_laser_control(self, enable, galvo_index=0):
        """发布激光控制命令"""
        if self.laser_pub is None:
            return False
        
        try:
            # 目前系统主要使用 /laser_control (Bool) 话题
            # 如果未来需要支持多振镜独立激光控制，这里可以根据 galvo_index 发布到不同话题
            # 或者改用包含 index 的自定义消息
            msg = Bool()
            msg.data = enable
            self.laser_pub.publish(msg)
            
            # 记录日志以便调试
            # print(f"DEBUG: 发布激光控制: galvo_{galvo_index} = {enable}")
            
            return True
        except Exception as e:
            self.status_message.emit(f"发布激光控制失败: {str(e)}", "error")
            return False
    
    def run(self):
        """线程主循环"""
        # 在线程中创建所有可能依赖线程的对象，避免线程问题
        try:
            self.bridge = CvBridge()
            self.mutex = QMutex()
        except Exception as e:
            self.status_message.emit(f"创建对象失败: {str(e)}", "error")
            return
        
        if not self.initialize_ros():
            return
        
        self.setup_publishers()
        self.setup_subscribers()
        
        self.running = True
        rate = rospy.Rate(30)  # 30 Hz
        last_topic_check = 0
        topic_check_interval = 2.0  # 每2秒检查一次话题
        
        while self.running and not rospy.is_shutdown():
            try:
                # 定期检查话题并重新连接
                current_time = rospy.get_time()
                if current_time - last_topic_check > topic_check_interval:
                    self.check_and_reconnect_topics()
                    last_topic_check = current_time
                
                rate.sleep()
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                self.status_message.emit(f"ROS 循环错误: {str(e)}", "error")
                break
    
    def check_and_reconnect_topics(self):
        """检查话题是否存在，如果存在但未订阅则重新连接"""
        if not self.ros_initialized:
            return
        
        try:
            available_topics = [topic[0] for topic in rospy.get_published_topics()]
            
            # 检查并连接检测图像话题
            if '/det_img/image_raw' in available_topics and self.det_img_sub is None:
                self.det_img_sub = rospy.Subscriber('/det_img/image_raw', Image, self.det_image_callback, queue_size=1)
            
            # 检查并连接原始图像话题
            raw_topic = '/camera/color/image_raw'
            if raw_topic in available_topics and self.raw_img_sub is None:
                self.raw_img_sub = rospy.Subscriber(raw_topic, Image, self.raw_image_callback, queue_size=1)
            
            # 检查并连接标定图像话题
            if '/manual_calibration_image' in available_topics and self.calib_img_sub is None:
                self.calib_img_sub = rospy.Subscriber('/manual_calibration_image', Image, self.calib_image_callback, queue_size=1)
            
            # 检查并连接标定状态话题
            if '/manual_calibration_status' in available_topics and self.calib_status_sub is None:
                self.calib_status_sub = rospy.Subscriber('/manual_calibration_status', String, self.calib_status_callback, queue_size=1)
            
            # 检查并连接系统状态话题
            if '/system_status' in available_topics:
                if self.status_sub is None:
                    print("DEBUG: 检测到 /system_status 话题，正在连接...")
                    self.status_sub = rospy.Subscriber(
                        '/system_status',
                        String,
                        self.status_callback,
                        queue_size=5
                    )
                    self.status_message.emit("已连接到系统状态话题", "info")
            
            # 检查并连接当前目标话题
            if '/current_target' in available_topics:
                if self.target_sub is None:
                    print("DEBUG: 检测到 /current_target 话题，正在连接...")
                    self.target_sub = rospy.Subscriber(
                        '/current_target',
                        String,
                        self.target_callback,
                        queue_size=5
                    )
                    self.status_message.emit("已连接到当前目标话题", "info")
        except Exception as e:
            print(f"DEBUG: 检查话题时出错: {e}")
    
    def stop(self):
        """停止线程"""
        self.running = False
        if self.det_img_sub:
            self.det_img_sub.unregister()
        if self.status_sub:
            self.status_sub.unregister()
        if self.target_sub:
            self.target_sub.unregister()
        self.wait()


class LaunchManager:
    """Launch 文件管理类"""
    
    def __init__(self, launch_dir):
        self.launch_dir = launch_dir
        self.current_process = None
        self.launch_files = []
        self._scan_launch_files()
    
    def _scan_launch_files(self):
        """扫描 launch 目录下的所有 launch 文件"""
        print(f"LaunchManager: 扫描 launch 目录: {self.launch_dir}")
        print(f"LaunchManager: 目录绝对路径: {os.path.abspath(self.launch_dir)}")
        print(f"LaunchManager: 目录是否存在: {os.path.exists(self.launch_dir)}")
        
        if not os.path.exists(self.launch_dir):
            print(f"LaunchManager: 警告: launch 目录不存在: {self.launch_dir}")
            # 尝试使用绝对路径
            abs_path = os.path.abspath(self.launch_dir)
            if os.path.exists(abs_path):
                print(f"LaunchManager: 使用绝对路径: {abs_path}")
                self.launch_dir = abs_path
            else:
                print(f"LaunchManager: 绝对路径也不存在: {abs_path}")
                return
        
        try:
            files = os.listdir(self.launch_dir)
            print(f"LaunchManager: 目录中的文件: {files}")
        except Exception as e:
            print(f"LaunchManager: 无法列出目录内容: {e}")
            return
        
        for filename in files:
            if filename.endswith('.launch'):
                name = filename[:-7]  # 去掉 .launch 后缀
                
                # 只保留指定的 launch 文件
                if name not in ['main', 'main_offline']:
                    continue
                    
                launch_path = os.path.join(self.launch_dir, filename)
                launch_info = {
                    'name': name,
                    'path': launch_path,
                    'display_name': self._get_display_name(name)
                }
                self.launch_files.append(launch_info)
                print(f"LaunchManager: 找到 launch 文件: {launch_info}")
        
        print(f"LaunchManager: 总共找到 {len(self.launch_files)} 个 launch 文件")
    
    def _get_display_name(self, name):
        """获取显示名称"""
        name_map = {
            'main': '主系统 (在线)',
            'main_offline': '主系统 (离线测试)',
            'calibration': '标定系统',
            'track_one_object': '单目标跟踪',
            'track_one_object_kalman': '单目标跟踪 (卡尔曼)'
        }
        return name_map.get(name, name)
    
    def get_launch_files(self):
        """获取所有 launch 文件列表"""
        return self.launch_files
    
    def start_launch(self, launch_name):
        """启动指定的 launch 文件"""
        print(f"LaunchManager: 收到启动请求: {launch_name}")
        if self.current_process is not None and self.current_process.poll() is None:
            print("LaunchManager: 已有进程在运行")
            return False, "已有 launch 进程在运行"
        
        launch_path = None
        for launch in self.launch_files:
            if launch['name'] == launch_name:
                launch_path = launch['path']
                break
        
        if launch_path is None:
            print(f"LaunchManager: 未找到 launch 文件: {launch_name}")
            print(f"LaunchManager: 可用的 launch 文件: {[l['name'] for l in self.launch_files]}")
            return False, f"未找到 launch 文件: {launch_name}"
        
        print(f"LaunchManager: 找到 launch 文件: {launch_path}")
        
        try:
            # 使用 roslaunch 启动
            # 直接使用绝对路径
            launch_path_abs = os.path.abspath(launch_path)
            cmd = ['roslaunch', launch_path_abs]
            
            print(f"LaunchManager: 执行命令: {' '.join(cmd)}")
            print(f"LaunchManager: Launch 文件绝对路径: {launch_path_abs}")
            
            # 启动进程，捕获输出
            # 注意：使用 bash -c 时，不需要传递 env，因为 bash 会继承当前环境
            # 并且 setup.bash 会设置正确的环境变量
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False,  # 使用字节模式
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            print(f"LaunchManager: 进程已启动，PID: {self.current_process.pid}")
            
            return True, f"启动成功: {launch_name}"
        except FileNotFoundError as e:
            error_msg = f"启动失败: 找不到 roslaunch 命令。请确保 ROS 环境已正确配置。"
            print(f"LaunchManager: {error_msg}")
            print(f"LaunchManager: 异常详情: {e}")
            return False, error_msg
        except Exception as e:
            error_msg = f"启动失败: {str(e)}"
            print(f"LaunchManager: {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg
    
    def start_launch_by_name(self, launch_name, process_storage=None):
        """
        启动指定名称的 launch 文件（即使不在管理列表中）
        
        Args:
            launch_name: launch 文件名（不含 .launch 后缀）
            process_storage: 可选，用于存储进程对象的字典（如 {'calibration': process}）
        
        Returns:
            (success: bool, message: str, process: subprocess.Popen or None)
        """
        print(f"LaunchManager: 启动 launch 文件: {launch_name}")
        
        # 构建 launch 文件路径
        launch_path = os.path.join(self.launch_dir, f"{launch_name}.launch")
        launch_path_abs = os.path.abspath(launch_path)
        
        if not os.path.exists(launch_path_abs):
            error_msg = f"Launch 文件不存在: {launch_path_abs}"
            print(f"LaunchManager: {error_msg}")
            return False, error_msg, None
        
        # 检查是否已经在运行（如果提供了 process_storage）
        if process_storage and launch_name in process_storage:
            existing_process = process_storage[launch_name]
            if existing_process is not None and existing_process.poll() is None:
                print(f"LaunchManager: {launch_name} 已在运行中")
                return True, f"{launch_name} 已在运行", existing_process
        
        try:
            cmd = ['roslaunch', launch_path_abs]
            print(f"LaunchManager: 执行命令: {' '.join(cmd)}")
            print(f"LaunchManager: Launch 文件绝对路径: {launch_path_abs}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            print(f"LaunchManager: {launch_name} 进程已启动，PID: {process.pid}")
            
            # 如果提供了 process_storage，保存进程对象
            if process_storage is not None:
                process_storage[launch_name] = process
            
            return True, f"启动成功: {launch_name}", process
        except FileNotFoundError as e:
            error_msg = f"启动失败: 找不到 roslaunch 命令。请确保 ROS 环境已正确配置。"
            print(f"LaunchManager: {error_msg}")
            print(f"LaunchManager: 异常详情: {e}")
            return False, error_msg, None
        except Exception as e:
            error_msg = f"启动失败: {str(e)}"
            print(f"LaunchManager: {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg, None
    
    def stop_launch_by_name(self, launch_name, process_storage):
        """
        停止指定名称的 launch 进程
        
        Args:
            launch_name: launch 文件名（不含 .launch 后缀）
            process_storage: 存储进程对象的字典
        
        Returns:
            (success: bool, message: str)
        """
        if launch_name not in process_storage or process_storage[launch_name] is None:
            return False, f"没有运行中的 {launch_name} 进程"
        
        process = process_storage[launch_name]
        if process.poll() is not None:
            process_storage[launch_name] = None
            return True, f"{launch_name} 进程已结束"
        
        try:
            # 终止整个进程组
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                process.terminate()
            
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 强制杀死
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    process.kill()
                process.wait()
            
            process_storage[launch_name] = None
            return True, f"已停止 {launch_name} 进程"
        except Exception as e:
            return False, f"停止失败: {str(e)}"
    
    def stop_launch(self):
        """停止当前运行的 launch 进程"""
        if self.current_process is None:
            return False, "没有运行中的 launch 进程"
        
        # 停止输出读取线程
        
        if self.current_process.poll() is not None:
            # 进程已经结束
            self.current_process = None
            return True, "进程已结束"
        
        try:
            # 终止整个进程组
            try:
                pgid = os.getpgid(self.current_process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                # 如果无法获取进程组，直接终止进程
                self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 强制杀死
                try:
                    pgid = os.getpgid(self.current_process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    self.current_process.kill()
                self.current_process.wait()
            self.current_process = None
            return True, "已停止 launch 进程"
        except Exception as e:
            return False, f"停止失败: {str(e)}"
    
    def is_running(self):
        """检查是否有 launch 进程在运行"""
        if self.current_process is None:
            return False
        return self.current_process.poll() is None


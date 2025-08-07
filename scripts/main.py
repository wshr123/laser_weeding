#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import serial
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Float32, String
from detector import WeedDetector, WeedTracker
import json
import time
from collections import defaultdict


class LaserWeedingNode:
    def __init__(self):
        rospy.init_node('laser_weeding_node', anonymous=True)

        # Parameters
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.xy_pub = rospy.Publisher('/laser_xy', Int32MultiArray, queue_size=10)

        # Serial setup for Teensy
        self.serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        self.serial_port.flush()

        # 激光控制相关参数
        self.processed_weeds = set()  # 已处理的杂草ID集合
        self.current_target_id = None  # 当前正在处理的杂草ID
        self.laser_active = False  # 激光是否激活
        self.laser_duration = 0.1  # 激光照射持续时间（秒）
        self.laser_start_time = None

        # 杂草处理队列和状态
        self.weed_queue = []  # 待处理杂草队列 [(id, priority, first_seen_time), ...]
        self.weed_processing_log = {}  # 杂草处理日志

        # Detection model initialization
        self.yolo5_weed_id = 0  # 假设杂草ID为1
        self.yolo5_model_path = "your_model_path.pt"  # 替换为实际路径

        try:
            self.detector = WeedDetector(
                model_path=self.yolo5_model_path,
                yolo5_weed_id=self.yolo5_weed_id
            )
            rospy.loginfo('YOLO5 plant detection model loaded successfully')
        except Exception as e:
            rospy.logerr(f"Failed to load detection model: {e}")
            return

        # Calibration parameters
        self.image_width = 640
        self.image_height = 480
        self.xy_scale = 65535.0 / self.image_width
        self.left_img = None

        # Camera subscription
        self.left_img_sub = rospy.Subscriber('/camera/image_raw', Image, self.left_img_cb, queue_size=1)

        # 定时器用于激光控制
        self.laser_control_timer = rospy.Timer(rospy.Duration(0.1), self.laser_control_callback)

        rospy.loginfo("Laser Weeding Node Initialized")

    def left_img_cb(self, msg):
        try:
            self.left_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

    def image_callback(self, data):
        if self.left_img is None:
            return

        try:
            cv_image = self.left_img

            # 检测和跟踪杂草
            result_image, detection_results = self.detector.detect_and_track_weeds(cv_image)
            track_info = self.detector.get_track_info()

            # 更新杂草队列
            self.update_weed_queue(track_info)

            # 如果当前没有激光任务，选择下一个目标
            if not self.laser_active and not self.current_target_id:
                self.select_next_target()

        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")

    def update_weed_queue(self, track_info):
        """更新杂草处理队列"""
        current_time = time.time()
        current_weed_ids = set(track_info.keys())

        # 添加新检测到的杂草到队列
        for weed_id, info in track_info.items():
            if (weed_id not in self.processed_weeds and
                    weed_id not in [item[0] for item in self.weed_queue] and
                    self.is_reliable_weed(info)):
                # 计算优先级（基于大小、置信度等）
                priority = self.calculate_weed_priority(info)
                self.weed_queue.append((weed_id, priority, current_time))

                rospy.loginfo(f"Added weed {weed_id} to processing queue with priority {priority:.2f}")

        # 按优先级排序队列（ID小的优先，然后按优先级）
        self.weed_queue.sort(key=lambda x: (x[0], -x[1]))

        # 移除已经消失太久的杂草
        self.cleanup_disappeared_weeds(current_weed_ids, current_time)

    def is_reliable_weed(self, weed_info):
        """判断杂草是否可靠（可以开始处理）"""
        return (weed_info.get('consecutive_hits', 0) >= 3 and
                weed_info.get('confidence', 0) > 0.5 and
                weed_info.get('frames_skipped', 0) <= 2)

    def calculate_weed_priority(self, weed_info):
        """计算杂草处理优先级"""
        confidence = weed_info.get('confidence', 0)
        bbox = weed_info.get('bbox', [0, 0, 10, 10])
        area = bbox[2] * bbox[3]  # 宽度 * 高度

        # 优先级 = 置信度 * 0.6 + 标准化面积 * 0.4
        normalized_area = min(area / 10000.0, 1.0)  # 假设最大面积为10000
        priority = confidence * 0.6 + normalized_area * 0.4

        return priority

    def cleanup_disappeared_weeds(self, current_weed_ids, current_time):
        """清理已消失的杂草"""
        # 从队列中移除已消失超过5秒的杂草
        self.weed_queue = [
            (weed_id, priority, first_seen)
            for weed_id, priority, first_seen in self.weed_queue
            if weed_id in current_weed_ids or (current_time - first_seen) < 5.0
        ]

    def select_next_target(self):
        """选择下一个要处理的杂草目标"""
        if not self.weed_queue:
            return

        # 获取队列中的第一个杂草（优先级最高）
        target_id, priority, first_seen = self.weed_queue.pop(0)

        # 检查该杂草是否仍然存在
        track_info = self.detector.get_track_info()
        if target_id in track_info and self.is_reliable_weed(track_info[target_id]):
            self.current_target_id = target_id
            rospy.loginfo(f"Selected weed {target_id} as next target")
        else:
            rospy.logwarn(f"Target weed {target_id} no longer reliable, selecting next")
            self.select_next_target()  # 递归选择下一个

    def laser_control_callback(self, event):
        """激光控制回调函数"""
        if self.current_target_id is None:
            return

        track_info = self.detector.get_track_info()

        # 检查目标杂草是否仍然存在且可靠
        if (self.current_target_id not in track_info or
                not self.is_reliable_weed(track_info[self.current_target_id])):
            rospy.logwarn(f"Target weed {self.current_target_id} lost, aborting laser")
            self.abort_current_laser_task()
            return

        target_info = track_info[self.current_target_id]

        if not self.laser_active:
            # 开始激光照射
            self.start_laser_treatment(target_info)
        else:
            # 检查激光照射是否完成
            if time.time() - self.laser_start_time >= self.laser_duration:
                self.complete_laser_treatment()

    def start_laser_treatment(self, target_info):
        """开始激光处理"""
        bbox = target_info['bbox']
        centroid = [(bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)]

        # 计算激光坐标
        x, y = self.calculate_xy(centroid)

        # 发送激光控制命令
        self.send_to_teensy(x, y)

        # 更新状态
        self.laser_active = True
        self.laser_start_time = time.time()

        # 发布ROS消息用于监控
        xy_msg = Int32MultiArray()
        xy_msg.data = [x, y, 1]  # 第三个参数表示激光开启
        self.xy_pub.publish(xy_msg)

        rospy.loginfo(f"Started laser treatment for weed {self.current_target_id} at ({x}, {y})")

    def complete_laser_treatment(self):
        """完成激光处理"""
        # 记录处理完成的杂草
        self.processed_weeds.add(self.current_target_id)

        # 记录处理日志
        self.weed_processing_log[self.current_target_id] = {
            'completion_time': time.time(),
            'laser_duration': self.laser_duration
        }

        # 发送停止激光命令
        self.send_to_teensy(32768, 32768)  # 中心位置，停止激光

        # 发布停止消息
        xy_msg = Int32MultiArray()
        xy_msg.data = [32768, 32768, 0]  # 第三个参数表示激光关闭
        self.xy_pub.publish(xy_msg)

        rospy.loginfo(f"Completed laser treatment for weed {self.current_target_id}")

        # 重置状态
        self.laser_active = False
        self.current_target_id = None
        self.laser_start_time = None

        # 选择下一个目标
        self.select_next_target()

    def abort_current_laser_task(self):
        """中止当前激光任务"""
        if self.laser_active:
            self.send_to_teensy(32768, 32768)  # 停止激光

        self.laser_active = False
        self.current_target_id = None
        self.laser_start_time = None

        # 立即选择下一个目标
        self.select_next_target()

    def calculate_xy(self, centroid):
        """将像素坐标转换为激光控制坐标"""
        px, py = centroid
        x = int((px / self.image_width) * 65535)
        y = int((py / self.image_height) * 65535)

        # 确保坐标在有效范围内
        x = max(0, min(65535, x))
        y = max(0, min(65535, y))

        return x, y

    def send_to_teensy(self, x, y):
        """发送坐标到Teensy"""
        try:
            command = f"XY:{x},{y}\n"
            self.serial_port.write(command.encode())
            rospy.logdebug(f"Sent to Teensy: {command.strip()}")
        except Exception as e:
            rospy.logerr(f"Failed to send to Teensy: {e}")

    def get_processing_status(self):
        """获取处理状态信息"""
        return {
            'processed_weeds': list(self.processed_weeds),
            'queue_length': len(self.weed_queue),
            'current_target': self.current_target_id,
            'laser_active': self.laser_active,
            'total_processed': len(self.processed_weeds)
        }


if __name__ == '__main__':
    try:
        node = LaserWeedingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Node error: {e}")
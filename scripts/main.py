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
from detector import WeedDetector, WeedTracker
import json
import time
from collections import defaultdict

# 导入振镜控制器
from send_to_teensy import XY2_100Controller


class LaserWeedingNode:
    def __init__(self):
        try:
            rospy.init_node('laser_weeding_node', anonymous=True)
            rospy.loginfo("=" * 50)
            rospy.loginfo("Starting Laser Weeding Node...")
            rospy.loginfo("=" * 50)

            # Parameters
            self.bridge = CvBridge()

            # 使用XY2_100Controller替代原有的串口控制
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

            # 激光控制相关参数
            self.processed_weeds = set()
            self.current_target_id = None
            self.laser_active = False
            self.laser_duration = rospy.get_param('~laser_duration', 0.1)
            self.laser_start_time = None

            # 杂草处理队列和状态
            self.weed_queue = []
            self.weed_processing_log = {}

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
                rospy.loginfo(
                    f'{self.model_type.upper()} detection model with {self.tracker_type} tracker loaded successfully')
            except Exception as e:
                rospy.logerr(f"Failed to load detection model: {e}")
                rospy.logerr(traceback.format_exc())
                sys.exit(1)

            # Calibration parameters
            self.image_width = rospy.get_param('~image_width', 640)
            self.image_height = rospy.get_param('~image_height', 480)

            # 初始化图像变量
            self.left_img = None

            # Publishers
            self.xy_pub = rospy.Publisher('/laser_xy', Int32MultiArray, queue_size=10)
            self.det_img_pub = rospy.Publisher('/det_img/image_raw', Image, queue_size=2)

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

            # 定时器用于激光控制
            self.laser_control_timer = rospy.Timer(
                rospy.Duration(0.05),  # 减少到50ms，提高响应速度
                self.laser_control_callback
            )

            # 心跳日志
            self.heartbeat_timer = rospy.Timer(
                rospy.Duration(5.0),
                self.heartbeat_callback
            )

            rospy.loginfo("=" * 50)
            rospy.loginfo("Laser Weeding Node Initialized Successfully!")
            rospy.loginfo("Waiting for images...")
            rospy.loginfo("=" * 50)

        except Exception as e:
            rospy.logerr(f"Failed to initialize LaserWeedingNode: {e}")
            rospy.logerr(traceback.format_exc())
            sys.exit(1)

    def heartbeat_callback(self, event):
        """心跳回调，用于确认节点仍在运行"""
        status = self.get_processing_status()
        rospy.loginfo(f"Node alive - Processed: {status['total_processed']}, "
                      f"Queue: {status['queue_length']}, "
                      f"Active: {status['laser_active']}")

    def left_img_cb(self, msg):
        """图像回调"""
        try:
            self.left_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if not hasattr(self, '_first_image_received'):
                rospy.loginfo(f"First image received! Shape: {self.left_img.shape}")
                self._first_image_received = True
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

    def image_callback(self, data):
        """主图像处理回调"""
        if self.left_img is None:
            return

        try:
            cv_image = self.left_img

            # 检测和跟踪杂草
            result_image, detection_results = self.detector.detect_and_track_weeds(cv_image)
            track_info = self.detector.get_track_info()

            # 发布检测结果图像
            try:
                ros_det_image = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                self.det_img_pub.publish(ros_det_image)
            except CvBridgeError as e:
                rospy.logerr(f"Failed to convert detection image: {e}")
                return

            # 更新杂草队列
            self.update_weed_queue(track_info)

            # 如果当前没有激光任务，选择下一个目标
            if not self.laser_active and not self.current_target_id:
                self.select_next_target()

        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")
            rospy.logerr(traceback.format_exc())

    def update_weed_queue(self, track_info):
        """更新杂草处理队列"""
        current_time = time.time()
        current_weed_ids = set(track_info.keys())

        for weed_id, info in track_info.items():
            if (weed_id not in self.processed_weeds and
                    weed_id not in [item[0] for item in self.weed_queue] and
                    self.is_reliable_weed(info)):
                priority = self.calculate_weed_priority(info)
                self.weed_queue.append((weed_id, priority, current_time))
                rospy.logdebug(f"Added weed {weed_id} to queue with priority {priority:.2f}")

        self.weed_queue.sort(key=lambda x: (x[0], -x[1]))
        self.cleanup_disappeared_weeds(current_weed_ids, current_time)

    def is_reliable_weed(self, weed_info):
        """判断杂草是否可靠"""
        return (weed_info.get('consecutive_hits', 0) >= 3 and
                weed_info.get('confidence', 0) > 0.5 and
                weed_info.get('frames_skipped', 0) <= 2)

    def calculate_weed_priority(self, weed_info):
        """计算杂草处理优先级"""
        confidence = weed_info.get('confidence', 0)
        bbox = weed_info.get('bbox', [0, 0, 10, 10])
        area = bbox[2] * bbox[3]
        normalized_area = min(area / 10000.0, 1.0)
        priority = confidence * 0.6 + normalized_area * 0.4
        return priority

    def cleanup_disappeared_weeds(self, current_weed_ids, current_time):
        """清理已消失的杂草"""
        self.weed_queue = [
            (weed_id, priority, first_seen)
            for weed_id, priority, first_seen in self.weed_queue
            if weed_id in current_weed_ids or (current_time - first_seen) < 5.0
        ]

    def select_next_target(self):
        """选择下一个要处理的杂草目标"""
        if not self.weed_queue:
            return

        target_id, priority, first_seen = self.weed_queue.pop(0)
        track_info = self.detector.get_track_info()

        if target_id in track_info and self.is_reliable_weed(track_info[target_id]):
            self.current_target_id = target_id
            rospy.loginfo(f"Selected weed {target_id} as next target")
        else:
            rospy.logdebug(f"Target weed {target_id} no longer reliable")
            self.select_next_target()

    def laser_control_callback(self, event):
        """激光控制回调函数"""
        if self.current_target_id is None:
            return

        track_info = self.detector.get_track_info()

        if (self.current_target_id not in track_info or
                not self.is_reliable_weed(track_info[self.current_target_id])):
            rospy.logwarn(f"Target weed {self.current_target_id} lost")
            self.abort_current_laser_task()
            return

        target_info = track_info[self.current_target_id]

        if not self.laser_active:
            self.start_laser_treatment(target_info)
        else:
            # 检查是否需要更新振镜位置（跟踪移动的目标）
            self.update_laser_position(target_info)

            # 检查激光持续时间
            if time.time() - self.laser_start_time >= self.laser_duration:
                self.complete_laser_treatment()

    def start_laser_treatment(self, target_info):
        """开始激光处理"""
        bbox = target_info['bbox']
        centroid = [(bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)]
        galvo_x, galvo_y = self.pixel_to_galvo(centroid[0], centroid[1])

        # 使用galvo_controller移动到位置
        if self.galvo_controller:
            self.galvo_controller.move_to_position(galvo_x, galvo_y)
            # 等待振镜稳定
            rospy.sleep(0.05)
            # 开启激光
            self.galvo_controller.laser_on()

        self.laser_active = True
        self.laser_start_time = time.time()

        # 发布消息（为了与原有系统兼容）
        xy_msg = Int32MultiArray()
        xy_msg.data = [galvo_x, galvo_y, 1]
        self.xy_pub.publish(xy_msg)

        rospy.loginfo(f"Started laser treatment for weed {self.current_target_id} at ({galvo_x}, {galvo_y})")

    def update_laser_position(self, target_info):
        """更新激光位置以跟踪移动的目标"""
        if not self.laser_active or not self.galvo_controller:
            return

        bbox = target_info['bbox']
        centroid = [(bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)]
        galvo_x, galvo_y = self.pixel_to_galvo(centroid[0], centroid[1])

        # 更新振镜位置但保持激光开启
        self.galvo_controller.move_to_position(galvo_x, galvo_y)

        # 发布更新的位置
        xy_msg = Int32MultiArray()
        xy_msg.data = [galvo_x, galvo_y, 1]
        self.xy_pub.publish(xy_msg)

    def complete_laser_treatment(self):
        """完成激光处理"""
        self.processed_weeds.add(self.current_target_id)

        self.weed_processing_log[self.current_target_id] = {
            'completion_time': time.time(),
            'laser_duration': self.laser_duration
        }

        # 关闭激光并回到中心
        if self.galvo_controller:
            self.galvo_controller.laser_off()
            self.galvo_controller.move_to_center()

        xy_msg = Int32MultiArray()
        xy_msg.data = [32768, 32768, 0]
        self.xy_pub.publish(xy_msg)

        rospy.loginfo(f"Completed laser treatment for weed {self.current_target_id}")

        self.laser_active = False
        self.current_target_id = None
        self.laser_start_time = None

        self.select_next_target()

    def abort_current_laser_task(self):
        """中止当前激光任务"""
        if self.laser_active and self.galvo_controller:
            self.galvo_controller.laser_off()
            self.galvo_controller.move_to_center()

        self.laser_active = False
        self.current_target_id = None
        self.laser_start_time = None

        self.select_next_target()

    def pixel_to_galvo(self, pixel_x, pixel_y):
        """将像素坐标转换为振镜坐标"""
        if self.galvo_controller:
            return self.galvo_controller.pixel_to_galvo(
                pixel_x, pixel_y,
                self.image_width, self.image_height
            )
        else:
            # 回退到原有方法
            x = int((pixel_x / self.image_width) * 65535)
            y = int((pixel_y / self.image_height) * 65535)
            x = max(0, min(65535, x))
            y = max(0, min(65535, y))
            return x, y

    def get_processing_status(self):
        """获取处理状态信息"""
        return {
            'processed_weeds': list(self.processed_weeds),
            'queue_length': len(self.weed_queue),
            'current_target': self.current_target_id,
            'laser_active': self.laser_active,
            'total_processed': len(self.processed_weeds)
        }


def main():
    """主函数"""
    try:
        rospy.loginfo("Starting main function...")
        node = LaserWeedingNode()
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
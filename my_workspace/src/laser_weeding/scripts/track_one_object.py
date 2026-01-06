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
from detector import WeedDetector
import json
import time
from collections import defaultdict
"""
    使用三种跟踪方法对单个目标进行跟踪，测试跟踪效果
"""
# 导入振镜控制器
from one_object_to_teensy import XY2_100Controller


class GalvoTrackingNode:
    def __init__(self):
        try:
            rospy.init_node('galvo_tracking_node', anonymous=True)
            rospy.loginfo("=" * 50)
            rospy.loginfo("Starting Galvo Tracking Node...")
            rospy.loginfo("=" * 50)

            # Parameters
            self.bridge = CvBridge()

            # 振镜控制器初始化
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

            # 目标跟踪相关参数
            self.current_target = None  # 当前跟踪的最高置信度目标
            self.min_confidence = rospy.get_param('~min_confidence', 0.6)  # 最小置信度阈值
            self.target_stable_frames = rospy.get_param('~target_stable_frames', 3)  # 目标稳定帧数

            # 目标稳定性跟踪
            self.target_history = []  # 存储最近几帧的最高置信度目标
            self.stable_target_count = 0  # 当前目标连续出现的帧数

            # 振镜跟踪控制
            self.tracking_active = False  # 是否正在跟踪
            self.last_galvo_position = [0, 0]  # 上次振镜位置
            self.position_smoothing = rospy.get_param('~position_smoothing', 0.3)  # 位置平滑系数

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
                    f'{self.model_type.upper()} detection model loaded successfully')
            except Exception as e:
                rospy.logerr(f"Failed to load detection model: {e}")
                rospy.logerr(traceback.format_exc())
                sys.exit(1)

            # Calibration parameters
            self.image_width = rospy.get_param('~image_width', 640)
            self.image_height = rospy.get_param('~image_height', 480)

            # 初始化图像变量
            self.left_img = None
            self.frame_count = 0

            # Publishers
            self.xy_pub = rospy.Publisher('/galvo_xy', Int32MultiArray, queue_size=10)
            self.det_img_pub = rospy.Publisher('/det_img/image_raw', Image, queue_size=2)
            self.target_pub = rospy.Publisher('/current_target', String, queue_size=5)

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

            # 定时器用于振镜控制
            self.galvo_control_timer = rospy.Timer(
                rospy.Duration(0.01),  # 20Hz 快速响应
                self.galvo_control_callback
            )

            # 心跳日志
            self.heartbeat_timer = rospy.Timer(
                rospy.Duration(1.0),
                self.heartbeat_callback
            )

            rospy.loginfo("=" * 50)
            rospy.loginfo("Galvo Tracking Node Initialized!")
            rospy.loginfo("Waiting for images...")
            rospy.loginfo("=" * 50)

        except Exception as e:
            rospy.logerr(f"Failed to initialize GalvoTrackingNode: {e}")
            rospy.logerr(traceback.format_exc())
            sys.exit(1)

    def heartbeat_callback(self, event):
        """心跳回调"""
        status_msg = f"Frame: {self.frame_count}, "
        if self.current_target:
            status_msg += f"Tracking: conf={self.current_target['confidence']:.3f}, "
            status_msg += f"stable={self.stable_target_count}/{self.target_stable_frames}, "
            status_msg += f"galvo_pos=({self.last_galvo_position[0]}, {self.last_galvo_position[1]}), "
        status_msg += f"Active: {'YES' if self.tracking_active else 'NO'}"

        rospy.loginfo(status_msg)

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
            self.frame_count += 1
            cv_image = self.left_img

            # 检测杂草
            result_image, detection_results = self.detector.detect_and_track_weeds(cv_image)

            # 从检测结果中找到置信度最高的杂草
            highest_confidence_target = self.find_highest_confidence_weed(detection_results)

            # 更新当前目标
            self.update_current_target(highest_confidence_target)

            # 绘制当前目标信息
            result_image = self.draw_current_target_info(result_image)

            # 发布检测结果图像
            try:
                ros_det_image = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                self.det_img_pub.publish(ros_det_image)
            except CvBridgeError as e:
                rospy.logerr(f"Failed to convert detection image: {e}")
                return

            # 发布当前目标信息
            self.publish_target_info()

        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")
            rospy.logerr(traceback.format_exc())

    def find_highest_confidence_weed(self, detection_results):
        """从检测结果中找到置信度最高的杂草"""
        if not detection_results:
            return None

        highest_conf_target = None
        max_confidence = 0

        for track_id, bbox, confidence in detection_results:
            if confidence > max_confidence and confidence >= self.min_confidence:
                # 计算目标中心点
                x, y, w, h = bbox
                centroid = [x + w / 2, y + h / 2]

                highest_conf_target = {
                    'track_id': track_id,
                    'bbox': bbox,
                    'centroid': centroid,
                    'confidence': confidence
                }
                max_confidence = confidence

        return highest_conf_target

    def update_current_target(self, new_target):
        """更新当前跟踪目标"""
        if new_target is None:
            # 没有检测到合适的目标
            if self.current_target is not None:
                self.stable_target_count = 0
                rospy.logdebug("Lost target")
            self.current_target = None
            return

        if self.current_target is None:
            # 发现新目标
            self.current_target = new_target
            self.stable_target_count = 1
            rospy.loginfo(f"New target detected: track_id={new_target['track_id']}, "
                          f"conf={new_target['confidence']:.3f}")
        else:
            # 检查是否是同一个目标
            if self.is_same_target(self.current_target, new_target):
                # 更新目标信息
                self.current_target = new_target
                self.stable_target_count += 1
                rospy.logdebug(f"Target updated: stable_count={self.stable_target_count}")
            else:
                # 切换到新的更高置信度目标
                if new_target['confidence'] > self.current_target['confidence'] + 0.1:
                    rospy.loginfo(f"Switching to higher confidence target: "
                                  f"old_conf={self.current_target['confidence']:.3f}, "
                                  f"new_conf={new_target['confidence']:.3f}")
                    self.current_target = new_target
                    self.stable_target_count = 1
                else:
                    # 保持当前目标
                    self.stable_target_count = max(0, self.stable_target_count - 1)

    def is_same_target(self, target1, target2):
        """判断是否是同一个目标"""
        # 如果有track_id，优先使用track_id判断
        if target1['track_id'] == target2['track_id']:
            return True

        # 否则使用位置距离判断
        dist = np.sqrt((target1['centroid'][0] - target2['centroid'][0]) ** 2 +
                       (target1['centroid'][1] - target2['centroid'][1]) ** 2)
        return dist < 30  # 像素距离阈值

    def galvo_control_callback(self, event):
        """振镜控制回调函数"""
        # 检查是否有稳定的目标
        if (self.current_target is None or
                self.stable_target_count < self.target_stable_frames):
            # 如果正在跟踪但目标不稳定，停止跟踪
            if self.tracking_active:
                self.stop_tracking()
            return

        # 如果未开始跟踪，开始跟踪
        if not self.tracking_active:
            self.start_tracking()
        else:
            # 正在跟踪，更新振镜位置
            self.update_galvo_position()

    def start_tracking(self):
        """开始振镜跟踪"""
        if self.current_target is None:
            return

        centroid = self.current_target['centroid']
        galvo_x, galvo_y = self.pixel_to_galvo(centroid[0], centroid[1])

        # 使用galvo_controller移动到位置
        if self.galvo_controller:
            self.galvo_controller.move_to_position(galvo_x, galvo_y)

        self.tracking_active = True
        self.last_galvo_position = [galvo_x, galvo_y]

        # 发布消息
        xy_msg = Int32MultiArray()
        xy_msg.data = [galvo_x, galvo_y, 1]  # 第三个参数表示跟踪状态
        self.xy_pub.publish(xy_msg)

        rospy.loginfo(f"Started tracking target track_id={self.current_target['track_id']}, "
                      f"conf={self.current_target['confidence']:.3f}, "
                      f"galvo_pos=({galvo_x}, {galvo_y})")

    def update_galvo_position(self):
        """更新振镜位置以跟踪目标"""
        if not self.tracking_active or not self.current_target:
            return

        centroid = self.current_target['centroid']
        target_galvo_x, target_galvo_y = self.pixel_to_galvo(centroid[0], centroid[1])

        # 位置平滑处理
        smooth_x = (self.last_galvo_position[0] * (1 - self.position_smoothing) +
                    target_galvo_x * self.position_smoothing)
        smooth_y = (self.last_galvo_position[1] * (1 - self.position_smoothing) +
                    target_galvo_y * self.position_smoothing)

        galvo_x = int(smooth_x)
        galvo_y = int(smooth_y)

        # 检查位置变化是否足够大，避免过度抖动
        position_change = np.sqrt((galvo_x - self.last_galvo_position[0]) ** 2 +
                                  (galvo_y - self.last_galvo_position[1]) ** 2)

        if position_change > 10:  # 最小移动阈值
            # 更新振镜位置
            if self.galvo_controller:
                self.galvo_controller.move_to_position(galvo_x, galvo_y)

            self.last_galvo_position = [galvo_x, galvo_y]

            # 发布更新的位置
            xy_msg = Int32MultiArray()
            xy_msg.data = [galvo_x, galvo_y, 1]
            self.xy_pub.publish(xy_msg)

            rospy.logdebug(f"Updated galvo position: ({galvo_x}, {galvo_y}), "
                           f"change: {position_change:.1f}")

    def stop_tracking(self):
        """停止振镜跟踪"""
        # 可选择是否回到中心位置
        # if self.galvo_controller:
        #     self.galvo_controller.move_to_center()

        self.tracking_active = False

        # 发布停止状态
        xy_msg = Int32MultiArray()
        xy_msg.data = [self.last_galvo_position[0], self.last_galvo_position[1], 0]  # 第三个参数0表示停止跟踪
        self.xy_pub.publish(xy_msg)

        rospy.loginfo("Stopped galvo tracking")

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

    def draw_current_target_info(self, image):
        """在图像上绘制当前目标信息"""
        result_image = image.copy()

        # 绘制当前目标
        if self.current_target:
            bbox = self.current_target['bbox']
            centroid = self.current_target['centroid']
            confidence = self.current_target['confidence']
            track_id = self.current_target['track_id']

            x, y, w, h = bbox

            # 根据跟踪状态选择颜色
            if self.tracking_active:
                color = (0, 255, 255)  # 红色 - 正在跟踪
                thickness = 2
                status_text = "TRACKING"
            elif self.stable_target_count >= self.target_stable_frames:
                color = (0, 255, 0)  # 绿色 - 稳定目标
                thickness = 2
                status_text = "STABLE"
            else:
                color = (0, 255, 255)  # 黄色 - 候选目标
                thickness = 2
                status_text = "CANDIDATE"

            # 绘制边界框
            cv2.rectangle(result_image, (int(x), int(y)), (int(x + w), int(y + h)),
                          color, thickness)

            # 绘制中心点
            cv2.circle(result_image, (int(centroid[0]), int(centroid[1])), 5, color, -1)

            # # 绘制十字线
            # cv2.line(result_image,
            #          (int(centroid[0] - 15), int(centroid[1])),
            #          (int(centroid[0] + 15), int(centroid[1])),
            #          color, 2)
            # cv2.line(result_image,
            #          (int(centroid[0]), int(centroid[1] - 15)),
            #          (int(centroid[0]), int(centroid[1] + 15)),
            #          color, 2)

            # 绘制振镜位置指示
            if self.tracking_active:
                galvo_x, galvo_y = self.last_galvo_position
                # 将振镜坐标转换回像素坐标显示
                display_x = (galvo_x + 30000) / 60000 * self.image_width
                display_y = (galvo_y + 30000) / 60000 * self.image_height
                cv2.circle(result_image, (int(display_x), int(display_y)), 8, (255, 0, 255), 2)
                cv2.putText(result_image, "GALVO", (int(display_x - 20), int(display_y - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            # 绘制标签
            label_parts = [f'T_{track_id}']
            label_parts.append(f'C:{confidence:.3f}')
            label_parts.append(f'S:{self.stable_target_count}/{self.target_stable_frames}')
            label_parts.append(f'[{status_text}]')

            label = ' '.join(label_parts)

            # 计算标签位置
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = int(y - 10) if y > 30 else int(y + h + 25)

            # 绘制标签背景
            # cv2.rectangle(result_image,
            #               (int(x), label_y - label_size[1] - 5),
            #               (int(x + label_size[0] + 10), label_y + 5),
            #               color, -1)
            #
            # # 绘制标签文字
            # text_color = (255, 255, 255) if color != (0, 255, 255) else (0, 0, 0)
            # cv2.putText(result_image, label, (int(x + 5), label_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # 绘制统计信息
        # self.draw_statistics(result_image)

        return result_image

    def draw_statistics(self, image):
        """在图像上绘制统计信息"""
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Model: {self.model_type.upper()}",
            f"Min Conf: {self.min_confidence}",
            f"Tracking: {'ACTIVE' if self.tracking_active else 'INACTIVE'}",
        ]

        if self.current_target:
            info_lines.append(f"Target ID: {self.current_target['track_id']}")
            info_lines.append(f"Confidence: {self.current_target['confidence']:.3f}")
            info_lines.append(f"Stable: {self.stable_target_count}/{self.target_stable_frames}")

            # 显示像素和振镜坐标
            px, py = self.current_target['centroid']
            gx, gy = self.pixel_to_galvo(px, py)
            info_lines.append(f"Pixel: ({px:.0f}, {py:.0f})")
            info_lines.append(f"Galvo: ({gx}, {gy})")
        else:
            info_lines.append("Target: None")

        box_height = len(info_lines) * 25 + 10
        cv2.rectangle(image, (10, 10), (400, box_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (400, box_height), (255, 255, 255), 2)

        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(image, line, (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def publish_target_info(self):
        """发布当前目标信息"""
        if self.current_target:
            target_info = {
                'track_id': self.current_target['track_id'],
                'confidence': self.current_target['confidence'],
                'pixel_position': self.current_target['centroid'],
                'galvo_position': self.last_galvo_position,
                'stable_count': self.stable_target_count,
                'tracking_active': self.tracking_active
            }
            target_msg = String()
            target_msg.data = json.dumps(target_info)
            self.target_pub.publish(target_msg)


def main():
    """主函数"""
    try:
        rospy.loginfo("Starting Galvo Tracking Node...")
        node = GalvoTrackingNode()
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
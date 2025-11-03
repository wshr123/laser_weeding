#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""使用 rosbag 回放图像并重新发布到指定 Topic。"""

import os
from typing import Optional, Tuple

import rospy
import rosbag
from sensor_msgs.msg import Image, CameraInfo


class BagImagePublisher:
    """读取 rosbag 中的图像话题并按设定速率发布。"""

    def __init__(self):
        rospy.init_node('bag_image_publisher', anonymous=True)

        self.bag_path = rospy.get_param('~bag_path', '')
        if not self.bag_path:
            rospy.logerr('~bag_path 参数为空，无法播放 rosbag 文件')
            raise rospy.ROSInitException('bag_path is required')

        if not os.path.exists(self.bag_path):
            rospy.logerr(f'rosbag 文件不存在: {self.bag_path}')
            raise rospy.ROSInitException('bag file does not exist')

        self.image_topic_in = rospy.get_param('~image_topic_in', '/camera/image_raw')
        self.image_topic_out = rospy.get_param('~image_topic_out', self.image_topic_in)
        self.camera_info_topic_in = rospy.get_param('~camera_info_topic_in', '')
        self.camera_info_topic_out = rospy.get_param(
            '~camera_info_topic_out', self.camera_info_topic_in or '/camera/camera_info'
        )
        self.playback_rate = float(rospy.get_param('~playback_rate', 1.0))
        if self.playback_rate <= 0:
            rospy.logwarn('播放速率 <= 0，已自动设置为 1.0')
            self.playback_rate = 1.0

        self.loop = bool(rospy.get_param('~loop', False))
        self.update_header_stamp = bool(rospy.get_param('~update_header_stamp', True))

        queue_size = int(rospy.get_param('~queue_size', 1))
        latch = bool(rospy.get_param('~latch', False))

        self.image_pub = rospy.Publisher(
            self.image_topic_out, Image, queue_size=queue_size, latch=latch
        )

        self.publish_camera_info = bool(self.camera_info_topic_in)
        self.camera_info_pub = None
        if self.publish_camera_info:
            self.camera_info_pub = rospy.Publisher(
                self.camera_info_topic_out, CameraInfo, queue_size=queue_size, latch=latch
            )

        topics = [self.image_topic_in]
        if self.publish_camera_info and self.camera_info_topic_in not in topics:
            topics.append(self.camera_info_topic_in)
        self.topics: Tuple[str, ...] = tuple(topics)

        rospy.loginfo('=' * 50)
        rospy.loginfo('启动 bag 图像回放节点')
        rospy.loginfo(f'bag 文件: {self.bag_path}')
        rospy.loginfo(f'图像输入话题: {self.image_topic_in}')
        rospy.loginfo(f'图像输出话题: {self.image_topic_out}')
        if self.publish_camera_info:
            rospy.loginfo(f'CameraInfo 输入话题: {self.camera_info_topic_in}')
            rospy.loginfo(f'CameraInfo 输出话题: {self.camera_info_topic_out}')
        rospy.loginfo(f'播放速率: {self.playback_rate}x, loop={self.loop}')
        rospy.loginfo('=' * 50)

    def run(self):
        """主循环，按需循环播放 bag 文件。"""

        while not rospy.is_shutdown():
            try:
                self._play_bag_once()
            except rospy.ROSInterruptException:
                break
            except Exception as exc:  # pylint: disable=broad-except
                rospy.logerr(f'播放 rosbag 失败: {exc}')
                rospy.sleep(1.0)

            if not self.loop:
                break

    def _play_bag_once(self):
        """逐条读取 rosbag，并根据原始时间间隔重放。"""

        with rosbag.Bag(self.bag_path, 'r') as bag:
            start_stamp: Optional[rospy.rostime.Time] = None
            playback_start: Optional[rospy.Time] = None

            for topic, msg, stamp in bag.read_messages(topics=self.topics):
                if rospy.is_shutdown():
                    break

                if start_stamp is None:
                    start_stamp = stamp
                    playback_start = rospy.Time.now()
                assert playback_start is not None
                elapsed = (stamp - start_stamp).to_sec()
                if elapsed < 0:
                    elapsed = 0.0

                target_time = playback_start + rospy.Duration.from_sec(
                    elapsed / self.playback_rate
                )
                while rospy.Time.now() < target_time and not rospy.is_shutdown():
                    rospy.sleep(0.001)

                if self.update_header_stamp and hasattr(msg, 'header'):
                    msg.header.stamp = rospy.Time.now()

                if topic == self.image_topic_in:
                    self.image_pub.publish(msg)
                elif self.publish_camera_info and topic == self.camera_info_topic_in:
                    assert self.camera_info_pub is not None
                    self.camera_info_pub.publish(msg)


def main():
    publisher = BagImagePublisher()
    publisher.run()


if __name__ == '__main__':
    main()

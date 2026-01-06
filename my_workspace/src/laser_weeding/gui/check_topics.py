#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查 ROS 话题的工具脚本
用于验证 GUI 订阅的话题是否存在且有数据
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import time

def check_topics():
    """检查话题是否存在"""
    rospy.init_node('topic_checker', anonymous=True)
    
    topics_to_check = [
        '/det_img/image_raw',
        '/system_status',
        '/current_target',
        '/galvo_xy',
        '/laser_control'
    ]
    
    print("=" * 60)
    print("检查 ROS 话题")
    print("=" * 60)
    
    # 获取所有可用话题
    available_topics = rospy.get_published_topics()
    topic_dict = {topic[0]: topic[1] for topic in available_topics}
    
    print(f"\n总共找到 {len(available_topics)} 个已发布的话题\n")
    
    for topic in topics_to_check:
        if topic in topic_dict:
            msg_type = topic_dict[topic]
            print(f"✓ {topic:30s} [{msg_type}]")
        else:
            print(f"✗ {topic:30s} [未找到]")
    
    print("\n" + "=" * 60)
    print("检查话题数据流")
    print("=" * 60)
    
    # 检查数据流
    received_topics = {}
    
    def image_callback(msg):
        received_topics['/det_img/image_raw'] = True
        print(f"✓ 收到图像数据: {msg.width}x{msg.height}")
    
    def status_callback(msg):
        received_topics['/system_status'] = True
        print(f"✓ 收到状态数据: {len(msg.data)} 字节")
    
    def target_callback(msg):
        received_topics['/current_target'] = True
        print(f"✓ 收到目标数据: {len(msg.data)} 字节")
    
    # 订阅话题
    subs = []
    if '/det_img/image_raw' in topic_dict:
        subs.append(rospy.Subscriber('/det_img/image_raw', Image, image_callback, queue_size=1))
    if '/system_status' in topic_dict:
        subs.append(rospy.Subscriber('/system_status', String, status_callback, queue_size=1))
    if '/current_target' in topic_dict:
        subs.append(rospy.Subscriber('/current_target', String, target_callback, queue_size=1))
    
    if subs:
        print("\n等待 5 秒接收数据...")
        rospy.sleep(5)
        
        print("\n接收结果:")
        for topic in ['/det_img/image_raw', '/system_status', '/current_target']:
            if topic in received_topics:
                print(f"✓ {topic}: 有数据")
            else:
                print(f"✗ {topic}: 无数据")
    else:
        print("\n没有可订阅的话题")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    try:
        check_topics()
    except rospy.ROSInterruptException:
        pass


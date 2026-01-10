#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
手动振镜控制节点
订阅 /galvo_xy 话题，将命令转发到硬件
用于在没有launch运行时，通过GUI手动控制振镜
"""

import rospy
from std_msgs.msg import Int32MultiArray, Bool
import sys
import os

# 添加脚本目录到 Python 路径，以便导入 send_to_teensy
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from send_to_teensy import XY2_100Controller


class ManualGalvoControlNode:
    """手动振镜控制节点"""
    
    def __init__(self):
        rospy.init_node('manual_galvo_control', anonymous=True)
        
        # 获取参数
        self.serial_port = rospy.get_param('~serial_port', '/dev/ttyACM0')
        self.serial_baudrate = rospy.get_param('~serial_baudrate', 115200)
        self.galvo_count = rospy.get_param('~galvo_count', 2)
        
        # 初始化振镜控制器
        try:
            self.galvo_controller = XY2_100Controller(
                port=self.serial_port,
                baudrate=self.serial_baudrate,
                galvo_count=self.galvo_count
            )
            rospy.loginfo("Manual galvo control node initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize galvo controller: {e}")
            self.galvo_controller = None
        
        # 订阅器
        self.galvo_sub = rospy.Subscriber(
            '/galvo_xy',
            Int32MultiArray,
            self.galvo_command_callback,
            queue_size=10
        )
        
        self.laser_sub = rospy.Subscriber(
            '/laser_control',
            Bool,
            self.laser_control_callback,
            queue_size=10
        )
        
        rospy.loginfo("Manual galvo control node ready, subscribing to /galvo_xy and /laser_control")
    
    def galvo_command_callback(self, msg):
        """振镜命令回调
        支持的消息格式:
        - 格式1: [x, y] - 振镜0（向后兼容）
        - 格式2: [x, y, galvo_index] - 指定振镜（GUI手动控制使用）
        - 格式3: [x, y, laser_state, galvo_index] - main.py格式（兼容）
        """
        if self.galvo_controller is None:
            rospy.logwarn_throttle(1.0, "Galvo controller not initialized")
            return
        
        try:
            data_len = len(msg.data)
            
            if data_len < 2:
                rospy.logwarn(f"Invalid galvo command message: data length {data_len} < 2")
                return
            
            x = int(msg.data[0])
            y = int(msg.data[1])
            galvo_index = 0  # 默认振镜0
            
            # 判断消息格式
            if data_len == 2:
                # 格式1: [x, y] - 振镜0（向后兼容）
                galvo_index = 0
            elif data_len == 3:
                # 格式2: [x, y, galvo_index] - GUI手动控制格式
                galvo_index = int(msg.data[2])
            elif data_len >= 4:
                # 格式3: [x, y, laser_state, galvo_index] - main.py格式
                galvo_index = int(msg.data[3])
            
            # 限制galvo_index范围
            if galvo_index < 0 or galvo_index >= self.galvo_count:
                rospy.logwarn_throttle(1.0, f"Invalid galvo_index: {galvo_index}, using 0")
                galvo_index = 0
            
            # 发送命令到硬件
            # 确保传递 galvo_index（0或1对应振镜1和振镜2）
            rospy.loginfo(f"[Manual Control] Received: msg.data={list(msg.data)}, data_len={data_len}, parsed_galvo_index={galvo_index}, x={x}, y={y}")
            rospy.loginfo(f"[Manual Control] Calling move_to_position with galvo_index={galvo_index}, x={x}, y={y}")
            self.galvo_controller.move_to_position(x, y, galvo_index=galvo_index)
            
        except Exception as e:
            rospy.logerr(f"Error processing galvo command: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def laser_control_callback(self, msg):
        """激光控制回调"""
        if self.galvo_controller is None:
            rospy.logwarn_throttle(1.0, "Galvo controller not initialized")
            return
        
        try:
            if msg.data:
                self.galvo_controller.laser_on()
                rospy.logdebug("Laser turned ON")
            else:
                self.galvo_controller.laser_off()
                rospy.logdebug("Laser turned OFF")
        except Exception as e:
            rospy.logerr(f"Error controlling laser: {e}")
    
    def run(self):
        """运行节点"""
        rospy.spin()
        
        # 清理
        if self.galvo_controller:
            self.galvo_controller.close()


if __name__ == '__main__':
    try:
        node = ManualGalvoControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Manual galvo control node error: {e}")
        import traceback
        traceback.print_exc()


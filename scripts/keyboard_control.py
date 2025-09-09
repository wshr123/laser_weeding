#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
import sys
import select
import tty
import termios
import threading


class KeyboardControlNode:
    """
    键盘控制节点
    用于向校准节点发送键盘命令
    """

    def __init__(self):
        rospy.init_node('keyboard_control_node', anonymous=True)

        # 发布器
        self.command_pub = rospy.Publisher('/calibration/keyboard_input', String, queue_size=10)

        # 保存终端设置
        self.old_attr = termios.tcgetattr(sys.stdin)
        tty.cbreak(sys.stdin.fileno())

        rospy.loginfo("键盘控制节点启动")
        rospy.loginfo("使用说明:")
        rospy.loginfo("  n/p - 选择下一个/上一个目标")
        rospy.loginfo("  a   - 自动瞄准")
        rospy.loginfo("  c   - 回中心")
        rospy.loginfo("  方向键 - 手动调整")
        rospy.loginfo("  +/- - 调整步长")
        rospy.loginfo("  s   - 保存校准点")
        rospy.loginfo("  r   - 计算校准参数")
        rospy.loginfo("  x   - 清除数据")
        rospy.loginfo("  e   - 导出结果")
        rospy.loginfo("  q   - 退出")
        rospy.loginfo("-" * 50)

    def __del__(self):
        # 恢复终端设置
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_attr)

    def get_key(self):
        """获取键盘输入"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

    def run(self):
        """运行键盘监听"""
        while not rospy.is_shutdown():
            key = self.get_key()
            if key:
                command = self.key_to_command(key)
                if command:
                    msg = String()
                    msg.data = command
                    self.command_pub.publish(msg)
                    rospy.loginfo(f"发送命令: {command}")

                    if command in ['quit', 'exit']:
                        break

            rospy.sleep(0.01)

    def key_to_command(self, key):
        """将按键转换为命令"""
        key_map = {
            'n': 'next',
            'p': 'prev',
            'a': 'auto',
            'c': 'center',
            's': 'save',
            'r': 'calc',
            'x': 'clear',
            'e': 'export',
            'q': 'quit',
            '+': 'step_inc',
            '=': 'step_inc',  # 不需要按shift
            '-': 'step_dec',
            '\x1b': 'escape'  # ESC键，用于方向键检测
        }

        # 处理特殊键（方向键等）
        if key == '\x1b':  # ESC序列开始
            # 读取完整的方向键序列
            seq = key
            for _ in range(2):
                next_key = self.get_key()
                if next_key:
                    seq += next_key
                else:
                    break

            if seq == '\x1b[A':
                return 'up'
            elif seq == '\x1b[B':
                return 'down'
            elif seq == '\x1b[C':
                return 'right'
            elif seq == '\x1b[D':
                return 'left'

        return key_map.get(key, None)


def main():
    try:
        node = KeyboardControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        rospy.loginfo("键盘控制节点关闭")


if __name__ == '__main__':
    main()
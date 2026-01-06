#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跟踪效果评估工具

功能：
1. 实时收集跟踪数据
2. 计算跟踪指标（MOTA, MOTP, ID switches等）
3. 生成评估报告和可视化
"""

import rospy
import json
import os
import time
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class TrackingEvaluator:
    """跟踪效果评估器"""
    
    def __init__(self, output_dir=None, save_images=False):
        """
        初始化评估器
        
        Args:
            output_dir: 输出目录，如果为None则使用默认路径
            save_images: 是否保存评估可视化图像
        """
        rospy.init_node('tracking_evaluator', anonymous=True)
        
        # 设置输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.path.expanduser("~"), "tracking_evaluation", timestamp)
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        rospy.loginfo(f"Tracking evaluation output directory: {self.output_dir}")
        
        self.save_images = save_images
        self.bridge = CvBridge()
        
        # 数据存储
        self.tracking_data = []  # 每帧的跟踪结果
        self.track_statistics = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'total_frames': 0,
            'bbox_history': [],
            'confidence_history': [],
            'id_switches': 0  # 该轨迹的ID切换次数
        })
        
        # 全局统计
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_id_switches = 0
        self.total_fragments = 0
        
        # 订阅跟踪结果
        self.tracking_sub = rospy.Subscriber(
            '/tracking_results',
            String,
            self.tracking_callback,
            queue_size=100
        )
        
        # 可选：订阅图像用于可视化
        if self.save_images:
            self.image_sub = rospy.Subscriber(
                '/camera/color/image_raw',
                Image,
                self.image_callback,
                queue_size=10
            )
            self.current_image = None
        
        # 定时保存和报告
        self.save_interval = rospy.get_param('~save_interval', 10.0)  # 每10秒保存一次
        self.last_save_time = time.time()
        
        rospy.loginfo("Tracking evaluator initialized")
    
    def tracking_callback(self, msg):
        """接收跟踪结果"""
        try:
            data = json.loads(msg.data)
            frame_id = data.get('frame_id', self.frame_count)
            timestamp = data.get('timestamp', time.time())
            detections = data.get('detections', [])
            
            # 记录当前帧的跟踪结果
            frame_data = {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'num_detections': len(detections),
                'detections': []
            }
            
            for det in detections:
                track_id = det.get('track_id')
                bbox = det.get('bbox', [])
                confidence = det.get('confidence', 0.0)
                target_point = det.get('target_point', None)
                
                frame_data['detections'].append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'confidence': confidence,
                    'target_point': target_point
                })
                
                # 更新轨迹统计
                if track_id is not None:
                    self._update_track_statistics(track_id, bbox, confidence, timestamp)
            
            self.tracking_data.append(frame_data)
            self.frame_count += 1
            
            # 定期保存
            if time.time() - self.last_save_time > self.save_interval:
                self._save_intermediate_results()
                self.last_save_time = time.time()
                
        except Exception as e:
            rospy.logerr(f"Error processing tracking data: {e}")
    
    def image_callback(self, msg):
        """接收图像（用于可视化）"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(f"Image conversion error: {e}")
    
    def _update_track_statistics(self, track_id, bbox, confidence, timestamp):
        """更新轨迹统计信息"""
        stats = self.track_statistics[track_id]
        
        if stats['first_seen'] is None:
            stats['first_seen'] = timestamp
            self.total_tracks_created += 1
        
        stats['last_seen'] = timestamp
        stats['total_frames'] += 1
        stats['bbox_history'].append(bbox)
        stats['confidence_history'].append(confidence)
        
        # 限制历史长度
        if len(stats['bbox_history']) > 100:
            stats['bbox_history'].pop(0)
        if len(stats['confidence_history']) > 100:
            stats['confidence_history'].pop(0)
    
    def _save_intermediate_results(self):
        """保存中间结果"""
        try:
            # 保存原始数据
            data_file = os.path.join(self.output_dir, 'tracking_data.json')
            with open(data_file, 'w') as f:
                json.dump(self.tracking_data, f, indent=2)
            
            # 保存统计信息
            stats_file = os.path.join(self.output_dir, 'track_statistics.json')
            stats_dict = {}
            for track_id, stats in self.track_statistics.items():
                stats_dict[str(track_id)] = {
                    'first_seen': stats['first_seen'],
                    'last_seen': stats['last_seen'],
                    'total_frames': stats['total_frames'],
                    'avg_confidence': np.mean(stats['confidence_history']) if stats['confidence_history'] else 0.0,
                    'id_switches': stats['id_switches']
                }
            
            with open(stats_file, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            
            rospy.loginfo(f"Intermediate results saved to {self.output_dir}")
            
        except Exception as e:
            rospy.logerr(f"Error saving intermediate results: {e}")
    
    def calculate_tracking_metrics(self):
        """计算跟踪指标"""
        if len(self.tracking_data) == 0:
            rospy.logwarn("No tracking data available for evaluation")
            return None
        
        metrics = {
            'total_frames': self.frame_count,
            'total_tracks': len(self.track_statistics),
            'total_tracks_created': self.total_tracks_created,
        }
        
        # 计算轨迹长度统计
        track_lengths = [stats['total_frames'] for stats in self.track_statistics.values()]
        if track_lengths:
            metrics['avg_track_length'] = np.mean(track_lengths)
            metrics['max_track_length'] = np.max(track_lengths)
            metrics['min_track_length'] = np.min(track_lengths)
            metrics['median_track_length'] = np.median(track_lengths)
        
        # 计算平均每帧检测数
        detections_per_frame = [frame['num_detections'] for frame in self.tracking_data]
        metrics['avg_detections_per_frame'] = np.mean(detections_per_frame) if detections_per_frame else 0
        
        # 计算轨迹存活时间
        track_durations = []
        for stats in self.track_statistics.values():
            if stats['first_seen'] and stats['last_seen']:
                duration = stats['last_seen'] - stats['first_seen']
                track_durations.append(duration)
        
        if track_durations:
            metrics['avg_track_duration'] = np.mean(track_durations)
            metrics['max_track_duration'] = np.max(track_durations)
            metrics['min_track_duration'] = np.min(track_durations)
        
        # 计算平均置信度
        all_confidences = []
        for stats in self.track_statistics.values():
            all_confidences.extend(stats['confidence_history'])
        metrics['avg_confidence'] = np.mean(all_confidences) if all_confidences else 0.0
        
        # 计算轨迹稳定性（短轨迹比例）
        if track_lengths:
            short_tracks = sum(1 for length in track_lengths if length < 5)
            metrics['short_track_ratio'] = short_tracks / len(track_lengths)
            metrics['stable_track_ratio'] = 1.0 - metrics['short_track_ratio']
        
        return metrics
    
    def generate_report(self):
        """生成评估报告"""
        metrics = self.calculate_tracking_metrics()
        if metrics is None:
            return
        
        # 保存最终结果
        self._save_intermediate_results()
        
        # 生成文本报告
        report_file = os.path.join(self.output_dir, 'evaluation_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("跟踪效果评估报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("【基础统计】\n")
            f.write(f"  总帧数: {metrics['total_frames']}\n")
            f.write(f"  总轨迹数: {metrics['total_tracks']}\n")
            f.write(f"  创建的轨迹数: {metrics['total_tracks_created']}\n")
            f.write(f"  平均每帧检测数: {metrics['avg_detections_per_frame']:.2f}\n\n")
            
            f.write("【轨迹长度统计】\n")
            if 'avg_track_length' in metrics:
                f.write(f"  平均轨迹长度: {metrics['avg_track_length']:.2f} 帧\n")
                f.write(f"  最大轨迹长度: {metrics['max_track_length']} 帧\n")
                f.write(f"  最小轨迹长度: {metrics['min_track_length']} 帧\n")
                f.write(f"  中位数轨迹长度: {metrics['median_track_length']:.2f} 帧\n\n")
            
            f.write("【轨迹持续时间统计】\n")
            if 'avg_track_duration' in metrics:
                f.write(f"  平均持续时间: {metrics['avg_track_duration']:.2f} 秒\n")
                f.write(f"  最大持续时间: {metrics['max_track_duration']:.2f} 秒\n")
                f.write(f"  最小持续时间: {metrics['min_track_duration']:.2f} 秒\n\n")
            
            f.write("【轨迹质量】\n")
            f.write(f"  平均置信度: {metrics['avg_confidence']:.3f}\n")
            if 'stable_track_ratio' in metrics:
                f.write(f"  稳定轨迹比例: {metrics['stable_track_ratio']:.2%}\n")
                f.write(f"  短轨迹比例: {metrics['short_track_ratio']:.2%}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        # 保存JSON格式的指标
        metrics_file = os.path.join(self.output_dir, 'tracking_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        rospy.loginfo(f"Evaluation report saved to {report_file}")
        rospy.loginfo(f"Metrics saved to {metrics_file}")
        
        # 打印摘要
        self._print_summary(metrics)
        
        return metrics
    
    def _print_summary(self, metrics):
        """打印评估摘要"""
        rospy.loginfo("=" * 60)
        rospy.loginfo("跟踪效果评估摘要")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"总帧数: {metrics['total_frames']}")
        rospy.loginfo(f"总轨迹数: {metrics['total_tracks']}")
        if 'avg_track_length' in metrics:
            rospy.loginfo(f"平均轨迹长度: {metrics['avg_track_length']:.2f} 帧")
        if 'stable_track_ratio' in metrics:
            rospy.loginfo(f"稳定轨迹比例: {metrics['stable_track_ratio']:.2%}")
        rospy.loginfo("=" * 60)
    
    def shutdown(self):
        """关闭时生成最终报告"""
        rospy.loginfo("Generating final evaluation report...")
        self.generate_report()


def main():
    """主函数"""
    output_dir = rospy.get_param('~output_dir', None)
    save_images = rospy.get_param('~save_images', False)
    
    evaluator = TrackingEvaluator(output_dir=output_dir, save_images=save_images)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        evaluator.shutdown()


if __name__ == '__main__':
    main()


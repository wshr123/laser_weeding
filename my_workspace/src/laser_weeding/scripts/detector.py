import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO, RTDETR
import time
import rospy
from collections import defaultdict, deque
import yaml
import os

# 保留原有的WeedTracker类
class WeedTracker:
    """自定义跟踪器（保持原有实现）"""

    def __init__(self, max_distance=50, max_frames_to_skip=15, min_hits=3, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_frames_to_skip = max_frames_to_skip
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.confidence_history_length = 10
        self.recently_deleted_tracks = {}
        self.deletion_memory_time = 5.0
        self.position_tolerance = 80
        self.size_tolerance = 0.5
        self.quality_threshold = 0.4
        self.total_tracks_created = 0
        self.total_tracks_recovered = 0

    # ... 保持原有的所有方法不变 ...
    def create_kalman_filter(self, x, y):
        """创建卡尔曼滤波器用于位置预测"""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], dtype=np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([float(x), float(y), 0.0, 0.0], dtype=np.float32).reshape(4, 1)
        return kf

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def calculate_size_similarity(self, box1, box2):
        """计算两个边界框的尺寸相似度"""
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        area1 = w1 * h1
        area2 = w2 * h2
        if area1 == 0 or area2 == 0:
            return 0
        ratio = min(area1, area2) / max(area1, area2)
        return ratio

    def should_create_new_track(self, new_bbox, new_centroid, confidence):
        """检查是否应该创建新轨迹"""
        current_time = time.time()
        for track_id, track in self.tracks.items():
            if self.calculate_iou(new_bbox, track['bbox']) > self.iou_threshold:
                return False
            dist = np.linalg.norm(new_centroid - track['predicted_centroid'])
            adaptive_threshold = self.max_distance * (1 + track['frames_skipped'] * 0.1)
            if dist < adaptive_threshold:
                return False
        for deleted_id, deleted_info in list(self.recently_deleted_tracks.items()):
            if current_time - deleted_info['deletion_time'] > self.deletion_memory_time:
                del self.recently_deleted_tracks[deleted_id]
                continue
            deleted_centroid = deleted_info['last_centroid']
            deleted_bbox = deleted_info['last_bbox']
            dist = np.linalg.norm(new_centroid - deleted_centroid)
            size_sim = self.calculate_size_similarity(new_bbox, deleted_bbox)
            if (dist < self.position_tolerance and
                    size_sim > self.size_tolerance and
                    confidence > 0.3):
                # rospy.loginfo(f"Recovering deleted track {deleted_id}")
                self.recover_deleted_track(deleted_id, deleted_info, new_bbox, new_centroid, confidence)
                return False
        return True

    def recover_deleted_track(self, track_id, deleted_info, new_bbox, new_centroid, confidence):
        """恢复被删除的轨迹"""
        kf = self.create_kalman_filter(new_centroid[0], new_centroid[1])
        prev_confidence_history = deleted_info.get('confidence_history', [])
        new_confidence_history = prev_confidence_history[-5:] + [confidence]
        self.tracks[track_id] = {
            'kalman': kf,
            'centroid': new_centroid.copy(),
            'predicted_centroid': new_centroid.copy(),
            'bbox': new_bbox.copy(),
            'frames_skipped': 0,
            'consecutive_hits': max(deleted_info.get('consecutive_hits', 1), 1),
            'confidence_history': new_confidence_history,
            'avg_confidence': np.mean(new_confidence_history),
            'recovered': True,
            'recovery_time': time.time(),
            'total_hits': deleted_info.get('total_hits', 1) + 1,
            'creation_time': deleted_info.get('creation_time', time.time()),
            'quality_score': self.calculate_track_quality(new_confidence_history, 1)
        }
        del self.recently_deleted_tracks[track_id]
        if track_id >= self.next_id:
            self.next_id = track_id + 1
        self.total_tracks_recovered += 1

    def calculate_track_quality(self, confidence_history, consecutive_hits):
        """计算轨迹质量分数"""
        if not confidence_history:
            return 0
        avg_conf = np.mean(confidence_history)
        conf_stability = 1.0 - np.std(confidence_history) if len(confidence_history) > 1 else 1.0
        hit_ratio = min(consecutive_hits / self.min_hits, 1.0)
        quality = (avg_conf * 0.5 + conf_stability * 0.3 + hit_ratio * 0.2)
        return quality

    def delete_track(self, track_id):
        """删除轨迹时记录信息"""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            if track.get('quality_score', 0) > 0.3 or track.get('consecutive_hits', 0) >= 2:
                self.recently_deleted_tracks[track_id] = {
                    'deletion_time': time.time(),
                    'last_centroid': track['centroid'].copy(),
                    'last_confidence': track.get('avg_confidence', 0),
                    'last_bbox': track['bbox'].copy(),
                    'consecutive_hits': track.get('consecutive_hits', 0),
                    'confidence_history': track.get('confidence_history', []).copy(),
                    'total_hits': track.get('total_hits', 0),
                    'creation_time': track.get('creation_time', time.time())
                }
            del self.tracks[track_id]
            rospy.logdebug(f"Deleted track {track_id}")

    def update_track_confidence(self, track_id, confidence):
        """更新轨迹置信度历史"""
        if 'confidence_history' not in self.tracks[track_id]:
            self.tracks[track_id]['confidence_history'] = []
        history = self.tracks[track_id]['confidence_history']
        history.append(confidence)
        if len(history) > self.confidence_history_length:
            history.pop(0)
        self.tracks[track_id]['avg_confidence'] = np.mean(history)
        self.tracks[track_id]['quality_score'] = self.calculate_track_quality(
            history, self.tracks[track_id].get('consecutive_hits', 0)
        )

    def is_reliable_track(self, track_id):
        """判断轨迹是否可靠"""
        if track_id not in self.tracks:
            return False
        track = self.tracks[track_id]
        quality_score = track.get('quality_score', 0)
        consecutive_hits = track.get('consecutive_hits', 0)
        frames_skipped = track.get('frames_skipped', 0)
        return (quality_score > self.quality_threshold and
                consecutive_hits >= self.min_hits and
                frames_skipped <= 3)

    def get_predicted_bbox(self, track):
        """根据预测位置生成边界框"""
        pred_center = track['predicted_centroid']
        last_bbox = track['bbox']
        w, h = last_bbox[2], last_bbox[3]
        return [pred_center[0] - w / 2, pred_center[1] - h / 2, w, h]

    def update(self, detections_with_conf):
        """更新跟踪器"""
        current_time = time.time()

        # 预测所有现有轨迹的位置
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            predicted = track['kalman'].predict()
            track['predicted_centroid'] = np.array([predicted[0, 0], predicted[1, 0]], dtype=np.float32)

        if len(detections_with_conf) == 0:
            for track_id in list(self.tracks.keys()):
                track = self.tracks[track_id]
                track['frames_skipped'] += 1
                track['consecutive_hits'] = max(0, track['consecutive_hits'] - 1)
                if track['frames_skipped'] > self.max_frames_to_skip:
                    self.delete_track(track_id)
            return [(track_id, self.get_predicted_bbox(track), track.get('avg_confidence', 0))
                    for track_id, track in self.tracks.items()
                    if self.is_reliable_track(track_id) or track['frames_skipped'] <= 3]

        detections = [det[0] for det in detections_with_conf]
        confidences = [det[1] for det in detections_with_conf]
        detection_centroids = np.array([[bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                                        for bbox in detections], dtype=np.float32)

        if len(self.tracks) == 0:
            for i, bbox in enumerate(detections):
                if confidences[i] > 0.3:
                    centroid = detection_centroids[i]
                    kf = self.create_kalman_filter(centroid[0], centroid[1])
                    self.tracks[self.next_id] = {
                        'kalman': kf,
                        'centroid': centroid.copy(),
                        'predicted_centroid': centroid.copy(),
                        'bbox': bbox.copy(),
                        'frames_skipped': 0,
                        'consecutive_hits': 1,
                        'confidence_history': [confidences[i]],
                        'avg_confidence': confidences[i],
                        'total_hits': 1,
                        'creation_time': current_time,
                        'quality_score': self.calculate_track_quality([confidences[i]], 1)
                    }
                    self.next_id += 1
                    self.total_tracks_created += 1
        else:
            predicted_centroids = np.array([track['predicted_centroid']
                                            for track in self.tracks.values()], dtype=np.float32)
            track_ids = list(self.tracks.keys())

            if len(predicted_centroids) > 0 and len(detection_centroids) > 0:
                distances = cdist(predicted_centroids, detection_centroids)
                cost_matrix = distances.copy()

                for i, track_id in enumerate(track_ids):
                    track = self.tracks[track_id]
                    for j, detection in enumerate(detections):
                        distance_cost = distances[i, j]
                        iou = self.calculate_iou(track['bbox'], detection)
                        size_sim = self.calculate_size_similarity(track['bbox'], detection)
                        combined_cost = distance_cost - iou * 50 - size_sim * 20
                        adaptive_threshold = self.max_distance * (1 + track['frames_skipped'] * 0.2)
                        if distance_cost > adaptive_threshold * 2:
                            combined_cost = 1e6
                        cost_matrix[i, j] = combined_cost

                if cost_matrix.size > 0:
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                    used_detection_indices = set()
                    updated_tracks = set()

                    for row_idx, col_idx in zip(row_indices, col_indices):
                        if cost_matrix[row_idx, col_idx] < 1e6:
                            track_id = track_ids[row_idx]
                            distance = distances[row_idx, col_idx]
                            adaptive_threshold = self.max_distance * (1 + self.tracks[track_id]['frames_skipped'] * 0.2)

                            if distance < adaptive_threshold:
                                track = self.tracks[track_id]
                                centroid = detection_centroids[col_idx]
                                measurement = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
                                track['kalman'].correct(measurement)
                                track['centroid'] = centroid.copy()
                                track['bbox'] = detections[col_idx].copy()
                                track['frames_skipped'] = 0
                                track['consecutive_hits'] += 1
                                track['total_hits'] = track.get('total_hits', 0) + 1
                                self.update_track_confidence(track_id, confidences[col_idx])
                                if (track.get('recovered', False) and
                                        current_time - track.get('recovery_time', 0) > 2.0):
                                    track['recovered'] = False
                                used_detection_indices.add(col_idx)
                                updated_tracks.add(track_id)

                    for track_id in track_ids:
                        if track_id not in updated_tracks:
                            track = self.tracks[track_id]
                            track['frames_skipped'] += 1
                            track['consecutive_hits'] = max(0, track['consecutive_hits'] - 1)
                            if track['frames_skipped'] > self.max_frames_to_skip:
                                self.delete_track(track_id)

                    for i, bbox in enumerate(detections):
                        if (i not in used_detection_indices and confidences[i] > 0.3):
                            centroid = detection_centroids[i]
                            if self.should_create_new_track(bbox, centroid, confidences[i]):
                                kf = self.create_kalman_filter(centroid[0], centroid[1])
                                self.tracks[self.next_id] = {
                                    'kalman': kf,
                                    'centroid': centroid.copy(),
                                    'predicted_centroid': centroid.copy(),
                                    'bbox': bbox.copy(),
                                    'frames_skipped': 0,
                                    'consecutive_hits': 1,
                                    'confidence_history': [confidences[i]],
                                    'avg_confidence': confidences[i],
                                    'total_hits': 1,
                                    'creation_time': current_time,
                                    'quality_score': self.calculate_track_quality([confidences[i]], 1)
                                }
                                self.next_id += 1
                                self.total_tracks_created += 1

        stable_tracks = []
        for track_id, track in self.tracks.items():
            if self.is_reliable_track(track_id) or track['frames_skipped'] <= 2:
                if track['frames_skipped'] == 0:
                    bbox = track['bbox']
                else:
                    bbox = self.get_predicted_bbox(track)
                stable_tracks.append((track_id, bbox, track.get('avg_confidence', 0)))

        return stable_tracks

    def get_statistics(self):
        """获取跟踪器统计信息"""
        return {
            'active_tracks': len(self.tracks),
            'total_created': self.total_tracks_created,
            'total_recovered': self.total_tracks_recovered,
            'deleted_tracks_memory': len(self.recently_deleted_tracks),
            'reliable_tracks': sum(1 for tid in self.tracks.keys() if self.is_reliable_track(tid))
        }


class WeedDetector:
    def __init__(self, model_path, model_type='yolov8', weed_class_id=0,
                 crop_class_id=1, confidence_threshold=0.3, device='cuda:0',
                 tracker_type='custom', tracker_config=None,
                 detection_mode='yolo_pose', exg_params=None, seg_params=None, pose_params=None):
        """
        初始化杂草检测器，支持多种模型和跟踪器

        Args:
            model_path: 模型文件路径
            model_type: 模型类型 ('yolov8', 'yolov11', 'rtdetr')
            weed_class_id: 杂草类别ID
            crop_class_id: 作物类别ID
            confidence_threshold: 检测置信度阈值
            device: 设备类型 ('cuda:0' 或 'cpu')
            tracker_type: 跟踪器类型 ('custom', 'bytetrack', 'botsort')
            tracker_config: 跟踪器配置文件路径（用于Ultralytics跟踪器）
            detection_mode: 检测模式 ('bbox', 'yolo_world_exg', 'yolo_seg', 'yolo_pose')
            exg_params: ExG相关参数字典
            seg_params: 分割相关参数字典
            pose_params: 姿态相关参数字典
        """
        self.model_type = model_type.lower()
        self.weed_class_id = weed_class_id
        self.crop_class_id = crop_class_id
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.tracker_type = tracker_type.lower()
        self.detection_mode = detection_mode.lower()

        try:
            # 加载模型（根据detection_mode可能需要不同的模型类型）
            if self.detection_mode in ['yolo_pose', 'yolo_seg', 'yolo_world_exg']:
                # 这些模式都需要YOLO模型
                self.model = YOLO(model_path)
                rospy.loginfo(f"Successfully loaded YOLO model from {model_path} (mode: {self.detection_mode})")
            elif self.model_type in ['yolov8', 'yolov11']:
                self.model = YOLO(model_path)
                rospy.loginfo(f"Successfully loaded {self.model_type.upper()} model from {model_path}")
            elif self.model_type == 'rtdetr':
                self.model = RTDETR(model_path)
                rospy.loginfo(f"Successfully loaded RT-DETR model from {model_path}")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # 设置模型设备
            self.model.to(device)

        except Exception as e:
            rospy.logerr(f"Failed to load {self.model_type} model: {e}")
            raise

        # 初始化检测模式相关参数
        self.exg_params = exg_params or {
            'use_otsu': True,
            'morph_kernel_size': 3,
            'min_mask_area': 50
        }
        self.seg_params = seg_params or {
            'retina_masks': True,
            'imgsz': 640
        }
        self.pose_params = pose_params or {
            'keypoint_index': 0,  # 根部关键点索引
            'min_keypoint_confidence': 0.3
        }

        # 初始化跟踪器
        if self.tracker_type == 'custom':
            # 使用自定义跟踪器
            self.weed_tracker = WeedTracker(
                max_distance=80,
                max_frames_to_skip=20,
                min_hits=3,
                iou_threshold=0.2
            )
            self.use_custom_tracker = True
            rospy.loginfo("Using custom weed tracker")

        elif self.tracker_type in ['bytetrack', 'botsort']:
            # 使用Ultralytics内置跟踪器
            self.use_custom_tracker = False

            # 设置跟踪器配置
            if tracker_config and os.path.exists(tracker_config):
                with open(tracker_config, 'r') as f:
                    self.tracker_config = yaml.safe_load(f)
            else:
                # 默认配置
                self.tracker_config = {
                    'tracker_type': self.tracker_type,
                    'track_high_thresh': 0.5,
                    'track_low_thresh': 0.1,
                    'new_track_thresh': 0.6,
                    'track_buffer': 30,
                    'match_thresh': 0.8,
                    'min_box_area': 10,
                    'mot20': False,
                }

            rospy.loginfo(f"Using Ultralytics {self.tracker_type.upper()} tracker")

            # 为了兼容性，创建一个轨迹信息存储
            self.ultralytics_tracks = {}
            self.track_confidence_history = defaultdict(lambda: deque(maxlen=10))
            self.track_consecutive_hits = defaultdict(int)
            self.track_quality_scores = {}

        else:
            raise ValueError(f"Unsupported tracker type: {self.tracker_type}")

        # 检测结果缓存和平滑
        self.detection_history = deque(maxlen=5)
        self.frame_count = 0

        # 检测统计
        self.detection_stats = {
            'total_detections': 0,
            'total_frames': 0,
            'avg_detections_per_frame': 0,
            'model_type': self.model_type,
            'tracker_type': self.tracker_type
        }

        # 图像处理参数
        self.input_size = (640, 640)

        # 存储对靶点信息（用于新检测模式）
        self.target_points_dict = {}  # {track_id: [x, y]}

        rospy.loginfo(f"WeedDetector initialized with {self.model_type.upper()} model, {self.tracker_type} tracker, detection_mode: {self.detection_mode}")

    def preprocess_image(self, image):
        """预处理图像"""
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
        return enhanced_image

    # ==================== 新增检测模式的辅助函数 ====================
    
    def compute_exg_mask_roi(self, roi_image):
        """计算 ROI 区域的 ExG 掩膜"""
        if roi_image is None or roi_image.size == 0:
            return None
        
        img_float = roi_image.astype(np.float32)
        B, G, R = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
        exg = 2 * G - R - B
        
        # 归一化到0-255
        exg_min, exg_max = exg.min(), exg.max()
        if exg_max - exg_min > 0:
            exg = 255 * (exg - exg_min) / (exg_max - exg_min)
        exg = exg.astype(np.uint8)
        
        # 二值化
        if self.exg_params.get('use_otsu', True):
            _, mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(exg, self.exg_params.get('threshold', 127), 255, cv2.THRESH_BINARY)
        
        # 形态学操作去噪
        kernel_size = self.exg_params.get('morph_kernel_size', 3)
        if kernel_size > 0:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def calculate_mask_centroid(self, mask, bbox):
        """计算掩膜质心，返回全局坐标 [x, y]
        
        Args:
            mask: 二值掩膜（ROI区域）
            bbox: 边界框 [x, y, w, h]（全局坐标）
        
        Returns:
            (target_point, is_fallback): 对靶点坐标和是否使用了保底策略
        """
        if mask is None:
            # 保底：使用bbox中心
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            return [center_x, center_y], True
        
        # 找到所有白色像素(255)的坐标
        ys, xs = np.where(mask > 0)
        
        if len(xs) > 0:
            # 计算质心（局部坐标）
            local_root_x = int(np.mean(xs))
            local_root_y = int(np.mean(ys))
            # 转换为全局坐标
            global_x = bbox[0] + local_root_x
            global_y = bbox[1] + local_root_y
            return [global_x, global_y], False
        else:
            # 掩膜为空，使用bbox中心作为保底
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            return [center_x, center_y], True

    def mask_to_bbox(self, mask):
        """从分割掩膜生成边界框
        
        Args:
            mask: 二值掩膜
        
        Returns:
            [x, y, w, h] 或 None（如果掩膜为空）
        """
        if mask is None or mask.size == 0:
            return None
        
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        
        return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

    def attach_target_points(self, tracked_results, target_points_dict):
        """将 target_point 附加到跟踪结果
        
        Args:
            tracked_results: [(track_id, bbox, conf), ...]
            target_points_dict: {track_id: [x, y]}
        
        Returns:
            [(track_id, bbox, conf, target_point), ...]
        """
        result_with_points = []
        for track_id, bbox, conf in tracked_results:
            if track_id in target_points_dict:
                target_point = target_points_dict[track_id]
            else:
                # 如果没有对靶点，使用bbox中心
                target_point = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
            result_with_points.append((track_id, bbox, conf, target_point))
        return result_with_points

    def filter_detections(self, detections, image_shape):
        """过滤检测结果"""
        filtered_detections = []
        h, w = image_shape[:2]

        for detection in detections:
            bbox, conf = detection
            x, y, width, height = bbox

            if (x < 0 or y < 0 or x + width > w or y + height > h):
                continue

            area = width * height
            if area < 100 or area > w * h * 0.5:
                continue

            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue

            filtered_detections.append(detection)

        return filtered_detections

    def smooth_detections(self, current_detections):
        """检测结果时间平滑"""
        if len(self.detection_history) < 2:
            return current_detections
        return current_detections

    def detect_and_track_weeds_ultralytics(self, np_image):
        self.frame_count += 1
        det_image = np_image.copy()
        self.ctr_points = []    #center points

        processed_image = self.preprocess_image(np_image)

        try:
            if self.tracker_type == 'bytetrack':
                results = self.model.track(
                    processed_image,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.confidence_threshold,
                    device=self.device,
                    classes=[self.weed_class_id, self.crop_class_id],
                    verbose=False,
                )
            elif self.tracker_type == 'botsort':
                results = self.model.track(
                    processed_image,
                    persist=True,
                    tracker="botsort.yaml",
                    conf=self.confidence_threshold,
                    device=self.device,
                    classes=[self.weed_class_id, self.crop_class_id],
                    verbose=False,
                )

            tracked_weeds = []
            crop_detections = []

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                # 获取跟踪ID
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                else:
                    # 如果没有跟踪ID，生成临时ID
                    track_ids = np.arange(len(boxes.xyxy))

                bboxes_xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                for i in range(len(bboxes_xyxy)):
                    x1, y1, x2, y2 = bboxes_xyxy[i]
                    conf = float(confidences[i])
                    cls = int(classes[i])
                    track_id = int(track_ids[i])
                    #xyxy -> xywh
                    bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                    if cls == self.crop_class_id and conf > 0.4:
                        width, height = x2 - x1, y2 - y1
                        if width < 100 and height < 100:    #todo 根据实际作物尺寸改
                            crop_detections.append((bbox_xywh, conf))

                    elif cls == self.weed_class_id and conf > self.confidence_threshold:
                        # 更新轨迹信息
                        self.update_ultralytics_track_info(track_id, bbox_xywh, conf)

                        # 获取轨迹质量信息
                        track_info = self.get_ultralytics_track_quality(track_id)

                        # 添加到跟踪结果
                        tracked_weeds.append((track_id, bbox_xywh, track_info['avg_confidence']))

                        # 计算中心点
                        center_x = x1 + (x2 - x1) / 2
                        center_y = y1 + (y2 - y1) / 2
                        self.ctr_points.append([center_x, center_y])

            # 更新统计信息
            self.detection_stats['total_detections'] += len(tracked_weeds)
            self.detection_stats['total_frames'] += 1
            self.detection_stats['avg_detections_per_frame'] = (
                    self.detection_stats['total_detections'] / self.detection_stats['total_frames']
            )

            # 统一返回格式：添加target_point（使用bbox中心）
            tracked_weeds_with_points = self.attach_target_points(tracked_weeds, {})
            
            # 绘制结果
            # det_image = self.draw_results_ultralytics(det_image, plant_detections, tracked_weeds)

            return det_image, tracked_weeds_with_points

        except Exception as e:
            rospy.logerr(f"Ultralytics tracking failed: {e}")
            return np_image, []

    def update_ultralytics_track_info(self, track_id, bbox, confidence):
        # 更新置信度历史
        self.track_confidence_history[track_id].append(confidence)

        # 更新连续命中次数
        if track_id in self.ultralytics_tracks:
            self.track_consecutive_hits[track_id] += 1
        else:
            self.track_consecutive_hits[track_id] = 1

        # 存储轨迹信息
        self.ultralytics_tracks[track_id] = {
            'bbox': bbox,
            'confidence': confidence,
            'last_seen': time.time(),
            'centroid': np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
        }

        # 计算质量分数
        self.track_quality_scores[track_id] = self.calculate_ultralytics_track_quality(track_id)
        return

    def calculate_ultralytics_track_quality(self, track_id):
        history = list(self.track_confidence_history[track_id])
        if not history:
            return 0

        avg_conf = np.mean(history)
        conf_stability = 1.0 - np.std(history) if len(history) > 1 else 1.0
        consecutive_hits = self.track_consecutive_hits[track_id]
        hit_ratio = min(consecutive_hits / 3, 1.0)  # 连续3帧检测到为稳定

        quality = (avg_conf * 0.5 + conf_stability * 0.3 + hit_ratio * 0.2)
        return quality

    def get_ultralytics_track_quality(self, track_id):
        history = list(self.track_confidence_history[track_id])
        return {
            'avg_confidence': np.mean(history) if history else 0,
            'consecutive_hits': self.track_consecutive_hits[track_id],
            'quality_score': self.track_quality_scores.get(track_id, 0)
        }

    def detect_plants_and_weeds(self, np_image):
        """检测植物和杂草（用于自定义跟踪器）"""
        processed_image = self.preprocess_image(np_image)

        try:
            results = self.model(processed_image, conf=self.confidence_threshold, device=self.device, verbose=False)

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                bboxes_xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
            else:
                bboxes_xyxy = np.array([])
                confidences = np.array([])
                classes = np.array([])

        except Exception as e:
            rospy.logerr(f"{self.model_type.upper()} detection failed: {e}")
            return [], []

        plant_detections = []
        weed_detections = []

        for i in range(len(bboxes_xyxy)):
            x1, y1, x2, y2 = bboxes_xyxy[i]
            conf = float(confidences[i])
            cls = int(classes[i])

            bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            if cls == self.crop_class_id and conf > 0.4:
                width, height = x2 - x1, y2 - y1
                if width < 100 and height < 100:
                    plant_detections.append((bbox_xywh, conf))

            elif cls == self.weed_class_id and conf > self.confidence_threshold:
                weed_detections.append((bbox_xywh, conf))

        weed_detections = self.filter_detections(weed_detections, np_image.shape)
        plant_detections = self.filter_detections(plant_detections, np_image.shape)

        self.detection_stats['total_detections'] += len(weed_detections)
        self.detection_stats['total_frames'] += 1
        self.detection_stats['avg_detections_per_frame'] = (
                self.detection_stats['total_detections'] / self.detection_stats['total_frames']
        )

        return plant_detections, weed_detections

    def detect_and_track_weeds(self, np_image):
        """检测和跟踪路由函数"""
        if self.detection_mode == 'bbox':
            # 原有检测方法
            if self.use_custom_tracker:
                return self.detect_and_track_weeds_custom(np_image)
            else:
                return self.detect_and_track_weeds_ultralytics(np_image)
        elif self.detection_mode == 'yolo_pose':
            return self.detect_and_track_yolo_pose(np_image)
        elif self.detection_mode == 'yolo_seg':
            return self.detect_and_track_yolo_seg(np_image)
        elif self.detection_mode == 'yolo_world_exg':
            return self.detect_and_track_yolo_world_exg(np_image)
        else:
            rospy.logwarn(f"Unknown detection_mode: {self.detection_mode}, falling back to bbox")
            if self.use_custom_tracker:
                return self.detect_and_track_weeds_custom(np_image)
            else:
                return self.detect_and_track_weeds_ultralytics(np_image)

    def detect_and_track_weeds_custom(self, np_image):
        """使用自定义跟踪器检测并跟踪杂草"""
        self.frame_count += 1
        det_image = np_image.copy()
        self.ctr_points = []

        plant_detections, weed_detections = self.detect_plants_and_weeds(np_image)

        self.detection_history.append(weed_detections)

        smoothed_detections = self.smooth_detections(weed_detections)

        tracked_weeds = self.weed_tracker.update(smoothed_detections)

        for _, bbox, _ in tracked_weeds:
            x, y, w, h = bbox
            center_x = x + w / 2
            center_y = y + h / 2
            self.ctr_points.append([center_x, center_y])

        det_image = self.draw_results(det_image, plant_detections, tracked_weeds)

        # 统一返回格式：添加target_point（使用bbox中心）
        tracked_weeds_with_points = self.attach_target_points(tracked_weeds, {})
        return det_image, tracked_weeds_with_points

    # ==================== 新增检测模式实现 ====================

    def detect_and_track_yolo_pose(self, np_image):
        """使用 YOLO11-pose 检测并跟踪杂草（关键点检测）"""
        self.frame_count += 1
        det_image = np_image.copy()
        self.ctr_points = []
        self.target_points_dict = {}

        processed_image = self.preprocess_image(np_image)

        try:
            # 使用YOLO pose模型进行推理
            results = self.model.predict(
                processed_image,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
                stream=False
            )

            result = results[0]
            boxes = result.boxes
            keypoints = result.keypoints

            # 准备检测结果用于跟踪
            weed_detections = []
            target_points_dict = {}  # {index: [x, y]}

            if boxes is not None and len(boxes) > 0:
                bboxes_xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                for i in range(len(bboxes_xyxy)):
                    x1, y1, x2, y2 = bboxes_xyxy[i]
                    conf = float(confidences[i])
                    cls = int(classes[i])

                    if cls == self.weed_class_id and conf > self.confidence_threshold:
                        # 转换为xywh格式
                        bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        weed_detections.append((bbox_xywh, conf))

                        # 获取关键点（根部）
                        target_point = None
                        if keypoints is not None and keypoints.xy.shape[1] > 0:
                            kpts = keypoints.xy[i].cpu().numpy()
                            keypoint_idx = self.pose_params.get('keypoint_index', 0)
                            if keypoint_idx < len(kpts):
                                root_x, root_y = kpts[keypoint_idx]
                                if root_x > 0 and root_y > 0:
                                    target_point = [float(root_x), float(root_y)]

                        # 如果没有有效关键点，使用bbox中心
                        if target_point is None:
                            target_point = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]

                        target_points_dict[len(weed_detections) - 1] = target_point

            # 过滤检测结果
            weed_detections = self.filter_detections(weed_detections, np_image.shape)

            # 使用跟踪器
            if self.use_custom_tracker:
                tracked_weeds = self.weed_tracker.update(weed_detections)
            else:
                # 使用Ultralytics跟踪器
                if self.tracker_type == 'bytetrack':
                    track_results = self.model.track(
                        processed_image,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=self.confidence_threshold,
                        device=self.device,
                        classes=[self.weed_class_id],
                        verbose=False,
                    )
                elif self.tracker_type == 'botsort':
                    track_results = self.model.track(
                        processed_image,
                        persist=True,
                        tracker="botsort.yaml",
                        conf=self.confidence_threshold,
                        device=self.device,
                        classes=[self.weed_class_id],
                        verbose=False,
                    )
                else:
                    track_results = []

                tracked_weeds = []
                if len(track_results) > 0 and track_results[0].boxes is not None:
                    boxes = track_results[0].boxes
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = np.arange(len(boxes.xyxy))
                    
                    bboxes_xyxy = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    
                    for i in range(len(bboxes_xyxy)):
                        x1, y1, x2, y2 = bboxes_xyxy[i]
                        conf = float(confidences[i])
                        track_id = int(track_ids[i])
                        bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        # 获取关键点
                        if track_results[0].keypoints is not None:
                            kpts = track_results[0].keypoints.xy[i].cpu().numpy()
                            keypoint_idx = self.pose_params.get('keypoint_index', 0)
                            if keypoint_idx < len(kpts):
                                root_x, root_y = kpts[keypoint_idx]
                                if root_x > 0 and root_y > 0:
                                    self.target_points_dict[track_id] = [float(root_x), float(root_y)]
                        
                        if track_id not in self.target_points_dict:
                            self.target_points_dict[track_id] = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
                        
                        self.update_ultralytics_track_info(track_id, bbox_xywh, conf)
                        track_info = self.get_ultralytics_track_quality(track_id)
                        tracked_weeds.append((track_id, bbox_xywh, track_info['avg_confidence']))

            # 将关键点附加到跟踪结果
            if self.use_custom_tracker:
                # 需要将检测索引映射到跟踪ID
                final_target_points = {}
                for track_id, bbox, conf in tracked_weeds:
                    # 找到最匹配的检测结果
                    best_match_idx = -1
                    best_iou = 0
                    for idx, (det_bbox, _) in enumerate(weed_detections):
                        iou = self.weed_tracker.calculate_iou(bbox, det_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_idx = idx
                    
                    if best_match_idx >= 0 and best_match_idx in target_points_dict:
                        final_target_points[track_id] = target_points_dict[best_match_idx]
                    else:
                        # 使用bbox中心
                        final_target_points[track_id] = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                
                tracked_weeds = self.attach_target_points(tracked_weeds, final_target_points)
            else:
                tracked_weeds = self.attach_target_points(tracked_weeds, self.target_points_dict)

            # 更新中心点列表
            for track_id, bbox, conf, target_point in tracked_weeds:
                self.ctr_points.append(target_point)

            # 更新统计信息
            self.detection_stats['total_detections'] += len(tracked_weeds)
            self.detection_stats['total_frames'] += 1
            self.detection_stats['avg_detections_per_frame'] = (
                self.detection_stats['total_detections'] / self.detection_stats['total_frames']
            )

            # 绘制结果（可选）
            # det_image = self.draw_results_pose(det_image, tracked_weeds)

            return det_image, tracked_weeds

        except Exception as e:
            rospy.logerr(f"YOLO pose detection failed: {e}")
            return np_image, []

    def detect_and_track_yolo_seg(self, np_image):
        """使用 YOLO11-seg 检测并跟踪杂草（分割掩膜）"""
        self.frame_count += 1
        det_image = np_image.copy()
        self.ctr_points = []
        self.target_points_dict = {}

        processed_image = self.preprocess_image(np_image)

        try:
            # 使用YOLO seg模型进行推理
            results = self.model.predict(
                processed_image,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
                retina_masks=self.seg_params.get('retina_masks', True),
                imgsz=self.seg_params.get('imgsz', 640)
            )

            result = results[0]
            boxes = result.boxes
            masks = result.masks

            # 准备检测结果用于跟踪
            weed_detections = []
            target_points_dict = {}  # {index: [x, y]}

            if boxes is not None and len(boxes) > 0:
                bboxes_xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                for i in range(len(bboxes_xyxy)):
                    x1, y1, x2, y2 = bboxes_xyxy[i]
                    conf = float(confidences[i])
                    cls = int(classes[i])

                    if cls == self.weed_class_id and conf > self.confidence_threshold:
                        # 从分割掩膜生成bbox
                        if masks is not None and i < len(masks.data):
                            mask = masks.data[i].cpu().numpy()
                            # 将mask转换为uint8
                            if mask.dtype != np.uint8:
                                mask = (mask * 255).astype(np.uint8)
                            
                            # 从掩膜生成bbox
                            bbox_from_mask = self.mask_to_bbox(mask)
                            if bbox_from_mask is not None:
                                # 转换为全局坐标
                                bbox_xywh = [
                                    float(bbox_from_mask[0]),
                                    float(bbox_from_mask[1]),
                                    float(bbox_from_mask[2]),
                                    float(bbox_from_mask[3])
                                ]
                            else:
                                # 如果掩膜为空，使用检测框
                                bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                            
                            # 计算掩膜质心作为对靶点
                            mask_roi = mask[int(bbox_from_mask[1]):int(bbox_from_mask[1]+bbox_from_mask[3]),
                                          int(bbox_from_mask[0]):int(bbox_from_mask[0]+bbox_from_mask[2])] if bbox_from_mask else None
                            target_point, _ = self.calculate_mask_centroid(mask_roi, bbox_xywh)
                        else:
                            # 没有掩膜，使用检测框
                            bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                            target_point = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]

                        weed_detections.append((bbox_xywh, conf))
                        target_points_dict[len(weed_detections) - 1] = target_point

            # 过滤检测结果
            weed_detections = self.filter_detections(weed_detections, np_image.shape)

            # 使用跟踪器
            if self.use_custom_tracker:
                tracked_weeds = self.weed_tracker.update(weed_detections)
                
                # 将质心附加到跟踪结果
                final_target_points = {}
                for track_id, bbox, conf in tracked_weeds:
                    # 找到最匹配的检测结果
                    best_match_idx = -1
                    best_iou = 0
                    for idx, (det_bbox, _) in enumerate(weed_detections):
                        iou = self.weed_tracker.calculate_iou(bbox, det_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_idx = idx
                    
                    if best_match_idx >= 0 and best_match_idx in target_points_dict:
                        final_target_points[track_id] = target_points_dict[best_match_idx]
                    else:
                        # 使用bbox中心
                        final_target_points[track_id] = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                
                tracked_weeds = self.attach_target_points(tracked_weeds, final_target_points)
            else:
                # 使用Ultralytics跟踪器（需要重新推理以获取跟踪ID）
                if self.tracker_type == 'bytetrack':
                    track_results = self.model.track(
                        processed_image,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=self.confidence_threshold,
                        device=self.device,
                        classes=[self.weed_class_id],
                        verbose=False,
                        retina_masks=self.seg_params.get('retina_masks', True)
                    )
                elif self.tracker_type == 'botsort':
                    track_results = self.model.track(
                        processed_image,
                        persist=True,
                        tracker="botsort.yaml",
                        conf=self.confidence_threshold,
                        device=self.device,
                        classes=[self.weed_class_id],
                        verbose=False,
                        retina_masks=self.seg_params.get('retina_masks', True)
                    )
                else:
                    track_results = []

                tracked_weeds = []
                if len(track_results) > 0 and track_results[0].boxes is not None:
                    boxes = track_results[0].boxes
                    masks = track_results[0].masks
                    
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = np.arange(len(boxes.xyxy))
                    
                    bboxes_xyxy = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    
                    for i in range(len(bboxes_xyxy)):
                        x1, y1, x2, y2 = bboxes_xyxy[i]
                        conf = float(confidences[i])
                        track_id = int(track_ids[i])
                        
                        # 从掩膜计算质心
                        if masks is not None and i < len(masks.data):
                            mask = masks.data[i].cpu().numpy()
                            if mask.dtype != np.uint8:
                                mask = (mask * 255).astype(np.uint8)
                            
                            bbox_from_mask = self.mask_to_bbox(mask)
                            if bbox_from_mask:
                                bbox_xywh = [
                                    float(bbox_from_mask[0]),
                                    float(bbox_from_mask[1]),
                                    float(bbox_from_mask[2]),
                                    float(bbox_from_mask[3])
                                ]
                                mask_roi = mask[int(bbox_from_mask[1]):int(bbox_from_mask[1]+bbox_from_mask[3]),
                                              int(bbox_from_mask[0]):int(bbox_from_mask[0]+bbox_from_mask[2])]
                                target_point, _ = self.calculate_mask_centroid(mask_roi, bbox_xywh)
                            else:
                                bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                                target_point = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
                        else:
                            bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                            target_point = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
                        
                        self.target_points_dict[track_id] = target_point
                        self.update_ultralytics_track_info(track_id, bbox_xywh, conf)
                        track_info = self.get_ultralytics_track_quality(track_id)
                        tracked_weeds.append((track_id, bbox_xywh, track_info['avg_confidence']))
                
                tracked_weeds = self.attach_target_points(tracked_weeds, self.target_points_dict)

            # 更新中心点列表
            for track_id, bbox, conf, target_point in tracked_weeds:
                self.ctr_points.append(target_point)

            # 更新统计信息
            self.detection_stats['total_detections'] += len(tracked_weeds)
            self.detection_stats['total_frames'] += 1
            self.detection_stats['avg_detections_per_frame'] = (
                self.detection_stats['total_detections'] / self.detection_stats['total_frames']
            )

            return det_image, tracked_weeds

        except Exception as e:
            rospy.logerr(f"YOLO seg detection failed: {e}")
            return np_image, []

    def detect_and_track_yolo_world_exg(self, np_image):
        """使用 YOLO World + ExG 检测并跟踪杂草"""
        self.frame_count += 1
        det_image = np_image.copy()
        self.ctr_points = []
        self.target_points_dict = {}

        processed_image = self.preprocess_image(np_image)

        try:
            # 1. 全图ExG计算
            global_mask = self.compute_exg_mask_roi(np_image)

            # 2. YOLO World检测
            results = self.model.predict(
                processed_image,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
                stream=False
            )

            result = results[0]
            boxes = result.boxes

            # 准备检测结果用于跟踪
            weed_detections = []
            target_points_dict = {}  # {index: [x, y]}

            if boxes is not None and len(boxes) > 0:
                bboxes_xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                h, w = np_image.shape[:2]

                for i in range(len(bboxes_xyxy)):
                    x1, y1, x2, y2 = bboxes_xyxy[i]
                    conf = float(confidences[i])
                    cls = int(classes[i])

                    if cls == self.weed_class_id and conf > self.confidence_threshold:
                        # 边界保护
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, int(x2)), min(h, int(y2))
                        
                        if x2 <= x1 or y2 <= y1:
                            continue

                        bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        weed_detections.append((bbox_xywh, conf))

                        # 3. 在检测框ROI内计算ExG掩膜质心
                        if global_mask is not None:
                            roi_mask = global_mask[y1:y2, x1:x2]
                            target_point, is_fallback = self.calculate_mask_centroid(roi_mask, bbox_xywh)
                        else:
                            # 如果全局掩膜计算失败，使用bbox中心
                            target_point = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
                            is_fallback = True

                        target_points_dict[len(weed_detections) - 1] = target_point

            # 过滤检测结果
            weed_detections = self.filter_detections(weed_detections, np_image.shape)

            # 使用跟踪器
            if self.use_custom_tracker:
                tracked_weeds = self.weed_tracker.update(weed_detections)
                
                # 将质心附加到跟踪结果
                final_target_points = {}
                for track_id, bbox, conf in tracked_weeds:
                    # 找到最匹配的检测结果
                    best_match_idx = -1
                    best_iou = 0
                    for idx, (det_bbox, _) in enumerate(weed_detections):
                        iou = self.weed_tracker.calculate_iou(bbox, det_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_idx = idx
                    
                    if best_match_idx >= 0 and best_match_idx in target_points_dict:
                        final_target_points[track_id] = target_points_dict[best_match_idx]
                    else:
                        # 使用bbox中心
                        final_target_points[track_id] = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                
                tracked_weeds = self.attach_target_points(tracked_weeds, final_target_points)
            else:
                # 使用Ultralytics跟踪器
                if self.tracker_type == 'bytetrack':
                    track_results = self.model.track(
                        processed_image,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=self.confidence_threshold,
                        device=self.device,
                        classes=[self.weed_class_id],
                        verbose=False,
                    )
                elif self.tracker_type == 'botsort':
                    track_results = self.model.track(
                        processed_image,
                        persist=True,
                        tracker="botsort.yaml",
                        conf=self.confidence_threshold,
                        device=self.device,
                        classes=[self.weed_class_id],
                        verbose=False,
                    )
                else:
                    track_results = []

                tracked_weeds = []
                if len(track_results) > 0 and track_results[0].boxes is not None:
                    boxes = track_results[0].boxes
                    
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = np.arange(len(boxes.xyxy))
                    
                    bboxes_xyxy = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    
                    for i in range(len(bboxes_xyxy)):
                        x1, y1, x2, y2 = bboxes_xyxy[i]
                        conf = float(confidences[i])
                        track_id = int(track_ids[i])
                        bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        # 计算ExG质心
                        x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                        x2_int, y2_int = min(w, int(x2)), min(h, int(y2))
                        if global_mask is not None and x2_int > x1_int and y2_int > y1_int:
                            roi_mask = global_mask[y1_int:y2_int, x1_int:x2_int]
                            target_point, _ = self.calculate_mask_centroid(roi_mask, bbox_xywh)
                        else:
                            target_point = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
                        
                        self.target_points_dict[track_id] = target_point
                        self.update_ultralytics_track_info(track_id, bbox_xywh, conf)
                        track_info = self.get_ultralytics_track_quality(track_id)
                        tracked_weeds.append((track_id, bbox_xywh, track_info['avg_confidence']))
                
                tracked_weeds = self.attach_target_points(tracked_weeds, self.target_points_dict)

            # 更新中心点列表
            for track_id, bbox, conf, target_point in tracked_weeds:
                self.ctr_points.append(target_point)

            # 更新统计信息
            self.detection_stats['total_detections'] += len(tracked_weeds)
            self.detection_stats['total_frames'] += 1
            self.detection_stats['avg_detections_per_frame'] = (
                self.detection_stats['total_detections'] / self.detection_stats['total_frames']
            )

            return det_image, tracked_weeds

        except Exception as e:
            rospy.logerr(f"YOLO World + ExG detection failed: {e}")
            return np_image, []

    def draw_results(self, image, plant_detections, tracked_weeds):
        """绘制检测和跟踪结果（自定义跟踪器）"""
        result_image = image.copy()

        for plant_det, conf in plant_detections:
            x, y, w, h = plant_det
            cv2.rectangle(result_image, (int(x), int(y)), (int(x + w), int(y + h)),
                          (255, 255, 255), 2)
            label = f'Plant ({conf:.2f})'
            cv2.putText(result_image, label, (int(x), int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for track_id, bbox, avg_conf in tracked_weeds:
            x, y, w, h = bbox
            track_info = self.weed_tracker.tracks.get(track_id, {})

            if track_info.get('recovered', False):
                color = (0, 255, 0)
                thickness = 3
                line_type = cv2.LINE_4
            elif self.weed_tracker.is_reliable_track(track_id):
                color = (0, 255, 255)
                thickness = 2
                line_type = cv2.LINE_8
            else:
                color = (255, 0, 0)
                thickness = 2
                line_type = cv2.LINE_8

            cv2.rectangle(result_image, (int(x), int(y)), (int(x + w), int(y + h)),
                          color, thickness, line_type)

            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            cv2.circle(result_image, (center_x, center_y), 3, color, -1)

            label_parts = [f'W_{track_id}']
            label_parts.append(f'({avg_conf:.2f})')

            if track_info.get('recovered', False):
                label_parts.append('[R]')

            consecutive_hits = track_info.get('consecutive_hits', 0)
            if consecutive_hits >= self.weed_tracker.min_hits:
                label_parts.append(f'[H{consecutive_hits}]')

            quality_score = track_info.get('quality_score', 0)
            if quality_score > 0:
                label_parts.append(f'[Q{quality_score:.1f}]')

            label = ' '.join(label_parts)

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_y = int(y - 5) if y > 30 else int(y + h + 20)

            # cv2.rectangle(result_image,
            #               (int(x), label_y - label_size[1] - 5),
            #               (int(x + label_size[0] + 5), label_y + 5),
            #               color, -1)
            #
            # text_color = (0, 0, 0) if color == (0, 255, 255) else (255, 255, 255)
            # cv2.putText(result_image, label, (int(x + 2), label_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # self.draw_statistics(result_image)
        return result_image

    def draw_results_ultralytics(self, image, plant_detections, tracked_weeds):
        result_image = image.copy()

        for plant_det, conf in plant_detections:
            x, y, w, h = plant_det
            cv2.rectangle(result_image, (int(x), int(y)), (int(x + w), int(y + h)),
                          (255, 255, 255), 2)
            label = f'Plant ({conf:.2f})'
            cv2.putText(result_image, label, (int(x), int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for track_id, bbox, avg_conf in tracked_weeds:
            x, y, w, h = bbox

            # 获取轨迹质量信息
            quality_info = self.get_ultralytics_track_quality(track_id)
            quality_score = quality_info['quality_score']
            consecutive_hits = quality_info['consecutive_hits']

            # 根据质量选择颜色
            # if quality_score > 0.6 and consecutive_hits >= 3:
            #     color = (0, 255, 255)  # 黄色 - 稳定轨迹
            #     thickness = 2
            # elif consecutive_hits >= 1:
            #     color = (0, 255, 0)  # 绿色 - 新轨迹
            #     thickness = 2
            # else:
            #     color = (255, 0, 0)  # 蓝色 - 不稳定
            #     thickness = 1

            color = (0, 255, 255)
            thickness = 2
            cv2.rectangle(result_image, (int(x), int(y)), (int(x + w), int(y + h)),
                          color, thickness)

            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            cv2.circle(result_image, (center_x, center_y), 3, color, -1)

            label_parts = [f'W_{track_id}']
            label_parts.append(f'({avg_conf:.2f})')
            #
            # if consecutive_hits >= 3:
            #     label_parts.append(f'[H{consecutive_hits}]')
            #
            # if quality_score > 0:
            #     label_parts.append(f'[Q{quality_score:.1f}]')

            label = ' '.join(label_parts)

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_y = int(y - 5) if y > 30 else int(y + h + 20)

            cv2.rectangle(result_image,
                          (int(x), label_y - label_size[1] - 5),
                          (int(x + label_size[0] + 5), label_y + 5),
                          color, -1)

            text_color = (0, 0, 0) if color == (0, 255, 255) else (255, 255, 255)
            cv2.putText(result_image, label, (int(x + 2), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        #左上角显示信息
        # self.draw_statistics(result_image)
        return result_image

    def draw_statistics(self, image):
        """在图像上绘制统计信息"""
        if self.use_custom_tracker:
            stats = self.weed_tracker.get_statistics()
            info_lines = [
                f"Model: {self.model_type.upper()}",
                f"Tracker: {self.tracker_type}",
                f"Frame: {self.frame_count}",
                f"Active Tracks: {stats['active_tracks']}",
                f"Reliable: {stats['reliable_tracks']}",
                f"Total Created: {stats['total_created']}",
                f"Recovered: {stats['total_recovered']}",
                f"Avg Det/Frame: {self.detection_stats['avg_detections_per_frame']:.1f}"
            ]
        else:
            info_lines = [
                f"Model: {self.model_type.upper()}",
                f"Tracker: {self.tracker_type.upper()}",
                f"Frame: {self.frame_count}",
                f"Active Tracks: {len(self.ultralytics_tracks)}",
                f"Avg Det/Frame: {self.detection_stats['avg_detections_per_frame']:.1f}"
            ]

        box_height = len(info_lines) * 25 + 10
        cv2.rectangle(image, (10, 10), (300, box_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (300, box_height), (255, 255, 255), 2)

        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(image, line, (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return

    def get_track_info(self):
        """获取当前跟踪信息（统一接口）"""
        if self.use_custom_tracker:
            # 自定义跟踪器的信息
            track_info = {}
            for track_id, track in self.weed_tracker.tracks.items():
                track_info[track_id] = {
                    'position': track['centroid'].tolist(),
                    'bbox': track['bbox'],
                    'confidence': track.get('avg_confidence', 0),
                    'consecutive_hits': track['consecutive_hits'],
                    'frames_skipped': track['frames_skipped'],
                    'quality_score': track.get('quality_score', 0),
                    'is_reliable': self.weed_tracker.is_reliable_track(track_id),
                    'total_hits': track.get('total_hits', 0),
                    'recovered': track.get('recovered', False)
                }
        else:
            # Ultralytics跟踪器的信息
            track_info = {}
            current_time = time.time()
            for track_id, track_data in self.ultralytics_tracks.items():
                # 清理超时的轨迹
                if current_time - track_data['last_seen'] > 5.0:
                    continue

                quality_info = self.get_ultralytics_track_quality(track_id)
                track_info[track_id] = {
                    'position': track_data['centroid'].tolist(),
                    'bbox': track_data['bbox'],
                    'confidence': quality_info['avg_confidence'],
                    'consecutive_hits': quality_info['consecutive_hits'],
                    'frames_skipped': 0,  # Ultralytics不提供此信息
                    'quality_score': quality_info['quality_score'],
                    'is_reliable': quality_info['quality_score'] > 0.4 and quality_info['consecutive_hits'] >= 3,
                    'total_hits': quality_info['consecutive_hits'],
                    'recovered': False  # Ultralytics不提供此信息
                }

        return track_info

    def get_reliable_weeds(self):
        """获取可靠的杂草轨迹"""
        reliable_weeds = []
        track_info = self.get_track_info()

        for track_id, info in track_info.items():
            if info['is_reliable']:
                reliable_weeds.append({
                    'id': track_id,
                    'bbox': info['bbox'],
                    'centroid': info['position'],
                    'confidence': info['confidence'],
                    'quality_score': info['quality_score']
                })

        reliable_weeds.sort(key=lambda x: x['id'])
        return reliable_weeds

    def reset_tracker(self):
        """重置跟踪器"""
        if self.use_custom_tracker:
            self.weed_tracker = WeedTracker(
                max_distance=80,
                max_frames_to_skip=20,
                min_hits=3,
                iou_threshold=0.2
            )
        else:
            # 重置Ultralytics跟踪器状态
            self.ultralytics_tracks = {}
            self.track_confidence_history = defaultdict(lambda: deque(maxlen=10))
            self.track_consecutive_hits = defaultdict(int)
            self.track_quality_scores = {}

        self.detection_history.clear()
        self.frame_count = 0
        # rospy.loginfo(f"Weed tracker reset for {self.tracker_type} tracker")

    def get_statistics(self):
        """获取完整统计信息"""
        if self.use_custom_tracker:
            tracker_stats = self.weed_tracker.get_statistics()
        else:
            tracker_stats = {
                'active_tracks': len(self.ultralytics_tracks),
                'tracker_type': self.tracker_type
            }

        return {
            'tracker': tracker_stats,
            'detection': self.detection_stats,
            'frame_count': self.frame_count,
            'model_type': self.model_type,
            'tracker_type': self.tracker_type
        }
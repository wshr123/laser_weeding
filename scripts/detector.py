import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO, RTDETR
import time
import rospy
from collections import defaultdict, deque


class WeedTracker:
    def __init__(self, max_distance=50, max_frames_to_skip=15, min_hits=3, iou_threshold=0.3):
        """
        初始化杂草跟踪器
        """
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

    def create_kalman_filter(self, x, y):
        """创建卡尔曼滤波器用于位置预测"""
        kf = cv2.KalmanFilter(4, 2)  # 4个状态量(x,y,vx,vy), 2个观测量(x,y)

        # 测量矩阵 - 确保是float32类型
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], dtype=np.float32)

        # 状态转移矩阵
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=np.float32)

        # 过程噪声协方差
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # 测量噪声协方差
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1

        # 后验误差协方差
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        # 初始状态 - 确保是float32类型
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
                rospy.loginfo(f"Recovering deleted track {deleted_id}")
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
        """
        更新跟踪器
        detections_with_conf: [(bbox, confidence), ...] 其中bbox格式为[x, y, w, h]
        """
        current_time = time.time()

        # 预测所有现有轨迹的位置
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            # 预测返回的是4x1矩阵，需要正确提取
            predicted = track['kalman'].predict()
            # 确保predicted_centroid是一维数组
            track['predicted_centroid'] = np.array([predicted[0, 0], predicted[1, 0]], dtype=np.float32)

        if len(detections_with_conf) == 0:
            # 没有新检测，只更新现有轨迹状态
            for track_id in list(self.tracks.keys()):
                track = self.tracks[track_id]
                track['frames_skipped'] += 1
                track['consecutive_hits'] = max(0, track['consecutive_hits'] - 1)

                if track['frames_skipped'] > self.max_frames_to_skip:
                    self.delete_track(track_id)

            return [(track_id, self.get_predicted_bbox(track), track.get('avg_confidence', 0))
                    for track_id, track in self.tracks.items()
                    if self.is_reliable_track(track_id) or track['frames_skipped'] <= 3]

        # 提取检测信息
        detections = [det[0] for det in detections_with_conf]
        confidences = [det[1] for det in detections_with_conf]

        # 计算检测框的质心 - 确保是float32类型
        detection_centroids = np.array([[bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                                        for bbox in detections], dtype=np.float32)

        if len(self.tracks) == 0:
            # 第一帧或无现有轨迹，创建新轨迹
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
            # 计算预测位置与检测位置的距离矩阵
            predicted_centroids = np.array([track['predicted_centroid']
                                            for track in self.tracks.values()], dtype=np.float32)
            track_ids = list(self.tracks.keys())

            if len(predicted_centroids) > 0 and len(detection_centroids) > 0:
                # 计算距离矩阵
                distances = cdist(predicted_centroids, detection_centroids)

                # 创建成本矩阵
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

                    # 处理匹配结果
                    for row_idx, col_idx in zip(row_indices, col_indices):
                        if cost_matrix[row_idx, col_idx] < 1e6:
                            track_id = track_ids[row_idx]
                            distance = distances[row_idx, col_idx]

                            adaptive_threshold = self.max_distance * (1 + self.tracks[track_id]['frames_skipped'] * 0.2)

                            if distance < adaptive_threshold:
                                # 更新轨迹
                                track = self.tracks[track_id]
                                centroid = detection_centroids[col_idx]

                                # 更新卡尔曼滤波器 - 确保输入是float32类型的2x1矩阵
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

                    # 处理未更新的轨迹
                    for track_id in track_ids:
                        if track_id not in updated_tracks:
                            track = self.tracks[track_id]
                            track['frames_skipped'] += 1
                            track['consecutive_hits'] = max(0, track['consecutive_hits'] - 1)

                            if track['frames_skipped'] > self.max_frames_to_skip:
                                self.delete_track(track_id)

                    # 创建新轨迹
                    for i, bbox in enumerate(detections):
                        if (i not in used_detection_indices and
                                confidences[i] > 0.3):

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

        # 返回稳定的轨迹
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


# WeedDetector类保持不变，只展示相关部分确保兼容性
class WeedDetector:
    def __init__(self, model_path, model_type='yolov8', weed_class_id=0,
                 crop_class_id=1, confidence_threshold=0.3, device='cuda:0'):
        """
        初始化杂草检测器，支持多种模型
        """
        self.model_type = model_type.lower()
        self.weed_class_id = weed_class_id
        self.crop_class_id = crop_class_id
        self.confidence_threshold = confidence_threshold
        self.device = device

        try:
            if self.model_type in ['yolov8', 'yolov11']:
                self.model = YOLO(model_path)
                rospy.loginfo(f"Successfully loaded {self.model_type.upper()} model from {model_path}")
            elif self.model_type == 'rtdetr':
                self.model = RTDETR(model_path)
                rospy.loginfo(f"Successfully loaded RT-DETR model from {model_path}")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            self.model.to(device)

        except Exception as e:
            rospy.logerr(f"Failed to load {self.model_type} model: {e}")
            raise

        # 初始化跟踪器
        self.weed_tracker = WeedTracker(
            max_distance=80,
            max_frames_to_skip=20,
            min_hits=3,
            iou_threshold=0.2
        )

        self.detection_history = deque(maxlen=5)
        self.frame_count = 0

        self.detection_stats = {
            'total_detections': 0,
            'total_frames': 0,
            'avg_detections_per_frame': 0,
            'model_type': self.model_type
        }

        self.input_size = (640, 640)

        rospy.loginfo(f"WeedDetector initialized with {self.model_type.upper()} model")

    # 其余方法保持不变...
    def preprocess_image(self, image):
        """预处理图像"""
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
        return enhanced_image

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

    def detect_plants_and_weeds(self, np_image):
        """检测植物和杂草"""
        processed_image = self.preprocess_image(np_image)

        try:
            results = self.model(processed_image, conf=self.confidence_threshold, device=self.device)

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
            conf = float(confidences[i])  # 确保是Python float
            cls = int(classes[i])

            # 转换为 [x, y, w, h] 格式，确保是float类型
            bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            width, height = x2 - x1, y2 - y1
            if width < 100 and height < 100:
                plant_detections.append((bbox_xywh, conf))
            weed_detections.append((bbox_xywh, conf))

            # if cls == self.crop_class_id and conf > 0.4:
            #     width, height = x2 - x1, y2 - y1
            #     if width < 100 and height < 100:
            #         plant_detections.append((bbox_xywh, conf))
            #
            # elif cls == self.weed_class_id and conf > self.confidence_threshold:
            #     weed_detections.append((bbox_xywh, conf))

        weed_detections = self.filter_detections(weed_detections, np_image.shape)
        plant_detections = self.filter_detections(plant_detections, np_image.shape)

        self.detection_stats['total_detections'] += len(weed_detections)
        self.detection_stats['total_frames'] += 1
        self.detection_stats['avg_detections_per_frame'] = (
                self.detection_stats['total_detections'] / self.detection_stats['total_frames']
        )

        return plant_detections, weed_detections

    def detect_and_track_weeds(self, np_image):
        """检测并跟踪杂草"""
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

        return det_image, tracked_weeds

    def draw_results(self, image, plant_detections, tracked_weeds):
        """绘制检测和跟踪结果"""
        result_image = image.copy()

        for plant_det, conf in plant_detections:
            x, y, w, h = plant_det
            cv2.rectangle(result_image, (int(x), int(y)), (int(x + w), int(y + h)),
                          (255, 255, 255), 2)

            label = f'Plant ({conf:.2f})'
            cv2.putText(result_image, label, (int(x), int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

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

            cv2.rectangle(result_image,
                          (int(x), label_y - label_size[1] - 5),
                          (int(x + label_size[0] + 5), label_y + 5),
                          color, -1)

            text_color = (0, 0, 0) if color == (0, 255, 255) else (255, 255, 255)
            cv2.putText(result_image, label, (int(x + 2), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        self.draw_statistics(result_image)

        return result_image

    def draw_statistics(self, image):
        """在图像上绘制统计信息"""
        stats = self.weed_tracker.get_statistics()
        detection_stats = self.detection_stats

        info_lines = [
            # f"Model: {self.model_type.upper()}",
            # f"Frame: {self.frame_count}",
            f"Active Tracks: {stats['active_tracks']}",
            f"Reliable: {stats['reliable_tracks']}",
            f"Total Created: {stats['total_created']}",
            f"Recovered: {stats['total_recovered']}",
            f"Avg Det/Frame: {detection_stats['avg_detections_per_frame']:.1f}"
        ]

        box_height = len(info_lines) * 15 + 10
        cv2.rectangle(image, (10, 10), (150, box_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (150, box_height), (255, 255, 255), 2)

        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(image, line, (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def get_track_info(self):
        """获取当前跟踪信息"""
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
        return track_info

    def get_reliable_weeds(self):
        """获取可靠的杂草轨迹"""
        reliable_weeds = []
        for track_id, track in self.weed_tracker.tracks.items():
            if self.weed_tracker.is_reliable_track(track_id):
                reliable_weeds.append({
                    'id': track_id,
                    'bbox': track['bbox'],
                    'centroid': track['centroid'].tolist(),
                    'confidence': track.get('avg_confidence', 0),
                    'quality_score': track.get('quality_score', 0)
                })

        reliable_weeds.sort(key=lambda x: x['id'])
        return reliable_weeds

    def reset_tracker(self):
        """重置跟踪器"""
        self.weed_tracker = WeedTracker(
            max_distance=80,
            max_frames_to_skip=20,
            min_hits=3,
            iou_threshold=0.2
        )
        self.detection_history.clear()
        self.frame_count = 0
        rospy.loginfo(f"Weed tracker reset for {self.model_type.upper()} detector")

    def get_statistics(self):
        """获取完整统计信息"""
        tracker_stats = self.weed_tracker.get_statistics()
        return {
            'tracker': tracker_stats,
            'detection': self.detection_stats,
            'frame_count': self.frame_count,
            'model_type': self.model_type
        }
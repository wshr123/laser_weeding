import pyrealsense2 as rs
import numpy as np
import cv2
import math

#This code is for testing the black circle detection method

class CircleTrack:
    """单个圆圈的追踪状态类"""
    _id_counter = 0

    def __init__(self, center, radius, area, circularity, solidity):
        self.id = CircleTrack._id_counter
        CircleTrack._id_counter += 1

        self.center = center  # (x, y)
        self.radius = radius

        # 追踪状态
        self.hits = 1  # 连续命中次数
        self.misses = 0  # 丢失帧数
        self.active = False  # 是否确认为有效目标 (hits >= 3)

        # 记录属性用于调试或平滑
        self.area = area

    def update(self, new_center, new_radius):
        """
        步骤5: EMA权重更新 (0.75旧 + 0.25新)
        """
        alpha = 0.25  # 新值的权重

        # 更新位置
        self.center = (
            self.center[0] * (1 - alpha) + new_center[0] * alpha,
            self.center[1] * (1 - alpha) + new_center[1] * alpha
        )

        # 更新半径
        self.radius = self.radius * (1 - alpha) + new_radius * alpha

        self.hits += 1
        self.misses = 0

        # 连续命中阈值：3次
        if self.hits >= 3:
            self.active = True

    def mark_missed(self):
        self.misses += 1


class BlackCircleDetector:
    def __init__(self):
        self.tracks = []

        # 步骤2: 形态学核 5x5
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect(self, color_image):
        """
        执行步骤1-4：检测当前帧中的候选圆
        """
        # -------------------------------------------------
        # 步骤1: 方法B HSV + V通道法
        # -------------------------------------------------
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # V(亮度) < 80, S(饱和度) < 100
        # OpenCV中 H:0-180, S:0-255, V:0-255
        # 黑色范围: H任意, S较低, V较低
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 100, 80])

        mask = cv2.inRange(hsv, lower_black, upper_black)

        # -------------------------------------------------
        # 步骤2: 形态学去噪 (开运算 -> 闭运算)
        # -------------------------------------------------
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # -------------------------------------------------
        # 步骤3: 形状过滤
        # -------------------------------------------------
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            # 1. 面积过滤 (最小200)
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            # 计算外接圆 (用于半径过滤)
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            # 5. 半径过滤 (10-150像素)
            if not (10 <= radius <= 150):
                continue

            # 2. 圆度过滤 (>= 0.6)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.6:
                continue

            # 3. 填充度/实心度 (>= 0.65)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = area / hull_area
            if solidity < 0.65:
                continue

            # 4. 椭圆比过滤 (>= 0.65)
            # fitEllipse 需要至少5个点
            if len(cnt) < 5:
                # 点太少通常不是好的圆，或者可以用外接矩形代替
                continue

            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            MA, ma = axes  # Minor Axis, Major Axis
            if ma > 0:
                aspect_ratio = MA / ma
                if aspect_ratio < 0.65:
                    continue
            else:
                continue

            # 计算综合得分 (用于NMS)
            # 0.5*area + 0.3*circularity + 0.2*fill
            # 注意：Area数值很大，Circularity是0-1。
            # 为了让排序合理，这里假设是在同量级比较，或者用户希望大面积优先
            score = 0.5 * area + 0.3 * circularity * 1000 + 0.2 * solidity * 1000

            candidates.append({
                'center': (int(x), int(y)),
                'radius': radius,
                'area': area,
                'circularity': circularity,
                'solidity': solidity,
                'score': score
            })

        # -------------------------------------------------
        # 步骤4: NMS (非极大值抑制)
        # -------------------------------------------------
        # 按分数降序排列
        candidates.sort(key=lambda x: x['score'], reverse=True)

        final_detections = []
        nms_threshold = 35  # 最小圆心距 35像素

        while len(candidates) > 0:
            current = candidates.pop(0)
            final_detections.append(current)

            # 移除距离当前圆太近的其他候选者
            remaining = []
            for other in candidates:
                dist = math.sqrt((current['center'][0] - other['center'][0]) ** 2 +
                                 (current['center'][1] - other['center'][1]) ** 2)
                if dist >= nms_threshold:
                    remaining.append(other)
            candidates = remaining

        return final_detections, mask  # 返回mask仅供调试显示

    def update_tracks(self, detections):
        """
        步骤5: 时间一致性跟踪
        """
        # 简单的贪婪匹配 (可以用匈牙利算法，但贪婪匹配对于帧率高的情况通常足够)
        match_dist_threshold = 40  # 帧间匹配距离

        matched_track_indices = set()
        matched_detection_indices = set()

        # 尝试将检测结果匹配到现有的轨迹
        for det_idx, det in enumerate(detections):
            best_dist = float('inf')
            best_track_idx = -1

            cx, cy = det['center']

            for trk_idx, track in enumerate(self.tracks):
                if trk_idx in matched_track_indices:
                    continue

                dist = math.sqrt((cx - track.center[0]) ** 2 + (cy - track.center[1]) ** 2)

                if dist < match_dist_threshold and dist < best_dist:
                    best_dist = dist
                    best_track_idx = trk_idx

            if best_track_idx != -1:
                # 匹配成功：更新轨迹
                self.tracks[best_track_idx].update(det['center'], det['radius'])
                matched_track_indices.add(best_track_idx)
                matched_detection_indices.add(det_idx)
            else:
                # 无匹配：稍后创建新轨迹
                pass

        # 处理未匹配的检测 -> 新建轨迹
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detection_indices:
                new_track = CircleTrack(
                    det['center'],
                    det['radius'],
                    det['area'],
                    det['circularity'],
                    det['solidity']
                )
                self.tracks.append(new_track)

        # 处理未匹配的轨迹 -> 增加丢失计数
        # ID生命周期管理：超过10帧未见才删除
        max_misses_for_deletion = 10

        active_tracks = []
        for trk_idx, track in enumerate(self.tracks):
            if trk_idx not in matched_track_indices:
                track.mark_missed()

            if track.misses <= max_misses_for_deletion:
                active_tracks.append(track)

        self.tracks = active_tracks


def main():
    # -------------------------------------------------
    # RealSense 初始化
    # -------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用彩色流 (640x480 @ 30fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 开始采集
    print("Starting RealSense...")
    pipeline.start(config)

    detector = BlackCircleDetector()

    try:
        while True:
            # 等待一帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 转为numpy数组
            color_image = np.asanyarray(color_frame.get_data())

            # 1. 检测
            detections, binary_mask = detector.detect(color_image)

            # 2. 追踪更新
            detector.update_tracks(detections)

            # 3. 绘制结果
            display_img = color_image.copy()

            # 绘制处理过的二值图（画中画，方便调试）
            mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.resize(mask_bgr, (160, 120))
            display_img[0:120, 0:160] = mask_bgr
            cv2.putText(display_img, "Mask", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            for track in detector.tracks:
                # 丢失缓冲：5帧 (只有丢失小于5帧的才显示，虽然内存里保留到10帧)
                if track.misses < 5:

                    # 只有连续命中3次以上的才画实线，否则画虚线或不同颜色表示“确认中”
                    if track.active:
                        color = (0, 255, 0)  # 绿色：确认目标
                        thickness = 2
                    else:
                        color = (0, 255, 255)  # 黄色：获取中
                        thickness = 1

                    cx, cy = int(track.center[0]), int(track.center[1])
                    r = int(track.radius)

                    # 画圆
                    cv2.circle(display_img, (cx, cy), r, color, thickness)
                    # 画圆心
                    cv2.circle(display_img, (cx, cy), 2, (0, 0, 255), -1)

                    # 显示ID和状态
                    label = f"ID:{track.id}"
                    if track.misses > 0:
                        label += f" (Lost:{track.misses})"
                    cv2.putText(display_img, label, (cx - 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow('RealSense Black Circle Detection', display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
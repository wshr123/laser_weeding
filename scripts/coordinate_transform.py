#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import yaml
import cv2
from scipy.spatial.transform import Rotation
import os


class CameraGalvoTransform:
    """
    相机-振镜坐标变换类
    支持两种模式：
    1) 3D几何变换：像素 → 相机光线/深度 → 振镜系点 → 振镜角 → 码值
       反向：码值 → 振镜角 → 振镜射线 → 选择相同深度面/固定深度面/工作平面 → 相机像素
    2) 简单线性映射（回退方案）
    """

    def __init__(self, config_file=None, use_3d_transform=True):
        rospy.loginfo("Initializing camera-galvo coordinate transformer...")

        self.use_3d_transform = use_3d_transform

        self.default_params = {
            'transform_mode': {
                'use_3d_transform': use_3d_transform,
                'fallback_to_simple': True
            },
            'camera_matrix': {
                'fx': 500.0,
                'fy': 500.0,
                'cx': 320.0,
                'cy': 240.0
            },
            'distortion_coeffs': [],
            'extrinsics': {
                't_gc': [0.0, 100.0, -50.0],  # 相机在振镜坐标系的位置（mm）
                'q_gc': [0.0, 0.0, 0.0, 1.0]  # 相机→振镜的旋转（xyzw）
            },
            'work_plane': {
                'n_g': [0.0, 0.0, 1.0],  # 振镜系下工作平面法向
                'd_g': -200.0           # 平面方程 n·X + d = 0
            },
            'galvo_params': {
                'scan_angle': 30.0,  # 总扫描角（deg）
                'scale_x': 1.0,
                'scale_y': 1.0,
                'bias_x': 0.0,
                'bias_y': 0.0,
                'max_code': 32767
            },
            'simple_mapping': {
                'use_safe_range': True,
                'max_safe_range': 32767,
                'protocol_max': 32767,
                'scale_factor': 65536,
                'offset_x': 0,
                'offset_y': 0,
            }
        }

        self.load_config(config_file)

        if self.use_3d_transform:
            self.init_3d_transform()

        self.init_simple_mapping()

        # 运行态变量
        self.last_pixel_pos = None
        self.last_galvo_pos = None
        self.transform_valid = True
        self.transform_method_used = "Unknown"
        self.transform_fail_count = 0

        # 深度查询回调
        self.depth_query_func = None

        # 反向回投时优先使用的“上一命中点”（来自像素+深度的正向分支）
        self.last_hit_point_g = None  # 振镜系 [x,y,z] mm
        # 可选的固定深度面（振镜系 z = const，单位 mm）
        self.fixed_reverse_depth_z_mm = None

        rospy.loginfo("Camera-galvo coordinate transformer initialized successfully")

    # -------------------- 参数与初始化 --------------------

    def load_config(self, config_file):
        self.params = self.default_params.copy()

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self.update_params_recursive(self.params, config)

                if 'transform_mode' in config:
                    self.use_3d_transform = config['transform_mode'].get('use_3d_transform', self.use_3d_transform)
            except Exception as e:
                rospy.logwarn(f"Failed to load config file, using default parameters: {e}")
        else:
            rospy.loginfo("Using default configuration parameters")

    def update_params_recursive(self, default_dict, update_dict):
        for key, value in update_dict.items():
            if key in default_dict and isinstance(default_dict[key], dict) and isinstance(value, dict):
                self.update_params_recursive(default_dict[key], value)
            else:
                default_dict[key] = value

    def init_3d_transform(self):
        try:
            self.build_camera_matrix()
            self.build_extrinsics()
            self.build_work_plane()
            self.transform_3d_initialized = True
        except Exception as e:
            rospy.logerr(f"Failed to initialize 3D transform components: {e}")
            self.transform_3d_initialized = False

    def init_simple_mapping(self):
        simple = self.params['simple_mapping']
        self.simple_use_safe_range = simple['use_safe_range']
        self.simple_max_safe_range = simple['max_safe_range']
        self.simple_protocol_max = simple['protocol_max']
        self.simple_scale_factor = simple['scale_factor']
        self.simple_offset_x = simple['offset_x']
        self.simple_offset_y = simple['offset_y']

    def build_camera_matrix(self):
        cam = self.params['camera_matrix']
        self.K = np.array([
            [cam['fx'], 0, cam['cx']],
            [0, cam['fy'], cam['cy']],
            [0, 0, 1]
        ], dtype=np.float64)

        self.D = np.array(self.params['distortion_coeffs'], dtype=np.float64)
        self.use_distortion = len(self.D) > 0

    def build_extrinsics(self):
        ext = self.params['extrinsics']
        self.t_gc = np.array(ext['t_gc'], dtype=np.float64)
        q = ext['q_gc']  # [x, y, z, w]
        self.R_gc = Rotation.from_quat(q).as_matrix()
        # 约定：p_g = R_gc * p_c + t_gc

    def build_work_plane(self):
        plane = self.params['work_plane']
        self.n_g = np.array(plane['n_g'], dtype=np.float64)
        self.n_g = self.n_g / np.linalg.norm(self.n_g)
        self.d_g = float(plane['d_g'])

    # -------------------- 公共 API --------------------

    def set_depth_query(self, func):
        """
        设置像素深度查询回调。
        func(u, v) -> depth_in_meters or None
        """
        self.depth_query_func = func

    def set_fixed_depth_for_reverse(self, z_mm):
        """
        设置反向回投时的固定深度面（振镜坐标系 z = z_mm）。
        若设为 None，则不使用固定深度优先级。
        """
        self.fixed_reverse_depth_z_mm = z_mm

    def pixel_to_galvo_code(self, pixel_x, pixel_y, image_width=640, image_height=480):
        """
        正向：像素 -> 码值
        优先 3D；失败时按配置回退到简单映射
        """
        try:
            if self.use_3d_transform and getattr(self, 'transform_3d_initialized', False):
                result = self.pixel_to_galvo_3d(pixel_x, pixel_y, image_width, image_height)
                if result is not None:
                    self.transform_method_used = "3D geometric transform"
                    self.transform_valid = True
                    self.transform_fail_count = 0
                    return result
                else:
                    self.transform_fail_count += 1
                    if self.transform_fail_count <= 5:
                        rospy.logwarn(f"3D transform failed (count: {self.transform_fail_count}), trying fallback")

                    if self.params['transform_mode']['fallback_to_simple']:
                        result = self.pixel_to_galvo_simple(pixel_x, pixel_y, image_width, image_height)
                        self.transform_method_used = "Simple mapping (fallback)"
                        self.transform_valid = True
                        return result
                    else:
                        self.transform_valid = False
                        return None
            else:
                result = self.pixel_to_galvo_simple(pixel_x, pixel_y, image_width, image_height)
                self.transform_method_used = "Simple mapping"
                self.transform_valid = True
                return result

        except Exception as e:
            rospy.logerr(f"Coordinate transform failed: {e}")
            self.transform_valid = False
            try:
                result = self.pixel_to_galvo_simple(pixel_x, pixel_y, image_width, image_height)
                self.transform_method_used = "Simple mapping (exception fallback)"
                self.transform_valid = True
                return result
            except:
                return None

    def galvo_code_to_pixel(self, galvo_x, galvo_y, image_width=640, image_height=480):
        """
        反向：码值 -> 像素
        优先 3D；失败时回退到简单映射
        """
        try:
            if self.use_3d_transform and getattr(self, 'transform_3d_initialized', False):
                return self.galvo_code_to_pixel_3d(galvo_x, galvo_y, image_width, image_height)
            else:
                return self.galvo_code_to_pixel_simple(galvo_x, galvo_y, image_width, image_height)
        except Exception as e:
            rospy.logdebug(f"Reverse coordinate transform failed: {e}")
            return self.galvo_code_to_pixel_simple(galvo_x, galvo_y, image_width, image_height)

    # -------------------- 正向 3D --------------------

    def pixel_depth_to_point_galvo(self, pixel_x, pixel_y, depth_m):
        """
        使用像素与深度直接计算振镜坐标系中的3D点。
        depth_m: 以米为单位的深度（相机坐标系下Z>0）。
        返回: 3D点 [x_g, y_g, z_g]（毫米），与 t_gc 单位一致。
        """
        try:
            if depth_m is None or depth_m <= 0:
                return None

            x_c = (pixel_x - self.K[0, 2]) / self.K[0, 0] * depth_m
            y_c = (pixel_y - self.K[1, 2]) / self.K[1, 1] * depth_m
            z_c = depth_m

            p_c_mm = np.array([x_c * 1000.0, y_c * 1000.0, z_c * 1000.0], dtype=np.float64)
            p_g = self.R_gc @ p_c_mm + self.t_gc
            return p_g
        except Exception:
            return None

    def pixel_to_camera_ray(self, pixel_x, pixel_y):
        try:
            if self.use_distortion:
                # 如需去畸变，这里添加畸变模型反解
                pass
            x = (pixel_x - self.K[0, 2]) / self.K[0, 0]
            y = (pixel_y - self.K[1, 2]) / self.K[1, 1]
            ray_dir = np.array([x, y, 1.0], dtype=np.float64)
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            return ray_dir
        except Exception as e:
            rospy.logdebug(f"Failed to convert pixel to camera ray: {e}")
            return None

    def ray_plane_intersection(self, ray_origin, ray_direction):
        try:
            denominator = np.dot(self.n_g, ray_direction)
            if abs(denominator) < 1e-6:
                return None

            t = -(np.dot(self.n_g, ray_origin) + self.d_g) / denominator
            if t <= 0:
                return None

            intersection = ray_origin + t * ray_direction
            return intersection
        except Exception:
            return None

    def point_to_galvo_angles(self, point):
        x, y, z = point
        if abs(z) < 1e-9:
            return 0.0, 0.0
        theta_x = np.arctan2(x, z)
        theta_y = np.arctan2(y, z)
        return theta_x, theta_y

    def angles_to_codes(self, theta_x, theta_y):
        """
        角度（弧度）→ 码值。仅在3D路径中使用。
        """
        galvo = self.params['galvo_params']
        theta_x_deg = np.degrees(theta_x)
        theta_y_deg = np.degrees(theta_y)

        theta_x_corrected = theta_x_deg * galvo['scale_x'] + galvo['bias_x']
        theta_y_corrected = theta_y_deg * galvo['scale_y'] + galvo['bias_y']

        half_scan_angle = galvo['scan_angle'] / 2.0  # deg

        norm_x = theta_x_corrected / half_scan_angle
        norm_y = theta_y_corrected / half_scan_angle

        code_x = norm_x * galvo['max_code']
        code_y = norm_y * galvo['max_code']

        max_code = galvo['max_code']
        saturated = (abs(code_x) > max_code) or (abs(code_y) > max_code)
        if saturated:
            rospy.logwarn_throttle(1.0, "angles_to_codes saturated; visualization/roundtrip may be inaccurate")

        code_x = np.clip(code_x, -max_code, max_code)
        code_y = np.clip(code_y, -max_code, max_code)

        return code_x, code_y

    def pixel_to_galvo_3d(self, pixel_x, pixel_y, image_width, image_height):
        try:
            # 优先使用深度
            if callable(getattr(self, 'depth_query_func', None)):
                depth_m = self.depth_query_func(pixel_x, pixel_y)
                if depth_m is not None and depth_m > 0:
                    point_g = self.pixel_depth_to_point_galvo(pixel_x, pixel_y, depth_m)
                    if point_g is not None:
                        # 缓存命中点供反向回投使用（保持一致的深度面）
                        self.last_hit_point_g = point_g.copy()

                        theta_x, theta_y = self.point_to_galvo_angles(point_g)
                        code_x, code_y = self.angles_to_codes(theta_x, theta_y)

                        self.last_pixel_pos = (pixel_x, pixel_y)
                        self.last_galvo_pos = (code_x, code_y)
                        return (int(code_x), int(code_y))
            # 无深度或深度失败，则使用射线-工作平面交
            ray_dir_camera = self.pixel_to_camera_ray(pixel_x, pixel_y)
            if ray_dir_camera is None:
                return None

            ray_origin_galvo = self.t_gc
            ray_dir_galvo = self.R_gc @ ray_dir_camera

            intersection_point = self.ray_plane_intersection(ray_origin_galvo, ray_dir_galvo)
            if intersection_point is None:
                return None

            # 也可以缓存，以便反向回投使用（但与深度得到的面不同）
            self.last_hit_point_g = intersection_point.copy()

            theta_x, theta_y = self.point_to_galvo_angles(intersection_point)
            code_x, code_y = self.angles_to_codes(theta_x, theta_y)

            self.last_pixel_pos = (pixel_x, pixel_y)
            self.last_galvo_pos = (code_x, code_y)

            return (int(code_x), int(code_y))

        except Exception as e:
            rospy.logdebug(f"3D transform exception: {e}")
            return None

    def pixel_to_galvo_simple(self, pixel_x, pixel_y, image_width, image_height):
        norm_x = (pixel_x / image_width - 0.5)
        norm_y = (pixel_y / image_height - 0.5)

        if self.simple_use_safe_range:
            max_range = self.simple_max_safe_range
            scale_factor = max_range * 2
        else:
            max_range = self.simple_protocol_max
            scale_factor = max_range * 2

        galvo_x = norm_x * scale_factor + self.simple_offset_x
        galvo_y = norm_y * scale_factor + self.simple_offset_y

        galvo_x = max(-max_range, min(max_range, galvo_x))
        galvo_y = max(-max_range, min(max_range, galvo_y))

        self.last_pixel_pos = (pixel_x, pixel_y)
        self.last_galvo_pos = (galvo_x, galvo_y)

        return (int(galvo_x), int(galvo_y))

    # -------------------- 反向 3D（使用深度面优先） --------------------

    def codes_to_angles(self, code_x, code_y):
        """
        码值 → 角度（弧度）。与 angles_to_codes 互逆（在未饱和前提下）。
        """
        galvo = self.params['galvo_params']
        max_code = galvo['max_code']

        norm_x = code_x / max_code
        norm_y = code_y / max_code

        half_scan_angle = galvo['scan_angle'] / 2.0  # deg

        theta_x_corrected = norm_x * half_scan_angle
        theta_y_corrected = norm_y * half_scan_angle

        theta_x_deg = (theta_x_corrected - galvo['bias_x']) / galvo['scale_x']
        theta_y_deg = (theta_y_corrected - galvo['bias_y']) / galvo['scale_y']

        theta_x = np.radians(theta_x_deg)
        theta_y = np.radians(theta_y_deg)

        return theta_x, theta_y

    def galvo_angles_to_point_on_plane(self, theta_x, theta_y):
        """
        用工作平面求交点。
        """
        try:
            x = np.tan(theta_x)
            y = np.tan(theta_y)
            z = 1.0
            dir_g = np.array([x, y, z], dtype=np.float64)
            dir_g /= np.linalg.norm(dir_g)
            org_g = np.zeros(3, dtype=np.float64)
            return self.ray_plane_intersection(org_g, dir_g)
        except Exception:
            return None

    def galvo_angles_to_point_at_depth(self, theta_x, theta_y, z_ref_mm):
        """
        用固定深度面 z = z_ref_mm（振镜系）求交点。
        """
        try:
            x = np.tan(theta_x)
            y = np.tan(theta_y)
            z = 1.0
            dir_g = np.array([x, y, z], dtype=np.float64)
            dir_g /= np.linalg.norm(dir_g)
            org_g = np.zeros(3, dtype=np.float64)

            if abs(dir_g[2]) < 1e-9:
                return None
            t = (z_ref_mm - org_g[2]) / dir_g[2]
            if t <= 0:
                return None
            return org_g + t * dir_g
        except Exception:
            return None

    def choose_reverse_intersection(self, theta_x, theta_y):
        """
        反向回投的交点选择：
        1) 如果有 last_hit_point_g，优先在相同 z 面上取交点（保持与正向深度一致）
        2) 如果配置了固定深度 fixed_reverse_depth_z_mm，使用它
        3) 否则回退到工作平面
        返回：intersection_point_g 或 None
        """
        # 1) 上一次正向命中的深度（来自像素+深度路径或平面交）
        if isinstance(self.last_hit_point_g, np.ndarray) and self.last_hit_point_g.shape == (3,):
            z_ref = float(self.last_hit_point_g[2])
            p = self.galvo_angles_to_point_at_depth(theta_x, theta_y, z_ref)
            if p is not None:
                return p

        # 2) 固定深度
        if self.fixed_reverse_depth_z_mm is not None:
            p = self.galvo_angles_to_point_at_depth(theta_x, theta_y, float(self.fixed_reverse_depth_z_mm))
            if p is not None:
                return p

        # 3) 工作平面
        return self.galvo_angles_to_point_on_plane(theta_x, theta_y)

    def point_to_camera_ray(self, point_g):
        """
        振镜系点 → 相机系视线方向
        """
        try:
            camera_to_point = point_g - self.t_gc
            camera_to_point = camera_to_point / np.linalg.norm(camera_to_point)
            ray_dir_camera = self.R_gc.T @ camera_to_point
            return ray_dir_camera
        except Exception:
            return None

    def camera_ray_to_pixel(self, ray_direction, image_width, image_height):
        try:
            if abs(ray_direction[2]) < 1e-9:
                return None

            x = ray_direction[0] / ray_direction[2]
            y = ray_direction[1] / ray_direction[2]

            pixel_x = x * self.K[0, 0] + self.K[0, 2]
            pixel_y = y * self.K[1, 1] + self.K[1, 2]

            # 如需加畸变投影，在此处处理
            return (pixel_x, pixel_y)
        except Exception:
            return None

    def galvo_code_to_pixel_3d(self, galvo_x, galvo_y, image_width, image_height):
        """
        使用“深度优先”的反向回投：码值 -> 角度 -> 选择交点（同深度/固定深度/工作平面）-> 投影到像素
        """
        try:
            theta_x, theta_y = self.codes_to_angles(galvo_x, galvo_y)

            # 若码值饱和，反向不可能严格还原；提示但继续计算
            max_code = self.params['galvo_params']['max_code']
            if abs(galvo_x) >= max_code - 1 or abs(galvo_y) >= max_code - 1:
                rospy.logwarn_throttle(1.0, "galvo_code appears saturated; reverse projection may be inaccurate")

            intersection_point = self.choose_reverse_intersection(theta_x, theta_y)
            if intersection_point is None:
                return None

            ray_dir_camera = self.point_to_camera_ray(intersection_point)
            if ray_dir_camera is None:
                return None

            pixel = self.camera_ray_to_pixel(ray_dir_camera, image_width, image_height)
            return pixel

        except Exception as e:
            rospy.logdebug(f"3D reverse transform exception: {e}")
            return None

    def galvo_code_to_pixel_simple(self, galvo_x, galvo_y, image_width, image_height):
        if self.simple_use_safe_range:
            max_range = self.simple_max_safe_range
            scale_factor = max_range * 2
        else:
            max_range = self.simple_protocol_max
            scale_factor = max_range * 2

        galvo_x_centered = galvo_x - self.simple_offset_x
        galvo_y_centered = galvo_y - self.simple_offset_y

        norm_x = galvo_x_centered / scale_factor
        norm_y = galvo_y_centered / scale_factor

        pixel_x = (norm_x + 0.5) * image_width
        pixel_y = (norm_y + 0.5) * image_height

        return (pixel_x, pixel_y)

    # -------------------- 模式切换/工具 --------------------

    def switch_transform_mode(self, use_3d_transform):
        old_mode = "3D geometric transform" if self.use_3d_transform else "Simple linear mapping"
        new_mode = "3D geometric transform" if use_3d_transform else "Simple linear mapping"

        self.use_3d_transform = use_3d_transform
        self.params['transform_mode']['use_3d_transform'] = use_3d_transform

        if use_3d_transform and not getattr(self, 'transform_3d_initialized', False):
            self.init_3d_transform()

        self.transform_fail_count = 0

        rospy.loginfo(f"Coordinate transform mode switched: {old_mode} -> {new_mode}")

    def update_camera_matrix(self, camera_info_msg):
        try:
            K_flat = camera_info_msg.K
            self.K = np.array(K_flat).reshape(3, 3)
            self.D = np.array(camera_info_msg.D)
            self.use_distortion = len(self.D) > 0

            self.params['camera_matrix']['fx'] = self.K[0, 0]
            self.params['camera_matrix']['fy'] = self.K[1, 1]
            self.params['camera_matrix']['cx'] = self.K[0, 2]
            self.params['camera_matrix']['cy'] = self.K[1, 2]
            self.params['distortion_coeffs'] = self.D.tolist()

            rospy.loginfo("Camera intrinsics updated from CameraInfo message")

        except Exception as e:
            rospy.logwarn(f"Failed to update camera intrinsics from CameraInfo: {e}")

    def calibrate_with_points(self, pixel_points, galvo_points):
        # 这里保留接口；实际标定流程视你的需求实现
        if len(pixel_points) != len(galvo_points) or len(pixel_points) < 4:
            rospy.logerr("Calibration requires at least 4 corresponding points")
            return False
        try:
            rospy.loginfo(f"Calibrating with {len(pixel_points)} point pairs (placeholder)")
            return True
        except Exception as e:
            rospy.logerr(f"Calibration failed: {e}")
            return False

    def save_config(self, config_file):
        try:
            with open(config_file, 'w') as f:
                yaml.dump(self.params, f, default_flow_style=False)
            rospy.loginfo(f"Configuration saved to: {config_file}")
            return True
        except Exception as e:
            rospy.logerr(f"Failed to save configuration: {e}")
            return False

    def get_transform_info(self):
        info = {
            'last_pixel_pos': self.last_pixel_pos,
            'last_galvo_pos': self.last_galvo_pos,
            'transform_valid': self.transform_valid,
            'transform_method': self.transform_method_used,
            'use_3d_transform': self.use_3d_transform,
            'transform_fail_count': getattr(self, 'transform_fail_count', 0),
            'galvo_params': self.params['galvo_params'],
            'simple_mapping': self.params['simple_mapping'],
            'fixed_reverse_depth_z_mm': self.fixed_reverse_depth_z_mm
        }

        if self.use_3d_transform and hasattr(self, 't_gc'):
            info.update({
                'camera_position': self.t_gc.tolist(),
                'work_plane_distance': self.d_g,
                'camera_matrix': self.K.tolist() if hasattr(self, 'K') else None,
                'transform_3d_initialized': getattr(self, 'transform_3d_initialized', False)
            })

        return info


# 示例配置（可选）
EXAMPLE_CONFIG = """
transform_mode:
  use_3d_transform: true
  fallback_to_simple: true

camera_matrix:
  fx: 500.0
  fy: 500.0
  cx: 320.0
  cy: 240.0

distortion_coeffs: []

extrinsics:
  t_gc: [0.0, 100.0, -50.0]
  q_gc: [0.0, 0.0, 0.0, 1.0]

work_plane:
  n_g: [0.0, 0.0, 1.0]
  d_g: -200.0

galvo_params:
  scan_angle: 20.0
  scale_x: 1.0
  scale_y: 1.0
  bias_x: 0.0
  bias_y: 0.0
  max_code: 30000

simple_mapping:
  use_safe_range: true
  max_safe_range: 30000
  protocol_max: 32767
  scale_factor: 60000
  offset_x: 0
  offset_y: 0
"""

def create_example_config(file_path):
    with open(file_path, 'w') as f:
        f.write(EXAMPLE_CONFIG)
    print(f"Example configuration file created: {file_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create_config":
        config_path = sys.argv[2] if len(sys.argv) > 2 else "camera_galvo_config.yaml"
        create_example_config(config_path)
    else:
        print("Testing 3D geometric transform with depth-aware reverse...")
        t = CameraGalvoTransform(use_3d_transform=True)
        # 可选：固定反向深度
        # t.set_fixed_depth_for_reverse(241.0)
        test_points = [(320, 240), (100, 100), (540, 380)]
        for px, py in test_points:
            res = t.pixel_to_galvo_code(px, py)
            print("p2g:", (px, py), "->", res)
            if res:
                uv = t.galvo_code_to_pixel(res[0], res[1])
                print("g2p:", res, "->", uv)
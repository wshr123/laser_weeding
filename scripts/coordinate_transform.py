# !/usr/bin/env python
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
    1. 3D几何变换：像素 → 相机光线 → 地面交点 → 振镜角度 → 码值
       （可选：像素+深度 → 相机点 → 振镜点，优先使用）
    2. 简单线性映射：像素直接线性映射到振镜码值（原始方法）
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
                't_gc': [0.0, 100.0, -50.0],
                'q_gc': [0.0, 0.0, 0.0, 1.0]
            },
            'work_plane': {
                'n_g': [0.0, 0.0, 1.0],
                'd_g': -200.0
            },
            'galvo_params': {
                'scan_angle': 30.0,
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

        self.last_pixel_pos = None
        self.last_galvo_pos = None
        self.transform_valid = True
        self.transform_method_used = "Unknown"
        self.transform_fail_count = 0

        # 可选的深度查询函数
        self.depth_query_func = None

        # 新增：Y轴符号控制（影响3D路径的角度↔码值）
        # 默认翻转为 True（-1.0），根据你的现象先这样；可通过 set_y_axis_invert 配置
        self.y_sign = -1.0

        rospy.loginfo("Camera-galvo coordinate transformer initialized successfully")

    def set_y_axis_invert(self, invert):
        """
        设置Y轴是否翻转，仅作用于3D路径角度↔码值互换，确保正反一致。
        invert=True -> y_sign = -1.0; invert=False -> y_sign = +1.0
        """
        self.y_sign = -1.0 if bool(invert) else +1.0

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

        rospy.logdebug(f"Camera matrix K:\n{self.K}")

    def build_extrinsics(self):
        ext = self.params['extrinsics']
        self.t_gc = np.array(ext['t_gc'], dtype=np.float64)
        q = ext['q_gc']  # [x, y, z, w]
        self.R_gc = Rotation.from_quat(q).as_matrix()
        rospy.logdebug(f"Camera position t_gc: {self.t_gc}")
        rospy.logdebug(f"Camera orientation R_gc:\n{self.R_gc}")

    def build_work_plane(self):
        plane = self.params['work_plane']
        self.n_g = np.array(plane['n_g'], dtype=np.float64)
        self.n_g = self.n_g / np.linalg.norm(self.n_g)
        self.d_g = float(plane['d_g'])
        rospy.logdebug(f"Work plane normal: {self.n_g}")
        rospy.logdebug(f"Work plane distance: {self.d_g}")

    def pixel_to_galvo_code(self, pixel_x, pixel_y, image_width=640, image_height=480):
        try:
            if self.use_3d_transform and hasattr(self, 'transform_3d_initialized') and self.transform_3d_initialized:
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

    def set_depth_query(self, func):
        """
        设置像素深度查询回调。
        func(u, v) -> depth_in_meters or None
        """
        self.depth_query_func = func

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

    def pixel_to_galvo_3d(self, pixel_x, pixel_y, image_width, image_height):
        try:
            # 优先使用深度
            if hasattr(self, 'depth_query_func') and callable(self.depth_query_func):
                depth_m = self.depth_query_func(pixel_x, pixel_y)
                if depth_m is not None and depth_m > 0:
                    point_g = self.pixel_depth_to_point_galvo(pixel_x, pixel_y, depth_m)
                    if point_g is not None:
                        theta_x, theta_y = self.point_to_galvo_angles(point_g)
                        code_x, code_y = self.angles_to_codes(theta_x, theta_y)
                        self.last_pixel_pos = (pixel_x, pixel_y)
                        self.last_galvo_pos = (code_x, code_y)
                        return (int(code_x), int(code_y))
            else:
                # 无深度或失败则继续原有逻辑：光线与工作平面交点
                ray_dir_camera = self.pixel_to_camera_ray(pixel_x, pixel_y)
                if ray_dir_camera is None:
                    rospy.logdebug("Failed at step 1: pixel to camera ray")
                    return None

                ray_origin_galvo = self.t_gc
                ray_dir_galvo = self.R_gc @ ray_dir_camera

                intersection_point = self.ray_plane_intersection(ray_origin_galvo, ray_dir_galvo)
                if intersection_point is None:
                    rospy.logdebug("Failed at step 3: ray-plane intersection")
                    return None

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

        rospy.logdebug(
            f"Simple mapping: pixel({pixel_x:.1f},{pixel_y:.1f}) -> norm({norm_x:.3f},{norm_y:.3f}) -> codes({galvo_x:.0f},{galvo_y:.0f})")

        return (int(galvo_x), int(galvo_y))

    def pixel_to_camera_ray(self, pixel_x, pixel_y):
        try:
            if self.use_distortion:
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
                rospy.logdebug("Ray is parallel to the plane, no intersection")
                return None

            t = -(np.dot(self.n_g, ray_origin) + self.d_g) / denominator
            if t <= 0:
                rospy.logdebug(f"Intersection point is behind camera, t={t}")
                return None

            intersection = ray_origin + t * ray_direction
            return intersection
        except Exception as e:
            rospy.logdebug(f"Failed to compute ray-plane intersection: {e}")
            return None

    def point_to_galvo_angles(self, point):
        x, y, z = point
        if abs(z) < 1e-6:
            rospy.logdebug("Z coordinate too small, cannot compute angles")
            return 0.0, 0.0
        theta_x = np.arctan2(x, z)
        theta_y = np.arctan2(y, z)
        return theta_x, theta_y

    def angles_to_codes(self, theta_x, theta_y):
        """
        角度（弧度）→ 码值。仅在3D路径中使用。Y轴附加 y_sign 以统一正反向。
        """
        galvo = self.params['galvo_params']
        theta_x_deg = np.degrees(theta_x)
        theta_y_deg = np.degrees(theta_y)

        theta_x_corrected = theta_x_deg * galvo['scale_x'] + galvo['bias_x']
        theta_y_corrected = (theta_y_deg * galvo['scale_y'] + galvo['bias_y']) * self.y_sign

        half_scan_angle = galvo['scan_angle'] / 2.0

        norm_x = theta_x_corrected / half_scan_angle
        norm_y = theta_y_corrected / half_scan_angle

        code_x = norm_x * galvo['max_code']
        code_y = norm_y * galvo['max_code']

        max_code = galvo['max_code']
        code_x = np.clip(code_x, -max_code, max_code)
        code_y = np.clip(code_y, -max_code, max_code)

        return code_x, code_y

    def galvo_code_to_pixel(self, galvo_x, galvo_y, image_width=640, image_height=480):
        try:
            if self.use_3d_transform and hasattr(self, 'transform_3d_initialized') and self.transform_3d_initialized:
                return self.galvo_code_to_pixel_3d(galvo_x, galvo_y, image_width, image_height)
            else:
                return self.galvo_code_to_pixel_simple(galvo_x, galvo_y, image_width, image_height)
        except Exception as e:
            rospy.logdebug(f"Reverse coordinate transform failed: {e}")
            return self.galvo_code_to_pixel_simple(galvo_x, galvo_y, image_width, image_height)

    def galvo_code_to_pixel_3d(self, galvo_x, galvo_y, image_width, image_height):
        try:
            theta_x, theta_y = self.codes_to_angles(galvo_x, galvo_y)
            intersection_point = self.galvo_angles_to_point(theta_x, theta_y)
            if intersection_point is None:
                rospy.logdebug("Failed to compute intersection point from galvo angles")
                return None

            ray_dir_camera = self.point_to_camera_ray(intersection_point)
            if ray_dir_camera is None:
                rospy.logdebug("Failed to compute camera ray from intersection point")
                return None

            pixel = self.camera_ray_to_pixel(ray_dir_camera, image_width, image_height)
            if pixel is None:
                return None
            pixel_x, pixel_y = pixel

            rospy.logdebug(
                f"3D reverse transform: codes({galvo_x:.0f},{galvo_y:.0f}) -> pixel({pixel_x:.1f},{pixel_y:.1f})")

            return (pixel_x, pixel_y)

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

        rospy.logdebug(
            f"Simple reverse mapping: codes({galvo_x:.0f},{galvo_y:.0f}) -> norm({norm_x:.3f},{norm_y:.3f}) -> pixel({pixel_x:.1f},{pixel_y:.1f})")

        return (pixel_x, pixel_y)

    def codes_to_angles(self, code_x, code_y):
        """
        码值 → 角度（弧度）。仅在3D路径中使用。与 angles_to_codes 完全互逆，Y轴应用 y_sign 的反向补偿。
        """
        galvo = self.params['galvo_params']
        max_code = galvo['max_code']

        norm_x = code_x / max_code
        norm_y = code_y / max_code

        half_scan_angle = galvo['scan_angle'] / 2.0

        theta_x_corrected = norm_x * half_scan_angle
        theta_y_corrected = norm_y * half_scan_angle

        theta_x_deg = (theta_x_corrected - galvo['bias_x']) / galvo['scale_x']
        theta_y_deg = ((theta_y_corrected) / self.y_sign - galvo['bias_y']) / galvo['scale_y']

        theta_x = np.radians(theta_x_deg)
        theta_y = np.radians(theta_y_deg)

        return theta_x, theta_y

    def galvo_angles_to_point(self, theta_x, theta_y):
        try:
            x = np.tan(theta_x)
            y = np.tan(theta_y)
            z = 1.0
            ray_direction = np.array([x, y, z], dtype=np.float64)
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            ray_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            intersection = self.ray_plane_intersection(ray_origin, ray_direction)
            return intersection
        except Exception as e:
            rospy.logdebug(f"Failed to compute point from galvo angles: {e}")
            return None

    def point_to_camera_ray(self, point):
        try:
            camera_to_point = point - self.t_gc
            camera_to_point = camera_to_point / np.linalg.norm(camera_to_point)
            ray_dir_camera = self.R_gc.T @ camera_to_point
            return ray_dir_camera
        except Exception as e:
            rospy.logdebug(f"Failed to compute camera ray from point: {e}")
            return None

    def camera_ray_to_pixel(self, ray_direction, image_width, image_height):
        try:
            if abs(ray_direction[2]) < 1e-6:
                rospy.logdebug("Ray direction z component too small")
                return None

            x = ray_direction[0] / ray_direction[2]
            y = ray_direction[1] / ray_direction[2]

            pixel_x = x * self.K[0, 0] + self.K[0, 2]
            pixel_y = y * self.K[1, 1] + self.K[1, 2]

            if self.use_distortion:
                pass

            return (pixel_x, pixel_y)
        except Exception as e:
            rospy.logdebug(f"Failed to convert camera ray to pixel: {e}")
            return None

    def switch_transform_mode(self, use_3d_transform):
        old_mode = "3D geometric transform" if self.use_3d_transform else "Simple linear mapping"
        new_mode = "3D geometric transform" if use_3d_transform else "Simple linear mapping"

        self.use_3d_transform = use_3d_transform
        self.params['transform_mode']['use_3d_transform'] = use_3d_transform

        if use_3d_transform and not hasattr(self, 'transform_3d_initialized'):
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
        if len(pixel_points) != len(galvo_points) or len(pixel_points) < 4:
            rospy.logerr("Calibration requires at least 4 corresponding points")
            return False

        try:
            rospy.loginfo(f"Calibrating with {len(pixel_points)} point pairs")
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
            'simple_mapping': self.params['simple_mapping']
        }

        if self.use_3d_transform and hasattr(self, 't_gc'):
            info.update({
                'camera_position': self.t_gc.tolist(),
                'work_plane_distance': self.d_g,
                'camera_matrix': self.K.tolist() if hasattr(self, 'K') else None,
                'transform_3d_initialized': getattr(self, 'transform_3d_initialized', False)
            })

        return info


# 示例配置保留
EXAMPLE_CONFIG = """
# 相机-振镜坐标变换配置文件

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
        print("Testing 3D geometric transform:")
        transform_3d = CameraGalvoTransform(use_3d_transform=True)

        print("\nTesting simple linear mapping:")
        transform_simple = CameraGalvoTransform(use_3d_transform=False)

        test_points = [(320, 240), (100, 100), (540, 380)]
        for px, py in test_points:
            result_3d = transform_3d.pixel_to_galvo_code(px, py)
            result_simple = transform_simple.pixel_to_galvo_code(px, py)
            print(f"Pixel({px}, {py}) -> 3D transform{result_3d} | Simple mapping{result_simple}")
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import yaml
import cv2
from onnx.reference.ops.op_non_max_suppression import PrepareContext
from scipy.spatial.transform import Rotation
import os
from dataclasses import dataclass
from typing import Optional
import threading

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CONFIG_PATH = os.path.join(PACKAGE_ROOT, 'cam_params.yaml')


def resolve_config_path(config_file: Optional[str]) -> str:
    """Resolve configuration paths relative to the repository root."""
    if not config_file:
        return DEFAULT_CONFIG_PATH
    
    expanded = os.path.expanduser(config_file)
    if os.path.isabs(expanded):
        return expanded
    
    package_candidate = os.path.join(PACKAGE_ROOT, expanded)
    if os.path.exists(package_candidate):
        return package_candidate
    
    return os.path.abspath(expanded)


@dataclass
class GalvoParams:
    Z_mm: float = 107.0          # 工作面参考距离（正向时会用实时 point_g[2] 覆盖）
    bx: float = 0.0             # code 偏置
    by: float = 0.0
    kx: float = 0.0             # alpha <- cx
    ky: float = 0.0             # beta  <- cy
    axy: float = -1.298e-5      # alpha <- cy
    ayx: float = -1.269e-5      # beta  <- cx

class CameraGalvoTransform:
    """
    相机-振镜坐标变换类
    支持两种模式：
    1) 3D几何变换：像素 → 相机光线/深度 → 振镜系点 → 振镜角 → 码值
       反向：码值 → 振镜角 → 振镜射线 → 选择相同深度面/固定深度面/工作平面 → 相机像素
    2) 简单线性映射（回退方案）
    """

    def __init__(self, config_file="/home/zhong/my_workspace/src/laser_weeding/yaml/cam_params.yaml", use_3d_transform=True):
        rospy.loginfo("Initializing camera-galvo coordinate transformer...")

        # 解析并保存配置文件路径
        self.config_file_path = resolve_config_path(config_file)

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
                't_gc': [0.0, 100.0, 0.0],  # 相机在振镜坐标系的位置（mm）
                'q_gc': [0.0, 0.0, 0.0, 1.0]  # 相机→振镜的旋转（xyzw）
            },
            'work_plane': {
                'n_g': [0.0, 0.0, 1.0],  # 振镜系下工作平面法向
                'd_g': -1000.0           # 平面方程 n·X + d = 0
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
            },
            'fix_extrinsics': {
                't_fix': [0.0, 0.0, 0.0],
                'q_fix': [0.0, 0.0, 0.0, 1.0],
            },
            'galvos': []
        }

        self.load_config(self.config_file_path)
        
        # 多振镜支持：加载振镜配置
        self._galvo_profiles = []
        self._build_galvo_profiles()
        self.active_profile_index = 0
        # 线程锁：保护振镜配置切换（防止多线程竞态条件）
        self._profile_lock = threading.Lock()
        self._active_poly_correction = None  # 当前激活的多项式校正系数
        self._active_correction_method = 'extrinsics'  # 当前激活的校正方法
        if self._galvo_profiles:
            self.set_active_galvo_profile(0)

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


        self.reverse_theta_x = None
        self.reverse_theta_y = None
        self.galvo_lin = GalvoParams()
        self.use_mech_compensation = False
        
        # --- 新增：深度保持变量 ---
        self.last_valid_depth_m = None  # 记录上一次有效的深度值
        # ------------------------
        
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
        ext_fix = self.params.get('fix_extrinsics', {'t_fix': [0.0, 0.0, 0.0], 'q_fix': [0.0, 0.0, 0.0, 1.0]})
        self.t_fix = np.array(ext_fix.get('t_fix', [0.0, 0.0, 0.0]), dtype=np.float64)
        q_fix = ext_fix.get('q_fix', [0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        self.q_fix = Rotation.from_quat(q_fix).as_matrix()


    def build_work_plane(self):
        plane = self.params['work_plane']
        self.n_g = np.array(plane['n_g'], dtype=np.float64)
        self.n_g = self.n_g / np.linalg.norm(self.n_g)
        self.d_g = float(plane['d_g'])

    def _build_galvo_profiles(self):
        """构建多振镜配置（接口层，不修改运算逻辑）"""
        galvos = self.params.get('galvos', [])
        if not isinstance(galvos, list) or not galvos:
            # 如果没有配置多振镜，使用默认配置
            return
        
        base_extrinsics = self.params.get('extrinsics', {})
        base_galvo_params = self.params.get('galvo_params', {})
        
        for idx, galvo_entry in enumerate(galvos):
            if not isinstance(galvo_entry, dict):
                continue
            
            galvo_params = galvo_entry.get('galvo_params', base_galvo_params)
            
            # 计算 axis_angle_limits
            scan_angle = galvo_params.get('scan_angle', 30.0)
            axis_angle_limits = {
                'x_plus': galvo_params.get('scan_angle_x_plus', scan_angle / 2.0),
                'x_minus': galvo_params.get('scan_angle_x_minus', scan_angle / 2.0),
                'y_plus': galvo_params.get('scan_angle_y_plus', scan_angle / 2.0),
                'y_minus': galvo_params.get('scan_angle_y_minus', scan_angle / 2.0),
            }
            
            # 提取多项式校正（如果存在）
            poly_correction = galvo_entry.get('poly_correction', None)
            active_correction = galvo_entry.get('active_correction', 'extrinsics')
            
            # 提取振镜配置
            profile = {
                'index': idx,
                'name': galvo_entry.get('name', f'galvo_{idx}'),
                'extrinsics': galvo_entry.get('extrinsics', {}),
                'galvo_params': galvo_params,
                'code_offset': galvo_entry.get('code_offset', [0.0, 0.0]),
                'code_scale': galvo_entry.get('code_scale', [1.0, 1.0]),
                'code_limits': galvo_entry.get('code_limits', {'x': [-32767, 32767], 'y': [-32767, 32767]}),
                'max_code': galvo_entry.get('max_code', galvo_params.get('max_code', 32767)),
                'axis_angle_limits': axis_angle_limits,
                'poly_correction': poly_correction,  # 多项式校正系数
                'active_correction': active_correction,  # 激活的校正方法：'extrinsics' 或 'poly_correction'
            }
            self._galvo_profiles.append(profile)
    
    def set_active_galvo_profile(self, galvo_index):
        """切换活跃振镜配置（接口层，不修改运算逻辑）
        
        注意：此方法使用线程锁保护，防止多线程竞态条件
        """
        if not self._galvo_profiles:
            # 如果没有多振镜配置，使用默认配置
            return
        
        if galvo_index < 0 or galvo_index >= len(self._galvo_profiles):
            rospy.logwarn(f"Invalid galvo_index {galvo_index}, using index 0")
            galvo_index = 0
        
        # 如果已经是目标配置，直接返回（避免不必要的锁竞争）
        if self.active_profile_index == galvo_index:
            return
        
        # 使用线程锁保护整个配置切换过程（包括外参更新）
        with self._profile_lock:
            # 双重检查：可能在等待锁的过程中，其他线程已经切换了
            if self.active_profile_index == galvo_index:
                return
            
            profile = self._galvo_profiles[galvo_index]
            self.active_profile_index = galvo_index
            
            # 更新当前使用的参数（临时修改，用于运算函数）
            # 提取 extrinsics
            extrinsics = profile.get('extrinsics', {})
            if isinstance(extrinsics, dict):
                # 支持 rough/refined 结构
                active_key = extrinsics.get('active', 'refined')
                if active_key in extrinsics and isinstance(extrinsics[active_key], dict):
                    ext_data = extrinsics[active_key]
                elif 'refined' in extrinsics and isinstance(extrinsics['refined'], dict):
                    ext_data = extrinsics['refined']
                elif 'rough' in extrinsics and isinstance(extrinsics['rough'], dict):
                    ext_data = extrinsics['rough']
                else:
                    ext_data = {}
            else:
                ext_data = {}
            
            # 更新 extrinsics（临时）- 支持 t_gc_mm 和 t_gc 两种格式
            if ext_data:
                if 't_gc_mm' in ext_data:
                    t_data = ext_data['t_gc_mm']
                    self.t_gc = np.array(t_data, dtype=np.float64)
                elif 't_gc' in ext_data:
                    t_data = ext_data['t_gc']
                    self.t_gc = np.array(t_data, dtype=np.float64)
                
                if 'q_gc_xyzw' in ext_data:
                    q_data = ext_data['q_gc_xyzw']
                    self.R_gc = Rotation.from_quat(q_data).as_matrix()
                elif 'q_gc' in ext_data:
                    q_data = ext_data['q_gc']
                    self.R_gc = Rotation.from_quat(q_data).as_matrix()
            
            # 更新 galvo_params（临时）- 合并到现有参数中
            galvo_params = profile.get('galvo_params', {})
            if galvo_params:
                # 创建临时副本，避免修改原始配置
                temp_params = self.params['galvo_params'].copy()
                temp_params.update(galvo_params)
                self.params['galvo_params'] = temp_params
            
            # 更新多项式校正（临时）
            self._active_poly_correction = profile.get('poly_correction', None)
            self._active_correction_method = profile.get('active_correction', 'extrinsics')
    
    def get_galvo_profile_count(self):
        """获取振镜配置数量"""
        return len(self._galvo_profiles) if self._galvo_profiles else 1
    
    def get_code_limits(self, galvo_index=0):
        """获取振镜代码限制"""
        if not self._galvo_profiles or galvo_index >= len(self._galvo_profiles):
            return ((-32767, 32767), (-32767, 32767))
        
        profile = self._galvo_profiles[galvo_index]
        limits = profile.get('code_limits', {'x': [-32767, 32767], 'y': [-32767, 32767]})
        x_lim = limits.get('x', [-32767, 32767])
        y_lim = limits.get('y', [-32767, 32767])
        return ((int(x_lim[0]), int(x_lim[1])), (int(y_lim[0]), int(y_lim[1])))
    
    def _get_profile(self, galvo_index):
        """获取振镜配置（内部方法）"""
        if not self._galvo_profiles:
            # 如果没有多振镜配置，返回默认配置的包装
            class DefaultProfile:
                def __init__(self, params):
                    self.galvo_params = params.get('galvo_params', {})
                    scan_angle = self.galvo_params.get('scan_angle', 30.0)
                    self.axis_angle_limits = {
                        'x_plus': self.galvo_params.get('scan_angle_x_plus', scan_angle / 2.0),
                        'x_minus': self.galvo_params.get('scan_angle_x_minus', scan_angle / 2.0),
                        'y_plus': self.galvo_params.get('scan_angle_y_plus', scan_angle / 2.0),
                        'y_minus': self.galvo_params.get('scan_angle_y_minus', scan_angle / 2.0),
                    }
            return DefaultProfile(self.params)
        
        if galvo_index < 0 or galvo_index >= len(self._galvo_profiles):
            rospy.logwarn(f"Invalid galvo_index {galvo_index}, using index 0")
            galvo_index = 0
        
        # 将字典包装成对象，以便使用点号访问
        class ProfileWrapper:
            def __init__(self, profile_dict):
                for key, value in profile_dict.items():
                    setattr(self, key, value)
        
        return ProfileWrapper(self._galvo_profiles[galvo_index])

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

    def pixel_to_galvo_code(self, pixel_x, pixel_y, image_width=640, image_height=480, galvo_index=0):
        """像素坐标转振镜代码（支持多振镜接口）
        
        注意：此方法会切换振镜配置，使用线程锁保护以避免竞态条件
        """
        # 接口层：切换振镜配置（线程安全）
        if self._galvo_profiles and galvo_index != self.active_profile_index:
            self.set_active_galvo_profile(galvo_index)
        
        # 在锁保护下执行坐标变换（确保使用正确的外参）
        with self._profile_lock:
            try:
                # pixel_x = 640
                # pixel_y = 407  # 407
                if self.use_3d_transform and getattr(self, 'transform_3d_initialized', False):
                    result = self.pixel_to_galvo_3d(pixel_x, pixel_y, image_width, image_height)
                    if result is not None:
                        self.transform_method_used = "3D geometric transform"
                        self.transform_valid = True
                        self.transform_fail_count = 0
                        return result
                    else:
                        # self.transform_fail_count += 1
                        # if self.transform_fail_count <= 5:
                        #     rospy.logwarn(f"3D transform failed (count: {self.transform_fail_count}), trying fallback")

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

    def galvo_code_to_pixel(self, galvo_x, galvo_y, image_width=640, image_height=480, galvo_index=0):
        """振镜代码转像素坐标（支持多振镜接口）
        
        注意：此方法会切换振镜配置，使用线程锁保护以避免竞态条件
        """
        # 接口层：切换振镜配置（线程安全）
        if self._galvo_profiles and galvo_index != self.active_profile_index:
            self.set_active_galvo_profile(galvo_index)
        
        # 在锁保护下执行坐标变换（确保使用正确的外参）
        with self._profile_lock:
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
        像素坐标-相机坐标-振镜坐标
        根据标定方法选择不同的外参：
        - SVD方法：使用refined extrinsics（计算出的新外参）
        - 多项式方法：使用rough extrinsics（原始外参），因为多项式校正会处理畸变
        """
        try:
            if depth_m is None or depth_m <= 0:
                return None
            
            x_c = (pixel_x - self.K[0, 2]) / self.K[0, 0] * depth_m
            y_c = (pixel_y - self.K[1, 2]) / self.K[1, 1] * depth_m
            z_c = depth_m
            p_c_mm = np.array([x_c * 1000.0, y_c * 1000.0, z_c * 1000.0], dtype=np.float64)

            # 根据标定方法选择外参
            if (self._active_correction_method == 'poly_correction' and 
                self._active_poly_correction is not None):
                # 多项式方法：使用rough extrinsics（原始外参）
                # 获取当前profile的rough extrinsics
                if self._galvo_profiles and self.active_profile_index < len(self._galvo_profiles):
                    profile = self._galvo_profiles[self.active_profile_index]
                    extrinsics = profile.get('extrinsics', {})
                    if isinstance(extrinsics, dict) and 'rough' in extrinsics:
                        rough_ext = extrinsics['rough']
                        if 't_gc_mm' in rough_ext:
                            t_rough = np.array(rough_ext['t_gc_mm'], dtype=np.float64)
                        elif 't_gc' in rough_ext:
                            t_rough = np.array(rough_ext['t_gc'], dtype=np.float64)
                        else:
                            t_rough = self.t_gc
                        
                        if 'q_gc_xyzw' in rough_ext:
                            R_rough = Rotation.from_quat(rough_ext['q_gc_xyzw']).as_matrix()
                        elif 'q_gc' in rough_ext:
                            R_rough = Rotation.from_quat(rough_ext['q_gc']).as_matrix()
                        else:
                            R_rough = self.R_gc
                        
                        p_g = R_rough @ p_c_mm + t_rough
                    else:
                        # 如果没有rough extrinsics，使用当前外参
                        p_g = self.R_gc @ p_c_mm + self.t_gc
                else:
                    # 没有多振镜配置，使用当前外参
                    p_g = self.R_gc @ p_c_mm + self.t_gc
            else:
                # SVD方法：使用refined extrinsics（当前外参，已在set_active_galvo_profile中更新）
                p_g = self.R_gc @ p_c_mm + self.t_gc
                # print(p_g)

            # 应用fix extrinsics（如果配置了）
            # p_g = self.q_fix @ p_g + self.t_fix
            
            return p_g
        except Exception:
            return None

    def pixel_to_camera_ray(self, pixel_x, pixel_y):
        try:
            if self.use_distortion:
                # 去畸变
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


    def angles_to_codes(self, theta_x, theta_y):
        """
        角度转成振镜编码，xy2-100形式
        """
        galvo = self.params['galvo_params']
        theta_x_deg = np.degrees(theta_x)   #弧度转角度
        theta_y_deg = np.degrees(theta_y)
        # print("theta_x_deg,theta_y_deg ",theta_x_deg,theta_y_deg)
        theta_x_corrected = theta_x_deg * galvo['scale_x'] + galvo['bias_x']    #应用修正
        theta_y_corrected = theta_y_deg * galvo['scale_y'] + galvo['bias_y']
        # print("theta_x_deg,theta_y_deg:", theta_x_corrected,theta_y_corrected)

        # half_scan_angle = galvo['scan_angle'] / 2.0  # 一侧的最大旋转角度
        # norm_x = theta_x_corrected / half_scan_angle
        # norm_y = theta_y_corrected / half_scan_angle
        # print("norm_x,norm_y:", norm_x, norm_y)
        # code_x = theta_x_corrected / half_scan_angle * galvo['max_code'] #*91/89        #角度转成code值 todo
        # if theta_y_corrected > 0:
        #     code_y = theta_y_corrected / half_scan_angle * galvo['max_code'] #* 46.5 / 45.5
        # else:
        #     code_y = theta_y_corrected / half_scan_angle * galvo['max_code'] #*  44.5 / 45.5

        if theta_x_corrected >= 0:
            code_x = theta_x_corrected / galvo['scan_angle_x_plus'] * 2.0 * galvo['max_code']
        elif theta_x_corrected < 0:
            half_scan_angle = galvo['scan_angle_x_minus'] / 2.0
            code_x = theta_x_corrected / half_scan_angle * galvo['max_code']
        # print(theta_y_corrected)
        if theta_y_corrected >= 0:
            code_y = theta_y_corrected / galvo['scan_angle_y_plus'] * 2.0 * galvo['max_code']
        elif theta_y_corrected < 0:
            code_y = theta_y_corrected / galvo['scan_angle_y_minus'] * 2.0 * galvo['max_code']

        max_code = galvo['max_code']
        saturated = (abs(code_x) > max_code) or (abs(code_y) > max_code)
        if saturated:
            rospy.logwarn_throttle(1.0, "angles_to_codes saturated; visualization/roundtrip may be inaccurate")

        code_x = np.clip(code_x, -max_code, max_code)
        code_y = np.clip(code_y, -max_code, max_code)   #做限制

        return code_x, code_y

    def pixel_to_galvo_3d(self, pixel_x, pixel_y, image_width, image_height):
        try:
            # 1. 尝试获取深度
            current_depth_m = None
            if callable(getattr(self, 'depth_query_func', None)):
                raw_depth = self.depth_query_func(pixel_x, pixel_y)
                # --- 鲁棒性增强逻辑 ---
                # 判定深度是否有效
                if raw_depth is not None :
                    current_depth_m = raw_depth
                    
                    # (可选) 简单的低通滤波：防止深度数值本身的高频抖动
                    # 如果你想让深度变化更丝滑，可以解开下面这行的注释
                    # if self.last_valid_depth_m is not None:
                    #     current_depth_m = self.last_valid_depth_m * 0.4 + raw_depth * 0.6
                    
                    # 更新历史值的缓存
                    self.last_valid_depth_m = current_depth_m
                else:
                    # 如果当前帧深度丢失 (None 或 0)
                    if self.last_valid_depth_m is not None:
                        # 使用"历史值保持"策略
                        # rospy.logwarn_throttle(1.0, "Depth dropout! Holding last valid depth.")
                        current_depth_m = self.last_valid_depth_m
                # --------------------
            # current_depth_m  = 0.31
            # 2. 如果最终要到了深度 (无论是当前的还是历史的)
            if current_depth_m is not None and current_depth_m > 0:
                # print("pixel_x,pixel_y", pixel_x, pixel_y)
                point_g = self.pixel_depth_to_point_galvo(pixel_x, pixel_y, current_depth_m)
                # print("point_g", point_g)
                if point_g is not None:
                    # 缓存命中点供反向映射使用
                    self.last_hit_point_g = point_g.copy()

                    self.use_mech_compensation = False
                    if self.use_mech_compensation:
                        theta_x, theta_y = self.point_to_galvo_angles_fix(point_g)
                    else:
                        theta_x, theta_y = self.point_to_galvo_angles(point_g)
                    # print("theta_x, theta_y", theta_x, theta_y)
                    code_x, code_y = self.angles_to_codes(theta_x, theta_y)
                    # print("codex,codey",code_x,code_y)
                    self.last_pixel_pos = (pixel_x, pixel_y)
                    self.last_galvo_pos = (code_x, code_y)
                    return (int(code_x), int(code_y))
            # 3. 如果连历史深度都没有（系统刚启动），才回退到"射线-平面求交"
            # 无深度或深度失败，使用给定深度平面

            ray_dir_camera = self.pixel_to_camera_ray(pixel_x, pixel_y)
            if ray_dir_camera is None:
                return None
            
            ray_origin_galvo = self.t_gc
            ray_dir_galvo = self.R_gc @ ray_dir_camera

            intersection_point = self.ray_plane_intersection(ray_origin_galvo, ray_dir_galvo)
            if intersection_point is None:
                return None

            self.last_hit_point_g = intersection_point.copy()

            theta_x, theta_y = self.point_to_galvo_angles(intersection_point)
            code_x, code_y = self.angles_to_codes(theta_x, theta_y)

            self.last_pixel_pos = (pixel_x, pixel_y)
            self.last_galvo_pos = (code_x, code_y)

            return (int(code_x), int(code_y))

        except Exception as e:
            rospy.logdebug(f"3D transform exception: {e}")
            return None

    def point_to_galvo_angles_fix(self, point):
        x, y, z = point
        self.reverse_theta_x = np.arctan2(x, z)
        self.reverse_theta_y = np.arctan2(y, z)
        if z < 0:
            x, y, z = x, y, -z
        if abs(z) < 1e-9:
            return 0.0, 0.0
        #
        # theta_x = np.arctan2(x, z)
        # theta_y = np.arctan2(y * np.cos(theta_x), z)

        d_mm = 20
        #有振镜距离d_mm
        theta_x = np.arctan2(x, z)
        # 2. 然后利用 theta_x, d 和 z 来计算 theta_y
        cos_theta_x = np.cos(theta_x)
        # 根据反解公式计算分子和分母
        numerator_y = y * cos_theta_x
        # denominator_y = z - d_mm * cos_theta_x

        if numerator_y > 0:
            denominator_y = z - d_mm * cos_theta_x
        else:
            denominator_y = z - d_mm * cos_theta_x
        if abs(denominator_y) < 1e-9:
            return theta_x, 0.0
        theta_y = np.arctan2(numerator_y, denominator_y)


        return theta_x, theta_y
    def point_to_galvo_angles(self, point):
        """ 从坐标点计算与振镜的夹角"""
        x, y, z = point
        self.reverse_theta_x = np.arctan2(x, z)
        self.reverse_theta_y = np.arctan2(y, z)
        if z < 0:
            x, y, z = x, y, -z
        if abs(z) < 1e-9:
            return 0.0, 0.0
        theta_x = np.arctan2(x, z)      #得到弧度
        theta_y = np.arctan2(y, z)
        # print("thetax,thetay:",theta_x, theta_y)
        return theta_x, theta_y


    @staticmethod
    def _angles_yfirst_from_plane(X_mm, Y_mm, Z_mm):
        # alpha 先（Y/Z），beta 后（X/Z * cos(alpha)）
        Z = float(Z_mm)
        if Z < 0:
            X_mm, Y_mm, Z = X_mm, Y_mm, -Z
        alpha = np.arctan2(Y_mm, Z)
        ca = np.cos(alpha)
        beta = np.arctan2((X_mm / Z) * ca, 1.0)
        return alpha, beta

    def angles_to_codes_mech(self, alpha, beta):
        # [alpha; beta] = [[kx, axy],[ayx, ky]] * ([cx-bx; cy-by])  的反解
        p = self.galvo_lin
        A11, A12 = p.kx, p.axy
        A21, A22 = p.ayx, p.ky
        det = A11 * A22 - A12 * A21
        alpha = float(alpha)
        beta = float(beta)

        if abs(det) < 1e-16:
            cx = p.bx + beta / (p.ayx if abs(p.ayx) > 1e-16 else -1e-5)
            cy = p.by + alpha / (p.axy if abs(p.axy) > 1e-16 else -1e-5)
        else:
            dcx = (A22 * alpha - A12 * beta) / det
            dcy = (-A21 * alpha + A11 * beta) / det
            cx = p.bx + dcx
            cy = p.by + dcy

        cx = int(np.clip(np.rint(cx), -32767, 32767))
        cy = int(np.clip(np.rint(cy), -32767, 32767))
        return cx, cy

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


    def codes_to_angles(self, code_x, code_y):
        """
        振镜编码转角度，与 angles_to_codes 保持一致的逻辑
        使用不对称角度参数（scan_angle_x_plus, scan_angle_x_minus, scan_angle_y_plus, scan_angle_y_minus）
        """
        galvo = self.params['galvo_params']
        max_code = galvo['max_code']

        norm_x = code_x / max_code
        norm_y = code_y / max_code

        # 使用不对称角度参数（与 angles_to_codes 保持一致）
        # 对于 X 轴
        if norm_x >= 0:
            # 正向：code_x = theta_x_corrected / scan_angle_x_plus * 2.0 * max_code
            # 反向：theta_x_corrected = norm_x * scan_angle_x_plus / 2.0
            scan_angle_x = galvo.get('scan_angle_x_plus', galvo['scan_angle'] / 2.0)
            theta_x_corrected = norm_x * scan_angle_x / 2.0
        else:
            # 负向：code_x = theta_x_corrected / (scan_angle_x_minus / 2.0) * max_code
            # 反向：theta_x_corrected = norm_x * scan_angle_x_minus / 2.0
            scan_angle_x = galvo.get('scan_angle_x_minus', galvo['scan_angle'] / 2.0)
            theta_x_corrected = norm_x * scan_angle_x / 2.0

        # 对于 Y 轴
        if norm_y >= 0:
            # 正向：code_y = theta_y_corrected / scan_angle_y_plus * 2.0 * max_code
            # 反向：theta_y_corrected = norm_y * scan_angle_y_plus / 2.0
            scan_angle_y = galvo.get('scan_angle_y_plus', galvo['scan_angle'] / 2.0)
            theta_y_corrected = norm_y * scan_angle_y / 2.0
        else:
            # 负向：code_y = theta_y_corrected / scan_angle_y_minus * 2.0 * max_code
            # 反向：theta_y_corrected = norm_y * scan_angle_y_minus / 2.0
            scan_angle_y = galvo.get('scan_angle_y_minus', galvo['scan_angle'] / 2.0)
            theta_y_corrected = norm_y * scan_angle_y / 2.0

        # 应用反向修正（去除 scale 和 bias）
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

    def galvo_angles_to_point_depth_cam(self, theta_x, theta_y, z_ref_mm):
        # 显式公式：Pg = [-z_ref * tan(ax), -z_ref * tan(ay), z_ref]
        tx = np.tan(theta_x)
        ty = np.tan(theta_y)
        return np.array([-z_ref_mm * tx, -z_ref_mm * ty, z_ref_mm], dtype=np.float64)

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
        # 1) 目标的深度
        if isinstance(self.last_hit_point_g, np.ndarray) and self.last_hit_point_g.shape == (3,):
            z_ref = float(self.last_hit_point_g[2])
            p = self.galvo_angles_to_point_depth_cam(theta_x, theta_y, z_ref)
            # print("galvo 2 pixel p",p)
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
            xg, yg, zg =  self.t_gc[0], self.t_gc[1], self.t_gc[2]
            t_cg = [xg, -yg, zg]
            xfix,yfix,zfix =  self.t_fix[0], self.t_fix[1], self.t_fix[2]
            t_fix = [xfix, -yfix, zfix]
            # print("point_g1:",point_g)
            # print("r fix,t fix",self.q_fix,t_fix)
            point_g = self.q_fix.T @ point_g
            point_g = point_g + t_fix
            # print("point_g2:",point_g)
            # print("t_g:",self.t_gc)
            # print("camera 2 point1",camera_to_point)
            # camera_to_point = camera_to_point / np.linalg.norm(camera_to_point)
            p_gc = self.R_gc.T @ point_g  #from galvo corr to cam corr
            p_gc  = p_gc + t_cg
            # print("p_gc:",p_gc)
            return p_gc
        except Exception:
            return None

    def camera_ray_to_pixel(self, p_gc, image_width, image_height):
        try:
            if abs(p_gc[2]) < 1e-9:
                return None
            x = p_gc[0] / p_gc[2]
            y = p_gc[1] / p_gc[2]

            pixel_x = x * self.K[0, 0] + self.K[0, 2]
            pixel_y = y * self.K[1, 1] + self.K[1, 2]

            return (pixel_x, pixel_y)
        except Exception:
            return None
    def codes_to_angles_mech(self, cx, cy):
        # 直接乘矩阵： [alpha; beta] = M * ([cx;cy] - [bx;by])
        p = self.galvo_lin
        v = np.array([float(cx) - p.bx, float(cy) - p.by], dtype=float)
        M = np.array([[p.kx, p.axy],
                      [p.ayx, p.ky]], dtype=float)
        alpha, beta = (M @ v).tolist()
        return alpha, beta
    def galvo_code_to_pixel_3d(self, galvo_x, galvo_y, image_width, image_height):
        """
        code -> 角度 -> 交点-> pixel
        """
        try:
            # print("galvo x,y", galvo_x, galvo_y)
            # theta_x, theta_y = self.codes_to_angles(galvo_x, galvo_y)
            if self.use_mech_compensation:
                alpha, beta = self.codes_to_angles_mech(galvo_x, galvo_y)
                # 对应关系：alpha -> theta_y, beta -> theta_x
                theta_y, theta_x = alpha, beta
            else:
                theta_x, theta_y = self.codes_to_angles(galvo_x, galvo_y)
            # print("galvo 2 pixel thetax,y",theta_x,theta_y)
            max_code = self.params['galvo_params']['max_code']
            # if abs(galvo_x) >= max_code - 1 or abs(galvo_y) >= max_code - 1:
            #     rospy.logwarn_throttle(1.0, "galvo_code appears saturated; reverse projection may be inaccurate")

            intersection_point = self.choose_reverse_intersection(theta_x, theta_y)
            # print("cam_p ",intersection_point)
            if intersection_point is None:
                return None

            p_gc = self.point_to_camera_ray(intersection_point)
            # print("p_gc",p_gc)
            if p_gc is None:
                return None

            pixel = self.camera_ray_to_pixel(p_gc, image_width, image_height)
            # print("pixel ",pixel)
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



if __name__ == "__main__":
    #for test
    import sys
    print("Testing 3D geometric transform with depth-aware reverse...")
    t = CameraGalvoTransform(use_3d_transform=True)
    # 可选：固定反向深度
    t.set_fixed_depth_for_reverse(500.0)
    test_points = [(640, 407)]
    for px, py in test_points:
        res = t.galvo_code_to_pixel(px, py)
        print("p2g:", (px, py), "->", res)
        if res:
            uv = t.galvo_code_to_pixel(res[0], res[1])
            print("g2p:", res, "->", uv)
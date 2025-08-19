# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# import numpy as np
# import rospy
# import yaml
# import cv2
# from scipy.spatial.transform import Rotation
# import os
#
#
# class CameraGalvoTransform:
#     """
#     相机-振镜坐标变换类
#     支持两种模式：
#     1. 3D几何变换：像素 → 相机光线 → 地面交点 → 振镜角度 → 码值
#     2. 简单线性映射：像素直接线性映射到振镜码值（原始方法）
#     """
#
#     def __init__(self, config_file=None, use_3d_transform=True):
#         """
#         初始化坐标变换器
#
#         Args:
#             config_file: 配置文件路径，如果为None则使用默认参数
#             use_3d_transform: 是否使用3D几何变换，False则使用简单线性映射
#         """
#         rospy.loginfo("initialize coordinate transform...")
#
#         self.use_3d_transform = use_3d_transform
#         # 默认参数
#         self.default_params = {
#             # 变换模式
#             'transform_mode': {
#                 'use_3d_transform': use_3d_transform,
#                 'fallback_to_simple': True  # 3D变换失败时是否回退到简单模式
#             },
#
#             # 相机内参矩阵 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
#             'camera_matrix': {
#                 'fx': 500.0,  # 焦距x
#                 'fy': 500.0,  # 焦距y
#                 'cx': 320.0,  # 主点x
#                 'cy': 240.0  # 主点y
#             },
#
#             # 畸变系数 D = [k1, k2, p1, p2, k3]
#             'distortion_coeffs': [],  # 空表示不考虑畸变
#
#             # 相机相对振镜的外参（振镜坐标系下的相机位置和姿态）
#             'extrinsics': {
#                 # 平移向量 t_gc：相机在振镜坐标系中的位置(mm)
#                 't_gc': [0.0, 100.0, -50.0],  # [右, 前, 上]
#                 # 旋转四元数 q_gc：相机在振镜坐标系中的姿态 [x,y,z,w]
#                 'q_gc': [0.0, 0.0, 0.0, 1.0]  # 无旋转
#             },
#
#             # 工作平面参数（振镜坐标系下）
#             'work_plane': {
#                 # 平面法向量（单位向量）
#                 'n_g': [0.0, 0.0, 1.0],  # Z向上
#                 # 平面到原点距离（mm）振镜原点到地面的垂直距离
#                 'd_g': -200.0  # 地面在振镜下方200mm
#             },
#
#             # 振镜参数
#             'galvo_params': {
#                 'scan_angle': 30.0,  # 总扫描角度（度）
#                 'scale_x': 1.0,  # X轴比例因子
#                 'scale_y': 1.0,  # Y轴比例因子
#                 'bias_x': 0.0,  # X轴零点偏移（度）
#                 'bias_y': 0.0,  # Y轴零点偏移（度）
#                 'max_code': 32767  # 最大码值
#             },
#
#             # 简单线性映射参数（原始方法）
#             'simple_mapping': {
#                 'scale_factor': 65535,  # 映射比例因子（对应原来的60000）
#                 'offset_x': 0,  # X轴偏移
#                 'offset_y': 0,  # Y轴偏移
#                 'max_code': 32767  # 最大码值限制（对应原来的±30000）
#             }
#         }
#
#         # 加载配置
#         self.load_config(config_file)
#
#         # 根据模式初始化相应组件
#         if self.use_3d_transform:
#             self.init_3d_transform()
#
#         # 初始化简单映射（作为备用或主要方法）
#         self.init_simple_mapping()
#
#         # 初始化状态
#         self.last_pixel_pos = None
#         self.last_galvo_pos = None
#         self.transform_valid = True
#         self.transform_method_used = "未知"
#
#         rospy.loginfo("finish loading coordinate transform")
#         # self.print_config_summary()
#
#     def load_config(self, config_file):
#         """加载配置文件"""
#         self.params = self.default_params.copy()
#
#         if config_file and os.path.exists(config_file):
#             try:
#                 with open(config_file, 'r') as f:
#                     config = yaml.safe_load(f)
#                     self.update_params_recursive(self.params, config)
#                 rospy.loginfo(f"load params from yaml: {config_file}")
#
#                 # 从配置文件更新变换模式
#                 if 'transform_mode' in config:
#                     self.use_3d_transform = config['transform_mode'].get('use_3d_transform', self.use_3d_transform)
#
#             except Exception as e:
#                 rospy.logwarn(f"loading yaml fail, using default params: {e}")
#         else:
#             rospy.loginfo("using default params")
#
#     def update_params_recursive(self, default_dict, update_dict):
#         """递归更新参数字典"""
#         for key, value in update_dict.items():
#             if key in default_dict and isinstance(default_dict[key], dict) and isinstance(value, dict):
#                 self.update_params_recursive(default_dict[key], value)
#             else:
#                 default_dict[key] = value
#
#     def init_3d_transform(self):
#         """初始化3D几何变换组件"""
#         # 构建相机内参矩阵
#         self.build_camera_matrix()
#         # 构建外参变换矩阵
#         self.build_extrinsics()
#         # 构建工作平面
#         self.build_work_plane()
#
#     def init_simple_mapping(self):
#         """初始化简单线性映射组件"""
#         simple = self.params['simple_mapping']
#         self.simple_scale_factor = simple['scale_factor']
#         self.simple_offset_x = simple['offset_x']
#         self.simple_offset_y = simple['offset_y']
#         self.simple_max_code = simple['max_code']
#
#
#     def build_camera_matrix(self):
#         """构建相机内参矩阵"""
#         cam = self.params['camera_matrix']
#         self.K = np.array([
#             [cam['fx'], 0, cam['cx']],
#             [0, cam['fy'], cam['cy']],
#             [0, 0, 1]
#         ], dtype=np.float64)
#
#         # 畸变系数
#         self.D = np.array(self.params['distortion_coeffs'], dtype=np.float64)
#         self.use_distortion = len(self.D) > 0
#
#         rospy.logdebug(f"相机内参矩阵 K:\n{self.K}")
#
#     def build_extrinsics(self):
#         """构建外参变换矩阵"""
#         ext = self.params['extrinsics']
#
#         # 平移向量
#         self.t_gc = np.array(ext['t_gc'], dtype=np.float64)
#
#         # 旋转矩阵（从四元数）
#         q = ext['q_gc']  # [x, y, z, w]
#         self.R_gc = Rotation.from_quat(q).as_matrix()
#
#         rospy.logdebug(f"相机位置 t_gc: {self.t_gc}")
#         rospy.logdebug(f"相机姿态 R_gc:\n{self.R_gc}")
#
#     def build_work_plane(self):
#         """构建工作平面"""
#         plane = self.params['work_plane']
#         self.n_g = np.array(plane['n_g'], dtype=np.float64)
#         self.n_g = self.n_g / np.linalg.norm(self.n_g)  # 归一化
#         self.d_g = float(plane['d_g'])
#
#         rospy.logdebug(f"工作平面法向量: {self.n_g}")
#         rospy.logdebug(f"工作平面距离: {self.d_g}")
#
#     def print_config_summary(self):
#         """打印配置摘要"""
#         rospy.loginfo("=== 坐标变换配置摘要 ===")
#         rospy.loginfo(f"变换模式: {'3D几何变换' if self.use_3d_transform else '简单线性映射'}")
#
#         if self.use_3d_transform:
#             rospy.loginfo(
#                 f"相机内参: fx={self.K[0, 0]:.1f}, fy={self.K[1, 1]:.1f}, cx={self.K[0, 2]:.1f}, cy={self.K[1, 2]:.1f}")
#             rospy.loginfo(f"相机位置: {self.t_gc}")
#             rospy.loginfo(f"工作平面: n={self.n_g}, d={self.d_g}")
#             galvo = self.params['galvo_params']
#             rospy.loginfo(
#                 f"振镜参数: 扫描角={galvo['scan_angle']}°, 比例=({galvo['scale_x']:.2f},{galvo['scale_y']:.2f})")
#         else:
#             simple = self.params['simple_mapping']
#             rospy.loginfo(f"简单映射: 比例={simple['scale_factor']}, 最大码值=±{simple['max_code']}")
#
#         rospy.loginfo("========================")
#
#     def pixel_to_galvo_code(self, pixel_x, pixel_y, image_width=640, image_height=480):
#         """
#         主要接口：将像素坐标转换为振镜码值
#
#         Args:
#             pixel_x, pixel_y: 像素坐标
#             image_width, image_height: 图像尺寸
#
#         Returns:
#             tuple: (galvo_x_code, galvo_y_code) 振镜码值，失败返回None
#         """
#         try:
#             if self.use_3d_transform:
#                 result = self.pixel_to_galvo_3d(pixel_x, pixel_y, image_width, image_height)
#                 if result is not None:
#                     self.transform_method_used = "3d transform"
#                     self.transform_valid = True
#                     return result
#                 elif self.params['transform_mode']['fallback_to_simple']:
#                     # 3D变换失败，回退到简单映射
#                     rospy.logdebug("3d trans failed, using simple transform")
#                     result = self.pixel_to_galvo_simple(pixel_x, pixel_y, image_width, image_height)
#                     self.transform_method_used = "simple reflect"
#                     self.transform_valid = True
#                     return result
#                 else:
#                     self.transform_valid = False
#                     return None
#             else:
#                 # 直接使用简单线性映射
#                 result = self.pixel_to_galvo_simple(pixel_x, pixel_y, image_width, image_height)
#                 self.transform_method_used = "simple reflect"
#                 self.transform_valid = True
#                 return result
#
#         except Exception as e:
#             rospy.logdebug(f"coordinate transform failed: {e}")
#             self.transform_valid = False
#             return None
#
#     def pixel_to_galvo_3d(self, pixel_x, pixel_y, image_width, image_height):
#         """
#         3D几何坐标变换：像素 → 相机光线 → 地面交点 → 振镜角度 → 码值
#
#         Returns:
#             tuple: (galvo_x_code, galvo_y_code) 或 None
#         """
#         # 第1步：像素坐标转换为相机坐标系中的光线方向
#         ray_dir_camera = self.pixel_to_camera_ray(pixel_x, pixel_y)
#         if ray_dir_camera is None:
#             return None
#
#         # 第2步：相机光线 → 振镜坐标系射线
#         ray_origin_galvo = self.t_gc  # 相机在振镜坐标系中的位置
#         ray_dir_galvo = self.R_gc @ ray_dir_camera  # 光线方向转到振镜坐标系
#
#         # 第3步：射线与工作平面求交点
#         intersection_point = self.ray_plane_intersection(ray_origin_galvo, ray_dir_galvo)
#         if intersection_point is None:
#             return None
#
#         # 第4步：交点 → 振镜角度
#         theta_x, theta_y = self.point_to_galvo_angles(intersection_point)
#
#         # 第5步：角度 → 码值
#         code_x, code_y = self.angles_to_codes(theta_x, theta_y)
#
#         # 保存状态用于可视化
#         self.last_pixel_pos = (pixel_x, pixel_y)
#         self.last_galvo_pos = (code_x, code_y)
#
#         return (int(code_x), int(code_y))
#
#     def pixel_to_galvo_simple(self, pixel_x, pixel_y, image_width, image_height):
#         """
#         简单线性映射：像素直接线性映射到振镜码值（原始方法）
#
#         这是原来的简单变换方法，假设相机和振镜同轴
#
#         Returns:
#             tuple: (galvo_x_code, galvo_y_code)
#         """
#         # 归一化到[-0.5, 0.5]范围
#         norm_x = (pixel_x / image_width - 0.5)
#         norm_y = (pixel_y / image_height - 0.5)
#
#         # 应用比例因子
#         galvo_x = norm_x * self.simple_scale_factor + self.simple_offset_x
#         galvo_y = norm_y * self.simple_scale_factor + self.simple_offset_y
#
#         # 限制范围
#         galvo_x = max(-self.simple_max_code, min(self.simple_max_code, galvo_x))
#         galvo_y = max(-self.simple_max_code, min(self.simple_max_code, galvo_y))
#
#         # 保存状态用于可视化
#         self.last_pixel_pos = (pixel_x, pixel_y)
#         self.last_galvo_pos = (galvo_x, galvo_y)
#
#         return (int(galvo_x), int(galvo_y))
#
#     def pixel_to_camera_ray(self, pixel_x, pixel_y):
#         """
#         第1步：像素坐标转换为相机坐标系中的光线方向
#
#         Returns:
#             numpy.array: 归一化的光线方向向量 [x, y, z]
#         """
#         # 去畸变（如果需要）
#         if self.use_distortion:
#             #todo
#             pass
#
#         # 像素坐标转相机坐标
#         print(K)
#         x = (pixel_x - self.K[0, 2]) / self.K[0, 0]
#         y = (pixel_y - self.K[1, 2]) / self.K[1, 1]
#
#         # 构建光线方向（相机坐标系中）
#         ray_dir = np.array([x, y, 1.0], dtype=np.float64)
#
#         # 归一化
#         ray_dir = ray_dir / np.linalg.norm(ray_dir)
#
#         return ray_dir
#
#     def ray_plane_intersection(self, ray_origin, ray_direction):
#         """
#         第3步：计算射线与平面的交点
#
#         射线方程: P(t) = origin + t * direction
#         平面方程: n · P + d = 0
#
#         Returns:
#             numpy.array: 交点坐标，失败返回None
#         """
#         # 计算分母: n · direction
#         denominator = np.dot(self.n_g, ray_direction)
#
#         # 检查射线是否平行于平面
#         if abs(denominator) < 1e-6:
#             rospy.logdebug("射线与平面平行，无交点")
#             return None
#
#         # 计算参数t
#         t = -(np.dot(self.n_g, ray_origin) + self.d_g) / denominator
#
#         # 检查交点是否在相机前方
#         if t <= 0:
#             rospy.logdebug(f"交点在相机后方，t={t}")
#             return None
#
#         # 计算交点
#         intersection = ray_origin + t * ray_direction
#
#         return intersection
#
#     def point_to_galvo_angles(self, point):
#         """
#         第4步：将振镜坐标系中的点转换为振镜角度
#
#         Args:
#             point: 振镜坐标系中的点 [x, y, z]
#
#         Returns:
#             tuple: (theta_x, theta_y) 振镜角度（弧度）
#         """
#         x, y, z = point
#
#         # 从振镜原点看向目标点的角度
#         # X轴角度（左右）
#         theta_x = np.arctan2(x, z)
#         # Y轴角度（上下）
#         theta_y = np.arctan2(y, z)
#
#         return theta_x, theta_y
#
#     def angles_to_codes(self, theta_x, theta_y):
#         """
#         第5步：将角度转换为振镜码值
#
#         Args:
#             theta_x, theta_y: 振镜角度（弧度）
#
#         Returns:
#             tuple: (code_x, code_y) 振镜码值
#         """
#         galvo = self.params['galvo_params']
#
#         # 转换为度数
#         theta_x_deg = np.degrees(theta_x)
#         theta_y_deg = np.degrees(theta_y)
#
#         # 应用比例和偏移
#         theta_x_corrected = theta_x_deg * galvo['scale_x'] + galvo['bias_x']
#         theta_y_corrected = theta_y_deg * galvo['scale_y'] + galvo['bias_y']
#
#         # 计算半扫描角
#         half_scan_angle = galvo['scan_angle'] / 2.0
#
#         # 归一化到[-1, 1]
#         norm_x = theta_x_corrected / half_scan_angle
#         norm_y = theta_y_corrected / half_scan_angle
#
#         # 转换为码值
#         code_x = norm_x * galvo['max_code']
#         code_y = norm_y * galvo['max_code']
#
#         # 限制范围
#         max_code = galvo['max_code']
#         code_x = np.clip(code_x, -max_code, max_code)
#         code_y = np.clip(code_y, -max_code, max_code)
#
#         return code_x, code_y
#
#     def switch_transform_mode(self, use_3d_transform):
#         """
#         切换变换模式
#
#         Args:
#             use_3d_transform: True for 3D几何变换, False for 简单线性映射
#         """
#         old_mode = "3D几何变换" if self.use_3d_transform else "简单线性映射"
#         new_mode = "3D几何变换" if use_3d_transform else "简单线性映射"
#
#         self.use_3d_transform = use_3d_transform
#         self.params['transform_mode']['use_3d_transform'] = use_3d_transform
#
#         if use_3d_transform and not hasattr(self, 'K'):
#             # 如果切换到3D模式但还没初始化，则初始化
#             self.init_3d_transform()
#
#         rospy.loginfo(f"坐标变换模式切换: {old_mode} -> {new_mode}")
#
#     def update_camera_matrix(self, camera_info_msg):
#         """
#         从ROS CameraInfo消息更新相机内参
#         """
#         try:
#             # 更新内参矩阵
#             K_flat = camera_info_msg.K
#             self.K = np.array(K_flat).reshape(3, 3)
#
#             # 更新畸变系数
#             self.D = np.array(camera_info_msg.D)
#             self.use_distortion = len(self.D) > 0
#
#             # 更新参数字典
#             self.params['camera_matrix']['fx'] = self.K[0, 0]
#             self.params['camera_matrix']['fy'] = self.K[1, 1]
#             self.params['camera_matrix']['cx'] = self.K[0, 2]
#             self.params['camera_matrix']['cy'] = self.K[1, 2]
#             self.params['distortion_coeffs'] = self.D.tolist()
#
#             rospy.loginfo("相机内参已从CameraInfo更新")
#
#         except Exception as e:
#             rospy.logwarn(f"CameraInfo更新失败: {e}")
#
#     def calibrate_with_points(self, pixel_points, galvo_points):
#         """
#         使用标定点对进行标定
#
#         Args:
#             pixel_points: 像素坐标点列表 [(u1,v1), (u2,v2), ...]
#             galvo_points: 对应的振镜码值点列表 [(x1,y1), (x2,y2), ...]
#         """
#         if len(pixel_points) != len(galvo_points) or len(pixel_points) < 4:
#             rospy.logerr("标定需要至少4对对应点")
#             return False
#
#         try:
#             # 这里可以实现最小二乘标定
#             # 简化版本：只调整scale和bias
#             rospy.loginfo(f"使用{len(pixel_points)}个点进行标定")
#             # TODO: 实现具体的标定算法
#             return True
#
#         except Exception as e:
#             rospy.logerr(f"标定失败: {e}")
#             return False
#
#     def save_config(self, config_file):
#         """保存当前配置到文件"""
#         try:
#             with open(config_file, 'w') as f:
#                 yaml.dump(self.params, f, default_flow_style=False)
#             rospy.loginfo(f"配置已保存到: {config_file}")
#             return True
#         except Exception as e:
#             rospy.logerr(f"配置保存失败: {e}")
#             return False
#
#     def get_transform_info(self):
#         """获取变换信息用于可视化"""
#         info = {
#             'last_pixel_pos': self.last_pixel_pos,
#             'last_galvo_pos': self.last_galvo_pos,
#             'transform_valid': self.transform_valid,
#             'transform_method': self.transform_method_used,
#             'use_3d_transform': self.use_3d_transform,
#             'galvo_params': self.params['galvo_params'],
#             'simple_mapping': self.params['simple_mapping']
#         }
#
#         if self.use_3d_transform and hasattr(self, 't_gc'):
#             info.update({
#                 'camera_position': self.t_gc.tolist(),
#                 'work_plane_distance': self.d_g,
#                 'camera_matrix': self.K.tolist() if hasattr(self, 'K') else None
#             })
#
#         return info
#
#
# # 更新配置文件示例，包含两种模式的参数
# EXAMPLE_CONFIG = """
# # 相机-振镜坐标变换配置文件
#
# # 变换模式配置
# transform_mode:
#   use_3d_transform: true      # true: 3D几何变换, false: 简单线性映射
#   fallback_to_simple: true    # 3D变换失败时是否回退到简单映射
#
# # === 3D几何变换参数 ===
# camera_matrix:
#   fx: 500.0      # 焦距x (像素)
#   fy: 500.0      # 焦距y (像素)
#   cx: 320.0      # 主点x (像素)
#   cy: 240.0      # 主点y (像素)
#
# distortion_coeffs: []  # 畸变系数 [k1, k2, p1, p2, k3]
#
# extrinsics:
#   # 相机在振镜坐标系中的位置 (mm)
#   t_gc: [0.0, 100.0, -50.0]  # [右+左-, 前+后-, 上+下-]
#   # 相机在振镜坐标系中的姿态四元数 [x, y, z, w]
#   q_gc: [0.0, 0.0, 0.0, 1.0]  # 无旋转
#
# work_plane:
#   # 工作平面法向量 (单位向量)
#   n_g: [0.0, 0.0, 1.0]  # Z轴向上
#   # 平面到振镜原点的距离 (mm, 负值表示在下方)
#   d_g: -200.0
#
# galvo_params:
#   scan_angle: 20.0   # 总扫描角度 (度)
#   scale_x: 1.0       # X轴比例因子
#   scale_y: 1.0       # Y轴比例因子
#   bias_x: 0.0        # X轴零点偏移 (度)
#   bias_y: 0.0        # Y轴零点偏移 (度)
#   max_code: 32767    # 最大码值
#
# # === 简单线性映射参数（原始方法）===
# simple_mapping:
#   scale_factor: 60000  # 映射比例因子
#   offset_x: 0          # X轴偏移
#   offset_y: 0          # Y轴偏移
#   max_code: 30000      # 最大码值限制 (±30000)
# """
#
#
# def create_example_config(file_path):
#     """创建示例配置文件"""
#     with open(file_path, 'w') as f:
#         f.write(EXAMPLE_CONFIG)
#     print(f"示例配置文件已创建: {file_path}")
#
#
# if __name__ == "__main__":
#     # 测试代码
#     import sys
#
#     if len(sys.argv) > 1 and sys.argv[1] == "create_config":
#         config_path = sys.argv[2] if len(sys.argv) > 2 else "camera_galvo_config.yaml"
#         create_example_config(config_path)
#     else:
#         # 测试两种变换器
#         print("测试3D几何变换:")
#         transform_3d = CameraGalvoTransform(use_3d_transform=True)
#
#         print("\n测试简单线性映射:")
#         transform_simple = CameraGalvoTransform(use_3d_transform=False)
#
#         # 测试几个像素点
#         test_points = [(320, 240), (100, 100), (540, 380)]
#         for px, py in test_points:
#             result_3d = transform_3d.pixel_to_galvo_code(px, py)
#             result_simple = transform_simple.pixel_to_galvo_code(px, py)
#             print(f"像素({px}, {py}) -> 3D变换{result_3d} | 简单映射{result_simple}")

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
    2. 简单线性映射：像素直接线性映射到振镜码值（原始方法）
    """

    def __init__(self, config_file=None, use_3d_transform=True):
        """
        初始化坐标变换器

        Args:
            config_file: 配置文件路径，如果为None则使用默认参数
            use_3d_transform: 是否使用3D几何变换，False则使用简单线性映射
        """
        rospy.loginfo("Initializing camera-galvo coordinate transformer...")

        self.use_3d_transform = use_3d_transform

        # if self.use_3d_transform:
        #     rospy.loginfo("Using 3D geometric coordinate transform mode")
        # else:
        #     rospy.loginfo("Using simple linear mapping mode (original method)")

        # 默认参数
        self.default_params = {
            # 变换模式
            'transform_mode': {
                'use_3d_transform': use_3d_transform,
                'fallback_to_simple': True  # 3D变换失败时是否回退到简单模式
            },

            # 相机内参矩阵 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            'camera_matrix': {
                'fx': 500.0,  # 焦距x
                'fy': 500.0,  # 焦距y
                'cx': 320.0,  # 主点x
                'cy': 240.0  # 主点y
            },

            # 畸变系数 D = [k1, k2, p1, p2, k3]
            'distortion_coeffs': [],  # 空表示不考虑畸变

            # 相机相对振镜的外参（振镜坐标系下的相机位置和姿态）
            'extrinsics': {
                # 平移向量 t_gc：相机在振镜坐标系中的位置(mm)
                't_gc': [0.0, 100.0, -50.0],  # [右, 前, 上]
                # 旋转四元数 q_gc：相机在振镜坐标系中的姿态 [x,y,z,w]
                'q_gc': [0.0, 0.0, 0.0, 1.0]  # 无旋转
            },

            # 工作平面参数（振镜坐标系下）
            'work_plane': {
                # 平面法向量（单位向量）
                'n_g': [0.0, 0.0, 1.0],  # Z向上
                # 平面到原点距离（mm）振镜原点到地面的垂直距离
                'd_g': -200.0  # 地面在振镜下方200mm
            },

            # 振镜参数
            'galvo_params': {
                'scan_angle': 30.0,  # 总扫描角度（度）
                'scale_x': 1.0,  # X轴比例因子
                'scale_y': 1.0,  # Y轴比例因子
                'bias_x': 0.0,  # X轴零点偏移（度）
                'bias_y': 0.0,  # Y轴零点偏移（度）
                'max_code': 32767  # 最大码值（安全范围）
            },

            # 简单线性映射参数（原始方法）
            'simple_mapping': {
                'use_safe_range': True,  # 是否使用安全范围
                'max_safe_range': 32767,  # 安全使用范围 (±30000)
                'protocol_max': 32767,  # 协议理论最大值
                'scale_factor': 65536,  # 映射比例因子
                'offset_x': 0,  # X轴偏移
                'offset_y': 0,  # Y轴偏移
            }
        }

        # 加载配置
        self.load_config(config_file)

        # 根据模式初始化相应组件
        if self.use_3d_transform:
            self.init_3d_transform()

        # 初始化简单映射（作为备用或主要方法）
        self.init_simple_mapping()

        # 初始化状态
        self.last_pixel_pos = None
        self.last_galvo_pos = None
        self.transform_valid = True
        self.transform_method_used = "Unknown"
        self.transform_fail_count = 0  # 添加失败计数

        rospy.loginfo("Camera-galvo coordinate transformer initialized successfully")
        # self.print_config_summary()

    def load_config(self, config_file):
        """加载配置文件"""
        self.params = self.default_params.copy()

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self.update_params_recursive(self.params, config)
                # rospy.loginfo(f"Parameters loaded from config file: {config_file}")

                # 从配置文件更新变换模式
                if 'transform_mode' in config:
                    self.use_3d_transform = config['transform_mode'].get('use_3d_transform', self.use_3d_transform)

            except Exception as e:
                rospy.logwarn(f"Failed to load config file, using default parameters: {e}")
        else:
            rospy.loginfo("Using default configuration parameters")

    def update_params_recursive(self, default_dict, update_dict):
        """递归更新参数字典"""
        for key, value in update_dict.items():
            if key in default_dict and isinstance(default_dict[key], dict) and isinstance(value, dict):
                self.update_params_recursive(default_dict[key], value)
            else:
                default_dict[key] = value

    def init_3d_transform(self):
        """初始化3D几何变换组件"""
        try:
            # 构建相机内参矩阵
            self.build_camera_matrix()

            # 构建外参变换矩阵
            self.build_extrinsics()

            # 构建工作平面
            self.build_work_plane()

            # 标记3D变换组件已初始化
            self.transform_3d_initialized = True

            # rospy.loginfo("3D geometric transform components initialized successfully")

        except Exception as e:
            rospy.logerr(f"Failed to initialize 3D transform components: {e}")
            self.transform_3d_initialized = False

    def init_simple_mapping(self):
        """初始化简单线性映射组件"""
        simple = self.params['simple_mapping']
        self.simple_use_safe_range = simple['use_safe_range']
        self.simple_max_safe_range = simple['max_safe_range']
        self.simple_protocol_max = simple['protocol_max']
        self.simple_scale_factor = simple['scale_factor']
        self.simple_offset_x = simple['offset_x']
        self.simple_offset_y = simple['offset_y']

        # rospy.loginfo("Simple linear mapping components initialized successfully")

    def build_camera_matrix(self):
        """构建相机内参矩阵"""
        cam = self.params['camera_matrix']
        self.K = np.array([
            [cam['fx'], 0, cam['cx']],
            [0, cam['fy'], cam['cy']],
            [0, 0, 1]
        ], dtype=np.float64)

        # 畸变系数
        self.D = np.array(self.params['distortion_coeffs'], dtype=np.float64)
        self.use_distortion = len(self.D) > 0

        rospy.logdebug(f"Camera matrix K:\n{self.K}")

    def build_extrinsics(self):
        """构建外参变换矩阵"""
        ext = self.params['extrinsics']

        # 平移向量
        self.t_gc = np.array(ext['t_gc'], dtype=np.float64)

        # 旋转矩阵（从四元数）
        q = ext['q_gc']  # [x, y, z, w]
        self.R_gc = Rotation.from_quat(q).as_matrix()

        rospy.logdebug(f"Camera position t_gc: {self.t_gc}")
        rospy.logdebug(f"Camera orientation R_gc:\n{self.R_gc}")

    def build_work_plane(self):
        """构建工作平面"""
        plane = self.params['work_plane']
        self.n_g = np.array(plane['n_g'], dtype=np.float64)
        self.n_g = self.n_g / np.linalg.norm(self.n_g)  # 归一化
        self.d_g = float(plane['d_g'])

        rospy.logdebug(f"Work plane normal: {self.n_g}")
        rospy.logdebug(f"Work plane distance: {self.d_g}")

    def print_config_summary(self):
        """打印配置摘要"""
        rospy.loginfo("=== Coordinate Transform Configuration Summary ===")
        rospy.loginfo(
            f"Transform mode: {'3D geometric transform' if self.use_3d_transform else 'Simple linear mapping'}")

        if self.use_3d_transform and hasattr(self, 'K'):
            rospy.loginfo(
                f"Camera intrinsics: fx={self.K[0, 0]:.1f}, fy={self.K[1, 1]:.1f}, cx={self.K[0, 2]:.1f}, cy={self.K[1, 2]:.1f}")
            rospy.loginfo(f"Camera position: {self.t_gc}")
            rospy.loginfo(f"Work plane: n={self.n_g}, d={self.d_g}")
            galvo = self.params['galvo_params']
            rospy.loginfo(
                f"Galvo params: scan_angle={galvo['scan_angle']}°, scale=({galvo['scale_x']:.2f},{galvo['scale_y']:.2f})")

        simple = self.params['simple_mapping']
        rospy.loginfo(
            f"Simple mapping: scale_factor={simple['scale_factor']}, max_range=±{simple['max_safe_range'] if simple['use_safe_range'] else simple['protocol_max']}")
        rospy.loginfo("================================================")

    def pixel_to_galvo_code(self, pixel_x, pixel_y, image_width=640, image_height=480):
        """
        主要接口：将像素坐标转换为振镜码值

        Args:
            pixel_x, pixel_y: 像素坐标
            image_width, image_height: 图像尺寸

        Returns:
            tuple: (galvo_x_code, galvo_y_code) 振镜码值，失败返回None
        """
        try:
            if self.use_3d_transform and hasattr(self, 'transform_3d_initialized') and self.transform_3d_initialized:
                # 尝试使用3D几何变换
                result = self.pixel_to_galvo_3d(pixel_x, pixel_y, image_width, image_height)
                if result is not None:
                    self.transform_method_used = "3D geometric transform"
                    self.transform_valid = True
                    self.transform_fail_count = 0  # 重置失败计数
                    return result
                else:
                    self.transform_fail_count += 1
                    if self.transform_fail_count <= 5:  # 只在前几次失败时打印警告
                        rospy.logwarn(f"3D transform failed (count: {self.transform_fail_count}), trying fallback")

                    if self.params['transform_mode']['fallback_to_simple']:
                        # 3D变换失败，回退到简单映射
                        result = self.pixel_to_galvo_simple(pixel_x, pixel_y, image_width, image_height)
                        self.transform_method_used = "Simple mapping (fallback)"
                        self.transform_valid = True
                        return result
                    else:
                        self.transform_valid = False
                        return None
            else:
                # 直接使用简单线性映射
                result = self.pixel_to_galvo_simple(pixel_x, pixel_y, image_width, image_height)
                self.transform_method_used = "Simple mapping"
                self.transform_valid = True
                return result

        except Exception as e:
            rospy.logerr(f"Coordinate transform failed: {e}")
            self.transform_valid = False
            # 在异常情况下也尝试简单映射
            try:
                result = self.pixel_to_galvo_simple(pixel_x, pixel_y, image_width, image_height)
                self.transform_method_used = "Simple mapping (exception fallback)"
                self.transform_valid = True
                return result
            except:
                return None

    def pixel_to_galvo_3d(self, pixel_x, pixel_y, image_width, image_height):
        """
        3D几何坐标变换：像素 → 相机光线 → 地面交点 → 振镜角度 → 码值

        Returns:
            tuple: (galvo_x_code, galvo_y_code) 或 None
        """
        try:
            # 第1步：像素坐标转换为相机坐标系中的光线方向
            ray_dir_camera = self.pixel_to_camera_ray(pixel_x, pixel_y)
            if ray_dir_camera is None:
                rospy.logdebug("Failed at step 1: pixel to camera ray")
                return None

            # 第2步：相机光线 → 振镜坐标系射线
            ray_origin_galvo = self.t_gc  # 相机在振镜坐标系中的位置
            ray_dir_galvo = self.R_gc @ ray_dir_camera  # 光线方向转到振镜坐标系

            # 第3步：射线与工作平面求交点
            intersection_point = self.ray_plane_intersection(ray_origin_galvo, ray_dir_galvo)
            if intersection_point is None:
                rospy.logdebug("Failed at step 3: ray-plane intersection")
                return None

            # 第4步：交点 → 振镜角度
            theta_x, theta_y = self.point_to_galvo_angles(intersection_point)

            # 第5步：角度 → 码值
            code_x, code_y = self.angles_to_codes(theta_x, theta_y)

            # 保存状态用于可视化
            self.last_pixel_pos = (pixel_x, pixel_y)
            self.last_galvo_pos = (code_x, code_y)

            # 调试信息
            # rospy.logdebug(
            #     f"3D transform: pixel({pixel_x:.1f},{pixel_y:.1f}) -> intersection({intersection_point[0]:.1f},{intersection_point[1]:.1f},{intersection_point[2]:.1f}) -> angles({np.degrees(theta_x):.2f}°,{np.degrees(theta_y):.2f}°) -> codes({code_x:.0f},{code_y:.0f})")

            return (int(code_x), int(code_y))

        except Exception as e:
            rospy.logdebug(f"3D transform exception: {e}")
            return None

    def pixel_to_galvo_simple(self, pixel_x, pixel_y, image_width, image_height):
        """
        简单线性映射：像素直接线性映射到振镜码值（原始方法）

        这是原来的简单变换方法，假设相机和振镜同轴

        Returns:
            tuple: (galvo_x_code, galvo_y_code)
        """
        # 归一化到[-0.5, 0.5]范围
        norm_x = (pixel_x / image_width - 0.5)
        norm_y = (pixel_y / image_height - 0.5)

        # 根据配置选择使用范围
        if self.simple_use_safe_range:
            # 使用安全范围：±30000
            max_range = self.simple_max_safe_range  # 30000
            scale_factor = max_range * 2  # 60000
        else:
            # 使用理论最大范围：±32767
            max_range = self.simple_protocol_max  # 32767
            scale_factor = max_range * 2  # 65535

        # 应用比例因子
        galvo_x = norm_x * scale_factor + self.simple_offset_x
        galvo_y = norm_y * scale_factor + self.simple_offset_y

        # 限制范围
        galvo_x = max(-max_range, min(max_range, galvo_x))
        galvo_y = max(-max_range, min(max_range, galvo_y))

        # 保存状态用于可视化
        self.last_pixel_pos = (pixel_x, pixel_y)
        self.last_galvo_pos = (galvo_x, galvo_y)

        # 调试信息
        rospy.logdebug(
            f"Simple mapping: pixel({pixel_x:.1f},{pixel_y:.1f}) -> norm({norm_x:.3f},{norm_y:.3f}) -> codes({galvo_x:.0f},{galvo_y:.0f})")

        return (int(galvo_x), int(galvo_y))

    def pixel_to_camera_ray(self, pixel_x, pixel_y):
        """
        第1步：像素坐标转换为相机坐标系中的光线方向

        Returns:
            numpy.array: 归一化的光线方向向量 [x, y, z]
        """
        try:
            # 去畸变（如果需要）
            if self.use_distortion:
                # 这里简化处理，实际应该用cv2.undistortPoints
                pass
            # 像素坐标转相机坐标
            x = (pixel_x - self.K[0, 2]) / self.K[0, 0]
            y = (pixel_y - self.K[1, 2]) / self.K[1, 1]

            # 构建光线方向（相机坐标系中）
            ray_dir = np.array([x, y, 1.0], dtype=np.float64)

            # 归一化
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            return ray_dir

        except Exception as e:
            rospy.logdebug(f"Failed to convert pixel to camera ray: {e}")
            return None

    def ray_plane_intersection(self, ray_origin, ray_direction):
        """
        第3步：计算射线与平面的交点

        射线方程: P(t) = origin + t * direction
        平面方程: n · P + d = 0

        Returns:
            numpy.array: 交点坐标，失败返回None
        """
        try:
            # 计算分母: n · direction
            denominator = np.dot(self.n_g, ray_direction)

            # 检查射线是否平行于平面
            if abs(denominator) < 1e-6:
                rospy.logdebug("Ray is parallel to the plane, no intersection")
                return None

            # 计算参数t
            t = -(np.dot(self.n_g, ray_origin) + self.d_g) / denominator

            # 检查交点是否在相机前方
            if t <= 0:
                rospy.logdebug(f"Intersection point is behind camera, t={t}")
                return None

            # 计算交点
            intersection = ray_origin + t * ray_direction

            return intersection

        except Exception as e:
            rospy.logdebug(f"Failed to compute ray-plane intersection: {e}")
            return None

    def point_to_galvo_angles(self, point):
        """
        第4步：将振镜坐标系中的点转换为振镜角度

        Args:
            point: 振镜坐标系中的点 [x, y, z]

        Returns:
            tuple: (theta_x, theta_y) 振镜角度（弧度）
        """
        x, y, z = point

        # 检查z坐标，避免除零
        if abs(z) < 1e-6:
            rospy.logdebug("Z coordinate too small, cannot compute angles")
            return 0.0, 0.0

        # 从振镜原点看向目标点的角度
        # X轴角度（左右）
        theta_x = np.arctan2(x, z)
        # Y轴角度（上下）
        theta_y = np.arctan2(y, z)

        return theta_x, theta_y

    def angles_to_codes(self, theta_x, theta_y):
        """
        第5步：将角度转换为振镜码值

        Args:
            theta_x, theta_y: 振镜角度（弧度）

        Returns:
            tuple: (code_x, code_y) 振镜码值
        """
        galvo = self.params['galvo_params']

        # 转换为度数
        theta_x_deg = np.degrees(theta_x)
        theta_y_deg = np.degrees(theta_y)

        # 应用比例和偏移
        theta_x_corrected = theta_x_deg * galvo['scale_x'] + galvo['bias_x']
        theta_y_corrected = theta_y_deg * galvo['scale_y'] + galvo['bias_y']

        # 计算半扫描角
        half_scan_angle = galvo['scan_angle'] / 2.0

        # 检查角度是否超出扫描范围
        if abs(theta_x_corrected) > half_scan_angle or abs(theta_y_corrected) > half_scan_angle:
            rospy.logdebug(
                f"Angles exceed scan range: x={theta_x_corrected:.2f}°, y={theta_y_corrected:.2f}°, max=±{half_scan_angle:.2f}°")

        # 归一化到[-1, 1]
        norm_x = theta_x_corrected / half_scan_angle
        norm_y = theta_y_corrected / half_scan_angle

        # 转换为码值
        code_x = norm_x * galvo['max_code']
        code_y = norm_y * galvo['max_code']

        # 限制范围
        max_code = galvo['max_code']
        code_x = np.clip(code_x, -max_code, max_code)
        code_y = np.clip(code_y, -max_code, max_code)

        return code_x, code_y

    def galvo_code_to_pixel(self, galvo_x, galvo_y, image_width=640, image_height=480):
        """
        振镜码值反向变换为像素坐标：码值 → 角度 → 地面交点 → 相机投影 → 像素

        Args:
            galvo_x, galvo_y: 振镜码值
            image_width, image_height: 图像尺寸

        Returns:
            tuple: (pixel_x, pixel_y) 或 None
        """
        try:
            if self.use_3d_transform and hasattr(self, 'transform_3d_initialized') and self.transform_3d_initialized:
                # 使用3D几何反向变换
                return self.galvo_code_to_pixel_3d(galvo_x, galvo_y, image_width, image_height)
            else:
                # 使用简单线性反向映射
                return self.galvo_code_to_pixel_simple(galvo_x, galvo_y, image_width, image_height)

        except Exception as e:
            rospy.logdebug(f"Reverse coordinate transform failed: {e}")
            # 回退到简单映射
            return self.galvo_code_to_pixel_simple(galvo_x, galvo_y, image_width, image_height)

    def galvo_code_to_pixel_3d(self, galvo_x, galvo_y, image_width, image_height):
        """
        3D几何反向变换：码值 → 角度 → 地面交点 → 相机投影 → 像素

        Returns:
            tuple: (pixel_x, pixel_y) 或 None
        """
        try:
            # 第1步：码值 → 角度（反向第5步）
            theta_x, theta_y = self.codes_to_angles(galvo_x, galvo_y)

            # 第2步：角度 → 地面交点（反向第4步）
            intersection_point = self.galvo_angles_to_point(theta_x, theta_y)
            if intersection_point is None:
                rospy.logdebug("Failed to compute intersection point from galvo angles")
                return None

            # 第3步：地面交点 → 相机坐标系射线（反向第2-3步）
            ray_dir_camera = self.point_to_camera_ray(intersection_point)
            if ray_dir_camera is None:
                rospy.logdebug("Failed to compute camera ray from intersection point")
                return None

            # 第4步：相机射线 → 像素坐标（反向第1步）
            pixel_x, pixel_y = self.camera_ray_to_pixel(ray_dir_camera, image_width, image_height)

            rospy.logdebug(
                f"3D reverse transform: codes({galvo_x:.0f},{galvo_y:.0f}) -> angles({np.degrees(theta_x):.2f}°,{np.degrees(theta_y):.2f}°) -> intersection({intersection_point[0]:.1f},{intersection_point[1]:.1f},{intersection_point[2]:.1f}) -> pixel({pixel_x:.1f},{pixel_y:.1f})")

            return (pixel_x, pixel_y)

        except Exception as e:
            rospy.logdebug(f"3D reverse transform exception: {e}")
            return None

    def galvo_code_to_pixel_simple(self, galvo_x, galvo_y, image_width, image_height):
        """
        简单线性反向映射：码值直接线性映射到像素坐标

        Returns:
            tuple: (pixel_x, pixel_y)
        """
        # 根据配置选择使用范围
        if self.simple_use_safe_range:
            max_range = self.simple_max_safe_range  # 30000
            scale_factor = max_range * 2  # 60000
        else:
            max_range = self.simple_protocol_max  # 32767
            scale_factor = max_range * 2  # 65535

        # 移除偏移
        galvo_x_centered = galvo_x - self.simple_offset_x
        galvo_y_centered = galvo_y - self.simple_offset_y

        # 反向比例因子
        norm_x = galvo_x_centered / scale_factor
        norm_y = galvo_y_centered / scale_factor

        # 反归一化到像素坐标
        pixel_x = (norm_x + 0.5) * image_width
        pixel_y = (norm_y + 0.5) * image_height

        rospy.logdebug(
            f"Simple reverse mapping: codes({galvo_x:.0f},{galvo_y:.0f}) -> norm({norm_x:.3f},{norm_y:.3f}) -> pixel({pixel_x:.1f},{pixel_y:.1f})")

        return (pixel_x, pixel_y)

    def codes_to_angles(self, code_x, code_y):
        """
        反向第5步：将振镜码值转换为角度

        Args:
            code_x, code_y: 振镜码值

        Returns:
            tuple: (theta_x, theta_y) 振镜角度（弧度）
        """
        galvo = self.params['galvo_params']
        max_code = galvo['max_code']

        # 码值转归一化值 [-1, 1]
        norm_x = code_x / max_code
        norm_y = code_y / max_code

        # 计算半扫描角
        half_scan_angle = galvo['scan_angle'] / 2.0

        # 归一化值转角度（度）
        theta_x_corrected = norm_x * half_scan_angle
        theta_y_corrected = norm_y * half_scan_angle

        # 移除偏移和比例
        theta_x_deg = (theta_x_corrected - galvo['bias_x']) / galvo['scale_x']
        theta_y_deg = (theta_y_corrected - galvo['bias_y']) / galvo['scale_y']

        # 转换为弧度
        theta_x = np.radians(theta_x_deg)
        theta_y = np.radians(theta_y_deg)

        return theta_x, theta_y

    def galvo_angles_to_point(self, theta_x, theta_y):
        """
        反向第4步：将振镜角度转换为工作平面上的点

        Args:
            theta_x, theta_y: 振镜角度（弧度）

        Returns:
            numpy.array: 工作平面上的交点 [x, y, z]
        """
        try:
            # 从振镜角度构造方向向量
            # 振镜在原点，朝向工作平面
            x = np.tan(theta_x)  # x/z = tan(theta_x)
            y = np.tan(theta_y)  # y/z = tan(theta_y)
            z = 1.0  # 归一化z方向

            # 构造射线方向（从振镜原点出发）
            ray_direction = np.array([x, y, z], dtype=np.float64)
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            # 振镜原点
            ray_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

            # 计算与工作平面的交点
            intersection = self.ray_plane_intersection(ray_origin, ray_direction)

            return intersection

        except Exception as e:
            rospy.logdebug(f"Failed to compute point from galvo angles: {e}")
            return None

    def point_to_camera_ray(self, point):
        """
        反向第2-3步：将地面交点转换为相机坐标系中的射线方向

        Args:
            point: 振镜坐标系中的地面交点 [x, y, z]

        Returns:
            numpy.array: 相机坐标系中的射线方向 [x, y, z]
        """
        try:
            # 从相机位置到地面交点的向量（振镜坐标系）
            camera_to_point = point - self.t_gc

            # 归一化
            camera_to_point = camera_to_point / np.linalg.norm(camera_to_point)

            # 转换到相机坐标系（R_gc的逆变换）
            ray_dir_camera = self.R_gc.T @ camera_to_point

            return ray_dir_camera

        except Exception as e:
            rospy.logdebug(f"Failed to compute camera ray from point: {e}")
            return None

    def camera_ray_to_pixel(self, ray_direction, image_width, image_height):
        """
        反向第1步：将相机坐标系中的射线方向转换为像素坐标

        Args:
            ray_direction: 相机坐标系中的射线方向 [x, y, z]
            image_width, image_height: 图像尺寸

        Returns:
            tuple: (pixel_x, pixel_y)
        """
        try:
            # 归一化到z=1平面
            if abs(ray_direction[2]) < 1e-6:
                rospy.logdebug("Ray direction z component too small")
                return None

            x = ray_direction[0] / ray_direction[2]
            y = ray_direction[1] / ray_direction[2]

            # 相机坐标转像素坐标
            pixel_x = x * self.K[0, 0] + self.K[0, 2]
            pixel_y = y * self.K[1, 1] + self.K[1, 2]

            # 应用畸变（如果需要）
            if self.use_distortion:
                # 这里简化处理，实际应该用cv2.projectPoints
                pass

            return (pixel_x, pixel_y)

        except Exception as e:
            rospy.logdebug(f"Failed to convert camera ray to pixel: {e}")
            return None

    def switch_transform_mode(self, use_3d_transform):
        """
        切换变换模式

        Args:
            use_3d_transform: True for 3D几何变换, False for 简单线性映射
        """
        old_mode = "3D geometric transform" if self.use_3d_transform else "Simple linear mapping"
        new_mode = "3D geometric transform" if use_3d_transform else "Simple linear mapping"

        self.use_3d_transform = use_3d_transform
        self.params['transform_mode']['use_3d_transform'] = use_3d_transform

        if use_3d_transform and not hasattr(self, 'transform_3d_initialized'):
            # 如果切换到3D模式但还没初始化，则初始化
            self.init_3d_transform()

        # 重置失败计数
        self.transform_fail_count = 0

        rospy.loginfo(f"Coordinate transform mode switched: {old_mode} -> {new_mode}")

    def update_camera_matrix(self, camera_info_msg):
        """
        从ROS CameraInfo消息更新相机内参
        """
        try:
            # 更新内参矩阵
            K_flat = camera_info_msg.K
            self.K = np.array(K_flat).reshape(3, 3)

            # 更新畸变系数
            self.D = np.array(camera_info_msg.D)
            self.use_distortion = len(self.D) > 0

            # 更新参数字典
            self.params['camera_matrix']['fx'] = self.K[0, 0]
            self.params['camera_matrix']['fy'] = self.K[1, 1]
            self.params['camera_matrix']['cx'] = self.K[0, 2]
            self.params['camera_matrix']['cy'] = self.K[1, 2]
            self.params['distortion_coeffs'] = self.D.tolist()

            rospy.loginfo("Camera intrinsics updated from CameraInfo message")

        except Exception as e:
            rospy.logwarn(f"Failed to update camera intrinsics from CameraInfo: {e}")

    def calibrate_with_points(self, pixel_points, galvo_points):
        """
        使用标定点对进行标定

        Args:
            pixel_points: 像素坐标点列表 [(u1,v1), (u2,v2), ...]
            galvo_points: 对应的振镜码值点列表 [(x1,y1), (x2,y2), ...]
        """
        if len(pixel_points) != len(galvo_points) or len(pixel_points) < 4:
            rospy.logerr("Calibration requires at least 4 corresponding points")
            return False

        try:
            # 这里可以实现最小二乘标定
            # 简化版本：只调整scale和bias
            rospy.loginfo(f"Calibrating with {len(pixel_points)} point pairs")
            # TODO: 实现具体的标定算法
            return True

        except Exception as e:
            rospy.logerr(f"Calibration failed: {e}")
            return False

    def save_config(self, config_file):
        """保存当前配置到文件"""
        try:
            with open(config_file, 'w') as f:
                yaml.dump(self.params, f, default_flow_style=False)
            rospy.loginfo(f"Configuration saved to: {config_file}")
            return True
        except Exception as e:
            rospy.logerr(f"Failed to save configuration: {e}")
            return False

    def get_transform_info(self):
        """获取变换信息用于可视化"""
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


# 更新配置文件示例，包含两种模式的参数
EXAMPLE_CONFIG = """
# 相机-振镜坐标变换配置文件

# 变换模式配置
transform_mode:
  use_3d_transform: true      # true: 3D几何变换, false: 简单线性映射
  fallback_to_simple: true    # 3D变换失败时是否回退到简单映射

# === 3D几何变换参数 ===
camera_matrix:
  fx: 500.0      # 焦距x (像素)
  fy: 500.0      # 焦距y (像素)
  cx: 320.0      # 主点x (像素)
  cy: 240.0      # 主点y (像素)

distortion_coeffs: []  # 畸变系数 [k1, k2, p1, p2, k3]

extrinsics:
  # 相机在振镜坐标系中的位置 (mm)
  t_gc: [0.0, 100.0, -50.0]  # [右+左-, 前+后-, 上+下-]
  # 相机在振镜坐标系中的姿态四元数 [x, y, z, w]
  q_gc: [0.0, 0.0, 0.0, 1.0]  # 无旋转

work_plane:
  # 工作平面法向量 (单位向量)
  n_g: [0.0, 0.0, 1.0]  # Z轴向上
  # 平面到振镜原点的距离 (mm, 负值表示在下方)
  d_g: -200.0

galvo_params:
  scan_angle: 20.0   # 总扫描角度 (度)
  scale_x: 1.0       # X轴比例因子
  scale_y: 1.0       # Y轴比例因子
  bias_x: 0.0        # X轴零点偏移 (度)
  bias_y: 0.0        # Y轴零点偏移 (度)
  max_code: 30000    # 最大码值

# === 简单线性映射参数（原始方法）===
simple_mapping:
  use_safe_range: true     # 使用安全范围还是理论最大范围
  max_safe_range: 30000    # 安全范围：±30000
  protocol_max: 32767      # XY2-100协议最大值
  scale_factor: 60000      # 映射比例因子
  offset_x: 0              # X轴偏移
  offset_y: 0              # Y轴偏移
"""


def create_example_config(file_path):
    """创建示例配置文件"""
    with open(file_path, 'w') as f:
        f.write(EXAMPLE_CONFIG)
    print(f"Example configuration file created: {file_path}")


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create_config":
        config_path = sys.argv[2] if len(sys.argv) > 2 else "camera_galvo_config.yaml"
        create_example_config(config_path)
    else:
        # 测试两种变换器
        print("Testing 3D geometric transform:")
        transform_3d = CameraGalvoTransform(use_3d_transform=True)

        print("\nTesting simple linear mapping:")
        transform_simple = CameraGalvoTransform(use_3d_transform=False)

        # 测试几个像素点
        test_points = [(320, 240), (100, 100), (540, 380)]
        for px, py in test_points:
            result_3d = transform_3d.pixel_to_galvo_code(px, py)
            result_simple = transform_simple.pixel_to_galvo_code(px, py)
            print(f"Pixel({px}, {py}) -> 3D transform{result_3d} | Simple mapping{result_simple}")
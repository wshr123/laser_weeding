#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Camera ↔ Galvo coordinate utilities.

This module keeps the minimal surface that the rest of the project relies on
while staying readable: load parameters, manage multiple galvo heads and
provide simple helpers for forward/backward projections.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
from scipy.spatial.transform import Rotation

from .config import DEFAULT_CONFIG_PATH, load_yaml_file, resolve_config_path


@dataclass
class ExtrinsicsSet:
    t_gc: np.ndarray
    R_gc: np.ndarray
    q_gc: np.ndarray


@dataclass
class GalvoHeadProfile:
    index: int
    name: str
    extrinsics: Dict[str, ExtrinsicsSet]
    active_key: str
    t_gc: np.ndarray
    R_gc: np.ndarray
    q_gc: np.ndarray
    code_offset: np.ndarray
    code_scale: np.ndarray
    max_code: float
    code_limits: np.ndarray
    galvo_params: Dict[str, float]
    axis_angle_limits: Dict[str, float]


class CameraGalvoTransform:
    """Coordinate transform helper for multiple galvo heads."""

    DEFAULT_PARAMS: Dict[str, object] = {
        'transform_mode': {
            'use_3d_transform': True,
        },
        'camera_matrix': {
            'fx': 500.0,
            'fy': 500.0,
            'cx': 320.0,
            'cy': 240.0,
        },
        'distortion_coeffs': [],
        'extrinsics': {
            't_gc': [0.0, 100.0, 0.0],
            'q_gc': [0.0, 0.0, 0.0, 1.0],
        },
        'work_plane': {
            'n_g': [0.0, 0.0, 1.0],
            'd_g': -1000.0,
        },
        'galvo_params': {
            'scan_angle': 30.0,
            'scan_angle_x_plus': 15.0,
            'scan_angle_x_minus': 15.0,
            'scan_angle_y_plus': 15.0,
            'scan_angle_y_minus': 15.0,
            'scale_x': 1.0,
            'scale_y': 1.0,
            'bias_x': 0.0,
            'bias_y': 0.0,
            'max_code': 32767,
        },
        'simple_mapping': {
            'use_safe_range': True,
            'max_safe_range': 32767,
            'protocol_max': 32767,
            'offset_x': 0,
            'offset_y': 0,
        },
        'galvos': [],
        'manual_calibration': {
            'enabled': False,
            'result_file': '',
        },
    }

    def __init__(self, config_file: Optional[str] = None, use_3d_transform: bool = True):
        rospy.loginfo("Initializing camera-galvo coordinate transformer...")

        self.config_file_path = resolve_config_path(config_file)
        self.use_3d_transform = bool(use_3d_transform)

        self.params = self._load_params(self.config_file_path)
        mode_cfg = self.params.get('transform_mode', {})
        self.use_3d_transform = bool(mode_cfg.get('use_3d_transform', self.use_3d_transform))

        self.active_extrinsics_map: Dict[str, ExtrinsicsSet] = {}
        self.active_extrinsics_key: str = 'rough'

        self._build_camera_model()
        self._build_work_plane()
        self._build_galvo_profiles()

        self.depth_query_func = None
        self.fixed_reverse_depth_z_mm: Optional[float] = None
        self.last_hit_point_g: Optional[np.ndarray] = None
        self.last_pixel_pos: Optional[Tuple[float, float]] = None
        self.last_galvo_pos: Optional[Tuple[int, int]] = None
        self.transform_valid = True
        self.transform_method_used = 'Simple mapping'

        if not self._galvo_profiles:
            raise RuntimeError('No galvo profile available – please check cam_params.yaml')

        self.set_active_galvo_profile(0)

        rospy.loginfo("Camera-galvo coordinate transformer ready")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @classmethod
    def _merge_dict(cls, target: Dict[str, object], update: Dict[str, object]) -> None:
        for key, value in update.items():
            if isinstance(target.get(key), dict) and isinstance(value, dict):
                cls._merge_dict(target[key], value)
            else:
                target[key] = value

    def _load_params(self, config_file: str) -> Dict[str, object]:
        params = copy.deepcopy(self.DEFAULT_PARAMS)

        if config_file and os.path.exists(config_file):
            try:
                user_params = load_yaml_file(config_file)
                self._merge_dict(params, user_params)
            except Exception as exc:  # pragma: no cover - just a safety log
                rospy.logwarn(f"Failed to read config file {config_file}: {exc}")
        else:
            rospy.logwarn_once(f"Config file {config_file} not found – using defaults")

        self._apply_manual_calibration(params)
        return params

    def _apply_manual_calibration(self, params: Dict[str, object]) -> None:
        cfg = params.get('manual_calibration') or {}
        if not cfg.get('enabled'):
            return

        result_file = cfg.get('result_file') or cfg.get('file')
        if not result_file:
            rospy.logwarn_once('Manual calibration enabled but result_file is empty')
            return

        manual_path = resolve_config_path(result_file)
        if not os.path.exists(manual_path):
            rospy.logwarn_once(f'Manual calibration file {manual_path} not found')
            return

        try:
            manual_data = load_yaml_file(manual_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            rospy.logwarn(f'Failed to load manual calibration file {manual_path}: {exc}')
            return

        overrides = manual_data.get('galvos', []) if isinstance(manual_data, dict) else []
        if not isinstance(overrides, list) or not overrides:
            rospy.logwarn_once('Manual calibration file does not contain galvos list')
            return

        galvos = params.setdefault('galvos', [])

        name_to_index = {
            entry.get('name', f'galvo_{idx}'): idx
            for idx, entry in enumerate(galvos)
            if isinstance(entry, dict)
        }

        for override in overrides:
            if not isinstance(override, dict):
                continue

            idx: Optional[int] = None
            name = override.get('name')
            if name and name in name_to_index:
                idx = name_to_index[name]
            else:
                try:
                    candidate = int(override.get('id'))
                except (TypeError, ValueError):
                    candidate = None
                if candidate is not None and 0 <= candidate < len(galvos):
                    idx = candidate

            if idx is None:
                idx = len(galvos)
                galvos.append({'name': name or f'galvo_{idx}'})
                name_to_index[galvos[idx]['name']] = idx

            entry = galvos[idx]
            if name:
                entry.setdefault('name', name)

            refined = override.get('refined_extrinsics') or override.get('extrinsics')
            if isinstance(refined, dict):
                extr_cfg = entry.setdefault('extrinsics', {})
                extr_cfg['refined'] = refined
                active_name = override.get('active_extrinsics', 'refined')
                if not isinstance(active_name, str) or active_name not in ('rough', 'refined'):
                    active_name = 'refined'
                extr_cfg['active'] = active_name

            manual_info = override.get('manual_calibration')
            if isinstance(manual_info, dict):
                entry.setdefault('manual_calibration', {}).update(manual_info)
                updated = manual_info.get('updated_galvo_params')
                if isinstance(updated, dict):
                    entry.setdefault('galvo_params', {}).update(updated)

            code_limits = override.get('code_limits')
            if isinstance(code_limits, dict):
                entry['code_limits'] = code_limits

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------

    def _build_camera_model(self) -> None:
        cam = self.params.get('camera_matrix', {})
        fx = float(cam.get('fx', 500.0))
        fy = float(cam.get('fy', 500.0))
        cx = float(cam.get('cx', 320.0))
        cy = float(cam.get('cy', 240.0))

        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D = np.array(self.params.get('distortion_coeffs', []), dtype=np.float64)
        self.use_distortion = bool(len(self.D))

    def _build_work_plane(self) -> None:
        plane = self.params.get('work_plane', {})
        n_g = np.array(plane.get('n_g', [0.0, 0.0, 1.0]), dtype=np.float64)
        norm = np.linalg.norm(n_g)
        self.n_g = n_g / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.d_g = float(plane.get('d_g', -1000.0))

    def _combine_galvo_params(self, overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
        params = dict(self.params.get('galvo_params', {}))
        if isinstance(overrides, dict):
            params.update(overrides)
        return params

    @staticmethod
    def _parse_translation(source: Dict[str, object], fallback: Dict[str, object]) -> np.ndarray:
        if 't_gc_mm' in source:
            values = source['t_gc_mm']
        elif 't_gc' in source:
            values = source['t_gc']
        else:
            values = fallback.get('t_gc_mm', fallback.get('t_gc', [0.0, 0.0, 0.0]))
        return np.array([float(v) for v in values], dtype=np.float64)

    @staticmethod
    def _parse_rotation(source: Dict[str, object], fallback: Dict[str, object]) -> Rotation:
        if 'R_gc' in source:
            return Rotation.from_matrix(np.array(source['R_gc'], dtype=np.float64))

        quat = source.get('q_gc_xyzw', source.get('q_gc'))
        if quat is None:
            quat = fallback.get('q_gc_xyzw', fallback.get('q_gc', [0.0, 0.0, 0.0, 1.0]))
        quat = np.array(quat, dtype=np.float64)
        if quat.shape != (4,) or not np.isfinite(quat).all():
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-9:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            quat = quat / norm
        return Rotation.from_quat(quat)

    @staticmethod
    def _parse_code_limits(limits: Optional[Dict[str, List[float]]], default_max: float) -> np.ndarray:
        if isinstance(limits, dict):
            x_lim = limits.get('x', [-default_max, default_max])
            y_lim = limits.get('y', [-default_max, default_max])
        else:
            x_lim = [-default_max, default_max]
            y_lim = [-default_max, default_max]
        arr = np.array([x_lim, y_lim], dtype=np.float64)
        arr[:, 0] = np.minimum(arr[:, 0], arr[:, 1])
        return arr

    def _compute_axis_limits(self, galvo_params: Dict[str, float]) -> Dict[str, float]:
        total = float(galvo_params.get('scan_angle', self.DEFAULT_PARAMS['galvo_params']['scan_angle']))
        half = total / 2.0 if total > 0 else self.DEFAULT_PARAMS['galvo_params']['scan_angle'] / 2.0

        def pick(key: str) -> float:
            try:
                value = float(galvo_params.get(key, half))
            except (TypeError, ValueError):
                value = half
            return value if value > 0 else half

        return {
            'x_plus': pick('scan_angle_x_plus'),
            'x_minus': pick('scan_angle_x_minus'),
            'y_plus': pick('scan_angle_y_plus'),
            'y_minus': pick('scan_angle_y_minus'),
        }

    def _create_extrinsics_set(
        self,
        source: Optional[Dict[str, object]],
        fallback: Optional[Dict[str, object]]
    ) -> ExtrinsicsSet:
        src_dict = source or {}
        fallback_dict = fallback or {}
        translation = self._parse_translation(src_dict, fallback_dict)
        rotation = self._parse_rotation(src_dict, fallback_dict)
        return ExtrinsicsSet(
            t_gc=translation,
            R_gc=rotation.as_matrix(),
            q_gc=rotation.as_quat(),
        )

    def _prepare_extrinsics(
        self,
        entry_extr: Optional[Dict[str, object]],
        base_extr: Dict[str, object]
    ) -> Tuple[ExtrinsicsSet, Dict[str, ExtrinsicsSet], str]:
        extr_dict: Dict[str, ExtrinsicsSet] = {}
        entry = entry_extr or {}
        base_dict = base_extr or {}

        rough_source = entry.get('rough') if isinstance(entry, dict) else None
        refined_source = entry.get('refined') if isinstance(entry, dict) else None
        active_override = entry.get('active') if isinstance(entry, dict) else None

        extr_dict['rough'] = self._create_extrinsics_set(
            rough_source if isinstance(rough_source, dict) else None,
            base_dict,
        )

        if isinstance(refined_source, dict):
            refined_fallback = rough_source if isinstance(rough_source, dict) else base_dict
            extr_dict['refined'] = self._create_extrinsics_set(refined_source, refined_fallback)

        if isinstance(active_override, str) and active_override in extr_dict:
            active_key = active_override
        elif 'refined' in extr_dict:
            active_key = 'refined'
        else:
            active_key = 'rough'

        active = extr_dict[active_key]
        return active, extr_dict, active_key

    def _build_galvo_profiles(self) -> None:
        self._galvo_profiles: List[GalvoHeadProfile] = []

        galvos = self.params.get('galvos') or [{}]
        if not isinstance(galvos, list):
            galvos = [{}]
            self.params['galvos'] = galvos

        base_extrinsics = self.params.get('extrinsics', {})

        for idx, entry in enumerate(galvos):
            entry = entry or {}
            name = entry.get('name', f'galvo_{idx}')
            extr_config = entry.get('extrinsics', {}) if isinstance(entry, dict) else {}
            active_extr, extr_map, active_key = self._prepare_extrinsics(extr_config, base_extrinsics)
            galvo_params = self._combine_galvo_params(entry.get('galvo_params'))
            code_offset = np.array(entry.get('code_offset', [0.0, 0.0]), dtype=np.float64)
            code_scale = np.array(entry.get('code_scale', [1.0, 1.0]), dtype=np.float64)
            max_code = float(entry.get('max_code', galvo_params.get('max_code', 32767)))
            code_limits = self._parse_code_limits(entry.get('code_limits'), max_code)
            axis_limits = self._compute_axis_limits(galvo_params)

            profile = GalvoHeadProfile(
                index=idx,
                name=name,
                extrinsics=extr_map,
                active_key=active_key,
                t_gc=active_extr.t_gc,
                R_gc=active_extr.R_gc,
                q_gc=active_extr.q_gc,
                code_offset=code_offset,
                code_scale=code_scale,
                max_code=max_code,
                code_limits=code_limits,
                galvo_params=galvo_params,
                axis_angle_limits=axis_limits,
            )
            self._galvo_profiles.append(profile)

    # ------------------------------------------------------------------
    # Profile accessors
    # ------------------------------------------------------------------

    def _get_profile(self, index: int) -> GalvoHeadProfile:
        if not self._galvo_profiles:
            raise RuntimeError('Galvo profiles not initialised')
        index = int(index)
        if index < 0 or index >= len(self._galvo_profiles):
            raise IndexError(f'Galvo index {index} out of range')
        return self._galvo_profiles[index]

    def set_active_galvo_profile(self, index: int) -> None:
        profile = self._get_profile(index)
        self.active_profile_index = profile.index
        self.active_profile = profile
        self.active_profile_max_code = float(profile.max_code)
        self.active_profile_offset = profile.code_offset.copy()
        self.active_profile_scale = profile.code_scale.copy()
        self.active_galvo_params = dict(profile.galvo_params)
        self.axis_angle_limits = dict(profile.axis_angle_limits)
        self.active_extrinsics_map = profile.extrinsics
        self.active_extrinsics_key = profile.active_key

        self.t_gc = profile.t_gc.copy()
        self.R_gc = profile.R_gc.copy()
        self.q_gc = profile.q_gc.copy()

    def get_galvo_profile_count(self) -> int:
        return len(self._galvo_profiles)

    def get_code_limits(self, index: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        profile = self._get_profile(index)
        limits = profile.code_limits
        return (
            (int(round(limits[0, 0])), int(round(limits[0, 1]))),
            (int(round(limits[1, 0])), int(round(limits[1, 1]))),
        )

    def get_galvo_params(self, index: Optional[int] = None) -> Dict[str, float]:
        profile = self._get_profile(index if index is not None else self.active_profile_index)
        return dict(profile.galvo_params)

    def get_axis_angle_limits(self, index: Optional[int] = None) -> Dict[str, float]:
        profile = self._get_profile(index if index is not None else self.active_profile_index)
        return dict(profile.axis_angle_limits)

    def get_profile_metadata(self, index: int) -> Dict[str, object]:
        profile = self._get_profile(index)
        return {
            'index': profile.index,
            'name': profile.name,
            'active_extrinsics': profile.active_key,
            't_gc_mm': profile.t_gc.tolist(),
            'R_gc': profile.R_gc.tolist(),
            'q_gc_xyzw': profile.q_gc.tolist(),
            'extrinsics': self._format_extrinsics_map(profile.extrinsics),
            'code_offset': profile.code_offset.tolist(),
            'code_scale': profile.code_scale.tolist(),
            'code_limits': profile.code_limits.tolist(),
            'max_code': profile.max_code,
            'galvo_params': dict(profile.galvo_params),
            'axis_angle_limits': dict(profile.axis_angle_limits),
        }

    def list_galvo_profiles(self) -> List[Dict[str, object]]:
        return [self.get_profile_metadata(idx) for idx in range(len(self._galvo_profiles))]

    @staticmethod
    def _format_extrinsics_map(extr_map: Dict[str, ExtrinsicsSet]) -> Dict[str, Dict[str, object]]:
        formatted: Dict[str, Dict[str, object]] = {}
        for key, extr in extr_map.items():
            formatted[key] = {
                't_gc_mm': extr.t_gc.tolist(),
                'R_gc': extr.R_gc.tolist(),
                'q_gc_xyzw': extr.q_gc.tolist(),
            }
        return formatted

    # ------------------------------------------------------------------
    # Depth helpers
    # ------------------------------------------------------------------

    def set_depth_query(self, func) -> None:
        self.depth_query_func = func

    def set_fixed_depth_for_reverse(self, z_mm: Optional[float]) -> None:
        self.fixed_reverse_depth_z_mm = z_mm

    # ------------------------------------------------------------------
    # Forward transform: pixel → galvo codes
    # ------------------------------------------------------------------

    def pixel_to_galvo_code(
        self,
        pixel_x: float,
        pixel_y: float,
        image_width: int = 640,
        image_height: int = 480,
        galvo_index: int = 0,
    ) -> Optional[Tuple[int, int]]:
        profile = self._get_profile(galvo_index)
        if profile.index != getattr(self, 'active_profile_index', None):
            self.set_active_galvo_profile(profile.index)

        try:
            if self.use_3d_transform:
                result = self._pixel_to_galvo_3d(pixel_x, pixel_y)
                method = '3D geometric transform'
            else:
                result = self._pixel_to_galvo_simple(pixel_x, pixel_y, image_width, image_height)
                method = 'Simple mapping'
        except Exception as exc:  # pragma: no cover - defensive path
            rospy.logdebug(f'pixel_to_galvo_code failed, fallback to simple mapping: {exc}')
            result = self._pixel_to_galvo_simple(pixel_x, pixel_y, image_width, image_height)
            method = 'Simple mapping'

        self.transform_method_used = method
        self.transform_valid = result is not None
        return result

    def _pixel_to_galvo_simple(self, pixel_x: float, pixel_y: float, image_width: int, image_height: int) -> Tuple[int, int]:
        mapping = self.params.get('simple_mapping', {})
        use_safe = bool(mapping.get('use_safe_range', True))
        max_range = float(mapping.get('max_safe_range' if use_safe else 'protocol_max', 32767))
        offset_x = float(mapping.get('offset_x', 0.0))
        offset_y = float(mapping.get('offset_y', 0.0))

        norm_x = pixel_x / float(image_width) - 0.5
        norm_y = pixel_y / float(image_height) - 0.5
        scale = max_range * 2.0

        galvo_x = np.clip(norm_x * scale + offset_x, -max_range, max_range)
        galvo_y = np.clip(norm_y * scale + offset_y, -max_range, max_range)

        codes = (int(round(galvo_x)), int(round(galvo_y)))
        self.last_pixel_pos = (pixel_x, pixel_y)
        self.last_galvo_pos = codes
        return codes

    def _pixel_to_galvo_3d(self, pixel_x: float, pixel_y: float) -> Optional[Tuple[int, int]]:
        depth = None
        if callable(self.depth_query_func):
            depth = self.depth_query_func(pixel_x, pixel_y)
        point_g = None
        if depth and depth > 0:
            point_g = self.pixel_depth_to_point_galvo(pixel_x, pixel_y, depth)
        if point_g is None:
            ray_dir_cam = self._camera_ray_from_pixel(pixel_x, pixel_y)
            if ray_dir_cam is None:
                return None
            ray_origin_g = self.t_gc
            ray_dir_g = self.R_gc @ ray_dir_cam
            point_g = self._ray_plane_intersection(ray_origin_g, ray_dir_g)
        if point_g is None:
            return None

        self.last_hit_point_g = point_g.copy()
        theta_x, theta_y = self._angles_from_point(point_g)
        code_x, code_y = self.angles_to_codes(theta_x, theta_y)
        codes = (int(round(code_x)), int(round(code_y)))
        self.last_pixel_pos = (pixel_x, pixel_y)
        self.last_galvo_pos = codes
        return codes

    def _camera_ray_from_pixel(self, pixel_x: float, pixel_y: float) -> Optional[np.ndarray]:
        try:
            x = (pixel_x - self.K[0, 2]) / self.K[0, 0]
            y = (pixel_y - self.K[1, 2]) / self.K[1, 1]
            ray = np.array([x, y, 1.0], dtype=np.float64)
            norm = np.linalg.norm(ray)
            return ray / norm if norm > 1e-9 else None
        except Exception:  # pragma: no cover - defensive path
            return None

    def _ray_plane_intersection(self, origin: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        denom = np.dot(self.n_g, direction)
        if abs(denom) < 1e-9:
            return None
        t = -(np.dot(self.n_g, origin) + self.d_g) / denom
        if t <= 0:
            return None
        return origin + t * direction

    def _angles_from_point(self, point_g: np.ndarray) -> Tuple[float, float]:
        x, y, z = point_g
        if abs(z) < 1e-9:
            return 0.0, 0.0
        return float(np.arctan2(x, z)), float(np.arctan2(y, z))

    # ------------------------------------------------------------------
    # Code ↔ angle conversions
    # ------------------------------------------------------------------

    def angles_to_codes(self, theta_x: float, theta_y: float) -> Tuple[float, float]:
        profile = self.active_profile
        galvo = profile.galvo_params
        limits = profile.axis_angle_limits

        theta_x_deg = np.degrees(theta_x) * galvo.get('scale_x', 1.0) + galvo.get('bias_x', 0.0)
        theta_y_deg = np.degrees(theta_y) * galvo.get('scale_y', 1.0) + galvo.get('bias_y', 0.0)

        def to_norm(angle_deg: float, neg_limit: float, pos_limit: float) -> float:
            limit = pos_limit if angle_deg >= 0 else neg_limit
            return angle_deg / limit if abs(limit) > 1e-9 else 0.0

        norm_x = to_norm(theta_x_deg, limits['x_minus'], limits['x_plus'])
        norm_y = to_norm(theta_y_deg, limits['y_minus'], limits['y_plus'])

        base_x = np.clip(norm_x * profile.max_code, -profile.max_code, profile.max_code)
        base_y = np.clip(norm_y * profile.max_code, -profile.max_code, profile.max_code)

        code_x = base_x * profile.code_scale[0] + profile.code_offset[0]
        code_y = base_y * profile.code_scale[1] + profile.code_offset[1]

        code_x = np.clip(code_x, profile.code_limits[0, 0], profile.code_limits[0, 1])
        code_y = np.clip(code_y, profile.code_limits[1, 0], profile.code_limits[1, 1])

        return float(code_x), float(code_y)

    def codes_to_angles(self, code_x: float, code_y: float) -> Tuple[float, float]:
        profile = self.active_profile
        galvo = profile.galvo_params
        limits = profile.axis_angle_limits

        scale_x = profile.code_scale[0] if abs(profile.code_scale[0]) > 1e-9 else 1.0
        scale_y = profile.code_scale[1] if abs(profile.code_scale[1]) > 1e-9 else 1.0

        base_x = (code_x - profile.code_offset[0]) / scale_x
        base_y = (code_y - profile.code_offset[1]) / scale_y

        base_x = np.clip(base_x, -profile.max_code, profile.max_code)
        base_y = np.clip(base_y, -profile.max_code, profile.max_code)

        norm_x = base_x / profile.max_code if profile.max_code else 0.0
        norm_y = base_y / profile.max_code if profile.max_code else 0.0

        def from_norm(norm_value: float, neg_limit: float, pos_limit: float) -> float:
            limit = pos_limit if norm_value >= 0 else neg_limit
            return norm_value * limit

        angle_x_deg = from_norm(norm_x, limits['x_minus'], limits['x_plus'])
        angle_y_deg = from_norm(norm_y, limits['y_minus'], limits['y_plus'])

        angle_x_deg = (angle_x_deg - galvo.get('bias_x', 0.0)) / galvo.get('scale_x', 1.0)
        angle_y_deg = (angle_y_deg - galvo.get('bias_y', 0.0)) / galvo.get('scale_y', 1.0)

        return np.radians(angle_x_deg), np.radians(angle_y_deg)

    # ------------------------------------------------------------------
    # Reverse transform: galvo codes → pixel
    # ------------------------------------------------------------------

    def galvo_code_to_pixel(
        self,
        galvo_x: float,
        galvo_y: float,
        image_width: int = 640,
        image_height: int = 480,
    ) -> Optional[Tuple[float, float]]:
        if self.use_3d_transform:
            pixel = self._galvo_code_to_pixel_3d(galvo_x, galvo_y)
        else:
            pixel = self.galvo_code_to_pixel_simple(galvo_x, galvo_y, image_width, image_height)
        return pixel

    def _galvo_code_to_pixel_3d(self, galvo_x: float, galvo_y: float) -> Optional[Tuple[float, float]]:
        theta_x, theta_y = self.codes_to_angles(galvo_x, galvo_y)
        point_g = self._choose_reverse_point(theta_x, theta_y)
        if point_g is None:
            return None
        camera_point = self._galvo_point_to_camera(point_g)
        if camera_point is None:
            return None
        return self._camera_point_to_pixel(camera_point)

    def _choose_reverse_point(self, theta_x: float, theta_y: float) -> Optional[np.ndarray]:
        if isinstance(self.last_hit_point_g, np.ndarray):
            z_ref = float(self.last_hit_point_g[2])
            return self.galvo_angles_to_point_depth_cam(theta_x, theta_y, z_ref)
        if self.fixed_reverse_depth_z_mm is not None:
            return self.galvo_angles_to_point_at_depth(theta_x, theta_y, float(self.fixed_reverse_depth_z_mm))
        return self.galvo_angles_to_point_on_plane(theta_x, theta_y)

    def _galvo_point_to_camera(self, point_g: np.ndarray) -> Optional[np.ndarray]:
        relative = point_g - self.t_gc
        return self.R_gc.T @ relative

    def _camera_point_to_pixel(self, camera_point: np.ndarray) -> Optional[Tuple[float, float]]:
        z = camera_point[2]
        if abs(z) < 1e-9:
            return None
        x = camera_point[0] / z
        y = camera_point[1] / z
        pixel_x = x * self.K[0, 0] + self.K[0, 2]
        pixel_y = y * self.K[1, 1] + self.K[1, 2]
        return (float(pixel_x), float(pixel_y))

    def galvo_code_to_pixel_simple(
        self,
        galvo_x: float,
        galvo_y: float,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float]:
        mapping = self.params.get('simple_mapping', {})
        use_safe = bool(mapping.get('use_safe_range', True))
        max_range = float(mapping.get('max_safe_range' if use_safe else 'protocol_max', 32767))
        offset_x = float(mapping.get('offset_x', 0.0))
        offset_y = float(mapping.get('offset_y', 0.0))

        norm_x = (galvo_x - offset_x) / (max_range * 2.0)
        norm_y = (galvo_y - offset_y) / (max_range * 2.0)
        pixel_x = (norm_x + 0.5) * image_width
        pixel_y = (norm_y + 0.5) * image_height
        return (float(pixel_x), float(pixel_y))

    # ------------------------------------------------------------------
    # Geometry utilities reused by calibrators
    # ------------------------------------------------------------------

    def pixel_depth_to_point_galvo(self, pixel_x: float, pixel_y: float, depth_m: float) -> Optional[np.ndarray]:
        if depth_m is None or depth_m <= 0:
            return None
        x_c = (pixel_x - self.K[0, 2]) / self.K[0, 0] * depth_m * 1000.0
        y_c = (pixel_y - self.K[1, 2]) / self.K[1, 1] * depth_m * 1000.0
        z_c = depth_m * 1000.0
        point_c = np.array([x_c, y_c, z_c], dtype=np.float64)
        return self.R_gc @ point_c + self.t_gc

    def galvo_angles_to_point_on_plane(self, theta_x: float, theta_y: float) -> Optional[np.ndarray]:
        direction = self._direction_from_angles(theta_x, theta_y)
        return self._ray_plane_intersection(np.zeros(3, dtype=np.float64), direction)

    def galvo_angles_to_point_depth_cam(self, theta_x: float, theta_y: float, z_ref_mm: float) -> np.ndarray:
        tx = np.tan(theta_x)
        ty = np.tan(theta_y)
        return np.array([-z_ref_mm * tx, -z_ref_mm * ty, z_ref_mm], dtype=np.float64)

    def galvo_angles_to_point_at_depth(self, theta_x: float, theta_y: float, z_ref_mm: float) -> Optional[np.ndarray]:
        direction = self._direction_from_angles(theta_x, theta_y)
        if abs(direction[2]) < 1e-9:
            return None
        t = (z_ref_mm - 0.0) / direction[2]
        if t <= 0:
            return None
        return direction * t

    @staticmethod
    def _direction_from_angles(theta_x: float, theta_y: float) -> np.ndarray:
        dir_vec = np.array([np.tan(theta_x), np.tan(theta_y), 1.0], dtype=np.float64)
        norm = np.linalg.norm(dir_vec)
        return dir_vec / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    def switch_transform_mode(self, use_3d_transform: bool) -> None:
        self.use_3d_transform = bool(use_3d_transform)

    def update_camera_matrix(self, camera_info_msg) -> None:
        try:
            self.K = np.array(camera_info_msg.K, dtype=np.float64).reshape(3, 3)
            self.D = np.array(camera_info_msg.D, dtype=np.float64)
            self.use_distortion = bool(len(self.D))
        except Exception as exc:
            rospy.logwarn(f'Failed to update camera matrix from CameraInfo: {exc}')

    def get_transform_info(self) -> Dict[str, object]:
        return {
            'active_profile': self.get_profile_metadata(self.active_profile_index),
            'transform_valid': self.transform_valid,
            'transform_method': self.transform_method_used,
            'use_3d_transform': self.use_3d_transform,
            'last_pixel_pos': self.last_pixel_pos,
            'last_galvo_pos': self.last_galvo_pos,
            'fixed_reverse_depth_z_mm': self.fixed_reverse_depth_z_mm,
        }


if __name__ == '__main__':  # pragma: no cover - manual smoke test
    transform = CameraGalvoTransform()
    print('Loaded', transform.get_galvo_profile_count(), 'galvo profile(s) from', transform.config_file_path)

"""Visualization helpers shared by calibration and main nodes."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def draw_galvo_crosshair(frame: np.ndarray, position: Tuple[int, int], color, label: str) -> None:
    x, y = int(position[0]), int(position[1])
    cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    cv2.putText(
        frame,
        label,
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


__all__ = ["draw_galvo_crosshair"]

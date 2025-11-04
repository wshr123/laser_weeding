"""Serial controller for the Teensy XY2-100 firmware."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import rospy
import serial


class XY2_100Controller:
    """Minimal controller for one or more XY2-100 galvo heads."""

    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 115200, galvo_count: int = 1):
        self.port = port
        self.baudrate = baudrate
        self.serial_port: Optional[serial.Serial] = None
        self.galvo_count = max(1, galvo_count)
        self._active_galvo = 0
        self.current_positions = [[0, 0] for _ in range(self.galvo_count)]
        self.galvo_limits: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [
            ((-32767, 32767), (-32767, 32767)) for _ in range(self.galvo_count)
        ]
        self.laser_enabled = False
        self.connected = False
        self.connect()

    # ------------------------------------------------------------------
    # Serial connection helpers
    # ------------------------------------------------------------------

    def connect(self) -> None:
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            rospy.loginfo(f"connected to {self.port}")
            self._drain_startup_banner()
            time.sleep(0.5)
            self.initialize_galvo()
            self.connected = True
        except serial.SerialException as exc:
            rospy.logerr(f"failed to connect to {self.port}: {exc}")
            self.serial_port = None
            self.connected = False

    def _drain_startup_banner(self) -> None:
        if not self.serial_port:
            return
        time.sleep(0.1)
        banner = self.serial_port.read_all()
        if banner:
            rospy.loginfo(banner.decode("utf-8", errors="ignore").strip())

    def is_connected(self) -> bool:
        if not self.serial_port:
            self.connected = False
        else:
            self.connected = bool(self.serial_port.is_open)
        return self.connected

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------

    def send_command(self, command: str) -> bool:
        if not self.serial_port:
            return False
        try:
            payload = (command + "\n").encode("utf-8")
            self.serial_port.write(payload)
            self.serial_port.flush()
            return True
        except Exception as exc:  # pylint: disable=broad-except
            rospy.logerr(f"failed to send command '{command}': {exc}")
            return False

    def initialize_galvo(self) -> None:
        for galvo_index in range(self.galvo_count):
            self.move_to_center(galvo_index)
        self.laser_off()

    def select_galvo(self, galvo_index: int) -> bool:
        if galvo_index < 0 or galvo_index >= self.galvo_count:
            return False
        if galvo_index == self._active_galvo:
            return True
        if self.send_command(f"GALVO:{galvo_index + 1}"):
            self._active_galvo = galvo_index
            return True
        return False

    def move_to_position(self, x: int, y: int, galvo_index: Optional[int] = None) -> None:
        idx = self._sanitize_galvo_index(galvo_index)
        (x_min, x_max), (y_min, y_max) = self.galvo_limits[idx]
        x_clamped = max(x_min, min(x_max, int(x)))
        y_clamped = max(y_min, min(y_max, int(y)))
        prefix = f"XY{idx + 1}"
        if self.send_command(f"{prefix}:{x_clamped},{y_clamped}"):
            self.current_positions[idx] = [x_clamped, y_clamped]

    def move_to_center(self, galvo_index: Optional[int] = None) -> None:
        self.move_to_position(0, 0, galvo_index)

    def laser_on(self) -> None:
        if self.send_command("LASER:ON"):
            self.laser_enabled = True

    def laser_off(self) -> None:
        if self.send_command("LASER:OFF"):
            self.laser_enabled = False

    def set_laser_mode(self, mode: str) -> bool:
        mode_upper = mode.strip().upper()
        if mode_upper not in {"POINT", "SPIRAL"}:
            return False
        return self.send_command(f"MODE:{mode_upper}")

    def configure_limits(self, galvo_index: int, x_min: int, x_max: int, y_min: int, y_max: int) -> None:
        if galvo_index < 0 or galvo_index >= self.galvo_count:
            return
        self.galvo_limits[galvo_index] = ((x_min, x_max), (y_min, y_max))
        self.send_command(
            f"LIMITS:{galvo_index + 1},{x_min},{x_max},{y_min},{y_max}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sanitize_galvo_index(self, galvo_index: Optional[int]) -> int:
        if galvo_index is None:
            return self._active_galvo
        return max(0, min(self.galvo_count - 1, galvo_index))


__all__ = ["XY2_100Controller"]

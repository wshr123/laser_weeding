#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
import argparse
import serial
import serial.tools.list_ports as list_ports

FW_READY_PREFIX = b"READY"
DEFAULT_BAUD = 115200

def find_teensy_port(preferred=None):
    if preferred:
        return preferred
    ports = list(list_ports.comports())
    # 尝试匹配常见特征（描述里含 Teensy 或 USB Serial），否则返回第一个 ACM/USB 串口
    for p in ports:
        desc = f"{p.device} {p.description}".lower()
        if "teensy" in desc or "usb serial" in desc:
            return p.device
    for p in ports:
        if "ttyACM" in p.device or "ttyUSB" in p.device or "COM" in p.device:
            return p.device
    return None

def open_port(port, baud=DEFAULT_BAUD, timeout=1.0):
    ser = serial.Serial(port=port, baudrate=baud, timeout=timeout)
    # 清空缓冲
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser

def read_line(ser, timeout=2.0):
    end = time.time() + timeout
    buf = b""
    while time.time() < end:
        ch = ser.read(1)
        if ch == b"":
            continue
        if ch in (b"\n", b"\r"):
            if buf:
                return buf.decode(errors="replace").strip()
        else:
            buf += ch
    return None

def wait_ready(ser, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        line = read_line(ser, timeout=0.5)
        if line:
            if line.startswith("READY"):
                print(f"[PC] {line}")
                return True
            else:
                print(f"[PC] {line}")
    return False

def send_cmd(ser, cmd, expect_ok=True, timeout=2.0):
    if not cmd.endswith("\n"):
        cmd += "\n"
    ser.write(cmd.encode())
    ser.flush()
    line = read_line(ser, timeout=timeout)
    if line is None:
        raise RuntimeError(f"Timeout waiting response for: {cmd.strip()}")
    print(f"[PC] -> {cmd.strip()}")
    print(f"[PC] <- {line}")
    if expect_ok and not (line.startswith("OK") or line == "PONG"):
        raise RuntimeError(f"Unexpected response for {cmd.strip()}: {line}")
    return line

def ping(ser, n=3):
    rtt = []
    for i in range(n):
        t0 = time.time()
        ser.write(b"PING\n")
        ser.flush()
        line = read_line(ser, timeout=1.0)
        t1 = time.time()
        if line != "PONG":
            raise RuntimeError(f"PING failed, got: {line}")
        rtt.append((t1 - t0) * 1000.0)
        print(f"[PING] {i+1}/{n} RTT = {rtt[-1]:.2f} ms")
        time.sleep(0.1)
    if rtt:
        print(f"[PING] avg RTT = {sum(rtt)/len(rtt):.2f} ms")

def center(ser):
    send_cmd(ser, "CENTER")

def status(ser):
    send_cmd(ser, "STATUS", expect_ok=False)

def set_laser(ser, on):
    send_cmd(ser, f"LASER:{1 if on else 0}")

def set_xy(ser, x, y):
    x = max(0, min(65535, int(x)))
    y = max(0, min(65535, int(y)))
    send_cmd(ser, f"XY:{x},{y}")

def set_xys(ser, sx, sy):
    sx = max(-32768, min(32767, int(sx)))
    sy = max(-32768, min(32767, int(sy)))
    send_cmd(ser, f"XYS:{sx},{sy}")

def run_square(ser, size=15000, dwell_ms=200):
    # 围绕中心跑一个正方形四角
    cx, cy = 32768, 32768
    pts = [
        (cx - size, cy - size),
        (cx + size, cy - size),
        (cx + size, cy + size),
        (cx - size, cy + size),
    ]
    print("[TEST] Running square path...")
    for (x, y) in pts:
        set_xy(ser, x, y)
        time.sleep(dwell_ms / 1000.0)
    set_xy(ser, cx, cy)

def main():
    ap = argparse.ArgumentParser(description="Teensy XY2-100 serial test client")
    ap.add_argument("--port", help="Serial port (e.g. /dev/ttyACM0 or COM3)")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--square", action="store_true", help="Run square path demo")
    ap.add_argument("--laser", type=int, choices=[0,1], help="Toggle laser gate")
    args = ap.parse_args()

    port = find_teensy_port(args.port)
    if not port:
        print("No serial port found. Specify with --port", file=sys.stderr)
        sys.exit(1)

    print(f"[PC] Opening {port} @ {args.baud} ...")
    with open_port(port, args.baud, timeout=0.2) as ser:
        # 等待 READY（若已错过，可发送 STATUS/HELP 测试）
        ready = wait_ready(ser, timeout=2.0)
        if not ready:
            print("[PC] No READY banner, trying STATUS ...")
            try:
                status(ser)
            except Exception as e:
                print(f"[PC] STATUS failed: {e}")

        # 基础 PING
        ping(ser, n=3)

        # 可选：激光门控（默认建议保持 0）
        if args.laser is not None:
            set_laser(ser, bool(args.laser))

        # 居中
        center(ser)

        # 跑正方形
        if args.square:
            run_square(ser, size=15000, dwell_ms=250)

        # 结束状态
        status(ser)

if __name__ == "__main__":
    main()
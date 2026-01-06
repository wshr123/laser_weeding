# agents.md: Project Documentation

This document outlines the file structure, code style conventions, and focused code testing methodology for the `laser_weeding-depth_cam` project.

## 1. File Directory Structure

The project adheres to a standard ROS package structure, focusing on modularity for easy maintenance and scaling.
laser_weeding-depth_cam/ ├── cam_params.yaml # Configuration: Camera-to-Galvo extrinsic and intrinsic parameters. ├── launch/ # ROS launch files for system setup and testing. │ ├── calibration.launch # Launches the 3D depth calibration node. │ ├── main.launch # Main online system launch (RealSense + YOLO + Control). │ ├── main_offline.launch # Offline test launch using ROS bag data. │ └── track_one_object_kalman.launch # Predictive single-target tracking test. ├── scripts/ # Python executables (ROS Nodes and Utilities). │ ├── coordinate_transform.py # Core 3D geometry: Pixel-Depth-to-Galvo Code mapping. │ ├── detector.py # Computer Vision: YOLO model wrapper and WeedTracker logic. │ ├── galvo_calibrator_depth.py # ROS Node: Manual 3D calibration process (SVD-based). │ ├── main.py # ROS Node: Main scheduling, prediction, and execution logic. │ ├── send_to_teensy.py # Utility: Serial communication class for XY2-100 Galvo. │ ├── test_svd.py # Unit Test: SVD rigid body transform algorithm verification. │ └── test_teensy.py # Utility: Basic serial connectivity test (PING/XY commands). └── README.md # Project overview (Intel RealSense focus).
## 2. Code Style Guidelines

All Python code must follow PEP 8 standards with specific ROS conventions for readability and interoperability.

### A. Python & PEP 8 Standards
1.  **Indentation:** 4 spaces.
2.  **Naming Convention:** Classes use `CamelCase`, functions and variables use `snake_case`.
3.  **Type Hinting:** Required for public API methods within key modules.
4.  **String Formatting:** Use f-strings for readability in debug and log statements.

### B. ROS Node Conventions
1.  **Shebang:** Use `#!/usr/bin/env python` or `python3`.
2.  **Encoding:** Specify `# -*- coding: utf-8 -*-`.
3.  **Logging:** All runtime output must use `rospy.loginfo`, `rospy.logwarn`, or `rospy.logerr` for proper ROS integration and log filtering.

## 3. Code Testing Methods (Code-Centric Focus)

Testing is segmented into Unit, Module, and Integration phases to ensure algorithmic correctness and system stability.

### A. Unit Tests (Algorithmic Verification)

| Test Target | Description | Core Files/Scripts |
| :--- | :--- | :--- |
| **SVD Rigid Transform** | Verifies the mathematical integrity of the `find_rigid_transform_3d` function by testing its ability to recover known ground truth rotation $R$ and translation $t$ from generated point clouds, ensuring minimal RMSE. | `scripts/test_svd.py` |
| **Galvo Angle/Code Mapping** | Tests the bi-directional conversion functions (`codes_to_angles`, `angles_to_codes` in `coordinate_transform.py`) to confirm zero-point, scale factors, and boundary clipping are applied correctly. | `scripts/coordinate_transform.py` |

### B. Module Tests (Component Integrity)

| Test Target | Description | Core Files/Scripts |
| :--- | :--- | :--- |
| **3D Transform Logic** | Tests `CameraGalvoTransform` class methods using mock camera parameters and known physical coordinates, validating the output code for edge cases (e.g., saturation) and verifying inverse mapping symmetry. | `scripts/coordinate_transform.py` |
| **Detector Consistency** | Tests the `WeedTracker` logic in `detector.py` by providing sequential synthetic bounding boxes to verify ID assignment consistency, frame-skipping logic, and tracking quality score calculation. | `scripts/detector.py` |
| **Serial Command Encapsulation** | Tests the `XY2_100Controller` in debug/mock mode to ensure command strings (`XY:x,y`, `LASER:ON`) are formatted correctly and coordinate clamping/conversion is performed before transmission. | `scripts/send_to_teensy.py` |

### C. Integration Tests (System Flow and Communication)

| Test Target | Description | Core Files/Scripts |
| :--- | :--- | :--- |
| **Hardware Handshake** | Executes `test_teensy.py` to confirm stable serial port connection, successful `READY` banner reception, and low-latency `PING/PONG` round-trip time, validating the physical layer connection. | `scripts/test_teensy.py` |
| **Offline Pipeline Validation** | Launches `main_offline.launch` to replay a pre-recorded ROS bag. The test monitors the `/galvo_xy` topic output to verify that the `main.py` control node correctly processes recorded data and publishes prediction-corrected galvo commands. | `launch/main_offline.launch` |
| **3D Calibration Procedure** | Tests the full interactive workflow of `galvo_calibrator_depth.py` by launching `calibration.launch`, ensuring manual keyboard input controls the galvo position correctly, depth querying works, and the final result YAML file contains the calculated $R, t$ extrinsics. | `launch/calibration.launch`, `scripts/galvo_calibrator_depth.py` |

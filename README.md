# Laser Weeding 深度相机平台

该仓库提供基于 Intel RealSense 的双振镜激光除草方案。项目主要由以下部分组成：

- `launch/`：ROS Launch 文件。
- `scripts/`：ROS 节点与核心算法模块。
- `cam_params.yaml`：相机到振镜的配置文件（含每个振镜的粗/精外参、扫描角范围、码值限制等）。

## 运行前准备
1. 确保 `cam_params.yaml` 中已经为每个振镜配置正确的 `galvos` 条目，包括：
   - `rough_extrinsics` 与 `refined_extrinsics` 四组外参（分别对应左右振镜的粗测/精校值）。
   - `galvo_params` 中的正负扫描角、比例系数与零点偏移；不同振镜可以使用不同角度范围。
   - `code_limits`、`max_code` 等码值限制，用于和 Teensy 控制器保持一致。
2. 如果存在手工标定结果，可在 `manual_calibration` 区域开启并指定结果文件，运行时会自动覆盖对应振镜的外参与限幅。
3. Teensy 固件需使用 `XY2-100_multi` 库并支持串口指令 `READY`、`PING`、`GALVO`、`XY{n}`、`LASER`、`MODE` 与 `LIMITS`。

## Launch 文件
### `calibration.launch`
用于深度摄像头辅助的手动标定流程。

- `galvo_index`：选择当前要标定的振镜索引（从 `0` 开始，示例中为右侧振镜 `1`）。
- `transform_config_file`：传入 `cam_params.yaml` 的路径，脚本会根据 `galvos` 中的参数初始化坐标变换。
- `calibration_result_file`：标定完成后写入的结果文件，可在 `manual_calibration` 中引用。

运行方式：
```bash
roslaunch laser_weeding calibration.launch
```
键盘控制与标定结果会在终端给出提示，图像窗口会同时显示两个振镜的瞄准位置。

### `main.launch`
在线运行主程序：读取 RealSense 图像与深度、调用检测模型、调度两个振镜并发送 XY2-100 指令。

- 模型与检测参数可通过 `model_type`、`model_path`、`device`、`weed_class_id`、`confidence_threshold` 配置。
- 激光时序由 `aiming_time` 和 `laser_time` 控制，单位为秒。
- 双振镜调度可调整 `galvo_count`、`galvo_split_axis`、`galvo_split_ratio` 与 `galvo_overlap_px`，用于划分图像负责区域。
- 坐标变换默认启用 3D 模式，若只需 2D 反投影，可在参数服务器设置 `use_3d_transform:=false`。

运行方式：
```bash
roslaunch laser_weeding main.launch
```

### `main_offline.launch`
离线仿真流程，利用 rosbag 回放图像，主节点在 2D 模式下绘制双振镜指向以便快速验证调度逻辑。

- 通过 `bag_image_publisher` 节点指定 `~bag_path`、`~image_topic_in` 等参数即可播放离线数据。
- 主节点将 `use_reverse_projection` 设置为 `false`，直接在图像平面展示两个振镜的命中点。

运行方式：
```bash
roslaunch laser_weeding main_offline.launch bag_path:=/path/to/file.bag
```

## 代码结构简化说明
- 共享模块放在 `scripts/core/`：包括配置解析、坐标变换、串口控制与可视化工具。
- ROS 节点脚本保持轻量，只关注自身逻辑，避免重复的条件判断与注释代码。
- 如需进一步扩展功能，请在 `core` 模块中新建子模块，并在节点脚本中引用。


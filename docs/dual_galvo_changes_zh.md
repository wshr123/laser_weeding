# 双振镜与螺旋灼烧改动说明

下列内容总结了提交 `Support dual galvo routing and calibration` 中对固件与上位机的关键改动，并解释这些改动所解决的问题与实现原理。

## 1. Teensy 固件 `control_tennsy.ino`
- **核心状态与安全边界数组化**：所有位置、偏移、校准、心跳等状态变量统一改为按振镜索引存储，并额外维护 `x_min_limits / x_max_limits / y_min_limits / y_max_limits` 四组数组用于记录每个振镜独立的码值边界，默认值为 ±32767。【F:control_tennsy.ino†L41-L103】
- **串口新增 `LIMITS` 指令**：`parseCommand` 支持 `LIMITS:n:xmin,xmax,ymin,ymax`，在下位机侧调用 `setGalvoLimits` 调整对应振镜的边界并即时回写，同时对 `XY` 指令的目标值统一走 `clampToGalvoRange` 检查，防止越界命令落地。【F:control_tennsy.ino†L221-L319】【F:control_tennsy.ino†L741-L844】
- **螺旋灼烧与限幅联动**：`drawSpiral` 在生成轨迹时引用新边界自动收缩最大半径，并对每个采样点调用限幅函数，保证新增的螺旋模式不会超过配置的安全范围。【F:control_tennsy.ino†L741-L788】
- **持续的保活与状态管理**：保留多头平滑插值、保活定时器与急停逻辑，确保每个振镜在新边界生效后仍能收到心跳与超时保护。【F:control_tennsy.ino†L208-L370】

## 2. 坐标映射 `scripts/coordinate_transform.py`
- **`GalvoHeadProfile` 扩展字段**：配置驱动的 `GalvoHeadProfile` 现保存每个振镜的外参、码值缩放、偏移、`max_code`、二维 `code_limits` 区间，以及整合后的 `galvo_params` 与逐轴角度上限，初始化时同步到运行态缓存。【F:scripts/coordinate_transform.py†L30-L118】【F:scripts/coordinate_transform.py†L200-L321】
- **统一的限幅/元数据接口**：提供 `get_galvo_profile_count`、`get_code_limits`、`get_profile_metadata` 与 `list_galvo_profiles`，供上位机和标定工具检索每个振镜的独立边界与补偿参数；`get_transform_info` 会回传激活 profile 的角度/码值信息。【F:scripts/coordinate_transform.py†L337-L374】【F:scripts/coordinate_transform.py†L839-L855】
- **角度/码值双向转换带 Profile**：`angles_to_codes` 与 `codes_to_angles` 在计算后引用各自 profile 的缩放、偏移、`max_code` 与角度限值，保证正反向转换都受配置文件约束，与保存在控制器/固件中的上限一致。【F:scripts/coordinate_transform.py†L632-L807】
- **非对称扫描角映射**：`_compute_axis_angle_limits` 支持每个振镜独立的 `scan_angle_x_plus` / `scan_angle_x_minus` / `scan_angle_y_plus` / `scan_angle_y_minus`，并在 profile 切换时更新缓存，使不同方向的最大角度都能正确换算成码值并参与限幅。【F:scripts/coordinate_transform.py†L208-L273】【F:scripts/coordinate_transform.py†L614-L807】
- **手工标定覆盖与报告**：读取 `cam_params.yaml` 中新增的 `manual_calibration` 配置后，会解析 `manual_galvo_calibration.yaml`，按振镜名称或编号套用手工标定的外参、偏置与码值范围，并记录残差统计供运行时检查。【F:cam_params.yaml†L78-L85】【F:scripts/coordinate_transform.py†L194-L223】【F:scripts/coordinate_transform.py†L267-L455】

## 3. 主控制节点 `scripts/main.py`
- **按配置裁剪硬件数量**：根据 `CameraGalvoTransform` 返回的 profile 数量与 Teensy 实际支持数量计算 `galvo_count`，并将 `cam_params.yaml` 中的每头边界读入 `self.galvo_limits`，随后通过 `_configure_controller_limits` 主动下发到固件。【F:scripts/main.py†L92-L137】
- **统一的限幅与范围判定**：新增 `clamp_to_galvo_limits`、`is_within_galvo_limits` 等工具，将预测轨迹、实时调度及状态发布均约束在各自限幅内；`in_galvo_scan_range`、`galvo_control_loop`、`update_position_history` 等流程都调用该逻辑，确保软硬件边界一致。【F:scripts/main.py†L485-L758】
- **运行状态回传边界信息**：`publish_status` 增加 `galvo_limits` 字段，将每个振镜的 X/Y 码值范围发布到 ROS topic，便于上位机监控或调试配置。【F:scripts/main.py†L893-L927】
- **多振镜瞄准可视化**：`draw_info` 遍历所有振镜位置，按激活/未激活状态绘制不同颜色的十字光标；当 `use_reverse_projection` 为 `true` 时调用坐标变换把码值反投影成像素，否则退化为使用预测/调度阶段缓存的像素目标，让离线仿真可以直接以 2D 结果进行比对。【F:scripts/main.py†L45-L56】【F:scripts/main.py†L860-L939】
- **像素目标缓存**：`update_position_history` 在生成预测点时会记录各振镜当前瞄准的像素坐标，并在可视化阶段加锁读取作为兜底展示，确保即便反投影失败或者处于纯 2D 模式，也能看到每个振镜期望命中的画面位置。【F:scripts/main.py†L662-L695】【F:scripts/main.py†L887-L913】

## 4. 标定工具
- **单头偏移换算遵循限幅**：手动标定工具读取当前 profile 的 `galvo_params`、`max_code`、`code_scale` 与 `code_limits`，结合 `CameraGalvoTransform.get_axis_angle_limits(galvo_index)` 返回的正负扫描角，将码值偏移分轴按比例换算成角度并把最终结果写回 `manual_calibration.updated_galvo_params`，避免不同振镜共用相同上限。【F:scripts/galvo_calibrator.py†L653-L737】
- **三维标定写回独立参数**：`galvo_calibrator_depth.py` 生成更新配置时会深拷贝当前参数，并仅修改目标振镜的 `galvo_params.bias_x/bias_y`，确保多振镜独立角度与偏移不会互相覆盖。【F:scripts/galvo_calibrator_depth.py†L912-L925】
- **自动计算残差验证标定精度**：三维标定完成后立即把相机点云变换到振镜系，与实测点做差并输出 RMSE/最大误差，同时把每个样本的残差、均方统计写入结果文件，方便后续加载时核对精度。【F:scripts/galvo_calibrator_depth.py†L768-L868】

## 5. 位机串口控制 `scripts/send_to_teensy.py`
- **上位机缓存限幅并推送固件**：控制器维护 `self.galvo_limits`，在 `move_to_position` 前先做本地限幅，`configure_limits` 则向固件发送 `LIMITS` 命令并同步缓存，确保上下位机对安全范围认知一致。【F:scripts/send_to_teensy.py†L29-L137】【F:scripts/send_to_teensy.py†L218-L247】
- **螺旋模式配置**：提供 `set_laser_mode` 与 `configure_spiral`，使上位机可以切换点灼或螺旋灼烧，并调整半径、圈距和驻留时间等参数。【F:scripts/send_to_teensy.py†L180-L207】

- **按头定义工作范围**：`galvos` 列表为左右振镜分别记录外参、`max_code` 与 `code_limits`，供坐标映射、上位机与固件限幅使用，避免两只振镜共享相同的最大扫描范围。【F:cam_params.yaml†L33-L66】
- **每头独立角度参数**：每个 `galvos[].galvo_params` 保存独立的 `scan_angle`、各轴正负扫描角、比例因子与偏移；上位机按 profile 读取这些值进行角度与码值的正反转换。【F:cam_params.yaml†L41-L66】【F:scripts/coordinate_transform.py†L200-L321】
- **保留全局默认值**：顶层 `galvo_params` 继续提供缺省角度字段，作为未显式配置振镜或新 profile 的兜底值。【F:cam_params.yaml†L21-L31】【F:scripts/coordinate_transform.py†L208-L273】

## 6. 离线仿真回放

- **ROS bag 图像回放节点**：新增 `bag_image_publisher.py`，可读取指定 bag 文件中的图像/CameraInfo 话题，按设定播放速率重放并刷新时间戳，支持循环播放，方便在无实机摄像头时复现流程。【F:scripts/bag_image_publisher.py†L1-L146】
- **离线 Launch 配置**：`main_offline.launch` 改为启动新的 bag 回放节点，通过参数控制 bag 路径、输入输出话题与播放速率；同时将 `use_reverse_projection` 设为 `false`，关闭码值反投影，可直接以 2D 像素目标叠加离线画面。【F:launch/main_offline.launch†L74-L92】

以上改动共同实现了：
1. 一块 Teensy 同时驱动两个振镜并共享激光控制，同时保证每个振镜的安全码值范围独立可调。
2. 上位机根据图像区域为不同振镜分配目标，并在重叠区域内做优先级判断，所有调度都会尊重对应振镜的限幅。
3. 校准流程支持分别标定两个振镜，并将补偿/限幅写回配置文件供运行时直接读取。
4. 激光可以在单点与螺旋模式之间切换，以覆盖更大的灼烧面积，同时确保轨迹不超出配置的工作范围。

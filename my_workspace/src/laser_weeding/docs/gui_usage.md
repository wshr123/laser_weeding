# 激光除草系统 GUI 使用说明

## 概述

本 GUI 界面为激光除草系统提供完整的图形化控制功能，包括：
- 实时检测结果图像显示
- Launch 文件启动管理
- 手动振镜控制（双振镜独立控制）
- 系统统计信息显示（检测数、作业数、FPS等）

## 安装依赖

### 1. 安装 PyQt5

```bash
pip3 install PyQt5
```

### 2. 确保 ROS 环境已配置

```bash
source /opt/ros/<your_ros_distro>/setup.bash
source ~/my_workspace/devel/setup.bash
```

## 启动 GUI

### 方法 1: 使用启动脚本（推荐）

```bash
cd ~/my_workspace/src/laser_weeding/gui
./run_gui.sh
```

### 方法 2: 直接运行 Python 脚本

```bash
cd ~/my_workspace/src/laser_weeding/gui
python3 gui_main.py
```

## 界面功能说明

### 1. 检测结果图像窗口

- **位置**: 右侧主显示区域
- **功能**: 实时显示来自 `/det_img/image_raw` 话题的检测结果图像
- **特性**: 
  - 自动缩放以适应窗口大小
  - 保持图像宽高比
  - 实时更新（30 FPS）

### 2. Launch 启动选择

- **位置**: 左侧控制面板顶部
- **功能**: 
  - 选择并启动不同的 ROS launch 文件
  - 停止正在运行的 launch 进程
- **可用 Launch 文件**:
  - `main.launch`: 主系统（在线模式）
  - `main_offline.launch`: 主系统（离线测试模式）
  - `calibration.launch`: 标定系统
  - `track_one_object.launch`: 单目标跟踪
  - `track_one_object_kalman.launch`: 单目标跟踪（卡尔曼滤波）

**使用步骤**:
1. 从下拉框选择要启动的 launch 文件
2. 点击"启动"按钮
3. 启动后，"停止"按钮将变为可用
4. 点击"停止"按钮可停止当前运行的 launch 进程

**注意**: 
- 启动 launch 后，手动振镜控制将自动禁用
- 停止 launch 后，手动振镜控制将自动启用

### 3. 手动振镜控制

- **位置**: 左侧控制面板中部
- **功能**: 在没有 launch 运行时，手动控制两个振镜的位置
- **控制方式**:
  - **X/Y 轴滑块**: 拖动滑块调整振镜位置（范围: -32767 到 32767）
  - **X/Y 轴数值框**: 直接输入数值或使用上下箭头调整
  - **回中心按钮**: 快速将振镜移动到中心位置 (0, 0)
- **振镜选择**: 通过下拉框选择要控制的振镜（振镜 1 或 振镜 2）

**激光控制**:
- **开启激光**: 开启激光器
- **关闭激光**: 关闭激光器

**注意**: 
- 只有在没有 launch 进程运行时，手动控制才可用
- 手动控制命令通过 `/galvo_xy` 话题发布
- 激光控制通过 `/laser_control` 话题发布

### 4. 系统统计信息

- **位置**: 左侧控制面板中部
- **显示内容**:
  - **检测杂草数**: 系统检测到的杂草总数
  - **作业杂草数**: 已完成作业的杂草数量
  - **当前 FPS**: 图像处理帧率
  - **系统状态**: 当前系统状态（IDLE/TRACKING/FIRING）

**状态颜色**:
- **IDLE** (灰色): 空闲状态，等待目标
- **TRACKING** (橙色): 正在跟踪目标
- **FIRING** (红色): 激光照射中

## 界面特性

### 1. 自适应布局

- 使用 Qt 布局管理器，窗口大小改变时控件自动调整
- 支持窗口最大化、最小化和拉伸
- 不同分辨率屏幕自适应

### 2. 高分屏适配

- 启用 Qt 高 DPI 缩放
- 支持 2K/4K 屏幕，字体清晰不发虚

### 3. 现代化风格

- 深色主题，护眼设计
- 统一的控件样式
- 悬停效果和状态反馈

### 4. 配置持久化

- 自动保存窗口位置和大小
- 下次启动时自动恢复上次的窗口状态

### 5. 非阻塞 UI

- 所有 ROS 通信在独立线程中运行
- 界面操作不会卡顿
- 实时状态更新

## 键盘快捷键

- **Tab**: 在控件间切换焦点
- **Enter**: 激活当前焦点控件
- **Esc**: 取消当前操作

## 故障排除

### 1. GUI 无法启动

**问题**: 提示 "PyQt5 未安装"
**解决**: 
```bash
pip3 install PyQt5
```

### 2. ROS 节点初始化失败

**问题**: 提示 "ROS 初始化失败"
**解决**: 
- 确保已 source ROS 环境
- 检查 ROS_MASTER_URI 是否正确设置
- 确保 roscore 正在运行

### 3. 无法显示图像

**问题**: 图像窗口显示 "等待图像..."
**解决**: 
- 确保已启动相应的 launch 文件
- 检查 `/det_img/image_raw` 话题是否有数据发布
- 使用 `rostopic list` 检查话题是否存在

### 4. Launch 启动失败

**问题**: 点击启动后提示错误
**解决**: 
- 确保 ROS 环境正确配置
- 检查 launch 文件路径是否正确
- 查看终端错误信息

### 5. 手动振镜控制无响应

**问题**: 调整滑块后振镜不移动
**解决**: 
- 确保没有 launch 进程在运行
- 检查 ROS 节点是否正常运行
- 检查 `/galvo_xy` 话题是否有订阅者

## 技术架构

### 文件结构

```
gui/
├── __init__.py              # Python 包初始化文件
├── gui_main.py              # 主界面窗口
├── gui_ros_interface.py     # ROS 通信接口（QThread）
├── gui_components.py        # UI 组件（图像显示、统计、控制等）
└── run_gui.sh               # 启动脚本
```

### 核心组件

1. **MainWindow**: 主窗口类，管理整体布局和组件
2. **ROSInterfaceThread**: ROS 通信线程，负责订阅和发布话题
3. **LaunchManager**: Launch 文件管理器，负责启动和停止进程
4. **ImageDisplayWidget**: 图像显示组件
5. **StatisticsWidget**: 统计信息显示组件
6. **LaunchSelectorWidget**: Launch 选择组件
7. **ManualGalvoControlWidget**: 手动振镜控制组件

### ROS 话题

**订阅**:
- `/det_img/image_raw` (sensor_msgs/Image): 检测结果图像
- `/system_status` (std_msgs/String): 系统状态信息
- `/current_target` (std_msgs/String): 当前目标信息

**发布**:
- `/galvo_xy` (std_msgs/Int32MultiArray): 振镜控制命令
- `/laser_control` (std_msgs/Bool): 激光控制命令

## 开发说明

### 代码规范

- 遵循 PEP 8 代码风格
- 使用类型提示（Type Hints）
- 所有 UI 操作在主线程，ROS 通信在子线程
- 使用信号/槽机制进行线程间通信

### 扩展功能

如需添加新功能：

1. **添加新的 UI 组件**: 在 `gui/gui_components.py` 中创建新的 Widget 类
2. **添加新的 ROS 话题**: 在 `gui/gui_ros_interface.py` 的 `ROSInterfaceThread` 中添加订阅器或发布器
3. **添加新的 Launch 文件**: 将文件放入 `launch/` 目录，LaunchManager 会自动扫描

## 更新日志

### v1.0.0 (2024)
- 初始版本发布
- 实现基本功能：图像显示、Launch 管理、手动控制、统计信息
- 支持双振镜独立控制
- 深色主题界面
- 配置持久化


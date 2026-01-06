# GUI 模块说明

本目录包含激光除草系统的 PyQt5 图形界面所有相关文件。

## 文件结构

```
gui/
├── __init__.py              # Python 包初始化文件
├── gui_main.py              # 主界面窗口（入口文件）
├── gui_ros_interface.py     # ROS 通信接口（使用 QThread）
├── gui_components.py        # UI 组件（图像显示、统计、控制等）
├── run_gui.sh               # 启动脚本
└── README.md                # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip3 install PyQt5
```

### 2. 启动 GUI

```bash
cd ~/my_workspace/src/laser_weeding/gui
./run_gui.sh
```

## 模块说明

### gui_main.py
主界面窗口类 `MainWindow`，负责：
- 窗口布局和样式
- 组件初始化和连接
- 配置持久化
- 窗口状态管理

### gui_ros_interface.py
ROS 通信接口，包含：
- `ROSInterfaceThread`: ROS 通信线程类
  - 订阅 ROS 话题（图像、状态、目标）
  - 发布 ROS 话题（振镜控制、激光控制）
  - 非阻塞通信实现
- `LaunchManager`: Launch 文件管理器
  - 扫描 launch 目录
  - 启动/停止 launch 进程

### gui_components.py
UI 组件模块，包含：
- `ImageDisplayWidget`: 实时图像显示组件
- `StatisticsWidget`: 统计信息显示组件
- `LaunchSelectorWidget`: Launch 选择组件
- `ManualGalvoControlWidget`: 手动振镜控制组件

## 依赖关系

```
gui_main.py
  ├── gui_ros_interface.py (ROSInterfaceThread, LaunchManager)
  └── gui_components.py (各种 UI 组件)
```

## 注意事项

1. **Python 路径**: 启动脚本会自动将 `scripts/` 目录添加到 `PYTHONPATH`，以便导入项目其他模块（如果需要）
2. **ROS 环境**: 需要先 source ROS 环境
3. **Qt 库**: 如果使用 conda 环境，可能需要安装系统 Qt 库（见故障排除文档）

## 更多信息

- 详细使用说明: [../docs/gui_usage.md](../docs/gui_usage.md)
- 故障排除: [../docs/gui_troubleshooting.md](../docs/gui_troubleshooting.md)
- 快速开始: [../GUI_README.md](../GUI_README.md)


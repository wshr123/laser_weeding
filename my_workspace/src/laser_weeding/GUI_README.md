# 激光除草系统 GUI 界面

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

或者：

```bash
cd ~/my_workspace/src/laser_weeding/gui
python3 gui_main.py
```

### 3. 使用界面

1. **启动 ROS 核心**（如果还没有运行）:
   ```bash
   roscore
   ```

2. **在 GUI 中选择并启动 Launch 文件**:
   - 从下拉框选择 launch 文件（如 `main.launch`）
   - 点击"启动"按钮

3. **查看实时检测结果**:
   - 右侧窗口显示实时检测图像

4. **查看统计信息**:
   - 左侧面板显示检测数、作业数、FPS 等

5. **手动控制振镜**（可选）:
   - 停止 launch 后，可以使用手动控制面板
   - 调整滑块或输入数值控制振镜位置

## 文件说明

- `gui/gui_main.py`: 主界面窗口
- `gui/gui_ros_interface.py`: ROS 通信接口（使用 QThread）
- `gui/gui_components.py`: UI 组件（图像显示、统计、控制等）
- `gui/run_gui.sh`: 启动脚本
- `docs/gui_usage.md`: 详细使用说明

## 功能特性

✅ **实时图像显示** - 显示检测结果图像  
✅ **Launch 管理** - 启动/停止不同的 launch 文件  
✅ **手动振镜控制** - 双振镜独立控制  
✅ **统计信息** - 检测数、作业数、FPS、系统状态  
✅ **自适应布局** - 支持不同分辨率屏幕  
✅ **高分屏适配** - 2K/4K 屏幕支持  
✅ **深色主题** - 现代化 UI 设计  
✅ **配置持久化** - 自动保存窗口状态  

## 详细文档

更多详细信息请参考: [docs/gui_usage.md](docs/gui_usage.md)


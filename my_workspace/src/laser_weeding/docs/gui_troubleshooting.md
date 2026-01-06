# GUI 故障排除指南

## 问题 1: Qt platform plugin "xcb" 加载失败

### 错误信息
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "..."
This application failed to start because no Qt platform plugin could be initialized.
```

### 原因
- Conda 环境中的 PyQt5 与系统的 Qt 库冲突
- 缺少系统 Qt 库依赖

### 解决方案

#### 方案 1: 安装系统 Qt 库（推荐）
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-sync1 libxcb-xfixes0 libxcb-xkb1 libxkbcommon-x11-0

# 或者安装完整的 Qt5 库
sudo apt-get install -y qt5-default libqt5gui5 libqt5widgets5 libqt5core5a
```

#### 方案 2: 在 conda 环境中安装 Qt（如果使用 conda）
```bash
conda install -c conda-forge qt
pip install PyQt5
```

#### 方案 3: 使用系统 Python 而不是 conda（如果可能）
```bash
# 退出 conda 环境
conda deactivate

# 使用系统 Python 安装 PyQt5
pip3 install --user PyQt5

# 运行 GUI
python3 gui_main.py
```

#### 方案 4: 设置环境变量（临时解决）
```bash
# 在启动脚本中已经包含，但也可以手动设置
export QT_QPA_PLATFORM_PLUGIN_PATH=""
unset QT_PLUGIN_PATH
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# 然后运行 GUI
cd ~/my_workspace/src/laser_weeding/gui
python3 gui_main.py
```

## 问题 2: QObject::moveToThread 错误

### 错误信息
```
QObject::moveToThread: Current thread (0x...) is not the object's thread (0x...).
Cannot move to target thread (0x...)
```

### 原因
- 某些 Qt 对象（如 CvBridge）在主线程中创建，然后试图移动到子线程

### 解决方案
已修复：CvBridge 现在在线程的 run() 方法中创建，而不是在 __init__ 中。

如果仍然遇到此问题，请确保：
1. 所有 Qt 对象都在正确的线程中创建
2. 不要在 QThread 的 __init__ 中创建可能依赖线程的对象

## 问题 3: 核心转储 (Core Dump)

### 错误信息
```
已放弃 (核心已转储)
```

### 原因
- Qt platform plugin 加载失败导致程序崩溃
- 系统库缺失

### 解决方案
1. 先解决 Qt platform plugin 问题（见问题 1）
2. 检查系统库：
   ```bash
   ldd $(python3 -c "import PyQt5.QtCore; print(PyQt5.QtCore.__file__)") | grep "not found"
   ```
3. 安装缺失的库

## 问题 4: ROS 节点初始化失败

### 错误信息
```
ROS 初始化失败: ...
```

### 解决方案
1. 确保 roscore 正在运行：
   ```bash
   roscore
   ```
2. 检查 ROS 环境：
   ```bash
   echo $ROS_MASTER_URI
   echo $ROS_DISTRO
   ```
3. Source ROS 环境：
   ```bash
   source /opt/ros/<your_ros_distro>/setup.bash
   source ~/my_workspace/devel/setup.bash
   ```

## 问题 5: 图像不显示

### 可能原因
1. Launch 文件未启动
2. 话题没有数据发布
3. 话题名称不匹配

### 解决方案
1. 检查话题是否存在：
   ```bash
   rostopic list | grep det_img
   ```
2. 检查话题是否有数据：
   ```bash
   rostopic hz /det_img/image_raw
   ```
3. 确保已启动相应的 launch 文件

## 问题 6: 手动振镜控制无响应

### 可能原因
1. Launch 进程正在运行（手动控制被禁用）
2. ROS 节点未运行
3. 话题没有订阅者

### 解决方案
1. 停止所有 launch 进程
2. 检查 ROS 节点：
   ```bash
   rosnode list
   ```
3. 检查话题订阅者：
   ```bash
   rostopic info /galvo_xy
   ```

## 通用调试步骤

### 1. 检查依赖
```bash
# 检查 PyQt5
python3 -c "import PyQt5; print(PyQt5.__version__)"

# 检查 ROS
python3 -c "import rospy; print(rospy.__version__)"

# 检查 OpenCV
python3 -c "import cv2; print(cv2.__version__)"
```

### 2. 检查环境变量
```bash
echo $QT_QPA_PLATFORM_PLUGIN_PATH
echo $QT_PLUGIN_PATH
echo $LD_LIBRARY_PATH
```

### 3. 使用详细输出运行
```bash
cd ~/my_workspace/src/laser_weeding/gui
python3 gui_main.py 2>&1 | tee gui_debug.log
```

### 4. 检查系统日志
```bash
dmesg | tail -20
journalctl -xe | tail -20
```

## 联系支持

如果以上方法都无法解决问题，请提供：
1. 完整的错误信息
2. 系统信息（`uname -a`）
3. Python 版本（`python3 --version`）
4. 已安装的包列表（`pip3 list | grep -i qt`）
5. 环境变量（`env | grep -i qt`）


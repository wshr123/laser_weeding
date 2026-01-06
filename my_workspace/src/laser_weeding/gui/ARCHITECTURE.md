# Qt GUI 界面运行逻辑说明

## 一、整体架构

```
gui_main.py (主窗口)
    ├── gui_ros_interface.py (ROS 通信线程)
    │   ├── ROSInterfaceThread (QThread)
    │   └── LaunchManager (Launch 文件管理)
    └── gui_components.py (UI 组件)
        ├── ImageDisplayWidget (图像显示)
        ├── StatisticsWidget (统计信息)
        ├── LaunchSelectorWidget (Launch 选择)
        └── ManualGalvoControlWidget (手动振镜控制)
```

## 二、启动流程

### 1. 程序入口 (`main()` 函数)

```python
def main():
    # 1. 设置高 DPI 支持（必须在创建 QApplication 之前）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # 2. 创建 QApplication 实例
    app = QApplication(sys.argv)
    
    # 3. 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 4. 进入事件循环
    app.exec_()
```

### 2. 主窗口初始化 (`MainWindow.__init__`)

**执行顺序：**

1. **初始化 Launch 管理器**
   ```python
   self.init_launch_manager()
   # - 扫描 launch/ 目录
   # - 找到所有 .launch 文件
   # - 创建 LaunchManager 实例
   ```

2. **初始化 ROS 接口**
   ```python
   self.init_ros_interface()
   # - 创建 ROSInterfaceThread 实例（但不启动）
   # - 线程将在窗口显示后启动
   ```

3. **初始化 UI**
   ```python
   self.init_ui()
   # - 创建左侧控制面板
   #   - LaunchSelectorWidget
   #   - StatisticsWidget
   #   - ManualGalvoControlWidget
   # - 创建右侧图像显示面板
   #   - ImageDisplayWidget
   # - 创建状态栏
   ```

4. **连接信号和槽**
   ```python
   self.connect_signals()
   # - ROS 线程信号 → UI 组件
   # - UI 组件信号 → 主窗口处理函数
   ```

5. **恢复窗口状态**
   ```python
   self.restore_window_state()
   # - 从 QSettings 恢复窗口位置和大小
   ```

6. **启动状态更新定时器**
   ```python
   self.status_timer.start(100)  # 每 100ms 更新一次
   ```

### 3. 窗口显示事件 (`showEvent`)

当窗口显示时：

```python
def showEvent(self, event):
    # 启动 ROS 通信线程
    if not self.ros_thread.isRunning():
        self.ros_thread.start()
```

## 三、ROS 通信线程 (`ROSInterfaceThread`)

### 1. 线程初始化 (`run()` 方法)

```python
def run(self):
    # 1. 在线程中创建对象（避免线程问题）
    self.bridge = CvBridge()
    self.mutex = QMutex()
    
    # 2. 初始化 ROS 节点
    rospy.init_node('laser_weeding_gui', anonymous=True)
    
    # 3. 设置发布器
    self.setup_publishers()
    # - /galvo_xy (振镜控制)
    # - /laser_control (激光控制)
    
    # 4. 设置订阅器（初始）
    self.setup_subscribers()
    # - /det_img/image_raw (检测图像)
    # - /system_status (系统状态)
    # - /current_target (当前目标)
    
    # 5. 进入主循环
    while self.running:
        # 每 2 秒检查一次话题并自动重连
        check_and_reconnect_topics()
        rate.sleep()
```

### 2. 话题自动重连机制

```python
def check_and_reconnect_topics(self):
    # 每 2 秒执行一次
    available_topics = rospy.get_published_topics()
    
    # 如果话题存在但未订阅，则自动连接
    if '/det_img/image_raw' in available_topics:
        if self.det_img_sub is None:
            # 创建订阅器
            self.det_img_sub = rospy.Subscriber(...)
```

### 3. 数据回调流程

#### 图像数据流：
```
main.py 发布 /det_img/image_raw
    ↓
ROSInterfaceThread.det_image_callback()
    ↓
计算 FPS
    ↓
发送信号: image_received.emit(cv_image)
    ↓
ImageDisplayWidget.update_image()
    ↓
转换为 QPixmap 并显示
```

#### 状态数据流：
```
main.py 发布 /system_status (JSON)
    ↓
ROSInterfaceThread.status_callback()
    ↓
解析 JSON 数据
    ↓
更新 self.stats 字典
    ↓
发送信号: statistics_updated.emit(stats)
    ↓
StatisticsWidget.update_statistics()
    ↓
更新 UI 显示
```

## 四、信号和槽连接

### 1. ROS 线程 → UI 组件

```python
# 图像信号
self.ros_thread.image_received.connect(
    self.image_display.update_image
)

# 统计信息信号
self.ros_thread.statistics_updated.connect(
    self.statistics_widget.update_statistics
)

# 状态消息信号
self.ros_thread.status_message.connect(
    self.on_status_message
)
```

### 2. UI 组件 → 主窗口

```python
# Launch 启动请求
self.launch_selector.start_requested.connect(
    self.on_launch_start
)

# Launch 停止请求
self.launch_selector.stop_requested.connect(
    self.on_launch_stop
)

# 振镜控制命令
self.galvo_control.galvo_command.connect(
    self.on_galvo_command
)

# 激光控制命令
self.galvo_control.laser_control.connect(
    self.on_laser_control
)
```

## 五、Launch 文件管理

### 1. 启动流程

```python
def on_launch_start(self, launch_name):
    # 1. 检查是否已有进程运行
    if self.launch_manager.is_running():
        return
    
    # 2. 启动 launch 文件
    success, message = self.launch_manager.start_launch(launch_name)
    # - 执行: roslaunch laser_weeding <launch_file>
    # - 创建子进程
    
    # 3. 更新 UI 状态
    if success:
        self.launch_selector.set_running(True)
        self.galvo_control.set_enabled(False)  # 禁用手动控制
```

### 2. 停止流程

```python
def on_launch_stop(self):
    # 1. 停止进程
    success, message = self.launch_manager.stop_launch()
    # - 发送 SIGTERM
    # - 等待进程结束
    # - 必要时发送 SIGKILL
    
    # 2. 更新 UI 状态
    if success:
        self.launch_selector.set_running(False)
        self.galvo_control.set_enabled(True)  # 启用手动控制
```

## 六、手动振镜控制

### 1. 控制流程

```python
def on_galvo_command(self, galvo_index, x, y):
    # 1. 检查 launch 是否在运行
    if self.launch_manager.is_running():
        return  # 如果 launch 在运行，不允许手动控制
    
    # 2. 发布振镜命令
    if self.ros_thread.ros_initialized:
        self.ros_thread.publish_galvo_command(x, y, galvo_index)
        # - 发布到 /galvo_xy 话题
```

### 2. 激光控制

```python
def on_laser_control(self, enable):
    # 类似流程
    if not self.launch_manager.is_running():
        self.ros_thread.publish_laser_control(enable)
        # - 发布到 /laser_control 话题
```

## 七、状态更新机制

### 1. 定时器更新

```python
# 每 100ms 执行一次
def update_status(self):
    # 检查 launch 状态
    is_running = self.launch_manager.is_running()
    
    # 更新 UI 状态
    if is_running != self.launch_selector.stop_button.isEnabled():
        self.launch_selector.set_running(is_running)
        self.galvo_control.set_enabled(not is_running)
```

### 2. 实时数据更新

- **图像**: 通过 ROS 回调实时更新（30 FPS）
- **统计信息**: 通过 ROS 回调实时更新
- **系统状态**: 通过 ROS 回调实时更新

## 八、线程安全

### 1. ROS 线程

- 所有 ROS 操作在独立线程中执行
- 使用 `QMutex` 保护共享数据
- 使用信号/槽进行线程间通信

### 2. UI 线程

- 所有 UI 操作在主线程中执行
- 通过信号/槽接收 ROS 线程的数据
- 避免在 UI 线程中执行耗时操作

## 九、数据流图

```
┌─────────────────┐
│   main.py       │
│  (ROS 节点)     │
└────────┬────────┘
         │
         │ 发布话题
         │
         ▼
┌─────────────────┐
│ ROSInterface    │
│ Thread          │
│ (QThread)       │
└────────┬────────┘
         │
         │ 信号/槽
         │
         ▼
┌─────────────────┐
│  MainWindow     │
│  (主窗口)       │
└────────┬────────┘
         │
         ├──► ImageDisplayWidget
         ├──► StatisticsWidget
         ├──► LaunchSelectorWidget
         └──► ManualGalvoControlWidget
```

## 十、关键设计点

### 1. 非阻塞 UI
- ROS 通信在独立线程
- 使用信号/槽异步通信
- UI 始终保持响应

### 2. 自动重连
- 定期检查话题是否存在
- 话题出现时自动连接
- 适应节点启动延迟

### 3. 状态同步
- Launch 状态影响手动控制可用性
- 定时器确保 UI 状态一致
- 实时更新统计数据

### 4. 错误处理
- 完善的异常捕获
- 友好的错误提示
- 详细的调试信息

## 十一、配置文件

### QSettings 持久化

```python
# 保存窗口状态
self.settings.setValue('geometry', self.saveGeometry())
self.settings.setValue('windowState', self.saveState())

# 恢复窗口状态
self.restoreGeometry(self.settings.value('geometry'))
```

## 十二、退出流程

```python
def closeEvent(self, event):
    # 1. 保存窗口状态
    self.save_window_state()
    
    # 2. 检查并停止 Launch 进程
    if self.launch_manager.is_running():
        # 询问用户是否停止
        reply = QMessageBox.question(...)
        if reply == QMessageBox.Yes:
            self.launch_manager.stop_launch()
    
    # 3. 停止 ROS 线程
    if self.ros_thread.isRunning():
        self.ros_thread.stop()
        self.ros_thread.wait(3000)
    
    # 4. 接受关闭事件
    event.accept()
```

## 十三、调试信息

所有关键操作都有调试输出：
- `DEBUG: ...` - 一般调试信息
- `LaunchManager: ...` - Launch 管理相关
- 异常堆栈跟踪

可以通过终端输出追踪程序执行流程。


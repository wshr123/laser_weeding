# ROS 话题检查清单

## GUI 订阅的话题

| 话题名称 | 消息类型 | 用途 | 发布者 |
|---------|---------|------|--------|
| `/det_img/image_raw` | `sensor_msgs/Image` | 检测结果图像 | `main.py` |
| `/system_status` | `std_msgs/String` | 系统状态信息（JSON格式） | `main.py` |
| `/current_target` | `std_msgs/String` | 当前目标信息（JSON格式） | `main.py` |

## GUI 发布的话题

| 话题名称 | 消息类型 | 用途 | 订阅者 |
|---------|---------|------|--------|
| `/galvo_xy` | `std_msgs/Int32MultiArray` | 振镜控制命令 | `main.py` 或其他节点 |
| `/laser_control` | `std_msgs/Bool` | 激光控制命令 | `main.py` 或其他节点 |

## 检查步骤

### 1. 检查话题是否存在

运行检查脚本：
```bash
cd ~/my_workspace/src/laser_weeding/gui
python3 check_topics.py
```

或者使用 ROS 命令：
```bash
rostopic list | grep -E "(det_img|system_status|current_target|galvo_xy|laser_control)"
```

### 2. 检查话题数据流

```bash
# 检查图像话题
rostopic hz /det_img/image_raw

# 检查状态话题
rostopic echo /system_status

# 检查目标话题
rostopic echo /current_target
```

### 3. 常见问题

#### 问题 1: 没有图像显示
- **原因**: `main.py` 节点未运行或没有图像输入
- **解决**: 
  1. 确保已启动 `main.launch` 或 `main.py` 节点
  2. 检查 `/camera/color/image_raw` 话题是否有数据
  3. 查看 `main.py` 的日志，确认是否在发布 `/det_img/image_raw`

#### 问题 2: 统计信息不更新
- **原因**: `/system_status` 话题没有数据或格式错误
- **解决**:
  1. 检查 `main.py` 是否在发布状态信息
  2. 查看状态信息的 JSON 格式是否正确
  3. 检查 GUI 终端的调试输出

#### 问题 3: 按钮无响应
- **原因**: Launch 文件未找到或启动失败
- **解决**:
  1. 检查 launch 目录路径是否正确
  2. 查看终端调试输出
  3. 检查 ROS 环境是否正确配置

## 调试技巧

1. **查看 GUI 终端输出**: 所有调试信息都会输出到终端
2. **使用 `rostopic echo`**: 直接查看话题数据
3. **使用 `rqt_graph`**: 可视化话题连接关系
4. **检查节点状态**: `rosnode list` 和 `rosnode info <node_name>`


# 跟踪效果评估指南

## 概述

本系统提供了完整的跟踪效果评估工具，可以实时收集跟踪数据并生成详细的评估报告。

## 功能特性

- **实时数据收集**：自动收集每帧的跟踪结果（track_id, bbox, confidence等）
- **多维度指标**：计算轨迹长度、持续时间、稳定性等指标
- **自动报告生成**：生成文本和JSON格式的评估报告
- **可视化支持**：可选保存可视化图像

## 使用方法

### 方法1：使用专用launch文件（推荐）

```bash
roslaunch laser_weeding tracking_evaluation.launch
```

这会同时启动主系统和评估器。

### 方法2：手动启动

1. **启动主系统（启用跟踪评估）**：
```bash
roslaunch laser_weeding main.launch enable_tracking_evaluation:=true
```

2. **启动评估器**：
```bash
rosrun laser_weeding tracking_evaluator.py
```

### 方法3：在现有launch文件中添加参数

在 `main.launch` 或 `main_offline.launch` 中添加：

```xml
<param name="enable_tracking_evaluation" value="true" />
```

然后单独启动评估器：

```bash
rosrun laser_weeding tracking_evaluator.py _output_dir:=/path/to/output _save_images:=true
```

## 评估参数

### 评估器参数

- `output_dir` (string, 可选): 输出目录路径
  - 默认：`~/tracking_evaluation/YYYYMMDD_HHMMSS/`
  
- `save_images` (bool, 默认: false): 是否保存可视化图像
  - `true`: 保存每帧的可视化图像（占用空间较大）
  - `false`: 仅保存数据（推荐）

- `save_interval` (float, 默认: 10.0): 自动保存间隔（秒）
  - 定期保存中间结果，防止数据丢失

### 主系统参数

- `enable_tracking_evaluation` (bool, 默认: false): 是否启用跟踪评估
  - `true`: 发布跟踪结果到 `/tracking_results` 话题
  - `false`: 不发布（节省资源）

## 输出文件

评估器会在输出目录中生成以下文件：

### 1. `tracking_data.json`
每帧的跟踪结果原始数据，格式：
```json
[
  {
    "frame_id": 0,
    "timestamp": 1234567890.123,
    "num_detections": 3,
    "detections": [
      {
        "track_id": 1,
        "bbox": [100, 200, 50, 50],
        "confidence": 0.85,
        "target_point": [125, 225]
      }
    ]
  }
]
```

### 2. `track_statistics.json`
每个轨迹的统计信息：
```json
{
  "1": {
    "first_seen": 1234567890.123,
    "last_seen": 1234567890.456,
    "total_frames": 10,
    "avg_confidence": 0.82,
    "id_switches": 0
  }
}
```

### 3. `tracking_metrics.json`
整体评估指标：
```json
{
  "total_frames": 1000,
  "total_tracks": 50,
  "avg_track_length": 20.5,
  "stable_track_ratio": 0.75,
  "avg_confidence": 0.80
}
```

### 4. `evaluation_report.txt`
人类可读的文本报告，包含：
- 基础统计（总帧数、轨迹数等）
- 轨迹长度统计
- 轨迹持续时间统计
- 轨迹质量指标

## 评估指标说明

### 基础指标

- **总帧数** (`total_frames`): 处理的视频帧数
- **总轨迹数** (`total_tracks`): 检测到的唯一轨迹数量
- **平均每帧检测数** (`avg_detections_per_frame`): 平均每帧检测到的目标数

### 轨迹长度指标

- **平均轨迹长度** (`avg_track_length`): 轨迹平均持续帧数
- **最大/最小轨迹长度**: 最长和最短的轨迹
- **中位数轨迹长度**: 轨迹长度的中位数

### 轨迹持续时间指标

- **平均持续时间** (`avg_track_duration`): 轨迹平均存活时间（秒）
- **最大/最小持续时间**: 最长和最短的轨迹存活时间

### 轨迹质量指标

- **平均置信度** (`avg_confidence`): 所有检测的平均置信度
- **稳定轨迹比例** (`stable_track_ratio`): 长度≥5帧的轨迹比例
- **短轨迹比例** (`short_track_ratio`): 长度<5帧的轨迹比例

## 分析建议

### 1. 轨迹稳定性分析

查看 `stable_track_ratio`：
- **> 0.7**: 跟踪效果良好
- **0.5-0.7**: 跟踪效果一般，可能需要调整参数
- **< 0.5**: 跟踪效果较差，建议检查：
  - 检测置信度阈值是否合适
  - 跟踪器参数（max_distance, max_frames_to_skip等）
  - 图像质量

### 2. 轨迹长度分析

查看 `avg_track_length` 和 `max_track_length`：
- 如果平均长度很短（< 5帧），可能是：
  - 目标移动太快
  - 跟踪器参数设置不当
  - 检测不稳定

### 3. 置信度分析

查看 `avg_confidence`：
- **> 0.7**: 检测质量良好
- **0.5-0.7**: 检测质量一般
- **< 0.5**: 检测质量较差，建议：
  - 调整检测置信度阈值
  - 检查模型质量
  - 改善图像质量

## 高级用法

### 自定义输出目录

```bash
rosrun laser_weeding tracking_evaluator.py _output_dir:=/path/to/my/evaluation
```

### 保存可视化图像

```bash
rosrun laser_weeding tracking_evaluator.py _save_images:=true
```

注意：这会显著增加存储空间占用。

### 调整保存间隔

```bash
rosrun laser_weeding tracking_evaluator.py _save_interval:=5.0
```

## 与Ground Truth对比（未来扩展）

如果需要与标注数据进行对比计算MOTA/MOTP等指标，可以：

1. 准备标注数据（MOT格式）
2. 修改 `tracking_evaluator.py` 添加对比功能
3. 计算标准MOT指标

## 故障排除

### 问题：没有生成输出文件

**解决方案**：
- 检查是否启用了 `enable_tracking_evaluation`
- 检查 `/tracking_results` 话题是否有数据：`rostopic echo /tracking_results`
- 检查评估器节点是否正常运行：`rosnode list | grep tracking_evaluator`

### 问题：输出目录权限错误

**解决方案**：
- 确保输出目录有写权限
- 使用绝对路径指定输出目录

### 问题：数据文件过大

**解决方案**：
- 设置 `save_images:=false`（如果不需要可视化）
- 减少评估时间
- 定期清理旧数据

## 示例工作流

1. **启动评估**：
```bash
roslaunch laser_weeding tracking_evaluation.launch
```

2. **运行测试场景**（让系统运行一段时间）

3. **停止系统**（Ctrl+C），评估器会自动生成最终报告

4. **查看结果**：
```bash
cat ~/tracking_evaluation/YYYYMMDD_HHMMSS/evaluation_report.txt
```

5. **分析数据**：
```bash
# 查看JSON格式的指标
cat ~/tracking_evaluation/YYYYMMDD_HHMMSS/tracking_metrics.json | python -m json.tool
```

## 注意事项

1. **性能影响**：启用评估会增加少量CPU和内存开销，但通常可以忽略
2. **存储空间**：长时间运行会产生大量数据，建议定期清理
3. **实时性**：评估是异步进行的，不会影响主系统的实时性能


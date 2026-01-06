# 双振镜独立控制改造说明

## 问题描述

原有系统虽然支持双振镜硬件，但控制逻辑是串行的：同一时间只能有一个振镜工作，另一个振镜必须等待当前振镜完成任务后才能开始。这限制了系统的效率，因为两个振镜本应可以同时独立工作。

## 解决方案

将控制逻辑从**全局单目标串行状态机**改造为**每个振镜独立并行状态机**，使两个振镜可以同时跟踪和照射不同区域的目标。

## 核心修改

### 1. 控制循环重构 (`control_loop`)

**改造前：**
- 使用全局 `self.system_state` 和 `self.current_target`
- 同一时间只处理一个目标
- IDLE → TRACKING → FIRING → IDLE 串行循环

**改造后：**
- 每个振镜独立运行状态机：`_control_single_galvo(galvo_idx, current_time)`
- 每个振镜有自己的状态：`self.galvo_states[idx]`
- 每个振镜有自己的当前目标：`self.galvo_current_targets[idx]`
- 两个振镜可以同时处于不同状态（例如：G0 在 FIRING，G1 在 TRACKING）

### 2. 目标分配策略 (`update_targets` 和 `_update_galvo_queues`)

**改造前：**
- 所有目标进入单一全局队列 `self.target_queue`
- 按距离图像中心排序

**改造后：**
- 每个振镜有独立队列：`self.galvo_target_queues[idx]`
- 根据目标位置分配到对应振镜的区域
- 每个振镜内部按距离区域中心排序
- 避免重复处理：正在被处理的目标不会进入其他振镜队列

### 3. 激光控制独立化 (`set_galvo_laser`)

**改造前：**
- 全局激光状态 `self.laser_on`
- 切换振镜时会影响激光控制

**改造后：**
- 每个振镜独立的激光状态：`self.galvo_laser_on[idx]`
- 通过 `select_galvo(idx)` 切换后独立控制
- 互不干扰：G0 打开激光不影响 G1 的激光状态

### 4. 振镜控制循环 (`galvo_control_loop`)

**改造前：**
- 只根据全局 `self.laser_on` 发布激光状态

**改造后：**
- 每个振镜发布独立的位置和激光状态
- 消息中包含振镜索引 `idx`

### 5. 可视化更新 (`draw_info`)

**改造前：**
- 只显示当前活动振镜的状态
- 目标显示不区分由哪个振镜处理

**改造后：**
- 显示每个振镜的状态（IDLE/TRACKING/FIRING）
- 目标框标注由哪个振镜处理（例如：`[G0 laser]`）
- 显示各振镜的队列长度
- 十字准星根据振镜状态着色

### 6. 卡尔曼滤波器独立化 (`init_kalman_filters` 和 `predict_position`)

**改造前：**
- 全局单一卡尔曼滤波器 `self.kalman_filter`
- 两个振镜共享同一个滤波器状态
- 导致预测混乱：G0 跟踪目标A时，G1 跟踪目标B会污染滤波器状态

**改造后：**
- 每个振镜独立的卡尔曼滤波器：`self.kalman_filters[idx]`
- 每个振镜有独立的位置历史：`self.galvo_position_histories[idx]`
- 预测时使用对应振镜的滤波器，互不干扰
- 开始跟踪新目标时，重置对应振镜的滤波器状态

**关键方法：**
- `init_kalman_filters()`：为每个振镜创建独立的滤波器
- `reset_kalman_filter(galvo_idx, initial_position)`：重置指定振镜的滤波器
- `predict_position(dt, galvo_index)`：使用对应振镜的滤波器进行预测

### 7. 状态发布 (`publish_status`)

**改造后新增：**
- `galvo_states_info`：包含每个振镜的详细状态
  - 当前状态（IDLE/TRACKING/FIRING）
  - 激光开关状态
  - 当前处理的目标ID
  - 队列长度
  - 位置信息

## 数据结构对比

### 改造前（串行控制）
```python
self.system_state          # 全局状态
self.current_target        # 全局当前目标
self.target_queue          # 全局队列
self.laser_on              # 全局激光状态
```

### 改造后（并行控制）
```python
self.galvo_states[idx]              # 每个振镜的状态
self.galvo_current_targets[idx]     # 每个振镜的当前目标
self.galvo_target_queues[idx]       # 每个振镜的队列
self.galvo_laser_on[idx]            # 每个振镜的激光状态
self.galvo_state_start_times[idx]   # 每个振镜的状态开始时间
self.galvo_position_histories[idx]  # 每个振镜的位置历史
self.kalman_filters[idx]            # 每个振镜的卡尔曼滤波器（新增）
```

## 工作流程示例

假设有两个目标：
- **目标A** 在左侧（振镜G0的区域）
- **目标B** 在右侧（振镜G1的区域）

**改造前（串行）：**
1. T=0s: G0 开始跟踪目标A，G1 空闲等待
2. T=0.1s: G0 瞄准完成，开始照射目标A，G1 仍在等待
3. T=0.3s: G0 照射完成，释放控制权
4. T=0.3s: G1 开始跟踪目标B
5. T=0.4s: G1 瞄准完成，开始照射目标B
6. T=0.6s: G1 照射完成
**总耗时：0.6秒**

**改造后（并行）：**
1. T=0s: G0 开始跟踪目标A，**同时** G1 开始跟踪目标B
2. T=0.1s: G0 和 G1 **同时** 瞄准完成，**同时** 开始照射
3. T=0.3s: G0 和 G1 **同时** 照射完成
**总耗时：0.3秒（效率提升一倍）**

## 兼容性说明

为保持向后兼容，保留了以下全局变量：
- `self.system_state`：取第一个非空闲振镜的状态
- `self.current_target`：取第一个活跃振镜的目标
- `self.target_queue`：保留但不再主动使用
- `self.laser_on`：取所有振镜激光状态的OR结果

这些变量主要用于：
- 旧的日志输出
- 外部监控脚本
- 可视化的兼容性

## 测试建议

1. **单振镜测试**：设置 `galvo_count=1`，验证单振镜模式下功能正常
2. **双振镜顺序测试**：两个目标依次出现在不同区域，验证正确分配
3. **双振镜并行测试**：两个目标同时出现在不同区域，验证同时处理
4. **边界情况测试**：目标在重叠区域，验证振镜选择逻辑
5. **目标丢失测试**：目标突然消失，验证各振镜能正确恢复到IDLE

## 配置参数

在 `launch/main_offline.launch` 中的相关参数：
```xml
<param name="galvo_count" value="2" />
<param name="galvo_split_axis" value="vertical" />     <!-- 垂直分割 -->
<param name="galvo_split_ratio" value="0.5" />          <!-- 左右各50% -->
<param name="galvo_overlap_px" value="120" />           <!-- 重叠120像素 -->
```

## 修改文件清单

1. **scripts/main.py**：主控制逻辑
   - `control_loop()` - 重构为调度器
   - `_control_single_galvo()` - 新增单振镜状态机
   - `update_targets()` - 添加队列分配调用
   - `_update_galvo_queues()` - 新增队列构建方法
   - `set_galvo_laser()` - 增强独立控制逻辑
   - `init_kalman_filters()` - 为每个振镜创建独立滤波器（原 `init_kalman_filter`）
   - `reset_kalman_filter()` - 新增：重置指定振镜的滤波器状态
   - `predict_position()` - 使用对应振镜的滤波器进行预测
   - `update_position_history()` - 使用对应振镜的位置历史
   - `draw_info()` - 更新可视化
   - `publish_status()` - 新增振镜状态发布

2. **scripts/send_to_teensy.py**：无需修改（已支持多振镜）

3. **launch/main_offline.launch**：无需修改（参数已存在）

## 预期效果

✅ 两个振镜可以同时工作，互不干扰  
✅ 激光可以同时在两个位置照射  
✅ 处理效率理论上提升约一倍  
✅ 目标分配更加合理（按区域就近处理）  
✅ 可视化清晰显示各振镜状态  
✅ **每个振镜独立的卡尔曼滤波器，预测更准确，避免状态混乱**  

## 卡尔曼滤波器独立化详细说明

### 问题背景

在双振镜系统中，如果两个振镜共享同一个卡尔曼滤波器，会导致以下问题：

1. **状态污染**：G0 跟踪目标A时，滤波器学习目标A的运动模式；当G1开始跟踪目标B时，会使用已经被目标A"污染"的滤波器状态，导致预测不准确。

2. **预测混乱**：两个目标可能运动方向、速度完全不同，共享滤波器会导致预测偏差，影响激光瞄准精度。

3. **初始化冲突**：每次切换目标时重置滤波器，会影响另一个正在工作的振镜。

### 解决方案

为每个振镜创建独立的卡尔曼滤波器：

```python
# 改造前
self.kalman_filter = cv2.KalmanFilter(4, 2)  # 全局单一滤波器

# 改造后
self.kalman_filters = [None] * self.galvo_count  # 每个振镜一个
for idx in range(self.galvo_count):
    self.kalman_filters[idx] = cv2.KalmanFilter(4, 2)
```

### 关键实现细节

1. **独立的位置历史**
   - 每个振镜维护自己的位置历史：`self.galvo_position_histories[idx]`
   - 避免不同振镜的目标位置数据混合

2. **独立的滤波器状态**
   - 每个振镜的滤波器独立更新状态（位置、速度、协方差）
   - G0 的滤波器状态不会影响 G1 的预测

3. **目标切换时重置**
   - 当振镜开始跟踪新目标时，调用 `reset_kalman_filter(galvo_idx, initial_position)`
   - 将滤波器状态重置到新目标的初始位置，速度设为0
   - 确保新目标的预测从干净的状态开始

4. **预测时使用对应滤波器**
   - `predict_position(dt, galvo_index)` 根据 `galvo_index` 选择对应的滤波器
   - 确保每个振镜使用自己的滤波器进行预测

### 效果验证

改造后，两个振镜可以：
- ✅ 同时跟踪不同运动模式的目标（例如：G0跟踪向左移动的目标，G1跟踪向右移动的目标）
- ✅ 各自独立预测，互不干扰
- ✅ 激光瞄准精度提升，因为每个振镜的预测基于自己目标的真实运动模式

---
**修改日期**：2026-01-04  
**修改人**：AI Assistant  
**测试状态**：待验证


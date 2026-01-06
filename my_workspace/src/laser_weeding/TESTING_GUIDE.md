# Teensy 振镜控制器测试指南

本文档提供完整的测试流程，用于验证 `control_tennsy.ino` 代码是否正常工作。

## 前置准备

### 1. 硬件连接
- ✅ Teensy 3.2 已正确连接到电脑（USB）
- ✅ 振镜驱动板已连接到 Teensy（XY2-100 协议）
- ✅ 激光控制引脚（PIN 9）已连接（可选，用于激光测试）
- ✅ 确认串口设备路径（Linux: `/dev/ttyACM0` 或 `/dev/ttyUSB0`，Windows: `COM3` 等）

### 2. 软件环境
```bash
# 安装 Python 依赖
pip3 install pyserial

# 确认串口权限（Linux）
sudo chmod 666 /dev/ttyACM0  # 或使用你的设备路径
```

### 3. 上传代码
- 使用 Arduino IDE 或 PlatformIO 将 `control_tennsy.ino` 上传到 Teensy 3.2
- 确认波特率设置为 115200

---

## 测试步骤

### 阶段 1: 基础连接测试

#### 1.1 串口连接测试
```bash
# 使用 test_teensy.py 进行基础连接测试
python3 scripts/test_teensy.py --port /dev/ttyACM0
```

**预期结果：**
- 看到 "READY:DUAL_GALVO" 消息
- PING/PONG 测试成功，RTT < 10ms
- STATUS 命令返回系统状态

**如果失败：**
- 检查串口路径是否正确
- 确认 Teensy 已正确连接
- 检查波特率是否为 115200

#### 1.2 手动串口测试（可选）
```bash
# 使用 minicom 或 screen 连接串口
minicom -D /dev/ttyACM0 -b 115200
# 或
screen /dev/ttyACM0 115200
```

**测试命令：**
```
PING          # 应该返回 PONG
STATUS        # 应该返回系统状态
VERSION       # 应该返回版本信息
```

---

### 阶段 2: 振镜控制测试

#### 2.1 单振镜移动测试
```bash
# 测试振镜 0（默认）
python3 scripts/test_dual_galvo.py --galvo 0 --x 0 --y 0        # 回到中心
python3 scripts/test_dual_galvo.py --galvo 0 --x 10000 --y 0     # 向右移动
python3 scripts/test_dual_galvo.py --galvo 0 --x -10000 --y 0   # 向左移动
python3 scripts/test_dual_galvo.py --galvo 0 --x 0 --y 10000     # 向上移动
python3 scripts/test_dual_galvo.py --galvo 0 --x 0 --y -10000    # 向下移动
```

**观察要点：**
- 振镜是否移动到指定位置
- 移动是否平滑（无抖动）
- 响应时间是否合理

#### 2.2 双振镜独立控制测试
```bash
# 测试振镜 1
python3 scripts/test_dual_galvo.py --galvo 1 --x 5000 --y 5000
python3 scripts/test_dual_galvo.py --galvo 1 --x -5000 --y -5000

# 交替测试两个振镜
python3 scripts/test_dual_galvo.py --galvo 0 --x 10000 --y 0
python3 scripts/test_dual_galvo.py --galvo 1 --x -10000 --y 0
```

**观察要点：**
- 两个振镜是否能独立控制
- 切换振镜时是否影响另一个

#### 2.3 使用有符号坐标测试（XYS 命令）
```bash
# 通过串口直接测试
minicom -D /dev/ttyACM0 -b 115200

# 在串口终端中输入：
XYS:10000,0      # 振镜0向右移动
XYS1:5000,5000   # 振镜1移动到(5000,5000)
XYS2:-5000,-5000 # 振镜2移动到(-5000,-5000)（如果支持）
```

---

### 阶段 3: 高级功能测试

#### 3.1 振镜选择测试
```bash
# 通过串口测试
GALVO:1         # 选择振镜1
XY:10000,0      # 移动当前选中的振镜
GALVO:2         # 选择振镜2（如果支持）
XY:-10000,0     # 移动振镜2
```

#### 3.2 限制范围测试
```bash
# 设置振镜0的移动范围
LIMITS:1:-10000,10000,-10000,10000

# 尝试移动到范围外
XY:20000,0      # 应该被限制到 10000
XY:-20000,0     # 应该被限制到 -10000
```

#### 3.3 中心保持测试
```bash
CENTER          # 移动到中心并保持
# 等待几秒，观察振镜是否保持在中心位置
XY:10000,0      # 移动后应该解除保持模式
CENTER:ALL      # 所有振镜回到中心
```

#### 3.4 平滑移动测试
```bash
SMOOTH:ON       # 启用平滑移动
XY:10000,10000  # 观察是否平滑移动
SMOOTH:OFF      # 关闭平滑移动
XY:0,0          # 观察是否直接跳转
```

---

### 阶段 4: 激光控制测试（需要硬件支持）

⚠️ **警告：激光测试需要安全防护措施！**

#### 4.1 激光开关测试
```bash
# 通过串口测试
LASER:ON        # 打开激光（点模式）
# 观察激光是否开启
LASER:OFF       # 关闭激光
```

#### 4.2 激光模式测试
```bash
MODE:POINT      # 设置为点模式
LASER:ON        # 单点灼烧

MODE:SPIRAL     # 设置为螺旋模式
SPIRAL:CONFIG:5000,1000,1500,0.25  # 配置螺旋参数
LASER:ON        # 执行螺旋灼烧
```

---

### 阶段 5: 测试图案功能

#### 5.1 内置测试图案
```bash
# 通过串口执行
TEST            # 运行测试图案（中心点、四角、十字线、圆）
```

**观察要点：**
- 图案是否按预期执行
- 激光是否在正确位置开启/关闭

#### 5.2 画圆测试
```bash
CIRCLE:3000     # 画半径为3000的圆
```

#### 5.3 画矩形测试
```bash
RECT:5000,3000  # 画宽5000、高3000的矩形
```

---

### 阶段 6: 校准功能测试

#### 6.1 校准参数设置
```bash
# 设置振镜0的校准参数
GALVO:1
CAL:X:SCALE:1.05    # X轴缩放1.05
CAL:Y:SCALE:0.98    # Y轴缩放0.98
CAL:X:OFFSET:100    # X轴偏移+100
CAL:Y:OFFSET:-50    # Y轴偏移-50

# 测试校准效果
XY:10000,0      # 应该应用校准参数
```

---

## 完整测试脚本

创建一个自动化测试脚本：

```bash
#!/bin/bash
# complete_test.sh - 完整测试脚本

PORT="/dev/ttyACM0"  # 修改为你的串口

echo "=== 阶段1: 连接测试 ==="
python3 scripts/test_teensy.py --port $PORT

echo -e "\n=== 阶段2: 振镜移动测试 ==="
python3 scripts/test_dual_galvo.py --port $PORT --galvo 0 --x 0 --y 0
sleep 1
python3 scripts/test_dual_galvo.py --port $PORT --galvo 0 --x 10000 --y 0
sleep 1
python3 scripts/test_dual_galvo.py --port $PORT --galvo 0 --x 0 --y 0
sleep 1
python3 scripts/test_dual_galvo.py --port $PORT --galvo 1 --x 5000 --y 5000
sleep 1
python3 scripts/test_dual_galvo.py --port $PORT --galvo 1 --x 0 --y 0

echo -e "\n=== 测试完成 ==="
```

---

## 常见问题排查

### 问题 1: 无法连接串口
**症状：** `test_teensy.py` 找不到设备

**解决方案：**
```bash
# 列出所有串口设备
python3 -m serial.tools.list_ports

# 检查设备权限
ls -l /dev/ttyACM*

# 添加用户到 dialout 组（Linux）
sudo usermod -a -G dialout $USER
# 然后重新登录
```

### 问题 2: 振镜不移动
**症状：** 命令返回 OK，但振镜不动

**检查清单：**
- ✅ 振镜驱动板电源是否正常
- ✅ XY2-100 协议线是否连接正确
- ✅ 振镜驱动板是否支持双振镜模式
- ✅ 检查 `STATUS` 命令，确认目标位置是否正确

### 问题 3: 响应超时
**症状：** 命令发送后无响应

**解决方案：**
- 检查波特率是否匹配（115200）
- 尝试重置 Teensy（按复位按钮）
- 检查串口是否被其他程序占用

### 问题 4: 坐标轴反向
**症状：** X/Y 轴移动方向相反

**解决方案：**
- 检查代码中的坐标映射（注意代码中 XY 轴可能对调）
- 使用 `XYS` 命令直接指定有符号坐标
- 调整校准参数进行补偿

---

## 测试检查清单

- [ ] 串口连接正常，能收到 READY 消息
- [ ] PING/PONG 测试成功
- [ ] 振镜0能正常移动
- [ ] 振镜1能正常移动
- [ ] 两个振镜能独立控制
- [ ] 中心保持功能正常
- [ ] 限制范围功能正常
- [ ] 平滑移动功能正常（如果使用）
- [ ] 激光控制功能正常（如果使用）
- [ ] 测试图案功能正常
- [ ] 校准参数能正确应用
- [ ] 紧急停止功能正常

---

## 性能指标参考

- **PING 延迟：** < 10ms
- **命令响应时间：** < 50ms
- **振镜移动响应：** < 100ms（取决于硬件）
- **保活周期：** 1ms（1000微秒）

---

## 下一步

测试通过后，可以：
1. 集成到 ROS 系统中（使用 `send_to_teensy.py`）
2. 进行 3D 校准（使用 `galvo_calibrator_depth.py`）
3. 运行完整系统测试（使用 `main.launch`）


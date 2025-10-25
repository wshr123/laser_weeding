// teensy_xy2_100_simple.ino
// 简化版 Teensy 3.2 振镜控制器 - 使用XY2-100库
// 用于激光除草系统

#include <XY2_100.h>

// ===== 创建XY2-100对象 =====
XY2_100 galvo;

// ===== 激光控制引脚 =====
const int LASER_PIN = 9;        // 激光TTL控制 (开/关)

// ===== 系统状态 =====
// 位置控制
int16_t current_x = 0;          // 当前X位置 (-32768 to 32767)
int16_t current_y = 0;          // 当前Y位置 (-32768 to 32767)
int16_t target_x = 0;           // 目标X位置
int16_t target_y = 0;           // 目标Y位置

// 激光状态
bool laser_enabled = false;      // 激光开关状态

// 运动控制
bool is_moving = false;          // 是否正在移动
bool smooth_move = false;        // 平滑移动开关 (默认关闭，直接移动)

// 插值控制（用于平滑移动）
float x_position = 0.0;          // 浮点X位置
float y_position = 0.0;          // 浮点Y位置
const float MOVE_STEP = 500.0;   // 每次移动步长

// 校准参数
float x_scale = 1.0;             // X轴缩放系数
float y_scale = 1.0;             // Y轴缩放系数
int16_t x_offset = 0;            // X轴偏移
int16_t y_offset = 0;            // Y轴偏移

// 扫描范围限制
const int16_t X_MIN = -32767;    // X最小值
const int16_t X_MAX = 32767;     // X最大值
const int16_t Y_MIN = -32767;    // Y最小值
const int16_t Y_MAX = 32767;     // Y最大值

// 通信缓冲区
char serial_buffer[256];
size_t buffer_index = 0;

// 时间控制
unsigned long last_update_time = 0;
unsigned long last_status_time = 0;
unsigned long laser_start_time = 0;

// 保持中心与保活
bool hold_center = true;                 // 上电后保持在(0,0)直到收到移动指令
unsigned long last_keepalive_time = 0;   // 上次保活时间戳（micros）

const unsigned long UPDATE_INTERVAL = 10;      // 10us? 注意：原注释写10ms，但使用micros()，这里是10微秒
const unsigned long STATUS_INTERVAL = 1000;    // 1秒 状态报告间隔（原注释不一致，实际数值为1s）
const unsigned long MAX_LASER_ON_TIME = 500;   // 最大激光持续时间 0.5秒（原注释写5秒，实际为500ms）
const unsigned long KEEPALIVE_INTERVAL_US = 1000; // 保活周期：1ms（1000微秒），可按需调整为2000~5000微秒

// ===== 函数声明 =====
void parseCommand(String cmd);
void moveToPosition(int16_t x, int16_t y, bool immediate = false);
void updatePosition();
void setLaser(bool on);
void emergencyStop();
void sendStatus();
void sendResponse(String response);
void testPattern();
void drawCircle(int16_t cx, int16_t cy, int16_t radius, int points = 100);
void drawRectangle(int16_t x1, int16_t y1, int16_t x2, int16_t y2);
int16_t applyCalibration(int16_t value, float scale, int16_t offset);

// ===== 初始化 =====
void setup() {
  // 初始化串口
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    // 等待串口连接，最多等3秒
  }
  
  Serial.println("=====================================");
  Serial.println("  Simple XY2-100 Galvo Controller");
  Serial.println("     Laser Weeding System v1.0");
  Serial.println("=====================================");
  
  // 配置激光引脚
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);
  
  // 初始化XY2-100
  Serial.println("Initializing XY2-100 protocol...");
  galvo.begin();
  delay(100);
  
  // 移动到中心位置并设置保持
  galvo.setSignedXY(0, 0);
  current_x = 0;
  current_y = 0;
  x_position = 0.0;
  y_position = 0.0;
  hold_center = true;  // 上电即保持中心
  last_keepalive_time = micros();
  
  Serial.println("XY2-100 initialized");
  Serial.println("=====================================");
  Serial.println("Commands:");
  Serial.println("  XY:x,y     - Move to position");
  Serial.println("  LASER:ON   - Turn laser on");
  Serial.println("  LASER:OFF  - Turn laser off");
  Serial.println("  CENTER     - Move to center (0,0) and hold");
  Serial.println("  STOP       - Emergency stop");
  Serial.println("  STATUS     - Get system status");
  Serial.println("  TEST       - Run test pattern");
  Serial.println("  SMOOTH:ON  - Enable smooth move");
  Serial.println("  SMOOTH:OFF - Disable smooth move");
  Serial.println("=====================================");
  Serial.println("Ready!");
}

// ===== 主循环 =====
void loop() {
  // 处理串口命令
  if (Serial.available()) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (buffer_index > 0) {
        serial_buffer[buffer_index] = '\0';
        String command = String(serial_buffer);
        command.trim();
        if (command.length() > 0) {
          parseCommand(command);
        }
        buffer_index = 0;
      }
    } else if (buffer_index < sizeof(serial_buffer) - 1) {
      serial_buffer[buffer_index++] = c;
    }
  }
  
  // 更新位置（如果启用平滑移动）
  if (smooth_move && is_moving) {
    unsigned long current_time = micros();
    if (current_time - last_update_time >= UPDATE_INTERVAL) {
      updatePosition();
      last_update_time = current_time;
    }
  }
  
  // 检查激光超时
  if (laser_enabled && (millis() - laser_start_time > MAX_LASER_ON_TIME)) {
    Serial.println("WARNING: Laser timeout - auto shutdown");
    setLaser(false);
  }
  
  // 定期状态报告（可选）
  if (millis() - last_status_time > STATUS_INTERVAL) {
    // sendStatus();  // 默认关闭自动状态报告
    last_status_time = millis();
  }

  // 保持中心与保活：在保持模式且不在移动时，周期性重复发送当前坐标（通常为0,0）
  unsigned long now_us = micros();
  if (hold_center && !is_moving) {
    if (now_us - last_keepalive_time >= KEEPALIVE_INTERVAL_US) {
      galvo.setSignedXY(current_x, current_y);
      last_keepalive_time = now_us;
    }
  }
}

// ===== 命令解析 =====
void parseCommand(String cmd) {
  cmd.toUpperCase();
  
  // 调试输出
  Serial.print("CMD: ");
  Serial.println(cmd);
  
  // 紧急停止
  if (cmd == "STOP" || cmd == "!") {
    emergencyStop();
    return;
  }
  
  // XY位置命令
  if (cmd.startsWith("XY:")) {
    int commaIndex = cmd.indexOf(',', 3);
    if (commaIndex > 0) {
      String xStr = cmd.substring(3, commaIndex);
      String yStr = cmd.substring(commaIndex + 1);
      
      // 支持两种输入格式：
      // 1. 有符号格式 (-32768 to 32767)
      // 2. 无符号格式 (0 to 65535, 32768为中心)
      //把xy对调了
      long yVal = xStr.toInt();
      long xVal = yStr.toInt();
      
      // 如果输入大于32767，认为是无符号格式，转换为有符号
      if (xVal > 32767) xVal -= 65536;
      if (yVal > 32767) yVal -= 65536;
      
      int16_t x = constrain(xVal, X_MIN, X_MAX);
      int16_t y = constrain(yVal, Y_MIN, Y_MAX);
      
      moveToPosition(x, y);
      sendResponse("OK:XY:" + String(x) + "," + String(y));
    } else {
      sendResponse("ERROR:Invalid XY format");
    }
  }
  // 激光控制
  else if (cmd == "LASER:ON") {
    setLaser(true);
    sendResponse("OK:LASER:ON");
  }
  else if (cmd == "LASER:OFF") {
    setLaser(false);
    sendResponse("OK:LASER:OFF");
  }
  // 移动到中心并保持
  else if (cmd == "CENTER" || cmd == "HOME") {
    moveToPosition(0, 0, true);
    hold_center = true;  // 进入保持中心模式
    sendResponse("OK:CENTER");
  }
  // 状态查询
  else if (cmd == "STATUS" || cmd == "?") {
    sendStatus();
  }
  // 测试模式
  else if (cmd == "TEST") {
    testPattern();
  }
  // 平滑移动控制
  else if (cmd == "SMOOTH:ON") {
    smooth_move = true;
    sendResponse("OK:SMOOTH:ON");
  }
  else if (cmd == "SMOOTH:OFF") {
    smooth_move = false;
    sendResponse("OK:SMOOTH:OFF");
  }
  // 设置校准参数
  else if (cmd.startsWith("CAL:X:SCALE:")) {
    x_scale = cmd.substring(12).toFloat();
    sendResponse("OK:CAL:X:SCALE:" + String(x_scale));
  }
  else if (cmd.startsWith("CAL:Y:SCALE:")) {
    y_scale = cmd.substring(12).toFloat();
    sendResponse("OK:CAL:Y:SCALE:" + String(y_scale));
  }
  else if (cmd.startsWith("CAL:X:OFFSET:")) {
    x_offset = cmd.substring(13).toInt();
    sendResponse("OK:CAL:X:OFFSET:" + String(x_offset));
  }
  else if (cmd.startsWith("CAL:Y:OFFSET:")) {
    y_offset = cmd.substring(13).toInt();
    sendResponse("OK:CAL:Y:OFFSET:" + String(y_offset));
  }
  // 获取版本信息
  else if (cmd == "VERSION" || cmd == "VER") {
    sendResponse("VERSION:1.0:XY2-100:TEENSY32");
  }
  // 画圆测试
  else if (cmd.startsWith("CIRCLE:")) {
    String rStr = cmd.substring(7);
    int16_t radius = constrain(rStr.toInt(), 100, 10000);
    drawCircle(0, 0, radius);
    sendResponse("OK:CIRCLE:" + String(radius));
  }
  // 画矩形测试
  else if (cmd.startsWith("RECT:")) {
    int commaIndex = cmd.indexOf(',', 5);
    if (commaIndex > 0) {
      String wStr = cmd.substring(5, commaIndex);
      String hStr = cmd.substring(commaIndex + 1);
      int16_t width = constrain(wStr.toInt(), 100, 20000);
      int16_t height = constrain(hStr.toInt(), 100, 20000);
      drawRectangle(-width/2, -height/2, width/2, height/2);
      sendResponse("OK:RECT:" + String(width) + "," + String(height));
    }
  }
  else {
    sendResponse("ERROR:Unknown command: " + cmd);
  }
}

// ===== 位置控制 =====
void moveToPosition(int16_t x, int16_t y, bool immediate) {
  // 一旦有移动请求，解除中心保持
  hold_center = false;

  // 应用校准
  x = applyCalibration(x, x_scale, x_offset);
  y = applyCalibration(y, y_scale, y_offset);
  
  // 限制范围
  x = constrain(x, X_MIN, X_MAX);
  y = constrain(y, Y_MIN, Y_MAX);
  
  target_x = x;
  target_y = y;
  
  if (!smooth_move || immediate) {
    // 直接移动
    galvo.setSignedXY(x, y);
    current_x = x;
    current_y = y;
    x_position = float(x);
    y_position = float(y);
    is_moving = false;
  } else {
    // 开始平滑移动
    is_moving = true;
  }
}

// ===== 平滑位置更新 =====
void updatePosition() {
  if (!is_moving) return;
  
  // 计算到目标的距离
  float dx = target_x - x_position;
  float dy = target_y - y_position;
  float distance = sqrt(dx * dx + dy * dy);
  
  if (distance < 1.0) {
    // 到达目标
    x_position = target_x;
    y_position = target_y;
    current_x = target_x;
    current_y = target_y;
    galvo.setSignedXY(current_x, current_y);
    is_moving = false;
  } else {
    // 计算移动步长
    float step = min(MOVE_STEP, distance);
    float ratio = step / distance;
    
    // 更新位置
    x_position += dx * ratio;
    y_position += dy * ratio;
    
    // 转换为整数并发送
    current_x = int16_t(x_position);
    current_y = int16_t(y_position);
    galvo.setSignedXY(current_x, current_y);
  }
}

// ===== 激光控制 =====
void setLaser(bool on) {
  laser_enabled = on;
  digitalWrite(LASER_PIN, on ? HIGH : LOW);
  
  if (on) {
    laser_start_time = millis();
  }
}

// ===== 紧急停止 =====
void emergencyStop() {
  // 立即关闭激光
  setLaser(false);
  
  // 停止所有运动
  is_moving = false;
  target_x = current_x;
  target_y = current_y;
  
  // 发送当前位置（确保振镜停止）
  galvo.setSignedXY(current_x, current_y);

  // 进入保持中心模式如果当前位置就是中心
  if (current_x == 0 && current_y == 0) {
    hold_center = true;
  }
  
  sendResponse("EMERGENCY_STOP");
  Serial.println("!!! EMERGENCY STOP !!!");
}

// ===== 状态报告 =====
void sendStatus() {
  String status = "STATUS:";
  status += "X=" + String(current_x);
  status += ",Y=" + String(current_y);
  status += ",TX=" + String(target_x);
  status += ",TY=" + String(target_y);
  status += ",LASER=" + String(laser_enabled ? "ON" : "OFF");
  status += ",MOVING=" + String(is_moving ? "YES" : "NO");
  status += ",HOLD_CENTER=" + String(hold_center ? "YES" : "NO");
  
  sendResponse(status);
}

// ===== 发送响应 =====
void sendResponse(String response) {
  Serial.println(response);
}

// ===== 测试图案 =====
void testPattern() {
  Serial.println("Starting test pattern...");
  
  // 保存当前状态
  int16_t saved_x = current_x;
  int16_t saved_y = current_y;
  bool saved_laser = laser_enabled;
  bool saved_hold = hold_center;

  // 测试前解除保持，避免保活干扰
  hold_center = false;
  
  // 测试1：中心点
  Serial.println("Test 1: Center point");
  moveToPosition(0, 0, true);
  delay(500);
  setLaser(true);
  delay(200);
  setLaser(false);
  delay(500);
  
  // 测试2：四个角
  Serial.println("Test 2: Four corners");
  int16_t corners[4][2] = {
    {-10000, -10000},  // 左下
    {10000, -10000},   // 右下
    {10000, 10000},    // 右上
    {-10000, 10000}    // 左上
  };
  
  for (int i = 0; i < 4; i++) {
    moveToPosition(corners[i][0], corners[i][1], true);
    delay(300);
    setLaser(true);
    delay(200);
    setLaser(false);
    delay(200);
  }
  
  // 测试3：十字线
  Serial.println("Test 3: Cross");
  moveToPosition(-5000, 0, true);
  delay(200);
  setLaser(true);
  moveToPosition(5000, 0, true);
  delay(500);
  setLaser(false);
  
  moveToPosition(0, -5000, true);
  delay(200);
  setLaser(true);
  moveToPosition(0, 5000, true);
  delay(500);
  setLaser(false);
  
  // 测试4：小圆
  Serial.println("Test 4: Small circle");
  drawCircle(0, 0, 3000, 30);
  
  // 恢复状态
  moveToPosition(saved_x, saved_y, true);
  setLaser(saved_laser);
  hold_center = saved_hold;
  
  Serial.println("Test pattern completed");
}

// ===== 画圆 =====
void drawCircle(int16_t cx, int16_t cy, int16_t radius, int points) {
  // 临时解除保持，避免保活干扰
  bool prev_hold = hold_center;
  hold_center = false;

  // 移动到起始点
  int16_t start_x = cx + radius;
  int16_t start_y = cy;
  moveToPosition(start_x, start_y, true);
  delay(100);
  
  // 打开激光
  setLaser(true);
  
  // 画圆
  for (int i = 0; i <= points; i++) {
    float angle = 2.0 * PI * i / points;
    int16_t x = cx + radius * cos(angle);
    int16_t y = cy + radius * sin(angle);
    moveToPosition(x, y, true);
    delay(20);
  }
  
  // 关闭激光
  setLaser(false);

  // 恢复保持状态
  hold_center = prev_hold;
}

// ===== 画矩形 =====
void drawRectangle(int16_t x1, int16_t y1, int16_t x2, int16_t y2) {
  // 临时解除保持，避免保活干扰
  bool prev_hold = hold_center;
  hold_center = false;

  // 移动到起始点
  moveToPosition(x1, y1, true);
  delay(100);
  
  // 打开激光
  setLaser(true);
  
  // 画四条边
  moveToPosition(x2, y1, true);
  delay(100);
  moveToPosition(x2, y2, true);
  delay(100);
  moveToPosition(x1, y2, true);
  delay(100);
  moveToPosition(x1, y1, true);
  delay(100);
  
  // 关闭激光
  setLaser(false);

  // 恢复保持状态
  hold_center = prev_hold;
}

// ===== 辅助函数 =====
int16_t applyCalibration(int16_t value, float scale, int16_t offset) {
  return int16_t(value * scale) + offset;
}
// teensy_xy2_100_simple.ino
// 简化版 Teensy 3.2 振镜控制器 - 使用XY2-100/XY2-100_multi库
// 用于激光除草系统

#if !defined(__has_include)
#define __has_include(x) 0
#endif

#if __has_include(<XY2_100_multi.h>)
#include <XY2_100_multi.h>
using XY2Driver = XY2_100_multi;
const char XY2_LIB_LABEL[] = "XY2-100_multi";
#elif __has_include(<XY2_100.h>)
#include <XY2_100.h>
using XY2Driver = XY2_100;
const char XY2_LIB_LABEL[] = "XY2-100";
#else
#error "No XY2-100 driver library found (expected XY2_100_multi or XY2_100)"
#endif
#include <math.h>

// ===== 常量与类型 =====
const uint8_t GALVO_COUNT = 2;           // 支持的振镜数量

enum LaserMode {
  LASER_MODE_POINT = 0,
  LASER_MODE_SPIRAL = 1
};

// ===== 创建XY2-100对象 =====
XY2Driver galvo_primary;
XY2Driver galvo_secondary;
XY2Driver* galvos[GALVO_COUNT] = {&galvo_primary, &galvo_secondary};

// ===== 激光控制引脚 =====
const int LASER_PIN = 9;        // 激光TTL控制 (开/关)

// ===== 系统状态 =====
// 位置控制
int16_t current_x[GALVO_COUNT] = {0};          // 当前X位置 (-32768 to 32767)
int16_t current_y[GALVO_COUNT] = {0};          // 当前Y位置 (-32768 to 32767)
int16_t target_x[GALVO_COUNT] = {0};           // 目标X位置（应用校准后）
int16_t target_y[GALVO_COUNT] = {0};           // 目标Y位置（应用校准后）
int16_t requested_target_x[GALVO_COUNT] = {0}; // 原始目标X位置（未校准）
int16_t requested_target_y[GALVO_COUNT] = {0}; // 原始目标Y位置（未校准）
uint8_t active_galvo = 0;                      // 当前选中的振镜

// 激光状态
bool laser_enabled = false;      // 激光开关状态

// 运动控制
bool is_moving[GALVO_COUNT] = {false};          // 是否正在移动
bool smooth_move = false;        // 平滑移动开关 (默认关闭，直接移动)

// 插值控制（用于平滑移动）
float x_position[GALVO_COUNT] = {0.0};          // 浮点X位置
float y_position[GALVO_COUNT] = {0.0};          // 浮点Y位置
const float MOVE_STEP = 500.0;   // 每次移动步长

// 校准参数
float x_scale[GALVO_COUNT] = {1.0, 1.0};             // X轴缩放系数
float y_scale[GALVO_COUNT] = {1.0, 1.0};             // Y轴缩放系数
int16_t x_offset[GALVO_COUNT] = {0, 0};              // X轴偏移
int16_t y_offset[GALVO_COUNT] = {0, 0};              // Y轴偏移

// 扫描范围限制（可由上位机调整）
const int16_t CODE_LIMIT_MIN = -32767;
const int16_t CODE_LIMIT_MAX = 32767;
int16_t x_min_limits[GALVO_COUNT] = {CODE_LIMIT_MIN, CODE_LIMIT_MIN};
int16_t x_max_limits[GALVO_COUNT] = {CODE_LIMIT_MAX, CODE_LIMIT_MAX};
int16_t y_min_limits[GALVO_COUNT] = {CODE_LIMIT_MIN, CODE_LIMIT_MIN};
int16_t y_max_limits[GALVO_COUNT] = {CODE_LIMIT_MAX, CODE_LIMIT_MAX};

// 通信缓冲区
char serial_buffer[256];
size_t buffer_index = 0;

// 时间控制
unsigned long last_update_time[GALVO_COUNT] = {0};
unsigned long last_status_time = 0;
unsigned long laser_start_time = 0;

// 保持中心与保活
bool hold_center[GALVO_COUNT] = {true, true};                 // 上电后保持在(0,0)直到收到移动指令
unsigned long last_keepalive_time[GALVO_COUNT] = {0, 0};      // 上次保活时间戳（micros）

const unsigned long UPDATE_INTERVAL = 10;      // 10us? 注意：原注释写10ms，但使用micros()，这里是10微秒
const unsigned long STATUS_INTERVAL = 1000;    // 1秒 状态报告间隔（原注释不一致，实际数值为1s）
const unsigned long MAX_LASER_ON_TIME = 500;   // 最大激光持续时间 0.5秒（原注释写5秒，实际为500ms）
const unsigned long KEEPALIVE_INTERVAL_US = 1000; // 保活周期：1ms（1000微秒），可按需调整为2000~5000微秒

// 螺旋灼烧参数
LaserMode laser_mode = LASER_MODE_POINT; // 当前激光模式
int16_t spiral_max_radius = 6000;        // 螺旋最大半径
float spiral_spacing = 1200.0;           // 相邻圈之间的间距
float spiral_angle_step = 0.25;          // 每次旋转步进弧度
unsigned int spiral_point_delay_us = 1500; // 两个点之间的停留时间（微秒）

// ===== 函数声明 =====
void parseCommand(String cmd);
void moveToPosition(uint8_t galvo_index, int16_t x, int16_t y, bool immediate = false);
void moveToPosition(int16_t x, int16_t y, bool immediate = false);
void updatePosition(uint8_t galvo_index);
void setLaser(bool on);
void emergencyStop();
void sendStatus();
void sendResponse(String response);
void testPattern();
void drawCircle(uint8_t galvo_index, int16_t cx, int16_t cy, int16_t radius, int points = 100);
void drawCircle(int16_t cx, int16_t cy, int16_t radius, int points = 100);
void drawRectangle(uint8_t galvo_index, int16_t x1, int16_t y1, int16_t x2, int16_t y2);
void drawRectangle(int16_t x1, int16_t y1, int16_t x2, int16_t y2);
void drawSpiral(uint8_t galvo_index, int16_t cx, int16_t cy, int16_t max_radius, float spacing, float angle_step);
int16_t applyCalibration(int16_t value, float scale, int16_t offset);
const char* laserModeName();
void setGalvoLimits(uint8_t galvo_index, int16_t xmin, int16_t xmax, int16_t ymin, int16_t ymax);
int16_t clampToGalvoRange(uint8_t galvo_index, int16_t value, bool isX);

// ===== 初始化 =====
void setup() {
  // 初始化串口
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    // 等待串口连接，最多等3秒
  }
  
  Serial.println("=====================================");
  Serial.print("  Simple ");
  Serial.print(XY2_LIB_LABEL);
  Serial.println(" Galvo Controller");
  Serial.println("     Laser Weeding System v1.0");
  Serial.println("=====================================");
  
  // 配置激光引脚
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);
  
  // 初始化XY2-100
  Serial.print("Initializing ");
  Serial.print(XY2_LIB_LABEL);
  Serial.println(" protocol (dual head)...");
  for (uint8_t i = 0; i < GALVO_COUNT; i++) {
    galvos[i]->begin();
    delay(100);

    // 移动到中心位置并设置保持
    galvos[i]->setSignedXY(0, 0);
    current_x[i] = 0;
    current_y[i] = 0;
    target_x[i] = 0;
    target_y[i] = 0;
    requested_target_x[i] = 0;
    requested_target_y[i] = 0;
    x_position[i] = 0.0;
    y_position[i] = 0.0;
    hold_center[i] = true;  // 上电即保持中心
    last_keepalive_time[i] = micros();
  }

  Serial.print(XY2_LIB_LABEL);
  Serial.println(" initialized");
  Serial.println("=====================================");
  Serial.println("Commands:");
  Serial.println("  GALVO:n    - Select galvo head (1-2)");
  Serial.println("  XY:x,y     - Move active galvo to position");
  Serial.println("  XYn:x,y    - Move specific galvo to position");
  Serial.println("  XYS:x,y    - Move active galvo with signed codes");
  Serial.println("  LASER:ON   - Turn laser on");
  Serial.println("  LASER:OFF  - Turn laser off");
  Serial.println("  LASER:1/0  - Alternate laser control (ON/OFF)");
  Serial.println("  MODE:POINT - Set laser burn to single point");
  Serial.println("  MODE:SPIRAL- Set laser burn to spiral");
  Serial.println("  SPIRAL:CONFIG:R,S,D[,A] - Configure spiral radius, spacing, delay(us), angle step");
  Serial.println("  LIMITS:n:xmin,xmax,ymin,ymax - Set galvo code limits");
  Serial.println("  CENTER     - Move to center (0,0) and hold");
  Serial.println("  PING       - Health check response");
  Serial.println("  STOP       - Emergency stop");
  Serial.println("  STATUS     - Get system status");
  Serial.println("  TEST       - Run test pattern");
  Serial.println("  SMOOTH:ON  - Enable smooth move");
  Serial.println("  SMOOTH:OFF - Disable smooth move");
  Serial.println("=====================================");
  Serial.println("READY:DUAL_GALVO");
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
  if (smooth_move) {
    unsigned long current_time = micros();
    for (uint8_t i = 0; i < GALVO_COUNT; i++) {
      if (is_moving[i] && current_time - last_update_time[i] >= UPDATE_INTERVAL) {
        updatePosition(i);
        last_update_time[i] = current_time;
      }
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
  for (uint8_t i = 0; i < GALVO_COUNT; i++) {
    if (hold_center[i] && !is_moving[i]) {
      if (now_us - last_keepalive_time[i] >= KEEPALIVE_INTERVAL_US) {
        galvos[i]->setSignedXY(current_x[i], current_y[i]);
        last_keepalive_time[i] = now_us;
      }
    }
  }
}

// ===== 命令解析 =====
void parseCommand(String cmd) {
  cmd.toUpperCase();

  // 调试输出
  Serial.print("CMD: ");
  Serial.println(cmd);

  // 基础握手命令
  if (cmd == "PING") {
    sendResponse("PONG");
    return;
  }

  // 紧急停止
  if (cmd == "STOP" || cmd == "!") {
    emergencyStop();
    return;
  }

  // 选择振镜
  if (cmd.startsWith("GALVO:")) {
    int index = cmd.substring(6).toInt() - 1;
    if (index >= 0 && index < GALVO_COUNT) {
      active_galvo = static_cast<uint8_t>(index);
      sendResponse("OK:GALVO:" + String(active_galvo + 1));
    } else {
      sendResponse("ERROR:INVALID GALVO INDEX");
    }
    return;
  }

  if (cmd.startsWith("LIMITS:")) {
    int firstColon = cmd.indexOf(':', 7);
    if (firstColon > 7) {
      int galvoIdx = cmd.substring(7, firstColon).toInt() - 1;
      if (galvoIdx < 0 || galvoIdx >= GALVO_COUNT) {
        sendResponse("ERROR:INVALID GALVO INDEX");
        return;
      }

      String params = cmd.substring(firstColon + 1);
      int firstComma = params.indexOf(',');
      int secondComma = params.indexOf(',', firstComma + 1);
      int thirdComma = params.indexOf(',', secondComma + 1);

      if (firstComma > 0 && secondComma > firstComma && thirdComma > secondComma) {
        int16_t xmin = constrain(params.substring(0, firstComma).toInt(), CODE_LIMIT_MIN, CODE_LIMIT_MAX);
        int16_t xmax = constrain(params.substring(firstComma + 1, secondComma).toInt(), CODE_LIMIT_MIN, CODE_LIMIT_MAX);
        int16_t ymin = constrain(params.substring(secondComma + 1, thirdComma).toInt(), CODE_LIMIT_MIN, CODE_LIMIT_MAX);
        int16_t ymax = constrain(params.substring(thirdComma + 1).toInt(), CODE_LIMIT_MIN, CODE_LIMIT_MAX);

        setGalvoLimits(galvoIdx, xmin, xmax, ymin, ymax);
      } else {
        sendResponse("ERROR:INVALID LIMIT FORMAT");
      }
    } else {
      sendResponse("ERROR:INVALID LIMIT FORMAT");
    }
    return;
  }

  // XY位置命令（优先处理有符号格式）
  if (cmd.startsWith("XYS")) {
    int colonIndex = cmd.indexOf(':', 3);
    if (colonIndex > 0) {
      uint8_t target_galvo = active_galvo;
      if (colonIndex > 3) {
        int idx = cmd.substring(3, colonIndex).toInt() - 1;
        if (idx < 0 || idx >= GALVO_COUNT) {
          sendResponse("ERROR:INVALID GALVO INDEX");
          return;
        }
        target_galvo = static_cast<uint8_t>(idx);
      }

      int commaIndex = cmd.indexOf(',', colonIndex + 1);
      if (commaIndex > colonIndex) {
        String xStr = cmd.substring(colonIndex + 1, commaIndex);
        String yStr = cmd.substring(commaIndex + 1);
        xStr.trim();
        yStr.trim();

        long yVal = xStr.toInt();
        long xVal = yStr.toInt();

        int16_t x = clampToGalvoRange(target_galvo, xVal, true);
        int16_t y = clampToGalvoRange(target_galvo, yVal, false);

        moveToPosition(target_galvo, x, y);
        sendResponse("OK:G" + String(target_galvo + 1) + ":XYS:" + String(x) + "," + String(y));
      } else {
        sendResponse("ERROR:INVALID XYS FORMAT");
      }
    } else {
      sendResponse("ERROR:INVALID XYS FORMAT");
    }
    return;
  }

  // XY位置命令（支持 XY:, XY1:, XY2:, ...）
  if (cmd.startsWith("XY")) {
    int colonIndex = cmd.indexOf(':', 2);
    if (colonIndex > 0) {
      uint8_t target_galvo = active_galvo;
      if (colonIndex > 2) {
        int idx = cmd.substring(2, colonIndex).toInt() - 1;
        if (idx < 0 || idx >= GALVO_COUNT) {
          sendResponse("ERROR:INVALID GALVO INDEX");
          return;
        }
        target_galvo = static_cast<uint8_t>(idx);
      }

      int commaIndex = cmd.indexOf(',', colonIndex + 1);
      if (commaIndex > colonIndex) {
        String xStr = cmd.substring(colonIndex + 1, commaIndex);
        String yStr = cmd.substring(commaIndex + 1);
        xStr.trim();
        yStr.trim();

        // 支持两种输入格式：
        // 1. 有符号格式 (-32768 to 32767)
        // 2. 无符号格式 (0 to 65535, 32768为中心)
        // 注意：XY传输中需要对调XY轴
        long yVal = xStr.toInt();
        long xVal = yStr.toInt();

        if (xVal > 32767) xVal -= 65536;
        if (yVal > 32767) yVal -= 65536;

        int16_t x = clampToGalvoRange(target_galvo, xVal, true);
        int16_t y = clampToGalvoRange(target_galvo, yVal, false);

        moveToPosition(target_galvo, x, y);
        sendResponse("OK:G" + String(target_galvo + 1) + ":XY:" + String(x) + "," + String(y));
      } else {
        sendResponse("ERROR:INVALID XY FORMAT");
      }
    } else {
      sendResponse("ERROR:INVALID XY FORMAT");
    }
    return;
  }
  // 激光控制
  else if (cmd == "LASER:ON" || cmd == "LASER:1") {
    if (laser_mode == LASER_MODE_POINT) {
      setLaser(true);
      sendResponse("OK:LASER:ON");
    } else {
      int16_t cx = requested_target_x[active_galvo];
      int16_t cy = requested_target_y[active_galvo];
      drawSpiral(active_galvo, cx, cy, spiral_max_radius, spiral_spacing, spiral_angle_step);
      sendResponse("OK:LASER:SPIRAL");
    }
  }
  else if (cmd == "LASER:OFF" || cmd == "LASER:0") {
    setLaser(false);
    sendResponse("OK:LASER:OFF");
  }
  // 激光模式切换
  else if (cmd == "MODE:POINT") {
    laser_mode = LASER_MODE_POINT;
    sendResponse("OK:MODE:POINT");
  }
  else if (cmd == "MODE:SPIRAL") {
    laser_mode = LASER_MODE_SPIRAL;
    sendResponse("OK:MODE:SPIRAL");
  }
  else if (cmd.startsWith("SPIRAL:CONFIG:")) {
    String params = cmd.substring(14);
    int firstComma = params.indexOf(',');
    int secondComma = params.indexOf(',', firstComma + 1);
    int thirdComma = params.indexOf(',', secondComma + 1);

    if (firstComma > 0) {
      int16_t radius = constrain(params.substring(0, firstComma).toInt(), 100, 20000);
      float spacing = params.substring(firstComma + 1, secondComma > firstComma ? secondComma : params.length()).toFloat();
      unsigned int delay_us = spiral_point_delay_us;
      float angle = spiral_angle_step;

      if (secondComma > firstComma) {
        String delayStr = thirdComma > secondComma ? params.substring(secondComma + 1, thirdComma) : params.substring(secondComma + 1);
        delay_us = max(200, delayStr.toInt());
      }

      if (thirdComma > secondComma) {
        String angleStr = params.substring(thirdComma + 1);
        angle = max(0.02f, angleStr.toFloat());
      }

      spiral_max_radius = radius;
      spiral_spacing = max(10.0f, spacing);
      spiral_point_delay_us = delay_us;
      spiral_angle_step = angle;
      sendResponse("OK:SPIRAL:CONFIG:" + String(spiral_max_radius) + "," + String(spiral_spacing, 2) + "," + String(spiral_point_delay_us) + "," + String(spiral_angle_step, 3));
    } else {
      sendResponse("ERROR:INVALID SPIRAL CONFIG");
    }
  }
  // 移动到中心并保持
  else if (cmd == "CENTER" || cmd == "HOME") {
    moveToPosition(0, 0, true);
    hold_center[active_galvo] = true;  // 进入保持中心模式
    sendResponse("OK:CENTER:G" + String(active_galvo + 1));
  }
  else if (cmd == "CENTER:ALL") {
    for (uint8_t i = 0; i < GALVO_COUNT; i++) {
      moveToPosition(i, 0, 0, true);
      hold_center[i] = true;
    }
    sendResponse("OK:CENTER:ALL");
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
    x_scale[active_galvo] = cmd.substring(12).toFloat();
    sendResponse("OK:CAL:X:SCALE:G" + String(active_galvo + 1) + ":" + String(x_scale[active_galvo], 4));
  }
  else if (cmd.startsWith("CAL:Y:SCALE:")) {
    y_scale[active_galvo] = cmd.substring(12).toFloat();
    sendResponse("OK:CAL:Y:SCALE:G" + String(active_galvo + 1) + ":" + String(y_scale[active_galvo], 4));
  }
  else if (cmd.startsWith("CAL:X:OFFSET:")) {
    x_offset[active_galvo] = cmd.substring(13).toInt();
    sendResponse("OK:CAL:X:OFFSET:G" + String(active_galvo + 1) + ":" + String(x_offset[active_galvo]));
  }
  else if (cmd.startsWith("CAL:Y:OFFSET:")) {
    y_offset[active_galvo] = cmd.substring(13).toInt();
    sendResponse("OK:CAL:Y:OFFSET:G" + String(active_galvo + 1) + ":" + String(y_offset[active_galvo]));
  }
  // 获取版本信息
  else if (cmd == "VERSION" || cmd == "VER") {
    sendResponse(String("VERSION:1.1:") + XY2_LIB_LABEL + ":TEENSY32:DUAL");
  }
  // 画圆测试
  else if (cmd.startsWith("CIRCLE:")) {
    String rStr = cmd.substring(7);
    int16_t radius = constrain(rStr.toInt(), 100, 10000);
    drawCircle(active_galvo, 0, 0, radius);
    sendResponse("OK:CIRCLE:G" + String(active_galvo + 1) + ":" + String(radius));
  }
  // 画矩形测试
  else if (cmd.startsWith("RECT:")) {
    int commaIndex = cmd.indexOf(',', 5);
    if (commaIndex > 0) {
      String wStr = cmd.substring(5, commaIndex);
      String hStr = cmd.substring(commaIndex + 1);
      int16_t width = constrain(wStr.toInt(), 100, 20000);
      int16_t height = constrain(hStr.toInt(), 100, 20000);
      drawRectangle(active_galvo, -width / 2, -height / 2, width / 2, height / 2);
      sendResponse("OK:RECT:G" + String(active_galvo + 1) + ":" + String(width) + "," + String(height));
    }
  }
  else {
    sendResponse("ERROR:Unknown command: " + cmd);
  }
}

// ===== 位置控制 =====
void moveToPosition(uint8_t galvo_index, int16_t x, int16_t y, bool immediate) {
  if (galvo_index >= GALVO_COUNT) {
    return;
  }

  // 一旦有移动请求，解除中心保持
  hold_center[galvo_index] = false;

  // 记录原始目标（并限制范围）
  requested_target_x[galvo_index] = clampToGalvoRange(galvo_index, x, true);
  requested_target_y[galvo_index] = clampToGalvoRange(galvo_index, y, false);

  // 应用校准
  int16_t calibrated_x = applyCalibration(requested_target_x[galvo_index], x_scale[galvo_index], x_offset[galvo_index]);
  int16_t calibrated_y = applyCalibration(requested_target_y[galvo_index], y_scale[galvo_index], y_offset[galvo_index]);

  // 限制范围
  calibrated_x = clampToGalvoRange(galvo_index, calibrated_x, true);
  calibrated_y = clampToGalvoRange(galvo_index, calibrated_y, false);

  target_x[galvo_index] = calibrated_x;
  target_y[galvo_index] = calibrated_y;

  if (!smooth_move || immediate) {
    // 直接移动
    galvos[galvo_index]->setSignedXY(calibrated_x, calibrated_y);
    current_x[galvo_index] = calibrated_x;
    current_y[galvo_index] = calibrated_y;
    x_position[galvo_index] = float(calibrated_x);
    y_position[galvo_index] = float(calibrated_y);
    is_moving[galvo_index] = false;
    last_keepalive_time[galvo_index] = micros();
  } else {
    // 开始平滑移动
    is_moving[galvo_index] = true;
  }
}

void moveToPosition(int16_t x, int16_t y, bool immediate) {
  moveToPosition(active_galvo, x, y, immediate);
}

// ===== 平滑位置更新 =====
void updatePosition(uint8_t galvo_index) {
  if (galvo_index >= GALVO_COUNT || !is_moving[galvo_index]) return;

  // 计算到目标的距离
  float dx = target_x[galvo_index] - x_position[galvo_index];
  float dy = target_y[galvo_index] - y_position[galvo_index];
  float distance = sqrt(dx * dx + dy * dy);

  if (distance < 1.0f) {
    // 到达目标
    x_position[galvo_index] = target_x[galvo_index];
    y_position[galvo_index] = target_y[galvo_index];
    current_x[galvo_index] = target_x[galvo_index];
    current_y[galvo_index] = target_y[galvo_index];
    galvos[galvo_index]->setSignedXY(current_x[galvo_index], current_y[galvo_index]);
    is_moving[galvo_index] = false;
  } else {
    // 计算移动步长
    float step = min(MOVE_STEP, distance);
    float ratio = step / distance;

    // 更新位置
    x_position[galvo_index] += dx * ratio;
    y_position[galvo_index] += dy * ratio;

    // 转换为整数并发送
    current_x[galvo_index] = clampToGalvoRange(galvo_index, int16_t(x_position[galvo_index]), true);
    current_y[galvo_index] = clampToGalvoRange(galvo_index, int16_t(y_position[galvo_index]), false);
    galvos[galvo_index]->setSignedXY(current_x[galvo_index], current_y[galvo_index]);
  }

  last_keepalive_time[galvo_index] = micros();
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
  for (uint8_t i = 0; i < GALVO_COUNT; i++) {
    is_moving[i] = false;
    target_x[i] = current_x[i];
    target_y[i] = current_y[i];
    requested_target_x[i] = current_x[i];
    requested_target_y[i] = current_y[i];

    // 发送当前位置（确保振镜停止）
    galvos[i]->setSignedXY(current_x[i], current_y[i]);

    // 进入保持中心模式如果当前位置就是中心
    if (current_x[i] == 0 && current_y[i] == 0) {
      hold_center[i] = true;
    }
  }

  sendResponse("EMERGENCY_STOP");
  Serial.println("!!! EMERGENCY STOP !!!");
}

// ===== 状态报告 =====
void sendStatus() {
  String status = "STATUS:";
  status += "MODE=" + String(laserModeName());
  status += ",ACTIVE=G" + String(active_galvo + 1);
  status += ",LASER=" + String(laser_enabled ? "ON" : "OFF");
  status += ",SPIRAL(R=" + String(spiral_max_radius);
  status += ",S=" + String(spiral_spacing, 2);
  status += ",D=" + String(spiral_point_delay_us);
  status += ",A=" + String(spiral_angle_step, 3) + ")";

  for (uint8_t i = 0; i < GALVO_COUNT; i++) {
    status += ",G" + String(i + 1) + "(X=" + String(current_x[i]);
    status += ",Y=" + String(current_y[i]);
    status += ",TX=" + String(target_x[i]);
    status += ",TY=" + String(target_y[i]);
    status += ",REQX=" + String(requested_target_x[i]);
    status += ",REQY=" + String(requested_target_y[i]);
    status += ",MOV=" + String(is_moving[i] ? "YES" : "NO");
    status += ",HOLD=" + String(hold_center[i] ? "YES" : "NO") + ")";
  }

  sendResponse(status);
}

// ===== 发送响应 =====
void sendResponse(String response) {
  Serial.println(response);
}

// ===== 测试图案 =====
void testPattern() {
  uint8_t idx = active_galvo;
  Serial.println("Starting test pattern (G" + String(idx + 1) + ")...");

  // 保存当前状态
  int16_t saved_x = requested_target_x[idx];
  int16_t saved_y = requested_target_y[idx];
  bool saved_laser = laser_enabled;
  bool saved_hold = hold_center[idx];

  // 测试前解除保持，避免保活干扰
  hold_center[idx] = false;

  // 测试1：中心点
  Serial.println("Test 1: Center point");
  moveToPosition(idx, 0, 0, true);
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
    moveToPosition(idx, corners[i][0], corners[i][1], true);
    delay(300);
    setLaser(true);
    delay(200);
    setLaser(false);
    delay(200);
  }

  // 测试3：十字线
  Serial.println("Test 3: Cross");
  moveToPosition(idx, -5000, 0, true);
  delay(200);
  setLaser(true);
  moveToPosition(idx, 5000, 0, true);
  delay(500);
  setLaser(false);

  moveToPosition(idx, 0, -5000, true);
  delay(200);
  setLaser(true);
  moveToPosition(idx, 0, 5000, true);
  delay(500);
  setLaser(false);

  // 测试4：小圆
  Serial.println("Test 4: Small circle");
  drawCircle(idx, 0, 0, 3000, 30);

  // 恢复状态
  moveToPosition(idx, saved_x, saved_y, true);
  setLaser(saved_laser);
  hold_center[idx] = saved_hold;

  Serial.println("Test pattern completed");
}

// ===== 画圆 =====
void drawCircle(uint8_t galvo_index, int16_t cx, int16_t cy, int16_t radius, int points) {
  if (galvo_index >= GALVO_COUNT) {
    return;
  }

  // 临时解除保持，避免保活干扰
  bool prev_hold = hold_center[galvo_index];
  hold_center[galvo_index] = false;

  // 移动到起始点
  int16_t start_x = cx + radius;
  int16_t start_y = cy;
  moveToPosition(galvo_index, start_x, start_y, true);
  delay(100);

  // 打开激光
  setLaser(true);

  // 画圆
  for (int i = 0; i <= points; i++) {
    float angle = 2.0 * PI * i / points;
    int16_t x = cx + radius * cos(angle);
    int16_t y = cy + radius * sin(angle);
    moveToPosition(galvo_index, x, y, true);
    delay(20);
  }

  // 关闭激光
  setLaser(false);

  // 恢复保持状态
  hold_center[galvo_index] = prev_hold;
}

void drawCircle(int16_t cx, int16_t cy, int16_t radius, int points) {
  drawCircle(active_galvo, cx, cy, radius, points);
}

// ===== 画矩形 =====
void drawRectangle(uint8_t galvo_index, int16_t x1, int16_t y1, int16_t x2, int16_t y2) {
  if (galvo_index >= GALVO_COUNT) {
    return;
  }

  // 临时解除保持，避免保活干扰
  bool prev_hold = hold_center[galvo_index];
  hold_center[galvo_index] = false;

  // 移动到起始点
  moveToPosition(galvo_index, x1, y1, true);
  delay(100);

  // 打开激光
  setLaser(true);

  // 画四条边
  moveToPosition(galvo_index, x2, y1, true);
  delay(100);
  moveToPosition(galvo_index, x2, y2, true);
  delay(100);
  moveToPosition(galvo_index, x1, y2, true);
  delay(100);
  moveToPosition(galvo_index, x1, y1, true);
  delay(100);

  // 关闭激光
  setLaser(false);

  // 恢复保持状态
  hold_center[galvo_index] = prev_hold;
}

void drawRectangle(int16_t x1, int16_t y1, int16_t x2, int16_t y2) {
  drawRectangle(active_galvo, x1, y1, x2, y2);
}

// ===== 螺旋灼烧 =====
void drawSpiral(uint8_t galvo_index, int16_t cx, int16_t cy, int16_t max_radius, float spacing, float angle_step) {
  if (galvo_index >= GALVO_COUNT) {
    return;
  }

  bool prev_hold = hold_center[galvo_index];
  hold_center[galvo_index] = false;

  int16_t x_pos_range = max((int16_t)0, x_max_limits[galvo_index]);
  int16_t x_neg_range = max((int16_t)0, (int16_t)(-x_min_limits[galvo_index]));
  int16_t y_pos_range = max((int16_t)0, y_max_limits[galvo_index]);
  int16_t y_neg_range = max((int16_t)0, (int16_t)(-y_min_limits[galvo_index]));
  int16_t symmetric_limit = min(min(x_pos_range, x_neg_range), min(y_pos_range, y_neg_range));
  if (symmetric_limit < 100) {
    symmetric_limit = 100;
  }
  int16_t limited_radius = constrain(max_radius, 100, symmetric_limit);
  float effective_spacing = max(10.0f, spacing);
  float step = angle_step > 0.01f ? angle_step : 0.1f;

  int16_t saved_x = requested_target_x[galvo_index];
  int16_t saved_y = requested_target_y[galvo_index];

  moveToPosition(galvo_index, cx, cy, true);
  delay(50);

  setLaser(true);

  for (float angle = 0.0f;; angle += step) {
    float radius = (effective_spacing * angle) / (2.0f * PI);
    if (radius > limited_radius) {
      break;
    }

    int16_t x = cx + int16_t(radius * cos(angle));
    int16_t y = cy + int16_t(radius * sin(angle));
    x = clampToGalvoRange(galvo_index, x, true);
    y = clampToGalvoRange(galvo_index, y, false);
    moveToPosition(galvo_index, x, y, true);
    delayMicroseconds(spiral_point_delay_us);
  }

  setLaser(false);

  // 返回原始点位
  moveToPosition(galvo_index, saved_x, saved_y, true);
  hold_center[galvo_index] = prev_hold;
}

int16_t clampToGalvoRange(uint8_t galvo_index, int16_t value, bool isX) {
  if (galvo_index >= GALVO_COUNT) {
    return value;
  }

  int16_t minVal = isX ? x_min_limits[galvo_index] : y_min_limits[galvo_index];
  int16_t maxVal = isX ? x_max_limits[galvo_index] : y_max_limits[galvo_index];

  if (minVal > maxVal) {
    int16_t tmp = minVal;
    minVal = maxVal;
    maxVal = tmp;
  }

  return constrain(value, minVal, maxVal);
}

void setGalvoLimits(uint8_t galvo_index, int16_t xmin, int16_t xmax, int16_t ymin, int16_t ymax) {
  if (galvo_index >= GALVO_COUNT) {
    sendResponse("ERROR:INVALID GALVO INDEX");
    return;
  }

  if (xmin > xmax) {
    int16_t tmp = xmin;
    xmin = xmax;
    xmax = tmp;
  }
  if (ymin > ymax) {
    int16_t tmp = ymin;
    ymin = ymax;
    ymax = tmp;
  }

  x_min_limits[galvo_index] = constrain(xmin, CODE_LIMIT_MIN, CODE_LIMIT_MAX);
  x_max_limits[galvo_index] = constrain(xmax, CODE_LIMIT_MIN, CODE_LIMIT_MAX);
  y_min_limits[galvo_index] = constrain(ymin, CODE_LIMIT_MIN, CODE_LIMIT_MAX);
  y_max_limits[galvo_index] = constrain(ymax, CODE_LIMIT_MIN, CODE_LIMIT_MAX);

  current_x[galvo_index] = clampToGalvoRange(galvo_index, current_x[galvo_index], true);
  current_y[galvo_index] = clampToGalvoRange(galvo_index, current_y[galvo_index], false);
  target_x[galvo_index] = clampToGalvoRange(galvo_index, target_x[galvo_index], true);
  target_y[galvo_index] = clampToGalvoRange(galvo_index, target_y[galvo_index], false);
  requested_target_x[galvo_index] = clampToGalvoRange(galvo_index, requested_target_x[galvo_index], true);
  requested_target_y[galvo_index] = clampToGalvoRange(galvo_index, requested_target_y[galvo_index], false);

  galvos[galvo_index]->setSignedXY(current_x[galvo_index], current_y[galvo_index]);

  sendResponse(
    "OK:LIMITS:G" + String(galvo_index + 1) + ":" +
    String(x_min_limits[galvo_index]) + "," + String(x_max_limits[galvo_index]) + "," +
    String(y_min_limits[galvo_index]) + "," + String(y_max_limits[galvo_index])
  );
}

// ===== 辅助函数 =====
int16_t applyCalibration(int16_t value, float scale, int16_t offset) {
  return int16_t(value * scale) + offset;
}

const char* laserModeName() {
  switch (laser_mode) {
    case LASER_MODE_SPIRAL:
      return "SPIRAL";
    case LASER_MODE_POINT:
    default:
      return "POINT";
  }
}

#if !defined(__has_include)
#define __has_include(x) 0
#endif
#include <XY2_100_multi.h>

using XY2Driver = XY2_100;
const char XY2_LIB_LABEL[] = "XY2-100_multi";
#include <math.h>

// ==========================================
//           1. 硬件配置 (最关键的修改区)
// ==========================================
const uint8_t GALVO_COUNT = 2;

// --- 激光引脚 ---
// 建议保持 Pin 3, 4 以避免冲突
const int LASER_PINS[GALVO_COUNT] = {3, 4}; 

// --- 物理端口互换开关 ---
// 如果你发送 XY1 动的是你认为的 2号振镜，请把这个改成 true
// false: G1=PortD(Pin2...), G2=PortC(Pin15...)
// true:  G1=PortC(Pin15...), G2=PortD(Pin2...)
const bool SWAP_HARDWARE_PORTS = false; 

// ==========================================
<<<<<<< HEAD
//              全局变量
=======
//               全局变量
>>>>>>> ccfcea0209b2e31744988083664d289e28221df7
// ==========================================
enum LaserMode {
  LASER_MODE_POINT = 0,
  LASER_MODE_SPIRAL = 1
};

XY2Driver galvo_driver_A; // 原 Primary
XY2Driver galvo_driver_B; // 原 Secondary
// 根据开关动态分配指针
XY2Driver* galvos[GALVO_COUNT]; 

volatile int16_t base_x[GALVO_COUNT] = {0}; 
volatile int16_t base_y[GALVO_COUNT] = {0};
int16_t current_x[GALVO_COUNT] = {0};
int16_t current_y[GALVO_COUNT] = {0};

uint8_t active_galvo = 0; // 0 = G1, 1 = G2

bool laser_enabled[GALVO_COUNT] = {false, false};
bool is_moving[GALVO_COUNT] = {false};
bool smooth_move = false;

// 螺旋状态机
bool is_spiraling[GALVO_COUNT] = {false, false};
float spiral_current_angle[GALVO_COUNT] = {0.0, 0.0};
unsigned long last_spiral_update_us[GALVO_COUNT] = {0, 0};
unsigned long spiral_start_time_us[GALVO_COUNT] = {0, 0};  // 螺旋线开始时间（微秒）
unsigned long spiral_target_duration_us = 200000;  // 螺旋线目标持续时间（微秒），默认200ms

// 平滑插值
float smooth_x[GALVO_COUNT] = {0.0};
float smooth_y[GALVO_COUNT] = {0.0};
const float MOVE_STEP = 500.0;

// 范围限制
const int16_t CODE_LIMIT_MIN = -32767;
const int16_t CODE_LIMIT_MAX = 32767;
int16_t x_min_limits[GALVO_COUNT] = {CODE_LIMIT_MIN, CODE_LIMIT_MIN};
int16_t x_max_limits[GALVO_COUNT] = {CODE_LIMIT_MAX, CODE_LIMIT_MAX};
int16_t y_min_limits[GALVO_COUNT] = {CODE_LIMIT_MIN, CODE_LIMIT_MIN};
int16_t y_max_limits[GALVO_COUNT] = {CODE_LIMIT_MAX, CODE_LIMIT_MAX};

// 其他
char serial_buffer[256];
size_t buffer_index = 0;
unsigned long last_update_time[GALVO_COUNT] = {0};
unsigned long laser_start_time[GALVO_COUNT] = {0, 0};
bool hold_center[GALVO_COUNT] = {true, true};
unsigned long last_keepalive_time[GALVO_COUNT] = {0, 0};

const unsigned long UPDATE_INTERVAL = 10;
const unsigned long MAX_LASER_ON_TIME = 500;
const unsigned long KEEPALIVE_INTERVAL_US = 1000;

// 螺旋参数
LaserMode laser_mode = LASER_MODE_POINT;
int16_t spiral_max_radius = 6000;
float spiral_spacing = 1200.0;
float spiral_angle_step = 0.25;
unsigned int spiral_point_delay_us = 1000;

// ==========================================
//                 函数声明
// ==========================================
void parseCommand(String cmd);
void setBasePosition(uint8_t galvo_index, int16_t x, int16_t y, bool immediate = false);
void processSpiral(uint8_t galvo_index);
void updateSmoothMove(uint8_t galvo_index);
void sendToGalvo(uint8_t galvo_index, int16_t x, int16_t y);
void setLaser(uint8_t galvo_index, bool on);
void emergencyStop();
void sendResponse(String response);
int16_t clampToGalvoRange(uint8_t galvo_index, int16_t value, bool isX);

// ==========================================
//                 Setup
// ==========================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}

  Serial.println("=====================================");
  Serial.println("  Teensy Galvo Controller V2.4");
  Serial.println("  Feature: Auto-Switch Active Galvo");
<<<<<<< HEAD
=======
  Serial.println("  Feature: Spiral Overflow Fix");
>>>>>>> ccfcea0209b2e31744988083664d289e28221df7
  Serial.println("=====================================");

  for (uint8_t i = 0; i < GALVO_COUNT; i++) {
    pinMode(LASER_PINS[i], OUTPUT);
    digitalWrite(LASER_PINS[i], LOW);
  }

  // --- 振镜分配逻辑 (处理互换) ---
  if (!SWAP_HARDWARE_PORTS) {
    // 默认: G1=PortD, G2=PortC
    Serial.println("Config: G1->PortD, G2->PortC");
    galvos[0] = &galvo_driver_A; // G1
    galvos[1] = &galvo_driver_B; // G2
  } else {
    // 互换: G1=PortC, G2=PortD
    Serial.println("Config: G1->PortC, G2->PortD (SWAPPED)");
    galvos[0] = &galvo_driver_B; // G1
    galvos[1] = &galvo_driver_A; // G2
  }

  Serial.println("Initializing hardware...");
  // Port D (Pins 2, 14, 7, 8...)
  galvo_driver_A.begin(2, &GPIOD_PDOR, 2, 14, 7, 8, 6, 20, 21, 5);
  delay(50);
  // Port C (Pins 15, 22, 23...)
  galvo_driver_B.begin(1, &GPIOC_PDOR, 15, 22, 23, 9, 10, 13, 11, 12);
  delay(50);

  // 归零
  for (uint8_t i = 0; i < GALVO_COUNT; i++) {
    sendToGalvo(i, 0, 0);
    hold_center[i] = true;
    last_keepalive_time[i] = micros();
  }
  Serial.println("System Ready.");
}

// ==========================================
//                 Main Loop
// ==========================================
void loop() {
  if (Serial.available()) {
    while (Serial.available()) {
      char c = Serial.read();
      if (c == '\n' || c == '\r') {
        if (buffer_index > 0) {
          serial_buffer[buffer_index] = '\0';
          String command = String(serial_buffer);
          command.trim();
          if (command.length() > 0) parseCommand(command);
          buffer_index = 0;
        }
      } else if (buffer_index < sizeof(serial_buffer) - 1) {
        serial_buffer[buffer_index++] = c;
      }
    }
  }

  unsigned long now_us = micros();
  
  for (uint8_t i = 0; i < GALVO_COUNT; i++) {
    // 激光超时保护
    if (laser_enabled[i] && (millis() - laser_start_time[i] > MAX_LASER_ON_TIME)) {
      setLaser(i, false);
      is_spiraling[i] = false;
    }

    // 螺旋追踪 (最高优先级)
    if (is_spiraling[i]) {
      processSpiral(i); 
    } 
    // 平滑移动
    else if (smooth_move && is_moving[i]) {
      if (now_us - last_update_time[i] >= UPDATE_INTERVAL) {
        updateSmoothMove(i);
        last_update_time[i] = now_us;
      }
    }
    // 静态保活 (防止掉线)
    else if (hold_center[i]) {
      if (now_us - last_keepalive_time[i] >= KEEPALIVE_INTERVAL_US) {
        sendToGalvo(i, base_x[i], base_y[i]); 
        last_keepalive_time[i] = now_us;
      }
    }
  }
}

// ==========================================
<<<<<<< HEAD
//              核心：螺旋叠加
=======
//              核心：螺旋叠加 (修复溢出版)
>>>>>>> ccfcea0209b2e31744988083664d289e28221df7
// ==========================================
void processSpiral(uint8_t i) {
  unsigned long now = micros();
  if (now - last_spiral_update_us[i] < spiral_point_delay_us) return;
  last_spiral_update_us[i] = now;

  // 检查是否达到目标时间
  unsigned long elapsed_us = now - spiral_start_time_us[i];
  if (elapsed_us >= spiral_target_duration_us) {
    setLaser(i, false);
    is_spiraling[i] = false;
    hold_center[i] = false; 
    sendToGalvo(i, base_x[i], base_y[i]); // 归位到当前的基座位置
    Serial.println("INFO:G" + String(i+1) + ":SPIRAL_DONE");
    return;
  }

  float radius = (max(10.0f, spiral_spacing) * spiral_current_angle[i]) / (2.0f * PI);

<<<<<<< HEAD
  // 如果达到最大半径，也停止（双重保护）
=======
  // 如果达到最大半径，也停止
>>>>>>> ccfcea0209b2e31744988083664d289e28221df7
  if (radius > spiral_max_radius) {
    setLaser(i, false);
    is_spiraling[i] = false;
    hold_center[i] = false; 
<<<<<<< HEAD
    sendToGalvo(i, base_x[i], base_y[i]); // 归位到当前的基座位置
=======
    sendToGalvo(i, base_x[i], base_y[i]); 
>>>>>>> ccfcea0209b2e31744988083664d289e28221df7
    Serial.println("INFO:G" + String(i+1) + ":SPIRAL_DONE");
    return;
  }

<<<<<<< HEAD
  int16_t offset_x = int16_t(radius * cos(spiral_current_angle[i]));
  int16_t offset_y = int16_t(radius * sin(spiral_current_angle[i]));

  // 叠加基座 + 偏移
  int16_t final_x = base_x[i] + offset_x;
  int16_t final_y = base_y[i] + offset_y;

  sendToGalvo(i, final_x, final_y);
=======
  // --- 修复溢出开始 ---
  // 1. 使用 long 计算偏移量，防止溢出
  long offset_x = (long)(radius * cos(spiral_current_angle[i]));
  long offset_y = (long)(radius * sin(spiral_current_angle[i]));

  // 2. 计算最终坐标 (使用 long)
  long calc_x = (long)base_x[i] + offset_x;
  long calc_y = (long)base_y[i] + offset_y;

  // 3. 获取当前振镜的物理限制
  int16_t x_min = x_min_limits[i];
  int16_t x_max = x_max_limits[i];
  int16_t y_min = y_min_limits[i];
  int16_t y_max = y_max_limits[i];

  // 4. 手动限幅 (Clamp) - 关键步骤：防止溢出回绕
  if (calc_x > x_max) calc_x = x_max;
  if (calc_x < x_min) calc_x = x_min;
  if (calc_y > y_max) calc_y = y_max;
  if (calc_y < y_min) calc_y = y_min;

  // 5. 安全转换为 int16_t 并发送
  sendToGalvo(i, (int16_t)calc_x, (int16_t)calc_y);
  // --- 修复溢出结束 ---

>>>>>>> ccfcea0209b2e31744988083664d289e28221df7
  spiral_current_angle[i] += spiral_angle_step;
}

// ==========================================
//               辅助函数
// ==========================================
void setBasePosition(uint8_t i, int16_t x, int16_t y, bool immediate) {
  if (i >= GALVO_COUNT) return;
  hold_center[i] = false;

  if (!is_spiraling[i]) {
    if (!smooth_move || immediate) {
      base_x[i] = x;
      base_y[i] = y;
      sendToGalvo(i, base_x[i], base_y[i]);
      smooth_x[i] = (float)base_x[i];
      smooth_y[i] = (float)base_y[i];
      is_moving[i] = false;
    } else {
      base_x[i] = x;
      base_y[i] = y;
      is_moving[i] = true;
    }
  } else {
    // 螺旋模式：仅更新基座，让 processSpiral 处理叠加
    base_x[i] = x;
    base_y[i] = y;
  }
}

void sendToGalvo(uint8_t i, int16_t x, int16_t y) {
  int16_t cx = clampToGalvoRange(i, x, true);
  int16_t cy = clampToGalvoRange(i, y, false);
  current_x[i] = cx;
  current_y[i] = cy;
  galvos[i]->setSignedXY(cx, cy);
}

void updateSmoothMove(uint8_t i) {
  float dx = base_x[i] - smooth_x[i];
  float dy = base_y[i] - smooth_y[i];
  float dist = sqrt(dx*dx + dy*dy);
  if (dist < 1.0f) {
    smooth_x[i] = base_x[i];
    smooth_y[i] = base_y[i];
    is_moving[i] = false;
  } else {
    float step = min(MOVE_STEP, dist);
    smooth_x[i] += (dx/dist) * step;
    smooth_y[i] += (dy/dist) * step;
  }
  sendToGalvo(i, (int16_t)smooth_x[i], (int16_t)smooth_y[i]);
}

void setLaser(uint8_t i, bool on) {
  if (i >= GALVO_COUNT) return;
  laser_enabled[i] = on;
  digitalWrite(LASER_PINS[i], on ? HIGH : LOW);
  if (on) laser_start_time[i] = millis();
}

void emergencyStop() {
  for (uint8_t i = 0; i < GALVO_COUNT; i++) {
    setLaser(i, false);
    is_spiraling[i] = false;
    is_moving[i] = false;
    base_x[i] = current_x[i];
    base_y[i] = current_y[i];
    sendToGalvo(i, base_x[i], base_y[i]);
  }
  Serial.println("EMERGENCY_STOP");
}

int16_t clampToGalvoRange(uint8_t galvo_index, int16_t value, bool isX) {
  if (galvo_index >= GALVO_COUNT) return value;
  int16_t minVal = isX ? x_min_limits[galvo_index] : y_min_limits[galvo_index];
  int16_t maxVal = isX ? x_max_limits[galvo_index] : y_max_limits[galvo_index];
  return constrain(value, minVal, maxVal);
}

void sendResponse(String response) { Serial.println(response); }

// ==========================================
//           命令解析 (已修复焦点逻辑)
// ==========================================
void parseCommand(String cmd) {
  cmd.toUpperCase();
  if (cmd == "STOP" || cmd == "!") { emergencyStop(); return; }
  if (cmd == "PING") { sendResponse("PONG"); return; }

  // --- XY指令 (还原Code1逻辑 + 自动切换焦点) ---
  if (cmd.startsWith("XY")) {
    int colonIdx = cmd.indexOf(':');
    if (colonIdx > 0) {
      uint8_t target_id = active_galvo; // 默认为当前
      // 检测是否有明确的 ID (XY1, XY2)
      if (colonIdx > 2) {
        int id = cmd.substring(2, colonIdx).toInt();
        if (id > 0 && id <= GALVO_COUNT) {
          target_id = id - 1;
          active_galvo = target_id; // 【关键修正】：自动更新全局活动振镜
        }
      }

      int commaIdx = cmd.indexOf(',', colonIdx);
      if (commaIdx > colonIdx) {
        // 还原 Code 1 的解析逻辑 (Y在前, X在后? 不，Code1是参数对调)
        // Code 1 Logic: String1 -> Y, String2 -> X
        long val1 = cmd.substring(colonIdx + 1, commaIdx).toInt();
        long val2 = cmd.substring(commaIdx + 1).toInt();

        // 无符号溢出处理
        if (val1 > 32767) val1 -= 65536;
        if (val2 > 32767) val2 -= 65536;

        // Code 1 的硬件逻辑：setSignedXY(Y, X)
        // 所以这里 setBasePosition(target_id, val2, val1)
        // 参数1(X) = val2, 参数2(Y) = val1
        setBasePosition(target_id, (int16_t)val2, (int16_t)val1);
      }
    }
    return;
  }

  // --- 激光控制 (修改版：支持指令携带模式 LASERn:ON:SPIRAL) ---
  if (cmd.startsWith("LASER")) {
    int firstColon = cmd.indexOf(':');
    
    // 查找第二个冒号，用于分隔动作和模式 (例如 LASER1:ON:SPIRAL)
    int secondColon = cmd.indexOf(':', firstColon + 1);
    
    uint8_t target_id = active_galvo;
    
    // 1. 解析 ID (例如 LASER1 或 LASER2)
    if (firstColon > 5) { // "LASER" 长度是 5
      int id = cmd.substring(5, firstColon).toInt();
      if (id > 0 && id <= GALVO_COUNT) {
        target_id = id - 1;
        // active_galvo = target_id; // 可选：是否更新全局活动ID，建议保持不更新以免干扰
      }
    }
    
    // 2. 解析动作 (ON 或 OFF)
    String action;
    if (secondColon == -1) {
      // 没有模式参数，格式为 LASER:ON
      action = cmd.substring(firstColon + 1);
    } else {
      // 有模式参数，格式为 LASER:ON:SPIRAL
      action = cmd.substring(firstColon + 1, secondColon);
    }
    
    // 3. 解析临时模式 (如果存在)
    String tempMode = "";
    if (secondColon > 0) {
      tempMode = cmd.substring(secondColon + 1);
      tempMode.trim(); // 去除可能的换行符
    }
    
    // --- 执行控制 ---
    if (action == "ON" || action == "1") {
      // 决策：使用哪种模式？
      // 如果指令里明确带了 SPIRAL，就强制用螺旋；如果是 POINT，强制用点射
      // 否则沿用全局 laser_mode 变量
      bool use_spiral = (laser_mode == LASER_MODE_SPIRAL);
      
      if (tempMode == "SPIRAL") use_spiral = true;
      else if (tempMode == "POINT") use_spiral = false;
      
      if (use_spiral) {
        // === 启动螺旋模式 ===
        spiral_current_angle[target_id] = 0.0;
        last_spiral_update_us[target_id] = micros();
        spiral_start_time_us[target_id] = micros();
        is_spiraling[target_id] = true;
        
        setLaser(target_id, true);
        sendResponse("OK:SPIRAL_START:G" + String(target_id+1));
      } else {
        // === 启动点射模式 ===
        is_spiraling[target_id] = false; // 确保清除螺旋标志
        setLaser(target_id, true);
        sendResponse("OK:LASER_ON:G" + String(target_id+1));
      }
    } else {
      // OFF
      setLaser(target_id, false);
      is_spiraling[target_id] = false;
      sendResponse("OK:LASER_OFF:G" + String(target_id+1));
    }
    return;
  }
  
  // 其他指令
  if (cmd == "MODE:SPIRAL") { laser_mode = LASER_MODE_SPIRAL; sendResponse("OK:MODE:SPIRAL"); }
  else if (cmd == "MODE:POINT") { laser_mode = LASER_MODE_POINT; sendResponse("OK:MODE:POINT"); }
  else if (cmd.startsWith("SPIRAL:CONFIG:")) {
    String params = cmd.substring(14);
    int c1 = params.indexOf(',');
    int c2 = params.indexOf(',', c1+1);
    int c3 = params.indexOf(',', c2+1);
    if (c1 > 0) {
      spiral_max_radius = params.substring(0, c1).toInt();
      spiral_spacing = params.substring(c1+1, c2 > 0 ? c2 : params.length()).toFloat();
      if (c2 > 0) spiral_point_delay_us = params.substring(c2+1, c3 > 0 ? c3 : params.length()).toInt();
      if (c3 > 0) {
        // 第四个参数：目标持续时间（毫秒），转换为微秒
        unsigned long duration_ms = params.substring(c3+1).toInt();
        spiral_target_duration_us = duration_ms * 1000UL;
      }
      sendResponse("OK:SPIRAL_CONFIG");
    }
<<<<<<< HEAD
  }
  else if (cmd.startsWith("GALVO:")) {
    int idx = cmd.substring(6).toInt() - 1;
    if (idx >= 0 && idx < GALVO_COUNT) {
      active_galvo = idx;
      sendResponse("OK:GALVO:" + String(idx+1));
    }
  }
}
=======
  }
  else if (cmd.startsWith("GALVO:")) {
    int idx = cmd.substring(6).toInt() - 1;
    if (idx >= 0 && idx < GALVO_COUNT) {
      active_galvo = idx;
      sendResponse("OK:GALVO:" + String(idx+1));
    }
  }
}
>>>>>>> ccfcea0209b2e31744988083664d289e28221df7

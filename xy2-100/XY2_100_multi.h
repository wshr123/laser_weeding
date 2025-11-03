/*  XY2_100_multi 库
    重构以支持多个实例。
    基于 Lutz Lisseck 的原始工作, Copyright (c) 2018。
*/

#ifndef XY2_100_MULTI_H
#define XY2_100_MULTI_H

#include <Arduino.h>
#include "DMAChannel.h"

#if TEENSYDUINO < 121
#error "编译此库需要 Teensyduino 1.21 或更高版本。"
#endif
#ifdef __AVR__
#error "此库不适用于 Teensy 2.0 或 Teensy++ 2.0。"
#endif

// 如果您有足够的硬件资源（如 FTM 定时器、GPIO 端口），可以增加此值
#define MAX_GALVO_INSTANCES 2

class XY2_100 {
public:
    // 构造函数
    XY2_100();

    // begin 方法为该实例配置并启动硬件
    void begin(uint8_t ftm_module, volatile uint32_t* gpio_port_register, 
               uint8_t pin0, uint8_t pin1, uint8_t pin2, uint8_t pin3, 
               uint8_t pin4, uint8_t pin5, uint8_t pin6, uint8_t pin7, 
               uint32_t frequency = 4000000);

    void setXY(uint16_t X, uint16_t Y);
    void setSignedXY(int16_t X, int16_t Y);
    uint8_t stat(void);

private:
    // 实例特定的成员（非静态）
    uint16_t lastX;
    uint16_t lastY;
    void *pingBuffer; // 指向Ping缓冲区的指针
    void *pongBuffer; // 指向Pong缓冲区的指针
    DMAChannel dma;
    volatile uint8_t txPing;

    // *** 更改：移除了 DMAMEM 数组声明 ***
    // DMAMEM 数组不能作为非静态成员变量。
    // 它们将在 .cpp 文件中被静态分配。

    // 存储此实例的硬件配置
    uint8_t _ftm_module;
    volatile uint32_t* _gpio_port_register;
    uint8_t _pins[8];
    uint32_t _frequency;

    // 中断处理机制
    void isr_handler(void); // 实例的实际中断服务程序（ISR）逻辑
    
    // 由硬件中断调用的静态“跳板”函数
    static void isr_dispatcher_0(void);
    static void isr_dispatcher_1(void);
    
    // 静态数组，用于保存所有已创建实例的指针
    static XY2_100* instances[MAX_GALVO_INSTANCES];
    static uint8_t instance_count;
    uint8_t _instance_id; // 每个对象的唯一ID（0, 1等）
};

#endif // XY2_100_MULTI_H
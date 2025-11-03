/*  XY2_100_multi 库
    重构以支持多个实例。
    基于 Lutz Lisseck 的原始工作, Copyright (c) 2018。
*/

#include "XY2_100_multi.h"
#include <string.h>

// *** 更改：在此处静态分配所有实例所需的DMA内存 ***
// 编译器允许对静态或全局变量使用 DMAMEM 属性。
DMAMEM int dma_ping_buffers[MAX_GALVO_INSTANCES][10];
DMAMEM int dma_pong_buffers[MAX_GALVO_INSTANCES][10];


// 初始化静态成员
XY2_100* XY2_100::instances[MAX_GALVO_INSTANCES] = {nullptr};
uint8_t XY2_100::instance_count = 0;

// 构造函数：为这个新实例分配一个唯一的ID
XY2_100::XY2_100() {
    if (instance_count < MAX_GALVO_INSTANCES) {
        _instance_id = instance_count;
        instances[instance_count] = this;
        instance_count++;
        txPing = 0; // 初始化实例特定的状态
    }
}

void XY2_100::begin(uint8_t ftm_module, volatile uint32_t* gpio_port_register, 
                    uint8_t pin0, uint8_t pin1, uint8_t pin2, uint8_t pin3, 
                    uint8_t pin4, uint8_t pin5, uint8_t pin6, uint8_t pin7, 
                    uint32_t frequency) {
    // 存储此实例的配置
    _ftm_module = ftm_module;
    _gpio_port_register = gpio_port_register;
    _pins[0] = pin0; _pins[1] = pin1; _pins[2] = pin2; _pins[3] = pin3;
    _pins[4] = pin4; _pins[5] = pin5; _pins[6] = pin6; _pins[7] = pin7;
    _frequency = frequency;

    uint32_t bufsize = 40;

    // *** 更改：将实例的指针指向其在静态DMA内存池中的专属区域 ***
    pingBuffer = dma_ping_buffers[_instance_id];
    pongBuffer = dma_pong_buffers[_instance_id];
    memset(pingBuffer, 0, bufsize);
    memset(pongBuffer, 0, bufsize);

    // 配置此实例的输出引脚
    for (int i = 0; i < 8; i++) {
        pinMode(_pins[i], OUTPUT);
    }

    // 配置此实例的DMA通道
    dma.sourceBuffer((uint8_t *)pingBuffer, bufsize);
    dma.destination(*_gpio_port_register);
    dma.transferSize(1);
    dma.transferCount(bufsize);
    dma.disableOnCompletion();
    dma.interruptAtCompletion();

    // 根据实例ID附加正确的中断调度器
    if (_instance_id == 0) {
        dma.attachInterrupt(isr_dispatcher_0);
    } else if (_instance_id == 1) {
        dma.attachInterrupt(isr_dispatcher_1);
    }

    // 为此实例配置指定的FTM定时器
#if defined(__MK20DX256__) // 适用于 Teensy 3.1/3.2
    if (_ftm_module == 2) {
        FTM2_SC = 0;
        FTM2_CNT = 0;
        uint32_t mod = (F_BUS + _frequency / 2) / _frequency;
        FTM2_MOD = mod - 1;
        FTM2_SC = FTM_SC_CLKS(1) | FTM_SC_PS(0);
        FTM2_C0SC = 0x69;
        FTM2_C0V = (mod * 128) >> 8;
        dma.triggerAtHardwareEvent(DMAMUX_SOURCE_FTM2_CH0);
        
        noInterrupts();
        FTM2_C0SC = 0x28;
        volatile uint32_t tmp = FTM2_C0SC; (void)tmp; // 读取以清除标志位
        FTM2_C0SC = 0x69;
        dma.enable();
        FTM2_SC = FTM_SC_CLKS(1) | FTM_SC_PS(0);
        interrupts();

    } else if (_ftm_module == 1) {
        // FTM1 配置
        FTM1_SC = 0;
        FTM1_CNT = 0;
        uint32_t mod = (F_BUS + _frequency / 2) / _frequency;
        FTM1_MOD = mod - 1;
        FTM1_SC = FTM_SC_CLKS(1) | FTM_SC_PS(0);
        FTM1_C0SC = 0x69;
        FTM1_C0V = (mod * 128) >> 8;
        dma.triggerAtHardwareEvent(DMAMUX_SOURCE_FTM1_CH0); // 使用 FTM1 触发源
        
        noInterrupts();
        FTM1_C0SC = 0x28;
        volatile uint32_t tmp = FTM1_C0SC; (void)tmp; // 读取以清除标志位
        FTM1_C0SC = 0x69;
        dma.enable();
        FTM1_SC = FTM_SC_CLKS(1) | FTM_SC_PS(0);
        interrupts();
    }
#else
    #error "此重构代码目前仅支持 Teensy 3.1/3.2 的 FTM 配置。"
#endif
}

// 实例0的静态中断调度器
void XY2_100::isr_dispatcher_0(void) {
    if (instances[0]) {
        instances[0]->isr_handler();
    }
}

// 实例1的静态中断调度器
void XY2_100::isr_dispatcher_1(void) {
    if (instances[1]) {
        instances[1]->isr_handler();
    }
}

// 实例特定的中断服务程序（ISR）逻辑
void XY2_100::isr_handler(void) {
    dma.clearInterrupt();
    if (txPing & 2) {
        txPing &= ~2;
        if (txPing & 1) {
            dma.sourceBuffer((uint8_t *)pongBuffer, 40);
        } else {
            dma.sourceBuffer((uint8_t *)pingBuffer, 40);
        }
    }

#if defined(__MK20DX256__)
    // 为下一次触发重新准备定时器和DMA
    if (_ftm_module == 2) {
        FTM2_SC = 0;
        FTM2_SC = FTM_SC_TOF; // 清除溢出标志位
        volatile uint32_t tmp;
        FTM2_C0SC = 0x28;
        tmp = FTM2_C0SC; (void)tmp;
        FTM2_C0SC = 0x69;
        FTM2_CNT = 0;
        dma.enable();
        FTM2_SC = FTM_SC_CLKS(1) | FTM_SC_PS(0);
    } else if (_ftm_module == 1) {
        FTM1_SC = 0;
        FTM1_SC = FTM_SC_TOF; // 清除溢出标志位
        volatile uint32_t tmp;
        FTM1_C0SC = 0x28;
        tmp = FTM1_C0SC; (void)tmp;
        FTM1_C0SC = 0x69;
        FTM1_CNT = 0;
        dma.enable();
        FTM1_SC = FTM_SC_CLKS(1) | FTM_SC_PS(0);
    }
#endif
}

uint8_t XY2_100::stat(void) {
    uint8_t ret = txPing;
    txPing &= ~128;
    return ret;
}

void XY2_100::setSignedXY(int16_t X, int16_t Y) {
    int32_t xu = (int32_t)X + 32768L;
    int32_t yu = (int32_t)Y + 32768L;
    setXY((uint16_t)xu, (uint16_t)yu);
}

void XY2_100::setXY(uint16_t X, uint16_t Y) {
    uint32_t *p;
    uint32_t Ch1 = (((uint32_t)X << 1) | 0x20000ul) & 0x3fffeul;
    uint32_t Ch2 = (((uint32_t)Y << 1) | 0x20000ul) & 0x3fffeul;
    uint8_t parity1 = 0;
    uint8_t parity2 = 0;
    const uint16_t Sync1[4] = {0xd2c3, 0x9687, 0x5a4b, 0x1e0f};
    const uint16_t Sync0[4] = {0xf0e1, 0xb4a5, 0x7869, 0x3c2d};

    lastX = X;
    lastY = Y;

    for (int i = 0; i < 20; i++) {
        if (Ch1 & (1 << i)) parity1++;
        if (Ch2 & (1 << i)) parity2++;
    }
    if (parity1 & 1) Ch1 |= 1;
    if (parity2 & 1) Ch2 |= 1;

    if (txPing & 1) {
        p = ((uint32_t *)pingBuffer);
    } else {
        p = ((uint32_t *)pongBuffer);
    }

    for (int i = 19; i >= 0; i--) {
        int j = 0;
        uint32_t d;
        if (Ch1 & (1 << i)) j = 1;
        if (Ch2 & (1 << i)) j |= 2;
        d = Sync1[j];
        i--;
        j = 0;
        if (Ch1 & (1 << i)) j = 1;
        if (Ch2 & (1 << i)) j |= 2;
        if (i != 0) d |= (uint32_t)Sync1[j] << 16;
        else d |= (uint32_t)Sync0[j] << 16;
        *p++ = d;
    }

    noInterrupts();
    txPing ^= 1;
    txPing |= 2;
    interrupts();
}
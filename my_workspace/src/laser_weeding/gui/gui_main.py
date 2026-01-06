#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
激光除草系统 PyQt5 主界面
提供完整的图形化控制界面
"""

import sys
import os
import re
from pathlib import Path

# 在导入 PyQt5 之前，必须设置环境变量排除 OpenCV 的 Qt 插件
# 这是关键：必须在导入任何 PyQt5 模块之前完成

# 如果启动脚本已经设置了环境变量，就使用它
# 否则，尝试从环境变量推断路径
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    # 尝试从 Python 路径推断 PyQt5 位置（不导入 PyQt5）
    import site
    for site_packages in site.getsitepackages():
        pyqt5_plugins = Path(site_packages) / 'PyQt5' / 'Qt5' / 'plugins' / 'platforms'
        if pyqt5_plugins.exists():
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(pyqt5_plugins)
            os.environ['QT_PLUGIN_PATH'] = str(pyqt5_plugins.parent)
            break

# 现在可以安全地导入 PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStatusBar, QMessageBox, QSplitter, QTabWidget, QLabel, QPushButton,
    QDialog, QLineEdit, QTextEdit, QGridLayout, QScrollArea, QGroupBox
)
from PyQt5.QtCore import Qt, QSettings, pyqtSlot, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QKeyEvent

# 导入自定义组件
from gui_ros_interface import ROSInterfaceThread, LaunchManager
from gui_components import (
    ImageDisplayWidget, StatisticsWidget, LaunchSelectorWidget,
    ManualGalvoControlWidget
)


# 科技感深色主题样式
TECH_STYLE = """
/* 全局背景和字体 */
QWidget {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    font-size: 14px;
}

/* 按钮通用样式 */
QPushButton {
    background-color: #1f1f1f;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 8px 16px;
    color: #00bcd4; /* 青色文字作为强调 */
    font-weight: bold;
}
QPushButton:hover {
    background-color: #262626;
    border-color: #00bcd4; /* 悬停时边框变青 */
    color: #ffffff; /* 悬停时文字变白 */
}
QPushButton:pressed {
    background-color: #00bcd4;
    border-color: #00bcd4;
    color: #000000; /* 按下时背景变青，文字变黑 */
}
QPushButton:checked {
    background-color: #008ba3;
    color: #ffffff;
}
QPushButton:disabled {
    background-color: #1a1a1a;
    border-color: #2a2a2a;
    color: #555555;
}

/* 特殊按钮颜色：保存/确认 (绿色系) */
QPushButton[role="primary"] {
    color: #4caf50;
    border-color: #2e7d32;
}
QPushButton[role="primary"]:hover {
    border-color: #4caf50;
    background-color: #1b5e20;
    color: #ffffff;
}
QPushButton[role="primary"]:pressed {
    background-color: #4caf50;
}

/* 特殊按钮颜色：危险/停止 (红色系) */
QPushButton[role="danger"] {
    color: #f44336;
    border-color: #c62828;
}
QPushButton[role="danger"]:hover {
    border-color: #f44336;
    background-color: #b71c1c;
    color: #ffffff;
}
QPushButton[role="danger"]:pressed {
    background-color: #f44336;
}

/* 输入框、下拉框、微调框 */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
    background-color: #1f1f1f;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 5px;
    color: #ffffff;
    selection-background-color: #00bcd4;
    selection-color: #000000;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus {
    border: 1px solid #00bcd4;
    background-color: #262626;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox::down-arrow {
    image: none; /* 可以替换为自定义图标，此处简化 */
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #00bcd4;
    margin-top: 2px;
}

/* 分组框 */
QGroupBox {
    border: 1px solid #333333;
    border-radius: 6px;
    margin-top: 24px; /* 为标题留出空间 */
    font-weight: bold;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: #00bcd4;
}

/* 滚动条 */
QScrollBar:vertical {
    border: none;
    background: #121212;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #333333;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #00bcd4;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* 标签页 */
QTabWidget::pane {
    border: 1px solid #333333;
    background: #121212;
    top: -1px; 
}
QTabBar::tab {
    background: #1f1f1f;
    border: 1px solid #333333;
    padding: 8px 20px;
    margin-right: 2px;
    color: #888888;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #121212;
    border-bottom-color: #121212;
    color: #00bcd4;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background: #262626;
    color: #e0e0e0;
}

/* 状态栏 */
QStatusBar {
    background-color: #1f1f1f;
    border-top: 1px solid #333333;
    color: #888888;
}

/* 滑块 */
QSlider::groove:horizontal {
    border: 1px solid #333333;
    height: 4px;
    background: #1f1f1f;
    margin: 2px 0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #00bcd4;
    border: 1px solid #00bcd4;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
}
"""

class CalibrationDialog(QDialog):
    """系统标定控制对话框"""
    def __init__(self, parent=None, ros_thread=None):
        super().__init__(parent)
        self.ros_thread = ros_thread
        self.setWindowTitle("🎯 3D 系统标定 - 手动瞄准")
        self.setMinimumSize(1000, 700)
        self.init_ui()
        self.apply_dialog_style()
        # 应用统一样式
        self.setStyleSheet(TECH_STYLE)
        
        # 连接 ROS 信号
        if self.ros_thread:
            self.ros_thread.calib_image_received.connect(self.image_display.update_image)
            self.ros_thread.calib_status_updated.connect(self.update_status)
        
        # 设置对话框可以接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)

    def init_ui(self):
        main_layout = QHBoxLayout()
        
        # 左侧控制按钮
        ctrl_layout = QVBoxLayout()
        
        # 标定流程组
        flow_group = QGroupBox("标定流程")
        flow_layout = QGridLayout()
        btns = [
            ("开始标定 (B)", 'b'), ("记录点 (R)", 'record'),
            ("上一个 (P)", 'p'), ("下一个 (N)", 'n'),
            ("完成批次 (M)", 'm'), ("保存结果 (C)", 'save'),
            ("重置 (X)", 'reset'), ("停止 (K)", 'k')
        ]
        for i, (text, cmd) in enumerate(btns):
            btn = QPushButton(text)
            btn.clicked.connect(lambda ch, c=cmd: self.send_cmd(c))
            flow_layout.addWidget(btn, i//2, i%2)
        flow_group.setLayout(flow_layout)
        ctrl_layout.addWidget(flow_group)
        
        # 移动控制组
        move_group = QGroupBox("振镜移动")
        move_layout = QGridLayout()
        moves = [('W', 'w', 0, 1), ('A', 'a', 1, 0), ('S', 's', 1, 1), ('D', 'd', 1, 2)]
        for text, cmd, r, c in moves:
            btn = QPushButton(text)
            btn.clicked.connect(lambda ch, c=cmd: self.send_cmd(c))
            move_layout.addWidget(btn, r, c)
        
        btn_center = QPushButton("自动对准 (Space)")
        btn_center.clicked.connect(lambda: self.send_cmd(' '))
        move_layout.addWidget(btn_center, 2, 0, 1, 3)
        move_group.setLayout(move_layout)
        ctrl_layout.addWidget(move_group)
        
        # 步进与模式
        mode_group = QGroupBox("设置")
        mode_layout = QHBoxLayout()
        btn_fine = QPushButton("精细模式 (F)")
        btn_fine.setCheckable(True)
        btn_fine.clicked.connect(lambda: self.send_cmd('f'))
        mode_layout.addWidget(btn_fine)
        btn_laser = QPushButton("激光开关 (L)")
        btn_laser.setProperty("role", "danger")  # 使用 role 属性来应用红色样式
        # btn_laser.setStyleSheet("background-color: #f44336;") # 移除内联样式
        btn_laser.clicked.connect(lambda: self.send_cmd('l'))
        mode_layout.addWidget(btn_laser)
        mode_group.setLayout(mode_layout)
        ctrl_layout.addWidget(mode_group)
        
        # 状态显示
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        ctrl_layout.addWidget(QLabel("状态信息:"))
        ctrl_layout.addWidget(self.status_text)
        
        ctrl_layout.addStretch()
        main_layout.addLayout(ctrl_layout, 1)
        
        # 右侧图像显示
        self.image_display = ImageDisplayWidget()
        main_layout.addWidget(self.image_display, 2)
        
        self.setLayout(main_layout)

    def send_cmd(self, cmd):
        if self.ros_thread:
            self.ros_thread.send_calib_command(cmd)
    
    def keyPressEvent(self, event: QKeyEvent):
        """处理键盘按键事件，支持 WASD 控制振镜"""
        key = event.key()
        modifiers = event.modifiers()
        
        # 只处理没有修饰键（Ctrl/Alt/Shift）的按键
        if modifiers == Qt.NoModifier:
            # 将按键转换为小写字符
            if key == Qt.Key_W:
                self.send_cmd('w')
                event.accept()
                return
            elif key == Qt.Key_A:
                self.send_cmd('a')
                event.accept()
                return
            elif key == Qt.Key_S:
                self.send_cmd('s')
                event.accept()
                return
            elif key == Qt.Key_D:
                self.send_cmd('d')
                event.accept()
                return
            elif key == Qt.Key_Space:
                self.send_cmd(' ')
                event.accept()
                return
            elif key == Qt.Key_F:
                self.send_cmd('f')
                event.accept()
                return
            elif key == Qt.Key_L:
                self.send_cmd('l')
                event.accept()
                return
            elif key == Qt.Key_R:
                self.send_cmd('r')  # 记录点
                event.accept()
                return
            elif key == Qt.Key_B:
                self.send_cmd('b')  # 开始标定
                event.accept()
                return
            elif key == Qt.Key_N:
                self.send_cmd('n')  # 下一个
                event.accept()
                return
            elif key == Qt.Key_P:
                self.send_cmd('p')  # 上一个
                event.accept()
                return
            elif key == Qt.Key_M:
                self.send_cmd('m')  # 完成批次
                event.accept()
                return
            elif key == Qt.Key_C:
                self.send_cmd('c')  # 保存结果
                event.accept()
                return
            elif key == Qt.Key_X:
                self.send_cmd('x')  # 重置
                event.accept()
                return
            elif key == Qt.Key_K:
                self.send_cmd('k')  # 停止
                event.accept()
                return
        
        # 其他按键交给父类处理
        super().keyPressEvent(event)

    def update_status(self, data):
        info = f"状态: {data.get('state')}\n"
        info += f"当前目标: {data.get('current_target_index', 0)+1}/{data.get('total_targets', 0)}\n"
        info += f"累积点数: {data.get('accumulated_points', 0)}\n"
        info += f"激光状态: {'ON' if data.get('laser_on') else 'OFF'}"
        self.status_text.setText(info)
    
    def closeEvent(self, event):
        """对话框关闭事件 - 询问是否停止 calibration launch"""
        # 获取主窗口（MainWindow）
        main_window = self.parent()
        if main_window and hasattr(main_window, 'calibration_launch_process'):
            if main_window.calibration_launch_process is not None:
                if main_window.calibration_launch_process.poll() is None:
                    # calibration launch 正在运行
                    reply = QMessageBox.question(
                        self,
                        "关闭标定系统",
                        "标定系统正在运行（包括 RealSense 相机），是否停止？\n\n"
                        "如果选择否，相机将继续运行。",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes  # 默认选择"是"，确保相机被关闭
                    )
                    if reply == QMessageBox.Yes:
                        # 停止 calibration launch
                        process_storage = {'calibration': main_window.calibration_launch_process}
                        if hasattr(main_window, 'launch_manager'):
                            success, message = main_window.launch_manager.stop_launch_by_name(
                                'calibration', 
                                process_storage
                            )
                            if success:
                                main_window.calibration_launch_process = None
                                if hasattr(main_window, 'status_bar'):
                                    main_window.status_bar.showMessage("已停止标定系统", 2000)
                                print(f"DEBUG: {message}")
                            else:
                                QMessageBox.warning(self, "停止失败", message)
                                # 即使停止失败，也允许关闭对话框
                        else:
                            # 如果没有 launch_manager，直接终止进程
                            try:
                                import signal
                                pgid = os.getpgid(main_window.calibration_launch_process.pid)
                                os.killpg(pgid, signal.SIGTERM)
                                main_window.calibration_launch_process.wait(timeout=5)
                                main_window.calibration_launch_process = None
                            except Exception as e:
                                print(f"DEBUG: 停止 calibration launch 时出错: {e}")
        
        # 允许关闭对话框
        event.accept()
    
    def apply_dialog_style(self):
        """应用对话框统一样式"""
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0d0d0d);
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1a1a1a);
                color: #00D4FF;
                border: 1px solid #00D4FF;
                padding: 8px;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                border: 1px solid #00FFFF;
                color: #00FFFF;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0a0a0a);
            }
            QGroupBox {
                border: 2px solid #00D4FF;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #00FFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #1a1a1a;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #00D4FF;
                border-radius: 4px;
                padding: 5px;
            }
        """)

class SettingsDialog(QDialog):
    """参数设置对话框 - 支持修改 Launch 和 YAML 配置文件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ 系统参数设置")
        self.setMinimumSize(800, 700)
        # 应用统一样式
        self.setStyleSheet(TECH_STYLE)
        
        # 获取文件路径
        gui_dir = Path(__file__).resolve().parent
        project_dir = gui_dir.parent
        self.launch_path = project_dir / 'launch' / 'main.launch'
        self.params_path = project_dir / 'cam_params.yaml'
        
        # 存储原始参数值
        self.launch_params = {}
        self.yaml_params = {}
        
        self.init_ui()
        self.load_parameters()

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # 创建标签页
        self.tabs = QTabWidget()
        
        # Launch 参数标签页
        self.launch_tab = self.create_launch_tab()
        self.tabs.addTab(self.launch_tab, "Launch 参数")
        
        # YAML 参数标签页
        self.yaml_tab = self.create_yaml_tab()
        self.tabs.addTab(self.yaml_tab, "YAML 参数")
        
        main_layout.addWidget(self.tabs)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("保存")
        btn_save.setProperty("role", "primary") # 应用绿色样式
        # btn_save.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;")
        btn_save.clicked.connect(self.save_all_parameters)
        btn_layout.addWidget(btn_save)
        
        btn_reset = QPushButton("重置")
        # btn_reset.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        btn_reset.clicked.connect(self.reset_parameters)
        btn_layout.addWidget(btn_reset)
        
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)
        
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)
        self.apply_dialog_style()
    
    def apply_dialog_style(self):
        """应用对话框统一样式"""
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0d0d0d);
            }
            QLabel {
                color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 2px solid #00D4FF;
                border-radius: 8px;
                background: #1a1a1a;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1a1a1a);
                color: #00D4FF;
                border: 1px solid #00D4FF;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00D4FF, stop:1 #0088CC);
                color: #000;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                color: #00FFFF;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1a1a1a);
                color: #00D4FF;
                border: 1px solid #00D4FF;
                padding: 10px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                border: 1px solid #00FFFF;
                color: #00FFFF;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0a0a0a);
            }
            QLineEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #00D4FF;
                border-radius: 4px;
                padding: 6px;
            }
            QLineEdit:focus {
                border: 2px solid #00FFFF;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #00D4FF;
                border-radius: 4px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

    def create_launch_tab(self):
        """创建 Launch 参数编辑标签页"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 说明
        info_label = QLabel("修改 Launch 文件中的关键参数（main.launch）")
        info_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        layout.addWidget(info_label)
        
        # 滚动区域
        scroll = QWidget()
        scroll_layout = QGridLayout()
        scroll_layout.setSpacing(10)
        
        # Launch 参数字段（关键参数）
        self.launch_widgets = {}
        
        # 模型参数
        scroll_layout.addWidget(QLabel("<b>模型参数</b>"), 0, 0, 1, 2)
        self.add_launch_param(scroll_layout, 1, "model_type", "模型类型", "yolov11")
        self.add_launch_param(scroll_layout, 2, "model_path", "模型路径", "")
        self.add_launch_param(scroll_layout, 3, "device", "设备 (0/1/cpu)", "0")
        self.add_launch_param(scroll_layout, 4, "confidence_threshold", "置信度阈值", "0.3")
        
        # 预测参数
        scroll_layout.addWidget(QLabel("<b>预测参数</b>"), 5, 0, 1, 2)
        self.add_launch_param(scroll_layout, 6, "total_delay", "总延迟 (秒)", "0.08")
        self.add_launch_param(scroll_layout, 7, "prediction_time", "预测时间 (秒)", "0.5")
        self.add_launch_param(scroll_layout, 8, "use_kalman", "使用卡尔曼滤波", "true")
        
        # 激光控制
        scroll_layout.addWidget(QLabel("<b>激光控制</b>"), 9, 0, 1, 2)
        self.add_launch_param(scroll_layout, 10, "aiming_time", "瞄准时间 (秒)", "0.15")
        self.add_launch_param(scroll_layout, 11, "laser_time", "照射时间 (秒)", "0.4")
        self.add_launch_param(scroll_layout, 12, "laser_mode", "激光模式 (point/spiral)", "point")
        
        # 振镜配置
        scroll_layout.addWidget(QLabel("<b>振镜配置</b>"), 13, 0, 1, 2)
        self.add_launch_param(scroll_layout, 14, "galvo_count", "振镜数量", "2")
        self.add_launch_param(scroll_layout, 15, "galvo_split_axis", "分割方向 (vertical/horizontal)", "vertical")
        self.add_launch_param(scroll_layout, 16, "galvo_split_ratio", "分割比例 (0.0-1.0)", "0.5")
        self.add_launch_param(scroll_layout, 17, "galvo_overlap_px", "重叠像素数", "120")
        
        # 目标管理
        scroll_layout.addWidget(QLabel("<b>目标管理</b>"), 18, 0, 1, 2)
        self.add_launch_param(scroll_layout, 19, "target_timeout", "目标超时 (秒)", "0.3")
        self.add_launch_param(scroll_layout, 20, "min_stable_frames", "最小稳定帧数", "1")
        
        scroll.setLayout(scroll_layout)
        
        # 使用 QScrollArea
        scroll_area_widget = QScrollArea()
        scroll_area_widget.setWidget(scroll)
        scroll_area_widget.setWidgetResizable(True)
        
        layout.addWidget(scroll_area_widget)
        tab.setLayout(layout)
        return tab

    def add_launch_param(self, layout, row, param_name, label_text, default_value):
        """添加 Launch 参数输入控件"""
        layout.addWidget(QLabel(label_text + ":"), row, 0)
        widget = QLineEdit()
        widget.setPlaceholderText(default_value)
        self.launch_widgets[param_name] = widget
        layout.addWidget(widget, row, 1)

    def create_yaml_tab(self):
        """创建 YAML 参数编辑标签页"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 说明
        info_label = QLabel("修改 YAML 配置文件中的关键参数（cam_params.yaml）")
        info_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        layout.addWidget(info_label)
        
        # 使用文本编辑器（YAML 结构复杂，直接编辑更灵活）
        self.yaml_editor = QTextEdit()
        self.yaml_editor.setFont(QFont('Courier', 9))
        # 移除内联样式，使用全局样式
        # self.yaml_editor.setStyleSheet("""...""")
        layout.addWidget(self.yaml_editor)
        
        tab.setLayout(layout)
        return tab

    def load_parameters(self):
        """加载参数文件"""
        # 加载 Launch 文件
        if self.launch_path.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(self.launch_path)
                root = tree.getroot()
                
                # 查找所有 param 节点
                for node in root.findall('.//node[@name="laser_weeding_node"]'):
                    for param in node.findall('param'):
                        param_name = param.get('name')
                        param_value = param.get('value')
                        if param_name and param_value:
                            self.launch_params[param_name] = param_value
                            if param_name in self.launch_widgets:
                                self.launch_widgets[param_name].setText(param_value)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载 Launch 文件失败: {e}")
        
        # 加载 YAML 文件
        if self.params_path.exists():
            try:
                import yaml
                with open(self.params_path, 'r', encoding='utf-8') as f:
                    self.yaml_params = yaml.safe_load(f) or {}
                    # 将 YAML 内容显示在编辑器中
                    with open(self.params_path, 'r', encoding='utf-8') as f2:
                        self.yaml_editor.setText(f2.read())
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载 YAML 文件失败: {e}")

    def save_all_parameters(self):
        """保存所有参数"""
        try:
            # 保存 Launch 参数
            if self.launch_path.exists():
                import xml.etree.ElementTree as ET
                tree = ET.parse(self.launch_path)
                root = tree.getroot()
                
                # 更新参数值
                updated_count = 0
                for node in root.findall('.//node[@name="laser_weeding_node"]'):
                    for param in node.findall('param'):
                        param_name = param.get('name')
                        if param_name in self.launch_widgets:
                            new_value = self.launch_widgets[param_name].text().strip()
                            if new_value:
                                param.set('value', new_value)
                                updated_count += 1
                
                # 保存文件（使用自定义格式化）
                import xml.dom.minidom
                xml_str = ET.tostring(root, encoding='unicode')
                dom = xml.dom.minidom.parseString(xml_str)
                pretty_xml = dom.toprettyxml(indent='    ', encoding='utf-8')
                
                # 读取原始文件以保留注释和格式
                with open(self.launch_path, 'r', encoding='utf-8') as f:
                    original_lines = f.readlines()
                
                # 简单的行替换方式（保留格式）
                with open(self.launch_path, 'w', encoding='utf-8') as f:
                    for line in original_lines:
                        # 检查是否需要替换
                        replaced = False
                        for param_name, widget in self.launch_widgets.items():
                            new_value = widget.text().strip()
                            if new_value and f'name="{param_name}"' in line and 'value=' in line:
                                # 替换 value 属性
                                pattern = rf'name="{param_name}"\s+value="[^"]*"'
                                replacement = f'name="{param_name}" value="{new_value}"'
                                line = re.sub(pattern, replacement, line)
                                replaced = True
                                break
                        f.write(line)
            
            # 保存 YAML 参数
            if self.params_path.exists():
                yaml_content = self.yaml_editor.toPlainText()
                with open(self.params_path, 'w', encoding='utf-8') as f:
                    f.write(yaml_content)
            
            QMessageBox.information(
                self, 
                "成功", 
                "参数已保存！\n\n请重新启动 Launch 文件以使新参数生效。"
            )
            self.accept()
            
        except Exception as e:
            import traceback
            error_msg = f"保存失败: {e}\n\n{type(e).__name__}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "错误", error_msg)

    def reset_parameters(self):
        """重置参数到文件中的值"""
        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要重置所有参数到文件中的原始值吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.load_parameters()

class ManualControlDialog(QDialog):
    """手动控制对话框"""
    
    # 定义信号，转发 ManualGalvoControlWidget 的信号
    galvo_command = pyqtSignal(int, int, int)  # galvo_index, x, y
    laser_control = pyqtSignal(int, bool)  # galvo_index, enabled
    
    def __init__(self, parent=None, ros_thread=None):
        super().__init__(parent)
        self.ros_thread = ros_thread
        self.setWindowTitle("🎮 手动控制 - 振镜和激光")
        self.setMinimumSize(600, 500)
        self.init_ui()
        self.apply_dialog_style()
        # 应用统一样式
        self.setStyleSheet(TECH_STYLE)
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title = QLabel("手动振镜和激光控制")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # 说明文字
        info_label = QLabel("注意：当有 launch 进程运行时，手动控制将被禁用")
        info_label.setStyleSheet("color: #FF9800; font-size: 10px;") # 保留特定颜色，或考虑统一
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # 手动振镜控制组件
        self.galvo_control = ManualGalvoControlWidget(galvo_count=2)
        layout.addWidget(self.galvo_control)
        
        # 连接信号（转发到主窗口）
        self.galvo_control.galvo_command.connect(self.galvo_command.emit)
        self.galvo_control.laser_control.connect(self.laser_control.emit)
        
        # 关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        # 移除内联样式
        layout.addWidget(close_button)
        
        self.setLayout(layout)
        
        # 移除底部的内联样式设置，因为已经在 __init__ 中设置了全局样式
    
    def set_enabled(self, enabled):
        """设置控制是否启用"""
        if hasattr(self, 'galvo_control'):
            self.galvo_control.set_enabled(enabled)
    
    def apply_dialog_style(self):
        """应用对话框统一样式"""
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0d0d0d);
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #1a1a1a);
                color: #00D4FF;
                border: 1px solid #00D4FF;
                padding: 10px;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                border: 1px solid #00FFFF;
                color: #00FFFF;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0a0a0a);
            }
            QGroupBox {
                border: 2px solid #00D4FF;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #00FFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #1a1a1a;
            }
        """)


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 注意：高 DPI 支持应该在创建 QApplication 之前设置（在 main() 函数中）
        
        # 加载设置
        self.settings = QSettings('LaserWeeding', 'MainWindow')
        
        # 初始化 ROS 接口
        self.init_ros_interface()
        
        # 初始化 Launch 管理器
        self.init_launch_manager()
        
        # 初始化手动控制对话框（延迟创建，首次打开时创建）
        self.manual_control_dialog = None
        
        # 标定 launch 进程存储（用于跟踪 calibration.launch）
        self.calibration_launch_process = None
        
        # 初始化 UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
        
        # 恢复窗口状态
        self.restore_window_state()
        
        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)  # 100ms 更新一次
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("激光除草系统控制界面")
        self.setMinimumSize(1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        central_widget.setLayout(main_layout)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧面板（控制面板）
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧面板（图像显示）
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割器比例
        splitter.setSizes([400, 800])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        # 应用样式
        self.apply_styles()
        
    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Launch 选择器
        self.launch_selector = LaunchSelectorWidget(self.launch_manager)
        layout.addWidget(self.launch_selector)
        
        # 统计信息（支持双振镜分别统计）
        self.statistics_widget = StatisticsWidget(galvo_count=2)
        layout.addWidget(self.statistics_widget)
        
        # 手动控制按钮
        self.manual_control_button = QPushButton("🎮 手动控制")
        self.manual_control_button.setToolTip("打开手动振镜和激光控制窗口")
        self.manual_control_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #42A5F5, stop:1 #2196F3);
                box-shadow: 0 4px 8px rgba(33, 150, 243, 0.5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1976D2, stop:1 #0D47A1);
            }
            QPushButton:disabled {
                background: #555;
                color: #888;
            }
        """)
        layout.addWidget(self.manual_control_button)
        
        # 标定系统按钮
        self.calibration_button = QPushButton("📐 系统标定")
        self.calibration_button.setToolTip("打开 3D 标定控制界面")
        self.calibration_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FF9800, stop:1 #F57C00);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FFB74D, stop:1 #FF9800);
                box-shadow: 0 4px 8px rgba(255, 152, 0, 0.5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #F57C00, stop:1 #E65100);
            }
        """)
        layout.addWidget(self.calibration_button)
        
        # 参数设置按钮
        self.settings_button = QPushButton("⚙️ 参数设置")
        self.settings_button.setToolTip("修改系统运行参数 (YAML/Launch)")
        self.settings_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #607D8B, stop:1 #455A64);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #78909C, stop:1 #607D8B);
                box-shadow: 0 4px 8px rgba(96, 125, 139, 0.5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #455A64, stop:1 #263238);
            }
        """)
        layout.addWidget(self.settings_button)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_right_panel(self):
        """创建右侧图像显示面板（包含原始图像和检测结果）"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 使用分割器上下排列两个图像
        img_splitter = QSplitter(Qt.Vertical)
        
        # 原始图像显示
        self.raw_image_display = ImageDisplayWidget()
        self.raw_image_display.setObjectName("raw_image")
        # 修改内部标题
        for child in self.raw_image_display.children():
            if isinstance(child, QLabel) and child.text() == "检测结果":
                child.setText("原始图像 (RealSense)")
        
        # 检测结果显示
        self.image_display = ImageDisplayWidget()
        self.image_display.setObjectName("det_image")
        
        img_splitter.addWidget(self.raw_image_display)
        img_splitter.addWidget(self.image_display)
        
        # 设置初始大小比例
        img_splitter.setSizes([400, 400])
        
        layout.addWidget(img_splitter)
        panel.setLayout(layout)
        return panel
    
    def init_ros_interface(self):
        """初始化 ROS 接口"""
        self.ros_thread = ROSInterfaceThread()
        # 注意：不要在这里启动线程，等窗口显示后再启动
    
    def init_launch_manager(self):
        """初始化 Launch 管理器"""
        # 获取 launch 目录路径（gui/ 的父目录下的 launch/）
        # 使用 __file__ 的绝对路径来确保正确解析
        gui_file = Path(__file__).resolve()  # 获取 gui_main.py 的绝对路径
        gui_dir = gui_file.parent  # gui/ 目录
        launch_dir = gui_dir.parent / 'launch'  # 父目录下的 launch/
        launch_dir_str = str(launch_dir)
        print(f"DEBUG: GUI 文件: {gui_file}")
        print(f"DEBUG: GUI 目录: {gui_dir}")
        print(f"DEBUG: Launch 目录: {launch_dir_str}")
        print(f"DEBUG: Launch 目录是否存在: {os.path.exists(launch_dir_str)}")
        if not os.path.exists(launch_dir_str):
            # 如果路径不对，尝试直接使用项目根目录
            project_root = gui_dir.parent
            alt_launch_dir = project_root / 'launch'
            if os.path.exists(str(alt_launch_dir)):
                launch_dir_str = str(alt_launch_dir)
                print(f"DEBUG: 使用备用路径: {launch_dir_str}")
        
        # 创建 Launch 管理器
        self.launch_manager = LaunchManager(launch_dir_str)
    
    def connect_signals(self):
        """连接信号和槽"""
        # ROS 接口信号
        self.ros_thread.image_received.connect(self.image_display.update_image)
        self.ros_thread.raw_image_received.connect(self.raw_image_display.update_image)
        self.ros_thread.statistics_updated.connect(self.statistics_widget.update_statistics)
        self.ros_thread.status_message.connect(self.on_status_message)
        
        # Launch 选择器信号
        self.launch_selector.start_requested.connect(self.on_launch_start)
        self.launch_selector.stop_requested.connect(self.on_launch_stop)
        
        # 按钮信号
        self.manual_control_button.clicked.connect(self.open_manual_control)
        self.calibration_button.clicked.connect(self.open_calibration_ui)
        self.settings_button.clicked.connect(self.open_settings_dialog)
    
    def apply_styles(self):
        """应用样式表"""
        # 应用定义的全局科技感样式
        self.setStyleSheet(TECH_STYLE)
    
    @pyqtSlot(str, str)
    def on_status_message(self, message, level):
        """处理状态消息"""
        if level == "error":
            self.status_bar.showMessage(f"错误: {message}", 5000)
            QMessageBox.critical(self, "错误", message)
        elif level == "warning":
            self.status_bar.showMessage(f"警告: {message}", 3000)
        else:
            self.status_bar.showMessage(message, 2000)
    
    @pyqtSlot(str)
    def on_launch_start(self, launch_name):
        """启动 Launch 文件"""
        print(f"DEBUG: 收到启动请求: {launch_name}")  # 调试信息
        try:
            if self.launch_manager.is_running():
                QMessageBox.warning(self, "警告", "已有 Launch 进程在运行，请先停止")
                return
            
            print(f"DEBUG: 开始启动 launch: {launch_name}")  # 调试信息
            success, message = self.launch_manager.start_launch(launch_name)
            print(f"DEBUG: 启动结果 - success: {success}, message: {message}")  # 调试信息
            if success:
                self.launch_selector.set_running(True)
                self.manual_control_button.setEnabled(False)  # 禁用手动控制按钮
                if self.manual_control_dialog:
                    self.manual_control_dialog.set_enabled(False)
                self.status_bar.showMessage(f"启动成功: {launch_name}", 3000)
            else:
                QMessageBox.critical(self, "启动失败", message)
        except Exception as e:
            print(f"DEBUG: 启动异常: {e}")  # 调试信息
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "启动异常", f"启动过程中发生错误: {str(e)}")
    
    @pyqtSlot()
    def on_launch_stop(self):
        """停止 Launch 文件"""
        if not self.launch_manager.is_running():
            QMessageBox.warning(self, "警告", "没有运行中的 Launch 进程")
            return
        
        success, message = self.launch_manager.stop_launch()
        if success:
            self.launch_selector.set_running(False)
            self.manual_control_button.setEnabled(True)  # 启用手动控制按钮
            if self.manual_control_dialog:
                self.manual_control_dialog.set_enabled(True)
            self.status_bar.showMessage("已停止 Launch 进程", 3000)
        else:
            QMessageBox.critical(self, "停止失败", message)
    
    @pyqtSlot(int, int, int)
    def on_galvo_command(self, galvo_index, x, y):
        """处理振镜控制命令"""
        if self.launch_manager.is_running():
            # 如果 launch 在运行，不允许手动控制
            return
        
        if self.ros_thread.ros_initialized:
            self.ros_thread.publish_galvo_command(x, y, galvo_index)
    
    @pyqtSlot(int, bool)
    def on_laser_control(self, galvo_index, enabled):
        """处理激光控制命令"""
        if self.launch_manager.is_running():
            # 如果 launch 在运行，不允许手动控制
            return
        
        if self.ros_thread.ros_initialized:
            self.ros_thread.publish_laser_control(enabled, galvo_index)
    
    def open_manual_control(self):
        """打开手动控制窗口"""
        if self.manual_control_dialog is None:
            self.manual_control_dialog = ManualControlDialog(self, self.ros_thread)
            # 连接信号
            self.manual_control_dialog.galvo_command.connect(self.on_galvo_command)
            self.manual_control_dialog.laser_control.connect(self.on_laser_control)
            # 根据 launch 状态设置启用状态
            is_running = self.launch_manager.is_running()
            self.manual_control_dialog.set_enabled(not is_running)
        
        # 显示窗口
        self.manual_control_dialog.show()
        self.manual_control_dialog.raise_()
        self.manual_control_dialog.activateWindow()

    def open_calibration_ui(self):
        """打开标定控制窗口，并自动启动 calibration.launch"""
        # 检查 calibration.launch 是否已经在运行
        if self.calibration_launch_process is not None:
            if self.calibration_launch_process.poll() is None:
                # 进程正在运行，直接打开对话框
                print("DEBUG: calibration.launch 已在运行中")
            else:
                # 进程已结束，清除引用
                self.calibration_launch_process = None
        
        # 如果 calibration.launch 未运行，则启动它
        if self.calibration_launch_process is None:
            print("DEBUG: 启动 calibration.launch...")
            # 使用字典来存储进程（LaunchManager 需要）
            process_storage = {'calibration': None}
            success, message, process = self.launch_manager.start_launch_by_name(
                'calibration', 
                process_storage
            )
            
            if success:
                self.calibration_launch_process = process_storage['calibration']
                self.status_bar.showMessage(f"已启动标定系统: {message}", 3000)
                print(f"DEBUG: {message}")
            else:
                QMessageBox.warning(
                    self,
                    "启动失败",
                    f"无法启动 calibration.launch:\n{message}\n\n请手动在终端运行:\nroslaunch laser_weeding calibration.launch"
                )
                return  # 启动失败，不打开对话框
        
        # 打开标定控制对话框
        self.calib_dialog = CalibrationDialog(self, self.ros_thread)
        self.calib_dialog.show()

    def open_settings_dialog(self):
        """打开参数设置窗口"""
        self.settings_dialog = SettingsDialog(self)
        self.settings_dialog.show()
    
    def update_status(self):
        """更新状态（定时器回调）"""
        # 检查 launch 状态
        is_running = self.launch_manager.is_running()
        if is_running != self.launch_selector.stop_button.isEnabled():
            self.launch_selector.set_running(is_running)
            self.manual_control_button.setEnabled(not is_running)
            if self.manual_control_dialog:
                self.manual_control_dialog.set_enabled(not is_running)
    
    def restore_window_state(self):
        """恢复窗口状态"""
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        else:
            # 默认大小和位置
            self.resize(1400, 900)
            self.move(100, 100)
        
        window_state = self.settings.value('windowState')
        if window_state:
            self.restoreState(window_state)
    
    def save_window_state(self):
        """保存窗口状态"""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 保存窗口状态
        self.save_window_state()
        
        # 停止 Launch 进程
        if self.launch_manager.is_running():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "有 Launch 进程正在运行，是否停止并退出？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.launch_manager.stop_launch()
            else:
                event.ignore()
                return
        
        # 停止 calibration launch 进程
        if self.calibration_launch_process is not None:
            if self.calibration_launch_process.poll() is None:
                reply = QMessageBox.question(
                    self,
                    "确认退出",
                    "标定系统正在运行，是否停止并退出？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    process_storage = {'calibration': self.calibration_launch_process}
                    success, message = self.launch_manager.stop_launch_by_name('calibration', process_storage)
                    if success:
                        self.calibration_launch_process = None
                        self.status_bar.showMessage(message, 2000)
                    else:
                        QMessageBox.warning(self, "停止失败", message)
                else:
                    event.ignore()
                    return
        
        # 停止 ROS 线程
        if self.ros_thread.isRunning():
            self.ros_thread.stop()
            self.ros_thread.wait(3000)  # 等待最多 3 秒
        
        # 关闭手动控制窗口
        if self.manual_control_dialog:
            self.manual_control_dialog.close()
        
        # 关闭标定对话框
        if hasattr(self, 'calib_dialog') and self.calib_dialog:
            self.calib_dialog.close()
        
        event.accept()
    
    def showEvent(self, event):
        """窗口显示事件"""
        super().showEvent(event)
        # 窗口显示后启动 ROS 线程
        if not self.ros_thread.isRunning():
            self.ros_thread.start()


def main():
    """主函数"""
    # 设置高 DPI 支持（必须在创建 QApplication 之前）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("激光除草系统")
    app.setOrganizationName("LaserWeeding")
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


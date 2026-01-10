#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GUI 组件模块
包含各种可复用的 UI 组件
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QDoubleSpinBox, QGroupBox, QComboBox, QProgressBar,
    QFrame, QGridLayout, QSlider, QLCDNumber, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import cv2
import numpy as np


class ImageDisplayWidget(QWidget):
    """实时图像显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.current_image = None
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题
        title = QLabel("检测结果")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # 图像显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 2px solid #3a3a3a;
                border-radius: 5px;
            }
        """)
        self.image_label.setText("等待图像...")
        layout.addWidget(self.image_label)
        
        self.setLayout(layout)
    
    def update_image(self, cv_image):
        """更新显示的图像"""
        if cv_image is None or cv_image.size == 0:
            print("DEBUG: 收到空图像，跳过更新")
            return
        
        # print(f"DEBUG: 更新图像，尺寸: {cv_image.shape}")  # 可以取消注释来查看图像信息
        self.current_image = cv_image.copy()
        
        # 转换为 RGB
        if len(cv_image.shape) == 3:
            if cv_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv_image
        else:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        
        # 调整大小以适应显示区域
        h, w = rgb_image.shape[:2]
        label_size = self.image_label.size()
        
        # 计算缩放比例
        scale_w = label_size.width() / w
        scale_h = label_size.height() / h
        scale = min(scale_w, scale_h, 1.0)  # 不放大，只缩小
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为 QPixmap
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.image_label.setPixmap(pixmap)


class StatisticsWidget(QWidget):
    """统计信息显示组件（支持双振镜分别统计）"""
    
    def __init__(self, galvo_count=2, parent=None):
        super().__init__(parent)
        self.galvo_count = galvo_count
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title = QLabel("系统统计")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # 为每个振镜创建统计组
        for galvo_idx in range(self.galvo_count):
            galvo_group = self.create_galvo_statistics_group(galvo_idx)
            layout.addWidget(galvo_group)
        
        # 全局统计（FPS 和系统状态）
        global_stats_layout = QGridLayout()
        global_stats_layout.setSpacing(10)
        
        # FPS
        global_stats_layout.addWidget(QLabel("当前 FPS:"), 0, 0)
        self.fps_label = QLabel("0.0")
        self.fps_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF9800;")
        global_stats_layout.addWidget(self.fps_label, 0, 1)
        
        # 系统状态
        global_stats_layout.addWidget(QLabel("系统状态:"), 1, 0)
        self.status_label = QLabel("IDLE")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #9E9E9E;")
        global_stats_layout.addWidget(self.status_label, 1, 1)
        
        global_stats_widget = QWidget()
        global_stats_widget.setLayout(global_stats_layout)
        layout.addWidget(global_stats_widget)
        
        layout.addStretch()
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                margin-top: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
    
    def create_galvo_statistics_group(self, galvo_idx):
        """为单个振镜创建统计组"""
        group = QGroupBox(f"振镜 {galvo_idx + 1}")
        layout = QGridLayout()
        layout.setSpacing(10)
        
        # 检测数量
        layout.addWidget(QLabel("检测杂草数:"), 0, 0)
        detected_label = QLabel("0")
        detected_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(detected_label, 0, 1)
        
        # 作业数量
        layout.addWidget(QLabel("作业杂草数:"), 1, 0)
        processed_label = QLabel("0")
        processed_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3;")
        layout.addWidget(processed_label, 1, 1)
        
        group.setLayout(layout)
        
        # 存储标签引用
        if not hasattr(self, 'galvo_labels'):
            self.galvo_labels = []
        self.galvo_labels.append({
            'detected': detected_label,
            'processed': processed_label
        })
        
        return group
    
    def update_statistics(self, stats):
        """更新统计信息 - 按振镜区域分别统计"""
        # 为每个振镜分别获取统计数据
        for galvo_idx in range(self.galvo_count):
            detected_count = 0
            processed_count = 0
            
            # 优先使用扁平化格式（galvo_X_detected_count）
            detected_key = f'galvo_{galvo_idx}_detected_count'
            processed_key = f'galvo_{galvo_idx}_processed_count'
            
            if detected_key in stats:
                detected_count = int(stats[detected_key])
            elif f'galvo_{galvo_idx}' in stats and isinstance(stats[f'galvo_{galvo_idx}'], dict):
                # 兼容嵌套格式（galvo_0 对象）
                galvo_data = stats[f'galvo_{galvo_idx}']
                if 'detected_count' in galvo_data:
                    detected_count = int(galvo_data['detected_count'])
            
            if processed_key in stats:
                processed_count = int(stats[processed_key])
            elif f'galvo_{galvo_idx}' in stats and isinstance(stats[f'galvo_{galvo_idx}'], dict):
                # 兼容嵌套格式（galvo_0 对象）
                galvo_data = stats[f'galvo_{galvo_idx}']
                if 'processed_count' in galvo_data:
                    processed_count = int(galvo_data['processed_count'])
            
            # 兼容 galvo_states 数组格式
            if detected_count == 0 or processed_count == 0:
                if 'galvo_states' in stats and isinstance(stats['galvo_states'], list):
                    for galvo_info in stats['galvo_states']:
                        if isinstance(galvo_info, dict) and galvo_info.get('index') == galvo_idx:
                            if detected_count == 0 and 'detected_count' in galvo_info:
                                detected_count = int(galvo_info['detected_count'])
                            if processed_count == 0 and 'processed_count' in galvo_info:
                                processed_count = int(galvo_info['processed_count'])
                            break
            
            # 更新该振镜的显示
            self.galvo_labels[galvo_idx]['detected'].setText(str(detected_count))
            self.galvo_labels[galvo_idx]['processed'].setText(str(processed_count))
        
        # 更新 FPS
        if 'fps' in stats:
            self.fps_label.setText(f"{stats['fps']:.1f}")
        
        # 更新系统状态
        if 'system_status' in stats:
            status = stats['system_status']
            self.status_label.setText(status)
            # 根据状态改变颜色
            if status == 'IDLE':
                self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #9E9E9E;")
            elif status == 'TRACKING':
                self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF9800;")
            elif status == 'FIRING':
                self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #F44336;")


class LaunchSelectorWidget(QWidget):
    """Launch 文件选择组件"""
    
    launch_selected = pyqtSignal(str)  # 选择的 launch 文件名
    start_requested = pyqtSignal(str)  # 启动请求
    stop_requested = pyqtSignal()  # 停止请求
    
    def __init__(self, launch_manager, parent=None):
        super().__init__(parent)
        self.launch_manager = launch_manager
        self.init_ui()
        self.update_launch_list()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title = QLabel("Launch 控制")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Launch 选择下拉框
        layout.addWidget(QLabel("选择 Launch 文件:"))
        self.launch_combo = QComboBox()
        self.launch_combo.setToolTip("选择要启动的 ROS launch 文件")
        layout.addWidget(self.launch_combo)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("启动")
        self.start_button.setToolTip("启动选中的 launch 文件")
        self.start_button.clicked.connect(self.on_start_clicked)
        print("DEBUG: 启动按钮信号已连接")  # 调试信息
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.setToolTip("停止当前运行的 launch 进程")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        # 状态标签
        self.status_label = QLabel("未运行")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #9E9E9E;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_launch_list(self):
        """更新 launch 文件列表"""
        self.launch_combo.clear()
        launch_files = self.launch_manager.get_launch_files()
        print(f"DEBUG: 找到 {len(launch_files)} 个 launch 文件")  # 调试信息
        if len(launch_files) == 0:
            print("DEBUG: 警告: 没有找到任何 launch 文件！")  # 调试信息
            self.launch_combo.addItem("未找到 launch 文件", None)
        else:
            for launch in launch_files:
                print(f"DEBUG: 添加 launch: {launch['name']} -> {launch['display_name']}")  # 调试信息
                self.launch_combo.addItem(launch['display_name'], launch['name'])
    
    def on_start_clicked(self):
        """启动按钮点击"""
        print("DEBUG: 启动按钮被点击")  # 调试信息
        current_data = self.launch_combo.currentData()
        print(f"DEBUG: 当前选择的 launch: {current_data}")  # 调试信息
        if current_data:
            print(f"DEBUG: 发送启动信号: {current_data}")  # 调试信息
            self.start_requested.emit(current_data)
        else:
            print("DEBUG: 警告: 没有选择 launch 文件")  # 调试信息
    
    def on_stop_clicked(self):
        """停止按钮点击"""
        self.stop_requested.emit()
    
    def set_running(self, running):
        """设置运行状态"""
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if running:
            self.status_label.setText("运行中")
            self.status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.status_label.setText("未运行")
            self.status_label.setStyleSheet("color: #9E9E9E;")
    
    def clear_output(self):
        """清空输出（由外部调用）"""
        pass  # 输出显示在单独的组件中


class LaunchOutputWidget(QWidget):
    """Launch 输出显示组件"""
    
    # 定义信号用于线程安全更新
    output_received = pyqtSignal(str, str)  # text, stream_type
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        # 连接信号到槽
        self.output_received.connect(self._append_output_safe)
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title = QLabel("Launch 运行输出")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # 输出文本区域
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont('Courier', 9))
        self.output_text.setAcceptRichText(True)  # 允许 HTML 格式
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.output_text)
        
        # 清空按钮
        clear_button = QPushButton("清空")
        clear_button.setToolTip("清空输出内容")
        clear_button.clicked.connect(self.output_text.clear)
        layout.addWidget(clear_button)
        
        self.setLayout(layout)
    
    def append_output(self, text, stream_type='stdout'):
        """添加输出文本（线程安全版本，通过信号调用）"""
        # 通过信号发送，确保在主线程中执行
        self.output_received.emit(text, stream_type)
    
    def _append_output_safe(self, text, stream_type='stdout'):
        """在主线程中安全地添加输出文本"""
        # 根据流类型设置颜色和前缀
        if stream_type == 'stderr':
            color = '#ff6b6b'  # 红色表示错误
            prefix = '[ERROR] '
        else:
            color = '#4ecdc4'  # 青色表示标准输出
            prefix = '[INFO] '
        
        # 使用 HTML 格式添加带颜色的文本
        html_text = f"<span style='color: {color};'>{prefix}{text}</span>"
        self.output_text.append(html_text)
        
        # 自动滚动到底部
        scrollbar = self.output_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear(self):
        """清空输出"""
        self.output_text.clear()


class ManualGalvoControlWidget(QWidget):
    """手动振镜控制组件"""
    
    galvo_command = pyqtSignal(int, int, int)  # galvo_index, x, y
    laser_control = pyqtSignal(int, bool)  # galvo_index, enable
    
    def __init__(self, galvo_count=2, parent=None):
        super().__init__(parent)
        self.galvo_count = galvo_count
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title = QLabel("手动振镜控制")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # 为每个振镜创建控制面板
        self.galvo_panels = []
        for i in range(self.galvo_count):
            panel = self.create_galvo_panel(i)
            self.galvo_panels.append(panel)
            layout.addWidget(panel)
            if i < self.galvo_count - 1:
                # 添加分隔线
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setFrameShadow(QFrame.Sunken)
                layout.addWidget(line)
        
        # 为每个振镜创建独立的激光控制
        self.laser_buttons = []
        for galvo_idx in range(self.galvo_count):
            laser_group = QGroupBox(f"振镜 {galvo_idx + 1} 激光控制")
            laser_layout = QHBoxLayout()
            
            laser_on_btn = QPushButton("开启激光")
            laser_on_btn.setToolTip(f"开启振镜 {galvo_idx + 1} 的激光器")
            laser_on_btn.clicked.connect(lambda checked, idx=galvo_idx: self.on_laser_button_clicked(idx, True))
            laser_layout.addWidget(laser_on_btn)
            
            laser_off_btn = QPushButton("关闭激光")
            laser_off_btn.setToolTip(f"关闭振镜 {galvo_idx + 1} 的激光器")
            laser_off_btn.clicked.connect(lambda checked, idx=galvo_idx: self.on_laser_button_clicked(idx, False))
            laser_layout.addWidget(laser_off_btn)
            
            laser_group.setLayout(laser_layout)
            layout.addWidget(laser_group)
            self.laser_buttons.append((laser_on_btn, laser_off_btn))
        
        layout.addStretch()
        self.setLayout(layout)
    
    def create_galvo_panel(self, galvo_index):
        """创建单个振镜的控制面板"""
        group = QGroupBox(f"振镜 {galvo_index + 1}")
        layout = QGridLayout()
        
        # X 轴控制
        layout.addWidget(QLabel("X 轴:"), 0, 0)
        x_slider = QSlider(Qt.Horizontal)
        x_slider.setRange(-32767, 32767)
        x_slider.setValue(0)
        x_slider.setToolTip("X 轴位置 (-32767 到 32767)")
        layout.addWidget(x_slider, 0, 1)
        
        x_spinbox = QSpinBox()
        x_spinbox.setRange(-32767, 32767)
        x_spinbox.setValue(0)
        x_spinbox.setToolTip("X 轴位置数值")
        x_spinbox.valueChanged.connect(x_slider.setValue)
        x_slider.valueChanged.connect(x_spinbox.setValue)
        x_slider.valueChanged.connect(
            lambda v: self.on_galvo_value_changed(galvo_index, 'x', v)
        )
        layout.addWidget(x_spinbox, 0, 2)
        
        # Y 轴控制
        layout.addWidget(QLabel("Y 轴:"), 1, 0)
        y_slider = QSlider(Qt.Horizontal)
        y_slider.setRange(-32767, 32767)
        y_slider.setValue(0)
        y_slider.setToolTip("Y 轴位置 (-32767 到 32767)")
        layout.addWidget(y_slider, 1, 1)
        
        y_spinbox = QSpinBox()
        y_spinbox.setRange(-32767, 32767)
        y_spinbox.setValue(0)
        y_spinbox.setToolTip("Y 轴位置数值")
        y_spinbox.valueChanged.connect(y_slider.setValue)
        y_slider.valueChanged.connect(y_spinbox.setValue)
        y_slider.valueChanged.connect(
            lambda v: self.on_galvo_value_changed(galvo_index, 'y', v)
        )
        layout.addWidget(y_spinbox, 1, 2)
        
        # 中心按钮
        center_button = QPushButton("回中心")
        center_button.setToolTip("将振镜移动到中心位置 (0, 0)")
        center_button.clicked.connect(
            lambda: self.on_center_clicked(galvo_index)
        )
        layout.addWidget(center_button, 2, 0, 1, 3)
        
        group.setLayout(layout)
        
        # 存储控件引用到 group 对象
        group.x_slider = x_slider
        group.x_spinbox = x_spinbox
        group.y_slider = y_slider
        group.y_spinbox = y_spinbox
        
        return group
    
    def on_galvo_value_changed(self, galvo_index, axis, value):
        """振镜值改变回调"""
        panel = self.galvo_panels[galvo_index]
        if axis == 'x':
            y = panel.y_spinbox.value()
            print(f"DEBUG [GUI Component]: Emitting galvo_command: galvo_index={galvo_index}, x={value}, y={y}")
            self.galvo_command.emit(galvo_index, value, y)
        else:
            x = panel.x_spinbox.value()
            print(f"DEBUG [GUI Component]: Emitting galvo_command: galvo_index={galvo_index}, x={x}, y={value}")
            self.galvo_command.emit(galvo_index, x, value)
    
    def on_center_clicked(self, galvo_index):
        """回中心按钮点击"""
        panel = self.galvo_panels[galvo_index]
        panel.x_slider.setValue(0)
        panel.y_slider.setValue(0)
        print(f"DEBUG [GUI Component]: Center clicked for galvo_index={galvo_index}")
        self.galvo_command.emit(galvo_index, 0, 0)
    
    def on_laser_button_clicked(self, galvo_index, enabled):
        """激光按钮点击回调"""
        self.laser_control.emit(galvo_index, enabled)
    
    def set_enabled(self, enabled):
        """设置控件启用状态"""
        for panel in self.galvo_panels:
            panel.setEnabled(enabled)
        for laser_on_btn, laser_off_btn in self.laser_buttons:
            laser_on_btn.setEnabled(enabled)
            laser_off_btn.setEnabled(enabled)


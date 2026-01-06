#!/bin/bash
# -*- coding: utf-8 -*-

# 激光除草系统 GUI 启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到脚本目录
cd "$SCRIPT_DIR"

# ========== 自动设置 Conda 环境 ==========
# 初始化 conda（如果还没有初始化）
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "yolo" ]; then

    # 激活 yolo 环境
    if command -v conda &> /dev/null; then
        echo "正在激活 conda 环境: yolo"
        conda activate yolo
        if [ $? -ne 0 ]; then
            echo "警告: 无法激活 yolo 环境，继续使用当前环境"
        else
            echo "已激活 conda 环境: yolo"
        fi
    fi
fi

# ========== 自动设置 ROS 环境 ==========
if [ -z "$ROS_DISTRO" ]; then
    # 尝试自动 source ROS 环境
    ROS_DISTROS=("noetic")
    ROS_SOURCE_SUCCESS=false
    
    for distro in "${ROS_DISTROS[@]}"; do
        ROS_SETUP="/home/zhong/my_workspace/devel/setup.bash"
        if [ -f "$ROS_SETUP" ]; then
            echo "正在加载 ROS 环境: $distro"
            source "$ROS_SETUP"
            ROS_SOURCE_SUCCESS=true
            break
        fi
    done
    
fi

# Source workspace setup.bash
# 确保使用绝对路径
# gui/ -> ../ -> laser_weeding/ -> ../ -> src/ -> ../ -> my_workspace/
WORKSPACE_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
echo "Workspace 根目录: $WORKSPACE_ROOT"

# 明确指定 setup.bash 的路径
SETUP_BASH="$WORKSPACE_ROOT/devel/setup.bash"
if [ ! -f "$SETUP_BASH" ]; then
    SETUP_BASH="$WORKSPACE_ROOT/install/setup.bash"
fi

if [ -f "$SETUP_BASH" ]; then
    echo "正在加载 workspace 环境: $SETUP_BASH"
    # 使用绝对路径 source
    source "$SETUP_BASH"
    echo "已加载 setup.bash"
    echo "当前 ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"
else
    echo "警告: 未找到 workspace setup.bash"
    echo "尝试手动设置 ROS_PACKAGE_PATH..."
    # 手动添加 workspace src 到 ROS_PACKAGE_PATH
    WORKSPACE_SRC="$WORKSPACE_ROOT/src"
    if [ -d "$WORKSPACE_SRC" ]; then
        if [ -z "$ROS_PACKAGE_PATH" ]; then
            export ROS_PACKAGE_PATH="$WORKSPACE_SRC"
        else
            export ROS_PACKAGE_PATH="$WORKSPACE_SRC:$ROS_PACKAGE_PATH"
        fi
        echo "已设置 ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"
    fi
fi

# 验证 ROS 包路径
echo "当前 ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"
if command -v rospack &> /dev/null; then
    echo "验证 laser_weeding 包路径:"
    rospack find laser_weeding 2>&1 || echo "警告: 无法找到 laser_weeding 包"
fi

# 添加项目根目录到 Python 路径，以便导入其他模块
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT/scripts:$PYTHONPATH"

# 检查 Python 依赖
# 优先使用当前激活环境的 Python（如果使用 conda）
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
    echo "使用 conda 环境 Python: $PYTHON_CMD"
else
    PYTHON_CMD="python3"
    echo "使用系统 Python: $PYTHON_CMD"
fi

$PYTHON_CMD -c "import PyQt5" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: PyQt5 未安装，请运行: pip3 install PyQt5"
    exit 1
fi

# 修复 Qt platform plugin 问题
# 检测实际的 Python 环境路径（使用上面确定的 Python 命令）
PYTHON_ENV=$($PYTHON_CMD -c "import sys; print(sys.prefix)" 2>/dev/null)
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)

# 首先排除 OpenCV 的 Qt 插件干扰
# 临时重命名 OpenCV 的 Qt 插件目录，避免被 Qt 加载
CV2_QT_PLUGINS="$PYTHON_ENV/lib/python${PYTHON_VERSION}/site-packages/cv2/qt/plugins"
CV2_QT_PLUGINS_DISABLED="$PYTHON_ENV/lib/python${PYTHON_VERSION}/site-packages/cv2/qt/plugins.disabled"
if [ -d "$CV2_QT_PLUGINS" ] && [ ! -d "$CV2_QT_PLUGINS_DISABLED" ]; then
    # 临时重命名目录（需要写权限）
    if mv "$CV2_QT_PLUGINS" "$CV2_QT_PLUGINS_DISABLED" 2>/dev/null; then
        echo "已临时禁用 OpenCV Qt 插件: $CV2_QT_PLUGINS"
        # 设置清理函数，在脚本退出时恢复
        trap "mv '$CV2_QT_PLUGINS_DISABLED' '$CV2_QT_PLUGINS' 2>/dev/null || true" EXIT
    else
        echo "警告: 无法禁用 OpenCV Qt 插件（可能需要权限）: $CV2_QT_PLUGINS"
    fi
elif [ -d "$CV2_QT_PLUGINS_DISABLED" ]; then
    echo "OpenCV Qt 插件已被禁用: $CV2_QT_PLUGINS_DISABLED"
fi

# 如果使用 conda 环境，优先使用 conda 环境中的 Qt 库
if [ -n "$PYTHON_ENV" ] && [ -d "$PYTHON_ENV/lib/python${PYTHON_VERSION}/site-packages/PyQt5" ]; then
    # 使用实际 Python 环境的 PyQt5 Qt 库
    PYQT5_BASE="$PYTHON_ENV/lib/python${PYTHON_VERSION}/site-packages/PyQt5"
    if [ -d "$PYQT5_BASE/Qt5/lib" ]; then
        # 将 PyQt5 的 Qt 库路径添加到 LD_LIBRARY_PATH 的最前面
        export LD_LIBRARY_PATH="$PYQT5_BASE/Qt5/lib:$LD_LIBRARY_PATH"
        echo "使用 PyQt5 Qt 库: $PYQT5_BASE/Qt5/lib"
    fi
    # 设置 Qt 插件路径为 PyQt5 的插件目录
    if [ -d "$PYQT5_BASE/Qt5/plugins" ]; then
        # 只使用 PyQt5 的插件，明确指定路径（不包含 OpenCV 的路径）
        # 直接指定平台插件路径，这样 Qt 就不会搜索其他路径
        if [ -d "$PYQT5_BASE/Qt5/plugins/platforms" ]; then
            export QT_QPA_PLATFORM_PLUGIN_PATH="$PYQT5_BASE/Qt5/plugins/platforms"
            # 设置 QT_PLUGIN_PATH 只包含 PyQt5 的插件目录
            export QT_PLUGIN_PATH="$PYQT5_BASE/Qt5/plugins"
        else
            export QT_QPA_PLATFORM_PLUGIN_PATH="$PYQT5_BASE/Qt5/plugins"
            export QT_PLUGIN_PATH="$PYQT5_BASE/Qt5/plugins"
        fi
        echo "使用 Qt 插件路径: $QT_QPA_PLATFORM_PLUGIN_PATH"
    else
        # 如果找不到插件，清空路径避免冲突
        export QT_QPA_PLATFORM_PLUGIN_PATH=""
        unset QT_PLUGIN_PATH
    fi
else
    # 非 conda 环境，使用系统设置
    export QT_QPA_PLATFORM_PLUGIN_PATH=""
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
fi

# 启动 GUI
$PYTHON_CMD gui_main.py


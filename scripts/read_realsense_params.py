import pyrealsense2 as rs

# 创建管道
pipeline = rs.pipeline()
config = rs.config()

# 启动相机（可设置分辨率）
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取彩色图像的内参
color_stream = profile.get_stream(rs.stream.color)  # 获取彩色流配置
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

# 打印内参
print("Width:", intrinsics.width)
print("Height:", intrinsics.height)
print("fx:", intrinsics.fx)
print("fy:", intrinsics.fy)
print("ppx (cx):", intrinsics.ppx)
print("ppy (cy):", intrinsics.ppy)
print("Distortion model:", intrinsics.model)
print("Distortion coefficients:", intrinsics.coeffs)

pipeline.stop()
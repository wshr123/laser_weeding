import numpy as np
from scipy.spatial.transform import Rotation


def find_rigid_transform_3d(points_A, points_B):
    """
    使用SVD算法计算从点云A到点云B的三维刚体变换（旋转R和平移t）。
    目标是找到最佳的R和t，使得 B ≈ R * A + t。
    """
    if points_A.shape != points_B.shape:
        raise ValueError("输入点云的维度必须相同")
    if points_A.shape[0] < 3:
        raise ValueError("至少需要3个点来计算变换")

    # 1. 计算两个点云的质心
    centroid_A = np.mean(points_A, axis=0)
    centroid_B = np.mean(points_B, axis=0)

    # 2. 对点云进行去中心化处理
    A_centered = points_A - centroid_A
    B_centered = points_B - centroid_B

    # 3. 计算协方差矩阵 H = A'T * B'
    H = A_centered.T @ B_centered

    # 4. 对H进行SVD分解，求解旋转矩阵 R
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T

    # 5. 特殊情况处理：防止出现反射/镜像
    if np.linalg.det(R) < 0:
        print("检测到反射，正在进行修正...")
        V[:, -1] *= -1
        R = V @ U.T

    # 6. 利用质心和旋转矩阵计算平移向量 t
    t = centroid_B.T - R @ centroid_A.T

    return R, t.reshape(3, 1)


def main():
    """主验证函数"""
    print("=" * 60)
    print("正在用【给定点】验证 find_rigid_transform_3d 函数的正确性...")
    print("=" * 60)

    # 1. 创建给定的、易于判断的源点云 points_A
    #    包含原点、X/Y/Z轴单位向量，以及一个任意点
    points_A = np.array([
        [0., 0., 0.],  # 原点
        [1., 0., 0.],  # X轴方向
        [0., 1., 0.],  # Y轴方向
        [0., 0., 1.],  # Z轴方向
        [1., 2., 3.]  # 一个任意点
    ])
    print("给定的源点云 (points_A):\n", points_A)

    # 2. 设定一个“真实”的、易于判断的变换作为标准答案
    #    旋转：绕 Z 轴旋转 90 度
    R_true = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    #    平移：一个简单的整数向量
    t_true = np.array([[10.], [20.], [30.]])

    print("\n【标准答案】(我们期望函数计算出的结果)")
    print("真实的旋转矩阵 R_true (绕Z轴90度):\n", np.round(R_true, 4))
    print("真实的平移向量 t_true:\n", np.round(t_true, 4))

    # 我们可以手动预测一下变换结果：
    # 原点 [0,0,0] -> 平移到 [10,20,30]
    # X轴 [1,0,0] -> 旋转为 Y轴 [0,1,0] -> 再平移到 [10,21,30]
    # Y轴 [0,1,0] -> 旋转为 -X轴 [-1,0,0] -> 再平移到 [9,20,30]
    # Z轴 [0,0,1] -> 旋转不变 [0,0,1] -> 再平移到 [10,20,31]

    # 3. 生成目标点云 points_B
    points_B = np.array([(R_true @ p + t_true.flatten()) for p in points_A])

    # 在这个测试中，我们不添加噪声，以确保结果的精确匹配
    print("\n根据真实变换生成的目标点云 (points_B):\n", np.round(points_B, 4))

    # 4. 运行待验证的函数
    print("\n>>> 正在调用 find_rigid_transform_3d 函数进行计算...")
    R_calc, t_calc = find_rigid_transform_3d(points_A, points_B)

    print("\n【计算结果】")
    print("计算出的旋转矩阵 R_calc:\n", np.round(R_calc, 4))
    print("计算出的平移向量 t_calc:\n", np.round(t_calc, 4))

    # 5. 对比结果
    print("\n【验证过程】")
    # 使用 numpy.allclose 来比较浮点数矩阵是否“足够接近”
    rotation_match = np.allclose(R_true, R_calc)
    translation_match = np.allclose(t_true, t_calc)

    print(f"旋转矩阵R是否与真实值完全匹配: {'✅ 成功' if rotation_match else '❌ 失败'}")
    print(f"平移向量t是否与真实值完全匹配: {'✅ 成功' if translation_match else '❌ 失败'}")

    # 附加验证：计算重投影误差
    points_A_reprojected_to_B = np.array([(R_calc @ p + t_calc.flatten()) for p in points_A])
    error = np.sqrt(np.mean(np.sum((points_B - points_A_reprojected_to_B) ** 2, axis=1)))
    print(f"\n重投影均方根误差 (RMSE): {error:.15f}")  # 误差应该是一个非常接近0的数

    if rotation_match and translation_match:
        print("\n🎉 验证通过！函数能够精确地从给定点中恢复出原始的旋转和平移。")
    else:
        print("\n❌ 验证失败！请检查算法实现。")
    print("=" * 60)


if __name__ == "__main__":
    main()
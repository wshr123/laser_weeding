from dataclasses import dataclass
import numpy as np

@dataclass
class GalvoParams:
    # 几何与标定参数
    Z_mm: float = 76.0          # 工作平面法向距离
    bx: float = 0.0             # 零点偏置（code）
    by: float = 0.0
    # 2x2 线性矩阵（rad/code）
    kx: float = 0.0             # (cx->alpha)
    ky: float = 0.0             # (cy->beta)
    axy: float = -1.236e-5      # (cy->alpha)
    ayx: float = -1.218e-5      # (cx->beta)

def angles_from_plane_YFIRST(X_mm, Y_mm, Z_mm):
    """
    反解角度（Y先、X后）:
        alpha = atan( Y / Z )
        beta  = atan( (X / Z) * cos(alpha) )
    支持标量或 numpy 数组。
    """
    X = np.asarray(X_mm, dtype=float)
    Y = np.asarray(Y_mm, dtype=float)
    Z = float(Z_mm)
    alpha = np.arctan2(Y, Z)                         # rad
    ca = np.cos(alpha)
    beta  = np.arctan2((X / Z) * ca, 1.0)            # 等价于 atan( (X/Z)*ca )
    return alpha, beta

def codes_from_angles(alpha, beta, p: GalvoParams):
    """
    解 2x2 线性系统：
        [alpha]   [ kx  axy ] [cx-bx]
        [beta ] = [ ayx  ky ] [cy-by]
    返回 (cx,cy)（int16，已限幅）。
    """
    alpha = np.asarray(alpha, dtype=float)
    beta  = np.asarray(beta,  dtype=float)

    A11, A12 = p.kx,  p.axy
    A21, A22 = p.ayx, p.ky
    det = A11*A22 - A12*A21

    if abs(det) < 1e-16:
        # 退化到“纯交叉”：cx = bx + beta/ayx, cy = by + alpha/axy
        cx = p.bx + beta  / (p.ayx if abs(p.ayx)>1e-16 else -1e-5)
        cy = p.by + alpha / (p.axy if abs(p.axy)>1e-16 else -1e-5)
    else:
        dcx = ( A22*alpha - A12*beta ) / det
        dcy = (-A21*alpha + A11*beta ) / det
        cx = p.bx + dcx
        cy = p.by + dcy

    # 限幅到 16-bit 有符号
    cx = np.clip(np.rint(cx), -32767, 32767).astype(np.int16)
    cy = np.clip(np.rint(cy), -32767, 32767).astype(np.int16)
    return cx, cy

def plane_to_codes(X_mm, Y_mm, params: GalvoParams):
    """一站式：目标平面坐标 -> (alpha,beta) -> (cx,cy)。"""
    alpha, beta = angles_from_plane_YFIRST(X_mm, Y_mm, params.Z_mm)
    return codes_from_angles(alpha, beta, params)

# ---------- 示例：生成 40x40 mm 方框的码值点列 ----------
if __name__ == "__main__":
    p = GalvoParams()  # 用你的默认参数
    HALF_S = 20.0      # 半边长 20mm
    EDGE_N = 1000      # 每边点数
    DWELL  = 150       # 拐角驻留

    def edge_points(P0, P1, N):
        P0 = np.array(P0, float); P1 = np.array(P1, float)
        t = np.linspace(0, 1, N)
        XY = (1-t)[:,None]*P0[None,:] + t[:,None]*P1[None,:]
        return XY[:,0], XY[:,1]

    # 组装四条边
    Xs, Ys = [], []
    # 上边：(-S,+S)->(+S,+S)
    x,y = edge_points((-HALF_S,+HALF_S), (+HALF_S,+HALF_S), EDGE_N); Xs.append(x); Ys.append(y)
    # 右边：(+S,+S)->(+S,-S)
    x,y = edge_points((+HALF_S,+HALF_S), (+HALF_S,-HALF_S), EDGE_N); Xs.append(x); Ys.append(y)
    # 下边：(+S,-S)->(-S,-S)
    x,y = edge_points((+HALF_S,-HALF_S), (-HALF_S,-HALF_S), EDGE_N); Xs.append(x); Ys.append(y)
    # 左边：(-S,-S)->(-S,+S)
    x,y = edge_points((-HALF_S,-HALF_S), (-HALF_S,+HALF_S), EDGE_N); Xs.append(x); Ys.append(y)

    X = np.concatenate([*Xs, np.repeat(Xs[-1][-1], DWELL),  # 每条边后加驻留
                        *[]])
    Y = np.concatenate([*Ys, np.repeat(Ys[-1][-1], DWELL),
                        *[]])

    # 简单把每个拐角也多驻留一下（可按需添加更多驻留点）
    for k in range(3):
        X = np.concatenate([X, np.repeat(Xs[k+1][0], DWELL)])
        Y = np.concatenate([Y, np.repeat(Ys[k+1][0], DWELL)])

    cx, cy = plane_to_codes(X, Y, p)

    print("生成码值点数:", len(cx))
    print("前 5 个点示例:")
    for i in range(5):
        print(int(cx[i]), int(cy[i]))
    # 你可以把 (cx,cy) 通过串口/网口发送给下位机，按固定点率逐点输出

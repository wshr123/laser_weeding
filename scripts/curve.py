import numpy as np

pts = np.array([
    [44.5, 47.518],
    [-44.0, 47.162],
    [0.0, 44.5],
], dtype=float)

x = pts[:,0]
y = pts[:,1]

# 设计矩阵 [x^2, x, 1]
X = np.vstack([x**2, x, np.ones_like(x)]).T
a, b, c = np.linalg.solve(X, y)

print(f"a = {a:.10f}")
print(f"b = {b:.10f}")
print(f"c = {c:.10f}")
print(f"y(x) = {a:.10f} x^2 + {b:.10f} x + {c:.10f}")
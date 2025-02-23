import numpy as np


def f1(x):
    return 0.5 * np.exp(-0.5 * x) + 0.4 * np.exp(0.3 * x)


def f2(x):
    return (x - 4) ** 2


def f3(x):
    return 0.5 * x ** 2 * np.log(1 + x ** 2) + x ** 2


def f4(x):
    return x ** 2 + np.exp(0.1 * x)


def f5(x):
    return np.log(np.exp(-0.1 * x) + np.exp(0.3 * x)) + 0.1 * x ** 2


def f6(x):
    return x ** 2 / np.log(2 + x ** 2)


def f7(x):
    return 0.2 * np.exp(-0.2 * x) + 0.4 * np.exp(0.4 * x)


def f8(x):
    return x ** 4 + 2 * x ** 2 + 2


def f9(x):
    return x ** 2 / np.sqrt(x ** 2 + 1) + 0.1 * x ** 2


def f10(x):
    return (x + 2) ** 2


###############################################################
import numpy as np
import matplotlib.pyplot as plt

# 根据图像构建邻接矩阵（10节点有向图）
A = np.zeros((10, 10))
# 定义箭头连接关系（示例权重，需根据实际论文调整）
A[0, 1] = 1  # 1->2
A[1, 2], A[1, 9] = 1, 1  # 2->3,10
A[2, 3] = 1  # 3->4
A[5, 6] = 1  # 6->7
A[4, 5], A[4, 3] = 1, 1  # 5->6,4
A[3, 4], A[3, 1] = 1, 1  # 4->5,2
A[6, 7], A[6, 4] = 1, 1  # 7->8,5
A[7, 6], A[7, 9] = 1, 1  # 8->7,10
A[8, 7] = 1  # 9->8
A[9, 8], A[9, 0] = 1, 1  # 10->9, 1

# 确保权重平衡（每行归一化）
A = A / A.sum(axis=1, keepdims=True)


# 定义10个局部函数的梯度
def local_gradient(i, x):
    if i == 0:  # f1(x) = 0.5e^{-0.5x} + 0.4e^{0.3x}
        return -0.25 * np.exp(-0.5 * x) + 0.12 * np.exp(0.3 * x)
    elif i == 1:  # f2(x) = (x-4)^2
        return 2 * (x - 4)
    elif i == 2:  # f3(x) = 0.5x²ln(1+x²) + x²
        return x * np.log(1 + x ** 2) + (x ** 3) / (1 + x ** 2) + 2 * x
    elif i == 3:  # f4(x) = x² + e^{0.1x}
        return 2 * x + 0.1 * np.exp(0.1 * x)
    elif i == 4:  # f5(x) = ln(e^{-0.1x} + e^{0.3x}) + 0.1x²
        numerator = -0.1 * np.exp(-0.1 * x) + 0.3 * np.exp(0.3 * x)
        denominator = np.exp(-0.1 * x) + np.exp(0.3 * x)
        return numerator / denominator + 0.2 * x
    elif i == 5:  # f6(x) = x²/ln(2+x²)
        return (2 * x * np.log(2 + x ** 2) - x ** 2 * (2 * x) / (2 + x ** 2)) / (np.log(2 + x ** 2) ** 2)
    elif i == 6:  # f7(x) = 0.2e^{-0.2x} + 0.4e^{0.4x}
        return -0.04 * np.exp(-0.2 * x) + 0.16 * np.exp(0.4 * x)
    elif i == 7:  # f8(x) = x^4 + 2x² + 2
        return 4 * x ** 3 + 4 * x
    elif i == 8:  # f9(x) = x²/√(x²+1) + 0.1x²
        return (2 * x * np.sqrt(x ** 2 + 1) - x ** 3 / (np.sqrt(x ** 2 + 1))) / (x ** 2 + 1) + 0.2 * x
    elif i == 9:  # f10(x) = (x+2)^2
        return 2 * (x + 2)


# 初始化分布式算法参数
N = 10
alpha = 0.5
beta = 1.2
T = 40.0
dt = 0.005
steps = int(T / dt)

# 定义拉普拉斯矩阵
D_out = np.diag(A.sum(axis=1))
L = D_out - A

# 初始化状态
x = np.random.randn(N)
z = np.zeros(N)

# 存储历史数据
x_history = np.zeros((steps, N))
# 添加全局成本函数tilde_f的收敛过程计算
funcs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
tilde_f_history = np.zeros(steps)

# 数值积分
for k in range(steps):
    gradients = np.array([local_gradient(i, x[i]) for i in range(N)])
    Lx = L @ x
    Lz = L @ z

    dz = Lx
    dx = -alpha * gradients - Lx - Lz

    z += dz * dt
    x += dx * dt

    x_history[k, :] = x
    # 计算当前时刻的tilde_f
    tilde_f_history[k] = sum(funcs[i](x[i]) for i in range(N))

# 可视化收敛过程
plt.figure(figsize=(12, 6))
for i in range(N):
    plt.plot(np.linspace(0, T, steps), x_history[:, i], alpha=0.7,
             label=f"Agent {i + 1} (f{i + 1})")
plt.title("Distributed Optimization on 10-Node Network")
plt.xlabel("Time")
plt.ylabel("State Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
# plt.show()

# 绘制tilde_f的收敛过程
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, T, steps), tilde_f_history, color='purple')
plt.title(r'Convergence of $\tilde{f}(\mathbf{x})=\sum_{i=1}^{10} f^i(x^i)$')
plt.xlabel('Time')
plt.ylabel('Cost Value')
plt.grid(True)
# plt.show()

# 绘制总函数f(x)=Σf^i(x)的曲线
x_vals = np.linspace(-3, 6, 1000)
f_total = np.zeros_like(x_vals)
for j in range(len(x_vals)):
    f_total[j] = sum(f(x_vals[j]) for f in funcs)

# 找到全局最优解的近似值
x_opt = x_vals[np.argmin(f_total)]
f_opt = np.min(f_total)

# 绘制总函数和代理最终状态
plt.figure(figsize=(12, 6))
plt.plot(x_vals, f_total, label=r'$f(x)=\sum_{i=1}^{10} f^i(x)$', linewidth=2)
plt.scatter(x_history[-1, :], [sum(f(x) for f in funcs) for x in x_history[-1, :]],
            color='red', zorder=5, label='Agent Final States')
plt.axvline(x=x_opt, color='green', linestyle='--', label='Global Minimum')
plt.title('Global Cost Function and Optimization Result')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
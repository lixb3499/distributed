# 论文中(3)式，没有约束时
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义系统参数
k = 10  # 控制增益（调整收敛速度）
b = 0.1  # 扰动幅度（小正数）
omega = 10  # 扰动频率（需满足论文中的频率条件）
# delta = 0.1  # 小参数（k = delta * k_bar，这里简化直接使用k）
t_span = (0, 100)  # 仿真时间范围
x0_hat = 0.0  # 初始估计值 x̂(0)


# 目标函数 f(x) = (x-1)^2
def f(x):
    return (x - 1) ** 2


# 定义ODE系统（公式3的简化版）
def system(t, x_hat):
    # 计算当前实际状态 x = x̂ + b*sin(ωt)
    x = x_hat + b * np.sin(omega * t)
    # 计算导数 d(x̂)/dt = -k * f(x) * b * sin(ωt)
    dx_hat_dt = -k * f(x) * b * np.sin(omega * t)
    return dx_hat_dt


# 数值求解ODE
sol = solve_ivp(system, t_span, [x0_hat], method='RK45', dense_output=True)

# 提取结果
t_eval = np.linspace(0, sol.t[-1], 1000)
x_hat = sol.sol(t_eval).T.flatten()
x = x_hat + b * np.sin(omega * t_eval)  # 实际状态x(t)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(t_eval, x, label='x(t)')
plt.plot(t_eval, x_hat, '--', label='x̂(t)')
plt.axhline(1, color='r', linestyle=':', label='Optimal x=1')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('Extremum Seeking for f(x) = (x-1)^2')
plt.legend()
plt.grid(True)
plt.show()

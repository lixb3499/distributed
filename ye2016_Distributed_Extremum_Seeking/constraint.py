import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义系统参数
k = 20  # 控制增益（调整收敛速度）
b = 0.1  # 扰动幅度（小正数）
alpha = 0.1  # 拉格朗日乘子更新增益
omega = 20  # 扰动频率（需满足频率条件）
t_span = (0, 200)  # 仿真时间范围
x0_hat = 0.0  # 初始估计值 x̂(0)
lambda0 = 1  # 初始拉格朗日乘子 λ(0) > 0


# 目标函数和约束
def f(x):
    return (x - 1) ** 2


def g(x):
    return -(x - 0)  # 约束 g(x) = x - 3 ≤ 0


# 定义ODE系统（包含x̂和λ的更新）
def system(t, states):
    x_hat, lambda_val = states
    x = x_hat + b * np.sin(omega * t)  # 实际状态x(t)

    # 计算dx̂/dt和dλ/dt
    dx_hat_dt = -k * (f(x) + lambda_val * g(x)) * b * np.sin(omega * t)
    dlambda_dt = alpha * lambda_val * g(x) * (b ** 2) * (np.sin(omega * t) ** 2)

    return [dx_hat_dt, dlambda_dt]


# 数值求解ODE
sol = solve_ivp(system, t_span, [x0_hat, lambda0], method='RK45', dense_output=True)

# 提取结果
t_eval = np.linspace(0, sol.t[-1], 10000)
states = sol.sol(t_eval)
x_hat = states[0]
lambda_vals = states[1]
x = x_hat + b * np.sin(omega * t_eval)  # 实际状态x(t)

# 绘制结果
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t_eval, x, label='x(t)')
plt.plot(t_eval, x_hat, '--', label='x̂(t)')
plt.axhline(1, color='r', linestyle=':', label='Optimal x=1')
plt.axhline(3, color='g', linestyle='--', label='Constraint x=3')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('Extremum Seeking with Constraint g(x)=x-3 ≤ 0')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_eval, lambda_vals, label='λ(t)')
plt.xlabel('Time')
plt.ylabel('Lambda')
plt.title('Evolution of Lagrange Multiplier λ')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
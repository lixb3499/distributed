import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 系统参数设置
N = 3  # 智能体数量
n = 1  # 每个智能体的状态维度
k = 1  # 控制增益
b = 0.1  # 扰动幅度
omega = 500  # 扰动频率（需满足频率互质）
theta = 0.1  # 拉格朗日乘子更新增益
t_span = (0, 200)  # 仿真时间范围

# 通信拓扑（环形图，无向连通）
A = np.array([[0, 1, 1],
              [1, 0, 1],
              [1, 1, 0]])  # 邻接矩阵
D = np.diag(np.sum(A, axis=1))
L = D - A  # 拉普拉斯矩阵

# 正交矩阵U分解（U1为约束空间）
# _, U = np.linalg.eigh(L)
# U1 = U[:, :-n]  # 去除与零特征值对应的列

# 目标函数：各智能体的本地目标函数（c_i为本地最优值）
c = np.array([1.0, 2.0, 3.0])  # 各智能体的本地最优值


def local_cost(x_i, i):
    return (x_i - c[i]) ** 2  # f_i(x_i) = (x_i - c_i)^2


def d_local_cost(x_i, i):
    return 2 * (x_i - c[i])  # d_f_i(x_i) = 2 * (x_i - c_i)


def f_vec(x):
    return [local_cost(x[0], 0), local_cost(x[1], 1), local_cost(x[2], 2)]


def d_f_vec(x):
    return [d_local_cost(x[0], 0), d_local_cost(x[1], 1), d_local_cost(x[2], 2)]


def main1():
    # 定义式（10）的微分方程
    def system(t, states):
        hat_X = states[:N * n]  # 估计值 hat_X
        z = states[N * n:]  # 拉格朗日乘子 y

        # 计算实际状态 x_i = hat_X_i + b*sin(ωt)
        x = hat_X + b * np.sin(omega * t)

        # # 计算梯度项 [f_i(x_i) * sin(ωt)]_vec
        # grad_f = np.zeros(N * n)
        # for i in range(N):
        #     grad_f[i] = 2 * (x[i] - c[i]) * np.sin(omega * t)  # df_i/dx_i * sin(ωt)
        f_sin = [f * np.sin(omega * t) for f in f_vec(x)]

        # 计算拉普拉斯项 diag(b/2) * (L⊗I) hat_X 和 (L⊗I) U1 y
        L_hatX = L @ hat_X
        L_U1y = L @ z

        # 更新 hat_X 和 y
        d_hatX = -k * np.array(f_sin) + (b / 2) * (L_hatX + (b / 2) * L_U1y)
        dz = theta * L_hatX

        return np.concatenate([d_hatX.flatten(), dz])

    # 初始条件
    hat_X0 = np.zeros(N * n)  # 保持一维
    y0 = np.zeros(N * n)  # 可能是二维的
    initial_state = np.concatenate([hat_X0, y0.flatten()])

    # 数值求解
    sol = solve_ivp(system, t_span, initial_state, method='RK45', dense_output=True)

    # 提取结果
    t_eval = np.linspace(0, sol.t[-1], 1000)
    states = sol.sol(t_eval)
    hat_X = states[:N * n].reshape(N, -1)
    x = hat_X + b * np.sin(omega * t_eval)  # 实际状态x(t)

    # 绘制结果
    plt.figure(figsize=(12, 6))
    for i in range(N):
        plt.plot(t_eval, x[i], label=f'Agent {i + 1} (c={c[i]})')
    plt.axhline(np.mean(c), color='r', linestyle='--', label='Global Optimum')
    plt.xlabel('Time')
    plt.ylabel('State $x_i(t)$')
    plt.title('Distributed Extremum Seeking for Consensus')
    plt.legend()
    plt.grid(True)
    plt.show()



def main2():
    # 系统参数设置
    N = 3  # 智能体数量
    n = 1  # 每个智能体的状态维度
    k = 10  # 控制增益
    b = 0.1  # 扰动幅度
    omega = 50  # 扰动频率（需满足频率互质）
    theta = 1  # 拉格朗日乘子更新增益
    t_span = (0, 200)  # 仿真时间范围

    def system(t, states):
        hat_X = states[:N * n]  # 估计值 hat_X
        z = states[N * n:]  # 拉格朗日乘子 y

        # 计算实际状态 x_i = hat_X_i + b*sin(ωt)
        # x = hat_X + b * np.sin(omega * t)
        x = hat_X

        # # 计算梯度项 [f_i(x_i) * sin(ωt)]_vec
        # grad_f = np.zeros(N * n)
        # for i in range(N):
        #     grad_f[i] = 2 * (x[i] - c[i]) * np.sin(omega * t)  # df_i/dx_i * sin(ωt)
        f_sin = [f * np.sin(omega * t) for f in f_vec(x)]

        # 计算拉普拉斯项 diag(b/2) * (L⊗I) hat_X 和 (L⊗I) U1 y
        L_hatX = L @ hat_X
        L_U1y = L @ z

        # 更新 hat_X 和 y
        d_hatX = -k * (b ** 2) * (np.array(d_f_vec(hat_X)) + L_hatX + L_U1y)
        dz = (b ** 2) * theta * L_hatX

        return np.concatenate([d_hatX.flatten(), dz])

    # 初始条件
    hat_X0 = np.zeros(N * n)  # 保持一维
    y0 = np.zeros(N * n)  # 可能是二维的
    initial_state = np.concatenate([hat_X0, y0.flatten()])

    # 数值求解
    sol = solve_ivp(system, t_span, initial_state, method='RK45', dense_output=True)

    # 提取结果
    t_eval = np.linspace(0, sol.t[-1], 1000)
    states = sol.sol(t_eval)
    hat_X = states[:N * n].reshape(N, -1)
    # x = hat_X + b * np.sin(omega * t_eval)  # 实际状态x(t)
    x = hat_X
    # 绘制结果
    plt.figure(figsize=(12, 6))
    for i in range(N):
        plt.plot(t_eval, x[i], label=f'Agent {i + 1} (c={c[i]})')
    plt.axhline(np.mean(c), color='r', linestyle='--', label='Global Optimum')
    plt.xlabel('Time')
    plt.ylabel('State $x_i(t)$')
    plt.title('Distributed Extremum Seeking for Consensus')
    plt.legend()
    plt.grid(True)

    # 绘制总函数f(x)=Σf^i(x)的曲线
    x_vals = np.linspace(-3, 6, 1000)
    f_total = np.zeros_like(x_vals)
    for j in range(len(x_vals)):
        f_total[j] = local_cost(x_vals[j],0) + local_cost(x_vals[j],1) + local_cost(x_vals[j],2)

    # 找到全局最优解的近似值
    x_opt = x_vals[np.argmin(f_total)]
    f_opt = np.min(f_total)

    # 绘制总函数和代理最终状态
    plt.figure(figsize=(12, 6))
    plt.plot(x_vals, f_total, label=r'$f(x)=\sum_{i=1}^{3} f^i(x)$', linewidth=2)
    # plt.scatter(x_history[-1, :], [sum(f(x) for f in funcs) for x in x_history[-1, :]],
    #             color='red', zorder=5, label='Agent Final States')
    plt.axvline(x=x_opt, color='green', linestyle='--', label='Global Minimum')
    plt.title('Global Cost Function and Optimization Result')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.show()


if __name__ == "__main__":
    main1()

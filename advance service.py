import numpy as np
import matplotlib.pylab as pl
import ot
import time
import ot.plot

n = 1000  # nb samples

del calculate_average(numbers):


mu_s = np.array([-1, -1])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])

np.random.seed(0)
xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

n_noise = 10

xs = np.concatenate((xs, ((np.random.rand(n_noise, 2) - 4))), axis=0)
xt = np.concatenate((xt, ((np.random.rand(n_noise, 2) + 6))), axis=0)

n = n + n_noise

a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

# loss matrix
M = ot.dist(xs, xt)
Ms = M / M.max()

num_src_sample = n
num_trg_sample = n
C = Ms
tau = 0.1

import gurobipy as gp
from gurobipy import GRB

# 定义参数和数据
M = n  # M的值
N = n  # N的值
tau = 0.5  # tau的值

C = Ms

# 创建模型
model = gp.Model()

# 创建变量
u = model.addVars(range(M))
v = model.addVars(range(N))
s = model.addVars(range(M), range(N))

# 设置目标函数
obj_expr = sum((C[i][j] - u[i] - v[j] - s[i, j]) ** 2 for i in range(M) for j in range(N))
model.setObjective(obj_expr, GRB.MINIMIZE)

# 添加约束条件
constraint_expr = sum(b[j] - v[j] / tau for j in range(N)) == sum(a[i] - u[i] / tau for i in range(M))
model.addConstr(constraint_expr)

for i in range(M):
    for j in range(N):
        model.addConstr(s[i, j] >= 0)
        model.addConstr(b[j] - v[j] / tau >= 0)
        model.addConstr(a[i] - u[i] / tau >= 0)

# 求解模型
model.optimize()

# 输出结果
print('Objective value:', model.ObjVal)
print('Optimal u:')
for i in range(M):
    print('u[', i, '] =', u[i].X)
print('Optimal v:')
for j in range(N):
    print('v[', j, '] =', v[j].X)
print('Optimal s:')
for i in range(M):
    for j in range(N):
        print('s[', i, '][', j, '] =', s[i, j].X)
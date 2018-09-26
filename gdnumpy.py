# coding=utf-8
import time
import numpy as np
import matplotlib.pyplot as plt

dx = 1e-8


def f(nparr):  # 函数随便改
    nparr = np.array(nparr, dtype=np.float64)
    x, y, z = nparr
    return (x - 5) ** 2 + (y - 3) ** 2 + (z - 4) ** 2


def partial_derivative(func, paraindex, nparr):  # 指定对第几个参数求偏导数
    lis = nparr.copy()
    lis[paraindex] += dx
    return (func(lis) - func(nparr)) / dx


def gradientDescent(f, turns, nparr, a=1e-6):  # 迭代次数,函数参数,a学习率
    for _ in range(turns):
        for i in range(len(nparr)):  # 这个for很难用向量化去掉
            nparr[i] -= a * partial_derivative(f, i, nparr)
    return nparr


def f1(args):  # 最小二乘法的优化函数
    a, b = args
    return np.sum((a * xi + b - yi) ** 2)


if __name__ == '__main__':
    n = 100
    xi = np.arange(n)
    yi = xi + 0.1 * np.random.rand(n)
    # plt.scatter(xi,yi)
    tic = time.time()
    # res = gradientDescent(f, 1000, np.random.rand(3))
    res = gradientDescent(f1, 20000, np.array([1., -1.]))
    tok = time.time()
    print(res)
    print(tok - tic, 's')
#数据量不大时候原生编码速度更快,但n>50或者以上则numpy快了10倍
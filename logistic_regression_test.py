# coding=utf-8
from math import e, log


def load_data():
    from sklearn.datasets import load_iris
    dic = load_iris()
    return dic['data'][:100], dic['target'][:100]


def partial_derivative(func, paraindex, *args):  # 指定对第几个参数求偏导数
    dx = 1e-6
    arglist = list(args)
    arglist[paraindex] += dx
    arglist = tuple(arglist)
    return (func(*arglist) - func(*args)) / dx


def GradientDescent_Optimizer(f, turns, *args, a=0.004):  # 迭代次数,*args是初始值,a学习率,
    argslis = list(args)
    for _ in range(turns):
        for i in range(len(argslis)):
            argslis[i] -= a * partial_derivative(f, i, *tuple(argslis))
    return argslis


def sigmo(x):
    return 1. / (1. + e ** -x)


def logistic_regression_loss(*args, x=None, label=None):
    if not (x and label):
        x, label = load_data()
    w1, w2, w3, w4, b = args
    sum = 0
    for i in range(x.shape[0]):  # 因为载入的是numpy数组所以这里写的是shape
        xi = x[i]
        y = w1 * xi[0] + w2 * xi[1] + w3 * xi[2] + w4 * xi[3] + b  # 这里没有考虑代码的鲁棒性只考虑了容易理解
        sum += -sigmo(y) * log(label[i] + 1e-8) - (1 - sigmo(y)) * log(1 - label[i] + 1e-8)
    sum /= x.shape[0]
    return sum


if __name__ == '__main__':
    w1, w2, w3, w4, b = GradientDescent_Optimizer(logistic_regression_loss, 100, 0.1, 0.2, 0.3, 0.1, 0.0)
    print(w1, w2, w3, w4, b)
    x, label = load_data()
    xi = x[58]
    label = label[58]
    p = sigmo(w1 * xi[0] + w2 * xi[1] + w3 * xi[2] + w4 * xi[3] + b)
    print(p, label)

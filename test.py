# coding=utf-8
import mathmatical.SimpleMatrixLib as s


def f(*args):
    x, y = args
    return ((x - 3) ** 2) + ((y - 2) ** 2)


res = s.gradientDescent(f, 1000, 1., -0.1)
print(res)

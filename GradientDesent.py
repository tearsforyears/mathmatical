#coding=utf-8
def diff(func,x0):
    dx=1e-8
    return (func(x0+dx)-func(x0))/dx
def _diff(func,x0,y):
    dx=1e-8
    return (func(x0+dx,y)-func(x0,y))/dx
def __diff(func,x,y0):
    dx=1e-8
    return (func(x,y0+dx)-func(x,y0))/dx
def test(x):
    return (x-4)**2
def test2(x,y):
    return (x-4)**2+(y-2)**2
#一元函数的SlopeDesent
def slopeDescent(func,x0,a=0.01,n=1000):#提供函数接口,不设置自收敛条件
    x=x0
    for i in range(n):
        x=x-a*diff(func,x)
def GradientDescent(func,value,a=0.001,n=10000):#func是函数接口 value是自变量接口 必须以func(value)这种形式去给值
    #因为测试函数写的不好 所以value在这里以func(value[0],value[1])这种形式去给值
    #并且给出偏导数的求法_diff tensorflow有计算图这种东西进行反向传播很快 这里只是掩饰原理
    #可以写的更robust一点 把func接口写好一点 但是因为应用价值不高就没写
    x=value[0]
    y=value[1]
    for i in range(n):
        x=x-a*_diff(func,x,y)
        y=y-a*__diff(func,x,y)
    print(x,y)
GradientDescent(test2,[0.5,0.7])
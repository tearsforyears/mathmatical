#coding=utf-8
#import numpy as np
def f(*args):#函数随便改
    x,y,z=args
    return (x-5)**2+(y-3)**2+(z-4)**2
def partial_derivative(func,paraindex,*args):#指定对第几个参数求偏导数默认为0
    dx=1e-8
    arglist=list(args)
    arglist[paraindex]+=dx
    arglist=tuple(arglist)
    return (func(*arglist)-func(*args))/dx
def gradientDescent(f,turns,*args,a=0.01):#迭代次数,函数参数,a学习率
    argslis=list(args)
    for _ in range(turns):
        for i in range(len(argslis)):
            argslis[i]-=a*partial_derivative(f,i,*tuple(argslis))
    return argslis
if __name__ == '__main__':
    res=gradientDescent(f,10000,1,2,0)
    print(res)
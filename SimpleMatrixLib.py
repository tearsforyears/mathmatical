#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import decimal as d
import time
"""
@github :tearsforyears (now it is empty)
‎2018‎年‎6‎月‎1‎日 ‏‎ 13:13:38
2018年7月25日 09:27:49
For my 2 year in collage
A simple matrix libary depends on the Numerical Anylise
import numpy just use the data_construct array and np.poly(only on cuvee fitting)
Introduce
1.interpolation
2.Eigebvalue
3.LU QR decomposition
4.Equalstions linear non-linear
5.simple GradientDescent
"""
#测试函数 一些非数值分析支持函数
def cube(x):
    return x**3
def e_x(x):
    return np.e**x
def factorial(x):
    res=1.
    if x==0:return 1.
    for i in range(1,x+1):
        res*=i
    return res
def norm2(vec):
    return np.sqrt(np.sum(vec ** 2))
def e1(len):
    res=np.zeros(len)
    res[0]=1.
    return res.reshape((len,1))
def I(len):
    res=np.zeros((len,len))
    for i in range(len):
        res[i][i]=1.
    return res
#Basic Derivative and Integral
#这是数值计算最基础的东西,求导和积分,来自高等数学
def diff(func,x0,dx=1e-16):
    return (func(x0+dx)-func(x0))/dx
def diff_n(func,x0,n):
    #因为没用数值分析的技术所以高阶导数会有误差而且误差在3阶段就控制不住了
    dx = d.Decimal(1e-8)
    if n==0:
        return func(x0)
    return (diff_n(func,x0+dx,n-1)-diff_n(func,x0,n-1))/dx
def Integral(func,a,b):
    dx=1e-6
    n=int((b-a)/dx)#区间数
    res=0
    for i in range(n+1):
        res+=dx*func(a+dx*i)
    return res
#泰勒级数逼近,最慢也是精度最高的
#然而实际上并没有
#公式默认选用的是麦克劳林
def TaylorSeries_Approximate(func,x_predict,x0=0,iter=4):
    res=0.
    for i in range(iter):
        res+=(float(diff_n(func,x0,i))*(x_predict-x0)**i)/factorial(i)
    return res
#数值分析技术
#interpolation
def Lagrange_interpolation(x,y,x_predict):
    n=len(x)#获得插值点的个数
    y_head=[]
    for item in x_predict:#对预测点进行批量处理
        w0=1. #这tm脑子抽风了 之前有更快的公式 还以为是公式错了 妈蛋 这鸡儿玩意初始化不能为0 血的教训
        for i in range(n):
            w0*=(item-x[i])
        temp=0.
        for i in range(n):
            w1=1.
            for j in range(n):
                if i!=j:
                    w1*=(x[i]-x[j])
            temp+=(y[i]*w0)/((item-x[i])*w1)
        y_head.append(temp)
    return y_head
#newton_mean有两个实现版本v2对v1进行了小优化
#后面有些小地方没有进行优化
def newton_mean(x,y):#x y表示给定的点值对
    n=len(x)#或者写y随便
    mean_matrix=np.zeros((n,n))+np.nan#均值表
    mean_x=np.zeros(n)#对于x的参考表
    for i in range(n):
        mean_matrix[i][0]=y[i]#0阶段差分
        mean_x[i]=x[i]
    for i in range(1,n):
        for j in range(i,n):
            mean_matrix[j][i]=(mean_matrix[j][i-1]-mean_matrix[i-1][i-1])/(mean_x[j]-mean_x[i-1])
            #我的动态方程弄出来的矩阵和书上的有点不一样 但是对角线元素相同 无语了我
    tr=[]
    for i in range(n):
        tr.append(mean_matrix[i][i])
    return tr
def newton_mean2(x,y):#x y表示给定的点值对
    n=len(x)#或者写y随便
    line1=np.zeros(n)
    line2=np.zeros(n)
    mean_x=np.zeros(n)#对于x的参考表
    tr = []
    for i in range(n):
        line1[i]=y[i]#0阶段差分
        mean_x[i]=x[i]
    tr.append(line1[0])
    # mean_matrix[j][i]=(mean_matrix[j][i-1]-mean_matrix[i-1][i-1])/(mean_x[j]-mean_x[i-1])
    for i in range(1,n):
        for j in range(i,n):
            line2[j]=(line1[j]-line1[i-1])/(mean_x[j]-mean_x[i-1])
        tr.append(line2[i])
        temp=line1
        line1=line2
        line2=temp
    return tr
def Newton_interpolation(x,y,x_predict): #注意开发时候比较懒没有考虑到向量 x_predict只是一个单纯的标量
    para=newton_mean2(x,y)#得到那一串差分的值
    n=len(x)
    if n==1:
        return para[0]
    else:
        a=1
        an=[1]#这个1的妙用在0阶的差分
        for i in range(n):
            a*=(x_predict-x[i])#矩阵操作 借助numpy大法
            an.append(a)#动态规划思想 存储中间结果
        a=0#废物利用a存储求和结果
        for i in range(n):
            a+=(para[i]*an[i])
        return a
#迭代法解方程组
#选取对角矩阵提高求逆的速度
def solve_v1(A,b,turns=10):#基于Jacobi方法 SOR和Gauss-Seidel没写是因为不够numpy快
    l=A.shape[0]
    x=np.random.rand(l)
    print(x)
    M_I=np.zeros(shape=A.shape)
    M=np.zeros(shape=A.shape)
    for i in range(l):
        M_I[i][i]=1./A[i][i]
        M[i][i]=A[i][i]
    N=M-A
    B=np.matmul(M_I,N)
    f=np.matmul(M_I,b)
    for i in range(turns):
        x=np.matmul(B,x)+f
    return x
#matrix decomposition
def lu(matrix):
    #print(matrix)
    #print(matrix.shape)
    L=np.zeros((matrix.shape[0],matrix.shape[0]),dtype=np.float32)
    U=matrix
    for k in range(matrix.shape[0]):#k 表征std行 i表征目标行 start
        for j in range(matrix.shape[1]):#索引行x
            if U[k][j]!=0.:
                L[k:,k]=U[k:,j]/U[k][j]
                break
        #上面是建立矩阵L
        #下面是行阶梯的变化
        for i in range(k+1,matrix.shape[0]):#从本行开始每往下开始遍历
            start = 0
            for j in range(matrix.shape[1]):
                if U[k][j]!=0.:
                    start=j
                    break#计算非0数字开始的点
            if k>1 and start==0:
                for i in range(matrix.shape[0]):L[i][i]=1.#修正矩阵
                return L,U
            U[i][:]=U[i][:]-U[k][:]*(U[i][start])/U[k][start]#k 表征std行 i表征目标行 start
        #print(U,k)#得到各个阶段的行阶梯
    return L,U
#lu分解同时是获得行阶梯的一个方法 并且提供了线性方程组的简单解法
#lu分解就是行阶梯
def solve(A,b):#解方程Ax=b 仅处理解存在的情况 就是full rank的情况
    C=np.concatenate((A,b),axis=1)#C为ab的曾广矩阵
    _,U=lu(C)
    rank=C.shape[0]-1
    for k in range(rank,-1,-1):
        last_row = k
        last_col = k
        U[last_row]=U[last_row]/U[last_row][last_col]
        #处理最后一行
        for i in range(last_row,-1,-1):
            for j in range(i-1,-1,-1):
                if U[j][last_col]!=0.:
                    U[j]=U[j]-U[i]*U[j][last_col]
    #这时U为行最简 也就是说 我们确定了方程的解
    #print(U[:,(C.shape[1]-1)])
    return U[:,(C.shape[1]-1)]
#About Eigebvalue
#声明下 特征值的所有方法根据实矩阵去实现 鲁棒性比较差
def Main_Eigebvalue(A,u=np.array([1,1,1]).T,n=1000):
    for i in range(n):
        v=np.matmul(A,u)
        u=v/np.max(v)
    return np.max(v),u
def inverse_Eigebvalue(A,u=np.random.rand(3).T,n=1000):
    for i in range(n):
        v=solve_v1(A,u)
        u=v/np.max(v)
    return 1/np.max(v),u
#qr分解作为qr算法的核心 可以基于gram-schimidt givens
#householder变换去实现 这里基于后种速度比较快 A=QR
def reduce_shape(A):
    res=np.zeros(shape=(A.shape[0]-1,A.shape[1]-1))
    for i in range(1,A.shape[0]):
        for j in range(1,A.shape[1]):
            res[i-1][j-1]=A[i][j]
    return res
def add_shape(A):
    res=np.zeros(shape=(A.shape[0]+1,A.shape[1]+1))
    res[0][0]=1.
    for i in range(1,A.shape[0]+1):
        for j in range(1,A.shape[0]+1):
            res[i][j]=A[i-1][j-1]
    return res
def householder(A):#m*n=m*n@n*n
    a=np.reshape(A[0:,0],(A.shape[0],1))
    w=(a-norm2(a)*e1(a.shape[0]))/norm2(a-norm2(a)*e1(a.shape[0]))
    H=I(w.shape[0])-2*w*w.T
    return H
def qr(A):
    n=A.shape[1]
    H_STACK=[]
    for i in range(A.shape[1]-1):
        H=householder(A)
        if i!=0:H=add_shape(H)
        H_STACK.append(H)
        if i!=0:
            R=H@R#这里需要缓存下
        else:
            R=H@A
        A=reduce_shape(R)
    Q=I(n)
    for mat in H_STACK:
        Q=Q@mat
    return Q,R
def all_Eigebvalue(A,n=10):
    for i in range(n):
        Q,R=qr(A)
        A=R@Q
    return np.diagonal(A)
def binary_root(func,a,b):#前提fa<0 fb>0 解的方程需要化为 fx=0这种形式
    while True:
        if b-a<1e-8:
            return (a+b)/2
        if func((a + b)/2)*func(a)<0:
            b=(a + b)/2
        elif func((a + b)/2)*func(a)>1e-8:
            a=(a + b)/ 2
        else:
            return (a+b)/2
#不动点迭代相对比较简单 还要判断收敛性等问题 这里直接写不懂点迭代一个重要应用
#Steffensen迭代 注意函数接口 f(x)=0 要化成 fi(x)=x 这种不动点迭代方程
def Steffensen(fi,x0=0.5,n=10):
    arr=np.random.rand(3)
    arr[0]=x0
    arr[1]=fi(arr[0])
    arr[2]=fi(arr[1])
    for i in range(n):
        arr[0]=arr[0]-(arr[1]-arr[0]**2)/(arr[2]-2*arr[1]+arr[0])
        arr[1]=fi(arr[0])
        arr[2]=fi(arr[1])
    return arr[0]
def Newton(func,x0,n=100):
    x=x0
    for i in range(n):
        x=x-func(x)/diff(func,x,1e-8)
    return x
def Newton_v2(func,x0,n=100):
    x=x0
    for i in range(n):
        x=x-func(x)/diff(func,x0,1e-8)#减少了大量计算
    return x
#Legendre多项式和Chebyshev多项式的生成
#这两个多项式在最佳一致逼近和最佳平方逼近有很重要的应用
#没用动态规划进行优化 只是简单的递归
def Legendre_Polynomid(n):
    if n==0:
        return np.poly1d(np.array([1.]))
    if n==1:
        return np.poly1d(np.array([1.,0.]))
    return ((2*n-1)*np.poly1d(np.array([1.,0.]))*Legendre_Polynomid(n-1)-(n-1)*Legendre_Polynomid(n-2))/n
def least_square_approximation(func,n):#默认求-1 1的最佳平方逼近
    n+=1#为了不反人类
    dx=1e-6#节省计算速度
    num=int((2.)/dx)#区间数
    resPoly=[]
    for j in range(n):
        res=0
        poly=Legendre_Polynomid(j)
        for i in range(num+1):
            vx=-1+dx*i
            res+=dx*func(vx)*poly(vx)
        resPoly.append(res*(2*j+1)/2)
    res=0
    for i in range(len(resPoly)):
        res+=Legendre_Polynomid(i)*resPoly[i]
    return res
def Chebyshev_Polynomid(n):
    if n == 0:
        return np.poly1d(np.array([1.]))
    if n == 1:
        return np.poly1d(np.array([1., 0.]))
    return 2*np.poly1d(np.array([1., 0.]))*Chebyshev_Polynomid(n-1)-Chebyshev_Polynomid(n-2)
#print(Chebyshev_Polynomid(6))
def Chebyshev_Polynomid_Monoic(n):
    return Chebyshev_Polynomid(n)/2**(n-1)
#关于GradientDescent相关技术演示
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
#GradientDescent(test2,[0.5,0.7])
#高级语法实现多元函数的gradientDescent 2018年9月10日 09:35:03
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
#print(gradientDescent(f,10000,0,0,0))
'''
特征值测试
if __name__=="__main__":
    n = 10000
    A = np.array([[0, 3, 1], [0, 4, -2], [2, 1, 1]])
    tic=time.time()
    for i in range(n):qr(A)
    tok=time.time()
    print("i used",tok-tic,"ms")
    tic = time.time()
    for i in range(n):np.linalg.qr(A)
    tok = time.time()
    print("numpy used",tok - tic,"ms")
'''
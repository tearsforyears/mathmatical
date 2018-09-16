#coding=utf-8
"""以下均对维数较低的公式进行处理 多变量问题暂不考虑"""

"""带噪声数据生成"""
#在begin到end生成num个点 类似numpy的linspace带噪声的点 rand 表示噪声的大小 正常选1 不正常选0表示无噪声
import numpy as np
import matplotlib.pyplot as plt
import time
def noisy(f,point,rand):
    rd=np.random.rand()
    if rd<0.5:
        return f(point)+rand*np.random.rand() #产生小幅度的噪声
    else:
        return f(point)-rand*np.random.rand()  #懒得考虑其他复杂逻辑了

def getdata(f,begin=0,end=10,num=100,rand=0.01):
    x_lis=np.linspace(begin,end,num)
    y_lis=[]
    for item in x_lis:
        y_lis.append(noisy(f,item,rand))
    return (x_lis,y_lis)

def data_print(func):
    data=getdata(func)
    x=data[0]
    y=data[1]
    plt.plot(x,y,'b-')
    plt.plot(x,func(x),'r-')
    plt.show()
#data_print(np.cos)

"""拉格朗日插值(基本不用numpy api可以实现)"""
#此处说明下 x0 y0 x——predict 分别表示 已知点的列表 x_predict表示需要预测点的列表 用的是拉格朗日的曲线
#ln(x)=∑yi*li li=π(x-xi)/(xi-xj)

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


#把拉格朗日插值的信息打印到控制台
def Lagrange_print_console(predict_points,x=np.linspace(0,10,100),y=np.sin(np.linspace(0,10,100))):
    print("-------------------------------true value--------------------------")
    print([np.sin(x) for x in predict_points])
    print("-------------------value by Lagrange_interpolation-----------------")
    print(Lagrange_interpolation(x,y,x_predict=predict_points))
#Lagrange_print_console([1,2,3,4,5,6])

def Lagrange_print_matplotlib(predict_points,x=np.linspace(0,10,100),y=np.sin(np.linspace(0,10,100))):
    Lagrange_print_console(predict_points)
    plt.title("the blue for true,the red for pridect with Lagrange_interpolation")
    plt.plot(x,y,'b-')
    plt.plot([i for i in predict_points],Lagrange_interpolation(x,y,x_predict=predict_points),'ro')
    plt.show()
#Lagrange_print_matplotlib(np.linspace(1,9,20))


#用拉格朗日去拟合有噪音的曲线
def cube(x):
    return x**3
def app():
    data=getdata(cube,0,1,200)
    x=data[0]
    y=data[1]
    plt.plot(x,x**3,'g-')
    plt.plot(x,y,'b-')
    y_head=Lagrange_interpolation(x=x,y=y,x_predict=np.linspace(0.1,0.9,50))
    for i in range(len(y_head)):
        if y_head[i]<=0 or y_head[i]>=2 or y_head[i]==np.nan:
            y_head[i]=0
    print(y_head)
    plt.plot(np.linspace(0.1,0.9,50),y_head, 'r-')
    plt.show()
app()
#很明显的 对于有噪音的情况 拉格朗日插值并不好用


"""牛顿差分插值"""
'''差分这tm在高中算死的玩意终于让我用计算机给实现了mlgb 下面利用动态规划(暂时不优化空间复杂度)实现'''
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

#数据源自于课本 既然那啥矩阵都不一样 但是结果能出来 我tm还是用空间优化吧
#print(newton_mean(np.array([0.40,0.55,0.65,0.80,0.90,1.05]),np.array([0.41075,0.57815,0.69675,0.88811,1.02652,1.25382])))

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
'''
tic=time.time()
print(newton_mean2(np.array([0.40,0.55,0.65,0.80,0.90,1.05]),np.array([0.41075,0.57815,0.69675,0.88811,1.02652,1.25382])))
tok=time.time()
print(tok-tic,"ms")
tic=time.time()
print(newton_mean(np.array([0.40,0.55,0.65,0.80,0.90,1.05]),np.array([0.41075,0.57815,0.69675,0.88811,1.02652,1.25382])))
tok=time.time()
print(tok-tic,"ms")
'''
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

#tic=time.time()
#print(Newton_interpolation(np.array([0.40,0.55,0.65,0.80,0.90,1.05]),np.array([0.41075,0.57815,0.69675,0.88811,1.02652,1.25382]),2.1))
#print(Newton_interpolation(np.array([0.40,0.55,0.65,0.80,0.90,1.05]),np.array([0.41075,0.57815,0.69675,0.88811,1.02652,1.25382]),0.596))
#tok=time.time()
#print((tok-tic),"ms")

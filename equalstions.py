#coding=utf-8
import numpy as np
A=np.array([[8,-3,2],[4,11,-1],[6,3,12]])
b=np.array([20,33,36])
#solve the equalstion of Ax=b only the square
#选取对角矩阵提高求逆的速度
def solve_v1(A,b,turns=10):
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
    print(x)
    return x
solve_v1(A=A,b=b)#解决很快 精度比较高了

#下面的编程得用到动态规划 有空在写
#2018年6月18日 23:45:03
#考完期末考试和n2再写吧
def Gaussian_Seidel(A,b,turns=10):
    l=A.shape[0]
    x=np.random.rand(l)
    #
    print(x)
    return x
#successive over relaxation method
def SOR(A,b,w,turns=10):
    pass
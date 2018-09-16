#coding=utf-8
import numpy as np
def solve_v1(A,b,turns=100):
    l=A.shape[0]
    x=np.random.rand(l)
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
#主特征值计算 不优化 计算速度足够快
def Main_Eigebvalue(A,u=np.random.rand(3),n=1000):
    for i in range(n):
        v=np.matmul(A,u)
        u=v/np.max(v)
    return np.max(v),u
A=np.array([[1.,1.,.5],[1.,1.,.25],[.5,.25,2.]])
print(Main_Eigebvalue(A))
def inverse_Eigebvalue(A,u=np.random.rand(3).T,n=1000):
    for i in range(n):
        v=solve_v1(A,u)
        u=v/np.max(v)
    return 1/np.max(v),u
B=np.array([[2,1,0],[1,3,1],[0,1,4]])
print(inverse_Eigebvalue(B))

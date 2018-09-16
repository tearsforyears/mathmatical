#coding=utf-8
import numpy as np
#import numpy 只是为了借用numpy数组 并没有用到本身的方法
#拿python 干数值分析才知道自己有很多很多的不足
#这个矩阵库没有的方法 矩阵的加减乘 det rank 全部使用numpy
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
#lu分解同时是获得行阶梯的一个方法
'''
B=np.array([[3,-7,-2],[-3,5,1],[6,-4,0]])
C=np.array([[2,-4,-2,3],[6,-9,-5,8],[2,-7,-3,9],[4,-2,-2,-1],[-6,3,3,4]])
L,U=lu(C)
print(L,U)
'''
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
'''
A=np.array([[1,1,1],[0,4,-1],[2,-2,1]])
b=np.array([[6,5,1]]).reshape(3,1)
#solve(A,b)
C=np.array([[10,-7,0,1],[-3,2.09,6,2],[5,-1,5,-1],[2,1,0,2]])
d=np.array([[8,5.9,5,1]]).reshape(4,1)
solve(C,d)
print("------------------")
print(np.linalg.solve(C,d))
#print(np.linalg.eig(np.matmul(A,A.T))[0])#返回两个数组 一个是特征值数组 一个是特征向量数组
#print(np.linalg.eig(np.matmul(A,A.T))[1])
#测试结果和numpy一样
'''

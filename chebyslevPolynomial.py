#coding=utf-8
#关于numpy 多项式 api的应用
import numpy as np
p = np.array([1.0,0,-2,1])
p = np.poly1d(p)
#print(p) #生成多项式
#print(np.polyval(p, 5))
#print(p*2) #多项式可以直接乘积
#print(p.deriv(4))#m阶微分 derivative
#print(p.integ())#m阶积分+c intergration

#Legendre Polynomid
def Legendre_Polynomid(n):
    if n==0:
        return np.poly1d(np.array([1.]))
    if n==1:
        return np.poly1d(np.array([1.,0.]))
    return ((2*n-1)*np.poly1d(np.array([1.,0.]))*Legendre_Polynomid(n-1)+(n-1)*Legendre_Polynomid(n-2))/n
#print(Legendre_Polynomid(4))
def Chebyshev_Polynomid(n):
    if n == 0:
        return np.poly1d(np.array([1.]))
    if n == 1:
        return np.poly1d(np.array([1., 0.]))
    return 2*np.poly1d(np.array([1., 0.]))*Chebyshev_Polynomid(n-1)-Chebyshev_Polynomid(n-2)
#print(Chebyshev_Polynomid(6))
def Chebyshev_Polynomid_Monoic(n):
    return Chebyshev_Polynomid(n)/2**(n-1)
print(Chebyshev_Polynomid(6))
print(Chebyshev_Polynomid_Monoic(6))
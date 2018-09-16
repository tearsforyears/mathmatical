#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,1.5,100)+1e-8
y=x**x
#plt.plot(x,x,'b-')
plt.plot(x,y,'r.')
plt.show()
'''
from sympy import *
x=Symbol('x')
res=diff(x**x**x)
print(res)
'''
# p32.py - solve u_xx = exp(4x), u(-1)=0 u(1)=1 (compare with p13.py)
from chebPy import *
import numpy as np
from scipy.linalg import solve
from matplotlib import pyplot as plt

N=16
D,x=cheb(N)
D2=np.dot(D,D)
D2=D2[1:N,1:N]
f=np.exp(4.*x[1:N])
u=solve(D2,f)
s=np.zeros(N+1)
s[0]=0.0
s[N]=0.0
s[1:N]=u
s = s + (x+1.)/2. # Correction for BCs

exact=(np.exp(4.*x)-np.sinh(4.)*x-np.cosh(4))/16.+(x+1.)/2.

maxerr=np.abs(s-exact).max()

plt.title('max err = %e' % maxerr)
plt.plot(x,s,'b',x,exact,'o')
plt.show()


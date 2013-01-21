# p33.py - solve u_xx = exp(4x), u'(-1)=0 u(1)=1
from chebPy import *
import numpy as np
from scipy.linalg import solve
from matplotlib import pyplot as plt

N=16

# System matrix
D,x=cheb(N)
D2=np.dot(D,D)
D2[N,:] = D[N,:]       # New equation involving first derivative of unknown function 
D2=D2[1:N+1,1:N+1]     # (hence D instead of D2) - the Neumann BC at x=-1

# RHS
f=np.zeros(N)
f[1:N]=np.exp(4.*x[1:N])
f=f[::-1]              # reverse it

u=np.zeros(N+1)        # initialize
u[1:N+1]=solve(D2,f)   # solve
u=u[::-1]              #reverse this one too

exact=(np.exp(4.*x)-4.*np.exp(-4.)*(x-1.)-np.exp(4.))/16. # exact sol.

maxerr=np.abs(u-exact).max() # max. error

plt.title('max err = %e' % maxerr)
plt.plot(x,u,'b',x,exact,'o')
plt.show()


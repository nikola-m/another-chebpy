# p33.py - solve u_xx = exp(4x), u'(-1)=0 u(1)=1
# Unlike in N.Trefethen's example we remove first equation
# that is first row, from D2 and replace it with the first row of D.
from chebPy import *
import numpy as np
from scipy.linalg import solve
from matplotlib import pyplot as plt

N=24
D,x=cheb(N)
D2=np.dot(D,D)

D2[0,:] = D[0,:]     # New equation involving first derivative of unknown function 
                     # (hence D instead of D2) - the Neumann BC at x=-1
D2=D2[0:N,0:N]
f=np.zeros(N)
f[1:N]=np.exp(4.*x[1:N])
u=solve(D2,f)
s=np.zeros(N+1)
s[0:N]=u

exact=(np.exp(4.*x)-4.*np.exp(-4.)*(x-1.)-np.exp(4.))/16.

maxerr=np.abs(s-exact).max()

plt.title('max err = %e' % maxerr)
plt.plot(x,s,'b',x,exact,'o')
plt.show()


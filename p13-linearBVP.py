# p13 - solve linear BVP u_xx = exp(4x), u(-1)=u(1)=0
#
# Note: for N=16, maxerr = 1.18789283965e-11; In Trefethen's book max err = 1.261e-10
#
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

exact=(np.exp(4.*x)-np.sinh(4.)*x-np.cosh(4))/16.
maxerr=np.max(s-exact)

xx=np.linspace(-1.,1.,100)
uu = np.polyval(np.polyfit(x,s,N),xx)    # interpolate grid data

plt.title('max err = %e' % maxerr)
plt.plot(xx,uu,'b',x,exact,'o')
plt.show()


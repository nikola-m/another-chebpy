# p38 - solve u_xxxx = exp(x), u(-1)=u(1)=u'(-1)=u'(1)=0
#       compare with p13
#
from chebdif import *
import numpy as np
from scipy.linalg import solve
from matplotlib import pyplot as plt

# Construct dscrete biharmonic operator:
N = 15
x,DM = chebdif(N+1,4) # Weideman&Reddy function for differentiation matrix
v = np.zeros(N+1) # Work array
v[1:N] = 1./(1.-x[1:N]**2)
S = np.diag(v)
D2 = DM[1,:,:]  # Python counting starts from zero...
D3 = DM[2,:,:]
D4 = DM[3,:,:]
print D4.shape
D4 = np.dot((np.dot(np.diag(1.-x**2),D4) - 8*np.dot(np.diag(x),D3) - 12*D2),S)
D4 = D4[1:N,1:N]

# Solve boundary-value problem and plot results:
f=np.exp(x[1:N])
u=solve(D4,f)
u1=np.zeros(N+1)
u1[0]=0.0
u1[N]=0.0
u1[1:N]=u


# Exact solution and max err:
A = np.array([ [1,-1,1,-1],
               [0,1,-2,3],
               [1,1,1,1],
               [0,1,2,3] ])
V = np.vander(x)
V = V[:,N-3:N+1]
V = (V.T[::-1]).T # transpose  - reverse up-down - transpose
v = np.array([-1,-1,1,1])
c = solve(A,np.exp(v))
exact=(np.exp(x)-np.dot(V,c))
maxerr=np.max(u1-exact)

# Interpolate to more grid points
xx=np.linspace(-1.,1.,100)
uu = (1.-xx**2)*np.polyval(np.polyfit(x,np.dot(S,u1),N),xx)

plt.title('max err = %e' % maxerr)
plt.plot(xx,uu,'b',x,exact,'o')
plt.show()


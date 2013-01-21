# p14 - solve nonlinear BVP U_xx = exp(u), u(-1)=u(1)=0 by iteration.
import numpy as np
from chebPy import cheb
from scipy.linalg import solve
from matplotlib import pyplot as plt

N=16
D,x=cheb(N)
D2=np.dot(D,D)
D2=D2[1:N,1:N]

u=np.zeros(N-1)
err=np.zeros(N-1)
change = 0.1 # arbitrary small number
it = 0

while True:
  unew=solve(D2,np.exp(u))
  err = np.abs(unew-u)
  change = err.max()
  u = unew
  it += 1
  if(change < 1e-15):
    break

s=np.zeros(N+1)
s[0]=0.0
s[N]=0.0
s[1:N]=u

xx=np.linspace(-1.,1.,100)
uu = np.polyval(np.polyfit(x,s,N),xx)    # interpolate grid data

plt.title('no. steps = %d,  u(0) = %18.14f' %(it,u[N/2-1]) ) # u(0) = u[N/2-1] - For this to be correct N has to be even!
plt.plot(xx,uu,'b')
plt.show()


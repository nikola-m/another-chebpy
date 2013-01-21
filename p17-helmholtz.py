# p17 - Helmholtz eq. u_xx + u_yy + (k^2)u = f
#       on [-1,1]x[-1,1]
from chebPy import *
import numpy as np
from scipy.linalg import solve
from matplotlib import pyplot as plt

import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

N=32
D,x=cheb(N)
y=x # tensor product grid
xx,yy = np.meshgrid(x[1:N], y[1:N])
xx=xx.flatten(1)
yy=yy.flatten(1)

f=np.exp(-10.*((yy-1.)**2+(xx-0.5)**2))
D2=np.dot(D,D)
D2=D2[1:N,1:N]
I=np.eye(N-1)
k=9.
L=np.kron(I,D2)+np.kron(D2,I)+k**2*np.eye((N-1)**2)# Helmholtz operator

u=solve(L,f)  # Solve system

# Reshape long 1D results to 2D grid:
uu=np.zeros((N+1,N+1))
uu[1:N,1:N] = u.reshape(N-1,N-1).transpose()
xx,yy = np.meshgrid(x,y)
value = uu[N/2,N/2]

# Plot results
# wireframe:
fig=p.figure()
ax = p3.Axes3D(fig)
#ax.contourf(xx,yy,uu)
ax.plot_wireframe(xx,yy,uu)
ax.plot_surface(xx,yy,uu,rstride=1, cstride=1, cmap=cm.YlGnBu,
        linewidth=1, antialiased=False)
#ax.plot_surface(X,Y,u,rstride=1, cstride=1,alpha=0.5)
ax.text(-0.8,0.5,.022,'u(0,0) = %13.11f' % value)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
p.show()


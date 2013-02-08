# p17 - Helmholtz eq. u_xx + u_yy + (k^2)u = f
#       on [-1,1]x[-1,1]
from chebPy import *
import numpy as np
from scipy.linalg import solve
from scipy.interpolate import interp2d
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

# Interpolate to finer grid and plot
uu = uu.flatten(1)
uuu = interp2d(x,y,uu,kind='cubic')

a = np.linspace(-1.,1.,100)
b = np.linspace(-1.,1.,100)
xxx,yyy = np.meshgrid(a,b)
newfunc = uuu(a,b)

# Plot results
# wireframe:
fig=p.figure()
ax = p3.Axes3D(fig)
ax.plot_wireframe(xxx,yyy,newfunc)
ax.plot_surface(xxx,yyy,newfunc,rstride=1, cstride=1, cmap=cm.YlGnBu,
        linewidth=1, antialiased=False)
ax.text(-0.8,0.5,.022,'u(0,0) = %13.11f' % value)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
p.show()

# mlba plotting
from enthought.mayavi import mlab
mlab.surf(a,b,newfunc)
mlab.show()

# p16 - Poisson eq. on [-1,1]x[-1,1] with u=0 on boundary
from chebPy import *
import numpy as np
from scipy.linalg import solve
from scipy.interpolate import interp2d
from matplotlib import pyplot as plt

import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

N=32                          # If you use cubic interpolant and N<32 there will be some oscilations, which are not there in solution.                     
D,x=cheb(N)
y=x                           # tensor product grid
xx,yy = np.meshgrid(x[1:N], y[1:N])
xx=xx.flatten(1)
yy=yy.flatten(1)

f=10.*np.sin(8.*xx*(yy-1))
D2=np.dot(D,D)
D2=D2[1:N,1:N]
I=np.eye(N-1)
L=np.kron(I,D2)+np.kron(D2,I) # tensor product Laplacian

#sparsity pattern of L
#plt.title('Sparsity pattern of discretization matrix')
#plt.spy(L, precision=1e-15, marker='s', markersize=1)
#plt.show()

u=solve(L,f)  # Solve system

# Reshape long 1D results to 2D grid:
uu=np.zeros((N+1,N+1))
uu[1:N,1:N] = u.reshape(N-1,N-1).transpose()
xx,yy = np.meshgrid(x,y)
# Interpolate to finer grid and plot
uu = uu.flatten(1)
uuu = interp2d(x,y,uu,kind='cubic')

a = np.linspace(-1.,1.,100)
b = np.linspace(-1.,1.,100)
xxx,yyy = np.meshgrid(a,b)
newfunc = uuu(a,b)

# Test value
value = uuu(2.**(-1./2.),2.**(-1./2.))

# Plot results
fig=p.figure()
ax = p3.Axes3D(fig)
#ax.plot_wireframe(xxx,yyy,newfunc)
ax.plot_surface(xxx,yyy,newfunc,rstride=1, cstride=1, cmap=cm.YlGnBu,
        linewidth=0, antialiased=False)
ax.text(-1.1,-.8,-.5,'u($2^{-1/2}$,$2^{-1/2}$) = %14.11f' % value)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
p.show()

# If you have it installed:
from enthought.mayavi import mlab
mlab.surf(a,b,newfunc)
mlab.show()
#mlab.savefig('p16-mlab.png')

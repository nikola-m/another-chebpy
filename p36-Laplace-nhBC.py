# p36.py - Laplace eq. on [-1,1]x[-1,1] with nonzero BC's
from chebPy import *
import numpy as np
from scipy.linalg import solve
from scipy.interpolate import interp2d

import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

N=24
D,x=cheb(N)
y=x                           # tensor product grid
xx,yy = np.meshgrid(x, y)     # xx,yy = np.meshgrid(x[1:N], y[1:N])
xx=xx.flatten(1)
yy=yy.flatten(1)

D2=np.dot(D,D)                #D2=D2[1:N,1:N]
I=np.eye(N+1)                 # cf. p16 where I=np.eye(N-1)
L=np.kron(I,D2)+np.kron(D2,I) # Laplacian

# Impose boundary conditions by replacing appropriate rows of L:
b1 = np.where(abs(xx)==1.)[0]
b2 = np.where(abs(yy)==1.)[0]
b=np.append(b1,b2)

# Prepare matrix for implementatiopn of BCs;
# rows pertinent to collocation points belonging to boundary 
# (their indices are collected in array b), are set equal to zero, exept main diagonal entries which is set to one.
for i in range(b.size):
    L[b[i],:] = 0. 
    for j in range(b.size):
        if (b[i]==b[j]):
            L[b[i],b[j]] = 1.  
        else:
            L[b[i],b[j]] = 0.

f = np.zeros((N+1)**2)
for i in range(b.size):
    bdi=b[i]
    f[bdi] = int(yy[bdi]==1.)*int(xx[bdi]<0.)*np.sin(np.pi*xx[bdi])**4 +.2*int(xx[bdi]==1.)*np.sin(3.*np.pi*yy[bdi])

u=solve(L,f)  # Solve system

# Reshape long 1D results to 2D grid:
uu=np.zeros((N+1,N+1))
uu = u.reshape(N+1,N+1).transpose()
xx,yy = np.meshgrid(x,y)

# Test value
value = uu[N/2,N/2]

# Plot results
fig=p.figure()
ax = p3.Axes3D(fig)
# wireframe:
ax.plot_wireframe(xx,yy,uu)
# surface 
#ax.plot_surface(xx,yy,uu,rstride=1, cstride=1, cmap=cm.spectral,
#        linewidth=0, antialiased=False)
# surface
#ax.plot_surface(xx,yy,uu,rstride=1, cstride=1,alpha=0.5)
ax.text(-1.1,-.8,-.5,'u(0,0) = %14.11f' % value)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
p.show()


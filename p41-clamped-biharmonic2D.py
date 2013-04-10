# 
# Biharmonic eq. on [-1,1]x[-1,1] with clamped boundary conditions.
#
from chebdif import *
import numpy as np
from scipy.linalg import solve,inv
from scipy.interpolate import interp2d
from matplotlib import pyplot as plt

from math import log10

#import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

from time import time

start = time()

N=18                 

x,DM = chebdif(N+1,4)
D2 = DM[1,:,:]  # Python counting starts from zero...
D3 = DM[2,:,:]
D4 = DM[3,:,:]


y=x  # tensor product grid
xx,yy = np.meshgrid(x[1:N], y[1:N])
xx=xx.flatten(1)
yy=yy.flatten(1)

# RHS vector:
pi=np.pi
f=4.*np.cos(pi*xx)*np.cos(pi*yy)+np.cos(pi*xx)+np.cos(pi*yy)

v = np.zeros(N+1) # Work array
v[1:N] = 1./(1.-x[1:N]**2)
S = np.diag(v)
D4 = np.dot((np.dot(np.diag(1.-x**2),D4) - 8*np.dot(np.diag(x),D3) - 12*D2),S)
D4 = D4[1:N,1:N]; D2=D2[1:N,1:N]; I=np.eye((N-1))
L=np.kron(I,D4)+np.kron(D4,I)+2*np.dot(np.kron(D2,I),np.kron(I,D2))

u=solve(L,f) # Solve system
elapsed = (time() - start)
print 'Solved!'

# Interpolate to finer grid for plotting
uu=np.zeros((N+1,N+1))
uu[1:N,1:N] = u.reshape(N-1,N-1)
xx,yy = np.meshgrid(x,y)
xx=xx.flatten(1)
yy=yy.flatten(1)
uu=uu.flatten(1)

# Interpolate to finer grid and plot
uuu = interp2d(x,y,uu,kind='cubic')

a = np.linspace(-1.,1.,100)
b = np.linspace(-1.,1.,100)
xxx,yyy = np.meshgrid(a,b)
newfunc = uuu(a,b)

# Exact solution
exact=1./pi**4*(1.+np.cos(pi*xx))*(1.+np.cos(pi*yy))

# Exact solution and Error
maxerr=log10(max(abs(uu-exact)))
print N,maxerr,elapsed

# Prepare for plotting
uu=np.reshape(uu,(N+1,N+1))
xx=np.reshape(xx,(N+1,N+1))
yy=np.reshape(yy,(N+1,N+1))
exact=np.reshape(exact,(N+1,N+1))

# Plot results
fig=plt.figure()
ax = p3.Axes3D(fig)
plt.annotate('n = %d, $log_{10}$(max err) = %f, time = %f [s]' % (N,maxerr,elapsed), xy=(0.05, 0.95), xycoords='axes fraction')
ax.plot_wireframe(xxx,yyy,newfunc,color='black',linewidth=0.2)
ax.plot_surface(xxx, yyy, newfunc, rstride=1, cstride=1, cmap=cm.cool, linewidth=0.1, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.show()


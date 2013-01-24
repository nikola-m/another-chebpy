# p34.py - Allen-Cahn eq. u_t = u_xx+u-u^3, u(-1)=-1, u(1)=1
from chebPy import cheb
import numpy as np
from scipy.linalg import solve

import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

# Differentiation matrix and initial data:
N = 20
D,x = cheb(N)
D2 = np.dot(D,D)   # Use full size matrix
D2[0,:] = 0.       # for convenience
D2[N,:] = 0.

eps = 0.01
dt = min(.01,50.*N**(-4.)/eps)
t = 0.
v = .53*x + .47*np.sin(-1.5*np.pi*x)

# Solve PDE by Euler formula and plot results:
tmax = 100.
tplot = 2
nplots = int(round(tmax/tplot))
plotgap = int(round(tplot/dt))
dt = float(tplot)/float(plotgap)
xx = np.linspace(-1.,1.,40)
vv = np.polyval(np.polyfit(x,v,N),xx)
plotdata = np.zeros((nplots+1,xx.size))
plotdata[0,:] = vv
tdata = np.zeros(nplots+1)
tdata[0] = t

for i in range(nplots):
    for n in range(plotgap):
        t = t + dt
        v = v + dt*(eps*np.dot(D2,(v-x))+v-v**3)      # Euler step
#        v[N] = 1.+np.sin(t/5.)**2                 # BC 
#        v[0] = -1.  
# NOTE: V[N] and V[0] come in different order than in Trefethen's MATLAB script p35.m

    vv = np.polyval(np.polyfit(x,v,N),xx)
    plotdata[i+1,:] = vv
    tdata[i+1] = t

xxx,yyy = np.meshgrid(xx,tdata)

# Plot results
fig=p.figure()
ax = p3.Axes3D(fig)
ax.view_init(30,-150) 
ax.plot_wireframe(xxx,yyy,plotdata)
#ax.plot_surface(tdata,xx,plotdata,rstride=1, cstride=1, cmap=cm.hsv,
#        linewidth=0, antialiased=False)
#ax.plot_surface(xxx,yyy,plotdata,rstride=1, cstride=1,cmap=cm.Set3)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
p.show()

# If you have it installed:
#from enthought.mayavi import mlab
#mlab.mesh(xxx,yyy,plotdata)
#mlab.show()


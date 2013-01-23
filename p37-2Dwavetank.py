# p37 - 2D 'wave tank' with Neumann BCs for |y|=1
from chebPy import cheb
import numpy as np
from scipy.linalg import solve,toeplitz

from matplotlib import pyplot as plt
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

# x variable in [-A,A], Fourier - trigonometric series:
A = 3.
Nx = 50
dx = 2*A/float(Nx)
x = -A + dx*np.arange(1,Nx+1)
# Create entries for Derivative matrix - a Toeplitz matrix with an entry from 'entries' on each diagonal.
entries = np.zeros(Nx)
entries[0] = -1/(3*(dx/A)**2)-1/6.
entries[1:Nx] = .5*(-1)**np.arange(2,Nx+1)/np.sin((np.pi*dx/A)*np.arange(1,Nx)/2)**2
# Form derivative matrix as a Toeplitz matrix using 'entries'
D2x = (np.pi/A)**2*toeplitz(entries)

# y variable in [-1,1], Chebyshev:
Ny =15
Dy,y = cheb(Ny)
D2y = np.dot(Dy,Dy)

# Prepare boundary conditions
Abc = - np.array([ [Dy[0,0],Dy[0,Ny]],
                   [Dy[Ny,0],Dy[Ny,Ny]] ])
BC = np.zeros((2,Ny-1))
Dybc = np.zeros((2,Ny-1))
Dybc[0,0:Ny-1] = Dy[0,1:Ny]
Dybc[1,0:Ny-1] = Dy[Ny,1:Ny]
BC[:,0:Ny-1] = solve(Abc,Dybc[:,0:Ny-1])

# Grid and initial data:
xx,yy = np.meshgrid(x,y)
vv = np.exp(-8.*((xx+1.5)**2+yy**2))
dt = 5./float(Nx+Ny**2)
vvold = np.exp(-8.*((xx+dt+1.5)**2+yy**2))

# Time-stepping by leap-frog formula
plotgap = int(round(2/dt))
dt = 2./plotgap
for n in range(2*plotgap+1):
    t = n*dt

    if ( (n+0.5)%plotgap<1. ):
        # Plot

#       If Matplotlib version >= 1.0.0
#        fig=plt.figure()
#        # Add subplot
#        ax = fig.add_subplot(3, 1, n/plotgap+1, projection='3d')
#       else
        fig=p.figure(figsize=plt.figaspect(0.5))
        ax = p3.Axes3D(fig)

        ax.view_init(15,-100) 
        ax.text(-2.5,1.,.5,'t = %f' % t)
        ax.plot_surface(xx,yy,vv,rstride=1, cstride=1,alpha=0.5)
        ax.set_zlim3d(0., 4.0)

    vnew = 2*vv - vvold +dt**2*(np.dot(vv,D2x) + np.dot(D2y,vv))
    vvold = vv
    vv = vnew
    vv[0,:]  = np.dot(BC[0,:],vv[1:Ny,:])    # Neumann BC for |y|=1
    vv[Ny,:] = np.dot(BC[1,:],vv[1:Ny,:])

plt.show()

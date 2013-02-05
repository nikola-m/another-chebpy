# p15 - solve eigenvalue BVP u_xx = lambda8u, u(-1)=u(1)=0
import numpy as np
from chebPy import cheb
from scipy.linalg import solve, eig
from matplotlib import pyplot as plt

N=36
D,x=cheb(N)
D2=np.dot(D,D)
D2=D2[1:N,1:N]

# Get eigenvalues and right-eigenvectors using scipy.linalg.eig
# Notice: lam,V in reverse order that in Matlab where it's [V,Lam]=eig(D2)
# Also notice the difference In Matlab we get diagonal matrix lam, and here an numpy array lam!!!
lam,V=eig(D2)   
ii = np.argsort(-lam)          # sort eigenvalues
lam=lam[ii]
V=V[:,ii]

fig = plt.figure()
#fig, axes = plt.subplots(nrows=6, ncols=1)
#fig.tight_layout()
eigs=np.linspace(5,30,6)       # plot 6 eigenvectors
for j in eigs:              
    lv = np.shape(V)[0]+2
    u = np.zeros(lv)
    u[1:lv-1] = V[:,int(j)]  
    ax1 = fig.add_subplot(6,1,j/5)
    ax1.plot(x,u,'bo')
    xx=np.linspace(-1.,1.,100)
    uu = np.polyval(np.polyfit(x,u,N),xx)    # interpolate grid data
    ax1.text(-0.4,0.3,'eig %d = %20.13f * 4/$pi^2$' %(j,lam[j-1]*4/np.pi**2) )
    ax1.text(0.7,0.3,'%4.1f ppw' % (4*N/(np.pi*j)) )
    ax1.plot(xx,uu,'b')

plt.show()

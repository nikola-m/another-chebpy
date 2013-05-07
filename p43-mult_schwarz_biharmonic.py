#
# Multiplicative Schwarz Domain Decomposition solution of Biharmonic eq.
# Original Finite Difference code for Poisson eq. written by Greg von Wickel
# Check out http://www.scientificpython.net/1/post/2012/12/multiplicative-schwarzdomain-decomposition.html
# Chebyshev collocation by Nikola Mirkov.
#
import numpy as np
from scipy.sparse import csr_matrix, spdiags, kron, identity
from scipy.sparse.linalg import spsolve 
from chebdif import *
from math import log10

def neglap2d(n):
    """
       Returns the Chebyshev collocation approximation to -D_x^2-D_y^2 on
       the square [-1,1]^2 on the interior points only
    """

    x,DM = chebdif(n+2,4)
    D2 = DM[1,:,:]  # Python counting starts from zero...
    D3 = DM[2,:,:]
    D4 = DM[3,:,:]  

    v = np.zeros(n+2) 
    v[1:-1] = 1./(1.-x[1:-1]**2)
    S = np.diag(v)
    D4 = np.dot((np.dot(np.diag(1.-x**2),D4) - 8*np.dot(np.diag(x),D3) - 12*D2),S)
    D4 = D4[1:-1,1:-1]
    D2=D2[1:-1,1:-1]
    I = identity(n)
    L = csr_matrix(kron(I,D4)+kron(D4,I)+2*kron(D2,D2)) # Like this, or...
    #L=csr_matrix(kron(I,D4)+kron(D4,I)+2*np.dot(kron(D2,I),kron(I,D2))) # ...like this.
    return L 


class mult_schwarz(object):
    """
        Class for solving the 2D Poisson equation on the unit square 
        using Chebyshev collocation and multaplicative Schwarz domain 
        decomposition
    """
  
    def __init__(self,bksize,numbk):
   
        n = 2*bksize*numbk      # Degrees of freedom per dimension  
        
        self.L = neglap2d(n)

        T = spdiags(np.ones((2,n)),np.array([0,1]),n,n) # Matrix for overlaps
    
        Id = np.identity(4)
        
        Esz = np.ones((bksize,bksize))     # Template for size of blocks
        Enm = np.ones((numbk,numbk))       # Template for numbe of blocks     
        
        # Four colors of subdomain 
        self.colors = ('blue','yellow','red','green')   

        self.dex = []     # Indices of colored subdomains
        self.S = []       # Operator on each subdomain

        for k in xrange(4):  # Loop over four colors of subdomains

            # Determine color of subdomains associated with grid points
            q = csr_matrix(np.reshape(Id[:,k],(2,2)))
            mat = T*kron(kron(Enm,q),Esz)*T
            row,col = mat.nonzero()
            
            self.dex.append(n*row+col)               
            self.S.append(self.L[:,self.dex[k]][self.dex[k],:])     

    def update(self,u,f,color):
        """
            Update the solution of the Poisson equation on the specified
            color of blocks
        """
        k = self.colors.index(color)        
        r = f-self.L*u   # Update residual    
        v = u        
        v[self.dex[k]] += spsolve(self.S[k],r[self.dex[k]])
        return v




if __name__ == '__main__':

    from matplotlib import cm
    from matplotlib.pyplot import figure, show
    from mpl_toolkits.mplot3d import axes3d as p3
    from matplotlib import pyplot as plt


    bksize = 4   # Number of grid points along sides of blocks
    numbk = 2    # Number of blocks along sides of domain

    MS = mult_schwarz(bksize,numbk)

    maxit = 1400  # maximum number of iterations

    n = 2*bksize*numbk    # number of grid points along one side of the domain

    N=n+1
    t = np.cos(np.pi*np.linspace(N,0,N+1)/N) # CGL nodes

    x,y = np.meshgrid(t,t)

    # Forcing function (f is of size n^2):     
    pi=np.pi
    xx,yy = np.meshgrid(t[1:-1],t[1:-1])
    f=4.*np.cos(pi*xx)*np.cos(pi*yy)+np.cos(pi*xx)+np.cos(pi*yy) 
    f=f.flatten()   
         
 
    u = np.ones(n**2)     # Initial guess at solution
 
    for iter in xrange(maxit): # Iteratively compute solution 
        for clr in MS.colors: # loop over colors
            u = MS.update(u,f,clr)
       
    
    U = np.zeros((n+2,n+2))
    U[1:-1,1:-1] = np.reshape(u,(n,n))    

    exact=1./pi**4*(1.+np.cos(pi*x))*(1.+np.cos(pi*y)) # Exact solution
    maxerr=log10(max(abs(U.flatten()-exact.flatten()))) # Error

    fig1 = plt.figure(1)
    ax1 = p3.Axes3D(fig1)
    plt.annotate('bksize = %d, numbk = %d, $log_{10}$(max err) = %f, iteration: %d' % (bksize,numbk,maxerr,maxit), xy=(0.05, 0.95), xycoords='axes fraction')
    p=ax1.plot_wireframe(x,y,U,color='black',linewidth=0.2)
    p=ax1.plot_surface(x,y,U, rstride=1, cstride=1, cmap=cm.cool,linewidth=0.5, antialiased=True)
    ax1.set_xlabel('x' )
    ax1.set_ylabel('y')
    plt.show()


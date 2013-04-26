#
# Multiplicative Schwarz Domain Decomposition solution of Poisson eq.
# Original Finite Difference code written by Greg von Wickel
# Check out http://www.scientificpython.net/1/post/2012/12/multiplicative-schwarzdomain-decomposition.html
# Adaptation for Chebyshev collocation by Nikola Mirkov.
#
import numpy as np
from scipy.sparse import csr_matrix, spdiags, kron, identity
from scipy.sparse.linalg import spsolve 
from chebPy import cheb 

def neglap2d(n):
    """
       Returns the Chebyshev collocation approximation to -D_x^2-D_y^2 on
       the square [-1,1]^2 on the interior points only
    """
        
    D,x=cheb(n+1)
    D2=np.dot(D,D)
    D2=D2[1:-1,1:-1]
    D2=csr_matrix(D2)
    I = identity(n)
    L = -(kron(D2,I) + kron(I,D2))                # FD negative Laplacian
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
    numbk = 3    # Number of blocks along sides of domain

    MS = mult_schwarz(bksize,numbk)

    maxit = 50  # maximum number of iterations

    n = 2*bksize*numbk    # number of grid points along one side of the domain

    N=n+1
    t = np.cos(np.pi*np.linspace(N,0,N+1)/N) # CGL nodes

    x,y = np.meshgrid(t,t)

    f = np.ones(n**2)    # Forcing function             
 
    u = np.ones(n**2)     # Initial guess at solution
 
    for iter in xrange(maxit): # Iteratively compute solution 
        for clr in MS.colors: # loop over colors
            u = MS.update(u,f,clr)
       
    
    U = np.zeros((n+2,n+2))
    U[1:-1,1:-1] = np.reshape(u,(n,n))

    fig1 = plt.figure(1)
    ax1 = p3.Axes3D(fig1)
    ax1.plot_wireframe(x,y,U,color='black',linewidth=0.5)
    ax1.plot_surface(x,y,U, rstride=1, cstride=1, cmap=cm.autumn,
           linewidth=0.1, antialiased=True)
    ax1.set_xlabel('x' )
    ax1.set_ylabel('y')
    ax1.set_title('Computed Solution')
    show()


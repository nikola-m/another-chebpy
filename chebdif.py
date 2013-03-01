import numpy as np
from scipy.linalg import toeplitz

def chebdif(N,M):
    '''Adopted from Weideman&Reddy's Matlab function chebdif.m.
    Input:
    N - size of diff matrix - np+1 where np is polynomial order.
    M - Highest derivative matrix order we need.
    Output:
    DM - (ell x N x N) - where ell=0..M-1.

    I will just paste the explanation from the original chebdif.m file:

    %  The code implements two strategies for enhanced 
    %  accuracy suggested by W. Don and S. Solomonoff in 
    %  SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    %  The two strategies are (a) the use of trigonometric 
    %  identities to avoid the computation of differences 
    %  x(k)-x(j) and (b) the use of the "flipping trick"
    %  which is necessary since sin t can be computed to high
    %  relative precision when t is small whereas sin (pi-t) cannot.

    I notice they also use the Negative Sum Trick (NTS) at the end.
    NTS is when you set diagonal entries to be negative sum of the
    other elements in the row.
    '''
    I = np.eye(N)  # Identity matrix
    DM = np.zeros((M,N,N))

    n1 = N/2; n2 = int(round(N/2.))  # Indices used for flipping trick

    k = np.arange(N)  # Compute theta vector
    th = k*np.pi/(N-1)

#    x = np.cos(np.pi*np.linspace(N-1,0,N)/(N-1))               # Old way with cos function.
    x = np.sin(np.pi*((N-1)-2*np.linspace(N-1,0,N))/(2*(N-1)))  # Compute Chebyshev points in the way W&R did it, with sin function.
    x = x[::-1]
    T = np.tile(th/2,(N,1))
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T)               # Trigonometric identity.
    DX[n1:,:] = -np.flipud(np.fliplr(DX[0:n2,:]))    # Flipping trick.!!!
    DX[range(N),range(N)]=1.                    # Put 1's on the main diagonal of DX.
    DX=DX.T

    C = toeplitz((-1.)**k)   # C i sthe matrix with entries c(k)/c(j)
    C[0,:]  *= 2
    C[-1,:] *= 2
    C[:,0] *= 0.5
    C[:,-1] *= 0.5

    Z = 1./DX                        # Z contains entries 1/(x(k)-x(j))
    Z[range(N),range(N)] = 0.        # with zeros on the diagonal.          

    D = np.eye(N)                    # D contains diff. matrices.
                                          
    for ell in range(M):
        D = (ell+1)*Z*(C*np.tile(np.diag(D),(N,1)).T - D)  # Off-diagonals    
        D[range(N),range(N)]= -np.sum(D,axis=1)       # Correct main diagonal of D - Negative sum trick!
        DM[ell,:,:] = D                               # Store current D in DM

    return x,DM

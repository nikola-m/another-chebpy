#
# Python version of the Maple code found in:
#     J.P.Boyd - Chebyshev Spectral Methods and the Lane-Emden Problem, Numer. Math. Theor. Meth. Appl., Vol. 4, No. 2, pp. 142-157, May 2011.
#
# This code will be enough to create all the figures and tables from this paper.
# 
# Authors: Anandaram Mandayam Nayaka email: mnanandaram@gmail.com & Nikola Mirkov: largeddysimulation@gmail.com
#
from mpmath import *
mp.dps = 30  # variable precision

n = 100      # number of interpolation points
m = 0.5     # Lane-Emden exponent m \in [0,5]
itermax = 10 # number of Newton iters

# Differentiation matrices and collocation poins init.
D0=matrix(n,n); D1=matrix(n,n); D2=matrix(n,n); xCheb = matrix(n,1)
fa = zeros(n+1,1); ya0 = zeros(n,1)
ta=matrix(n,1)
for ii in range(n):
    ta[ii]=pi*ii/(n-1); xCheb[ii]=0.5*(1+cos(ta[ii]))
for ii in range(1,n-1):
    t=ta[ii]; ss=sin(t); cc=cos(t)
    for j in range(n):
        D0[ii,j]=cos(j*t); pt=-j*sin(j*t); ptt=-j*j*D0[ii,j]; 
        D1[ii,j]=-2*pt/ss;
        D2[ii,j]=4*(ptt/(ss*ss)-cc*pt/(ss*ss*ss))
# apply non-trig formulas at the endpoints
for j in range(n):
    D0[0,j]=1.; D0[n-1,j]=(-1)**j
    D1[0,j]=2*j**2; D1[n-1,j]=2*(-1)**j*j**2
    D2[0,j]=j**2*(j**2-1)*4./3.; D2[n-1,j]=(-1)**j*j**2*((j**2-1)*4./3.)
for ii in range(n):
    ya0[ii]=cos(pi/2*xCheb[ii])
xi0=3.0 
# ya0 and xi0 are first guess for Newton iteration
a0=lu_solve(D0,ya0) # Cheb coefs of ya0(x)
a=a0; xi=xi0
Jacobian=matrix(n+1,n+1)
print 'Newton iterations:'
for iters in range(1,itermax+1):
    # begin Newton-Kantorovich iteration
    ya=D0*a
    for ii in range(n-2):   
        fa[ii]=-xi*xi*ya[ii+1]**m
        for j in range(n):
            fa[ii]=fa[ii]-D2[ii+1,j]*a[j]-(2/xCheb[ii+1])*D1[ii+1,j]*a[j]
    yatzero=0.
    for j in range(n):
        yatzero=yatzero+D0[n-1,j]*a[j]
    fa[n]=-(yatzero-1.); fa[n-2]=0.; fa[n-1]=0.; Jacobian[n,n]=0.
    for ii in range(n-2):  
        for j in range(n):
            Jacobian[ii,j]=D2[ii+1,j]+(2/xCheb[ii+1])*D1[ii+1,j] \
            + xi*xi*m*ya[ii+1]**(m-1)*D0[ii+1,j]
    for j in range(n):
        Jacobian[n-2,j]=D0[0,j]; Jacobian[n-1,j]=D1[n-1,j]; Jacobian[n,j]=D0[n-1,j]
    for ii in range(n-2):
        Jacobian[ii,n]=2*xi*ya[ii+1]**m
    Jacobian[n-1,n]=0.; Jacobian[n-2,n]=0.
    delta_a_and_xi=lu_solve(Jacobian,fa)
    for j in range(n):
        a[j]=a[j]+delta_a_and_xi[j]
    xidelta=delta_a_and_xi[n]; xi=xi+xidelta; print iters,xidelta

print 'xi: '
print xi
print 'a0: '
print a


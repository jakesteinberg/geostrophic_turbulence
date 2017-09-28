import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs 

def vertical_modes(N2,Depth,omega,mmax): 
    z = -1*Depth
    n = np.size(z); 
    nm1 = n - 1; 
    nm2 = n - 2;
    gravity = 9.82    
    ###### vertical increments
    zm = 0.5*( z[0:-1] + z[1:])  
    dz = np.concatenate([ [0], z[1:] - z[0:nm1] ])  # depth increment [m]
    dzm = np.concatenate([ [0], 0.5*(z[2:] - z[0:nm2]), [0] ])  # depth increment between midpoints [m]     
    ###### sparse matrices   
    # A = row pos, B = col pos, C = val  
    A = np.concatenate([ [0], [0], np.arange(1,nm1), np.arange(1,nm1), np.arange(1,nm1), [n-1] ])
    B = np.concatenate([ [0], [1], np.arange(1,nm1), np.arange(0,nm2), np.arange(2,n), [n-1] ])
    C = np.concatenate([ [-1/dz[1]], [1/dz[1]], (1/dz[2:] + 1/dz[1:nm1])/dzm[1:nm1], -1/(dz[1:nm1]*dzm[1:nm1]), -1/(dz[2:n]*dzm[1:nm1]), [-1] ])
    mat1 = coo_matrix((C,(A,B)),shape=(n,n))    
    
    D = np.concatenate([ [0],np.arange(1,n) ])
    E = np.concatenate([ [0],np.arange(1,n) ])
    F = np.concatenate([ [gravity], N2[1:,10] - omega*omega ])
    mat2 = coo_matrix((F,(D,E)),shape=(n,n)) 
    
    # compute eigenvalues and vectors 
    vals, vecs = eigs(mat1,k=mmax+1,M=mat2,sigma=0)
    eigenvalue = np.real(vals)
    wmodes = np.real(vecs)
    s_ind = np.argsort(eigenvalue)
    eigenvalue = eigenvalue[s_ind]
    wmodes = wmodes[:,s_ind]
    m = np.size(eigenvalue)
    c = 1/np.sqrt(eigenvalue) # kelvin wave speed
    # normalize mode (shapes)
    Gz = np.zeros(np.shape(wmodes)) 
    G = np.zeros(np.shape(wmodes)) 
    for i in range(m):
        dwdz = np.gradient(wmodes[:,i],z)
        norm_constant = np.sqrt( np.trapz(dwdz*dwdz,z)/z[-1] )
        if dwdz[0] < 0:
            norm_constant = -1*norm_constant
        Gz[:,i] = dwdz/norm_constant
        G[:,i] = wmodes[:,i]/norm_constant
        
    return(G,Gz,c)

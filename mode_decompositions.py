import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs 
    
# solves G''(z) + (N^2(z) - omega^2)G(z)/c^2 = 0 
#   subject to G'(0) = gG(0)/c^2 (free surface) & G(-D) = 0 (flat bottom)
# G(z) is normalized so that the vertical integral of (G'(z))^2 is D
# G' is dimensionless, G has dimensions of length

# - N is buoyancy frequency [s^-1] (nX1 vector)
# - depth [m] (maximum depth is considered the sea floor) (nX1 vector)
# - omega is frequency [s^-1] (scalar)
# - mmax is the highest baroclinic mode calculated
# - m=0 is the barotropic mode
# - 0 < m <= mmax are the baroclinic modes
# - Modes are calculated by expressing in finite difference form 1) the
#  governing equation for interior depths (rows 2 through n-1) and 2) the
#  boundary conditions at the surface (1st row) and the bottome (last row).
# - Solution is found by solving the eigenvalue system A*x = lambda*B*x
def vertical_modes(N2_0,Depth,omega,mmax): 
    z = -1*Depth
    
    if np.size(np.shape(N2_0)) > 1: 
        N2 = np.nanmean(N2,axis=1)
    else:
        N2 = N2_0
        
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
    F = np.concatenate([ [gravity], N2[1:] - omega*omega ]) # originially says N2[1:,10]
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

def PE_Tide_GM(rho0,Depth,nmodes,N2,f_ref):
    modenum = np.arange(0,nmodes)   
    Navg = np.trapz(np.nanmean(np.sqrt(N2),1),Depth)/Depth[-1]
    
    TE_SD = (75 + 280 + 72)/(rho0*Depth[-1])          # SD tidal energy [m^2/s^2] Hendry 1977
    sigma_SD = 2*np.pi/(12*3600)                      # SD frequency [s^-1]
    PE_SD = TE_SD*(sigma_SD**2 - f_ref**2)/(2*sigma_SD**2)
    
    bGM = 1300                  # GM internal wave depth scale [m]
    N0_GM = 5.2e-3              # GM N scale [s^-1];
    jstar = 3                   # GM vertical mode number scale
    EGM = 6.3e-5                # GM energy level [no dimensions]
    HHterm = 1/(modenum[1:]*modenum[1:] + jstar*jstar)
    HH = HHterm/np.sum(HHterm)
    
    PE_GM = bGM*bGM*N0_GM*Navg*HH*EGM/2
    
    return(PE_SD, PE_GM)

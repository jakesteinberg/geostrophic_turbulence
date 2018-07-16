import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs
from scipy.integrate import cumtrapz


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
def vertical_modes(N2_0, Depth, omega, mmax):
    z = -1 * Depth

    if np.size(np.shape(N2_0)) > 1:
        N2 = np.nanmean(N2_0, axis=1)
    else:
        N2 = N2_0

    n = np.size(z)
    nm1 = n - 1
    nm2 = n - 2
    gravity = 9.82
    # ----- vertical increments
    dz = np.concatenate([[0], z[1:] - z[0:nm1]])  # depth increment [m]
    dzm = np.concatenate([[0], 0.5 * (z[2:] - z[0:nm2]), [0]])  # depth increment between midpoints [m]
    # ----- sparse matrices
    # A = row pos, B = col pos, C = val  
    A = np.concatenate([[0], [0], np.arange(1, nm1), np.arange(1, nm1), np.arange(1, nm1), [n - 1]])
    B = np.concatenate([[0], [1], np.arange(1, nm1), np.arange(0, nm2), np.arange(2, n), [n - 1]])
    C = np.concatenate(
        [[-1 / dz[1]], [1 / dz[1]], (1 / dz[2:] + 1 / dz[1:nm1]) / dzm[1:nm1], -1 / (dz[1:nm1] * dzm[1:nm1]),
         -1 / (dz[2:n] * dzm[1:nm1]), [-1]])
    mat1 = coo_matrix((C, (A, B)), shape=(n, n))

    D = np.concatenate([[0], np.arange(1, n)])
    E = np.concatenate([[0], np.arange(1, n)])
    F = np.concatenate([[gravity], N2[1:] - omega * omega])  # originially says N2[1:,10]
    mat2 = coo_matrix((F, (D, E)), shape=(n, n))

    # compute eigenvalues and vectors 
    vals, vecs = eigs(mat1, k=mmax + 1, M=mat2, sigma=0)
    eigenvalue = np.real(vals)
    wmodes = np.real(vecs)
    s_ind = np.argsort(eigenvalue)
    eigenvalue = eigenvalue[s_ind]
    wmodes = wmodes[:, s_ind]
    m = np.size(eigenvalue)
    c = 1 / np.sqrt(eigenvalue)  # kelvin wave speed
    # normalize mode (shapes)
    Gz = np.zeros(np.shape(wmodes))
    G = np.zeros(np.shape(wmodes))
    for i in range(m):
        dw_dz = np.gradient(wmodes[:, i], z)
        norm_constant = np.sqrt(np.trapz((dw_dz * dw_dz), (-1 * z)) / (-1 * z[-1]))
        # norm_constant = np.abs(np.trapz(dw_dz * dw_dz, z) / Depth.max())
        if dw_dz[0] < 0:
            norm_constant = -1 * norm_constant
        Gz[:, i] = dw_dz / norm_constant
        G[:, i] = wmodes[:, i] / norm_constant

    epsilon = np.nan * np.zeros((2, 3, 3))
    for i in range(0, 2):  # i modes
        for j in range(0, 3):  # j modes
            for m in range(0, 3):  # k modes
                epsilon[i, j, m] = np.trapz((Gz[:, i] * Gz[:, j] * Gz[:, m]), z) / (z[-1])

    return G, Gz, c, epsilon


def vertical_modes_f(N2_0, depth, omega, mmax, bc_bot):
    z = -1 * depth
    # ensure that N2 is 1-D array (take average if not)
    if np.size(np.shape(N2_0)) > 1:
        N2 = np.nanmean(N2_0, axis=1)
    else:
        N2 = N2_0  # buoyancy frequency squared

    # make first element = 0 if not already so
    if depth[0] != 0:
        z = np.nan * np.zeros(len(depth))
        z[0] = 0
        z[1:] = -depth[0:-1]
        N2 = np.concatenate([[0], N2[0:-1]])

    n = np.size(z)
    nm1 = n - 1
    nm2 = n - 2
    gravity = 9.82
    # ---- fix N2 profiles with zeros
    fixer = np.where(N2 == 0)[0]
    fixer2 = fixer[(fixer > 10) & (fixer < len(depth) - 20)]
    if len(fixer2) >= 1:
        for i in range(len(fixer2)):
            N2[fixer2[i]] = N2[fixer2[i] - 1] + (N2[fixer2[i] + 1] - N2[fixer2[i] - 1]) / 2
    n2_first = np.where(N2 > 0)[0][0]
    n2_last = np.where(N2 > 0)[0][-1]
    N2[n2_last + 1:] = N2[n2_last]
    N2[1:n2_first] = N2[n2_first]
    # all of this removes N2 = 0 at the end of the array and leaves N2 = 0 only for the first element of the array

    N2_inter = N2.copy()
    N2_inter[0] = N2_inter[1]

    # 1st approximate (k(x)u'(x)) at halfway points
    z12 = (z[1:] - z[0:-1]) / 2 + z[0:-1]
    k = 1 / N2_inter
    k12 = np.flipud(np.interp(np.flipud(z12), np.flipud(z), np.flipud(k)))

    # ----- vertical increments
    dz = np.concatenate([[0], z[1:] - z[0:nm1]])  # depth increment [m]
    dzm = np.concatenate([[0], 0.5 * (z[2:] - z[0:nm2]), [0]])  # depth increment between midpoints [m]
    # ----- sparse matrices
    # A = row pos, B = col pos, C = val
    if bc_bot > 1:
        A = np.concatenate([[0], [0], np.arange(1, nm1), np.arange(1, nm1), np.arange(1, nm1), [n - 1]])
        B = np.concatenate([[0], [1], np.arange(1, nm1), np.arange(0, nm2), np.arange(2, n), [n - 1]])
        # rough_bottom
        C = np.concatenate(
            [[-1 / dz[1]], [1 / dz[1]], (1 / dz[1:nm1] / dz[1:nm1]) * (k12[0:-1] + k12[1:]),
             -1 * k12[0:-1] / (dz[1:nm1] * dz[1:nm1]), -1 * k12[1:] / (dz[2:n] * dz[1:nm1]), [-1]])
    else:
        A = np.concatenate([[0], [0], np.arange(1, nm1), np.arange(1, nm1), np.arange(1, nm1), [n - 1], [n - 1]])
        B = np.concatenate([[0], [1], np.arange(1, nm1), np.arange(0, nm2), np.arange(2, n), [n - 2], [n - 1]])
        # flat bottom
        C = np.concatenate(
            [[-1 / dz[1]], [1 / dz[1]], (1 / dz[1:nm1] / dz[1:nm1]) * (k12[0:-1] + k12[1:]),
             -1 * k12[0:-1] / (dz[1:nm1] * dzm[1:nm1]), -1 * k12[1:] / (dz[2:n] * dzm[1:nm1]), [1 / dz[nm1]],
             [-1 / dz[nm1]]])

    mat1 = coo_matrix((C, (A, B)), shape=(n, n))

    D = np.concatenate([[0], np.arange(1, nm1), [n - 1]])
    E = np.concatenate([[0], np.arange(1, nm1), [n - 1]])
    F = np.concatenate([[N2[0] / gravity], np.ones(len(np.arange(1, nm1))), [0]])
    mat2 = coo_matrix((F, (D, E)), shape=(n, n))

    # compute eigenvalues and vectors
    vals, vecs = eigs(mat1, k=mmax + 1, M=mat2, sigma=0)
    eigenvalue = np.real(vals)
    wmodes = np.real(vecs)
    s_ind = np.argsort(eigenvalue)
    eigenvalue = eigenvalue[s_ind]
    wmodes = wmodes[:, s_ind]
    m = np.size(eigenvalue)
    c = 1 / np.sqrt(eigenvalue)  # kelvin wave speed
    # normalize mode (shapes)
    F = np.zeros(np.shape(wmodes))
    G = np.zeros(np.shape(wmodes))
    for i in range(m):
        norm_constant = np.sqrt(np.trapz(wmodes[:, i] * wmodes[:, i], z) / z[-1])
        if wmodes[0, i] < 0:
            norm_constant = -1 * norm_constant
        F[:, i] = wmodes[:, i] / norm_constant
        G[:, i] = (1 / N2_inter) * np.gradient(wmodes[:, i], z) / norm_constant
        # G[:, i] = cumtrapz(wmodes[:, i], z, initial=0) / norm_constant

    epsilon = np.nan * np.zeros((2, 3, 3))
    for i in range(0, 2):  # i modes
        for j in range(0, 3):  # j modes
            for m in range(0, 3):  # k modes
                epsilon[i, j, m] = np.trapz((F[:, i] * F[:, j] * F[:, m]), z) / (z[-1])

    return G, F, c, norm_constant, epsilon


def PE_Tide_GM(rho0, Depth, nmodes, N2, f_ref):
    modenum = np.arange(0, nmodes)
    Navg = np.trapz(np.nanmean(np.sqrt(N2), 1), Depth) / Depth[-1]

    TE_SD = (75 + 280 + 72) / (rho0 * Depth[-1])  # SD tidal energy [m^2/s^2] Hendry 1977
    sigma_SD = 2 * np.pi / (12 * 3600)  # SD frequency [s^-1]
    PE_SD = TE_SD * (sigma_SD ** 2 - f_ref ** 2) / (2 * sigma_SD ** 2)

    bGM = 1300  # GM internal wave depth scale [m]
    N0_GM = 5.2e-3  # GM N scale [s^-1];
    jstar = 3  # GM vertical mode number scale
    EGM = 6.3e-5  # GM energy level [no dimensions]
    HHterm = 1 / (modenum[1:] * modenum[1:] + jstar * jstar)
    HH = HHterm / np.sum(HHterm)

    PE_GM = bGM * bGM * N0_GM * Navg * HH * EGM / 2

    return (PE_SD, PE_GM)


def eta_fit(num_profs, grid, nmodes, n2, G, c, eta, eta_fit_dep_min, eta_fit_dep_max):
    bvf = np.sqrt(n2)
    Neta = np.nan * np.zeros(eta.shape)
    NEta_m = np.nan * np.zeros(eta.shape)
    Eta_m = np.nan * np.zeros(eta.shape)
    AG = np.zeros([nmodes, num_profs])
    PE_per_mass = np.nan * np.zeros([nmodes, num_profs])
    for i in range(num_profs):
        # fit to eta profiles
        this_eta = eta[:, i].copy()
        # obtain matrix of NEta
        Neta[:, i] = bvf * this_eta
        # find indices within fitting range
        iw = np.where((grid >= eta_fit_dep_min) & (grid <= eta_fit_dep_max))
        if iw[0].size > 1:
            eta_fs = eta[:, i].copy()  # ETA

            i_sh = np.where((grid < eta_fit_dep_min))
            eta_fs[i_sh[0]] = grid[i_sh] * this_eta[iw[0][0]] / grid[iw[0][0]]

            i_dp = np.where((grid > eta_fit_dep_max))
            eta_fs[i_dp[0]] = (grid[i_dp] - grid[-1]) * this_eta[iw[0][-1]] / (grid[iw[0][-1]] - grid[-1])

            AG[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], np.transpose(np.atleast_2d(eta_fs)))[0])
            Eta_m[:, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG[:, i])))
            NEta_m[:, i] = bvf * np.array(np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG[:, i]))))
            PE_per_mass[:, i] = (1 / 2) * AG[:, i] * AG[:, i] * c * c

    return (AG, Eta_m, NEta_m, PE_per_mass)

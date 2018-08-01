import numpy as np
import matplotlib.pyplot as plt
import glob
import datetime
import gsw
import pickle
from mode_decompositions import vertical_modes, vertical_modes_f, PE_Tide_GM
from toolkit import unq_searchsorted, plot_pro

file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Shipboard/ABACO/*.pkl')

# --- LOAD ABACO SHIPBOARD CTD DATA
abaco = []
for i in file_list:
    pkl_file = open(i, 'rb')
    abaco.append(pickle.load(pkl_file))
    pkl_file.close()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f0, (ax3, ax4, ax5, ax6) = plt.subplots(4, 1, sharex=True)
HKE_U_out = []
HKE_V_out = []
goodz = [0, 1, 2, 4]
# loop over each Cruise (Once or Twice per year)
for i in goodz:  # len(abaco)):
    den = abaco[i]['den_grid']
    bin_depth = abaco[i]['bin_depth']
    u = abaco[i]['adcp_u'] / 100
    v = abaco[i]['adcp_v'] / 100
    adcp_dep = abaco[i]['adcp_depth']
    adcp_dist = abaco[i]['adcp_dist']
    adcp_np = u.shape[1]

    ax3.plot(adcp_dist, u[50, :], 'r')
    ax3.plot(adcp_dist, v[50, :], 'b')
    ax4.plot(adcp_dist, u[100, :], 'r')
    ax4.plot(adcp_dist, v[100, :], 'b')
    ax5.plot(adcp_dist, u[300, :], 'r')
    ax5.plot(adcp_dist, v[300, :], 'b')
    ax6.plot(adcp_dist, u[400, :], 'r')
    ax6.plot(adcp_dist, v[400, :], 'b')


    avg_den = np.nanmean(den[:, 5:-3], axis=1)
    N2 = (-9.81/np.nanmean(avg_den))*np.gradient(avg_den, -1 * bin_depth)
    N2[N2 < 0] = -1 * N2[N2 < 0]
    bad_up = np.where(np.isnan(N2[0:30]))[0]
    bad_bot = np.where(np.isnan(N2[150:]))[0]
    for l in bad_up:
        N2[l] = N2[len(bad_up)]
    for l in bad_bot:
        N2[l+150] = N2[150 + l - 1] + 1*10**-8

    # ax0.plot(N2, bin_depth)
    # ax0.invert_yaxis()

    # ------- Eta_fit / Mode Decomposition --------------
    # define G grid
    omega = 0  # frequency zeroed for geostrophic modes
    mmax = 75  # highest baroclinic mode to be calculated
    nmodes = mmax + 1

    G, Gz, c, epsilon = vertical_modes(N2, bin_depth, omega, mmax)

    # --- compute alternate vertical modes
    bc_bot = 2  # 1 = flat, 2 = rough
    grid2 = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 5480, 10)])
    n2_interp = np.interp(grid2, bin_depth, N2)
    F_int_g2, F_g2, c_ff, norm_constant, epsilon2 = vertical_modes_f(n2_interp, grid2, omega, mmax, bc_bot)
    F = np.nan * np.ones((np.size(bin_depth), mmax + 1))
    F_int = np.nan * np.ones((np.size(bin_depth), mmax + 1))
    for m in range(mmax + 1):
        F[:, m] = np.interp(bin_depth, grid2, F_g2[:, m])
        F_int[:, m] = np.interp(bin_depth, grid2, F_int_g2[:, m])

    HKE_noise_threshold_adcp = 1e-5
    AGz = np.zeros([nmodes, adcp_np])
    U_m = np.nan * np.zeros([np.size(bin_depth), adcp_np])
    V_m = np.nan * np.zeros([np.size(bin_depth), adcp_np])
    HKE_U_per_mass = np.nan * np.zeros([nmodes, adcp_np])
    HKE_V_per_mass = np.nan * np.zeros([nmodes, adcp_np])
    modest = np.arange(11, nmodes)
    good_prof_U = np.zeros(adcp_np)
    good_prof_V = np.zeros(adcp_np)
    for j in range(abaco[i]['adcp_u'].shape[1]):
        U = u[:, j]
        V = v[:, j]
        bad = np.where(np.isnan(U[0:20]))[0]
        U[bad] = U[len(bad)]
        bad = np.where(np.isnan(V[0:20]))[0]
        V[bad] = V[len(bad)]
        # fit to velocity profiles
        if adcp_dep[~np.isnan(V)][-1] > 5250:
            this_U_0 = U.copy()
            this_U = np.interp(bin_depth, adcp_dep, this_U_0)
            iv = np.where(~np.isnan(this_U))
            if iv[0].size > 1:
                AGz[:, j] = np.squeeze(np.linalg.lstsq(np.squeeze(F[iv, :]), np.transpose(np.atleast_2d(this_U[iv])))[0])
                U_m[:, j] = np.squeeze(np.matrix(F) * np.transpose(np.matrix(AGz[:, j])))  # Gz*AGz[:,i];
                HKE_U_per_mass[:, j] = AGz[:, j] * AGz[:, j]
                ival = np.where(HKE_U_per_mass[modest, j] >= HKE_noise_threshold_adcp)
                if np.size(ival) > 0:
                    good_prof_U[j] = 1  # flag profile as noisy
                else:
                    good_prof_U[j] = 1  # flag empty profile as noisy as well

            this_V_0 = V.copy()
            this_V = np.interp(bin_depth, adcp_dep, this_V_0)
            iv = np.where(~np.isnan(this_V))
            if iv[0].size > 1:
                AGz[:, j] = np.squeeze(np.linalg.lstsq(np.squeeze(F[iv, :]), np.transpose(np.atleast_2d(this_V[iv])))[0])
                V_m[:, j] = np.squeeze(np.matrix(F) * np.transpose(np.matrix(AGz[:, j])))  # Gz*AGz[:,i];
                HKE_V_per_mass[:, j] = AGz[:, j] * AGz[:, j]
                ival = np.where(HKE_V_per_mass[modest, j] >= HKE_noise_threshold_adcp)
                if np.size(ival) > 0:
                    good_prof_V[j] = 1  # flag profile as noisy
                else:
                    good_prof_V[j] = 1  # flag empty profile as noisy as well

            ax1.plot(u[:, j], adcp_dep, linewidth=0.5)
            ax1.plot(U_m[:, j], bin_depth, linewidth=0.5, color='k', linestyle='--')
            ax2.plot(v[:, j], adcp_dep, linewidth=0.5)
            ax2.plot(V_m[:, j], bin_depth, linewidth=0.5, color='k', linestyle='--')
    HKE_U_out.append(HKE_U_per_mass)
    HKE_V_out.append(HKE_V_per_mass)
ax1.invert_yaxis()
ax1.set_xlim([-.75, 0.75])
ax1.grid()
ax2.set_xlim([-.75, 0.75])
plot_pro(ax2)

ax3.grid()
ax3.set_ylim([-.5, .5])
ax4.grid()
ax4.set_ylim([-.5, .5])
ax5.grid()
ax5.set_ylim([-.5, .5])
ax6.set_ylim([-.5, .5])
plot_pro(ax6)

# --- KE plot
f_ref = np.pi * np.sin(np.deg2rad(26.5)) / (12 * 1800)
dk = f_ref / c[1]
sc_x = 1000 * f_ref / c[1:]

f, ax = plt.subplots()
for i in range(len(HKE_U_out)):
    KE_adcp = ax.plot(sc_x, np.nanmean(HKE_U_out[i], axis=1)[1:] / dk, color='k', linewidth=2)
    KE_adcp = ax.plot(sc_x, np.nanmean(HKE_V_out[i], axis=1)[1:] / dk, color='b', linewidth=2)
ax.set_xlabel('vertical wavenumber')
ax.set_ylabel('variance per vert. wavenumber')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([10**-1, 2*10**3])
plot_pro(ax)
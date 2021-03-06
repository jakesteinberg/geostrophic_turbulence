import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import gsw
import pickle
from netCDF4 import Dataset
from scipy.signal import savgol_filter
from grids import make_bin_gen
from toolkit import plot_pro, nanseg_interp
from mode_decompositions import vertical_modes, vertical_modes_f, eta_fit, PE_Tide_GM

ref_lat = 24 + (45 / 60)
ref_lon = 158 - 360
HD1 = Dataset('/Users/jake/Documents/baroclinic_modes/HOTS/HOTS_20065.nc', 'r')
HD2 = Dataset('/Users/jake/Documents/baroclinic_modes/HOTS/HOTS_190036.nc', 'r')
h_s = np.concatenate([HD1['sal'][:], HD2['sal'][:]])
h_temp = np.concatenate([HD1['temp'][:], HD2['temp'][:]])
h_p = np.concatenate([HD1['press'][:], HD2['press'][:]])
h_time = np.concatenate([HD1['days'][:], HD2['days'][:]]) + 726011  # days from Oct 1 1988
h_s[h_s < 0] = np.nan
h_temp[h_temp < 0] = np.nan

# ----

# reference grid
GD = Dataset('/Users/jake/Documents/geostrophic_turbulence/BATs_2015_gridded_apr04.nc', 'r')
grid = np.concatenate([GD.variables['grid'][:], np.arange(GD.variables['grid'][:][-1] + 20, 4720, 20)])
z = -1 * grid
grid_p = gsw.p_from_z(-1 * grid, ref_lat)

dates = np.unique(h_time)
SA = np.nan * np.zeros((len(grid), 1))
CT = np.nan * np.zeros((len(grid), 1))
sig0 = np.nan * np.zeros((len(grid), 1))
time = np.nan * np.zeros(1)
d_max = np.nan * np.zeros(len(dates))
count = 0
for i in range(len(dates)):
    d_i = np.where(h_time == dates[i])
    t_i = h_temp[d_i]
    s_i = h_s[d_i]
    p_i = h_p[d_i]
    depth = -1 * gsw.z_from_p(p_i, ref_lat)
    d_max[i] = depth.max()

    test_t_grad = np.gradient(t_i, -1 * depth)
    test_s_grad = np.gradient(s_i, -1 * depth)
    if (d_max[i] > 4700) & (np.sum(np.abs(test_s_grad[40:]) > 0.018) < 1) &\
            (np.sum(np.abs(test_t_grad[40:]) > 0.15) < 1):
        temp_g, sal_g = make_bin_gen(grid, depth, t_i, s_i)
        if count < 1:
            SA[:, 0] = gsw.SA_from_SP(sal_g, grid, ref_lon * np.ones(len(sal_g)), ref_lat * np.ones(len(sal_g)))
            CT[:, 0] = gsw.CT_from_t(SA[:, 0], temp_g, grid_p)
            sig0[:, 0] = gsw.sigma0(SA[:, 0], CT[:, 0])
            time[0] = dates[i]
        else:
            SA = np.append(SA, gsw.SA_from_SP(sal_g, grid, ref_lon * np.ones(len(sal_g)),
                                              ref_lat * np.ones(len(sal_g)))[:, np.newaxis], axis=1)
            CT = np.append(CT, gsw.CT_from_t(SA[:, count], temp_g, grid_p)[:, np.newaxis], axis=1)
            sig0 = np.append(sig0, gsw.sigma0(SA[:, count], CT[:, count])[:, np.newaxis], axis=1)
            time = np.append(time, dates[i])
        count = count + 1


num_profs = np.shape(SA)[1]
t_s = datetime.date.fromordinal(np.int(np.min(time)))  # t_start 726377
t_e = datetime.date.fromordinal(np.int(np.max(time)))
SA[SA < 0] = np.nan
CT[CT < 0] = np.nan
SA_avg = np.nanmean(SA, axis=1)
CT_avg = np.nanmean(CT, axis=1)
sig0_avg = np.nanmean(sig0, axis=1)
sig0_anom = sig0 - np.tile(sig0_avg[:, np.newaxis], (1, num_profs))
ddz_avg_sigma = np.gradient(sig0_avg, z)
N2 = gsw.Nsquared(SA_avg, CT_avg, grid_p, lat=ref_lat)[0]
# N2 is one value less than grid_p
N2[N2 < 0] = np.nan
N2 = nanseg_interp(grid, N2)
N2 = np.append(N2, (N2[-1] - N2[-1]/20))

window_size = 11
poly_order = 3
N2 = savgol_filter(N2, window_size, poly_order)

eta = sig0_anom / np.tile(ddz_avg_sigma[:, np.newaxis], (1, num_profs))

# MODE PARAMETERS
# frequency zeroed for geostrophic modes
omega = 0
# highest baroclinic mode to be calculated
mmax = 60
nmodes = mmax + 1
eta_fit_dep_min = 50
eta_fit_dep_max = 4000

# -- computer vertical mode shapes
G, Gz, c, epsilon = vertical_modes(N2, grid, omega, mmax)
# --- compute alternate vertical modes
bc_bot = 1  # 1 = flat, 2 = rough
grid2 = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4500, 10)])
n2_interp = np.interp(grid2, grid, N2)
F_int_g2, F_g2, c_ff, norm_constant, epsilon2 = vertical_modes_f(n2_interp, grid2, omega, mmax, bc_bot, 22, 10**-4)
F = np.nan * np.ones((np.size(grid), mmax + 1))
F_int = np.nan * np.ones((np.size(grid), mmax + 1))
for i in range(mmax + 1):
    F[:, i] = np.interp(grid, grid2, F_g2[:, i])
    F_int[:, i] = np.interp(grid, grid2, F_int_g2[:, i])

# -- project modes onto eta profiles
AG, eta_m, Neta_m, PE_per_mass = eta_fit(num_profs, grid, nmodes, N2, G, c, eta, eta_fit_dep_min, eta_fit_dep_max)

# --- find EOFs of dynamic vertical displacement (Eta mode amplitudes)
AG_avg = AG.copy()
good_prof_eof = np.where(~np.isnan(AG_avg[2, :]))
num_profs_2 = np.size(good_prof_eof)
AG2 = AG_avg[:, good_prof_eof[0]]
C = np.transpose(np.tile(c, (num_profs_2, 1)))
AGs = C * AG2
AGq = AGs[1:, :]  # ignores barotropic mode
nqd = num_profs_2
avg_AGq = np.nanmean(AGq, axis=1)
AGqa = AGq - np.transpose(np.tile(avg_AGq, [nqd, 1]))  # mode amplitude anomaly matrix
cov_AGqa = (1 / nqd) * np.matrix(AGqa) * np.matrix(np.transpose(AGqa))  # nmodes X nmodes covariance matrix
var_AGqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGqa)))) * np.matrix(np.sqrt(np.diag(cov_AGqa)))
cor_AGqa = cov_AGqa / var_AGqa  # nmodes X nmodes correlation matrix

D_AGqa, V_AGqa = np.linalg.eig(cov_AGqa)  # columns of V_AGzqa are eigenvectors
EOFetaseries = np.transpose(V_AGqa) * np.matrix(AGqa)  # EOF "timeseries' [nmodes X nq]
EOFetashape = np.matrix(G[:, 1:]) * V_AGqa  # depth shape of eigenfunctions [ndepths X nmodes]
EOFetashape1_BTpBC1 = G[:, 1:3] * V_AGqa[0:2, 0]  # truncated 2 mode shape of EOF#1
EOFetashape2_BTpBC1 = G[:, 1:3] * V_AGqa[0:2, 1]  # truncated 2 mode shape of EOF#2

# --- ENERGY parameters
rho0 = 1027
f_ref = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)
dk = f_ref / c[1]
sc_x = 1000 * f_ref / c[1:]
vert_wavenumber = f_ref / c[1:]
dk_ke = 1000 * f_ref / c[1]
# PE_SD, PE_GM, GMPE, GMKE = PE_Tide_GM(rho0, grid, nmodes, np.transpose(np.atleast_2d(N2)), f_ref)
avg_PE = np.nanmean(PE_per_mass, 1)

# load in Station BATs PE Comparison
pkl_file = open('/Users/jake/Desktop/bats/station_bats_pe_apr11.pkl', 'rb')
SB = pickle.load(pkl_file)
pkl_file.close()
sta_bats_depth = SB['depth']
sta_bats_pe = SB['PE']
sta_bats_c = SB['c']
sta_bats_f = np.pi * np.sin(np.deg2rad(31.6)) / (12 * 1800)
sta_bats_dk = sta_bats_f / sta_bats_c[1]
sta_bats_n2 = np.nanmean(SB['N2_per_season'], axis=1)
G_B, Gz_B, c_B, epsilon_B = vertical_modes(sta_bats_n2, SB['depth'], omega, mmax)
# PE_SD_bats, PE_GM_bats, GMPE_bats, GMKE_bats = PE_Tide_GM(rho0, sta_bats_depth, nmodes,
#                                                           np.transpose(np.atleast_2d(sta_bats_n2)), sta_bats_f)

# load in Station PAPA PE Comparison
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Line_P/canada_DFO/papa_energy_spectra_jun13.pkl', 'rb')
SP = pickle.load(pkl_file)
pkl_file.close()
sta_papa_depth = SP['depth']
sta_papa_pe = SP['PE']
sta_papa_c = SP['c']
sta_papa_f = np.pi * np.sin(np.deg2rad(49.98)) / (12 * 1800)
sta_papa_dk = sta_papa_f / sta_papa_c[1]
sta_papa_n2 = SP['N2']
G_P, Gz_P, c_P, epsilon_P = vertical_modes(sta_papa_n2, SP['depth'], omega, mmax)
# PE_SD_papa, PE_GM_papa, GMPE_papa, GMKE_papa = PE_Tide_GM(rho0, sta_papa_depth, nmodes,
#                                                           np.transpose(np.atleast_2d(sta_papa_n2)), sta_papa_f)

# load in Station Deep Argo NZ PE Comparison
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Deep_Argo/float6036_oct17.pkl', 'rb')
SNZ = pickle.load(pkl_file)
pkl_file.close()
sta_nz_depth = SNZ['bin_depth']
sta_nz_pe = SNZ['avg_PE']
sta_nz_c = SNZ['c']
sta_nz_f = np.pi * np.sin(np.deg2rad(49.98)) / (12 * 1800)
sta_nz_dk = sta_nz_f / sta_nz_c[1]
sta_nz_n2 = SNZ['N2_avg']
G_P, Gz_P, c_P, epsilon_P = vertical_modes(sta_nz_n2, SNZ['bin_depth'], omega, mmax)
# PE_SD_nz, PE_GM_nz, GMPE_nz, GMKE_nz = PE_Tide_GM(rho0, sta_nz_depth, nmodes,
#                                                   np.transpose(np.atleast_2d(sta_nz_n2)), sta_nz_f)


# f, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.scatter(SA, CT, s=3)
# ax0.grid()
# for i in range(num_profs):
#     ax1.plot(sig0_anom[:, i], grid, linewidth=0.75)
#     ax2.plot(eta[:, i], grid, linewidth=0.75)
#     ax2.plot(eta_m[:, i], grid, linewidth=0.5, linestyle='--', color='k')
# ax2.set_xlim([-500, 500])
# ax1.invert_yaxis()
# ax1.grid()
# ax2.invert_yaxis()
# plot_pro(ax2)

# ----- PLOT DENSITY ANOM, ETA, AND MODE SHAPES
colors = plt.cm.Dark2(np.arange(0, 4, 1))
f, (ax, ax2, ax3) = plt.subplots(1, 3, sharey=True)
for i in range(num_profs):
    ax.plot(sig0_anom[:, i], grid, linewidth=0.75)
    ax2.plot(eta[0:247, i], grid[0:247], linewidth=.75, color='#808000')
    ax2.plot(eta_m[:, i], grid, linewidth=.5, color='k', linestyle='--')
n2p = ax3.plot((np.sqrt(N2) * (1800 / np.pi)) / 6, grid, color='k', label='N(z) [cph]')
for j in range(3):
    ax3.plot(G[:, j] / grid.max(), grid, color='#2F4F4F', linestyle='--')
    p_eof = ax3.plot(EOFetashape[:, j] / grid.max(), grid, color=colors[j, :], label='EOF # = ' + str(j + 1),
                     linewidth=2.5)
ax.text(0.2, 4000, str(num_profs) + ' profiles', fontsize=10)
ax.grid()
ax.set_title('ALOHA ' + str(t_s) + ' - ' + str(t_e))
ax.set_xlabel(r'$\sigma_{\theta} - \overline{\sigma_{\theta}}$')
ax.set_ylabel('Depth [m]')
ax.set_xlim([-.5, .5])
ax2.grid()
ax2.set_title('Isopycnal Displacement [m]')
ax2.set_xlabel('[m]')
ax2.axis([-600, 600, 0, 4750])
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels, fontsize=10)
ax3.set_title('EOFs of mode amplitudes G(z)')
ax3.set_xlabel('Normalized Mode Amplitude')
ax3.set_xlim([-1, 1])
ax3.invert_yaxis()
plot_pro(ax3)

# ----- energy spectra
matplotlib.rcParams['figure.figsize'] = (16/3 + .75, 6)
f, ax = plt.subplots()
mode_num = np.arange(1, 61, 1)
# BATS
PE_sta_p = ax.plot(mode_num, np.nanmean(sta_bats_pe[1:], axis=1),
                   color='g', label='APE$_{BATS}$', linewidth=2)  # 1000 * sta_bats_f / sta_bats_c[1:]
# HOTS
PE_p = ax.plot(mode_num, avg_PE[1:], color='#B22222', label='APE$_{ALOHA}$', linewidth=2)  # sc_x
# PAPA
PE_sta_papa_p = ax.plot(mode_num, np.nanmean(sta_papa_pe[1:], axis=1),
                   color='#4682B4', label='APE$_{PAPA}$', linewidth=2)
# NZ
# PE_sta_nz_p = ax.plot(mode_num, sta_nz_pe[1:] / sta_nz_dk,
#                    color='c', label='APE$_{NZ}$', linewidth=2)

# GM
# ax.plot(mode_num, PE_GM, linestyle='--', color='#B22222', linewidth=1)
# ax.plot(mode_num, PE_GM_bats, linestyle='--', color='g', linewidth=1)  # #00FF7F
# ax.plot(mode_num, PE_GM_papa, linestyle='--', color='#4682B4', linewidth=1)
# ax.plot(mode_num, PE_GM_nz / sta_nz_dk, linestyle='--', color='c', linewidth=0.75)
# ax.plot(1000 * sta_bats_f / sta_bats_c[1:], PE_GM_bats / sta_bats_dk, linestyle='--', color='#FF8C00', linewidth=0.75)
# ax.text(sc_x[0] - .005, PE_GM[1] / dk, r'$PE_{GM}$', fontsize=13)
# -3 slope
# ax.plot([3*10**-1, 3*10**0], [1.5*10**1, 1.5*10**-2],color='k', linewidth=0.75)
# ax.plot([4*10**-2, 4*10**-1], [1.5*10**0, 1.5*10**-2], color='k', linewidth=0.75)
# ax.text(3.4*10**-1, 1.3*10**1, '-3', fontsize=11)
# ax.text(4.4*10**-2, 1.3*10**0, '-2', fontsize=11)
# ax.plot([10**1, 10**2], [10**1, 10**-2], color='k', linewidth=0.75)
# ax.plot([10**0, 10**2], [10**2, 10**-2], color='k', linewidth=0.75)
# ax.plot([10**0, 10**2], [3.333 * 10**1, 10**-2], color='k', linewidth=0.75)
# ax.text(1.1*10**1, 10**1, '-3', fontsize=11)
# ax.text(1.2*10**0, 1.3*10**2, '-2', fontsize=11)
# ax.text(0.78*10**0, 3.33*10**1, '-5/3', fontsize=11)
# ax.plot([10**-1, 10**2], [10**5, 10**-4], color='k', linewidth=0.5)
# ax.plot([10**-1, 10**2], [10**4, 10**-2], color='k', linewidth=0.5)
ax.plot([10**-1, 10**2], [9 * 10**0, 9 * 10**-9], color='k', linewidth=0.5)
ax.text(6*10**1, 4*10**-8, '-3', fontsize=12)
ax.plot([10**-1, 10**2], [9 * 10**-1, 9 * 10**-7], color='k', linewidth=0.5)
ax.text(7*10**1, 8*10**-7, '-2', fontsize=12)

ax.set_yscale('log')
ax.set_xscale('log')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=14)
# ax.axis([10 ** -2, 10 ** 1, 10 ** (-4), 10 ** 3])
# ax.axis([7 * 10 ** -1, 10 ** 2, 3 * 10 ** (-4), 10 ** 3])
ax.set_xlim([0.6 * 10 ** 0, 1 * 10 ** 2])
ax.set_ylim([10 ** -9, 1 * 10 ** -1])
ax.set_xlabel('Mode Number', fontsize=16)
ax.set_ylabel('Variance', fontsize=16)  # ' (and Hor. Wavenumber)')
ax.set_title('PE: ALOHA, BATS, PAPA', fontsize=18)
plot_pro(ax)
# f.savefig("/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_05_17/ship_pe_energy_comp.jpg", dpi=300)

cmap = matplotlib.cm.get_cmap('Blues')
matplotlib.rcParams['figure.figsize'] = (12,8)
f, arm = plt.subplots(3, 3, sharex=True, sharey=True)
vmi = 0
vma = 2.5
arm[0, 0].pcolor(epsilon[0, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[0, 0].set_title('ALOHA mode k=0')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[0, 0].set_ylabel('Mode i')
arm[0, 1].pcolor(epsilon[1, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[0, 1].set_title('ALOHA mode k=1')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[0, 2].pcolor(epsilon[2, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[0, 2].set_title('ALOHA mode k=2')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[0, 0].invert_yaxis()

arm[1, 0].pcolor(epsilon_B[0, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[1, 0].set_title('BATS mode k=0')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[1, 0].set_ylabel('Mode i')
arm[1, 1].pcolor(epsilon_B[1, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[1, 1].set_title('BATS mode k=1')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[1, 2].pcolor(epsilon_B[2, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[1, 2].set_title('BATS mode k=2')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))

arm[2, 0].pcolor(epsilon_P[0, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[2, 0].set_title('PAPA mode k=0')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[2, 1].pcolor(epsilon_P[1, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[2, 1].set_title('PAPA mode k=1')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[2, 2].pcolor(epsilon_P[2, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[2, 2].set_title('PAPA mode k=2')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[2, 0].set_xlabel('Mode j')
arm[2, 0].set_ylabel('Mode i')
arm[2, 1].set_xlabel('Mode j')
arm[2, 2].set_xlabel('Mode j')

c_map_ax = f.add_axes([0.933, 0.1, 0.02, 0.8])
norm = matplotlib.colors.Normalize(vmin=vmi, vmax=vma)
cb1 = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=cmap, norm=norm, orientation='vertical')
cb1.set_label('Epsilon')
arm[2, 2].grid()
plot_pro(arm[2, 2])

f, ax = plt.subplots()
ax.plot(Gz[:, 0], grid, linestyle='--', color='r')
ax.plot(Gz[:, 1], grid, linestyle='--', color='b')
ax.plot(Gz[:, 2], grid, linestyle='--', color='g')
ax.plot(Gz[:, 3], grid, linestyle='--', color='m')
ax.plot(N2 * 30000, grid, color='k')
ax.plot(F[:, 0], grid, color='r')
ax.plot(F[:, 1], grid, color='b')
ax.plot(F[:, 2], grid, color='g')
ax.plot(F[:, 3], grid, color='m')
ax.set_title('HOTS Vertical Modes')
ax.invert_yaxis()
plot_pro(ax)

f, ax = plt.subplots()
ax.plot(N2, grid, label='HOTS', linewidth=2)
ax.plot(sta_bats_n2, SB['depth'], label='BATS', linewidth=2)
ax.plot(sta_papa_n2, SP['depth'], label='PAPA', linewidth=2)
ax.set_title('Station N2')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=14)
ax.set_ylim([0, 3000])
ax.invert_yaxis()
plot_pro(ax)

# --- SAVE ---
# write python dict to a file
sa = 1
if sa > 0:
    mydict = {'bin_depth': grid, 'N2': N2, 'PE': avg_PE, 'PE_per_prof': PE_per_mass, 'c': c, 'f': f, 'sig0': sig0, 'date': time}
    output = open('/Users/jake/Documents/baroclinic_modes/Shipboard/HOTS_92_10_2.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gsw
import pickle
from scipy.signal import savgol_filter
from mode_decompositions import vertical_modes
from toolkit import plot_pro


# load in N2 from various sites
omega = 0
mmax = 30
nmodes = mmax + 1
rho0 = 1025

# BATS DG (2015)
pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg035_energy.pkl', 'rb')
bats_dg = pickle.load(pkl_file)
pkl_file.close()
dg_depth = bats_dg['depth']
dg_N2 = bats_dg['N2']
DG_G, DG_Gz, DG_c, DG_epsilon = vertical_modes(dg_N2, dg_depth, omega, mmax)

# --------------------------------------------------------------------------------------------------------------------
# BATS SHIP
pkl_file = open('/Users/jake/Desktop/bats/station_bats_pe_apr11.pkl', 'rb')
SB = pickle.load(pkl_file)
pkl_file.close()
sta_bats_depth = SB['depth']
sta_bats_pe = SB['PE']
sta_bats_c = SB['c']
sta_bats_f = np.pi * np.sin(np.deg2rad(31.6)) / (12 * 1800)
sta_bats_dk = sta_bats_f / sta_bats_c[1]
sta_bats_n2 = SB['N2_per_season'][:, 2]
G_B, Gz_B, c_B, epsilon_B = vertical_modes(sta_bats_n2, SB['depth'], omega, mmax)
ratio = sta_bats_n2 / sta_bats_f
# --------------------------------------------------------------------------------------------------------------------
# ABACO SHIP
# just one year!!
pkl_file = open('/Users/jake/Documents/baroclinic_modes/SHIPBOARD/ABACO/ship_ladcp_2017-05-08.pkl', 'rb')
abaco_ship = pickle.load(pkl_file)
pkl_file.close()
ship_depth = abaco_ship['bin_depth']
ship_SA = abaco_ship['SA']
ship_CT = abaco_ship['CT']
ship_sig0 = abaco_ship['den_grid']
ship_dist = abaco_ship['den_dist']
ref_lat = 26
# Shipboard CTD N2
ship_p = gsw.p_from_z(-1 * ship_depth, ref_lat)
ship_N2 = np.nan*np.ones(len(ship_p))
# select profiles to average over away from DWBC
ship_N2[0:-1] = gsw.Nsquared(np.nanmean(ship_SA[:, ship_dist > 100], axis=1),
                             np.nanmean(ship_CT[:, ship_dist > 100], axis=1), ship_p, lat=ref_lat)[0]
ship_N2[1] = ship_N2[2] - 1*10**(-5)
ship_N2[0] = ship_N2[1] - 1*10**(-5)
ship_N2[ship_N2 < 0] = np.nan
for i in np.where(np.isnan(ship_N2))[0]:
    ship_N2[i] = ship_N2[i - 1] - 1*10**(-8)
this_ship_n2 = savgol_filter(ship_N2, 15, 3)
G_AB, Gz_AB, c_AB, epsilon_AB = vertical_modes(this_ship_n2, ship_depth, omega, mmax)
# --------------------------------------------------------------------------------------------------------------------
# HOTS
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Shipboard/HOTS_92_10.pkl', 'rb')
SH = pickle.load(pkl_file)
pkl_file.close()
sta_aloha_depth = SH['bin_depth']
sta_aloha_pe = SH['PE']
sta_aloha_c = SH['c']
sta_aloha_f = np.pi * np.sin(np.deg2rad(49.98)) / (12 * 1800)
sta_aloha_dk = sta_aloha_f / sta_aloha_c[1]
sta_aloha_n2 = SH['N2']
G_AL, Gz_AL, c_AL, epsilon_AL = vertical_modes(sta_aloha_n2, sta_aloha_depth, omega, mmax)
ratio_hots = sta_aloha_n2 / sta_aloha_f
# --------------------------------------------------------------------------------------------------------------------
# PAPA
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Line_P/canada_DFO/papa_energy_spectra_sept17.pkl', 'rb')
SP = pickle.load(pkl_file)
pkl_file.close()
sta_papa_depth = SP['depth']
sta_papa_time = SP['time']
sta_papa_pe = SP['PE']
sta_papa_c = SP['c']
sta_papa_f = np.pi * np.sin(np.deg2rad(49.98)) / (12 * 1800)
sta_papa_dk = sta_papa_f / sta_papa_c[1]

sta_papa_n2 = np.nan * np.ones(np.shape(SP['Sigma0']))
# f, ax = plt.subplots()
for i in range(np.shape(SP['Sigma0'])[1]):
    if sta_papa_time[i, 1] < 4:  # season selection because PAPA is strongly seasonal
        sta_papa_n2[0:-1, i] = gsw.Nsquared(SP['SA'][:, i], SP['CT'][:, i], SP['bin_press'], lat=ref_lat)[0]
        # ax.plot(sta_papa_n2[:, i], sta_papa_depth, linewidth=0.75)
# ax.set_ylim([0, 800])
# ax.invert_yaxis()
# ax.grid()
# plot_pro(ax)

# sta_papa_n2 = SP['N2']
papa_mean_corrected = np.nanmean(sta_papa_n2, axis=1)
papa_mean_corrected[-1] = papa_mean_corrected[-2]
G_P, Gz_P, c_P, epsilon_P = vertical_modes(papa_mean_corrected, SP['depth'], omega, mmax)
ratio_papa = sta_papa_n2 / sta_papa_f
# --------------------------------------------------------------------------------------------------------------------
# DEEP ARGO
# ATL
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Deep_Argo/float6025_mar17.pkl', 'rb')
abaco_argo = pickle.load(pkl_file)
pkl_file.close()
argo_depth = abaco_argo['bin_depth']
argo_N2 = abaco_argo['N2_avg']
argo_G, argo_Gz, argo_c, argo_epsilon = vertical_modes(argo_N2, argo_depth, omega, mmax)
# NZ
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Deep_Argo/float6036_oct17.pkl', 'rb')
abaco_argo = pickle.load(pkl_file)
pkl_file.close()
argo2_depth = abaco_argo['bin_depth']
argo2_N2 = abaco_argo['N2_avg']
argo2_G, argo2_Gz, argo2_c, argo2_epsilon = vertical_modes(argo2_N2, argo2_depth, omega, mmax)
# --------------------------------------------------------------------------------------------------------------------
# Constant N2
const_G, const_Gz, const_c, epsilon_const = vertical_modes(0.00005 * np.ones(np.shape(argo_depth)),
                                                           argo_depth, omega, mmax)

# --------------------------------------------------------------------------------------------------------------------
# PLOTTING
f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
ax0.plot(sta_aloha_n2, sta_aloha_depth, label='ALOHA')
ax0.plot(this_ship_n2, ship_depth, label='ABACO')
ax0.plot(sta_bats_n2, sta_bats_depth, label='BATS Ship')
ax0.plot(papa_mean_corrected, sta_papa_depth, label='PAPA')
# ax0.plot(argo_N2, argo_depth, label='Argo Atl')
# ax0.plot(argo2_N2, argo2_depth, label='Argo NZ')
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles, labels, fontsize=10)

ax1.plot(Gz_AL[:, 1], sta_aloha_depth, label='ALOHA (22$^{\circ}$N)')
ax1.plot(Gz_AB[:, 1], ship_depth, label='ABACO (26.5$^{\circ}$N)')
ax1.plot(Gz_B[:, 1], sta_bats_depth, label='BATS Ship (32$^{\circ}$N)')
ax1.plot(Gz_P[:, 1], sta_papa_depth, label=r'PAPA (50$^{\circ}$N)')
# ax1.plot(argo_Gz[:, 1], argo_depth, label='Argo Atl')
# ax1.plot(argo2_Gz[:, 1], argo2_depth, label='Argo NZ')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=10)

ax0.set_ylim([0, 2000])
ax0.set_ylabel('Depth')
ax0.invert_yaxis()
ax0.grid()
ax0.grid()
ax1.grid()
plot_pro(ax1)


cmap = matplotlib.cm.get_cmap('Blues')
f, arm = plt.subplots(3, 7, sharex=True, sharey=True)
vmi = 0
vma = 2.5
fs = 8
arm[0, 0].pcolor(epsilon_const[0, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[0, 0].set_title('Const. N2 mode 0', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[0, 1].pcolor(epsilon_AL[0, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[0, 1].set_title('ALOHA mode 0', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[0, 2].pcolor(epsilon_P[0, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[0, 2].set_title('PAPA mode 0', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[0, 3].pcolor(epsilon_B[0, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[0, 3].set_title('BATS Ship mode 0', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[0, 4].pcolor(epsilon_AB[0, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[0, 4].set_title('ABACO Ship mode 0', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[0, 5].pcolor(argo_epsilon[0, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[0, 5].set_title('Deep Argo Atl mode 0', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[0, 6].pcolor(argo2_epsilon[0, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[0, 6].set_title('Deep Argo NZ mode 0', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[0, 0].invert_yaxis()

arm[1, 0].pcolor(epsilon_const[1, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[1, 0].set_title('Const. N2 mode 1', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[1, 1].pcolor(epsilon_AL[1, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[1, 1].set_title('ALOHA mode 1', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[1, 2].pcolor(epsilon_P[1, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[1, 2].set_title('PAPA mode 1', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[1, 3].pcolor(epsilon_B[1, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[1, 3].set_title('BATS Ship mode 1', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[1, 4].pcolor(epsilon_AB[1, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[1, 4].set_title('ABACO Ship mode 1', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[1, 5].pcolor(argo_epsilon[1, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[1, 5].set_title('Deep Argo Atl mode 1', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[1, 6].pcolor(argo2_epsilon[1, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[1, 6].set_title('Deep Argo NZ mode 1', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))

arm[2, 0].pcolor(epsilon_const[2, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[2, 0].set_title('Const. N2 mode 2', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[2, 1].pcolor(epsilon_AL[2, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[2, 1].set_title('ALOHA mode 2', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[2, 2].pcolor(epsilon_P[2, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[2, 2].set_title('PAPA mode 2', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[2, 3].pcolor(epsilon_B[2, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[2, 3].set_title('BATS Ship mode 2', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[2, 4].pcolor(epsilon_AB[2, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[2, 4].set_title('ABACO Ship mode 2', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[2, 5].pcolor(argo_epsilon[2, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[2, 5].set_title('Deep Argo Atl mode 2', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
arm[2, 6].pcolor(argo2_epsilon[2, :, :], cmap=cmap, vmin=vmi, vmax=vma)
arm[2, 6].set_title('Deep Argo NZ mode 2', fontsize=fs)
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))


c_map_ax = f.add_axes([0.925, 0.1, 0.02, 0.8])
norm = matplotlib.colors.Normalize(vmin=vmi, vmax=vma)
cb1 = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=cmap, norm=norm, orientation='vertical')
cb1.set_label('Epsilon')
arm[2, 5].grid()
plot_pro(arm[2, 5])



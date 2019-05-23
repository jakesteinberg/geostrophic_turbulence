import numpy as np
import scipy.io as si
from scipy.integrate import cumtrapz
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gsw
import pickle
from scipy.signal import savgol_filter
from mode_decompositions import vertical_modes, vertical_modes_f, PE_Tide_GM
from toolkit import plot_pro, find_nearest


# load in N2 from various sites
omega = 0
mmax = 30
nmodes = mmax + 1
rho0 = 1025

# DG041 36N (2018)
pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg041_2018_initial_processing.pkl', 'rb')
dg_36n = pickle.load(pkl_file)
pkl_file.close()
dg_36n_depth = dg_36n['depth']
dg_36n_N2 = np.nanmean(dg_36n['N2_by_season_quad'][:, 1:], axis=1)
DG_36n_G, DG_36n_Gz, DG_36n_c, DG_36n_epsilon = vertical_modes(dg_36n_N2, dg_36n_depth, omega, mmax)
# --------------------------------------------------------------------------------------------------------------------
# BATS DG (2015)
pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg035_2015_initial_processing.pkl', 'rb')
bats_dg = pickle.load(pkl_file)
pkl_file.close()
dg_depth = bats_dg['depth']
dg_N2 = bats_dg['N2_by_season_quad']
ref_lat = bats_dg['ref_lat']
bats_f = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)
# DG_G, DG_Gz, DG_c, DG_epsilon = vertical_modes(dg_N2, dg_depth, omega, mmax)

# winter1
dg_bats_n2_0 = np.nanmean(bats_dg['N2_by_season_quad'][:, 0:4], axis=1)
G_B_0, Gz_B_0, c_B_0, epsilon_B_0 = vertical_modes(dg_bats_n2_0, bats_dg['depth'], omega, mmax)
# summer
dg_bats_n2_1 = np.nanmean(bats_dg['N2_by_season_quad'][:, 4:8], axis=1)
G_B_1, Gz_B_1, c_B_1, epsilon_B_1 = vertical_modes(dg_bats_n2_1, bats_dg['depth'], omega, mmax)
# winter2 (Fall)
dg_bats_n2_2 = np.nanmean(bats_dg['N2_by_season_quad'][:, 8:], axis=1)
G_B_2, Gz_B_2, c_B_2, epsilon_B_2 = vertical_modes(dg_bats_n2_2, bats_dg['depth'], omega, mmax)
ratio = dg_bats_n2_2 / bats_f
# --------------------------------------------------------------------------------------------------------------------
# BATS SHIP
pkl_file = open('/Users/jake/Desktop/bats/station_bats_pe_jan30.pkl', 'rb')  # jan 30 2018
SB = pickle.load(pkl_file)
pkl_file.close()
sta_bats_depth = SB['depth']
sta_bats_pe = SB['PE']
sta_bats_c = SB['c']
sta_bats_f = np.pi * np.sin(np.deg2rad(31.6)) / (12 * 1800)
sta_bats_dk = sta_bats_f / sta_bats_c[1]

# # winter1
sta_bats_n2_0 = SB['N2_per_season'][:, 0]
sta_G_B_0, sta_Gz_B_0, sta_c_B_0, sta_epsilon_B_0 = vertical_modes(sta_bats_n2_0, SB['depth'], omega, mmax)
# # summer
sta_bats_n2_1 = SB['N2_per_season'][:, 1]
sta_G_B_1, sta_Gz_B_1, sta_c_B_1, sta_epsilon_B_1 = vertical_modes(sta_bats_n2_1, SB['depth'], omega, mmax)
# # winter2 (Fall)
# sta_bats_n2_2 = SB['N2_per_season'][:, 2]
# G_B_2, Gz_B_2, c_B_2, epsilon_B_2 = vertical_modes(sta_bats_n2_2, SB['depth'], omega, mmax)
# ratio = sta_bats_n2_2 / sta_bats_f

# --- compute alternate vertical modes (allowing for a sloping bottom)
bc_bot = 1  # 1 = flat, 2 = rough
slope = 5.0*10**(-3)
grid2 = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4500, 10)])
n2_interp = np.interp(grid2, SB['depth'], sta_bats_n2_0)
n2_interp[0] = n2_interp[1] - 0.000001
F_int_g2, F_g2, c_ff, norm_constant, epsilon2 = vertical_modes_f(n2_interp, grid2, omega, mmax, bc_bot, 31.5, slope)
F = np.nan * np.ones((np.size(SB['depth']), mmax + 1))
F_int = np.nan * np.ones((np.size(SB['depth']), mmax + 1))
for i in range(mmax + 1):
    F[:, i] = np.interp(SB['depth'], grid2, F_g2[:, i])
    F_int[:, i] = np.interp(SB['depth'], grid2, F_int_g2[:, i])

plt.rcParams['figure.figsize'] = 11, 5.5
f, (ax1, ax15, ax2, ax3) = plt.subplots(1, 4, sharey=True)
ax1.plot(sta_bats_n2_0 * 40000000, sta_bats_depth, label=r'N$^2$ $\times$ $ \left( 4 \times 10^{7} \right) $',color='k')
ax1.plot(sta_G_B_0[:, 1], sta_bats_depth, label='m=1', color='#228B22', linewidth=1.5)
ax1.plot(sta_G_B_0[:, 2], sta_bats_depth, label='m=2', color='#1E90FF', linewidth=1.5)
ax1.plot(sta_G_B_0[:, 3], sta_bats_depth, label='m=3', color='#FF4500', linewidth=1.5)
ax1.plot(sta_G_B_0[:, 4], sta_bats_depth, label='m=4', color='#778899', linewidth=1.5)
ax1.set_xlim([-3000, 3000])
ax1.set_ylim([0, 4800])
ax1.set_title(r'G$_{m}$(z) (Flat Bottom)')
ax1.set_ylabel('Depth [m]')
ax1.set_xlabel('Normalized Mode Amplitude')
ax15.plot(F_int[:, 1], sta_bats_depth, color='k')
ax15.plot(F_int[:, 2], sta_bats_depth, color='#228B22', linewidth=1.5)
ax15.plot(F_int[:, 3], sta_bats_depth, color='#1E90FF', linewidth=1.5)
ax15.plot(F_int[:, 4], sta_bats_depth, color='#FF4500', linewidth=1.5)
ax15.set_title(r'G$_{m}$(z) (Slope = $5*10^{-3}$)')
ax15.set_xlabel('Normalized Mode Amplitude')
ax15.grid()

ax2.plot(sta_Gz_B_0[:, 0], sta_bats_depth, color='k')
ax2.plot(sta_Gz_B_0[:, 1], sta_bats_depth, color='#228B22', linewidth=1.5)
ax2.plot(sta_Gz_B_0[:, 2], sta_bats_depth, color='#1E90FF', linewidth=1.5)
ax2.plot(sta_Gz_B_0[:, 3], sta_bats_depth, color='#FF4500', linewidth=1.5)
ax3.plot(F[:, 0], sta_bats_depth, color='k')
ax3.plot(F[:, 1], sta_bats_depth, color='#228B22', linewidth=1.5)
ax3.plot(F[:, 2], sta_bats_depth, color='#1E90FF', linewidth=1.5)
ax3.plot(F[:, 3], sta_bats_depth, color='#FF4500', linewidth=1.5)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=8)
ax1.invert_yaxis()
ax1.grid()
ax2.set_xlim([-3, 3])
ax2.set_title(r'F$_{m}$(z) (Flat Bottom)')
ax2.set_xlabel('Normalized Mode Amplitude')
ax2.grid()
ax3.set_title(r'F$_{m}$(z) (Slope = $5*10^{-3}$)')
ax3.set_xlabel('Normalized Mode Amplitude')
plot_pro(ax3)
# f.savefig("/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_04_18/mode_shapes.jpg", dpi=300)

test = sta_bats_n2_0 * (sta_G_B_0[:, 1]**2) /(sta_bats_c[1]**2)
test2 = sta_bats_n2_0 * (sta_G_B_0[:, 2]**2) /(sta_bats_c[2]**2)
h = np.nanmax(sta_bats_depth)
normG = (1/h)*np.trapz(test, sta_bats_depth)
normG_pr = (1/h)*np.trapz(sta_Gz_B_0[:, 1]**2, sta_bats_depth)
normG_2 = (1/h)*np.trapz(test2, sta_bats_depth)
normG_pr_2 = (1/h)*np.trapz(sta_Gz_B_0[:, 2]**2, sta_bats_depth)
# --------------------------------------------------------------------------------------------------------------------
# ABACO SHIP
# just one year!!
# pkl_file = open('/Users/jake/Documents/baroclinic_modes/SHIPBOARD/ABACO/ship_ladcp_2017-05-08.pkl', 'rb')
# abaco_ship = pickle.load(pkl_file)
# pkl_file.close()
# ship_depth = abaco_ship['bin_depth']
# ship_SA = abaco_ship['SA']
# ship_CT = abaco_ship['CT']
# ship_sig0 = abaco_ship['den_grid']
# ship_dist = abaco_ship['den_dist']
# ref_lat = 26
# # Shipboard CTD N2
# ship_p = gsw.p_from_z(-1 * ship_depth, ref_lat)
# ship_N2 = np.nan*np.ones(len(ship_p))
# # select profiles to average over away from DWBC
# ship_N2[0:-1] = gsw.Nsquared(np.nanmean(ship_SA[:, ship_dist > 100], axis=1),
#                              np.nanmean(ship_CT[:, ship_dist > 100], axis=1), ship_p, lat=ref_lat)[0]
# ship_N2[1] = ship_N2[2] - 1*10**(-5)
# ship_N2[0] = ship_N2[1] - 1*10**(-5)
# ship_N2[ship_N2 < 0] = np.nan
# for i in np.where(np.isnan(ship_N2))[0]:
#     ship_N2[i] = ship_N2[i - 1] - 1*10**(-8)

pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/background_profiles.pkl', 'rb')
abaco_dg = pickle.load(pkl_file)
pkl_file.close()
this_ship_n2 = savgol_filter(abaco_dg['N2_avg'][:], 5, 3)
G_AB, Gz_AB, c_AB, epsilon_AB = vertical_modes(this_ship_n2, abaco_dg['depth'][:], omega, mmax)
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
    if sta_papa_time[i, 1] > 7:  # season selection because PAPA is strongly seasonal
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
# DEEP ARGO (ATLANTIC)
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
matplotlib.rcParams['figure.figsize'] = (14, 6)
f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharey=True)
ax0.plot(sta_aloha_n2, sta_aloha_depth, label='ALOHA', color='#B22222')
ax0.plot(this_ship_n2, abaco_dg['depth'][:], label='ABACO', color='#FF8C00')
ax0.plot(sta_bats_n2_1, sta_bats_depth, label='BATS Ship Sum.', color='#00FF7F')
ax0.plot(sta_bats_n2_0, sta_bats_depth, label='BATS Ship Win.', color='k')
# ax0.plot(dg_36n_N2, dg_36n_depth, label='36$^{\circ}$N DG Fall', color='k')
ax0.plot(papa_mean_corrected, sta_papa_depth, label='PAPA', color='#4682B4')
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles, labels, fontsize=10)

ax1.plot(Gz_AL[:, 1], sta_aloha_depth, label='ALOHA (22$^{\circ}$N) (Mode$_{111}$ IC = ' + str(np.round(epsilon_AL[1, 1, 1], 2)) + ')', color='#B22222')
ax1.plot(Gz_AB[:, 1], abaco_dg['depth'][:], label='ABACO (26.5$^{\circ}$N) (Mode$_{111}$ IC = ' + str(np.round(epsilon_AB[1, 1, 1], 2)) + ')', color='#FF8C00')
ax1.plot(sta_Gz_B_1[:, 1], sta_bats_depth, label='BATS$_{DG}$ Sum. (32$^{\circ}$N) (Mode$_{111}$ IC = ' + str(np.round(sta_epsilon_B_1[1, 1, 1], 2)) + ')', color='#00FF7F')
ax1.plot(sta_Gz_B_0[:, 1], sta_bats_depth, label='BATS$_{DG}$ Win. (32$^{\circ}$N) (Mode$_{111}$ IC = ' + str(np.round(epsilon_B_0[1, 1, 1], 2)) + ')', color='k')
# ax1.plot(DG_36n_Gz[:, 1], dg_36n_depth, label='36$^{\circ}$N DG Fall (Mode$_{111}$ IC = ' + str(np.round(epsilon_B_1[1, 1, 1], 2)) + ')', color='k')
ax1.plot(Gz_P[:, 1], sta_papa_depth, label=r'PAPA Sum. (50$^{\circ}$N) (Mode$_{111}$ IC = ' + str(np.round(epsilon_P[1, 1, 1], 2)) + ')', color='#4682B4')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=8)

ax2.plot(Gz_AL[:, 2], sta_aloha_depth, label='ALOHA (22$^{\circ}$N) (Mode$_{222}$ IC = ' + str(np.round(epsilon_AL[2, 2, 2], 2)) + ')', color='#B22222')
ax2.plot(Gz_AB[:, 2], abaco_dg['depth'][:], label='ABACO (26.5$^{\circ}$N) (Mode$_{222}$ IC = ' + str(np.round(epsilon_AB[2, 2, 2], 2)) + ')', color='#FF8C00')
ax2.plot(sta_Gz_B_1[:, 2], sta_bats_depth, label='BATS$_{DG}$ Sum. (32$^{\circ}$N) (Mode$_{222}$ IC = ' + str(np.round(sta_epsilon_B_1[2, 2, 2], 2)) + ')', color='#00FF7F')
ax2.plot(sta_Gz_B_0[:, 2], sta_bats_depth, label='BATS$_{DG}$ Win. (32$^{\circ}$N) (Mode$_{222}$ IC = ' + str(np.round(epsilon_B_0[2, 2, 2], 2)) + ')', color='k')
# ax2.plot(DG_36n_Gz[:, 2], dg_36n_depth, label='36$^{\circ}$N DG Fall (Mode$_{222}$ IC = ' + str(np.round(epsilon_B_1[2, 2, 2], 2)) + ')', color='k')
ax2.plot(Gz_P[:, 2], sta_papa_depth, label=r'PAPA Sum. (50$^{\circ}$N) (Mode$_{222}$ IC = ' + str(np.round(epsilon_P[2, 2, 2], 2)) + ')', color='#4682B4')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=8)

ax3.plot(Gz_AL[:, 3], sta_aloha_depth, label='ALOHA (22$^{\circ}$N) (Mode$_{333}$ IC = ' + str(np.round(epsilon_AL[3, 3, 3], 2)) + ')', color='#B22222')
ax3.plot(Gz_AB[:, 3], abaco_dg['depth'][:], label='ABACO (26.5$^{\circ}$N) (Mode$_{333}$ IC = ' + str(np.round(epsilon_AB[3, 3, 3], 2)) + ')', color='#FF8C00')
ax3.plot(sta_Gz_B_1[:, 3], sta_bats_depth, label='BATS$_{DG}$ Sum. (32$^{\circ}$N) (Mode$_{333}$ IC = ' + str(np.round(sta_epsilon_B_1[3, 3, 3], 2)) + ')', color='#00FF7F')
ax3.plot(sta_Gz_B_0[:, 3], sta_bats_depth, label='BATS$_{DG}$ Win. (32$^{\circ}$N) (Mode$_{333}$ IC = ' + str(np.round(epsilon_B_0[3, 3, 3], 2)) + ')', color='k')
# ax3.plot(DG_36n_Gz[:, 3], dg_36n_depth, label='36$^{\circ}$N DG Fall (Mode$_{333}$ IC = ' + str(np.round(epsilon_B_1[3, 3, 3], 2)) + ')', color='k')
ax3.plot(Gz_P[:, 3], sta_papa_depth, label=r'PAPA Sum. (50$^{\circ}$N) (Mode$_{333}$ IC = ' + str(np.round(epsilon_P[3, 3, 3], 2)) + ')', color='#4682B4')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels, fontsize=8)

ax0.set_ylim([0, 2000])
ax0.set_ylabel('Depth [m]')
ax0.set_xlabel(r'N$^2$ [s$^{-2}$]')
ax1.set_title('Mode 1 Strucutre')
ax2.set_title('Mode 2 Strucutre')
ax3.set_title('Mode 2 Strucutre')
ax0.set_title('Variations in Buoyancy Frequency')
ax0.invert_yaxis()
ax0.grid()
ax1.grid()
ax2.grid()
plot_pro(ax3)
f.savefig("/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_05_17/site_n2_mode1_interaction_shapes.jpg", dpi=300)
# ---------------------------------------------------------------------------------------------------------------------
# Block Diagram of Interaction Coefficients
cmap = matplotlib.cm.get_cmap('hot_r')

matplotlib.rcParams['figure.figsize'] = (18, 6)

vmi = 0
vma = 2.75
fs = 10
modes = [0, 1, 2, 3]
mode_labs = '0', '1', '2', '3', '4'
epsils = [epsilon_const, epsilon_AL, epsilon_P, epsilon_AB, DG_36n_epsilon, epsilon_B_0, epsilon_B_1]
epsils_labs = ['Constant N$^2$', 'ALOHA', 'PAPA Sum.', 'ABACO', '36$^{\circ}$N Win.','BATS Win.', 'BATS Sum.']
# epsils = [epsilon_const, epsilon_B_0, epsilon_B_1]
# epsils_labs = ['Constant N$^2$','BATS Winter', 'BATS Summer']
f, arm = plt.subplots(3, len(epsils_labs), sharex=True, sharey=True)
for i in range(len(epsils)):
    # arm[0, i].pcolor(epsils[i][modes[0], :, :], cmap=cmap, vmin=vmi, vmax=vma)
    # arm[0, i].set_title(epsils_labs[i] + ', i=' + str(modes[0]), fontsize=fs)
    # plt.xticks(np.arange(0.5, len(mode_labs) + .5, 1), mode_labs)
    # plt.yticks(np.arange(0.5, len(mode_labs) + .5, 1), mode_labs)

    arm[0, i].pcolor(epsils[i][modes[1], :, :], cmap=cmap, vmin=vmi, vmax=vma)
    arm[0, i].set_title(epsils_labs[i] + ', i=' + str(modes[1]), fontsize=fs)
    plt.xticks(np.arange(0.5, len(mode_labs) + .5, 1), mode_labs)
    plt.yticks(np.arange(0.5, len(mode_labs) + .5, 1), mode_labs)

    arm[1, i].pcolor(epsils[i][modes[2], :, :], cmap=cmap, vmin=vmi, vmax=vma)
    arm[1, i].set_title(epsils_labs[i] + ', i=' + str(modes[2]), fontsize=fs)
    plt.xticks(np.arange(0.5, len(mode_labs) + .5, 1), mode_labs)
    plt.yticks(np.arange(0.5, len(mode_labs) + .5, 1), mode_labs)

    arm[2, i].pcolor(epsils[i][modes[3], :, :], cmap=cmap, vmin=vmi, vmax=vma)
    arm[2, i].set_title(epsils_labs[i] + ', i=' + str(modes[3]), fontsize=fs)
    plt.xticks(np.arange(0.5, len(mode_labs) + .5, 1), mode_labs)
    plt.yticks(np.arange(0.5, len(mode_labs) + .5, 1), mode_labs)
arm[0, 0].invert_yaxis()
arm[0, 0].set_ylabel('j', fontsize=fs)
arm[1, 0].set_ylabel('j', fontsize=fs)
arm[2, 0].set_ylabel('j', fontsize=fs)
# arm[3, 0].set_ylabel('j', fontsize=fs)
arm[2, 0].set_xlabel('m', fontsize=fs)
arm[2, 1].set_xlabel('m', fontsize=fs)
arm[2, 2].set_xlabel('m', fontsize=fs)
arm[2, 3].set_xlabel('m', fontsize=fs)
arm[2, 4].set_xlabel('m', fontsize=fs)
c_map_ax = f.add_axes([0.925, 0.1, 0.02, 0.8])
norm = matplotlib.colors.Normalize(vmin=vmi, vmax=vma)
cb1 = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=cmap, norm=norm, orientation='vertical')
cb1.set_label('Epsilon')
arm[2, len(epsils_labs)-1].grid()
plot_pro(arm[2, len(epsils_labs)-1])
# f.savefig("/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_05_17/mode_interactions_blocks.jpg", dpi=300)

# ---------------------------------------------------------------------------------------------------------------------
# PE across geographic locations
# --- ABACO ---
pkl_file = open('/Users/jake/Documents/geostrophic_turbulence/ABACO_2017_energy.pkl', 'rb')
abaco_energies = pickle.load(pkl_file)
pkl_file.close()

# --- PAPA ---
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
for i in range(np.shape(SP['Sigma0'])[1]):
    if sta_papa_time[i, 1] < 4:  # season selection because PAPA is strongly seasonal
        sta_papa_n2[0:-1, i] = gsw.Nsquared(SP['SA'][:, i], SP['CT'][:, i], SP['bin_press'], lat=ref_lat)[0]
papa_mean_corrected = np.nanmean(sta_papa_n2, axis=1)
papa_mean_corrected[-1] = papa_mean_corrected[-2]
PE_SD_papa, PE_GM_papa, GMPE_papa, GMKE_papa = PE_Tide_GM(rho0, sta_papa_depth,
                                                          np.shape(sta_papa_pe)[0], papa_mean_corrected[:, None], sta_papa_f)

# --- HOTS ---
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Shipboard/HOTS_92_10.pkl', 'rb')
SH = pickle.load(pkl_file)
pkl_file.close()
sta_aloha_depth = SH['bin_depth']
sta_aloha_pe = SH['PE']
sta_aloha_c = SH['c']
sta_aloha_f = np.pi * np.sin(np.deg2rad(49.98)) / (12 * 1800)
sta_aloha_dk = sta_aloha_f / sta_aloha_c[1]
sta_aloha_n2 = SH['N2']
PE_SD_aloha, PE_GM_aloha, GMPE_aloha, GMKE_aloha = PE_Tide_GM(rho0, sta_aloha_depth,
                                                              len(sta_aloha_pe), sta_aloha_n2[:, None], sta_aloha_f)

# --- BATS ---
pkl_file = open('/Users/jake/Desktop/bats/station_bats_pe_jan30.pkl', 'rb')  # update jan 2019
SB = pickle.load(pkl_file)
pkl_file.close()
sta_bats_pe = SB['PE_by_season']
sta_bats_c = SB['c']
sta_bats_depth = SB['depth']
sta_bats_f = np.pi * np.sin(np.deg2rad(31.6)) / (12 * 1800)
sta_bats_dk = sta_bats_f / sta_bats_c[1]
sta_bats_pe_total = np.nanmean(np.concatenate((np.nanmean(sta_bats_pe[0], axis=1)[:, None],
                                    np.nanmean(sta_bats_pe[1], axis=1)[:, None],
                                    np.nanmean(sta_bats_pe[2], axis=1)[:, None]), axis=1), axis=1)
sta_bats_n2_1 = SB['N2_per_season'][:, 1]
PE_SD_bats, PE_GM_bats, GMPE_bats, GMKE_bats = PE_Tide_GM(rho0, sta_bats_depth,
                                                          len(sta_bats_pe_total), sta_bats_n2_1[:, None], sta_bats_f)

# --- BATS DG
pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg035_2015_energy_may2019.pkl', 'rb')
DGB = pickle.load(pkl_file)
pkl_file.close()
bats_dg_KE = DGB['KE']
bats_dg_PE = DGB['PE']
bats_dg_bckgrds = DGB['background_eddy_indicies_for_energy'][0]  # winter index
bats_dg_KE_all = np.nanmean(DGB['KE_all'][:, bats_dg_bckgrds], axis=1)
bats_dg_PE_all = np.nanmean(DGB['PE_all'][:, bats_dg_bckgrds], axis=1)
bats_dg_c = DGB['c']
bats_dg_f = DGB['f']
bats_dg_depth = DGB['depth']
bats_dg_GMKE = DGB['GMKE']
bats_dg_GMPE = DGB['GMPE']
dk_bats = bats_dg_f / bats_dg_c[1]
sc_x_bats = 1000 * bats_dg_f / bats_dg_c[1:]

# --- 36N DG
pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg041_2018_energy.pkl', 'rb')
DG36 = pickle.load(pkl_file)
pkl_file.close()
dg_36n_KE = DG36['KE']
dg_36n_PE = DG36['PE']
dg_36n_bckgrds = DG36['background_eddy_indicies_for_energy'][1]  # winter index
dg_36n_KE_all = np.nanmean(DG36['KE_all'][:, dg_36n_bckgrds], axis=1)
dg_36n_PE_all = np.nanmean(DG36['PE_all'][:, dg_36n_bckgrds], axis=1)
dg_36n_c = DG36['c']
dg_36n_f = DG36['f']
dg_36n_depth = DG36['depth']
dg_36n_GMKE = DG36['GMKE']
dg_36n_GMPE = DG36['GMPE']
dk_36n = dg_36n_f / dg_36n_c[1]
sc_x_36n = 1000 * dg_36n_f / dg_36n_c[1:]

mode_num = np.arange(1, 61, 1)
# --- PLOT
# PE and KE by mode number for DG missions, BATS, 36N, ABACO
matplotlib.rcParams['figure.figsize'] = (7, 6)
scols = ['#00BFFF', '#6B8E23', '#800000']
fig01, ax0 = plt.subplots()
mode_num = np.arange(1, 61, 1)
# PE_ab = ax0.plot(mode_num, abaco_energies['avg_PE'][1:] / (abaco_energies['f_ref'] / abaco_energies['c'][1]),
#                  label='ABACO')
PE_bats = ax0.plot(mode_num[0:45], bats_dg_PE_all[1:] / dk_bats, label='BATS', linewidth=2, color=scols[0])
PE_36n = ax0.plot(mode_num[0:40], dg_36n_PE_all[1:] / dk_36n, label='36N', linewidth=2, color=scols[1])
KE_bats = ax0.plot(mode_num[0:45], bats_dg_KE_all[1:] / dk_bats, label='BATS',
                   linewidth=2, color=scols[0], linestyle='--')
KE_36n = ax0.plot(mode_num[0:40], dg_36n_KE_all[1:] / dk_36n, label='36N',
                  linewidth=2, color=scols[1], linestyle='--')
ax0.plot([10 ** 0, 10 ** 2], [10 ** 3, 10 ** -3], color='k', linewidth=0.5)
ax0.plot([10 ** 0, 10 ** 3], [3 * 10 ** 2, 3 * 10 ** -4], color='k', linewidth=0.5)
ax0.text(8 * 10 ** 1, 6 * 10 ** -2, '-2', fontsize=10)
ax0.text(8 * 10 ** 1, 3 * 10 ** -3, '-3', fontsize=10)

ax0.set_xlabel('Mode Number', fontsize=14)
ax0.set_ylabel('Spectral Density', fontsize=14)
ax0.set_title('Potential Energy Spectra (Across DG Missions)', fontsize=16)
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.axis([8 * 10 ** -1, 10 ** 2, 10 ** (-3), 1 * 10 ** 3])
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles, labels, fontsize=12)
plot_pro(ax0)

# PE (by mode number) COMPARISON BY SITE
matplotlib.rcParams['figure.figsize'] = (7, 6)
scols = ['#00BFFF', '#6B8E23', '#800000']
fig01, ax0 = plt.subplots()
mode_num = np.arange(1, 61, 1)
PE_sta_p1 = ax0.plot(mode_num, sta_bats_pe_total[1:] / sta_bats_dk, label='BATS', linewidth=2, color=scols[0])
PE_sta_hots = ax0.plot(mode_num, sta_aloha_pe[1:] / sta_aloha_dk, label='ALOHA', linewidth=2, color=scols[1])
PE_sta_papa = ax0.plot(mode_num, np.nanmean(sta_papa_pe[1:], axis=1) / sta_papa_dk, label='PAPA', linewidth=2,
                       color=scols[2])
ax0.plot(mode_num, PE_GM_bats / sta_bats_dk, linewidth=0.75, linestyle='--', color=scols[0])
ax0.plot(mode_num, PE_GM_aloha / sta_aloha_dk, linewidth=0.75, linestyle='--', color=scols[1])
ax0.plot(mode_num, PE_GM_papa / sta_papa_dk, linewidth=0.75, linestyle='--', color=scols[2])

ax0.plot([10 ** 0, 10 ** 2], [10 ** 3, 10 ** -3], color='k', linewidth=0.5)
ax0.plot([10 ** 0, 10 ** 3], [3 * 10 ** 2, 3 * 10 ** -4], color='k', linewidth=0.5)
ax0.text(8 * 10 ** 1, 6 * 10 ** -2, '-2', fontsize=10)
ax0.text(8 * 10 ** 1, 3 * 10 ** -3, '-3', fontsize=10)

ax0.set_xlabel('Mode Number', fontsize=14)
ax0.set_ylabel('Spectral Density', fontsize=14)
ax0.set_title('Potential Energy Spectra (Site Comparison)', fontsize=16)
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.axis([8 * 10 ** -1, 10 ** 2, 10 ** (-3), 1 * 10 ** 3])
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles, labels, fontsize=12)
plot_pro(ax0)

# ---------------------------------------------------------------------------------------------------------------------
# --- Internal Wave Computations / Testing of new formulation
Depth = sta_bats_depth.copy()
N2 = sta_bats_n2_0.copy()
f_ref = sta_bats_f.copy()
modenum = np.arange(0, 61)
Navg = np.trapz(np.sqrt(N2), Depth) / Depth[-1]

dk = f_ref / sta_aloha_c[1]
sc_x = 1000 * f_ref / sta_aloha_c[1:]

omega = np.arange(np.round(f_ref, 5), np.round(Navg, 5), 0.00001)
if omega[0] < f_ref:
    omega[0] = f_ref.copy() + 0.000001

bGM = 1300  # GM internal wave depth scale [m]
N0_GM = 5.2e-3  # GM N scale [s^-1];
jstar = 3  # GM vertical mode number scale
EGM = 6.3e-5  # GM energy level [no dimensions]

HHterm = 1 / (modenum[1:] * modenum[1:] + jstar * jstar)
HH = HHterm / np.sum(HHterm)

BBterm = (2 / 3.14159) * (f_ref / omega) * (1 / np.sqrt((omega**2) - (f_ref**2)))
BBint = np.trapz(BBterm, omega)
BBterm2 = (1/BBint) * BBterm

EE = np.tile(BBterm2[:, None], (1, len(modenum) - 1)) * np.tile(HH[None, :], (len(omega), 1)) * EGM
omega_g = np.tile(omega[:, None], (1, len(modenum) - 1))
FPE = (1/2) * (Navg**2) * (bGM**2 * N0_GM * (1 / Navg) * (omega_g**2 - f_ref**2) * (1 / (omega_g**2)) * EE)
FKE = (1/2) * bGM * bGM * N0_GM * Navg * (omega_g**2 + f_ref**2) * (1 / (omega_g**2)) * EE

PE_GM = bGM * bGM * N0_GM * Navg * HH * EGM / 2

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(omega_g, np.tile(modenum[None, 1:], (len(omega), 1)), FPE)
# ax.set_zlim([10**-2, 10**0])
# ax.set_ylim([10**0, 6 * 10**1])
# ax.zaxis._set_scale('log')
# ax.set_xlabel('Frequency')
# ax.set_ylabel('Mode Number')
# ax.set_zlabel('Energy')
# ax.view_init(20, -20)
# plot_pro(ax)

FPE_int = np.nan * np.ones(np.shape(FPE[0, :]))
FKE_int = np.nan * np.ones(np.shape(FPE[0, :]))
for i in range(len(modenum[1:])):
    FPE_int[i] = np.trapz(FPE[:, i], omega)
    FKE_int[i] = np.trapz(FKE[:, i], omega)

# f, ax = plt.subplots()
# ax.plot(sc_x, PE_GM / dk, linestyle='--', color='r', linewidth=0.75)
# ax.plot(sc_x, FPE_int / dk, color='r', linewidth=0.75)
# ax.plot(sc_x, FKE_int / dk, color='g', linewidth=0.75)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plot_pro(ax)

# ---------------------------------------------------------------------------------------------------------------------
# ---- energy in vertical modes from paper (Manita Chouksey thesis)
# Ri = np.array([3, 13, 377, 915])
# modes = np.arange(0, 5)
# r3_KE = np.array([5236, 2481.48, 861.81, 329.12, 267.72]) * np.array([1, 0.66, 0.33, 0.4, 0.26])
# r3_PE = np.array([5236, 2481.48, 861.81, 329.12, 267.72]) * (1 - np.array([1, 0.66, 0.33, 0.4, 0.26]))
# r13_KE = np.array([1553, 268, 124, 49.9, 40.6]) * np.array([1, 0.51, 0.22, 0.26, 0.16])
# r13_PE = np.array([1553, 268, 124, 49.9, 40.6]) * (1 - np.array([1, 0.51, 0.22, 0.26, 0.16]))
# r377_KE = np.array([341, 36.2, 27.9, 9.5, 9.9]) * np.array([1, 0.19, 0.02, 0.03, 0.01])
# r377_PE = np.array([341, 36.2, 27.9, 9.5, 9.9]) * (1 - np.array([1, 0.19, 0.02, 0.03, 0.01]))
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
# ax1.plot(modes, r3_KE, color='r')
# ax1.plot(modes[1:], r3_PE[1:], color='b')
# ax1.set_ylim([10**-1, 10**4])
# ax1.set_yscale('log')
# ax1.set_xlabel('Vertical Mode')
# ax1.set_ylabel('Energy')
# ax1.set_title('Ri = 3')
# ax1.grid()
# ax2.plot(modes, r13_KE, color='r')
# ax2.plot(modes[1:], r13_PE[1:], color='b')
# ax2.set_yscale('log')
# ax2.set_xlabel('Vertical Mode')
# ax2.set_title('Ri = 13')
# ax2.grid()
# ax3.plot(modes, r377_KE, color='r', label='KE')
# ax3.plot(modes[1:], r377_PE[1:], color='b', label='PE')
# ax3.set_yscale('log')
# ax3.set_xlabel('Vertical Mode')
# ax3.set_title('Ri = 377')
# handles, labels = ax3.get_legend_handles_labels()
# ax3.legend(handles, labels, fontsize=10)
# plot_pro(ax3)
# ---------------------------------------------------------------------------------------------------------------------
# SAMPLE EDDY (MODE INTERACTIONS)
samp_ed = 0
if samp_ed > 0:
    # Schematic of Mode Shapes and Interactions
    y_grid = np.arange(-40000, 40000, 1000)
    # z_grid = np.arange(0, -4000, -10)
    load_F = si.loadmat('/Users/jake/Documents/Cuddy_tailored/normalized_anomaly.mat')
    Fz = load_F['out']['F_z'][0][0]
    z_grid_Fz = load_F['out']['depth'][0][0][:, 0]

    Fz_2 = savgol_filter(Fz[:, 0], 45, 5) - .16
    z_3 = np.concatenate((2. * z_grid_Fz, np.arange(2050, 4000, 50)))
    Fz_3 = np.concatenate((Fz_2, 0 * np.ones(len(np.arange(2050, 4000, 50)))))
    Fz_4 = savgol_filter(Fz_3, 55, 5)

    z_grid = -1 * SB['depth']
    rho_1 = savgol_filter(SB['Sigma0'][:, 0], 7, 5) + 1000
    rho_1 = savgol_filter(rho_1, 15, 5)
    Fz_5 = np.interp(-1. * z_grid, z_3, Fz_4)

    [Y, Z] = np.meshgrid(y_grid, z_grid)
    y_i = np.where((y_grid >= 0) & (y_grid <= 35000))[0]
    y_i_2 = np.where((y_grid >= -35000) & (y_grid <= 0))[0]
    z_i = np.where(z_grid > -5000)[0]

    # physical parameters
    g = 9.80665
    om = 7.2921 * 10**-5 # rot rate of earth
    rho0 = 1025.0
    f1 = 2.0 * om * np.sin(np.deg2rad((47.5)))

    A = 0.17
    L = 12000.0
    H = 1600.0
    rho_0 = 1026.0

    # -- surface eddy
    A2 = 0.000175
    L2 = 10000.0
    H_sc = 0.4
    H_sc_2 = 0.25
    rho_pr_s = -1.0 * (A2) * rho_0 * np.e**(1.0 * Z[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1] /(H_sc_2 * H)) * \
               np.e**(-1.0 * (Y[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1]**2) / (3 * L2**2))
    rho_surf = np.nan*np.ones(np.shape(Y))
    rho_b = np.tile(rho_1[:, None], (1, len(y_grid)))
    rho_surf[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1] = rho_b[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1] + rho_pr_s
    rho_surf[z_i[0]:z_i[-1]+1, y_i_2[0]:y_i_2[-1]] = np.fliplr(rho_b[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]] + rho_pr_s[:, 1:])
    v_s = (450.0 * A2 * g / (rho_0 * f1)) * (2 * Y[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1] / (3 * L2**2)) * \
        np.e**(-(Y[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1]**2)/(3 * L2**2)) * \
        (1.0 * H_sc * H * np.e**(Z[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1] / (H_sc * H)))
    v_surf = np.nan*np.ones(np.shape(Y))
    v_surf[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1] = -1.0 * v_s
    v_surf[z_i[0]:z_i[-1]+1, y_i_2[0]:y_i_2[-1]] = np.fliplr(v_s[:, 1:])

    # -- subsurface eddy
    A = 0.00007
    rho_pr = A * rho_0 * np.transpose(np.tile(Fz_5, (len(y_i), 1))) * \
            np.e**((-Y[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1]**2)/(2 * L**2))
    rho = np.nan*np.ones(np.shape(Y))
    rho_b = np.tile(rho_1[:, None], (1, len(y_grid)))
    rho[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1] = rho_b[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1] + rho_pr
    rho[z_i[0]:z_i[-1]+1, y_i_2[0]:y_i_2[-1]] = np.fliplr(rho_b[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]] + rho_pr[:, 1:])

    int_Fz = np.concatenate((np.array([-4]), cumtrapz(Fz_5, z_grid)))
    v_r = (A * g * Y[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1])/(f1 * L**2) * \
        np.e**((-Y[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1]**2) / (2*L**2)) * np.transpose(np.tile(int_Fz, (len(y_i), 1)));
    v = np.nan*np.ones(np.shape(Y))
    v[z_i[0]:z_i[-1]+1, y_i[0]:y_i[-1]+1] = v_r
    v[z_i[0]:z_i[-1]+1, y_i_2[0]:y_i_2[-1]] = np.fliplr(-1. * v_r[:, 1:])
    # v[np.abs(v) < 0.01] = 0.0

    # -- alternate computation of displacement
    ddz_avg_sigma = savgol_filter(np.gradient(rho_1 - 1000.0, z_grid), 35, 5)

    # surface
    eddy_rho_surf = rho_surf[:, y_grid == 0][:, 0]
    eta_surf = np.nan * np.ones(len(eddy_rho_surf))
    for i in range(len(eddy_rho_surf)):
        idx, rho_idx = find_nearest(rho_1, eddy_rho_surf[i])
        if idx <= 1:
            z_rho_1 = -1.0 * z_grid[0:idx+3]
            eta_surf[i] = np.interp(eddy_rho_surf[i], rho_1[0:idx + 3], z_rho_1) - (-1.0 * z_grid[i])
        else:
            z_rho_1 = -1.0 * z_grid[idx-2:idx+3]
            eta_surf[i] = np.interp(eddy_rho_surf[i], rho_1[idx-2:idx+3], z_rho_1) - (-1.0 * z_grid[i])

    # subsurface
    eddy_rho = rho[:, y_grid == 0][:, 0]
    eta = np.nan * np.ones(len(eddy_rho))
    for i in range(len(eddy_rho)):
        idx, rho_idx = find_nearest(rho_1, eddy_rho[i])
        if idx <= 2:
            z_rho_1 = -1.0 * z_grid[0:idx+3]
            eta[i] = np.interp(eddy_rho[i], rho_1[0:idx + 3], z_rho_1) - (-1.0 * z_grid[i])
        else:
            z_rho_1 = -1.0 * z_grid[idx-2:idx+3]
            eta[i] = np.interp(eddy_rho[i], rho_1[idx-2:idx+3], z_rho_1) - (-1.0 * z_grid[i])
        # z_rho_1 = -1.0 * z_grid[idx-2:idx+3]
        # eta[i] = np.interp(eddy_rho[i], rho_1[idx-2:idx+3], z_rho_1) - (-1.0 * z_grid[i])

    # -- plotting
    rho_levels = np.concatenate((np.arange(1024, 1027.8, 0.2), np.array([1027.76]) ,np.arange(1027.81, 1028, 0.01)))
    u_levels = np.arange(-.4, .4, 0.05)
    u_levels_2 = [-.4, -.3, -.2, -.1, .1, .2, .3, .4]

    cmap = matplotlib.cm.get_cmap('RdBu')
    # surface eddy
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.contour(Y/1000, Z, rho_surf, levels=rho_levels, colors='#A9A9A9', linewidth=0.5)
    ax1.contourf(Y/1000, Z, v_surf, levels=u_levels, cmap=cmap)
    ax1.contour(Y/1000, Z, v_surf, levels=u_levels_2, colors='k', linewidth=0.75)
    ax1.set_xlabel('Km')
    ax1.set_ylabel('Z [m]')
    ax1.set_ylim([-3500, 0])
    ax1.set_title('Surface Eddy')
    ax1.grid()
    ax2.plot(savgol_filter(eta_surf, 25, 7), z_grid, linewidth=2)
    ax2.plot(G_B_0[:, 1] / 50, z_grid, color='r', linestyle='--')
    ax2.set_xlim([-150, 300])
    ax2.set_xlabel('[m]')
    ax2.set_title('Vertical Isopycnal Displacement')
    ax2.grid()
    ax3.plot(v_surf[:, y_grid == L / 2], z_grid, linewidth=2)
    ax3.plot(-1. * .2 * Gz_B_0[:, 0], -1.0 * sta_bats_depth, color='r', linewidth=0.5)
    ax3.plot(Gz_B_0[:, 1] * (np.nanmin(v) * .2), -1.0 * sta_bats_depth, color='r', linewidth=0.5)
    ax3.plot((-0.02 * Gz_B_0[:, 0]) + (Gz_B_0[:, 1] * (np.nanmin(v) * .2)), -1.0 * sta_bats_depth,
            color='r', linestyle='--')
    ax3.set_title('Velocity Profile and Mode Shapes')
    ax3.set_xlabel('[m/s]')
    plot_pro(ax3)

    # subsurface eddy
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.contour(Y/1000, Z, rho, levels=rho_levels, colors='#A9A9A9', linewidth=0.5)
    ax1.contourf(Y/1000, Z, v, levels=u_levels, cmap=cmap)
    ax1.contour(Y/1000, Z, v, levels=u_levels_2, colors='k', linewidth=0.75)
    ax1.set_xlabel('Km')
    ax1.set_ylabel('Z [m]')
    ax1.set_ylim([-3500, 0])
    ax1.set_title('Subsurface Eddy')
    ax1.grid()
    ax2.plot(savgol_filter(eta, 25, 7), z_grid)
    ax2.plot(G_B_0[:, 1] / 50, z_grid, color='r', linestyle='--')
    # ax2.plot((rho[:, y_grid == 0][:, 0] - rho_1) / ddz_avg_sigma, z_grid)
    # ax2.plot(rho[:, y_grid == 0][:, 0], z_grid)
    # ax2.plot(rho_1, z_grid)
    ax2.set_xlim([-150, 300])
    ax2.set_xlabel('[m]')
    ax2.set_title('Vertical Isopycnal Displacement')
    ax2.grid()
    ax3.plot(v[:, y_grid == L / 2], z_grid, linewidth=2)
    ax3.plot(-1. * .2 * Gz_B_0[:, 0], -1.0 * sta_bats_depth, color='r', linewidth=0.5)
    ax3.plot(Gz_B_0[:, 1] * (np.nanmin(v) * .2), -1.0 * sta_bats_depth, color='r', linewidth=0.5)
    ax3.plot((-0.02 * Gz_B_0[:, 0]) + (Gz_B_0[:, 1] * (np.nanmin(v) * .2)), -1.0 * sta_bats_depth,
            color='r', linestyle='--')
    ax3.set_title('Velocity Profile and Mode Shapes')
    ax3.set_xlabel('[m/s]')
    plot_pro(ax3)
# ---------------------------------------------------------------------------------------------------------------------
# MODE SHAPE PLOT
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(G_B_0[:, 1], z_grid, linewidth=2)
# ax1.plot(G_B_0[:, 2], z_grid, linewidth=2)
# ax1.plot(G_B_0[:, 3], z_grid, linewidth=2)
# ax2.plot(Gz_B_0[:, 1], z_grid, linewidth=2)
# ax2.plot(Gz_B_0[:, 2], z_grid, linewidth=2)
# ax2.plot(Gz_B_0[:, 3], z_grid, linewidth=2)
# ax1.set_xlim([-2000, 2000])
# ax2.set_xlim([-3, 3])
# ax1.set_ylabel('Depth [m]')
# ax1.set_title('Displacement Modes')
# ax2.set_title('Velocity Modes')
# ax1.grid()
# plot_pro(ax2)


# ---------------------------------------------------------------------------------------------------------------------

# OLD
# # rho_1 = np.concatenate((1026 * np.ones(len(z_grid[0:20])), np.arange(1026, 1027, (1027 - 1026)/len(z_grid[20:100])),
# #                         np.arange(1027, 1028, (1028 - 1027)/len(z_grid[100:]))))
# rho_1 = SB['Sigma0'][0:200, 0] + 1000
# rho_pr = np.nan * np.ones(np.shape(Y))
# rho_pr[z_i[0]:z_i[-1], y_i[0]:y_i[-1]] = A * np.sin(np.pi * Z[z_i[0]:z_i[-1], y_i[0]:y_i[-1]] / H) * \
#                                          np.cos(np.pi * Y[z_i[0]:z_i[-1], y_i[0]:y_i[-1]] / (2 * L))**2
# rho_pr[z_i[0]:z_i[-1], y_i_2[0]:y_i_2[-1]] = np.fliplr(A * np.sin(np.pi * Z[z_i[0]:z_i[-1], y_i[0]:y_i[-1]] / H) * \
#                                          np.cos(np.pi * Y[z_i[0]:z_i[-1], y_i[0]:y_i[-1]] / (2 * L))**2)
# rho = np.tile(rho_1[:, None], (1, len(y_grid)))
# rho[z_i[0]:z_i[-1], y_i_2[0]:y_i[-1]] = rho[z_i[0]:z_i[-1], y_i_2[0]:y_i[-1]] + rho_pr[z_i[0]:z_i[-1], y_i_2[0]:y_i[-1]]
#
# # using thermal wind
# v = 0.0 * np.ones(np.shape(Y))
# v[z_i[0]:z_i[-1], y_i[0]:y_i[-1]] = (g * A * H / (2 * f1 * rho_0 * L)) * \
#                                     -1 * np.sin(np.pi * Y[z_i[0]:z_i[-1], y_i[0]:y_i[-1]] / L) * \
#                                     (np.cos(np.pi * Z[z_i[0]:z_i[-1], y_i[0]:y_i[-1]] / (2 * H))**2)
# v[z_i[0]:z_i[-1], y_i_2[0]:y_i_2[-1]] = -1 * (g * A * H / (2 * f1 * rho_0 * L)) * \
#                                         -1 * np.sin(np.pi * Y[z_i[0]:z_i[-1], y_i[0]:y_i[-1]] / L) * \
#                                         (np.cos(np.pi * Z[z_i[0]:z_i[-1], y_i[0]:y_i[-1]] / (2 * H))**2)
import numpy as np
import pickle
import glob
import datetime
from netCDF4 import Dataset
import gsw
import time as TT
from scipy.integrate import cumtrapz
from mode_decompositions import eta_fit, vertical_modes, PE_Tide_GM, vertical_modes_f
# -- plotting
import matplotlib
import matplotlib.pyplot as plt
from toolkit import plot_pro, nanseg_interp
from zrfun import get_basic_info, get_z


file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/LiveOcean/simulated_dg_velocities/ve_ew_v*y40_*.pkl')
tagg = 'yall_v08_slp3'
save_metr = 0  # ratio
save_e = 1  # save energy spectra
save_rms = 0  # save v error plot

direct_anom = []
# igw_var = np.nan * np.ones(len(file_list))
for i in range(len(file_list)):
    pkl_file = open(file_list[i], 'rb')
    MOD = pickle.load(pkl_file)
    pkl_file.close()

    glider_v = MOD['dg_v'][:]
    model_v = MOD['model_u_at_mwv']
    model_v_avg = MOD['model_u_at_mw_avg']
    z_grid = MOD['dg_z'][:]
    slope_error = MOD['shear_error'][:]
    igw = MOD['igw_var']
    ke_mod_0 = MOD['KE_mod'][:]
    pe_mod_0 = MOD['PE_model'][:]
    ke_dg_0 = MOD['KE_dg'][:]
    pe_dg_0 = MOD['PE_dg_avg'][:]
    pe_dg_ind_0 = MOD['PE_dg'][:]
    w_tag_0 = MOD['dg_w'][:]
    slope_tag_0 = MOD['glide_slope'][:]

    model_mean_per_mw = np.nan * np.ones(np.shape(glider_v))
    for j in range(len(model_v)):
        this_mod = model_v[j]
        g_rep = np.repeat(np.tile(glider_v[:, j][:, None], np.shape(this_mod)[1])[:, :, None],
                          np.shape(this_mod)[2], axis=2)
        direct_anom = g_rep - this_mod
        mod_space = np.nanmean(this_mod, axis=2)  # average across time
        mod_time = np.nanmean(this_mod, axis=1)  # average across space
        spatial_anom = np.nanmean(direct_anom, axis=2)  # average across time
        time_anom = np.nanmean(direct_anom, axis=1)  # average across space

        model_mean_per_mw[:, j] = np.nanmean(np.nanmean(model_v[j], axis=2), axis=1)

    if i < 1:
        v = glider_v.copy()
        mod_v = model_mean_per_mw.copy()
        mod_v_avg = model_v_avg.copy()
        mod_space_out = mod_space
        mod_time_out = mod_time
        anoms = glider_v - model_mean_per_mw
        anoms_space = spatial_anom
        anoms_time = time_anom
        slope_er = slope_error.copy()
        igw_var = igw.copy()
        ke_mod = ke_mod_0.copy()
        pe_mod = pe_mod_0.copy()
        ke_dg = ke_dg_0.copy()
        pe_dg = pe_dg_0.copy()
        pe_dg_ind = pe_dg_ind_0.copy()
        w_tag = 100 * w_tag_0.copy()
        slope_tag = slope_tag_0.copy()
    else:
        v = np.concatenate((v, glider_v.copy()), axis=1)
        mod_v = np.concatenate((mod_v, model_mean_per_mw), axis=1)
        mod_v_avg = np.concatenate((mod_v_avg, model_v_avg), axis=1)
        mod_space_out = np.concatenate((mod_space_out, mod_space), axis=1)
        mod_time_out = np.concatenate((mod_time_out, mod_time), axis=1)
        anoms = np.concatenate((anoms, glider_v - model_mean_per_mw), axis=1)
        anoms_space = np.concatenate((anoms_space, spatial_anom), axis=1)
        anoms_time = np.concatenate((anoms_time, time_anom), axis=1)
        slope_er = np.concatenate((slope_er, slope_error), axis=1)
        igw_var = np.concatenate((igw_var, igw), axis=1)
        ke_mod = np.concatenate((ke_mod, ke_mod_0), axis=1)
        pe_mod = np.concatenate((pe_mod, pe_mod_0), axis=1)
        ke_dg = np.concatenate((ke_dg, ke_dg_0), axis=1)
        pe_dg = np.concatenate((pe_dg, pe_dg_0), axis=1)
        pe_dg_ind = np.concatenate((pe_dg_ind, pe_dg_ind_0), axis=1)
        w_tag = np.concatenate((w_tag, 100 * w_tag_0), axis=0)
        slope_tag = np.concatenate((slope_tag, slope_tag_0), axis=0)

# vertical shear error as a function depth and igwsignal
matplotlib.rcParams['figure.figsize'] = (6.5, 8)
f, ax = plt.subplots()
low_er_mean = np.nan * np.ones(len(z_grid))
for i in range(len(z_grid)):
    low = np.where(igw_var[i, :] < 0.75)[0]
    ax.scatter(slope_er[i, :], z_grid[i] * np.ones(len(slope_er[i, :])), s=2, color='b')
    ax.scatter(slope_er[i, low], z_grid[i] * np.ones(len(slope_er[i, low])), s=4, color='r')
    ax.scatter(np.nanmean(slope_er[i, low]), z_grid[i], s=20, color='r')
    low_er_mean[i] = np.nanmean(slope_er[i, low])
ax.plot(low_er_mean, z_grid, linewidth=2.5, color='r', label='Low Noise Error Mean')
ax.scatter(np.nanmean(slope_er[i, low]), z_grid[i], s=15, color='r', label=r'var$_{igw}$/var$_{gstr}$ < 0.2')
ax.plot(np.nanmedian(slope_er, axis=1), z_grid, color='b', linewidth=2.5, label='Error Median')
ax.plot([20, 20], [0, -3000], color='k', linewidth=2.5, linestyle='--')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=12)
ax.set_xscale('log')
ax.set_xlabel('Percent Error')
ax.set_ylabel('z [m]')
ax.set_title('Percent Error between Model Shear and Glider Shear (' + str(MOD['glide_slope']) + ':1, w=' + str(MOD['dg_w']) + ' m/s)')
ax.set_xlim([1, 10**4])
ax.set_ylim([-3000, 0])
plot_pro(ax)
if save_metr > 0:
    f.savefig('/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_05_17/model_dg_per_shear_err_' + str(tagg) + '.png', dpi=200)


# all velocity anomalies colored by max abs vel attained
# cmap = plt.cm.get_cmap('Spectral_r')
# f, ax = plt.subplots()
# this_one = mod_time_out
# this_anom = anoms_time
# max_mod_v = np.nan * np.ones(np.shape(this_one)[1])
# avg_anom1 = np.nan * np.ones(np.shape(this_one)[1])
# avg_anom2 = np.nan * np.ones(np.shape(this_one)[1])
# avg_anom3 = np.nan * np.ones(np.shape(this_one)[1])
# for i in range(np.shape(this_one)[1]):
#     max_mod_v[i] = np.nanmax(this_one[:, i]**2)
#     avg_anom1[i] = np.nanmean(this_anom[0:50, i]**2)
#     avg_anom2[i] = np.nanmean(this_anom[5:100, i]**2)
#     avg_anom3[i] = np.nanmean(this_anom[100:, i]**2)
#     if np.nanmax(np.abs(this_one[:, i])) > 0.25:
#         ax.plot(this_anom[:, i], z_grid, color=cmap(1))
#     elif (np.nanmax(np.abs(this_one[:, i])) > 0.2) & (np.nanmax(np.abs(this_one[:, i])) < 0.25):
#         ax.plot(this_anom[:, i], z_grid, color=cmap(.8))
#     elif (np.nanmax(np.abs(this_one[:, i])) > 0.15) & (np.nanmax(np.abs(this_one[:, i])) < 0.2):
#         ax.plot(this_anom[:, i], z_grid, color=cmap(.6))
#     elif (np.nanmax(np.abs(this_one[:, i])) > 0.1) & (np.nanmax(np.abs(this_one[:, i])) < 0.15):
#         ax.plot(this_anom[:, i], z_grid, color=cmap(.4))
#     elif (np.nanmax(np.abs(this_one[:, i])) > 0.05) & (np.nanmax(np.abs(this_one[:, i])) < 0.1):
#         ax.plot(this_anom[:, i], z_grid, color=cmap(.2))
#     else:
#         ax.plot(this_anom[:, i], z_grid, color=cmap(.01))
# ax.plot(np.nanmean(this_anom, axis=1), z_grid, linewidth=2.2, color='k')
# plot_pro(ax)

# # RMS at different depths
# matplotlib.rcParams['figure.figsize'] = (9.5, 8)
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# anomy = v - mod_v_avg
# for i in range(np.shape(anomy)[1]):
#     ax1.plot(anomy[:, i], z_grid, color='k', linewidth=0.6)
# ax1.plot(np.nanmean(anomy, axis=1), z_grid, color='r')
# ax1.set_xlim([-.12, .12])
# ax1.set_xlabel('m/s')
# ax1.set_ylabel('Depth [m]')
# ax1.set_title(r'M/W Vel. Error ($u_{g}$ - $\overline{u_{model}}$) ('
#              + str(MOD['glide_slope']) + ':1, w=' + str(MOD['dg_w']) + ' m/s)')
# mm = np.nanmean(anomy**2, axis=1)
# mm_std = np.nan * np.ones(len(z_grid))
# for i in range(len(z_grid)):
#     mm_std[i] = np.nanstd(anomy[i, :]**2)
# ax1.text(0.04, -2600, str(np.shape(anomy)[1]) + ' profiles')
# ax2.plot(mm, z_grid, linewidth=2.2, color='k')
# ax2.plot(mm_std, z_grid, linewidth=1.5, color='k', linestyle='--')
# # ax.text(0.005, -2400, 'Mean Error = ' + str(np.round(np.nanmean(anoms), 3)) + ' m/s')
# ax2.set_xlim([0, .005])
# ax2.set_ylim([-3000, 0])
# ax2.set_xlabel(r'm$^2$/s$^2$')
# ax2.set_title(r'M/W Vel. RMS Error ($u_{g}$ - $\overline{u_{model}}$)$^2$')
# ax1.grid()
# plot_pro(ax2)
# if savee > 0:
#     f.savefig('/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_05_17/model_dg_vel_error_' + str(tagg) + '.png', dpi=200)

# RMS at different depths
anomy = v - mod_v_avg  # velocity anomaly
# estimate rms error by w
w_s = np.unique(w_tag)
slope_s = np.unique(slope_tag)
w_cols = '#191970', 'g', '#FF8C00', '#B22222'
mm = np.nan * np.ones((len(slope_s), len(z_grid), len(w_s)))
avg_anom = np.nan * np.ones((len(slope_s), len(z_grid), len(w_s)))
for ss in range(len(np.unique(slope_tag))):
    for i in range(len(np.unique(w_tag))):
        inn = np.where((w_tag == w_s[i]) & (slope_tag == slope_s[ss]))[0]
        mm[ss, :, i] = np.nanmean(anomy[:, inn]**2, axis=1)  # rms error
        avg_anom[ss, :, i] = np.nanmean(anomy[:, inn], axis=1)
# std about error
min_a = np.nan * np.ones((len(slope_s), len(z_grid), 4))
max_a = np.nan * np.ones((len(slope_s), len(z_grid), 4))
for ss in range(len(np.unique(slope_tag))):
    for i in range(np.shape(anomy)[0]):
        for j in range(len(w_s)):
            inn = np.where((w_tag == w_s[j]) & (slope_tag == slope_s[ss]))[0]
            min_a[ss, i, j] = np.nanmean(anomy[i, inn]) - np.nanstd(anomy[i, inn])
            max_a[ss, i, j] = np.nanmean(anomy[i, inn]) + np.nanstd(anomy[i, inn])

matplotlib.rcParams['figure.figsize'] = (12, 6.5)
f, ax = plt.subplots(1, 4, sharey=True)
# w_cols_2 = '#48D1CC', '#32CD32', '#FFA500', '#CD5C5C'
w_cols_2 = '#40E0D0', '#2E8B57', '#FFA500', '#CD5C5C'
for i in range(len(w_s)):
    ax[i].fill_betweenx(z_grid, min_a[0, :, i], x2=max_a[0, :, i], color=w_cols_2[i], zorder=i, alpha=0.95)
    ax[i].plot(avg_anom[0, :, i], z_grid, color=w_cols[i], linewidth=3, zorder=4, label='dg w = ' + str(w_s[i]) + ' cm/s')
    ax[i].set_xlim([-.2, .2])
    ax[i].set_xlabel('m/s')
    ax[i].set_title(r'($u_{g}$ - $\overline{u_{model}}$) (w=$\mathbf{' + str(w_s[i]) + '}$ cm/s) ('
                    + str(np.int(slope_s[0])) + ':1)', fontsize=10)
ax[0].set_ylabel('z [m]')
ax[0].set_ylim([-3200, 0])
ax[0].text(0.025, -4200, str(np.shape(anomy[:, slope_tag > 2])[1]) + ' profiles')
ax[0].grid()
ax[1].grid()
ax[2].grid()
plot_pro(ax[3])
if save_rms > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/LO_model_dg_vel_error_bats.png', dpi=200)

matplotlib.rcParams['figure.figsize'] = (6, 6)
f, ax2 = plt.subplots()
for i in range(np.shape(mm)[2]):
    ax2.plot(mm[0, :, i], z_grid, linewidth=2.2, color=w_cols[i],
             label='w=' + str(w_s[i]) + ' cm/s) (gs=' + str(np.int(slope_s[0])) + ':1)')
    # ax2.plot(mm[1, :, i], z_grid, linewidth=2.2, color=w_cols[i], linestyle='--',
    #          label='w=' + str(w_s[i]) + ' cm/s) (gs=' + str(np.int(slope_s[1])) + ':1)')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=10)
ax2.set_xlim([0, .03])
ax2.set_ylim([-3200, 0])
ax2.set_xlabel(r'm$^2$/s$^2$')
ax2.set_title(r'Glider/Model Velocity rms error ($u_{g}$ - $\overline{u_{model}}$)$^2$')
ax2.set_ylabel('z [m]')
plot_pro(ax2)
if save_rms > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/LO_model_dg_vel_rms_e.png', dpi=200)

# ---------------------------------------------------------------------------------------------------------------------
# --- PLOT ENERGY SPECTRA
ff = np.pi * np.sin(np.deg2rad(44)) / (12 * 1800)  # Coriolis parameter [s^-1]
# --- Background density
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Model/LiveOcean/background_density.pkl', 'rb')
MODb = pickle.load(pkl_file)
pkl_file.close()
bck_sa = np.flipud(MODb['sa_back'][:][:, 40:120])
bck_ct = np.flipud(MODb['ct_back'][:][:, 40:120])
z_bm = [0, len(z_grid)-14]
p_grid = gsw.p_from_z(z_grid, 44)

# or if no big feature is present, avg current profiles for background
N2_bck_out = gsw.Nsquared(np.nanmean(bck_sa[0:z_bm[-1]+1, :], axis=1), np.nanmean(bck_ct[0:z_bm[-1]+1, :], axis=1),
                          p_grid[0:z_bm[-1]+1], lat=44)[0]
N2_bck_out[N2_bck_out < 0] = 1*10**-7

omega = 0
mmax = 25
mm = 25
G, Gz, c, epsilon = vertical_modes(N2_bck_out, -1.0 * z_grid[0:146], omega, mmax)  # N2

sc_x = 1000 * ff / c[1:mm]
l_lim = 3 * 10 ** -2
sc_x = np.arange(1, mm)
l_lim = 0.7
dk = ff / c[1]

avg_PE = np.nanmean(pe_dg, axis=1)
avg_PE_ind = np.nanmean(pe_dg_ind, axis=1)
avg_KE = np.nanmean(ke_dg, axis=1)
avg_PE_model = np.nanmean(pe_mod, axis=1)
avg_KE_model = np.nanmean(ke_mod, axis=1)

matplotlib.rcParams['figure.figsize'] = (10, 6)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# DG
ax1.plot(sc_x, avg_PE[1:mm] / dk, 'r', label='PE$_{DG}$', linewidth=2)
ax1.scatter(sc_x, avg_PE[1:mm] / dk, color='r', s=20)
ax1.plot(sc_x, avg_PE_ind[1:mm] / dk, 'c', label='PE$_{DG_{ind}}$', linewidth=2)
ax1.scatter(sc_x, avg_PE_ind[1:mm] / dk, color='c', s=20)
ax2.plot(sc_x, avg_KE[1:mm] / dk, 'r', label='KE$_{DG}$', linewidth=3)
ax2.scatter(sc_x, avg_KE[1:mm] / dk, color='r', s=20)  # DG KE
ax2.plot([l_lim, sc_x[0]], avg_KE[0:2] / dk, 'r', linewidth=3)  # DG KE_0
ax2.scatter(l_lim, avg_KE[0] / dk, color='r', s=25, facecolors='none')  # DG KE_0
# Model
ax1.plot(sc_x, avg_PE_model[1:mm] / dk, color='k', label='PE$_{Model}$', linewidth=2)
ax1.scatter(sc_x, avg_PE_model[1:mm] / dk, color='k', s=20)
ax2.plot(sc_x, avg_KE_model[1:mm] / dk, color='k', label='KE$_{Model}$', linewidth=2)
ax2.scatter(sc_x, avg_KE_model[1:mm] / dk, color='k', s=20)
ax2.plot([l_lim, sc_x[0]], avg_KE_model[0:2] / dk, color='k', linewidth=2)
ax2.scatter(l_lim, avg_KE_model[0] / dk, color='k', s=25, facecolors='none')

modeno = '1', '2', '3', '4', '5', '6', '7', '8'
for j in range(len(modeno)):
    ax2.text(sc_x[j], (avg_KE[j + 1] + (avg_KE[j + 1] / 2)) / dk, modeno[j], color='k', fontsize=10)

limm = 5
ax1.set_xlim([l_lim, 0.5 * 10 ** 2])
ax2.set_xlim([l_lim, 0.5 * 10 ** 2])
ax2.set_ylim([10 ** (-4), 1 * 10 ** 2])
ax1.set_yscale('log')
ax1.set_xscale('log')
ax2.set_xscale('log')

# ax1.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
ax1.set_ylabel('Spectral Density', fontsize=12)  # ' (and Hor. Wavenumber)')
# ax2.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
ax1.set_xlabel('Mode Number', fontsize=12)
ax2.set_xlabel('Mode Number', fontsize=12)
ax1.set_title('PE Spectrum (' + str(np.int(MOD['glide_slope'][0])) + ':1)', fontsize=12)
ax2.set_title('KE Spectrum (' + str(np.int(MOD['glide_slope'][0])) + ':1)', fontsize=12)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=12)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=12)
ax1.grid()
plot_pro(ax2)
if save_e > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/LO_model_dg_vel_energy_eddy.png', dpi=200)

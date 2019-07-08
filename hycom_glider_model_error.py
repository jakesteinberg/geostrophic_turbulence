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


file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/HYCOM/simulated_dg_velocities_bats_36n/ve_b*_v*_slp*_y*.pkl')
tagg = 'yall_v20_slp3'
savee = 0
save_rms = 1  # vel error plot
save_e = 1  # energy spectra plot
plot_se = 0  # plot shear error scatter plot

direct_anom = []
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
    c = MOD['c'][:]
    if 'PE_mod_ALL' in MOD.keys():
        ke_mod_tot = MOD['KE_mod_ALL'][:]
        pe_mod_tot = MOD['PE_mod_ALL'][:]
        avg_N2 = MOD['avg_N2'][:]
        z_grid_n2 = MOD['z_grid_n2'][:]

    model_mean_per_mw = np.nan * np.ones(np.shape(glider_v))
    for j in range(len(model_v)):
        this_mod = model_v[j]
        g_rep = np.repeat(np.tile(glider_v[:, j][:, None], np.shape(this_mod)[2])[None, :, :],
                          np.shape(this_mod)[0], axis=0)
        direct_anom = g_rep - this_mod
        mod_space = np.nanmean(this_mod, axis=0)  # average across time
        mod_time = np.transpose(np.nanmean(this_mod, axis=2))  # average across space
        spatial_anom = np.nanmean(direct_anom, axis=0)  # average across time
        time_anom = np.transpose(np.nanmean(direct_anom, axis=2))  # average across space

        model_mean_per_mw[:, j] = np.nanmean(np.nanmean(model_v[j], axis=2), axis=0)

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
if plot_se > 0:
    matplotlib.rcParams['figure.figsize'] = (6.5, 8)
    f, ax = plt.subplots()
    low_er_mean = np.nan * np.ones(len(z_grid))
    for i in range(len(z_grid)):
        frac_lim = 0.2
        low = np.where(igw_var[i, :] < frac_lim)[0]
        ax.scatter(slope_er[i, :], z_grid[i] * np.ones(len(slope_er[i, :])), s=2, color='b')
        ax.scatter(slope_er[i, low], z_grid[i] * np.ones(len(slope_er[i, low])), s=4, color='r')
        ax.scatter(np.nanmean(slope_er[i, low]), z_grid[i], s=30, color='r')
        low_er_mean[i] = np.nanmean(slope_er[i, low])
    ax.plot(low_er_mean, z_grid, linewidth=2.5, color='r', label='Low Noise Error Mean')
    ax.scatter(np.nanmean(slope_er[i, low]), z_grid[i], s=15, color='r',
            label=r'var$_{igw}$/var$_{gstr}$ < ' + str(frac_lim))
    ax.plot(np.nanmedian(slope_er, axis=1), z_grid, color='b', linewidth=2.5, label='Error Median')
    ax.plot([20, 20], [0, -3000], color='k', linewidth=2.5, linestyle='--')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12)
    ax.set_xscale('log')
    ax.set_xlabel('Percent Error')
    ax.set_ylabel('z [m]')
    ax.set_title('Percent Error between Model Shear and Glider Shear (' + str(MOD['glide_slope']) +
                 ':1, w=' + str(MOD['dg_w']) + ' m/s)')
    ax.set_xlim([1, 10**4])
    ax.set_ylim([-3000, 0])
    plot_pro(ax)
    if savee > 0:
        f.savefig('/Users/jake/Documents/glider_flight_sim_paper/model_dg_per_shear_err_' + str(tagg) + '.png', dpi=200)

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
for i in range(4):
    ax[i].fill_betweenx(z_grid, min_a[1, :, i], x2=max_a[1, :, i], color=w_cols_2[i], zorder=i, alpha=0.95)
    ax[i].plot(avg_anom[1, :, i], z_grid, color=w_cols[i], linewidth=3, zorder=4, label='dg w = ' + str(w_s[i]) + ' cm/s')
    ax[i].set_xlim([-.2, .2])
    ax[i].set_xlabel('m/s')
    ax[i].set_title(r'($u_{g}$ - $\overline{u_{model}}$) (w=$\mathbf{' + str(w_s[i]) + '}$ cm/s) ('
                    + str(np.int(slope_s[1])) + ':1)', fontsize=10)
ax[0].set_ylabel('z [m]')
ax[0].text(0.025, -4200, str(np.shape(anomy[:, slope_tag > 2])[1]) + ' profiles')
ax[0].grid()
ax[1].grid()
ax[2].grid()
plot_pro(ax[3])
if save_rms > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/model_dg_vel_error_bats.png', dpi=200)

matplotlib.rcParams['figure.figsize'] = (6, 6)
f, ax2 = plt.subplots()
for i in range(np.shape(mm)[2]):
    ax2.plot(mm[0, :, i], z_grid, linewidth=2.2, color=w_cols[i],
             label='w=' + str(w_s[i]) + ' cm/s) (gs=' + str(np.int(slope_s[0])) + ':1)')
    ax2.plot(mm[1, :, i], z_grid, linewidth=2.2, color=w_cols[i], linestyle='--',
             label='w=' + str(w_s[i]) + ' cm/s) (gs=' + str(np.int(slope_s[1])) + ':1)')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=10)
ax2.set_xlim([0, .03])
ax2.set_ylim([-4750, 0])
ax2.set_xlabel(r'm$^2$/s$^2$')
ax2.set_title(r'Glider/Model Velocity rms error ($u_{g}$ - $\overline{u_{model}}$)$^2$')
ax2.set_ylabel('z [m]')
plot_pro(ax2)
if save_rms > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/model_dg_vel_rms_e_bats.png', dpi=200)

# ---------------------------------------------------------------------------------------------------------------------
# --- PLOT ENERGY SPECTRA
ff = np.pi * np.sin(np.deg2rad(44)) / (12 * 1800)  # Coriolis parameter [s^-1]

omega = 0
mmax = 30
mm = 30

sc_x = 1000 * ff / c[1:mm]
l_lim = 3 * 10 ** -2
sc_x = np.arange(1, mm)
l_lim = 0.7
dk = ff / c[1]

good = np.where((ke_mod[1, :] < 1*10**0) & (slope_tag > 2))[0]
avg_PE_model = np.nanmean(pe_mod[:, good], axis=1)
avg_KE_model = np.nanmean(ke_mod[:, good], axis=1)

matplotlib.rcParams['figure.figsize'] = (10, 6)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# DG
for i in range(len(w_s)):
    inn = np.where((w_tag == w_s[i]) & (slope_tag > 2))[0]
    # PE
    avg_PE = np.nanmean(pe_dg[:, inn], axis=1)
    ax1.plot(sc_x, avg_PE[1:mm] / dk, color=w_cols[i], label=r'PE$_{w = ' + str(w_s[i]) + '}$', linewidth=1.5)
    ax1.scatter(sc_x, avg_PE[1:mm] / dk, color=w_cols[i], s=15)
    # KE
    avg_KE = np.nanmean(ke_dg[:, inn], axis=1)
    ax2.plot(sc_x, avg_KE[1:mm] / dk, color=w_cols[i], label=r'KE$_{w = ' + str(w_s[i]) + '}$', linewidth=1.5)
    ax2.scatter(sc_x, avg_KE[1:mm] / dk, color=w_cols[i], s=15)  # DG KE
    ax2.plot([l_lim, sc_x[0]], avg_KE[0:2] / dk, color=w_cols[i], linewidth=1.5)  # DG KE_0
    ax2.scatter(l_lim, avg_KE[0] / dk, color=w_cols[i], s=15, facecolors='none')  # DG KE_0
    if i < 1:
        modeno = '1', '2', '3', '4', '5', '6', '7', '8'
        for j in range(len(modeno)):
            ax2.text(sc_x[j], (avg_KE[j + 1] + (avg_KE[j + 1] / 2)) / dk, modeno[j], color='k', fontsize=10)

# Model
ax1.plot(sc_x, avg_PE_model[1:mm] / dk, color='k', label='PE$_{Model}$', linewidth=2)
ax1.scatter(sc_x, avg_PE_model[1:mm] / dk, color='k', s=20)
ax2.plot(sc_x, avg_KE_model[1:mm] / dk, color='k', label='KE$_{Model}$', linewidth=2)
ax2.scatter(sc_x, avg_KE_model[1:mm] / dk, color='k', s=20)
ax2.plot([l_lim, sc_x[0]], avg_KE_model[0:2] / dk, color='k', linewidth=2)
ax2.scatter(l_lim, avg_KE_model[0] / dk, color='k', s=25, facecolors='none')


avg_KE_model_ind_all = np.nanmean(ke_mod_tot, axis=1)
avg_PE_model_ind_all = np.nanmean(pe_mod_tot, axis=1)
# lg = np.where(avg_N2 < 1*10**-10)[0]
# for i in range(len(avg_N2) - lg[1]):
#     avg_N2[lg[1] + i] = avg_N2[lg[1] - 1] - 1*10**-10
# PE_SD, PE_GM, GMPE, GMKE = PE_Tide_GM(1025, -1.0 * z_grid_n2, len(avg_KE_model), avg_N2[:, None], ff)
ax2.plot(sc_x, avg_KE_model_ind_all[1:mm] / dk, color='k', label='KE$_{Model_{inst}}$', linewidth=1, linestyle='--')
ax2.scatter(sc_x, avg_KE_model_ind_all[1:mm] / dk, color='k', s=5)
ax2.plot([l_lim, sc_x[0]], avg_KE_model_ind_all[0:2] / dk, color='k', linewidth=1, linestyle='--')
ax2.scatter(l_lim, avg_KE_model_ind_all[0] / dk, color='k', s=5, facecolors='none')
# model GMKE
# ax2.plot(sc_x, GMKE[1:mm] / dk, color='k', linewidth=0.75)
ax1.plot(sc_x, avg_PE_model_ind_all[1:mm] / dk, color='k', label='PE$_{Model_{inst}}$', linewidth=1, linestyle='--')
ax1.scatter(sc_x, avg_PE_model_ind_all[1:mm] / dk, color='k', s=5)

limm = 5
ax1.set_xlim([l_lim, 0.5 * 10 ** 2])
ax2.set_xlim([l_lim, 0.5 * 10 ** 2])
ax2.set_ylim([10 ** (-4), 3 * 10 ** 2])
ax1.set_yscale('log')
ax1.set_xscale('log')
ax2.set_xscale('log')

# ax1.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
ax1.set_ylabel('Spectral Density', fontsize=12)  # ' (and Hor. Wavenumber)')
# ax2.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
ax1.set_xlabel('Mode Number', fontsize=12)
ax2.set_xlabel('Mode Number', fontsize=12)
ax1.set_title('PE Spectrum (gs='
             + str(np.int(slope_s[1])) + ':1)', fontsize=12)
ax2.set_title('KE Spectrum (gs='
             + str(np.int(slope_s[1])) + ':1)', fontsize=12)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=12)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=12)
ax1.grid()
plot_pro(ax2)
if save_e > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/model_dg_vel_energy_bats.png', dpi=200)
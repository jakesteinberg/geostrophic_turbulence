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


file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/vel_anom_slow2*.pkl')
savee = 0

direct_anom = []
# igw_var = np.nan * np.ones(len(file_list))
for i in range(len(file_list)):
    pkl_file = open(file_list[i], 'rb')
    MOD = pickle.load(pkl_file)
    pkl_file.close()

    glider_v = MOD['dg_v'][:]
    model_v = MOD['model_u_at_mwv']
    z_grid = MOD['dg_z'][:]
    slope_error = MOD['shear_error'][:]
    igw = MOD['igw_var']
    ke_mod_0 = MOD['KE_mod'][:]
    pe_mod_0 = MOD['PE_model'][:]
    ke_dg_0 = MOD['KE_dg'][:]
    pe_dg_0 = MOD['PE_dg_avg'][:]
    pe_dg_ind_0 = MOD['PE_dg'][:]

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
    else:
        v = np.concatenate((v, glider_v.copy()), axis=1)
        mod_v = np.concatenate((mod_v, model_mean_per_mw), axis=1)
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

# vertical shear error as a function depth and igwsignal
matplotlib.rcParams['figure.figsize'] = (6.5, 8)
f, ax = plt.subplots()
low_er_mean = np.nan * np.ones(len(z_grid))
for i in range(len(z_grid)):
    low = np.where(igw_var[i, :] < .2)[0]
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
if savee > 0:
    f.savefig("/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_04_18/model_dg_per_shear_err_steep.png", dpi=200)


# all velocity anomalies colored by max abs vel attained
cmap = plt.cm.get_cmap('Spectral_r')
f, ax = plt.subplots()
this_one = mod_time_out
this_anom = anoms_time
max_mod_v = np.nan * np.ones(np.shape(this_one)[1])
avg_anom1 = np.nan * np.ones(np.shape(this_one)[1])
avg_anom2 = np.nan * np.ones(np.shape(this_one)[1])
avg_anom3 = np.nan * np.ones(np.shape(this_one)[1])
for i in range(np.shape(this_one)[1]):
    max_mod_v[i] = np.nanmax(this_one[:, i]**2)
    avg_anom1[i] = np.nanmean(this_anom[0:50, i]**2)
    avg_anom2[i] = np.nanmean(this_anom[5:100, i]**2)
    avg_anom3[i] = np.nanmean(this_anom[100:, i]**2)
    if np.nanmax(np.abs(this_one[:, i])) > 0.25:
        ax.plot(this_anom[:, i], z_grid, color=cmap(1))
    elif (np.nanmax(np.abs(this_one[:, i])) > 0.2) & (np.nanmax(np.abs(this_one[:, i])) < 0.25):
        ax.plot(this_anom[:, i], z_grid, color=cmap(.8))
    elif (np.nanmax(np.abs(this_one[:, i])) > 0.15) & (np.nanmax(np.abs(this_one[:, i])) < 0.2):
        ax.plot(this_anom[:, i], z_grid, color=cmap(.6))
    elif (np.nanmax(np.abs(this_one[:, i])) > 0.1) & (np.nanmax(np.abs(this_one[:, i])) < 0.15):
        ax.plot(this_anom[:, i], z_grid, color=cmap(.4))
    elif (np.nanmax(np.abs(this_one[:, i])) > 0.05) & (np.nanmax(np.abs(this_one[:, i])) < 0.1):
        ax.plot(this_anom[:, i], z_grid, color=cmap(.2))
    else:
        ax.plot(this_anom[:, i], z_grid, color=cmap(.01))
ax.plot(np.nanmean(this_anom, axis=1), z_grid, linewidth=2.2, color='k')
plot_pro(ax)

# RMS at different depths
matplotlib.rcParams['figure.figsize'] = (6.5, 8)
f, ax = plt.subplots()
# ax.plot(np.nanmean(anoms, axis=1), z_grid, linewidth=2.2, color='k')
mm = np.nanmean(anoms**2, axis=1)
mm_std = np.nan * np.ones(len(z_grid))
for i in range(len(z_grid)):
    mm_std[i] = np.nanstd(anoms[i, :]**2)
# ax.errorbar(mm, z_grid, xerr=mm_std)
ax.plot(mm, z_grid, linewidth=2.2, color='k')
ax.plot(mm_std, z_grid, linewidth=1.5, color='k', linestyle='--')
# ax.text(0.005, -2400, 'Mean Error = ' + str(np.round(np.nanmean(anoms), 3)) + ' m/s')
ax.set_xlim([0, .01])
ax.set_ylim([-3000, 0])
ax.set_xlabel(r'm$^2$/s$^2$')
ax.set_ylabel('Depth [m]')
ax.set_title(r'Glider M/W Velocity Profile Error ($u_{g}$ - $\overline{u_{model}}$)$^2$ ('
             + str(MOD['glide_slope']) + ':1, w=' + str(MOD['dg_w']) + ' m/s)')
plot_pro(ax)
if savee > 0:
    f.savefig("/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_04_18/model_dg_vel2_error_steep.png", dpi=200)

# binss = np.arange(0, 0.002, 0.0001)
# subax = f.add_axes([0.2, 0.75, .225, .08])
# subax.hist(this_anom[9, :]**2, bins=binss)
# subax.set_xlim([0, 0.002])
# subax.set_ylim([0, 100])
# subax.set_title(str(z_grid[9]) + 'm', fontsize=7)
# subax.set_xticks([0, 0.001])
# subax.tick_params(labelsize=7)
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
#
# subax = f.add_axes([0.2, 0.55, .225, .08])
# subax.hist(this_anom[49, :]**2, bins=binss)
# subax.set_xlim([0, 0.002])
# subax.set_ylim([0, 100])
# subax.set_title(str(z_grid[49]) + 'm', fontsize=7)
# subax.set_xticks([0, 0.001])
# subax.tick_params(labelsize=7)
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
#
# subax = f.add_axes([0.2, 0.35, .225, .08])
# subax.hist(this_anom[74, :]**2, bins=binss)
# subax.set_xlim([0, 0.002])
# subax.set_ylim([0, 100])
# subax.set_title(str(z_grid[74]) + 'm', fontsize=7)
# subax.set_xticks([0, 0.001])
# subax.tick_params(labelsize=7)
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
#
# subax = f.add_axes([0.2, 0.15, .225, .08])
# subax.hist(this_anom[129, :]**2, bins=binss)
# subax.set_xlim([0, 0.002])
# subax.set_ylim([0, 100])
# subax.set_title(str(z_grid[129]) + 'm', fontsize=7)
# subax.set_xticks([0, 0.001])
# subax.tick_params(labelsize=7)
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)

f, ax = plt.subplots()
for i in range(np.shape(anoms)[1]):
    ax.plot(anoms[:, i], z_grid, color='#D3D3D3')
ax.plot(np.nanmean(anoms, axis=1), z_grid, color='r', linewidth=2)
ax.text(0.06, -2400, 'Mean Error = ' + str(np.round(np.nanmean(anoms), 3)) + ' m/s')
ax.set_title(r'Glider M/W Velocity Profile Error ($u_{g}$ - $\overline{u_{model}}$)')
ax.set_ylabel('z [m]')
ax.set_xlabel('Velocity Error [m/s]')
ax.set_xlim([-.15, .15])
plot_pro(ax)
if savee > 0:
    f.savefig("/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_04_18/model_dg_vel_error_steep.png", dpi=200)

# ---------------------------------------------------------------------------------------------------------------------
# --- PLOT ENERGY SPECTRA
ff = np.pi * np.sin(np.deg2rad(44)) / (12 * 1800)  # Coriolis parameter [s^-1]
# --- Background density
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Model/background_density.pkl', 'rb')
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
mm = 20
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

# test
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
ax1.set_title('PE Spectrum ('
             + str(MOD['glide_slope']) + ':1, w=' + str(MOD['dg_w']) + ' m/s)', fontsize=12)
ax2.set_title('KE Spectrum ('
             + str(MOD['glide_slope']) + ':1, w=' + str(MOD['dg_w']) + ' m/s)', fontsize=12)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=12)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=12)
ax1.grid()
plot_pro(ax2)
# f.savefig("/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_04_18/model_dg_vel_energy_slow2.png", dpi=200)
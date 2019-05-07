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


file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/vel*.pkl')

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

# vertical shear error as a function depth and igwsignal
f, ax = plt.subplots()
for i in range(len(z_grid)):
    low = np.where(igw_var[i, :] < .5)[0]
    ax.scatter(slope_er[i, :], z_grid[i] * np.ones(len(slope_er[i, :])), s=2, color='b')
    ax.scatter(slope_er[i, low], z_grid[i] * np.ones(len(slope_er[i, low])), s=2, color='r')
    ax.scatter(np.nanmean(slope_er[i, low]), z_grid[i], s=15, color='r')
ax.plot(np.nanmedian(slope_er, axis=1), z_grid, color='k', linewidth=2)
ax.set_xscale('log')
ax.set_xlabel('Percent Error')
ax.set_ylabel('z [m]')
ax.set_title('Percent Error between Model Shear and Glider Shear')
ax.set_xlim([1, 10**4])
plot_pro(ax)

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

# Ratio of error magnitude at all depths to model velocity at all depths
f, ax = plt.subplots()
for i in range(np.shape(mod_v)[1]):
    ax.scatter(np.abs(anoms[:, i]/mod_v[:, i]), z_grid, color='#D3D3D3', s=4)
    if np.nanmax(np.abs(mod_v[:, i])) > 0.175:
        ax.scatter(np.abs(anoms[:, i] / mod_v[:, i]), z_grid, color='r', s=3)
    if np.nanmax(np.abs(mod_v[:, i])) < 0.05:
        ax.scatter(np.abs(anoms[:, i] / mod_v[:, i]), z_grid, color='g', s=3)
ax.plot(np.nanmean(np.abs(anoms), axis=1)/np.nanmean(np.abs(mod_v), axis=1), z_grid, color='k')
ax.set_xlim([0, 4])
plot_pro(ax)

# RMS at different depths
matplotlib.rcParams['figure.figsize'] = (6.5, 8)
f, ax = plt.subplots()
ax.plot(np.nanmean(anoms, axis=1), z_grid, linewidth=2.2, color='k')
mm = np.nanmean(anoms, axis=1)
mm_std = np.nan * np.ones(len(z_grid))
for i in range(len(z_grid)):
    mm_std[i] = np.nanstd(this_anom[i, :])
ax.errorbar(mm, z_grid, xerr=mm_std)
ax.text(0.04, -2400, 'Mean Error = ' + str(np.round(np.nanmean(anoms), 3)) + ' m/s')

binss = np.arange(0, 0.002, 0.0001)
subax = f.add_axes([0.2, 0.75, .225, .08])
subax.hist(this_anom[9, :]**2, bins=binss)
subax.set_xlim([0, 0.002])
subax.set_ylim([0, 100])
subax.set_title(str(z_grid[9]) + 'm', fontsize=7)
subax.set_xticks([0, 0.001])
subax.tick_params(labelsize=7)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

subax = f.add_axes([0.2, 0.55, .225, .08])
subax.hist(this_anom[49, :]**2, bins=binss)
subax.set_xlim([0, 0.002])
subax.set_ylim([0, 100])
subax.set_title(str(z_grid[49]) + 'm', fontsize=7)
subax.set_xticks([0, 0.001])
subax.tick_params(labelsize=7)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

subax = f.add_axes([0.2, 0.35, .225, .08])
subax.hist(this_anom[74, :]**2, bins=binss)
subax.set_xlim([0, 0.002])
subax.set_ylim([0, 100])
subax.set_title(str(z_grid[74]) + 'm', fontsize=7)
subax.set_xticks([0, 0.001])
subax.tick_params(labelsize=7)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

subax = f.add_axes([0.2, 0.15, .225, .08])
subax.hist(this_anom[129, :]**2, bins=binss)
subax.set_xlim([0, 0.002])
subax.set_ylim([0, 100])
subax.set_title(str(z_grid[129]) + 'm', fontsize=7)
subax.set_xticks([0, 0.001])
subax.tick_params(labelsize=7)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

ax.set_xlim([-.15, .1])
ax.set_ylim([-3000, 0])
ax.set_xlabel('m/s')
ax.set_ylabel('Depth [m]')
ax.set_title(r'Glider M/W Velocity Profile Error ($v_{dg}$ - $\overline{v_{model}}$)')
plot_pro(ax)
# f.savefig("/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_04_18/model_dg_vel_error_w10_gs2.png", dpi=200)

f, ax = plt.subplots()
x_range = np.arange(0, .2, .02)
ax.plot(x_range, np.zeros(len(x_range)), linestyle='--', color='k')
ax.scatter(max_mod_v, avg_anom1, s=5, color='r', label='0-1000m')
ax.scatter(max_mod_v, avg_anom2, s=5, color='g', label='1000-2000m')
ax.scatter(max_mod_v, avg_anom3, s=5, color='b', label='2000m-')
ax.plot(x_range, np.nanmean(avg_anom1) * np.ones(len(x_range)), color='r', linewidth=0.5)
ax.plot(x_range, np.nanmean(avg_anom2) * np.ones(len(x_range)), color='g', linewidth=0.5)
ax.plot(x_range, np.nanmean(avg_anom3) * np.ones(len(x_range)), color='b', linewidth=0.5)
a1p = np.polyfit(max_mod_v, avg_anom1, 1)
a1f = np.polyval(a1p, max_mod_v)
ax.plot(max_mod_v, a1f, color='r')
a2p = np.polyfit(max_mod_v, avg_anom2, 1)
a2f = np.polyval(a2p, max_mod_v)
ax.plot(max_mod_v, a2f, color='g')
a3p = np.polyfit(max_mod_v, avg_anom3, 1)
a3f = np.polyval(a3p, max_mod_v)
ax.plot(max_mod_v, a3f, color='b')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=12)
# ax.set_xlim([0, .5])
# ax.set_ylim([0, .1])
ax.set_xlabel('Maximum Square Model Velocity ')
ax.set_ylabel(r'Average ( DG Vel. - Model Vel. ) $^2$ ')
ax.set_title('M/W Velocity Profile Error As Function of Model Velocity and Depth')
plot_pro(ax)

f, ax = plt.subplots()
for i in range(np.shape(anoms)[1]):
    ax.plot(anoms[:, i], z_grid, color='#D3D3D3')
ax.plot(np.nanmean(anoms, axis=1), z_grid, color='r', linewidth=2)
ax.text(0.06, -2400, 'Mean Error = ' + str(np.round(np.nanmean(anoms), 3)) + ' m/s')
ax.set_title(r'Glider M/W Velocity Profile Error ($v_{dg}$ - $\overline{v_{model}}$)')
ax.set_ylabel('z [m]')
ax.set_xlabel('Velocity Error [m/s]')
ax.set_xlim([-.15, .15])
plot_pro(ax)

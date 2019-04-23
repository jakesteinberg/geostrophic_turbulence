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
import matplotlib.pyplot as plt
from toolkit import plot_pro, nanseg_interp
from zrfun import get_basic_info, get_z


file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/velocity*.pkl')

direct_anom = []
for i in range(len(file_list)):
    pkl_file = open(file_list[i], 'rb')
    MOD = pickle.load(pkl_file)
    pkl_file.close()

    glider_v = MOD['dg_v'][:]
    model_v = MOD['model_u_at_mwv']
    z_grid = MOD['dg_z'][:]

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

    else:
        v = np.concatenate((v, glider_v.copy()), axis=1)
        mod_v = np.concatenate((mod_v, model_mean_per_mw), axis=1)
        mod_space_out = np.concatenate((mod_space_out, mod_space), axis=1)
        mod_time_out = np.concatenate((mod_time_out, mod_time), axis=1)
        anoms = np.concatenate((anoms, glider_v - model_mean_per_mw), axis=1)
        anoms_space = np.concatenate((anoms_space, spatial_anom), axis=1)
        anoms_time = np.concatenate((anoms_time, time_anom), axis=1)

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
plot_pro(ax)

f, ax = plt.subplots()
x_range = np.arange(0, .25, .02)
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
ax.set_title(r'Glider M/W Velocity Profile Error ($v_{dg}$ - $\overline{v_model}$)')
ax.set_ylabel('z [m]')
ax.set_xlabel('Velocity Error [m/s]')
ax.set_xlim([-.10, .15])
plot_pro(ax)

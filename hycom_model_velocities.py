import numpy as np
import glob
import pickle
from netCDF4 import Dataset
import scipy.io as si
# -- plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from toolkit import plot_pro

# -- BATS --
pp = '1'
filename = 'HYCOM_hrly_b_3087N_214_01_228_00'  # E_W profile 1
# filename = 'HYCOM_hrly_b_6336W_214_01_228_00'  # N_S profile 1
# pp = '2'
# filename = 'HYCOM_hrly_b_3128N_214_01_228_00'  # E_W profile 2
# filename = 'HYCOM_hrly_b_6376W_214_01_228_00'  # N_S profile 2

# -- 36N,65W --
# pp = '1'
# filename = 'HYCOM_hrly_n_3532N_214_01_228_00'  # E_W profile 1
# filename = 'HYCOM_hrly_n_6424W_214_01_228_00'  # N_S profile 1
# pp = '2'
# filename = 'HYCOM_hrly_n_3597N_214_01_228_00'  # E_W profile 2
# filename = 'HYCOM_hrly_n_6504W_214_01_228_00'  # N_S profile 2

import_hy = si.loadmat('/Users/jake/Documents/baroclinic_modes/Model/HYCOM/' + filename + '.mat')
MOD = import_hy['out']
ref_lat = MOD['ref_lat'][0][0][0][0]
sig0_out_s = MOD['gamma'][0][0][:]
ct_out_s = MOD['temp'][0][0][:]  # temp
sa_out_s = MOD['salin'][0][0][:]  # salin
transect_dir = MOD['transect'][0][0][0][0]  # matlab keys are stupidly flipped (E_W = 0, while here E_W = 1)
if transect_dir < 1:  # E/W transect
    u_out_s = MOD['v'][0][0][:] # vel across transect
    u_off_out_s = MOD['u'][0][0][:] # vel along transect
    xy_grid = MOD['xy_grid'][0][0][0] * 1000.0
    E_W = 1
else:
    u_out_s = MOD['u'][0][0][:]  # vel across transect
    u_off_out_s = MOD['v'][0][0][:]  # vel along transect
    xy_grid = np.squeeze(MOD['xy_grid'][0][0] * 1000.0)
    E_W = 0

time_ord_s = np.arange(1, np.shape(sig0_out_s)[0]) / 24.0
z_grid = -1.0 * MOD['z_grid'][0][0][:, 0]

mod_u_rec = u_out_s
mod_u_off_rec = u_off_out_s
msz = np.array([20, 40, 80, 180, 230])  # first index is 100m, second is 500m, third is 2000m
# msz = np.array([10, 30, 105, 150, 200])
msx = np.array([3, 10, 20, 45])  # first ~ 10km, second 100 km
mod_u_rec = mod_u_rec[:, :, msx]
mod_u_off_rec = mod_u_off_rec[:, :, msx]
mod_u_rec = mod_u_rec[:, msz, :]
mod_u_off_rec = mod_u_off_rec[:, msz, :]

# --- Look at LiveOcean model velocity fields
# load model velocity time series
these_paths = np.flipud(glob.glob('/Users/jake/Documents/baroclinic_modes/Model/LiveOcean/e_w_extraction_nov*'))
for i in range(len(these_paths)):
    pkl_file = open(these_paths[i] + '/gamma_output/extracted_gridded_gamma.pkl', 'rb')
    lo_MOD = pickle.load(pkl_file)
    pkl_file.close()
    u_out_s = lo_MOD['vel_cross_transect'][:]
    u_off_out_s = lo_MOD['vel_along_transect'][:]
    date_out_s = lo_MOD['date'][:]
    lo_xy_grid = lo_MOD['dist_grid'][:]
    lo_z_grid = np.flipud(lo_MOD['z'][:])
    lo_mod_u_rec = np.flipud(u_out_s)
    lo_mod_u_off_rec = np.flipud(u_off_out_s)
    lo_msz = np.array([7, 14, 34, 74, 139]) # first is 100m, second is 500m, third is 2000m (select to match moorings)
    # old one lo_msz = np.array([4, 24, 99, 99, 134])
    lo_msx = np.array([10, 40, 80, 110]) # horizontal location along transect
    lo_mod_u_rec = lo_mod_u_rec[:, lo_msx, :]
    lo_mod_u_off_rec = lo_mod_u_off_rec[:, lo_msx, :]
    if i < 1:
        lo_mod_u_rec_0 = lo_mod_u_rec[lo_msz, :, :]
        lo_mod_u_off_rec_0 = lo_mod_u_off_rec[lo_msz, :, :]
    else:
        lo_mod_u_rec_0 = np.concatenate((lo_mod_u_rec_0, lo_mod_u_rec[lo_msz, :, :]), axis=2)
        lo_mod_u_off_rec_0 = np.concatenate((lo_mod_u_off_rec_0, lo_mod_u_off_rec[lo_msz, :, :]), axis=2)

# --- LOW PASS FILTER velocity time series (like mooring data)
hy_test1 = mod_u_rec  # [time, dep_levs, xy_pos]
hy_test1_filt = np.nan * np.ones(np.shape(hy_test1))
hy_test1_off = mod_u_off_rec  # [time, dep_levs, xy_pos]
hy_test1_off_filt = np.nan * np.ones(np.shape(hy_test1_off))
lo_test1 = lo_mod_u_rec_0  # [dep_levs, xy_pos, time]
lo_test1_filt = np.nan * np.ones((np.shape(lo_test1)[2], np.shape(lo_test1)[0], np.shape(lo_test1)[1]))
lo_test1_off = lo_mod_u_off_rec_0  # [dep_levs, xy_pos, time]
lo_test1_off_filt = np.nan * np.ones((np.shape(lo_test1)[2], np.shape(lo_test1)[0], np.shape(lo_test1)[1]))
for m in range(len(lo_msx)):
    hy_test1 = np.squeeze(mod_u_rec[:, :, m])  # [time, dep_levs, xy_pos]
    hy_test1_off = np.squeeze(mod_u_off_rec[:, :, m])  # [time, dep_levs, xy_pos]
    for i in range(12, len(hy_test1)-12):
        hy_test1_filt[i, :, m] = np.nanmean(hy_test1[i-12:i+12, :], axis=0)
        hy_test1_off_filt[i, :, m] = np.nanmean(hy_test1_off[i - 12:i + 12, :], axis=0)

    lo_test1 = np.transpose(np.squeeze(lo_mod_u_rec_0[:, m, :])) # [dep_levs, xy_pos, time] --> [time, dep]
    lo_test1_off = np.transpose(np.squeeze(lo_mod_u_off_rec_0[:, m, :]))  # [dep_levs, xy_pos, time] --> [time, dep]
    for i in range(12, len(lo_test1)-12):
        lo_test1_filt[i, :, m] = np.nanmean(lo_test1[i-12:i+12, :], axis=0)
        lo_test1_off_filt[i, :, m] = np.nanmean(lo_test1_off[i - 12:i + 12, :], axis=0)

# --- SAVE for EOF time series of both model outputs, sampled as if they were moorings
# write python dict to a file
sa = 1
if sa > 0:
    my_dict = {'hy_depth': z_grid, 'hy_mod_u': mod_u_rec, 'hy_mod_u_off': mod_u_off_rec,
               'hy_z_samp_deps': z_grid[msz], 'hy_x_samp_pos': xy_grid[msx], 'hy_time': 'time step = 1 hour',
               'hy_mod_u_xy1_lo_pass': hy_test1_filt, 'hy_mod_u_off_xy1_lo_pass': hy_test1_off_filt,
               'lo_depth': lo_z_grid, 'lo_mod_u': lo_mod_u_rec_0, 'lo_mod_u_off': lo_mod_u_off_rec_0,
               'lo_z_samp_deps': lo_z_grid[lo_msz], 'lo_x_samp_pos': lo_xy_grid[lo_msx], 'lo_time': 'time step = 1 hour',
               'lo_mod_u_xy1_lo_pass': lo_test1_filt, 'lo_mod_u_off_xy1_lo_pass': lo_test1_off_filt,}
    output = open('/Users/jake/Documents/baroclinic_modes/Model/model_mooring_samplings.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()

# PLOT
matplotlib.rcParams['figure.figsize'] = (12, 7)
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(np.arange(0, len(np.squeeze(mod_u_rec[:, 0, 0]))), np.squeeze(mod_u_rec[:, 0, 0]), color='b')
ax1.plot(np.arange(0, len(np.squeeze(mod_u_rec[:, 0, 0]))), hy_test1_filt[:, 0, 0], color='k')
ax1.plot(np.arange(0, len(np.squeeze(mod_u_off_rec[:, 0, 0]))), np.squeeze(mod_u_off_rec[:, 0, 0]), color='r')
ax1.set_ylim([-.5, .5])
# ax1.set_title('E_W = ' + str(E_W) + ', x = ' + str(xy_grid[msx[0]]) + ' km, z = ' + str(z_grid[msz[0]]) + ' m')
ax1.set_ylabel('m s$^{-1}$')
ax2.plot(np.arange(0, len(np.squeeze(mod_u_rec[:, 1, 0]))), np.squeeze(mod_u_rec[:, 1, 0]), color='b')
ax2.plot(np.arange(0, len(np.squeeze(mod_u_rec[:, 1, 0]))), hy_test1_filt[:, 1, 0], color='k')
ax2.plot(np.arange(0, len(np.squeeze(mod_u_off_rec[:, 1, 0]))), np.squeeze(mod_u_off_rec[:, 1, 0]), color='r')
ax2.set_ylim([-.5, .5])
# ax2.set_title('z = ' + str(z_grid[msz[1]]) + ' m')
ax2.set_ylabel('m s$^{-1}$')
ax3.plot(np.arange(0, len(np.squeeze(mod_u_rec[:, 3, 0]))), np.squeeze(mod_u_rec[:, 3, 0]), label='HYCOM cross-track', color='b')
ax3.plot(np.arange(0, len(np.squeeze(mod_u_rec[:, 3, 0]))), hy_test1_filt[:, 3, 0], color='k')
ax3.plot(np.arange(0, len(np.squeeze(mod_u_off_rec[:, 3, 0]))), np.squeeze(mod_u_off_rec[:, 3, 0]), label='HYCOM along-track', color='r')
ax3.set_ylim([-.5, .5])
# ax3.set_title('z = ' + str(z_grid[msz[3]]) + ' m')
ax3.set_ylabel('m s$^{-1}$')
ax3.set_xlabel('hour')
# handles, labels = ax3.get_legend_handles_labels()
# ax3.legend(handles, labels, fontsize=12)
# plot_pro(ax3)
# f.savefig('/Users/jake/Documents/glider_flight_sim_paper/lo_back_uv_10.png', dpi=300)

# LiveOcean
ax1.plot(np.arange(0, len(np.squeeze(lo_mod_u_rec_0[0, 0, :]))), np.squeeze(lo_mod_u_rec_0[0, 0, :]), color='b', linestyle='--')
ax1.plot(np.arange(0, len(np.squeeze(lo_mod_u_rec_0[0, 0, :]))), lo_test1_filt[:, 0, 0], color='k')
ax1.plot(np.arange(0, len(np.squeeze(lo_mod_u_off_rec_0[0, 0, :]))), np.squeeze(lo_mod_u_off_rec_0[0, 0, :]), color='r', linestyle='--')
ax1.set_ylim([-.5, .5])
ax1.set_title('E_W = ' + str(E_W) + ', x = 10 km, z = ' + str(lo_z_grid[lo_msz[0]]) + ' m')
ax1.set_ylabel('m s$^{-1}$')
ax1.grid()
ax2.plot(np.arange(0, len(np.squeeze(lo_mod_u_rec_0[1, 0, :]))), np.squeeze(lo_mod_u_rec_0[1, 0, :]), color='b', linestyle='--')
ax2.plot(np.arange(0, len(np.squeeze(lo_mod_u_off_rec_0[1, 0, :]))), np.squeeze(lo_mod_u_off_rec_0[1, 0, :]), color='r', linestyle='--')
ax2.set_ylim([-.5, .5])
ax2.set_title('z = ' + str(lo_z_grid[lo_msz[1]]) + ' m')
ax2.set_ylabel('m s$^{-1}$')
ax2.grid()
ax3.plot(np.arange(0, len(np.squeeze(lo_mod_u_rec_0[3, 0, :]))), np.squeeze(lo_mod_u_rec_0[3, 0, :]), label='LiveOcean cross-track', color='b', linestyle='--')
ax3.plot(np.arange(0, len(np.squeeze(lo_mod_u_off_rec_0[3, 0, :]))), np.squeeze(lo_mod_u_off_rec_0[3, 0, :]), label='LiveOcean along-track', color='r', linestyle='--')
ax3.set_ylim([-.5, .5])
ax3.set_title('z = ' + str(lo_z_grid[lo_msz[3]]) + ' m')
ax3.set_ylabel('m s$^{-1}$')
ax3.set_xlabel('hour')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels, fontsize=8)
plot_pro(ax3)
# f.savefig('/Users/jake/Documents/glider_flight_sim_paper/hy_lo_back_uv_10_2.png', dpi=300)

# --
# -- second location along transect
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(np.arange(0, len(np.squeeze(mod_u_rec[:, 0, 2]))), np.squeeze(mod_u_rec[:, 0, 2]), color='b')
ax1.plot(np.arange(0, len(np.squeeze(mod_u_off_rec[:, 0, 2]))), np.squeeze(mod_u_off_rec[:, 0, 2]), color='r')
ax1.set_ylim([-.5, .5])
# ax1.set_title('hy_E_W = ' + str(E_W) + ', x = ' + str(xy_grid[msx[2]]) + ' km, z = ' + str(z_grid[msz[0]]) + ' m')
ax1.set_ylabel('m s$^{-1}$')
ax2.plot(np.arange(0, len(np.squeeze(mod_u_rec[:, 1, 2]))), np.squeeze(mod_u_rec[:, 1, 2]), color='b')
ax2.plot(np.arange(0, len(np.squeeze(mod_u_off_rec[:, 1, 2]))), np.squeeze(mod_u_off_rec[:, 1, 2]), color='r')
ax2.set_ylim([-.5, .5])
# ax2.set_title('z = ' + str(z_grid[msz[1]]) + ' m')
ax2.set_ylabel('m s$^{-1}$')
ax3.plot(np.arange(0, len(np.squeeze(mod_u_rec[:, 3, 2]))), np.squeeze(mod_u_rec[:, 3, 2]), label='HYCOM cross-track', color='b')
ax3.plot(np.arange(0, len(np.squeeze(mod_u_off_rec[:, 3, 2]))), np.squeeze(mod_u_off_rec[:, 3, 2]), label='HYCOM along-track', color='r')
ax3.set_ylim([-.5, .5])
# ax3.set_title('z = ' + str(z_grid[msz[3]]) + ' m')
ax3.set_xlabel('hour')
ax3.set_ylabel('m s$^{-1}$')

# LiveOcean
ax1.plot(np.arange(0, len(np.squeeze(lo_mod_u_rec_0[0, 2, :]))), np.squeeze(lo_mod_u_rec_0[0, 2, :]), color='b', linestyle='--')
ax1.plot(np.arange(0, len(np.squeeze(lo_mod_u_off_rec_0[0, 2, :]))), np.squeeze(lo_mod_u_off_rec_0[0, 2, :]), color='r', linestyle='--')
ax1.set_ylim([-.5, .5])
ax1.set_title('hy_E_W = ' + str(E_W) + ', x = 100 km, z = ' + str(lo_z_grid[lo_msz[0]]) + ' m')
ax1.set_ylabel('m s$^{-1}$')
ax1.grid()
ax2.plot(np.arange(0, len(np.squeeze(lo_mod_u_rec_0[1, 2, :]))), np.squeeze(lo_mod_u_rec_0[1, 2, :]), color='b', linestyle='--')
ax2.plot(np.arange(0, len(np.squeeze(lo_mod_u_off_rec_0[1, 2, :]))), np.squeeze(lo_mod_u_off_rec_0[1, 2, :]), color='r', linestyle='--')
ax2.set_ylim([-.5, .5])
ax2.set_title('z = ' + str(lo_z_grid[lo_msz[1]]) + ' m')
ax2.set_ylabel('m s$^{-1}$')
ax2.grid()
ax3.plot(np.arange(0, len(np.squeeze(lo_mod_u_rec_0[3, 2, :]))), np.squeeze(lo_mod_u_rec_0[3, 2, :]), label='LiveOcean cross-track', color='b', linestyle='--')
ax3.plot(np.arange(0, len(np.squeeze(lo_mod_u_off_rec_0[3, 2, :]))), np.squeeze(lo_mod_u_off_rec_0[3, 2, :]), label='LiveOcean along-track', color='r', linestyle='--')
ax3.set_ylim([-.5, .5])
ax3.set_title('z = ' + str(lo_z_grid[lo_msz[3]]) + ' m')
ax3.set_xlabel('hour')
ax3.set_ylabel('m s$^{-1}$')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels, fontsize=8)
plot_pro(ax3)
# f.savefig('/Users/jake/Documents/glider_flight_sim_paper/hy_lo_back_uv_100_2.png', dpi=300)
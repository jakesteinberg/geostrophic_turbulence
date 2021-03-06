import numpy as np
import matplotlib.animation as manimation
import pickle
import gsw
import time as TT
import matlab.engine
import scipy.io as si
from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter
from mode_decompositions import eta_fit, vertical_modes, PE_Tide_GM, vertical_modes_f
# -- plotting
import matplotlib
import matplotlib.pyplot as plt
from toolkit import plot_pro, nanseg_interp, find_nearest

# load mat file

# -- BATS --
# pp = '1'
# filename = 'HYCOM_hrly_b_3087N_214_01_228_00'  # E_W profile 1
# filename = 'HYCOM_hrly_b_6336W_214_01_228_00'  # N_S profile 1
pp = '2'
filename = 'HYCOM_hrly_b_3128N_214_01_228_00'  # E_W profile 2
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
p_grid = gsw.p_from_z(z_grid, ref_lat)
# dive to the bottom or a specified max D_TGT
dep_bot = 1
if dep_bot > 0:
    max_depth_per_xy = np.nan * np.ones(len(xy_grid))
    max_depth_ind = np.nan * np.ones(len(xy_grid))
    for i in range(np.shape(sig0_out_s)[2]):
        deepest = z_grid[np.isnan(sig0_out_s[0, :, i])]
        if len(deepest) > 0:
            max_depth_per_xy[i] = deepest[0]
            max_depth_ind[i] = np.where(z_grid < deepest[0])[0][0]
        else:
            max_depth_per_xy[i] = z_grid[-1]
            max_depth_ind[i] = len(z_grid)
else:
    max_depth_ind = np.where(z_grid == -3000.0)[0][0] * np.ones(len(xy_grid))
    max_depth_per_xy = -3000.0 * np.ones(len(xy_grid))


sigth_levels = np.array([23, 24, 24.5, 25, 25.5, 26, 26.2, 26.4, 26.6, 26.8,
                         27, 27.2, 27.4, 27.6, 27.7, 27.8, 27.9, 27.95,
                         28, 28.05, 28.1, 28.15])

g = 9.81
rho0 = 1025.0
ff = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)  # Coriolis parameter [s^-1]
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# CREATE LOCATIONS OF FAKE GLIDER DIVE
# approximate glider as flying with vertical velocity of 0.08 m/s, glide slope of 1:3
# assume glider measurements are made every 10, 15, 30, 45, 120 second intervals
# time between each model time step is 1hr.

# main set of parameters
# errors
dac_error = 0.01/2.0  # m/s
g_error = 0.00001/2.0
t_error = 0.003/10.0
s_error = 0.01/10.0

# t_st_ind = 10      # time start index
# dg_vertical_speeds = np.array([0.1])  # m/s
# dg_glide_slope = 3
# num_dives = 4
# y_dg_s = 20000     # horizontal position, start of glider dives (75km)

z_dg_s = 0        # depth, start of glider dives
partial_mw = 0    # include exclude partial m/w estimates

plot0 = 0  # plot model, glider cross section
plot_v = 0  # plot velocity eta profiles
plot_rho = 0  # plot density at 4 depths in space and time
plot_sp = 0  # run and save gif

save_anom = 1
save_p = 0
save_v = 0
save_rho = 0

# use these settings as the background to compute instant model fields (use n/s transect)
params = np.array([[0.1, 3.0, 3.0, 50000.0, 10.0]])
u_mod_all = 1
# if on, compute model pe/ke spectrum over many many instantaneous model vel profiles

# iterate over everything
# params = np.array([[0.06, 3, 4, 10000, 10], [0.075, 3, 4, 10000, 10], [0.1, 3, 4, 10000, 10], [0.2, 3, 4, 10000, 10],
#                    [0.06, 3, 4, 10000, 144], [0.075, 3, 4, 10000, 144],
#                    [0.1, 3, 4, 10000, 144], [0.2, 3, 4, 10000, 144],
#                    [0.06, 3, 4, 50000, 10], [0.075, 3, 4, 50000, 10], [0.1, 3, 4, 50000, 10], [0.2, 3, 4, 50000, 10],
#                    [0.06, 3, 4, 50000, 144], [0.075, 3, 4, 50000, 144],
#                    [0.1, 3, 4, 50000, 144], [0.2, 3, 4, 50000, 144],
#                    [0.06, 2, 4, 10000, 10], [0.075, 2, 4, 10000, 10], [0.1, 2, 4, 10000, 10], [0.2, 2, 4, 10000, 10],
#                    [0.06, 2, 4, 10000, 144], [0.075, 2, 4, 10000, 144],
#                    [0.1, 2, 4, 10000, 144], [0.2, 2, 4, 10000, 144],
#                    [0.06, 2, 4, 50000, 10], [0.075, 2, 4, 50000, 10], [0.1, 2, 4, 50000, 10], [0.2, 2, 4, 50000, 10],
#                    [0.06, 2, 4, 50000, 144], [0.075, 2, 4, 50000, 144],
#                    [0.1, 2, 4, 50000, 144], [0.2, 2, 4, 50000, 144]])

# loop over varied flight parameters
for master in range(np.shape(params)[0]):
    dg_vertical_speed = params[master, 0]
    dg_glide_slope = params[master, 1]
    num_dives = np.int(params[master, 2])
    y_dg_s = params[master, 3]
    t_st_ind = np.int(params[master, 4])

    # need to specify D_TGT or have glider 'fly' until it hits bottom
    data_loc = np.nanmean(sig0_out_s, axis=0)  # (depth X xy_grid)


    # glider dives are simulated by considered the depth grid as fixed and march down and up while stepping horizontally
    # following the desired glide slope and vertical velocity

    # --- DG Z
    dg_z = z_grid.copy()  # vertical grid points
    # grid of vertical grid points across all dives
    dg_z_g = np.tile(np.concatenate((z_grid.copy()[:, None], z_grid.copy()[:, None]), axis=1), (1, num_dives))

    # --- DG X,Y
    dg_y = np.nan * np.ones((len(dg_z), num_dives*2))
    dg_y[0, 0] = y_dg_s

    # --- DIVE
    # first dive
    # hor start loc
    xyg_start = np.where(xy_grid > y_dg_s)[0][0]  # hor start loc
    # projected hor position at dive end
    xyg_proj_end = np.where(xy_grid >
                            (y_dg_s + dg_glide_slope*2*np.nanmean(np.abs(max_depth_per_xy[xyg_start:xyg_start+6]))))[0][0]
    if dep_bot > 0:
        data_depth_max = z_grid[np.where(np.isnan(np.mean(data_loc[:, xyg_start:xyg_proj_end+1], 1)))[0][0]] \
                         + np.abs(z_grid[-2] - z_grid[-1])  # depth of water (depth glider will dive to) estimate
    else:
        data_depth_max = z_grid[np.int(max_depth_ind[0])]

    dg_z_ind = np.where(dg_z > data_depth_max)[0]
    for i in range(1, len(dg_z_ind)):
        dg_y[i, 0] = dg_glide_slope * (dg_z[i - 1] - dg_z[i]) + dg_y[i - 1, 0]
    # first climb
    dg_y[dg_z_ind[-1], 1] = dg_y[dg_z_ind[-1], 0]
    for i in range(len(dg_z_ind) - 2, 0, -1):
        dg_y[dg_z_ind[i], 1] = dg_glide_slope * (dg_z[dg_z_ind[i]] - dg_z[dg_z_ind[i + 1]]) + dg_y[i + 1, 1]
    dg_y[0, 1] = dg_glide_slope * (dg_z[0] - dg_z[1]) + dg_y[1, 1]

    # all subsequent dive-climb cycles
    for i in range(2, num_dives*2, 2):
        # hor start loc of dive i
        xyg_start = np.where(xy_grid > (dg_y[0, i - 1] + 10))[0][0]
        # projected hor position at dive end
        # xyg_proj_end = np.where(xy_grid > ((dg_y[0, i - 1] + 10) + np.abs(2500) * 3 * 2))[0][0]
        xyg_proj_end = np.where(xy_grid >
                                ((dg_y[0, i - 1] + 10) + dg_glide_slope * 2 *
                                 np.nanmean(np.abs(max_depth_per_xy[xyg_start:xyg_start + 6]))))[0][0]
        if dep_bot > 0:
            data_depth_max = z_grid[np.where(np.isnan(np.mean(data_loc[:, xyg_start:xyg_proj_end + 1], 1)))[0][0]] \
                             + np.abs(z_grid[-2] - z_grid[-1])  # depth of water (depth glider will dive to) estimate
        else:
            data_depth_max = z_grid[np.int(max_depth_ind[0])]
        dg_z_ind = np.where(dg_z > data_depth_max)[0]
        # dive
        dg_y[0, i] = (dg_y[0, i - 1] + 10)
        for j in range(1, len(dg_z_ind)):
            dg_y[j, i] = dg_glide_slope * (dg_z[j - 1] - dg_z[j]) + dg_y[j - 1, i]
        # climb
        dg_y[dg_z_ind[-1], i + 1] = dg_y[dg_z_ind[-1], i]
        for j in range(len(dg_z_ind) - 2, 0, -1):
            dg_y[dg_z_ind[j], i + 1] = dg_glide_slope * (dg_z[dg_z_ind[j]] - dg_z[dg_z_ind[j + 1]]) + dg_y[j + 1, i + 1]
        dg_y[0, i + 1] = dg_glide_slope * (dg_z[0] - dg_z[1]) + dg_y[1, i + 1]

    # --- DG time
    dg_t = np.nan * np.ones(dg_y.shape)
    dg_t[0, 0] = time_ord_s[t_st_ind]
    for j in range(np.shape(dg_y)[1]):
        # climb portion
        if np.mod(j, 2):
            # loop over z's as the glider climbs
            max_ind = np.where(np.isnan(dg_y[:, j]))[0][0]
            dg_t[max_ind - 1, j] = dg_t[max_ind - 1, j - 1] + 50. * (1/(60.*60.*24.)) # first value in climb
            this_dg_z = dg_z[0:max_ind]
            for i in range(len(dg_z[0:max_ind]) - 2, 0, -1):
                dg_t[i, j] = dg_t[i + 1, j] + (np.abs(this_dg_z[i] - this_dg_z[i - 1]) /
                                               dg_vertical_speed) / (60 * 60 * 24)
            dg_t[0, j] = dg_t[1, j] + (np.abs(this_dg_z[0] - this_dg_z[1]) / dg_vertical_speed) / (60 * 60 * 24)
        # dive portion
        else:
            if j < 1:
                dg_t[0, j] = dg_t[0, 0]
            else:
                dg_t[0, j] = dg_t[0, j - 1] + 50. * (1/(60.*60.*24.))
            # loop over z's as the glider dives
            max_ind = np.where(np.isnan(dg_y[:, j]))[0][0]
            for i in range(1, len(dg_z[0:max_ind])):
                dg_t[i, j] = dg_t[i - 1, j] + (np.abs(dg_z_g[i, j] - dg_z_g[i - 1, j]) /
                                               dg_vertical_speed) / (60 * 60 * 24)

    # saving labels
    dg_t_s = dg_t[0, 0]
    dg_t_e = dg_t[0, -1]
    tag = str(np.int(filename[19])) + str(np.int(np.int(filename[20:22]) + np.floor(dg_t_s))) + \
          str(np.int((dg_t[0, 0] - np.floor(dg_t_s)) * 24)) + \
          '_' + str(np.int(filename[19])) + str(np.int(np.int(filename[20:22]) + np.floor(dg_t_e))) + \
          str(np.int((dg_t[0, -1] - np.floor(dg_t_e)) * 24))
    # save filename
    if E_W > 0:
        output_filename = '/Users/jake/Documents/baroclinic_modes/Model/HYCOM/simulated_dg_velocities_bats/ve_bew' + \
                          pp + '_v' + str(np.int(100*dg_vertical_speed)) + '_slp' + str(np.int(dg_glide_slope)) + \
                          '_y' + str(np.int(y_dg_s/1000)) + '_' + tag + '.pkl'
    else:
        output_filename = '/Users/jake/Documents/baroclinic_modes/Model/HYCOM/simulated_dg_velocities_bats/ve_bns' + \
                          pp + '_v' + str(np.int(100 * dg_vertical_speed)) + '_slp' + str(np.int(dg_glide_slope)) + \
                          '_y' + str(np.int(y_dg_s/1000)) + '_' + tag + '.pkl'

    # --- Interpolation of Model to each glider measurements
    data_interp = sig0_out_s
    sa_interp = sa_out_s
    ct_interp = ct_out_s
    dg_sig0 = np.nan * np.ones(np.shape(dg_t))
    dg_sa = np.nan * np.ones(np.shape(dg_t))
    dg_ct = np.nan * np.ones(np.shape(dg_t))
    dg_p_per_prof = []
    for i in range(np.shape(dg_y)[1]):  # xy_grid
        max_ind = np.where(np.isnan(dg_y[:, i]))[0][0]
        this_dg_z = dg_z[0:max_ind]
        dg_p_per_prof.append(gsw.p_from_z(this_dg_z, 32))
        for j in range(len(this_dg_z)):  # depth
            if (i < 1) & (j < 1):
                # interpolation to horizontal position begins at t_st_ind
                dg_sig0[j, i] = np.interp(dg_y[j, i], xy_grid, data_interp[t_st_ind, j, :] + (np.random.normal(0, g_error)))
                dg_sa[j, i] = np.interp(dg_y[j, i], xy_grid, sa_interp[t_st_ind, j, :] + (np.random.normal(0, s_error)))
                dg_ct[j, i] = np.interp(dg_y[j, i], xy_grid, ct_interp[t_st_ind, j, :] + (np.random.normal(0, t_error)))
            else:
                this_t = dg_t[j, i]
                # find time bounding model runs
                nearest_model_t_over = np.where(time_ord_s > this_t)[0][0]
                # print(this_t - time_ord_s[nearest_model_t_over])
                # interpolate to hor position of glider dive for time before and after
                sig_t_before = np.interp(dg_y[j, i], xy_grid, data_interp[nearest_model_t_over - 1, j, :] + (np.random.normal(0, g_error)))  # (g_error * np.random.rand(1)))
                sig_t_after = np.interp(dg_y[j, i], xy_grid, data_interp[nearest_model_t_over, j, :] + (np.random.normal(0, g_error)))
                sa_t_before = np.interp(dg_y[j, i], xy_grid, sa_interp[nearest_model_t_over - 1, j, :] + (np.random.normal(0, s_error)))
                sa_t_after = np.interp(dg_y[j, i], xy_grid, sa_interp[nearest_model_t_over, j, :] + (np.random.normal(0, s_error)))
                ct_t_before = np.interp(dg_y[j, i], xy_grid, ct_interp[nearest_model_t_over - 1, j, :] + (np.random.normal(0, t_error)))
                ct_t_after = np.interp(dg_y[j, i], xy_grid, ct_interp[nearest_model_t_over, j, :] + (np.random.normal(0, t_error)))
                # interpolate across time
                dg_sig0[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                          [sig_t_before, sig_t_after])
                dg_sa[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                        [sa_t_before, sa_t_after])
                dg_ct[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                        [ct_t_before, ct_t_after])

    print('Simulated Glider Flight')
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # # --- convert in matlab (estimate neutral density from t,s,p)
    eng = matlab.engine.start_matlab()
    eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/')
    eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/library/')
    gamma = np.nan * np.ones(np.shape(dg_sa))
    profile_lon = np.nanmean(dg_y, 0)/(1852. * 60. * np.cos(np.deg2rad(ref_lat))) + (-63.5)
    print('Opened Matlab')
    tic = TT.clock()
    for j in range(np.shape(dg_sa)[1]):  # loop over columns
        good_ind = np.where(~np.isnan(dg_sa[:, j]))[0]
        gamma[good_ind, j] = np.squeeze(np.array(eng.eos80_legacy_gamma_n(
                matlab.double(dg_sa[good_ind, j].tolist()), matlab.double(dg_ct[good_ind, j].tolist()),
                matlab.double(dg_p_per_prof[j][good_ind].tolist()), matlab.double([profile_lon[j]]), matlab.double([ref_lat]))))
        toc = TT.clock()
        print('Time step = ' + str(j) + ' = ' + str(toc - tic) + 's')
    eng.quit()
    print('Closed Matlab')
    dg_sig0 = gamma.copy()
    # ------------------------------------------------------------------------------------------------------------------

    num_profs = np.shape(dg_sig0)[1]
    order_set = np.arange(0, np.shape(dg_sig0)[1], 2)

    # --- Estimate of DAC
    dg_dac = np.nan * np.ones(np.shape(dg_sig0)[1] - 1)
    dg_dac_off = np.nan * np.ones(np.shape(dg_sig0)[1] - 1)
    dg_dac_mid = np.nan * np.ones(np.shape(dg_sig0)[1] - 1)
    mw_time = np.nan * np.ones(np.shape(dg_sig0)[1] - 1)
    for i in range(0, num_profs-1, 2):
        min_t = np.nanmin(dg_t[:, i:i + 2])
        max_t = np.nanmax(dg_t[:, i:i + 2])
        # mto = np.where((time_ord_s >= min_t) & (time_ord_s <= max_t))[0]
        mtun = np.where(time_ord_s <= min_t)[0][-1]
        mtov = np.where(time_ord_s > max_t)[0][0]

        this_ys = np.nanmin(dg_y[:, i:i + 2])
        this_ye = np.nanmax(dg_y[:, i:i + 2])
        this_ygs = np.where(xy_grid <= this_ys)[0][-1]
        this_yge = np.where(xy_grid >= this_ye)[0][0]
        bot_lim = np.int(np.round(np.nanmean(max_depth_ind[this_ygs+1:this_yge])))
        dg_dac[i] = np.nanmean(np.nanmean(np.nanmean(u_out_s[mtun:mtov, 0:bot_lim, this_ygs+1:this_yge], axis=2), axis=0))
        dg_dac_off[i] = np.nanmean(np.nanmean(np.nanmean(u_off_out_s[mtun:mtov, :, this_ygs+1:this_yge], axis=2), axis=0))
        dg_dac_mid[i] = np.nanmean([this_ys, this_ye])
        mw_time[i] = np.nanmean([min_t, max_t])
        if i < num_profs-2:  # exclude timing of last climb (for now)
            mw_time[i + 1] = max_t

    ws = np.where(np.isnan(dg_dac_mid))[0]
    for i in range(len(ws)):
        dg_dac_mid[ws[i]] = np.nanmean([dg_dac_mid[ws[i] - 1], dg_dac_mid[ws[i] + 1]])
    # now dg_dac_mid is the same as mw_y computed later (the horizontal position of each m/w profile)

    if E_W:
        dg_dac_v = nanseg_interp(dg_dac_mid, dg_dac)
        dg_dac_u = nanseg_interp(dg_dac_mid, dg_dac_off)
    else:
        dg_dac_u = nanseg_interp(dg_dac_mid, dg_dac)
        dg_dac_v = nanseg_interp(dg_dac_mid, dg_dac_off)

    # model velocity profiles at locations within domain spanned by one m/w pattern and in the same time
    u_model = u_out_s
    u_mod_at_mw = []
    gamma_mod_at_mw = []
    u_mod_at_mw_avg = np.nan * np.ones((len(dg_z), len(dg_dac_mid)))
    gamma_mod_at_mw_avg = np.nan * np.ones((len(dg_z), len(dg_dac_mid)))
    for i in range(len(dg_dac_mid)):
        # time/space bounds of m/w profiles
        # incomplete M
        if i < 1:
            min_t = dg_t[0, 0]
            max_t = np.nanmax(dg_t[:, 2])
            min_y = np.nanmin(dg_y[:, 0])
            max_y = np.nanmean(dg_y[:, 2])
        elif (i > 0) & (i < len(dg_dac_mid) - 2):
            min_t = np.nanmin(dg_t[:, i - 1])
            max_t = np.nanmax(dg_t[:, i + 2])
            min_y = np.nanmin(dg_y[:, i - 1])
            max_y = np.nanmax(dg_y[:, i + 2])
        elif i >= len(dg_dac_mid) - 2:
            min_t = np.nanmin(dg_t[:, i - 1])
            max_t = np.nanmax(dg_t[:, -1])
            min_y = np.nanmin(dg_y[:, i - 1])
            max_y = np.nanmax(dg_y[:, -1])

        mtun = np.where(time_ord_s <= min_t)[0][-1]
        mtov = np.where(time_ord_s > max_t)[0][0]
        this_ygs = np.where(xy_grid <= min_y)[0][-1]
        this_yge = np.where(xy_grid >= max_y)[0][0]

        print(str(min_t) + '-' + str(max_t))
        print(str(time_ord_s[mtun]) + '-' + str(time_ord_s[mtov]))
        print(str(min_y) + '-' + str(max_y))
        print(str(xy_grid[this_ygs]) + '-' + str(xy_grid[this_yge]))

        u_mod_at_mw.append(u_model[mtun:mtov, :, this_ygs+1:this_yge])
        gamma_mod_at_mw.append(data_interp[mtun:mtov, :, this_ygs+1:this_yge])
        u_mod_at_mw_avg[:, i] = np.nanmean(np.nanmean(u_model[mtun:mtov, :, this_ygs+1:this_yge], axis=2), axis=0)
        gamma_mod_at_mw_avg[:, i] = np.nanmean(np.nanmean(data_interp[mtun:mtov, :, this_ygs+1:this_yge], axis=2), axis=0)
    # ---------------------------------------------------------------------------------------------------------------------
    # --- M/W estimation
    mw_y = np.nan * np.zeros(num_profs - 1)
    sigma_theta_out = np.nan * np.zeros((np.size(z_grid), num_profs - 1))
    shear = np.nan * np.zeros((np.size(z_grid), num_profs - 1))
    avg_sig_pd = np.nan * np.zeros((np.size(z_grid), num_profs - 1))
    avg_ct_pd = np.nan * np.zeros((np.size(z_grid), num_profs - 1))
    avg_sa_pd = np.nan * np.zeros((np.size(z_grid), num_profs - 1))
    isopycdep = np.nan * np.zeros((np.size(sigth_levels), num_profs))
    isopycx = np.nan * np.zeros((np.size(sigth_levels), num_profs))
    for i in order_set:
        sigma_theta_pa_M = np.nan * np.zeros(np.size(z_grid))
        sigma_theta_pa_W = np.nan * np.zeros(np.size(z_grid))
        shearM = np.nan * np.zeros(np.size(z_grid))
        shearW = np.nan * np.zeros(np.size(z_grid))
        p_avg_sig_M = np.nan * np.zeros(np.size(z_grid))
        p_avg_sig_W = np.nan * np.zeros(np.size(z_grid))
        p_avg_ct_M = np.nan * np.zeros(np.size(z_grid))
        p_avg_ct_W = np.nan * np.zeros(np.size(z_grid))
        p_avg_sa_M = np.nan * np.zeros(np.size(z_grid))
        p_avg_sa_W = np.nan * np.zeros(np.size(z_grid))
        yy_M = np.nan * np.zeros(np.size(z_grid))
        yy_W = np.nan * np.zeros(np.size(z_grid))
        # LOOP OVER EACH BIN_DEPTH
        for j in range(np.size(z_grid)):
            # find array of indices for M / W sampling
            if i < 2:
                if partial_mw:
                    c_i_m = np.arange(i, i + 3)
                else:
                    c_i_m = []  # omit partial "M" estimate
                c_i_w = np.arange(i, i + 4)
            elif (i >= 2) and (i < num_profs - 2):
                c_i_m = np.arange(i - 1, i + 3)
                c_i_w = np.arange(i, i + 4)
            elif i >= num_profs - 2:
                if partial_mw:
                    c_i_m = np.arange(i - 1, num_profs)
                else:
                    c_i_m = []  # omit partial "M" estimated
                c_i_w = []
            nm = np.size(c_i_m)
            nw = np.size(c_i_w)

            # for M profile compute shear and eta
            if nm > 2 and np.size(dg_sig0[j, c_i_m]) > 2:
                sigmathetaM = dg_sig0[j, c_i_m]
                sigma_theta_pa_M[j] = np.nanmean(sigmathetaM)  # average density across 4 profiles
                yy_M[j] = np.nanmean(dg_y[j, c_i_m])
                imv = ~np.isnan(np.array(dg_sig0[j, c_i_m]))
                c_i_m_in = c_i_m[imv]

                if np.size(c_i_m_in) > 1:
                    d_anom0M = sigmathetaM[imv] - np.nanmean(sigmathetaM[imv])
                    # drhodatM = np.nanmean(np.gradient(sigmathetaM[imv], dg_y[j, c_i_m]))

                    # local density referenced to the current pressure
                    local_sig = gsw.rho(dg_sa[j, c_i_m], dg_ct[j, c_i_m], p_grid[j]) - 1000

                    # drhodatM = np.polyfit(dg_y[j, c_i_m], local_sig, 1)[0]
                    drhodatM = np.polyfit(dg_y[j, c_i_m], dg_sig0[j, c_i_m], 1)[0]
                    shearM[j] = -g * drhodatM / (rho0 * ff)  # shear to port of track [m/s/km]

                    p_avg_sig_M[j] = np.nanmean(dg_sig0[j, c_i_m])
                    p_avg_sa_M[j] = np.nanmean(dg_sa[j, c_i_m])
                    p_avg_ct_M[j] = np.nanmean(dg_ct[j, c_i_m])

            # for W profile compute shear and eta
            if nw > 2 and np.size(dg_sig0[j, c_i_w]) > 2:
                sigmathetaW = dg_sig0[j, c_i_w]
                sigma_theta_pa_W[j] = np.nanmean(sigmathetaW)  # average density across 4 profiles
                yy_W[j] = np.nanmean(dg_y[j, c_i_w])
                iwv = ~np.isnan(np.array(dg_sig0[j, c_i_w]))
                c_i_w_in = c_i_w[iwv]

                if np.size(c_i_w_in) > 1:
                    d_anom0W = sigmathetaW[iwv] - np.nanmean(sigmathetaW[iwv])
                    # drhodatW = np.nanmean(np.gradient(sigmathetaW[iwv], dg_y[j, c_i_w]))

                    # local density referenced to the current pressure
                    local_sig = gsw.rho(dg_sa[j, c_i_w], dg_ct[j, c_i_w], p_grid[j]) - 1000

                    # drhodatW = np.polyfit(dg_y[j, c_i_w], local_sig, 1)[0]
                    drhodatW = np.polyfit(dg_y[j, c_i_w], dg_sig0[j, c_i_w], 1)[0]
                    shearW[j] = -g * drhodatW / (rho0 * ff)  # shear to port of track [m/s/km]

                    p_avg_sig_W[j] = np.nanmean(dg_sig0[j, c_i_w])
                    p_avg_sa_W[j] = np.nanmean(dg_sa[j, c_i_w])
                    p_avg_ct_W[j] = np.nanmean(dg_ct[j, c_i_w])

        # OUTPUT FOR EACH TRANSECT (at least 2 DIVES)
        # because this is M/W profiling, for a 3 dive transect, only 5 profiles of shear and eta are compiled
        sigma_theta_out[:, i] = sigma_theta_pa_M
        shear[:, i] = shearM
        avg_sig_pd[:, i] = p_avg_sig_M
        avg_sa_pd[:, i] = p_avg_sa_M
        avg_ct_pd[:, i] = p_avg_ct_M
        iq = np.where(~np.isnan(shearM))[0]
        if len(iq) > 0:
            mw_y[i] = dg_y[iq[-1], i]

        if i < num_profs - 2:
            sigma_theta_out[:, i + 1] = sigma_theta_pa_W
            shear[:, i + 1] = shearW
            avg_sig_pd[:, i + 1] = p_avg_sig_W
            avg_sa_pd[:, i + 1] = p_avg_sa_W
            avg_ct_pd[:, i + 1] = p_avg_ct_W
            mw_y[i + 1] = dg_y[0, i + 1]

        # ISOPYCNAL DEPTHS ON PROFILES ALONG EACH TRANSECT
        sigthmin = np.nanmin(np.array(dg_sig0[:, i]))
        sigthmax = np.nanmax(np.array(dg_sig0[:, i]))
        isigth = (sigth_levels > sigthmin) & (sigth_levels < sigthmax)
        isopycdep[isigth, i] = np.interp(sigth_levels[isigth], dg_sig0[:, i], z_grid)
        isopycx[isigth, i] = np.interp(sigth_levels[isigth], dg_sig0[:, i], dg_y[:, i])

        sigthmin = np.nanmin(np.array(dg_sig0[:, i + 1]))
        sigthmax = np.nanmax(np.array(dg_sig0[:, i + 1]))
        isigth = (sigth_levels > sigthmin) & (sigth_levels < sigthmax)
        isopycdep[isigth, i + 1] = np.interp(sigth_levels[isigth], dg_sig0[:, i + 1], z_grid)
        isopycx[isigth, i + 1] = np.interp(sigth_levels[isigth], dg_sig0[:, i + 1], dg_y[:, i + 1])

    # interpolate dac to be centered on m/w locations
    if E_W:
        dg_dac_use = 1. * nanseg_interp(dg_dac_mid, dg_dac) + dac_error * np.random.rand(len(dg_dac))  # added noise to DAC
    else:
        dg_dac_use = -1. * nanseg_interp(dg_dac_mid, dg_dac) + dac_error * np.random.rand(len(dg_dac))  # added noise to DAC
    # FOR EACH TRANSECT COMPUTE GEOSTROPHIC VELOCITY
    vbc_g = np.nan * np.zeros(np.shape(shear))
    v_g = np.nan * np.zeros((np.size(z_grid), num_profs))
    for m in range(num_profs - 1):
        iq = np.where(~np.isnan(shear[:, m]))[0]
        if np.size(iq) > 10:
            z2 = dg_z[iq]
            vrel = cumtrapz(shear[iq, m], x=z2, initial=0)
            # vrel_av = np.trapz(vrel / (z2[-1] - 0), x=z2)
            H = (z2[-1] - 0)
            vrel_av = np.trapz(vrel, x=z2) * (1.0 / H)
            vbc = vrel - vrel_av
            vbc_g[iq, m] = vbc
            v_g[iq, m] = dg_dac_use[m] + vbc
        else:
            vbc_g[iq, m] = np.nan
            v_g[iq, m] = np.nan
    if E_W:
        v_g = 1. * v_g[:, 0:-1]
    else:
        v_g = -1. * v_g[:, 0:-1]
    print('Completed M/W Vel Estimation')
    # ---------------------------------------------------------------------------------------------------------------------
    # remove nan profiles that are the result of deciding to use partial m/w profiles or not
    v_g_0 = v_g.copy()
    v_g = v_g[:, ~np.isnan(v_g_0[30, :])]
    goodie = np.where(~np.isnan(v_g_0[30, :]))[0]
    u_mod_at_mw = u_mod_at_mw[goodie[0]:goodie[-1]+1]
    gamma_mod_at_mw = gamma_mod_at_mw[goodie[0]:goodie[-1]+1]
    avg_sig_pd = avg_sig_pd[:, ~np.isnan(avg_sig_pd[30, :])]
    num_profs_eta = np.shape(avg_sig_pd)[1]
    shear = shear[:, ~np.isnan(shear[30, :])]
    # --------------------------------------------------------------------------------------------------------------------
    # -- DENSITY AT FIXED DEPTHS (this is how we estimate density gradients (drho/dx)
    # --- depth of isopycnals in time and  glider aliasing
    dive_1_s = dg_y[0, 0]
    dive_1_e = dg_y[0, 1]
    dive_1_t_s = dg_t[0, 0]
    dive_1_t_e = dg_t[0, 1]
    dive_1_span = dive_1_e - dive_1_s
    dive_1_t_span = dive_1_t_e - dive_1_t_s
    xy_grid_near = np.where(xy_grid <= dive_1_s)[0][-1]
    xy_grid_near_end = np.where(xy_grid >= dive_1_e)[0][0]
    t_near = np.where(time_ord_s <= dive_1_t_s)[0][-1]
    t_near_end = np.where(time_ord_s >= dive_1_t_e)[0][0]

    isops = [26.8, 27.4, 27.8, 27.9]
    t_steps = [t_near, t_near + 6, t_near + 12, t_near + 18, t_near + 24, t_near + 30]

    deps = [250, 1000, 1500, 2000]
    t_step_m = np.int(np.floor(24 * (dg_t[0, 3] - dg_t[0, 0])))
    t_steps = range(t_near, t_near + t_step_m, 2)
    isop_dep = np.nan * np.ones((len(deps), len(xy_grid), len(t_steps)))
    for i in range(len(xy_grid)):  # loop over each horizontal grid point
        for j in range(len(deps)):  # loop over each dep
            for k in range(len(t_steps)):  # loop over each 6 hr increment
                isop_dep[j, i, k] = np.interp(deps[j], -1. * dg_z, data_interp[t_steps[k], :, i])

    # -- glider measured isopycnal depth (over time span)
    dg_isop_dep = np.nan * np.ones((len(deps), 4))
    dg_isop_xy = np.nan * np.ones((len(deps), 4))
    for j in range(len(deps)):
        for k in range(4):  # loop over dive, climb, dive, climb
            dg_isop_dep[j, k] = np.interp(deps[j], -1. * dg_z, dg_sig0[:, k])
            dg_isop_xy[j, k] = np.interp(deps[j], -1. * dg_z, dg_y[:, k])

    # xy within glider dives (model xy over which to estimate density gradients)
    inn_per_mw = []
    inn_per_mw_t = []
    mw_ind_s = np.arange(0, np.shape(shear)[1])
    mw_ind_e = np.arange(3, 3 + np.shape(shear)[1])
    dg_t_hrs = 24. * (dg_t - dg_t[0, 0])
    model_t_hrs = 24. * (time_ord_s - time_ord_s[0])
    for i in range(np.shape(shear)[1]):
        if np.mod(i, 2):  # M
            inn_per_mw.append(np.where((xy_grid >= np.nanmin(dg_y[:, mw_ind_s[i]])) &
                                       (xy_grid <= np.nanmax(dg_y[:, mw_ind_e[i]])))[0])
            inn_per_mw_t.append(np.where((model_t_hrs > np.nanmin(dg_t_hrs[:, mw_ind_s[i]])) &
                                         (model_t_hrs <= np.nanmax(dg_t_hrs[:, mw_ind_e[i]])))[0])
        else:  # W
            inn_per_mw.append(np.where((xy_grid >= dg_y[0, mw_ind_s[i]]) & (xy_grid <= dg_y[0, mw_ind_e[i]]))[0])
            inn_per_mw_t.append(np.where((model_t_hrs >= dg_t_hrs[0, mw_ind_s[i]]) &
                                         (model_t_hrs <= dg_t_hrs[0, mw_ind_e[i]]))[0])

    # vertical shear difference between model and glider at all depths
    model_isop_slope_all = np.nan * np.ones((len(dg_z), 2, np.shape(shear)[1]))
    den_var = np.nan * np.ones((len(dg_z), np.shape(shear)[1]))
    igw_var = np.nan * np.ones((len(dg_z), np.shape(shear)[1]))
    ed_var = np.nan * np.ones((len(dg_z), np.shape(shear)[1]))
    # model_mean_isop_all = np.nan * np.ones((len(dg_z), len(xy_grid[inn]), np.shape(shear)[1]))
    for i in range(np.shape(shear)[1]):  # loop over each horizontal profile location
        for j in range(len(dg_z)):  # loop over each depth
            if np.sum(np.isnan(np.nanmean(data_interp[:, j, inn_per_mw[i]], axis=0))) < 1:
                # gradient of density at depth j over model grid cells that span m_w limits
                model_isop_slope_all[j, :, i] = np.polyfit(xy_grid[inn_per_mw[i]],
                                                           np.nanmean(data_interp[inn_per_mw_t[i][0]:inn_per_mw_t[i][-1] + 1,
                                                                      j, inn_per_mw[i][0]:inn_per_mw[i][-1]+1], axis=0), 1)
                # create linear gradient
                # model_mean_isop_all = np.polyval(model_isop_slope_all[j, :, i], xy_grid[inn_per_mw[i]])

                # consider density at depth j over model grid cells that span fixed xy limits and 24hrs and not m_w time
                dive_mid_t_hr = 24. * np.nanmean(time_ord_s[inn_per_mw_t[i]])
                dive_min_t_hr = 24. * time_ord_s[inn_per_mw_t[i][0]]
                dive_max_t_hr = 24. * time_ord_s[inn_per_mw_t[i][-1]]
                dive_mid_xy = np.nanmean(xy_grid[inn_per_mw[i][0]:inn_per_mw[i][-1]+1])
                xy_low = np.where(xy_grid >= (dive_mid_xy - 15000))[0][0]
                xy_up = np.where(xy_grid <= (dive_mid_xy + 15000))[0][-1] + 1
                if dive_mid_t_hr <= 24:
                    t_to_add = 36 - (dive_mid_t_hr - dive_min_t_hr)
                    t_up_win = np.where((24. * time_ord_s) <= (dive_mid_t_hr + t_to_add))[0][-1]
                    model_isop_slope_all[j, :, i] = np.polyfit(xy_grid[xy_low:xy_up],
                                                               np.nanmean(data_interp[0:t_up_win + 1,
                                                                          j, xy_low:xy_up], axis=0), 1)
                    den_in = data_interp[0:t_up_win + 1, j, xy_low:xy_up]
                elif dive_mid_t_hr >= (np.nanmax(24. * time_ord_s) - 36):
                    t_to_sub = 36 - (dive_max_t_hr - dive_mid_t_hr)
                    t_bot_win = np.where((24. * time_ord_s) >= (dive_mid_t_hr - t_to_sub))[0][0]
                    model_isop_slope_all[j, :, i] = np.polyfit(xy_grid[xy_low:xy_up],
                                                               np.nanmean(data_interp[t_bot_win:,
                                                                          j, xy_low:xy_up], axis=0), 1)
                    den_in = data_interp[t_bot_win:, j, xy_low:xy_up]
                else:
                    midd = np.int(np.round(np.nanmean(inn_per_mw_t[i])))
                    midd_mi = midd - 18
                    midd_pl = midd + 18
                    model_isop_slope_all[j, :, i] = np.polyfit(xy_grid[xy_low:xy_up],
                                                               np.nanmean(data_interp[midd_mi:midd_pl,
                                                                          j, xy_low:xy_up], axis=0), 1)
                    den_in = data_interp[midd_mi:midd_pl, j, xy_low:xy_up]


                # try 2 at linear gradient of model density
                model_mean_isop_all = np.polyval(model_isop_slope_all[j, :, i], xy_grid[xy_low:xy_up])

                # # check if doing right
                # if i > 0 & i < 3:
                #     if j == 100:
                #         print(str(i))
                #         print(str(j))
                #         f, ax = plt.subplots()
                #         ax.plot(xy_grid[xy_low:xy_up], np.nanmean(den_in, axis=0), color='k')
                #         for kk in range(np.shape(den_in)[0]):
                #             ax.plot(xy_grid[xy_low:xy_up], den_in[kk, :], linewidth=0.5)
                #         plot_pro(ax)

                # variance of the linear fit
                # signal_var = np.nanvar(model_mean_isop_all)
                signal_var = (1/len(model_mean_isop_all))*np.nansum((model_mean_isop_all - np.nanmean(model_mean_isop_all))**2)
                #
                # vary = np.nan * np.ones(len(xy_grid))
                for l in range(len(xy_grid[xy_low:xy_up])):
                    # variance in time of iso about model mean
                    # vary[l] = (1/len(den_in[:, l])) * np.nansum((den_in[:, l] - model_mean_isop_all[l])**2)
                    # square difference over all times at each grid point spanning each m/w pattern
                    vary = (den_in[:, l] - model_mean_isop_all[l])**2
                    if l < 1:
                        vary_out = vary.copy()
                    else:
                        vary_out = np.concatenate((vary_out, vary))
                vary_tot = (1/len(vary_out)) * np.nansum(vary_out)
                igw_var[j, i] = vary_tot.copy()
                ed_var[j, i] = signal_var.copy()
                den_var[j, i] = vary_tot/signal_var  # np.nanmean(vary)/signal_var
                # note: igw variance is a function of glider speed because I consider variance of isopycnals about mean over
                # the time it takes for the glider to complete an m/w
                # faster w's will yield a better ratio because isopycnal variance will be lower because less time has
                # elapsed for heaving to occur

    # difference in shear between glider and model
    shear_error = np.abs(100. * (shear - (-g * model_isop_slope_all[:, 0, :]/(rho0 * ff))) /
                         (-g * model_isop_slope_all[:, 0, :]/(rho0 * ff)))
    # --------------------------------------------------------------------------------------------------------------------
    # --- Background density
    overlap = np.where((xy_grid > dg_y[0, 0]) & (xy_grid < dg_y[0, -1]))[0]
    bck_gamma = np.nanmean(np.nanmean(sig0_out_s, axis=0)[:, overlap], axis=1)
    bck_sa = np.nanmean(np.nanmean(sa_out_s, axis=0)[:, overlap], axis=1)
    bck_ct = np.nanmean(np.nanmean(ct_out_s, axis=0)[:, overlap], axis=1)
    sig0_bck_out = bck_gamma
    sig0_bck_out = sig0_bck_out[~np.isnan(sig0_bck_out)]
    N2_bck_out = gsw.Nsquared(bck_sa, bck_ct, p_grid, lat=ref_lat)[0]
    for i in range(len(N2_bck_out)-20, len(N2_bck_out)):
        N2_bck_out[i] = N2_bck_out[i - 1] - 1*10**(-8)
    max_ind = np.where(np.isnan(bck_gamma))[0][0]
    z_back = dg_z[0:max_ind]
    N2_bck_out = N2_bck_out[0:max_ind]
    # ---------------------------------------------------------------------------------------------------------------------
    # testing of variance of igw to eddy signals
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    # for i in range(np.shape(shear)[1]):
    #     ax1.plot(igw_var[:, i], z_grid, linewidth=0.75)
    #     ax2.plot(ed_var[:, i], z_grid, linewidth=0.75)
    #     ax3.plot((igw_var[0:230, i]/ed_var[0:230, i]), z_grid[0:230], linewidth=0.75)
    # ax1.set_xlim([0, 0.004])
    # ax2.set_xlim([0, 0.004])
    # ax3.set_xlim([0, 0.0001])
    # ax1.grid()
    # ax2.grid()
    # plot_pro(ax3)
    # --------------------------------------------------------------------------------------------------------------
    # --- MODE PARAMETERS
    # frequency zeroed for geostrophic modes
    omega = 0
    # highest baroclinic mode to be calculated
    mmax = 35
    nmodes = mmax + 1
    # maximum allowed deep shear [m/s/km]
    deep_shr_max = 0.1
    # minimum depth for which shear is limited [m]
    deep_shr_max_dep = 2000.0
    # fit limits
    eta_fit_dep_min = 200.0
    eta_fit_dep_max = 3750.0

    # adjust z_grid and N2 profile such that N2=0 at surface and that z_grid min = -2500
    # match background z_grid and N2 to current data
    z_grid_inter = z_back[0:-1] + (z_back[1:] - z_back[0:-1])/2
    z_grid_n2 = np.concatenate((np.array([0]), z_back))
    avg_N2 = np.concatenate((np.array([0]), N2_bck_out))
    avg_N2_0 = avg_N2.copy()
    for i in range(len(avg_N2) - 25, len(avg_N2)):
        avg_N2[i] = avg_N2[i - 1] - 1 * 10 ** -9
    avg_sig0 = np.concatenate((np.array([sig0_bck_out[0] - 0.01]), sig0_bck_out))
    # avg_sig0 = sig0_bck_out

    # need ddz profile to compute eta
    ddz_avg_sig0 = np.nan * np.ones(np.shape(avg_sig0))
    ddz_avg_sig0[0] = (avg_sig0[1] - avg_sig0[0])/(z_grid_n2[1] - z_grid_n2[0])
    ddz_avg_sig0[-1] = (avg_sig0[-1] - avg_sig0[-2])/(z_grid_n2[-1] - z_grid_n2[-2])
    ddz_avg_sig0[1:-1] = (avg_sig0[2:] - avg_sig0[0:-2]) / (z_grid_n2[2:] - z_grid_n2[0:-2])

    # # --- compute vertical mode shapes
    G, Gz, c, epsilon = vertical_modes(avg_N2, -1.0 * z_grid_n2, omega, mmax)  # N2
    print('Computed Vertical Modes from Background Profiles')
    # ---------------------------------------------------------------------------------------------------------------------
    # -- DG eta from individual profiles
    eta = np.nan * np.ones(np.shape(dg_sig0))
    AG_dg = np.zeros((nmodes, num_profs))
    PE_per_mass_dg = np.nan * np.ones((nmodes, num_profs))
    eta_m_dg = np.nan * np.ones((len(dg_z), num_profs))
    Neta_m_dg = np.nan * np.ones((len(dg_z), num_profs))
    for i in range(num_profs):
        good = np.where(~np.isnan(dg_sig0[:, i]))[0]

        overlap = np.where((xy_grid > np.nanmin(dg_y[:, i])) & (xy_grid < np.nanmax(dg_y[:, i])))[0]
        bck_gamma_xy = np.nanmean(np.nanmean(sig0_out_s, axis=0)[:, overlap], axis=1)

        avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid), bck_gamma_xy)  # np.abs(z_grid_n2), avg_sig0
        ddz_avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), ddz_avg_sig0)
        avg_N2_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_N2)

        eta[good, i] = (dg_sig0[good, i] - avg_sig0_match) / ddz_avg_sig0_match

        grid = -1. * dg_z[good]
        bvf = np.sqrt(avg_N2_match)
        this_eta = eta[good, i].copy()
        eta_fs = eta[good, i].copy()
        iw = np.where((grid >= eta_fit_dep_min) & (grid <= eta_fit_dep_max))
        if iw[0].size > 1:
            G, Gz, c, epsilon = vertical_modes(avg_N2_match, -1.0 * dg_z[good], omega, mmax)  # N2

            i_sh = np.where((grid < eta_fit_dep_min))
            eta_fs[i_sh[0]] = grid[i_sh] * this_eta[iw[0][0]] / grid[iw[0][0]]

            i_dp = np.where((grid > eta_fit_dep_max))
            eta_fs[i_dp[0]] = (grid[i_dp] - grid[-1]) * this_eta[iw[0][-1]] / (grid[iw[0][-1]] - grid[-1])

            AG_dg[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], eta_fs)[0])
            # AG[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], np.transpose(np.atleast_2d(eta_fs)))[0])
            eta_m_dg[good, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG_dg[:, i])))
            Neta_m_dg[good, i] = bvf * np.array(np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG_dg[:, i]))))
            PE_per_mass_dg[:, i] = (0.5) * AG_dg[:, i] * AG_dg[:, i] * c * c

    G, Gz, c, epsilon = vertical_modes(avg_N2, -1.0 * z_grid_n2, omega, mmax)  # N2
    # ---------------------------------------------------------------------------------------------------------------------
    # -- DG eta avg across four profiles
    eta_sm_old = np.nan * np.ones(np.shape(avg_sig_pd))
    AG_dg_sm = np.zeros((nmodes, num_profs_eta))
    PE_per_mass_dg_sm = np.nan * np.ones((nmodes, num_profs_eta))
    eta_m_dg_sm = np.nan * np.ones((len(dg_z), num_profs_eta))
    Neta_m_dg_sm = np.nan * np.ones((len(dg_z), num_profs_eta))
    eta_sm = np.nan * np.ones(np.shape(avg_sig_pd))
    for i in range(np.shape(avg_sig_pd)[1]):
        good = np.where(~np.isnan(avg_sig_pd[:, i]))[0]

        overlap = np.where((xy_grid > (dg_dac_mid[i + 1] - 10000)) & (xy_grid < (dg_dac_mid[i + 1] + 10000)))[0]
        bck_gamma_xy = np.nanmean(np.nanmean(sig0_out_s, axis=0)[:, overlap], axis=1)
        avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid), bck_gamma_xy)  # np.abs(z_grid_n2), avg_sig0
        ddz_avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), ddz_avg_sig0)
        avg_N2_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_N2)

        ddz_avg_sig0_match_2 = savgol_filter(ddz_avg_sig0_match, 11, 2)

        # ETA ALT 3 (direct search )
        this_z = dg_z[good]
        avg_sig_pd_cut = avg_sig_pd[good, i]
        for j in range(len(this_z)):
            # find this profile density at j along avg profile
            idx, rho_idx = find_nearest(avg_sig0_match, avg_sig_pd_cut[j])
            if idx <= 2:
                z_rho_1 = this_z[0:idx + 3]
                eta_sm[j, i] = np.interp(avg_sig_pd_cut[j], avg_sig0_match[0:idx + 3], z_rho_1) - this_z[j]
            else:
                z_rho_1 = this_z[idx - 2:idx + 3]
                eta_sm[j, i] = np.interp(avg_sig_pd_cut[j], avg_sig0_match[idx - 2:idx + 3], z_rho_1) - \
                                  this_z[j]

        eta_sm_old[good, i] = (avg_sig_pd[good, i] - avg_sig0_match) / ddz_avg_sig0_match_2

        grid = -1. * dg_z[good]
        bvf = np.sqrt(avg_N2_match)
        this_eta = eta_sm[good, i].copy()
        eta_fs = eta_sm[good, i].copy()
        iw = np.where((grid >= eta_fit_dep_min) & (grid <= eta_fit_dep_max))
        if iw[0].size > 1:
            G, Gz, c, epsilon = vertical_modes(avg_N2_match, -1.0 * dg_z[good], omega, mmax)  # N2

            i_sh = np.where((grid < eta_fit_dep_min))
            eta_fs[i_sh[0]] = grid[i_sh] * this_eta[iw[0][0]] / grid[iw[0][0]]

            i_dp = np.where((grid > eta_fit_dep_max))
            eta_fs[i_dp[0]] = (grid[i_dp] - grid[-1]) * this_eta[iw[0][-1]] / (grid[iw[0][-1]] - grid[-1])

            AG_dg_sm[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], eta_fs)[0])
            # AG[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], np.transpose(np.atleast_2d(eta_fs)))[0])
            eta_m_dg_sm[good, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG_dg_sm[:, i])))
            Neta_m_dg_sm[good, i] = bvf * np.array(np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG_dg_sm[:, i]))))
            PE_per_mass_dg_sm[:, i] = (0.5) * AG_dg_sm[:, i] * AG_dg_sm[:, i] * c * c
    # ---------------------------------------------------------------------------------------------------------------------
    # -- eta Model (avg across m_w space)
    in_range = np.where((xy_grid > np.nanmin(dg_y)) & (xy_grid < np.nanmax(dg_y)))[0]
    eta_model = np.nan * np.ones(np.shape(sig0_out_s[:, in_range, 0]))

    eta_model = np.nan * np.ones(np.shape(eta_sm))
    AG_model = np.zeros((nmodes, np.shape(eta_model)[1]))
    PE_per_mass_model = np.nan * np.ones((nmodes, np.shape(eta_model)[1]))
    eta_m_model = np.nan * np.ones((len(dg_z), np.shape(eta_model)[1]))
    Neta_m_model = np.nan * np.ones((len(dg_z), np.shape(eta_model)[1]))
    for i in range(np.shape(eta_model)[1]):
        # range over which m/w profile takes up,
        # consider each model density profile, avg at each grid point in time, estimate eta
        # this_model = np.flipud(np.nanmean(sig0_out_s[:, in_range[i], :], axis=1))
        this_model = gamma_mod_at_mw_avg[:, i + 1]
        # + 1 because this variable includes first and last gamma that we exclude because it is a partial m_w
        good = np.where(~np.isnan(this_model))[0]

        # avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_sig0)
        # ddz_avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), ddz_avg_sig0)
        # avg_N2_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_N2)

        overlap = np.where((xy_grid > (dg_dac_mid[i + 1] - 10000)) & (xy_grid < (dg_dac_mid[i + 1] + 10000)))[0]
        bck_gamma_xy = np.nanmean(np.nanmean(sig0_out_s, axis=0)[:, overlap], axis=1)
        avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid), bck_gamma_xy)  # np.abs(z_grid_n2), avg_sig0
        ddz_avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), ddz_avg_sig0)
        avg_N2_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_N2)

        eta_model[good, i] = (this_model[good] - avg_sig0_match) / ddz_avg_sig0_match

        grid = -1. * dg_z[good]
        bvf = np.sqrt(avg_N2_match)
        this_eta = eta_model[good, i].copy()
        eta_fs = eta_model[good, i].copy()
        iw = np.where((grid >= eta_fit_dep_min) & (grid <= eta_fit_dep_max))
        if iw[0].size > 1:
            G, Gz, c, epsilon = vertical_modes(avg_N2_match, -1.0 * dg_z[good], omega, mmax)  # N2

            i_sh = np.where((grid < eta_fit_dep_min))
            eta_fs[i_sh[0]] = grid[i_sh] * this_eta[iw[0][0]] / grid[iw[0][0]]

            i_dp = np.where((grid > eta_fit_dep_max))
            eta_fs[i_dp[0]] = (grid[i_dp] - grid[-1]) * this_eta[iw[0][-1]] / (grid[iw[0][-1]] - grid[-1])

            AG_model[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], eta_fs)[0])
            # AG[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], np.transpose(np.atleast_2d(eta_fs)))[0])
            eta_m_model[good, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG_model[:, i])))
            Neta_m_model[good, i] = bvf * np.array(np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG_model[:, i]))))
            PE_per_mass_model[:, i] = (0.5) * AG_model[:, i] * AG_model[:, i] * c * c
    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # --- DG velocities ---
    HKE_per_mass_dg = np.nan * np.zeros([nmodes, num_profs_eta])
    modest = np.arange(11, nmodes)
    good_ke_prof = np.ones(num_profs_eta)
    AGz = np.zeros([nmodes, num_profs_eta])
    HKE_noise_threshold = 1e-5  # 1e-5
    V_m = np.nan * np.zeros((len(dg_z), num_profs_eta))
    for i in range(num_profs_eta):

        good = np.where(~np.isnan(v_g[:, i]))[0]

        if len(good) > 20:
            avg_N2_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_N2)
            G, Gz, c, epsilon = vertical_modes(avg_N2_match, -1.0 * dg_z[good], omega, mmax)  # N2

            # fit to velocity profiles
            this_V = v_g[good, i].copy()
            iv = np.where(~np.isnan(this_V))
            if iv[0].size > 1:
                AGz[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(Gz[iv, :]), np.transpose(np.atleast_2d(this_V[iv])))[0])
                # Gz(iv,:)\V_g(iv,ip)
                V_m[good, i] = np.squeeze(np.matrix(Gz) * np.transpose(np.matrix(AGz[:, i])))
                # Gz*AGz[:,i];
                HKE_per_mass_dg[:, i] = 0.5 * (AGz[:, i] * AGz[:, i])
                ival = np.where(HKE_per_mass_dg[modest, i] >= HKE_noise_threshold)
                if np.size(ival) > 0:
                    good_ke_prof[i] = 0  # flag profile as noisy
            else:
                good_ke_prof[i] = 0  # flag empty profile as noisy as well
    # ---------------------------------------------------------------------------------------------------------------------
    # Model
    # select avg model profiles that coincide with dg velocity profiles in space and time
    avg_mod_u = u_mod_at_mw_avg[:, 1:-1]
    HKE_per_mass_model = np.nan * np.zeros([nmodes, np.shape(avg_mod_u)[1]])
    good_ke_prof = np.ones(np.shape(avg_mod_u)[1])
    AGz_model = np.zeros([nmodes, np.shape(avg_mod_u)[1]])
    HKE_noise_threshold = 1e-5  # 1e-5
    V_m_model = np.nan * np.zeros((len(dg_z), np.shape(avg_mod_u)[1]))
    # avg_mod_u = np.flipud(np.nanmean(u_out_s[:, in_range, :], axis=2))
    for i in range(np.shape(avg_mod_u)[1]):
        good = np.where(~np.isnan(avg_mod_u[:, i]))[0]
        avg_N2_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_N2)

        G, Gz, c, epsilon = vertical_modes(avg_N2_match, -1.0 * dg_z[good], omega, mmax)  # N2

        # fit to velocity profiles
        this_V = avg_mod_u[good, i]
        iv = np.where(~np.isnan(this_V))
        if iv[0].size > 1:
            AGz_model[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(Gz[iv, :]), np.transpose(np.atleast_2d(this_V[iv])))[0])
            # Gz(iv,:)\V_g(iv,ip)
            V_m_model[good, i] = np.squeeze(np.matrix(Gz) * np.transpose(np.matrix(AGz_model[:, i])))
            # Gz*AGz[:,i];
            HKE_per_mass_model[:, i] = 0.5 * (AGz_model[:, i] * AGz_model[:, i])
            ival = np.where(HKE_per_mass_model[modest, i] >= HKE_noise_threshold)
            if np.size(ival) > 0:
                good_ke_prof[i] = 0  # flag profile as noisy
        else:
            good_ke_prof[i] = 0  # flag empty profile as noisy as well
    # ----------------------------------------------------------------------------------------------------------------------
    ff = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)  # Coriolis parameter [s^-1]
    omega = 0
    mm = mmax
    sc_x = 1000 * ff / c[1:mm]
    l_lim = 3 * 10 ** -2
    sc_x = np.arange(1, mm)
    dk = ff / c[1]
    avg_KE_model = np.nanmean(HKE_per_mass_model, axis=1)
    avg_PE_model = np.nanmean(PE_per_mass_model, axis=1)
    # ----------------------------------------------------------------------------------------------------------------------
    # Model (instant)
    if u_mod_all > 0:
        for m in range(len(u_mod_at_mw)):
            this_mod_u = u_mod_at_mw[m]
            this_mod_gamma = gamma_mod_at_mw[m]
            for mt in range(np.shape(this_mod_u)[0]):  # loop in time
                # PE
                overlap = np.where((xy_grid > (dg_dac_mid[m + 1] - 10000)) & (xy_grid < (dg_dac_mid[m + 1] + 10000)))[0]
                bck_gamma_xy = np.nanmean(np.nanmean(sig0_out_s, axis=0)[:, overlap], axis=1)

                good = np.where(~np.isnan(bck_gamma_xy))[0]
                avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid), bck_gamma_xy)
                ddz_avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), ddz_avg_sig0)
                avg_N2_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_N2)

                this_model = this_mod_gamma[mt, good, :]
                eta_model = (this_model - np.tile(bck_gamma_xy[good, None],
                                                  (1, np.shape(this_model)[1]))) / np.tile(ddz_avg_sig0_match[:, None],
                                                                                           (1, np.shape(this_model)[1]))

                # KE
                this_mod_u_sp = this_mod_u[mt, good, :]
                HKE_per_mass_model_tot = np.nan * np.zeros([nmodes, np.shape(this_mod_u_sp)[1]])
                good_ke_prof = np.ones(np.shape(this_mod_u_sp)[1])
                AGz_model = np.zeros([nmodes, np.shape(this_mod_u_sp)[1]])
                HKE_noise_threshold = 1e-5  # 1e-5
                AG_model = np.zeros((nmodes, np.shape(eta_model)[1]))
                PE_per_mass_model_tot = np.nan * np.ones((nmodes, np.shape(eta_model)[1]))
                V_m_model_tot = np.nan * np.zeros((len(dg_z), np.shape(this_mod_u_sp)[1]))
                eta_m_model_tot = np.nan * np.ones((len(dg_z), np.shape(eta_model)[1]))

                for i in range(np.shape(this_mod_u_sp)[1]):
                    good = np.where(~np.isnan(this_mod_u_sp[:, i]))[0]
                    avg_N2_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_N2)
                    G, Gz, c, epsilon = vertical_modes(avg_N2_match, -1.0 * dg_z[good], omega, mmax)  # N2

                    grid = -1. * dg_z[good]
                    this_eta = eta_model[good, i].copy()
                    eta_fs = eta_model[good, i].copy()

                    # fit to eta profiles
                    iw = np.where((grid >= eta_fit_dep_min) & (grid <= eta_fit_dep_max))
                    if iw[0].size > 1:
                        i_sh = np.where((grid < eta_fit_dep_min))
                        eta_fs[i_sh[0]] = grid[i_sh] * this_eta[iw[0][0]] / grid[iw[0][0]]

                        i_dp = np.where((grid > eta_fit_dep_max))
                        eta_fs[i_dp[0]] = (grid[i_dp] - grid[-1]) * this_eta[iw[0][-1]] / (grid[iw[0][-1]] - grid[-1])

                        AG_model[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], eta_fs)[0])
                        eta_m_model_tot[good, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG_model[:, i])))
                        PE_per_mass_model_tot[:, i] = (0.5) * AG_model[:, i] * AG_model[:, i] * c * c

                    # fit to velocity profiles
                    this_V = this_mod_u_sp[good, i]
                    iv = np.where(~np.isnan(this_V))
                    if iv[0].size > 1:
                        AGz_model[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(Gz[iv, :]), np.transpose(np.atleast_2d(this_V[iv])))[0])
                        # Gz(iv,:)\V_g(iv,ip)
                        V_m_model_tot[good, i] = np.squeeze(np.matrix(Gz) * np.transpose(np.matrix(AGz_model[:, i])))
                        # Gz*AGz[:,i];
                        HKE_per_mass_model_tot[:, i] = 0.5 * (AGz_model[:, i] * AGz_model[:, i])
                        ival = np.where(HKE_per_mass_model_tot[modest, i] >= HKE_noise_threshold)
                        if np.size(ival) > 0:
                            good_ke_prof[i] = 0  # flag profile as noisy
                    else:
                        good_ke_prof[i] = 0  # flag empty profile as noisy as well
                print(str(mt))
                if (m < 1) & (mt < 1):
                    HKE_mod_TOT = HKE_per_mass_model_tot.copy()
                    PE_mod_TOT = PE_per_mass_model_tot.copy()
                else:
                    HKE_mod_TOT = np.concatenate((HKE_mod_TOT, HKE_per_mass_model_tot), axis=1)
                    PE_mod_TOT = np.concatenate((PE_mod_TOT, PE_per_mass_model_tot), axis=1)

    print('fraction completed = ' + str(master / np.float64(np.shape(params)[0])))
    # ----------------------------------------------------------------------------------------------------------------------
    # Save
    if save_anom:
        if u_mod_all:
            my_dict = {'dg_z': dg_z, 'dg_v': v_g, 'model_u_at_mwv': u_mod_at_mw,
                       'model_u_at_mw_avg': avg_mod_u, 'eta': eta_sm,
                       'shear_error': shear_error, 'igw_var': den_var,
                       'c': c,
                       'KE_dg': HKE_per_mass_dg, 'KE_mod': HKE_per_mass_model,
                       'eta_m_dg': eta_m_dg, 'PE_dg': PE_per_mass_dg,
                       'eta_m_dg_avg': eta_m_dg_sm, 'PE_dg_avg': PE_per_mass_dg_sm,
                       'eta_model': eta_m_model, 'PE_model': PE_per_mass_model,
                       'glide_slope': np.ones(np.shape(v_g)[1]) * dg_glide_slope,
                       'dg_w': np.ones(np.shape(v_g)[1]) * dg_vertical_speed,
                       'avg_N2': avg_N2, 'z_grid_n2': z_grid_n2,
                       'KE_mod_ALL': HKE_mod_TOT, 'PE_mod_ALL': PE_mod_TOT}
        else:
            my_dict = {'dg_z': dg_z, 'dg_v': v_g, 'model_u_at_mwv': u_mod_at_mw,
                       'model_u_at_mw_avg': avg_mod_u, 'eta': eta_sm,
                       'shear_error': shear_error, 'igw_var': den_var,
                       'c': c,
                       'KE_dg': HKE_per_mass_dg, 'KE_mod': HKE_per_mass_model,
                       'eta_m_dg': eta_m_dg, 'PE_dg': PE_per_mass_dg,
                       'eta_m_dg_avg': eta_m_dg_sm, 'PE_dg_avg': PE_per_mass_dg_sm,
                       'eta_model': eta_m_model, 'PE_model': PE_per_mass_model,
                       'glide_slope': np.ones(np.shape(v_g)[1]) * dg_glide_slope,
                       'dg_w': np.ones(np.shape(v_g)[1]) * dg_vertical_speed}
        output = open(output_filename, 'wb')
        pickle.dump(my_dict, output)
        output.close()

# END OF FUNCTION
# ----------------------------------------------------------------------------------------------------------------------

# PLOTTING
# -- plot cross section --
if plot0 > 0:

    h_max = np.round(np.nanmax(xy_grid), -3) / 1000 - 10  # np.nanmax(dg_y/1000 + 20)  # horizontal domain limit
    z_max = -5200
    # u_levels = np.array([-.8, -.7, -.6, -.5, -.4, -0.3, -.25, - .2, -.15, -.125, -.1, -.075, -.05, -0.025, 0,
    #                      0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, .7, .8])
    u_levels = np.array([-.5, -.4, -0.3, -.25, - .2, -.15, -.125, -.1, -.075, -.05, -0.025, 0,
                         0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])

    cmap = plt.cm.get_cmap("Spectral")

    dg_t_s = dg_t[0, 0]
    dg_t_e = dg_t[0, -1]
    taggt = str(np.int(filename[19])) + '/' + str(np.int(filename[20:22]) + np.floor(dg_t_s)) + ' ' + \
            str(np.int((dg_t[0, 0] - np.floor(dg_t_s)) * 24)) + 'hr' + \
            ' - ' + str(np.int(filename[19])) + '/' + str(np.int(filename[20:22]) + np.floor(dg_t_e)) + ' ' + \
            str(np.int((dg_t[0, -1] - np.floor(dg_t_e)) * 24)) + 'hr'


    t_in_low = np.where(time_ord_s >= np.nanmin(dg_t))[0][0]
    t_in_high = np.where(time_ord_s <= np.nanmax(dg_t))[0][-1]


    matplotlib.rcParams['figure.figsize'] = (13, 6)
    cmap = plt.cm.get_cmap("Spectral")
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    uvcf = ax1.contourf(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                 np.nanmean(u_out_s[t_in_low:t_in_high, :, :], axis=0), levels=u_levels, cmap=cmap)
    uvc = ax1.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                      np.nanmean(u_out_s[t_in_low:t_in_high, :, :], axis=0), levels=u_levels, colors='#2F4F4F', linewidths=0.5)
    ax1.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
    rhoc = ax1.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       np.nanmean(sig0_out_s, axis=0), levels=sigth_levels, colors='#A9A9A9', linewidths=0.5)
    ax1.scatter(dg_y/1000, dg_z_g, 4, color='k', label='glider path')

    t_tot = np.int(np.round(24.0*(dg_t[0,-1] - dg_t[0,0])))
    ax1.set_title('Model Avg. ' + taggt)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=11)
    ax1.set_ylim([z_max, 0])
    ax1.set_xlim([0, h_max])
    ax1.set_xlabel('E/W distance [km]')
    ax1.set_ylabel('z [m]')
    # plt.colorbar(uvcf, label='N/S Velocity [m/s]', ticks=u_levels)
    # ax1.grid()

    u_levels_i = np.array([-0.4, -.25, -.15, -.1, -.05, 0, 0.05, 0.1, 0.15, 0.25, 0.4])
    vh = ax2.contourf(np.tile(mw_y[1:-1]/1000, (len(z_grid), 1)), np.tile(dg_z[:, None], (1, len(mw_y[1:-1]))),
                 v_g, levels=u_levels, cmap=cmap)
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(vh, cax=cax, ticks=u_levels, label='N/S Velocity [m/s]', location='left')
    uvc = ax2.contour(np.tile(mw_y[1:-1]/1000, (len(z_grid), 1)), np.tile(dg_z[:, None], (1, len(mw_y[1:-1]))), v_g,
                      levels=u_levels, colors='#2F4F4F', linewidths=0.5)
    ax2.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')

    rhoc = ax2.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       np.nanmean(sig0_out_s[t_in_low:t_in_high, :, :], axis=0),
                       levels=sigth_levels, colors='#A9A9A9', linewidths=0.5)
    for r in range(np.shape(isopycdep)[0]):
        ax2.plot(isopycx[r, :]/1000, isopycdep[r, :], color='r', linewidth=0.75)
    ax2.plot(isopycx[r, :] / 1000, isopycdep[r, :], color='r', linewidth=0.5, label='glider measured isopycnals')
    ax2.scatter(dg_y/1000, dg_z_g, 4, color='k')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=11)
    ax2.set_xlim([0, h_max])
    ax2.set_title(r'Cross-Track Glider Vel. u$_g$(x,z)')
    ax2.set_xlabel('E/W distance [km]')
    if save_p > 0:
        fig.savefig('/Users/jake/Documents/glider_flight_sim_paper/hycom_ind_cross.png', dpi=200)
    else:
        ax2.grid()
        plot_pro(ax2)

# -- plot velocity --
if plot_v > 0:
    avg_mod_u = u_mod_at_mw_avg[:, 1:-1]
    matplotlib.rcParams['figure.figsize'] = (7, 7)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    for i in range(np.shape(v_g)[1]):
        ax1.plot(v_g[:, i], dg_z, color='#4682B4', linewidth=1.25, label='DG')
        ax1.plot(V_m[:, i], dg_z, color='k', linestyle='--', linewidth=.75)
        ax1.plot(avg_mod_u[0:-20, i], dg_z[0:-20], color='r', linewidth=0.5, label='Model')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[-2], handles[-1]], [labels[-2], labels[-1]], fontsize=10)
    ax1.set_xlim([-.4, .4])
    ax1.set_title('Avg. Model and DG Velocity (u_g), w = ' + str(dg_vertical_speed))
    ax1.set_xlabel('Velocity [m/s]')
    ax1.set_ylabel('z [m]')
    ax1.grid()

    for i in range(np.shape(eta_sm)[1]):
        ax2.plot(eta_sm[:, i], dg_z, color='#4682B4', linewidth=1.25, label='DG')
        ax2.plot(eta_m_dg_sm[:, i], dg_z, color='k', linestyle='--', linewidth=.75)
        ax2.plot(eta_model[0:-20, i], dg_z[0:-20], linewidth=0.5, color='r', label='Model')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend([handles[-2], handles[-1]], [labels[-2], labels[-1]], fontsize=10)
    ax2.set_xlabel('Isopycnal Displacement [m]')
    ax2.set_title('Vertical Displacement')
    ax2.set_xlim([-100, 100])

    if save_v > 0:
        ax2.grid()
        f.savefig('/Users/jake/Documents/glider_flight_sim_paper/vel_eta_y' +
                  str(np.int(y_dg_s/1000)) + '_v' + str(np.int(100*dg_vertical_speed)) +
                  '_slp' + str(np.int(dg_glide_slope)) + '_' + tag + '.png', dpi=200)
    else:
        plot_pro(ax2)

if plot_rho > 0:
    # density gradient computation for 4 depths for plot
    inn = np.where((xy_grid >= dg_y[0, 0]) & (xy_grid <= dg_y[0, 3]))[0]
    dg_isop_slope = np.nan * np.ones((len(deps), 2))
    dg_mean_isop = np.nan * np.ones((len(deps), len(dg_isop_xy[0, :])))
    model_isop_slope = np.nan * np.ones((len(deps), 2))
    model_mean_isop = np.nan * np.ones((len(deps), len(xy_grid[inn])))
    for i in range(4):
        dg_isop_slope[i, :] = np.polyfit(dg_isop_xy[i, :], dg_isop_dep[i, :], 1)
        dg_mean_isop[i, :] = np.polyval(dg_isop_slope[i, :], dg_isop_xy[i, :])

        model_isop_slope[i, :] = np.polyfit(xy_grid[inn], np.nanmean(isop_dep[i, inn, :], axis=1), 1)
        model_mean_isop[i, :] = np.polyval(model_isop_slope[i, :], xy_grid[inn])

    cmap = plt.cm.get_cmap("YlGnBu_r")
    # cmaps = [0.15, 0.22, 0.3, 0.37, 0.45, 0.52, 0.6, 0.67, 0.75, 0.82, .9, .98]
    cmaps = np.arange(0, 1, 1.0/len(t_steps))
    lab_y = [np.round(np.nanmin(isop_dep[0, :, 0]), 2), np.round(np.nanmin(isop_dep[1, :, 0]), 2),
             np.round(np.nanmin(isop_dep[2, :, 0]), 2), np.round(np.nanmin(isop_dep[3, :, 0]), 2)]
    matplotlib.rcParams['figure.figsize'] = (10, 7)
    f, ax = plt.subplots(4, 1, sharex=True)
    for i in range(len(deps)):
        ax[i].set_facecolor('#DCDCDC')
        for j in range(len(t_steps)):
            ax[i].plot(xy_grid / 1000, isop_dep[i, :, j], color=cmap(cmaps[j]))
        ax[i].plot(xy_grid / 1000, np.nanmean(isop_dep[i, :, :], axis=1), color='r', linestyle='--')
        ax[i].scatter(dg_isop_xy[i, :] / 1000, dg_isop_dep[i, :], s=40, color='k', zorder=10)

        ax[i].plot([dg_y[0, 0] / 1000, dg_y[0, 0] / 1000], [20, 30], linestyle='--', color='k')
        ax[i].plot([dg_y[0, 1] / 1000, dg_y[0, 1] / 1000], [20, 30], linestyle='--', color='k')
        ax[i].plot([dg_y[0, 3] / 1000, dg_y[0, 3] / 1000], [20, 30], linestyle='--', color='k')
        ax[i].text(10, lab_y[i], str(deps[i]) + 'm', fontweight='bold')
        ax[i].plot(xy_grid[inn] / 1000, model_mean_isop[i, :], color='#FF8C00', linewidth=2)
        ax[i].plot(dg_isop_xy[i, :] / 1000, dg_mean_isop[i, :], color='m', linewidth=2)
        ax[i].text(h_max - 45,
                   lab_y[i], r'du$_g$/dz error = ' + str(
                np.round(100. * np.abs((dg_isop_slope[i, 0] - model_isop_slope[i, 0]) / model_isop_slope[i, 0]),
                         0)) + '%', fontweight='bold')

    ax[i].plot(xy_grid / 1000, np.nanmean(isop_dep[i, :, :], axis=1), color='r', linestyle='--', linewidth=1.3,
               label='avg. density over 2 dive-cycle period')
    handles, labels = ax[3].get_legend_handles_labels()
    ax[3].legend(handles, labels, fontsize=9, loc='lower right')
    ax[0].set_ylabel('Neutral Density')
    ax[0].set_title(r'Density along z=z$_i$')
    ax[0].set_ylim([np.round(np.nanmin(isop_dep[0, :, 0]), 2) - 0.1, np.round(np.nanmax(isop_dep[0, :, 0]), 2) + 0.1])
    ax[0].invert_yaxis()
    ax[0].grid()
    ax[0].set_xlim([0, h_max])
    ax[1].set_ylabel('Neutral Density')
    ax[1].set_ylim([np.round(np.nanmin(isop_dep[1, :, 0]), 2) - 0.02, np.round(np.nanmax(isop_dep[1, :, 0]), 2) + 0.02])
    ax[1].invert_yaxis()
    ax[1].grid()
    ax[2].set_ylabel('Neutral Density')
    ax[2].set_ylim([np.round(np.nanmin(isop_dep[2, :, 0]), 2) - 0.01, np.round(np.nanmax(isop_dep[2, :, 0]), 2) + 0.01])
    ax[2].invert_yaxis()
    ax[2].grid()
    ax[3].set_ylabel('Neutral Density')
    ax[3].set_ylim([np.round(np.nanmin(isop_dep[3, :, 0]), 2) - 0.01, np.round(np.nanmax(isop_dep[3, :, 0]), 2) + 0.01])
    ax[3].invert_yaxis()
    ax[3].set_xlabel('Transect Distance [km]')

    if save_rho > 0:
        ax[3].grid()
        # f.savefig('/Users/jake/Documents/glider_flight_sim_paper/g_m_rho_y' +
        #           str(np.int(y_dg_s/1000)) + '_v' + str(np.int(100*dg_vertical_speed)) +
        #           '_slp' + str(np.int(dg_glide_slope)) + '_' + tag + '.png', dpi=200)
        f.savefig('/Users/jake/Documents/glider_flight_sim_paper/hycom_ind_den_grad.png', dpi=200)
    else:
        plot_pro(ax[3])

if plot_sp > 0:

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='glider_dives', artist='JS')
    writer = FFMpegWriter(fps=4, metadata=metadata)

    fig = plt.figure(figsize=(11, 8))

    with writer.saving(fig, "/Users/jake/Documents/baroclinic_modes/Model/HYCOM/glider_ts_movie/glider_dive_4.mp4", 150):
        for i in range(0, 6):  # range(np.shape(dg_sig0)[1]):
            max_ind = np.where(np.isnan(dg_y[:, i]))[0][0] - 1
            for j0 in range(0, max_ind, 8):  # range(1, np.shape(dg_sig0)[0]):
                if np.mod(i, 2):
                    j = max_ind - j0
                else:
                    j = j0
                this_t = dg_t[j, i]
                # find time bounding model runs
                nearest_model_t_over = np.where(time_ord_s > this_t)[0][0]

                this_u_before = u_out_s[nearest_model_t_over - 1, :, :]
                this_u_after = u_out_s[nearest_model_t_over, :, :]
                this_u = np.nan * np.ones(np.shape(this_u_before))
                this_sig_before = sig0_out_s[nearest_model_t_over - 1, :, :]
                this_sig_after = sig0_out_s[nearest_model_t_over, :, :]
                this_sig0 = np.nan * np.ones(np.shape(this_sig_before))
                for k in range(np.shape(this_u_before)[1]):
                    for l in range(np.shape(this_u_before)[0]):
                        this_u[l, k] = np.interp(this_t,
                                                 [time_ord_s[nearest_model_t_over - 1],
                                                  time_ord_s[nearest_model_t_over]],
                                                 [this_u_before[l, k], this_u_after[l, k]])
                        this_sig0[l, k] = np.interp(this_t,
                                                 [time_ord_s[nearest_model_t_over - 1],
                                                  time_ord_s[nearest_model_t_over]],
                                                 [this_sig_before[l, k], this_sig_after[l, k]])

                uvcf = plt.contourf(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                                    this_u, levels=u_levels, cmap=cmap)
                uvc = plt.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                                this_u, levels=u_levels, colors='#2F4F4F', linewidths=0.5)
                plt.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k', fontsize=8)
                rhoc = plt.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                                   this_sig0,
                                   levels=sigth_levels, colors='#A9A9A9', linewidths=0.5, label='Isopycnals')
                if i < 1:  # first dive
                    plt.scatter(dg_y[0:j+1, i]/1000, dg_z_g[0:j+1, i], 4, color='k', label='glider path')
                elif np.mod(i, 2):  # climb
                    plt.scatter(dg_y[:, 0:i] / 1000, dg_z_g[:, 0:i], 4, color='k')  # up to
                    plt.scatter(dg_y[j:, i] / 1000, dg_z_g[j:, i], 4, color='k', label='glider path')
                else:  # dive
                    plt.scatter(dg_y[:, 0:i] / 1000, dg_z_g[:, 0:i], 4, color='k')  # up to
                    plt.scatter(dg_y[0:j+1, i] / 1000, dg_z_g[0:j+1, i], 4, color='k', label='glider path')

                tagg = r'Slice at 30.86$^{\circ}$N, ' + str(np.int(filename[19])) + '/' + \
                       str(np.int(np.int(filename[20:22]) + np.floor(this_t))) + ' ' + \
                       str(nearest_model_t_over - (np.floor(this_t) * 24)) + 'hr'

                # if this_t < 1:
                #     tagg = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14])) + ' ' + str(
                #         nearest_model_t_over) + 'hr'
                # elif (this_t > 1) & (this_t < 2):
                #     tagg = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 1) + ' ' + str(
                #         nearest_model_t_over - 24) + 'hr'
                # elif (this_t > 2) & (this_t < 3):
                #     tagg = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 2) + ' ' + str(
                #         nearest_model_t_over - 48) + 'hr'
                # elif (this_t > 3) & (this_t < 4):
                #     tagg = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 3) + ' ' + str(
                #         nearest_model_t_over - 72) + 'hr'
                # elif (this_t > 4) & (this_t < 5):
                #     tagg = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 4) + ' ' + str(
                #         nearest_model_t_over - 96) + 'hr'

                ax1 = plt.gca()
                ax1.set_title(tagg)
                handles, labels = ax1.get_legend_handles_labels()
                ax1.legend(handles, labels, fontsize=11)
                ax1.set_ylim([z_max, 0])
                ax1.set_xlim([0, h_max])
                ax1.set_xlabel('E/W distance [km]')
                ax1.set_ylabel('z [m]')
                plt.colorbar(uvcf, label='N/S Velocity [m/s]', ticks=u_levels)

                writer.grab_frame()
                plt.clf()

            # f.savefig("/Users/jake/Documents/baroclinic_modes/Model/HYCOM/glider_ts_movie/fr_" +
            #           str(count) + ".png", dpi=200)
            # count = count + 1
import numpy as np
import matplotlib.animation as manimation
import glob
import datetime
from netCDF4 import Dataset
import gsw
import time as TT
import scipy.io as si
from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter
from mode_decompositions import eta_fit, vertical_modes, PE_Tide_GM, vertical_modes_f
# -- plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from toolkit import plot_pro, nanseg_interp
from zrfun import get_basic_info, get_z

# load mat file
filename = 'HYCOM_hrly_222_00_228_00'
import_hy = si.loadmat('/Users/jake/Documents/baroclinic_modes/Model/' + filename + '.mat')
MOD = import_hy['out']
ref_lat = 31
sig0_out_s = MOD['gamma'][0][0][:]
ct_out_s = MOD['temp'][0][0][:]  # temp
sa_out_s = MOD['salin'][0][0][:]  # salin
u_out_s = MOD['v'][0][0][:] # vel across transect
u_off_out_s = MOD['u'][0][0][:] # vel along transect
time_ord_s = np.arange(1, np.shape(sig0_out_s)[0]) / 24.0
z_grid = -1.0 * MOD['z_grid'][0][0][:, 0]
p_grid = gsw.p_from_z(z_grid, ref_lat)
xy_grid = MOD['x_grid'][0][0][0] * 1000.0
max_depth_per_xy = np.nan * np.ones(len(xy_grid))
for i in range(np.shape(sig0_out_s)[2]):
    deepest = z_grid[np.isnan(sig0_out_s[0, :, i])]
    if len(deepest) > 0:
        max_depth_per_xy[i] = deepest[0]
    else:
        max_depth_per_xy[i] = z_grid[0]


sigth_levels = np.array([23, 24, 24.5, 25, 25.5, 26, 26.2, 26.4, 26.6, 26.8,
                         27, 27.2, 27.4, 27.6, 27.7, 27.8, 27.9, 27.95,
                         28, 28.05, 28.1, 28.15])

E_W = 1

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
# to adjust
dg_vertical_speed = 0.08  # m/s
dg_glide_slope = 3
num_dives = 3
dac_error = 0.00001  # m/s
g_error = 0.00001
t_error = 0.0001
s_error = 0.0001
y_dg_s = 20000     # horizontal position, start of glider dives (75km)
z_dg_s = 0        # depth, start of glider dives
partial_mw = 0    # include exclude partial m/w estimates

tag = filename[11:]
t_st_ind = 10
t_e = np.nanmax(time_ord_s)
output_filename = '/Users/jake/Documents/baroclinic_modes/Model/HYCOM/vel_er_y10_v01_slp3_' + tag + '.pkl'
save_anom = 0
save_p = 0
save_p_g = 0

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
                        (y_dg_s + 3*2*np.nanmean(np.abs(max_depth_per_xy[xyg_start:xyg_start+6]))))[0][0]
data_depth_max = z_grid[np.where(np.isnan(np.mean(data_loc[:, xyg_start:xyg_proj_end+1], 1)))[0][0]] \
                 + np.abs(z_grid[-2] - z_grid[-1])  # depth of water (depth glider will dive to) estimate
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
                            ((dg_y[0, i - 1] + 10) + 3 * 2 *
                             np.nanmean(np.abs(max_depth_per_xy[xyg_start:xyg_start + 6]))))[0][0]
    data_depth_max = z_grid[np.where(np.isnan(np.mean(data_loc[:, xyg_start:xyg_proj_end + 1], 1)))[0][0]] \
                     + np.abs(z_grid[-2] - z_grid[-1])  # depth of water (depth glider will dive to) estimate
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
            dg_t[i, j] = dg_t[i + 1, j] + (np.abs(this_dg_z[i] - this_dg_z[i - 1]) / dg_vertical_speed) / (60 * 60 * 24)
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
            dg_t[i, j] = dg_t[i - 1, j] + (np.abs(dg_z_g[i, j] - dg_z_g[i - 1, j]) / dg_vertical_speed) / (60 * 60 * 24)

# --- Interpolation of Model to each glider measurements
data_interp = sig0_out_s
sa_interp = sa_out_s
ct_interp = ct_out_s
dg_sig0 = np.nan * np.ones(np.shape(dg_t))
dg_sa = np.nan * np.ones(np.shape(dg_t))
dg_ct = np.nan * np.ones(np.shape(dg_t))
for i in range(np.shape(dg_y)[1]):  # xy_grid
    max_ind = np.where(np.isnan(dg_y[:, i]))[0][0]
    this_dg_z = dg_z[0:max_ind]
    for j in range(len(this_dg_z)):  # depth
        if (i < 1) & (j < 1):
            # interpolation to horizontal position begins at t_st_ind
            dg_sig0[j, i] = np.interp(dg_y[j, i], xy_grid, data_interp[t_st_ind, j, :] + (g_error * np.random.rand(1)))
            dg_sa[j, i] = np.interp(dg_y[j, i], xy_grid, sa_interp[t_st_ind, j, :] + (s_error * np.random.rand(1)))
            dg_ct[j, i] = np.interp(dg_y[j, i], xy_grid, ct_interp[t_st_ind, j, :] + (t_error * np.random.rand(1)))
        else:
            this_t = dg_t[j, i]
            # find time bounding model runs
            nearest_model_t_over = np.where(time_ord_s > this_t)[0][0]
            # print(this_t - time_ord_s[nearest_model_t_over])
            # interpolate to hor position of glider dive for time before and after
            sig_t_before = np.interp(dg_y[j, i], xy_grid, data_interp[nearest_model_t_over - 1, j, :] + (g_error * np.random.rand(1)))
            sig_t_after = np.interp(dg_y[j, i], xy_grid, data_interp[nearest_model_t_over, j, :] + (g_error * np.random.rand(1)))
            sa_t_before = np.interp(dg_y[j, i], xy_grid, sa_interp[nearest_model_t_over - 1, j, :] + (s_error * np.random.rand(1)))
            sa_t_after = np.interp(dg_y[j, i], xy_grid, sa_interp[nearest_model_t_over, j, :] + (s_error * np.random.rand(1)))
            ct_t_before = np.interp(dg_y[j, i], xy_grid, ct_interp[nearest_model_t_over - 1, j, :] + (t_error * np.random.rand(1)))
            ct_t_after = np.interp(dg_y[j, i], xy_grid, ct_interp[nearest_model_t_over, j, :] + (t_error * np.random.rand(1)))
            # interpolate across time
            dg_sig0[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                   [sig_t_before, sig_t_after])
            dg_sa[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                   [sa_t_before, sa_t_after])
            dg_ct[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                   [ct_t_before, ct_t_after])

print('Simulated Glider Flight')
# ---------------------------------------------------------------------------------------------------------------------
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
    dg_dac[i] = np.nanmean(np.nanmean(np.nanmean(u_out_s[mtun:mtov, :, this_ygs+1:this_yge], axis=2), axis=0))
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
    isopycdep[isigth, i] = np.interp(sigth_levels[isigth], dg_sig0[:, i], np.flipud(z_grid))
    isopycx[isigth, i] = np.interp(sigth_levels[isigth], dg_sig0[:, i], dg_y[:, i])

    sigthmin = np.nanmin(np.array(dg_sig0[:, i + 1]))
    sigthmax = np.nanmax(np.array(dg_sig0[:, i + 1]))
    isigth = (sigth_levels > sigthmin) & (sigth_levels < sigthmax)
    isopycdep[isigth, i + 1] = np.interp(sigth_levels[isigth], dg_sig0[:, i + 1], np.flipud(z_grid))
    isopycx[isigth, i + 1] = np.interp(sigth_levels[isigth], dg_sig0[:, i + 1], dg_y[:, i + 1])

# interpolate dac to be centered on m/w locations
if E_W:
    dg_dac_use = 1. * nanseg_interp(dg_dac_mid, dg_dac) + dac_error * np.random.rand(len(dg_dac))  # added noise to DAC
else:
    dg_dac_use = -1. * nanseg_interp(dg_dac_mid, dg_dac)
# FOR EACH TRANSECT COMPUTE GEOSTROPHIC VELOCITY
vbc_g = np.nan * np.zeros(np.shape(shear))
v_g = np.nan * np.zeros((np.size(z_grid), num_profs))
for m in range(num_profs - 1):
    iq = np.where(~np.isnan(shear[:, m]))
    if np.size(iq) > 10:
        z2 = dg_z[iq]
        vrel = cumtrapz(shear[iq, m], x=z2, initial=0)
        vrel_av = np.trapz(vrel / (z2[-1] - 0), x=z2)
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
v_g = v_g[:, ~np.isnan(v_g[30, :])]
goodie = np.where(~np.isnan(v_g[30, :]))[0]
u_mod_at_mw = u_mod_at_mw[goodie[0]:goodie[-1]+1]
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
t_steps = [t_near, t_near + 6, t_near + 12, t_near + 18, t_near + 24, t_near + 30, t_near + 36,
           t_near + 42, t_near + 48, t_near + 54, t_near + 60, t_near + 66, t_near + 71]
isop_dep = np.nan * np.ones((len(deps), len(xy_grid), 6))
for i in range(len(xy_grid)):  # loop over each horizontal grid point
    for j in range(len(deps)):  # loop over each dep
        for k in range(6):  # loop over each 6 hr increment
            isop_dep[j, i, k] = np.interp(deps[j], -1. * dg_z, data_interp[t_steps[k], :, i])

# -- glider measured isopycnal depth (over time span)
dg_isop_dep = np.nan * np.ones((len(deps), 4))
dg_isop_xy = np.nan * np.ones((len(deps), 4))
for j in range(len(deps)):
    for k in range(4):  # loop over dive, climb, dive, climb
        dg_isop_dep[j, k] = np.interp(deps[j], -1. * dg_z, dg_sig0[:, k])
        dg_isop_xy[j, k] = np.interp(deps[j], -1. * dg_z, dg_y[:, k])
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# PLOTTING
h_max = np.nanmax(dg_y/1000 + 20)  # horizontal domain limit
z_max = -4800
u_levels = np.array([-.5, -.4, -0.3, -.25, - .2, -.15, -.125, -.1, -.075, -.05, -0.025, 0,
                     0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])

cmap = plt.cm.get_cmap("Spectral")

if dg_t[0, 0] < 1:
    taggt = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14])) + ' ' + str(np.int(dg_t[0, 0]*24)) + 'hr' + \
            ' - ' + str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 2) + ' ' + str(np.int((dg_t[0, -1] - 2)*24)) + 'hr'
elif (dg_t[0, 0] > 1) & (dg_t[0, 0] < 2):
    taggt = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 1) + ' ' + str(np.int((dg_t[0, 0] - 1)*24)) + 'hr' + \
            ' - ' + str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 3) + ' ' + str(np.int((dg_t[0, -1] - 3)*24)) + 'hr'
elif (dg_t[0, 0] > 2) & (dg_t[0, 0] < 3):
    taggt = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 2) + ' ' + str(np.int((dg_t[0, 0] - 2)*24)) + 'hr' + \
            ' - ' + str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 4) + ' ' + str(np.int((dg_t[0, -1] - 4)*24)) + 'hr'
elif (dg_t[0, 0] > 3) & (dg_t[0, 0] < 4):
    taggt = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 3) + ' ' + str(np.int((dg_t[0, 0] - 3)*24)) + 'hr' + \
            ' - ' + str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 5) + ' ' + str(np.int((dg_t[0, -1] - 5)*24)) + 'hr'
elif (dg_t[0, 0] > 4) & (dg_t[0, 0] < 5):
    taggt = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 4) + ' ' + str(np.int((dg_t[0, 0] - 4)*24)) + 'hr' + \
            ' - ' + str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 6) + ' ' + str(np.int((dg_t[0, -1] - 6)*24)) + 'hr'
elif (dg_t[0, 0] > 5) & (dg_t[0, 0] < 6):
    taggt = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 5) + ' ' + str(np.int((dg_t[0, 0] - 5)*24)) + 'hr' + \
            ' - ' + str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 7) + ' ' + str(np.int((dg_t[0, -1] - 7)*24)) + 'hr'

plot0 = 1
if plot0 > 0:
    matplotlib.rcParams['figure.figsize'] = (12, 6)

    t_in_low = np.where(time_ord_s >= np.nanmin(dg_t))[0][0]
    t_in_high = np.where(time_ord_s <= np.nanmax(dg_t))[0][-1]

    f, ax1 = plt.subplots()
    uvcf = ax1.contourf(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                 np.nanmean(u_out_s[t_in_low:t_in_high, :, :], axis=0), levels=u_levels, cmap=cmap)
    uvc = ax1.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                      np.nanmean(u_out_s[t_in_low:t_in_high, :, :], axis=0), levels=u_levels, colors='#2F4F4F', linewidths=0.5)
    ax1.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
    rhoc = ax1.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       np.nanmean(sig0_out_s, axis=0), levels=sigth_levels, colors='#A9A9A9', linewidths=0.5)
    ax1.scatter(dg_y/1000, dg_z_g, 4, color='k', label='glider path')
    # ax1.set_title(str(datetime.date.fromordinal(np.int(time_ord_s[0]))) +
    #               ' - ' + str(datetime.date.fromordinal(np.int(time_ord_s[-2]))) + ', 72hr. Model Avg.')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=11)
    ax1.set_ylim([z_max, 0])
    ax1.set_xlim([0, h_max])
    ax1.set_xlabel('E/W distance [km]')
    ax1.set_ylabel('z [m]')
    ax1.set_title(taggt)
    plt.colorbar(uvcf, label='N/S Velocity [m/s]', ticks=u_levels)
    ax1.grid()
    plot_pro(ax1)

    # matplotlib.rcParams['figure.figsize'] = (6, 6)
    # f, ax1 = plt.subplots()
    # for i in range(np.shape(dg_sa)[1]):
    #     ax1.scatter(dg_sa[:, i], dg_ct[:, i], s=1)
    # plot_pro(ax1)

plot_v = 1
if plot_v > 0:
    avg_mod_u = u_mod_at_mw_avg[:, 1:-1]
    matplotlib.rcParams['figure.figsize'] = (7, 7)
    f, ax1 = plt.subplots()
    for i in range(np.shape(v_g)[1]):
        ax1.plot(v_g[:, i], dg_z, color='#4682B4', linewidth=1.25, label='DG')
        # ax1.plot(V_m[:, i], dg_z, color='k', linestyle='--', linewidth=.75)
        ax1.plot(avg_mod_u[0:-20, i], dg_z[0:-20], color='r', linewidth=0.5, label='Model')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[-2], handles[-1]], [labels[-2], labels[-1]], fontsize=10)
    ax1.set_xlim([-.4, .4])
    ax1.set_title('Avg. Model and DG Velocity (u_g)')
    ax1.set_xlabel('Velocity [m/s]')
    ax1.set_ylabel('z [m]')
    plot_pro(ax1)

plot_rho = 1
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
    cmaps = [0.15, 0.22, 0.3, 0.37, 0.45, 0.52, 0.6, 0.67, 0.75, 0.82, .9, .98]
    lab_y = [25.875, 27.6, 27.91, 27.97]
    matplotlib.rcParams['figure.figsize'] = (10, 7)
    f, ax = plt.subplots(4, 1, sharex=True)
    for i in range(len(deps)):
        ax[i].set_facecolor('#DCDCDC')
        for j in range(6):
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
               label='72hr. avg. density ')
    handles, labels = ax[3].get_legend_handles_labels()
    ax[3].legend(handles, labels, fontsize=12)
    ax[0].set_ylabel('Neutral Density')
    ax[0].set_title(r'Density along z=z$_i$')
    ax[0].set_ylim([25.7, 26])
    ax[0].invert_yaxis()
    ax[0].grid()
    ax[0].set_xlim([0, h_max])
    ax[1].set_ylabel('Neutral Density')
    ax[1].set_ylim([27.5, 27.75])
    ax[1].invert_yaxis()
    ax[1].grid()
    ax[2].set_ylabel('Neutral Density')
    ax[2].set_ylim([27.89, 27.94])
    ax[2].invert_yaxis()
    ax[2].grid()
    ax[3].set_ylabel('Neutral Density')
    ax[3].set_ylim([27.96, 28])
    ax[3].invert_yaxis()
    ax[3].set_xlabel('Transect Distance [km]')
    plot_pro(ax[3])

plot_sp = 0
if plot_sp > 0:

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='glider_dives', artist='JS')
    writer = FFMpegWriter(fps=3, metadata=metadata)

    fig = plt.figure(figsize=(11,8))

    with writer.saving(fig, "/Users/jake/Documents/baroclinic_modes/Model/HYCOM/glider_ts_movie/glider_diver.mp4", 150):
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
                plt.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
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

                if this_t < 1:
                    tagg = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14])) + ' ' + str(
                        nearest_model_t_over) + 'hr'
                elif (this_t > 1) & (this_t < 2):
                    tagg = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 1) + ' ' + str(
                        nearest_model_t_over - 24) + 'hr'
                elif (this_t > 2) & (this_t < 3):
                    tagg = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 2) + ' ' + str(
                        nearest_model_t_over - 48) + 'hr'
                elif (this_t > 3) & (this_t < 4):
                    tagg = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 3) + ' ' + str(
                        nearest_model_t_over - 72) + 'hr'
                elif (this_t > 4) & (this_t < 5):
                    tagg = str(np.int(filename[11])) + '/' + str(np.int(filename[12:14]) + 4) + ' ' + str(
                        nearest_model_t_over - 96) + 'hr'

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
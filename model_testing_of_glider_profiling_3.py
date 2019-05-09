import numpy as np
import pickle
import glob
import datetime
from netCDF4 import Dataset
import gsw
import time as TT
from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter
from mode_decompositions import eta_fit, vertical_modes, PE_Tide_GM, vertical_modes_f
# -- plotting
import matplotlib
import matplotlib.pyplot as plt
from toolkit import plot_pro, nanseg_interp
from zrfun import get_basic_info, get_z

# because we load pickle protocol 2 (needed for matlab engine) we need 'glider' environment (not 'geo_env')

this_path = 'e_w_extraction_nov19_nov21_offshore'  # 'n_s_extraction_eddy_nov1_nov3'  #

# -- LOAD extracted and PROCESSED MODEL TIME STEPS WITH COMPUTED GAMMA
# this file has combined all model output and computed gamma (using model_testing_of_glider_profiling_2.py)
# pkl_file = open('/Users/jake/Documents/baroclinic_modes/Model/e_w_extraction_eddy_oct29_oct31/'
#                 'gamma_output/extracted_gridded_gamma.pkl', 'rb')
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Model/' + this_path +
                '/gamma_output/extracted_gridded_gamma.pkl', 'rb')
MOD = pickle.load(pkl_file)
pkl_file.close()
ref_lat = 44
sig0_out_s = MOD['gamma'][:]
ct_out_s = MOD['CT'][:]  # temp
sa_out_s = MOD['SA'][:]  # salin
u_out_s = MOD['vel_cross_transect'][:]
u_off_out_s = MOD['vel_along_transect']
time_out_s = MOD['time'][:]
time_ord_s = MOD['time_ord'][:]
date_out_s = MOD['date'][:]
z_grid = MOD['z'][:]
p_grid = gsw.p_from_z(z_grid, ref_lat)
xy_grid = MOD['dist_grid'][:]
lon_grid = MOD['lon_grid'][:]
lat_grid = MOD['lat_grid'][:]
ref_lon = np.nanmin(lon_grid)
ref_lat = np.nanmin(lat_grid)
max_depth_per_xy = np.nan * np.ones(len(xy_grid))
for i in range(np.shape(sig0_out_s)[1]):
    deepest = z_grid[np.isnan(sig0_out_s[:, i, 0])]
    if len(deepest) > 0:
        max_depth_per_xy[i] = np.nanmax(deepest[-1])
    else:
        max_depth_per_xy[i] = z_grid[0]

# check if profile is N_S or E_W
# transects will be South to North or East to West
# if South to North, m/w technique estimates relative velocity to port of the glider. this is in the NEGATIVE u
# direction, so dac u or v values will have to be flipped to estimate abs geostrophic vel
# if East to West, m/w technique similary will estimate relative velocity to port... in the POSITIVE v direction
# dac values should NOT be flipped when estimating abs geostrophic vel.
lon_check = lon_grid[-1] - lon_grid[0]
E_W = 0
N_S = 0
if lon_check > 0.05:
    E_W = 1
else:
    N_S = 1

sigth_levels = np.array([25, 26, 26.2, 26.4, 26.6, 26.8, 27, 27.2, 27.4, 27.6, 27.7, 27.8, 27.9])
g = 9.81
rho0 = 1025.0
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# CREATE LOCATIONS OF FAKE GLIDER DIVE
# approximate glider as flying with vertical velocity of 0.08 m/s, glide slope of 1:3
# assume glider measurements are made every 10, 15, 30, 45, 120 second intervals
# time between each model time step is 1hr.

# main set of parameters to adjust
dg_vertical_speed = 0.1  # m/s
dg_glide_slope = 3
num_dives = 4
dac_error = 0.01  # m/s
y_dg_s = 70000     # horizontal position, start of glider dives (75km)
z_dg_s = 0        # depth, start of glider dives
partial_mw = 0    # include exclude partial m/w estimates

t_s = datetime.date.fromordinal(np.int(time_ord_s[0]))
t_e = datetime.date.fromordinal(np.int(time_ord_s[-1]))
tag = str(t_s.month) + '_' + str(t_s.day) + '_' + str(t_e.month) + '_' + str(t_e.day)
output_filename = '/Users/jake/Documents/baroclinic_modes/Model/vel_anom_fast_calm_' + tag + '.pkl'

# need to specify D_TGT or have glider 'fly' until it hits bottom
data_loc = np.nanmean(sig0_out_s, axis=2)  # (depth X xy_grid)

# glider dives are simulated by considered the depth grid as fixed and march down and up while stepping horizontally
# following the desired glide slope and vertical velocity
# --- DG Z
dg_z = np.flipud(z_grid.copy())  # vertical grid points
dg_z_g = np.tile(np.concatenate((np.flipud(z_grid.copy())[:, None], np.flipud(z_grid.copy())[:, None]), axis=1),
                 (1, num_dives))  # grid of vertical grid points across all dives
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
data_depth_max = z_grid[np.where(np.isnan(np.mean(data_loc[:, xyg_start:xyg_proj_end+1], 1)))[0][-1]] \
                 + np.abs(z_grid[-2] - z_grid[-1])  # depth of water (depth glider will dive to) estimate
dg_z_ind = np.where(dg_z >= data_depth_max)[0]
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
    data_depth_max = z_grid[np.where(np.isnan(np.mean(data_loc[:, xyg_start:xyg_proj_end + 1], 1)))[0][-1]] \
                     + np.abs(z_grid[-2] - z_grid[-1])  # depth of water (depth glider will dive to) estimate
    dg_z_ind = np.where(dg_z >= data_depth_max)[0]
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
dg_t[0, 0] = np.nanmin(time_ord_s)
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
data_interp = np.flipud(sig0_out_s)
sa_interp = np.flipud(sa_out_s)
ct_interp = np.flipud(ct_out_s)
dg_sig0 = np.nan * np.ones(np.shape(dg_t))
dg_sa = np.nan * np.ones(np.shape(dg_t))
dg_ct = np.nan * np.ones(np.shape(dg_t))
for i in range(np.shape(dg_y)[1]):  # xy_grid
    max_ind = np.where(np.isnan(dg_y[:, i]))[0][0]
    this_dg_z = dg_z[0:max_ind]
    for j in range(len(this_dg_z)):  # depth
        if (i < 1) & (j < 1):
            dg_sig0[j, i] = np.interp(dg_y[j, i], xy_grid, data_interp[j, :, 0])
            dg_sa[j, i] = np.interp(dg_y[j, i], xy_grid, sa_interp[j, :, 0])
            dg_ct[j, i] = np.interp(dg_y[j, i], xy_grid, ct_interp[j, :, 0])
        else:
            this_t = dg_t[j, i]
            # find time bounding model runs
            nearest_model_t_over = np.where(time_ord_s > this_t)[0][0]
            # print(this_t - time_ord_s[nearest_model_t_over])
            # interpolate to hor position of glider dive for time before and after
            sig_t_before = np.interp(dg_y[j, i], xy_grid, data_interp[j, :, nearest_model_t_over - 1])
            sig_t_after = np.interp(dg_y[j, i], xy_grid, data_interp[j, :, nearest_model_t_over])
            sa_t_before = np.interp(dg_y[j, i], xy_grid, sa_interp[j, :, nearest_model_t_over - 1])
            sa_t_after = np.interp(dg_y[j, i], xy_grid, sa_interp[j, :, nearest_model_t_over])
            ct_t_before = np.interp(dg_y[j, i], xy_grid, ct_interp[j, :, nearest_model_t_over - 1])
            ct_t_after = np.interp(dg_y[j, i], xy_grid, ct_interp[j, :, nearest_model_t_over])
            # interpolate across time
            dg_sig0[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                   [sig_t_before, sig_t_after])
            dg_sa[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                   [sa_t_before, sa_t_after])
            dg_ct[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                   [ct_t_before, ct_t_after])

print('Simulated Glider Flight')
# ---------------------------------------------------------------------------------------------------------------------
ff = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)  # Coriolis parameter [s^-1]
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
    dg_dac[i] = np.nanmean(np.nanmean(np.nanmean(u_out_s[:, this_ygs+1:this_yge, mtun+1:mtov], axis=2), axis=0))
    dg_dac_off[i] = np.nanmean(np.nanmean(np.nanmean(u_off_out_s[:, this_ygs+1:this_yge, mtun+1:mtov], axis=2), axis=0))
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
u_model = np.flipud(u_out_s)
u_mod_at_mw = []
u_mod_at_mw_avg = np.nan * np.ones((len(dg_z), len(dg_dac_mid)))
for i in range(len(dg_dac_mid)):
    # time bounds of m/w profiles
    if i <= 1:
        min_t = dg_t[0, 0]
        max_t = mw_time[i + 2]
        min_y = 0
        max_y = dg_dac_mid[i + 2]
    elif (i > 1) & (i < len(dg_dac_mid) - 2):
        min_t = mw_time[i - 2]
        max_t = mw_time[i + 2]
        min_y = dg_dac_mid[i - 2]
        max_y = dg_dac_mid[i + 2]
    elif i > len(dg_dac_mid) - 2:
        min_t = mw_time[i - 2]
        max_t = dg_t[0, -1]
        min_y = dg_dac_mid[i - 2]
        max_y = np.nanmax(dg_y)
    mtun = np.where(time_ord_s <= min_t)[0][-1]
    mtov = np.where(time_ord_s > max_t)[0][0]
    this_ygs = np.where(xy_grid <= min_y)[0][-1]
    this_yge = np.where(xy_grid >= max_y)[0][0]

    u_mod_at_mw.append(u_model[:, this_ygs+1:this_yge, mtun:mtov])
    u_mod_at_mw_avg[:, i] = np.nanmean(np.nanmean(u_model[:, this_ygs+1:this_yge, mtun:mtov], axis=2), axis=1)

    # mean_t = np.nanmean(dg_t[:, i:i + 2])  # mean time of one dive_climb cycle
    # nearest_t = np.where(time_ord_s < mean_t)[0][-1]
    # mw_time[i] = np.nanmean(time_ord_s[nearest_t:nearest_t + 2])
    # avg_u_at = np.nanmean(u_out_s[:, :, nearest_t:nearest_t + 2], axis=2)
    # for j in range(len(z_grid)):
    #     u_mod_at_mw_y[j, i] = np.interp(mw_y[i], xy_grid, avg_u_at[j, :])

# ----------------------------------------------------------------------------------------------------------------------
# transect plan view with DAC
if E_W:
    dac_lon = dg_dac_mid / (1852. * 60. * np.cos(np.deg2rad(ref_lat))) + ref_lon
    dac_lat = ref_lat * np.ones(len(dac_lon))
else:
    dac_lat = dg_dac_mid / (1852. * 60) + ref_lat
    dac_lon = ref_lon * np.ones(len(dac_lat))

# --- PLAN VIEW plot
plan_plot = 0
if plan_plot:
    # -- LOAD processed output from model_testing_of glider.py (location of transect)
    file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/' + this_path + '/pickle_protocol_2/*.pkl')
    file_name = open(file_list[0], 'rb')
    D = pickle.load(file_name)
    file_name.close()
    # for bathymetry
    file_name = '/Users/jake/Documents/baroclinic_modes/Model/LiveOcean_11_01_18/ocean_his_0001.nc'
    Db = Dataset(file_name, 'r')
    lon_rho = Db['lon_rho'][:]
    lat_rho = Db['lat_rho'][:]
    zeta = Db['zeta'][0]  # SSH
    h = Db['h'][:]  # bathymetry
    G, S, T = get_basic_info(file_name, only_G=False, only_S=False, only_T=False)
    z = get_z(h, zeta, S)[0]

    f, ax = plt.subplots()
    ax.contour(lon_rho, lat_rho, z[0, :, :], levels=[-25, -20, -15, -10, -5], colors='k', fontsize=6)
    bc = ax.contour(lon_rho, lat_rho, z[0, :, :],
                    levels=[-3000, -2900, -2800, -2700, -2600, -2500, -2000, -1000], colors='b', linewidth=0.5)
    ax.quiver(dac_lon, dac_lat, dg_dac_u, dg_dac_v, scale=.25, width=0.01)
    ax.clabel(bc, inline_spacing=-3, fmt='%.4g', colors='b')
    ax.scatter(D['lon_rho'][:], D['lat_rho'][:], 10, color='r')
    w = 1 / np.cos(np.deg2rad(46))
    ax.set_aspect(w)
    ax.axis([-128, -123, 42, 48])
    plot_pro(ax)
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
# --- PLOTTING
h_max = np.nanmax(dg_y/1000 + 20)  # horizontal domain limit
z_max = -3250
u_levels = np.array([-.4, -.35, -.3, -.25, - .2, -.15, -.125, -.1, -.075, -.05, -0.025, 0,
                     0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, .4])
plot0 = 1
if plot0 > 0:
    matplotlib.rcParams['figure.figsize'] = (12, 6)
    cmap = plt.cm.get_cmap("viridis")
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.contourf(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                 np.nanmean(u_out_s, axis=2), levels=u_levels, cmap=cmap)
    uvc = ax1.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                      np.nanmean(u_out_s, axis=2), levels=u_levels, colors='k', linewidth=0.75)
    ax1.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
    rhoc = ax1.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       np.nanmean(sig0_out_s, axis=2), levels=sigth_levels, colors='#A9A9A9', linewidth=0.35)
    ax1.scatter(dg_y/1000, dg_z_g, 4, color='k')
    for r in range(np.shape(isopycdep)[0]):
        ax1.plot(isopycx[r, :]/1000, isopycdep[r, :], color='r', linewidth=0.45)
    ax1.set_title(str(datetime.date.fromordinal(np.int(time_ord_s[0]))) +
                  ' - ' + str(datetime.date.fromordinal(np.int(time_ord_s[-2]))) + ', 72hr. Model Avg.')
    ax1.set_ylim([z_max, 0])
    ax1.set_xlim([0, h_max])
    ax1.set_xlabel('Km')
    ax1.set_ylabel('z [m]')
    ax1.grid()

    ax2.contourf(np.tile(mw_y/1000, (len(z_grid), 1)), np.tile(dg_z[:, None], (1, len(mw_y))),
                 v_g, levels=u_levels, cmap=cmap)
    uvc = ax2.contour(np.tile(mw_y/1000, (len(z_grid), 1)), np.tile(dg_z[:, None], (1, len(mw_y))), v_g,
                      levels=u_levels, colors='k', linewidth=0.75)
    ax2.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
    ax2.scatter(dg_y/1000, dg_z_g, 4, color='k')
    for r in range(np.shape(isopycdep)[0]):
        ax2.plot(isopycx[r, :]/1000, isopycdep[r, :], color='r')
    ax2.set_xlim([0, h_max])
    ax2.set_title('Glider Cross-Track Vel.')
    ax2.set_xlabel('Km')
    ax2.grid()

    ax3.pcolor(np.tile(mw_y/1000, (len(z_grid), 1)), np.tile(dg_z[:, None], (1, len(mw_y))),
               u_mod_at_mw_avg - v_g, vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels), cmap=cmap)
    uvc = ax3.contour(np.tile(mw_y/1000, (len(z_grid), 1)), np.tile(dg_z[:, None], (1, len(mw_y))),
                      u_mod_at_mw_avg - v_g, levels=u_levels, colors='k', linewidth=0.75)
    ax3.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
    ax3.set_title('Model, Glider Velocity Difference')
    ax3.set_xlabel('Km')
    ax3.set_xlim([0, h_max])
    plot_pro(ax3)

# ------------------------------
# model - glider velocity error

ua_mean = np.nan * np.ones((len(dg_z), len(dg_dac_mid)))
ua_std = np.nan * np.ones((len(dg_z), len(dg_dac_mid)))
ua_avg_mean = np.nan * np.ones((len(dg_z), len(dg_dac_mid)))
ua_avg_std = np.nan * np.ones((len(dg_z), len(dg_dac_mid)))
for i in range(len(dg_dac_mid)):
    these_u = u_mod_at_mw[i]  # each u_mod_at_mw[i] contains all model profiles spanning the distance covered by
    # one m/w set of profiles and including the amount of time it took to collect m/w profiles
    for j in range(np.shape(these_u)[2]):
        each_anoms = these_u[:, :, j] - np.tile(v_g[:, i][:, None], (1, np.shape(these_u)[1]))
        if j < 1:
            u_anoms = each_anoms.copy()
        else:
            u_anoms = np.concatenate((u_anoms, each_anoms), axis=1)
    # u_anoms = these_u - np.transpose(np.tile(v_g[:, i], (np.shape(these_u)[1], np.shape(these_u)[2], 1)), (2, 0, 1))

    these_u_space_avg = np.nanmean(these_u, axis=1)  # avg spatially then get statistic on time variability
    # difference between this v_g profile and the avg profile of all model profiles spanning this dive-climb cycle
    # at each time step
    u_avg_anoms = these_u_space_avg - np.tile(v_g[:, i][:, None], (1, np.shape(these_u_space_avg)[1]))
    for j in range(len(dg_z)):
        ua_mean[j, i] = np.nanmean(u_anoms[j, :])
        ua_std[j, i] = np.nanstd(u_anoms[j, :])
        ua_avg_mean[j, i] = np.nanmean(u_avg_anoms[j, :])
        ua_avg_std[j, i] = np.nanstd(u_avg_anoms[j, :])

plot1 = 0
if plot1 > 0:
    f, ax = plt.subplots()
    # anomaly from each model profile and glider profile (in space and time)
    ax.errorbar(np.nanmean(ua_mean, axis=1), dg_z, xerr=np.nanmean(ua_std, axis=1))
    # anomaly from spatially averaged profiles that span the m/w section, but vary in time
    ax.errorbar(np.nanmean(ua_avg_mean, axis=1), dg_z, xerr=np.nanmean(ua_avg_std, axis=1))
    ax.plot(np.nanmean(ua_mean, axis=1), dg_z, color='k', linewidth=2.5)
    ax.set_ylim([z_max, 0])
    ax.set_xlim([-0.2, 0.2])
    ax.set_title('Model V. - DG V.')
    ax.set_xlabel('Velocity Error [m/s]')
    ax.set_ylabel('z [m]')
    plot_pro(ax)

    f, ax = plt.subplots()
    for i in range(np.shape(shear)[1]):
        these_u = u_mod_at_mw[i]
        these_u_space_avg = np.nanmean(these_u, axis=1)  # avg spatially then get statistic on time variability
        for j in range(len(shear[:, i])):
            u_avg_anoms = these_u_space_avg[j, :] - v_g[j, i]  # choose specific depth to look at u anomalies
            ax.scatter(shear[j, i], np.nanmean(u_avg_anoms), s=3) # avg velocity anomaly at depth x for given shear
    plot_pro(ax)

# model snapshots over the 72hr period
plot2 = 0
if plot2 > 0:
    ti1 = 23
    ti2 = 46
    ti3 = 69
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    ax1.pcolor(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
               u_out_s[:, :, ti1], vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
    uvc = ax1.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       u_out_s[:, :, ti1], levels=u_levels, colors='k', linewidth=0.75)
    rhoc = ax1.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       sig0_out_s[: ,:, ti1], levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

    ax2.pcolor(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
               u_out_s[:, :, ti2], vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
    uvc = ax2.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       u_out_s[:, :, ti2], levels=u_levels, colors='k', linewidth=0.75)
    rhoc = ax2.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       sig0_out_s[: ,:, ti2], levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

    ax3.pcolor(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
               u_out_s[:, :, ti3], vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
    uvc = ax3.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       u_out_s[:, :, ti3], levels=u_levels, colors='k', linewidth=0.75)
    rhoc = ax3.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       sig0_out_s[: ,:, ti3], levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

    ax4.pcolor(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
               np.nanmean(u_out_s, 2), vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
    uvc = ax4.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       np.nanmean(u_out_s, axis=2), levels=u_levels, colors='k', linewidth=0.75)
    rhoc = ax4.contour(np.tile(xy_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(xy_grid))),
                       np.nanmean(sig0_out_s, axis=2), levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

    ax1.set_xlim([0, h_max])
    ax1.set_ylim([z_max, 0])
    ax1.set_title(str(datetime.date.fromordinal(np.int(time_ord_s[ti1]))) + ', ' +
                  str(np.int(date_out_s[ti1, 1])) + 'hrs.')
    ax1.set_ylabel('Z [m]')
    ax1.set_xlabel('Km')
    ax2.set_xlim([0, h_max])
    ax2.set_title(str(datetime.date.fromordinal(np.int(time_ord_s[ti2]))) + ', ' +
                  str(np.int(date_out_s[ti2, 1])) + 'hrs.')
    ax2.set_xlabel('Km')
    ax3.set_xlim([0, h_max])
    ax3.set_title(str(datetime.date.fromordinal(np.int(time_ord_s[ti3]))) + ', ' +
                  str(np.int(date_out_s[ti3, 1])) + 'hrs.')
    ax3.set_xlabel('Km')
    ax4.set_xlim([0, h_max])
    ax4.set_title('72hr avg.')
    ax4.set_xlabel('Km')
    plot_pro(ax4)
# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
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

# -- model isopycnal depths
# these_data = data_interp[:, xy_grid_near, :]
# these_data2 = data_interp[:, xy_grid_near + 2, :]
these_data = data_interp[:, :, t_steps[0]]
these_data2 = data_interp[:, :, t_steps[1]]
these_data3 = data_interp[:, :, t_steps[2]]
these_data4 = data_interp[:, :, t_steps[3]]
these_data5 = data_interp[:, :, t_steps[4]]
these_data6 = data_interp[:, :, t_steps[5]]
isop_dep = np.nan * np.ones((len(isops), np.shape(these_data)[1]))
isop_dep2 = np.nan * np.ones((len(isops), np.shape(these_data)[1]))
isop_dep3 = np.nan * np.ones((len(isops), np.shape(these_data)[1]))
isop_dep4 = np.nan * np.ones((len(isops), np.shape(these_data)[1]))
isop_dep5 = np.nan * np.ones((len(isops), np.shape(these_data)[1]))
isop_dep6 = np.nan * np.ones((len(isops), np.shape(these_data)[1]))
for i in range(np.shape(these_data)[1]):
    for j in range(len(isops)):
        # each of these is at a different time in the ~ 30 hr window
        isop_dep[j, i] = np.interp(isops[j], these_data[:, i], -1. * dg_z)
        isop_dep2[j, i] = np.interp(isops[j], these_data2[:, i], -1. * dg_z)
        isop_dep3[j, i] = np.interp(isops[j], these_data3[:, i], -1. * dg_z)
        isop_dep4[j, i] = np.interp(isops[j], these_data4[:, i], -1. * dg_z)
        isop_dep5[j, i] = np.interp(isops[j], these_data5[:, i], -1. * dg_z)
        isop_dep6[j, i] = np.interp(isops[j], these_data6[:, i], -1. * dg_z)

# -- glider measured isopycnal depth (over time span)
dg_isop_dep = np.nan * np.ones((len(isops), 4))
dg_isop_xy = np.nan * np.ones((len(isops), 4))
# first dive
for j in range(len(isops)):
    dg_isop_dep[j, 0] = np.interp(isops[j], dg_sig0[:, 0], -1. * dg_z)
    dg_isop_xy[j, 0] = np.interp(isops[j], dg_sig0[:, 0], dg_y[:, 0])
# first climb
for j in range(len(isops)):
    dg_isop_dep[j, 1] = np.interp(isops[j], dg_sig0[:, 1], -1. * dg_z)
    dg_isop_xy[j, 1] = np.interp(isops[j], dg_sig0[:, 1], dg_y[:, 1])
# second dive
for j in range(len(isops)):
    dg_isop_dep[j, 2] = np.interp(isops[j], dg_sig0[:, 2], -1. * dg_z)
    dg_isop_xy[j, 2] = np.interp(isops[j], dg_sig0[:, 2], dg_y[:, 2])
# second climb
for j in range(len(isops)):
    dg_isop_dep[j, 3] = np.interp(isops[j], dg_sig0[:, 3], -1. * dg_z)
    dg_isop_xy[j, 3] = np.interp(isops[j], dg_sig0[:, 3], dg_y[:, 3])

# ---------------------------------------------------------------------------------------------------------------------
# -- Variances
# fraction of variance contained in igw/tide fluctuations (compared to total variance of isopynal depth
l=0
# 30 hr avg model isopycnal depth as a function of xy_grid
avg_isop_pos = np.nanmean(np.squeeze(np.array([[isop_dep[l, :]], [isop_dep2[l, :]], [isop_dep3[l, :]],
                                               [isop_dep4[l, :]], [isop_dep5[l, :]], [isop_dep6[l, :]]])), axis=0)

# xy within glider dives
inn = np.where((xy_grid >= dg_y[0,0]) & (xy_grid <= dg_y[0, -1]))[0]

# at each xy_grid point square difference between 30 hr avg and hourly output
isop_anom = (isop_dep[l, inn] - avg_isop_pos[inn])**2
isop_anom2 = (isop_dep2[l, inn] - avg_isop_pos[inn])**2
isop_anom3 = (isop_dep3[l, inn] - avg_isop_pos[inn])**2
isop_anom4 = (isop_dep4[l, inn] - avg_isop_pos[inn])**2
isop_anom5 = (isop_dep5[l, inn] - avg_isop_pos[inn])**2
isop_anom6 = (isop_dep6[l, inn] - avg_isop_pos[inn])**2
# list of square differences
combo_0 = np.concatenate((isop_anom, isop_anom2, isop_anom3, isop_anom4, isop_anom5, isop_anom6))
# mean square differences
igw_var = (1./len(combo_0)) * np.nansum(combo_0)  # model noise signal
# at each xy_grid point square difference between avg of 30 hr avg (horizontal line) and hourly output
isop_anom_t = (isop_dep[l, inn] - np.nanmean(avg_isop_pos[inn]))**2
isop_anom2_t = (isop_dep2[l, inn] - np.nanmean(avg_isop_pos[inn]))**2
isop_anom3_t = (isop_dep3[l, inn] - np.nanmean(avg_isop_pos[inn]))**2
isop_anom4_t = (isop_dep4[l, inn] - np.nanmean(avg_isop_pos[inn]))**2
isop_anom5_t = (isop_dep5[l, inn] - np.nanmean(avg_isop_pos[inn]))**2
isop_anom6_t = (isop_dep6[l, inn] - np.nanmean(avg_isop_pos[inn]))**2
combo_1 = np.concatenate((isop_anom_t, isop_anom2_t, isop_anom3_t, isop_anom4_t, isop_anom5_t, isop_anom6_t))
# mean square differences
tot_var = (1./len(combo_1)) * np.nansum(combo_1)
frac_igw_var = igw_var / tot_var

# mean square of difference from avg of 30 hr avg and 30 hr avg
model_eddy_rms = np.nanmean((avg_isop_pos[inn] - np.nanmean(avg_isop_pos[inn]))**2)  # model eddy signal

inn = np.where((xy_grid >= dg_y[0,0]) & (xy_grid <= dg_y[0, 3]))[0]
# fit model slope as glider does (for
model_isop_slope = np.polyfit(xy_grid[inn], isop_dep[l, inn], 1)
model_mean_isop = np.polyval(model_isop_slope, xy_grid[inn])

# glider slope of 4 points
dg_isop_slope = np.polyfit(dg_isop_xy[l, :], dg_isop_dep[l, :], 1)
dg_mean_isop = np.polyval(dg_isop_slope, dg_isop_xy[l, :])
dg_igw_var = np.nanmean((dg_isop_dep[l, :] - dg_mean_isop)**2)  # noise signal
dg_tot_var = np.nanmean((dg_isop_dep[l, :] - np.nanmean(dg_mean_isop))**2)  # total signal

# mean square of difference from avg of dg_isop_slope and dg_isop_slope
dg_eddy_rms = np.nanmean((dg_mean_isop - np.nanmean(dg_mean_isop))**2)  # eddy signal

# # variance of difference between glider measurement and mean isop pos
# avg_isop_dep_at_dg = np.nan * np.ones(len(dg_isop_xy[l, :]))
# dg_mod_anom = np.nan * np.ones(len(dg_isop_xy[l, :]))
# for i in range(len(dg_isop_xy[l, :])):
#     avg_isop_dep_at_dg[i] = np.interp(dg_isop_xy[l, i], xy_grid, avg_isop_pos)
#     dg_mod_anom[i] = (dg_isop_dep[l, i] - avg_isop_dep_at_dg[i])**2
# # mean square difference between DG isopycnal depth and avg model isopycnal depth
# dg_mod_var = (1./len(dg_mod_anom)) * np.nansum(dg_mod_anom)
# mod_var_at_dg_loc = (1./len(avg_isop_dep_at_dg)) * np.nansum((avg_isop_dep_at_dg - np.nanmean(avg_isop_dep_at_dg))**2)  # variance of model over the m/w pattern
#
# # fraction of variance from igw over m/w pattern
# inn = np.where((xy_grid >= dg_y[0,0]) & (xy_grid <= dg_y[0, 3]))[0]
# combo_3 = np.concatenate((isop_anom[inn], isop_anom2[inn], isop_anom3[inn], isop_anom4[inn], isop_anom5[inn], isop_anom6[inn]))
# igw_var_in = (1./len(combo_3)) * np.nansum(combo_3)
# combo_4 = np.concatenate((isop_anom_t[inn], isop_anom2_t[inn], isop_anom3_t[inn], isop_anom4_t[inn], isop_anom5_t[inn], isop_anom6_t[inn]))
# tot_var_in = (1./len(combo_4)) * np.nansum(combo_4)
# frac_var_from_igw = igw_var_in / tot_var_in

dg_isop_slope = np.nan * np.ones((len(isops), 2))
dg_mean_isop = np.nan * np.ones((len(isops), len(dg_isop_xy[0, :])))
model_isop_slope = np.nan * np.ones((len(isops), 2))
model_mean_isop = np.nan * np.ones((len(isops), len(xy_grid[inn])))
for i in range(4):
    dg_isop_slope[i, :] = np.polyfit(dg_isop_xy[i, :], dg_isop_dep[i, :], 1)
    dg_mean_isop[i, :] = np.polyval(dg_isop_slope[i, :], dg_isop_xy[i, :])

    model_isop_slope[i, :] = np.polyfit(xy_grid[inn], isop_dep[i, inn], 1)
    model_mean_isop[i, :] = np.polyval(model_isop_slope[i, :], xy_grid[inn])

# ---------------------------------------------------------------------------------------------------------------------
# PLOT DEPTH OF ISOPYCNAL IN SPACE AND TIME
matplotlib.rcParams['figure.figsize'] = (10, 6)
cmap = plt.cm.get_cmap("YlGnBu_r")
f, ax = plt.subplots(4, 1, sharex=True)
for i in range(len(isops)):
    avg_isop_pos_i = np.nanmean(np.squeeze(np.array([[isop_dep[i, :]], [isop_dep2[i, :]], [isop_dep3[i, :]],
                                                   [isop_dep4[i, :]], [isop_dep5[i, :]], [isop_dep6[i, :]]])), axis=0)

    ax[i].set_facecolor('#DCDCDC')
    ax[i].plot(xy_grid/1000, isop_dep[i, :])
    ax[i].plot(xy_grid/1000, isop_dep2[i, :], color=cmap(0.1))
    ax[i].plot(xy_grid/1000, isop_dep3[i, :], color=cmap(0.3))
    ax[i].plot(xy_grid/1000, isop_dep4[i, :], color=cmap(0.5))
    ax[i].plot(xy_grid/1000, isop_dep5[i, :], color=cmap(0.7))
    ax[i].plot(xy_grid/1000, isop_dep6[i, :], color=cmap(0.9))
    ax[i].plot(xy_grid/1000, avg_isop_pos_i, color='r', linestyle='--')
    ax[i].scatter(dg_isop_xy[i, :]/1000, dg_isop_dep[i, :], s=40, color='k', zorder=10)
    ax[i].plot([dg_y[0, 0]/1000, dg_y[0, 0]/1000], [0, 2000], linestyle='--', color='k')
    ax[i].plot([dg_y[0, 1]/1000, dg_y[0, 1]/1000], [0, 2000], linestyle='--', color='k')
    ax[i].plot([dg_y[0, 3]/1000, dg_y[0, 3]/1000], [0, 2000], linestyle='--', color='k')
    ax[i].plot(xy_grid[inn] / 1000, model_mean_isop[i, :], color='#FF8C00', linewidth=2)
    ax[i].plot(dg_isop_xy[i, :] / 1000, dg_mean_isop[i, :], color='m', linewidth=2)
ax[0].set_ylabel('Depth [m]')
ax[0].set_title('Isopycnal Depths over a 30hr Period (two glider dive-climb cycles)')
ax[0].set_ylim([225, 375])
ax[0].invert_yaxis()
ax[0].grid()
ax[0].set_xlim([0, h_max])
ax[1].set_ylabel('Depth [m]')
ax[1].set_ylim([700, 850])
ax[1].invert_yaxis()
ax[1].grid()
ax[2].set_ylabel('Depth [m]')
ax[2].set_ylim([1500, 1650])
ax[2].invert_yaxis()
ax[2].grid()
ax[3].set_ylabel('Depth [m]')
ax[3].set_ylim([1825, 1975])
ax[3].invert_yaxis()
ax[3].set_xlabel('Transect Distance [km]')
plot_pro(ax[3])

# - colormap key
# f, ax = plt.subplots()
# xx = np.arange(0, 1., 1./100)*30
# ax.set_facecolor('#DCDCDC')
# for i in range(100):
#     ax.scatter(xx[i], 0, color=cmap(xx[i]/30), s=100)
# ax.set_xlabel('Hour')
# ax.set_ylim([-0.001, 0.001])
# ax.yaxis.set_visible(False)
# ax.grid()
# plot_pro(ax)
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
deps = [250, 1000, 1500, 2000]
t_steps = [t_near, t_near + 6, t_near + 12, t_near + 18, t_near + 24, t_near + 30, t_near + 36,
           t_near + 42, t_near + 48, t_near + 54, t_near + 60, t_near + 66, t_near + 71]
isop_dep = np.nan * np.ones((len(deps), np.shape(these_data)[1], 6))
for i in range(np.shape(these_data)[1]):  # loop over each horizontal grid point
    for j in range(len(deps)):  # loop over each dep
        for k in range(6):  # loop over each 6 hr increment
            isop_dep[j, i, k] = np.interp(deps[j], -1. * dg_z, data_interp[:, i, t_steps[k]])

# -- glider measured isopycnal depth (over time span)
dg_isop_dep = np.nan * np.ones((len(deps), 4))
dg_isop_xy = np.nan * np.ones((len(deps), 4))
for j in range(len(deps)):
    for k in range(4):  # loop over dive, climb, dive, climb
        dg_isop_dep[j, k] = np.interp(deps[j], -1. * dg_z, dg_sig0[:, k])
        dg_isop_xy[j, k] = np.interp(deps[j], -1. * dg_z, dg_y[:, k])

# ---------
# xy within glider dives (model xy over which to estimate density gradients)
inn_per_mw = []
inn_per_mw_t = []
mw_ind_s = np.arange(0, np.shape(shear)[1])
mw_ind_e = np.arange(3, 3 + np.shape(shear)[1])
dg_t_hrs = 24. * (dg_t - dg_t[0, 0])
model_t_hrs = (time_out_s - time_out_s[0]) / 3600.
for i in range(np.shape(shear)[1]):
    if np.mod(i, 2):  # M
        inn_per_mw.append(np.where((xy_grid >= np.nanmin(dg_y[:, mw_ind_s[i]])) &
                                   (xy_grid <= np.nanmax(dg_y[:, mw_ind_e[i]])))[0])
        inn_per_mw_t.append(np.where((model_t_hrs > np.nanmin(dg_t_hrs[:, mw_ind_s[i]])) &
                                      (model_t_hrs <= np.nanmax(dg_t_hrs[:, mw_ind_e[i]])))[0])
    else:  # W
        inn_per_mw.append(np.where((xy_grid >= dg_y[0, mw_ind_s[i]]) & (xy_grid <= dg_y[0, mw_ind_e[i]]))[0])
        inn_per_mw_t.append(np.where((model_t_hrs >= dg_t_hrs[0, mw_ind_s[i]]) & (model_t_hrs <= dg_t_hrs[0, mw_ind_e[i]]))[0])

# density interpolated to each depth at 6hr increments should go through 48 hr period?
# t_steps = [t_near, t_near + 6, t_near + 12, t_near + 18, t_near + 24, t_near + 30, t_near + 36, t_near + 42, t_near + 48]
# dens_at_z = np.nan * np.ones((len(dg_z), np.shape(data_interp)[1], len(t_steps)))
# for i in range(np.shape(data_interp)[1]):  # loop over each horizontal grid point
#     for j in range(len(dg_z)):  # loop over each dep
#         for k in range(6):  # loop over each 6 hr increment
#             dens_at_z[j, i, k] = np.interp(deps[j], -1. * dg_z, data_interp[:, i, t_steps[k]])

# vertical shear difference between model and glider at all depths
model_isop_slope_all = np.nan * np.ones((len(dg_z), 2, np.shape(shear)[1]))
den_var = np.nan * np.ones((len(dg_z), np.shape(shear)[1]))
# model_mean_isop_all = np.nan * np.ones((len(dg_z), len(xy_grid[inn]), np.shape(shear)[1]))
for i in range(np.shape(shear)[1]):  # loop over each horizontal profile location
    for j in range(len(dg_z)):  # loop over each depth
        if np.sum(np.isnan(np.nanmean(data_interp[j, inn_per_mw[i], :], axis=1))) < 1:
            model_isop_slope_all[j, :, i] = np.polyfit(xy_grid[inn_per_mw[i]],
                                                       np.nanmean(data_interp[j, inn_per_mw[i][0]:inn_per_mw[i][-1]+1,
                                                                  inn_per_mw_t[i][0]:inn_per_mw_t[i][-1]+1], axis=1), 1)
            model_mean_isop_all = np.polyval(model_isop_slope_all[j, :, i], xy_grid[inn_per_mw[i]])
            den_in = data_interp[j, inn_per_mw[i][0]:inn_per_mw[i][-1]+1,
                                                                  inn_per_mw_t[i][0]:inn_per_mw_t[i][-1]+1]
            signal_var = np.nanvar(model_mean_isop_all)
            vary = np.nan * np.ones(len(xy_grid))
            for l in range(len(xy_grid[inn_per_mw[i]])):
                vary[l] = np.nanvar(den_in[l, :] - model_mean_isop_all[l])  # variance in time of iso about model mean
            den_var[j, i] = np.nanmean(vary)/signal_var

# difference in shear between glider and model
shear_error = np.abs(100. * (shear - (-g * model_isop_slope_all[:, 0, :]/(rho0 * ff))) / (-g * model_isop_slope_all[:, 0, :]/(rho0 * ff)))
# f, ax = plt.subplots()
# for i in range(len(dg_z)):
#     ax.scatter(shear_error[i, :], dg_z[i] * np.ones(len(shear_error[i, :])), s=2, color='b')
# # filtered_error = savgol_filter(np.nanmean(np.log10(shear_error), axis=1), 5, 3)
# ax.plot(np.nanmean(shear_error, axis=1), dg_z, color='k', linewidth=2)
# ax.set_xscale('log')
# ax.set_xlim([1, 10**4])
# plot_pro(ax)
# ------------

# density gradient computation for 4 depths for plot
inn = np.where((xy_grid >= dg_y[0,0]) & (xy_grid <= dg_y[0, 3]))[0]
dg_isop_slope = np.nan * np.ones((len(deps), 2))
dg_mean_isop = np.nan * np.ones((len(deps), len(dg_isop_xy[0, :])))
model_isop_slope = np.nan * np.ones((len(deps), 2))
model_mean_isop = np.nan * np.ones((len(deps), len(xy_grid[inn])))
for i in range(4):
    dg_isop_slope[i, :] = np.polyfit(dg_isop_xy[i, :], dg_isop_dep[i, :], 1)
    dg_mean_isop[i, :] = np.polyval(dg_isop_slope[i, :], dg_isop_xy[i, :])

    model_isop_slope[i, :] = np.polyfit(xy_grid[inn], np.nanmean(isop_dep[i, inn, :], axis=1), 1)
    model_mean_isop[i, :] = np.polyval(model_isop_slope[i, :], xy_grid[inn])

cmaps = [0.15, 0.22, 0.3, 0.37, 0.45, 0.52, 0.6, 0.67, 0.75, 0.82, .9, .98]
lab_y = [26.66, 27.53, 27.74, 27.92]
matplotlib.rcParams['figure.figsize'] = (10, 6)
f, ax = plt.subplots(4, 1, sharex=True)
for i in range(len(deps)):
    ax[i].set_facecolor('#DCDCDC')
    for j in range(6):
        ax[i].plot(xy_grid/1000, isop_dep[i, :, j], color=cmap(cmaps[j]))
    ax[i].plot(xy_grid/1000, np.nanmean(isop_dep[i, :, :], axis=1), color='r', linestyle='--')
    ax[i].scatter(dg_isop_xy[i, :]/1000, dg_isop_dep[i, :], s=40, color='k', zorder=10)

    ax[i].plot([dg_y[0, 0]/1000, dg_y[0, 0]/1000], [20, 30], linestyle='--', color='k')
    ax[i].plot([dg_y[0, 1]/1000, dg_y[0, 1]/1000], [20, 30], linestyle='--', color='k')
    ax[i].plot([dg_y[0, 3]/1000, dg_y[0, 3]/1000], [20, 30], linestyle='--', color='k')
    ax[i].text(10, lab_y[i], str(deps[i]) + 'm', fontweight='bold')
    ax[i].plot(xy_grid[inn] / 1000, model_mean_isop[i, :], color='#FF8C00', linewidth=2)
    ax[i].plot(dg_isop_xy[i, :] / 1000, dg_mean_isop[i, :], color='m', linewidth=2)
    ax[i].text(h_max - 45,
               lab_y[i], r'du$_g$/dz error = ' + str(np.round(100.*np.abs((dg_isop_slope[i, 0] - model_isop_slope[i, 0])/model_isop_slope[i, 0]), 0)) + '%', fontweight='bold')

ax[i].plot(xy_grid/1000, np.nanmean(isop_dep[i, :, :], axis=1), color='r', linestyle='--', linewidth=1.3, label='72hr. avg. density ')
handles, labels = ax[3].get_legend_handles_labels()
ax[3].legend(handles, labels, fontsize=12)
ax[0].set_ylabel('Neutral Density')
ax[0].set_title(r'Density along z=z$_i$')
ax[0].set_ylim([26.6, 26.9])
ax[0].invert_yaxis()
ax[0].grid()
ax[0].set_xlim([0, h_max])
ax[1].set_ylabel('Neutral Density')
ax[1].set_ylim([27.5, 27.6])
ax[1].invert_yaxis()
ax[1].grid()
ax[2].set_ylabel('Neutral Density')
ax[2].set_ylim([27.72, 27.82])
ax[2].invert_yaxis()
ax[2].grid()
ax[3].set_ylabel('Neutral Density')
ax[3].set_ylim([27.9, 28])
ax[3].invert_yaxis()
ax[3].set_xlabel('Transect Distance [km]')
plot_pro(ax[3])
# --------------------------------------------------------------------------------------------------------------------
# --- Background density
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Model/background_density.pkl', 'rb')
MOD = pickle.load(pkl_file)
pkl_file.close()
overlap = np.where((xy_grid > dg_y[0, 0]) & (xy_grid < dg_y[0, -1]))[0]
bck_sa = np.flipud(MOD['sa_back'][:][:, overlap])
bck_ct = np.flipud(MOD['ct_back'][:][:, overlap])
z_back = np.flipud(z_grid)
z_bm = np.where(z_back <= dg_z)[0]

# or if no big feature is present, avg current profiles for background
N2_bck_out = gsw.Nsquared(np.nanmean(bck_sa[0:z_bm[-1]+1, :], axis=1), np.nanmean(bck_ct[0:z_bm[-1]+1, :], axis=1),
                          np.flipud(p_grid)[0:z_bm[-1]+1], lat=ref_lat)[0]
sig0_bck_out = np.nanmean(avg_sig_pd, axis=1)  # avg of glider profiles over this time period

sig0_bck_out = sig0_bck_out[~np.isnan(sig0_bck_out)]
N2_bck_out = N2_bck_out[~np.isnan(sig0_bck_out)]
for i in range(len(N2_bck_out)-7, len(N2_bck_out)):
    N2_bck_out[i] = N2_bck_out[i - 1] - 1*10**(-8)
z_back = z_back[~np.isnan(sig0_bck_out)]
# ---------------------------------------------------------------------------------------------------------------------
# -- MODES ------------------------------------------------------------------------------------------------------------
# # --- MODE PARAMETERS
# frequency zeroed for geostrophic modes
omega = 0
# highest baroclinic mode to be calculated
mmax = 25
nmodes = mmax + 1
# maximum allowed deep shear [m/s/km]
deep_shr_max = 0.1
# minimum depth for which shear is limited [m]
deep_shr_max_dep = 2000.0
# fit limits
eta_fit_dep_min = 200.0
eta_fit_dep_max = 2500.0

# adjust z_grid and N2 profile such that N2=0 at surface and that z_grid min = -2500
# match background z_grid and N2 to current data
z_grid_inter = z_back[0:-1] + (z_back[1:] - z_back[0:-1])/2
z_grid_n2 = np.concatenate((np.array([0]), z_back))
avg_N2 = np.concatenate((np.array([0]), N2_bck_out))
avg_sig0 = np.concatenate((np.array([sig0_bck_out[0] - 0.01]), sig0_bck_out))
# avg_sig0 = sig0_bck_out

# need ddz profile to compute eta
ddz_avg_sig0 = np.nan * np.ones(np.shape(avg_sig0))
ddz_avg_sig0[0] = (avg_sig0[1] - avg_sig0[0])/(z_grid_n2[1] - z_grid_n2[0])
ddz_avg_sig0[-1] = (avg_sig0[-1] - avg_sig0[-2])/(z_grid_n2[-1] - z_grid_n2[-2])
ddz_avg_sig0[1:-1] = (avg_sig0[2:] - avg_sig0[0:-2]) / (z_grid_n2[2:] - z_grid_n2[0:-2])

# # --- compute vertical mode shapes
G, Gz, c, epsilon = vertical_modes(avg_N2, -1.0 * z_grid_n2, omega, mmax)  # N2
#
print('Computed Vertical Modes from Background Profiles')


# DG ETA
# -- eta from individual profiles
eta = np.nan * np.ones(np.shape(dg_sig0))
AG_dg = np.zeros((nmodes, num_profs))
PE_per_mass_dg = np.nan * np.ones((nmodes, num_profs))
eta_m_dg = np.nan * np.ones((len(dg_z), num_profs))
Neta_m_dg = np.nan * np.ones((len(dg_z), num_profs))
for i in range(num_profs):
    good = np.where(~np.isnan(dg_sig0[:, i]))[0]
    avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_sig0)
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

    # AG_dg_0, eta_m_dg_0, Neta_m_dg_0, PE_per_mass_dg_0 = eta_fit(1, -1.0 * dg_z[good], nmodes, avg_N2_match, G, c,
    #                                                              np.transpose(np.atleast_2d(eta[good, i])),
    #                                                              eta_fit_dep_min, eta_fit_dep_max)


G, Gz, c, epsilon = vertical_modes(avg_N2, -1.0 * z_grid_n2, omega, mmax)  # N2
# ---------------------------------------------------------------------------------------------------------------------

# -- eta avg across four profiles
eta_sm = np.nan * np.ones(np.shape(avg_sig_pd))
AG_dg_sm = np.zeros((nmodes, num_profs_eta))
PE_per_mass_dg_sm = np.nan * np.ones((nmodes, num_profs_eta))
eta_m_dg_sm = np.nan * np.ones((len(dg_z), num_profs_eta))
Neta_m_dg_sm = np.nan * np.ones((len(dg_z), num_profs_eta))
for i in range(np.shape(avg_sig_pd)[1]):
    good = np.where(~np.isnan(avg_sig_pd[:, i]))[0]
    avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_sig0)
    ddz_avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), ddz_avg_sig0)
    avg_N2_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_N2)

    eta_sm[good, i] = (avg_sig_pd[good, i] - avg_sig0_match) / ddz_avg_sig0_match

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

        # AG_dg_sm, eta_m_dg_sm, Neta_m_dg_sm, PE_per_mass_dg_sm = eta_fit(np.shape(avg_sig_pd)[1], -1.0 * z_grid_n2,
        #                                                                  nmodes, avg_N2, G, c, eta_sm[0:126, :],
        #                                                                  eta_fit_dep_min, eta_fit_dep_max)

# -- eta Model
in_range = np.where((xy_grid > np.nanmin(dg_y)) & (xy_grid < np.nanmax(dg_y)))[0]
eta_model = np.nan * np.ones(np.shape(sig0_out_s[:, in_range, 0]))
AG_model = np.zeros((nmodes, len(in_range)))
PE_per_mass_model = np.nan * np.ones((nmodes, len(in_range)))
eta_m_model = np.nan * np.ones((len(dg_z), len(in_range)))
Neta_m_model = np.nan * np.ones((len(dg_z), len(in_range)))
for i in range(len(in_range)):
    this_model = np.flipud(np.nanmean(sig0_out_s[:, in_range[i], :], axis=1))

    good = np.where(~np.isnan(this_model))[0]
    avg_sig0_match = np.interp(np.abs(dg_z[good]), np.abs(z_grid_n2), avg_sig0)
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

    # AG_model, eta_m_model, Neta_m_model, PE_per_mass_model = eta_fit(np.shape(eta_model)[1], -1.0 * z_grid_n2,
    #                                                              nmodes, avg_N2, G, c, eta_model[0:126, :],
    #                                                              eta_fit_dep_min, eta_fit_dep_max)


# --- fit velocities ---
# DG
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

# Model
HKE_per_mass_model = np.nan * np.zeros([nmodes, np.shape(eta_model)[1]])
modest = np.arange(11, nmodes)
good_ke_prof = np.ones(len(in_range))
AGz_model = np.zeros([nmodes, len(in_range)])
HKE_noise_threshold = 1e-5  # 1e-5
V_m_model = np.nan * np.zeros((len(dg_z), len(in_range)))
avg_mod_u = np.flipud(np.nanmean(u_out_s[:, in_range, :], axis=2))
for i in range(len(in_range)):
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

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
z_grid_f = z_grid_n2
dg_sig0_f = np.flipud(dg_sig0)

# --- PLOT DENSITY ANOMALIES, ETA, AND VELOCITIES
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
for i in range(np.shape(dg_sig0)[1]):
    ax1.plot(dg_sig0[0:len(avg_sig0)-1, i] - avg_sig0[1:], z_grid_n2[1:])
ax1.grid()
ax1.set_xlim([-.5, .5])
ax1.set_ylim([z_max, 0])
ax1.set_ylabel('Z [m]')
ax1.set_xlabel('Density Anomaly')
ax1.set_title('Density Anomaly')
for i in range(np.shape(eta)[1]):
    ax2.plot(eta[:, i], dg_z, color='#4682B4', linewidth=1.25)
    ax2.plot(eta_m_dg[:, i], dg_z, color='k', linestyle='--', linewidth=.75)
ax2.grid()
ax2.set_xlabel('Isopycnal Displacement [m]')
ax2.set_title('Vertical Displacement')

avg_mod_u = np.flipud(np.nanmean(u_out_s[:, in_range, :], axis=2))
for i in range(np.shape(avg_mod_u)[1]):
    ax3.plot(avg_mod_u[:, i], dg_z, linewidth=0.75, color='#DCDCDC')
    ax3.plot(V_m_model[:, i], dg_z, color='k', linestyle='--', linewidth=.4)
for i in range(np.shape(v_g)[1]):
    ax3.plot(v_g[:, i], dg_z)
    ax3.plot(V_m[:, i], dg_z, color='k', linestyle='--', linewidth=.75)
ax2.set_xlim([-200, 200])
ax3.set_xlim([-.4, .4])
ax3.set_title('Model and DG Velocity (u)')
ax3.set_xlabel('Velocity [m/s]')
plot_pro(ax3)

# ---------------------------------------------------------------------------------------------------------------------
# select only dg profiles with dephts greater than 2000m
max_eta_dep = np.nan * np.ones(np.shape(eta)[1])
for i in range(np.shape(eta)[1]):
    max_eta_dep[i] = dg_z[np.where(~np.isnan(eta[:, i]))[0][-1]]

max_eta_sm_dep = np.nan * np.ones(np.shape(eta)[1])
max_v_dep = np.nan * np.ones(np.shape(eta)[1])
for i in range(np.shape(eta_sm)[1]):
    max_eta_sm_dep[i] = dg_z[np.where(~np.isnan(eta_sm[:, i]))[0][-1]]
    max_v_dep[i] = dg_z[np.where(~np.isnan(v_g[:, i]))[0][-1]]

max_model_dep = np.nan * np.ones(np.shape(PE_per_mass_model)[1])
for i in range(np.shape(PE_per_mass_model)[1]):
    max_model_dep[i] = dg_z[np.where(~np.isnan(avg_mod_u[:, i]))[0][-1]]

good_eta = np.where(max_eta_dep < -2000)[0]
good_sm = np.where(max_eta_sm_dep < -2000)[0]
good_vg = np.where(max_v_dep < -2000)[0]
good_mod = np.where(max_model_dep < -2000)[0]

# --- ENERGY
avg_PE = np.nanmean(PE_per_mass_dg[:, good_eta], axis=1)
avg_PE_smooth = np.nanmean(PE_per_mass_dg_sm[:, good_sm], axis=1)
avg_KE = np.nanmean(HKE_per_mass_dg[:, good_vg], axis=1)
avg_PE_model = np.nanmean(PE_per_mass_model[:, good_mod], axis=1)
avg_KE_model = np.nanmean(HKE_per_mass_model[:, good_mod], axis=1)
dk = ff / c[1]
PE_SD, PE_GM, GMPE, GMKE = PE_Tide_GM(rho0, -1.0 * z_grid_f, nmodes, np.flipud(np.transpose(np.atleast_2d(avg_N2))), ff)

# --- PLOT ENERGY SPECTRA
mm = 10
sc_x = 1000 * ff / c[1:mm]
l_lim = 3 * 10 ** -2
sc_x = np.arange(1, mm)
l_lim = 0.7
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# DG
ax1.plot(sc_x, avg_PE[1:mm] / dk, color='#B22222', label='APE$_{DG_{ind}}$', linewidth=3, linestyle='--')
ax1.scatter(sc_x, avg_PE[1:mm] / dk, color='#B22222', s=20)
ax1.plot(sc_x, avg_PE_smooth[1:mm] / dk, color='#B22222', label='APE$_{DG_{avg}}$', linewidth=3)
ax1.scatter(sc_x, avg_PE_smooth[1:mm] / dk, color='#B22222', s=20)

ax2.plot(sc_x, avg_KE[1:mm] / dk, 'g', label='KE$_{DG}$', linewidth=3)
ax2.scatter(sc_x, avg_KE[1:mm] / dk, color='g', s=20)  # DG KE
# ax2.plot([l_lim, 1000 * ff / c[1]], avg_KE[0:2] / dk, 'g', linewidth=3)  # DG KE_0
# ax2.scatter(l_lim, avg_KE[0] / dk, color='g', s=25, facecolors='none')  # DG KE_0
ax2.plot([l_lim, sc_x[0]], avg_KE[0:2] / dk, 'g', linewidth=3)  # DG KE_0
ax2.scatter(l_lim, avg_KE[0] / dk, color='g', s=25, facecolors='none')  # DG KE_0
# Model
ax1.plot(sc_x, avg_PE_model[1:mm] / dk, color='#FF8C00', label='APE$_{Model}$', linewidth=2)
ax1.scatter(sc_x, avg_PE_model[1:mm] / dk, color='#FF8C00', s=20)
ax2.plot(sc_x, avg_KE_model[1:mm] / dk, color='#FF8C00', label='KE$_{Model}$', linewidth=2)
ax2.scatter(sc_x, avg_KE_model[1:mm] / dk, color='#FF8C00', s=20)
# ax2.plot([l_lim, 1000 * ff / c[1]], avg_KE_model[0:2] / dk, color='#FF8C00', linewidth=2)
# ax2.scatter(l_lim, avg_KE_model[0] / dk, color='g', s=25, facecolors='none')
ax2.plot([l_lim, sc_x[0]], avg_KE_model[0:2] / dk, color='#FF8C00', linewidth=2)
ax2.scatter(l_lim, avg_KE_model[0] / dk, color='g', s=25, facecolors='none')

# test
limm = 5
# ax2.plot(1000 * ff / c[1:limm], (avg_KE_model[1:limm] - GMKE[0:limm - 1]) / dk, 'b', linewidth=4, label='Model - GM')

# GM
# ax1.plot(sc_x, 0.25 * PE_GM / dk, linestyle='--', color='k', linewidth=0.75)
# ax1.plot(1000 * ff / c[1:], 0.25 * GMPE / dk, color='k', linewidth=0.75)
# ax1.text(sc_x[0] - .01, 0.3 * PE_GM[1] / dk, r'$1/4 PE_{GM}$', fontsize=10)
# ax2.plot(1000 * ff / c[1:], 0.25 * GMKE / dk, color='k', linewidth=0.75)
# ax2.text(sc_x[0] - .01, 0.5 * GMKE[1] / dk, r'$1/4 KE_{GM}$', fontsize=10)

ax1.set_xlim([l_lim, 0.2 * 10 ** 2])
ax2.set_xlim([l_lim, 0.2 * 10 ** 2])
ax2.set_ylim([10 ** (-3), 2 * 10 ** 2])
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.grid()
ax2.set_xscale('log')

# ax1.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
ax1.set_xlabel('Mode Number', fontsize=12)
ax1.set_ylabel('Spectral Density', fontsize=12)  # ' (and Hor. Wavenumber)')
ax1.set_title('Potential Energy Spectrum', fontsize=12)
# ax2.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
ax2.set_xlabel('Mode Number', fontsize=12)
ax2.set_title('Kinetic Energy Spectrum', fontsize=12)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=12)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=12)

plot_pro(ax2)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

save_anom = 0
if save_anom:
    my_dict = {'dg_z': dg_z, 'dg_v': v_g, 'model_u_at_mwv': u_mod_at_mw,
               'shear_error': shear_error, 'igw_var': den_var,
               'KE_dg': HKE_per_mass_dg, 'KE_mod': HKE_per_mass_model,
               'eta_m_dg': eta_m_dg, 'PE_dg': PE_per_mass_dg,
               'eta_m_dg_avg': eta_m_dg_sm, 'PE_dg_avg': PE_per_mass_dg_sm,
               'eta_model': eta_m_model, 'PE_model': PE_per_mass_model,
               'glide_slope': dg_glide_slope, 'dg_w': dg_vertical_speed}
    output = open(output_filename, 'wb')
    pickle.dump(my_dict, output)
    output.close()

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# # --- DEPTH OF ISOPYCNALS IN TIME ----------------------------------------------
# # isopycnals i care about
# rho1 = 26.6
# rho2 = 27.2
# rho3 = 27.9
#
# t_s = time_ord_s[0]
# d_time_per_prof = np.nanmean(dg_t, axis=0)
# d_time_per_prof_date = []
# d_dep_rho1 = np.nan * np.ones((3, len(d_time_per_prof)))
# for i in range(len(d_time_per_prof)):
#     d_time_per_prof_date.append(datetime.date.fromordinal(np.int(d_time_per_prof[i])))
#     d_dep_rho1[0, i] = np.interp(rho1, dg_sig0_f[:, i], z_grid_f)
#     d_dep_rho1[1, i] = np.interp(rho2, dg_sig0_f[:, i], z_grid_f)
#     d_dep_rho1[2, i] = np.interp(rho3, dg_sig0_f[:, i], z_grid_f)
# mw_time_ordered = mw_time
# mw_sig_ordered = np.flipud(avg_sig_pd)
# mw_time_date = []
# mw_dep_rho1 = np.nan * np.ones((3, len(mw_time_ordered)))
# for i in range(len(mw_time)):
#     mw_time_date.append(datetime.date.fromordinal(np.int(np.round(mw_time_ordered[i]))))
#     mw_dep_rho1[0, i] = np.interp(rho1, mw_sig_ordered[:, i], z_grid_f)
#     mw_dep_rho1[1, i] = np.interp(rho2, mw_sig_ordered[:, i], z_grid_f)
#     mw_dep_rho1[2, i] = np.interp(rho3, mw_sig_ordered[:, i], z_grid_f)
#
#
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# ax1.scatter(24*(d_time_per_prof - t_s), d_dep_rho1[0, :], color='g', s=15, label=r'DG$_{ind}$')
# ax2.scatter(24*(d_time_per_prof - t_s), d_dep_rho1[1, :], color='g', s=15)
# ax3.scatter(24*(d_time_per_prof - t_s), d_dep_rho1[2, :], color='g', s=15)
#
# ax1.plot(24*(mw_time - t_s), mw_dep_rho1[0, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')
# ax1.scatter(24*(mw_time - t_s), mw_dep_rho1[0, :], color='b', s=6)
# ax2.plot(24*(mw_time - t_s), mw_dep_rho1[1, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')
# ax2.scatter(24*(mw_time - t_s), mw_dep_rho1[1, :], color='b', s=6)
# ax3.plot(24*(mw_time - t_s), mw_dep_rho1[2, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')
# ax3.scatter(24*(mw_time - t_s), mw_dep_rho1[2, :], color='b', s=6)
#
# ax1.set_title('Depth of $\gamma^{n}$ = ' + str(rho1))
# ax2.set_title('Depth of $\gamma^{n}$ = ' + str(rho2))
# ax3.set_title('Depth of $\gamma^{n}$ = ' + str(rho3))
# ax1.set_ylabel('Depth [m]')
# ax2.set_ylabel('Depth [m]')
# ax3.set_ylabel('Depth [m]')
# ax3.set_xlabel('Time [Hour]')
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(handles, labels, fontsize=10)
# ax1.set_ylim([-350, -150])
# ax2.set_ylim([-750, -450])
# ax3.set_ylim([-2100, -1800])
# ax1.grid()
# ax2.grid()
# plot_pro(ax3)
#
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# for i in range(len(time_ord_s)):
#     ax1.contour(y_grid/1000, z_grid, sig0_out_s[:, :, i], levels=[rho1])
#     ax2.contour(y_grid/1000, z_grid, sig0_out_s[:, :, i], levels=[rho2])
#     ax3.contour(y_grid/1000, z_grid, sig0_out_s[:, :, i], levels=[rho3])
# ax1.set_ylim([-350, -150])
# ax2.set_ylim([-750, -550])
# ax3.set_ylim([-2050, -1850])
# ax1.set_ylabel('Z [m]')
# ax2.set_ylabel('Z [m]')
# ax3.set_ylabel('Z [m]')
# ax1.set_title('Time Variability of Isopycnal Depth; ' + '$\gamma^{n}$ = ' + str(rho1))
# ax2.set_title('Depth of $\gamma^{n}$ = ' + str(rho2))
# ax3.set_title('Depth of $\gamma^{n}$ = ' + str(rho3))
# ax3.set_xlabel('Domain [km]')
# ax1.grid()
# ax2.grid()
# plot_pro(ax3)
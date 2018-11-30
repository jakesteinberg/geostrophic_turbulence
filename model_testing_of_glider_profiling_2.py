import numpy as np
import pickle
import glob
import datetime
import gsw
import time as TT
from scipy.integrate import cumtrapz
from mode_decompositions import eta_fit, vertical_modes, PE_Tide_GM, vertical_modes_f
# -- plotting
import matplotlib.pyplot as plt
from toolkit import plot_pro

# ---- LOAD PROCESSED FILES ---------------------------------------------------------------------------------------
# file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/n_s_extraction/*.pkl')

file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/n_s_extraction/pickle_protocol_2/*.pkl')
# file_name = open('/Users/jake/Documents/baroclinic_modes/Model/test_extraction_1541203200.pkl', 'rb')

file_name = open(file_list[0], 'rb')
D = pickle.load(file_name)
file_name.close()
y_grid = np.arange(0, 125000, 500)
z_grid = np.concatenate((np.arange(-2600, -1000, 20), np.arange(-1000, 0, 10)))
ref_lat = np.nanmin(D['lat_rho'][:])
ref_lon = np.nanmin(D['lon_rho'][:])
p_out = gsw.p_from_z(z_grid, ref_lat)
y = 1852. * 60. * (D['lat_rho'][:] - ref_lat)
lat_yy = np.interp(y_grid, y, D['lat_rho'][:])
lon_yy = D['lon_rho'][:][0] * np.ones(len(y_grid))

# -- LOOP OVER FILES to collect
num_files = len(file_list)
sig0_raw = np.nan * np.ones((len(D['z']), len(y), num_files))
z_raw = np.nan * np.ones((len(D['z']), len(y), num_files))
t_out = np.nan * np.ones((len(z_grid), len(y_grid), num_files))
s_out = np.nan * np.ones((len(z_grid), len(y_grid), num_files))
sig0_out = np.nan * np.ones((len(z_grid), len(y_grid), num_files))
u_out = np.nan * np.ones((len(z_grid), len(y_grid), num_files))
time_out = np.nan * np.ones(num_files)
time_ord = np.nan * np.ones(num_files)
date_out = np.nan * np.ones((num_files, 2))
for m in range(num_files):
    file_name = open(file_list[m], 'rb')
    D = pickle.load(file_name)
    file_name.close()

    time = D['ocean_time']
    u = D['u']
    v = D['v']
    t = D['temp']
    s = D['salt']
    lon_rho = D['lon_rho'][:]
    lat_rho = D['lat_rho'][:]
    lon_u = D['lon_u'][:]
    lat_u = D['lat_u'][:]
    lon_v = D['lon_v'][:]
    lat_v = D['lat_v'][:]
    z = D['z'][:]

    # orient for either E/W or N/S section
    vel = u

    # -- TIME
    time_cor = time/(60*60*24) + 719163  # correct to start (1970)
    time_hour = 24. * (time_cor - np.int(time_cor))
    date_time = datetime.date.fromordinal(np.int(time_cor))

    # # ---- SAVE OUTPUT for old PYTHON/PICKLE ------------------------------------------------------------------------
    # date_str_out = str(date_time.year) + '_' + str(date_time.month) + '_' + str(date_time.day) + \
    #                '_' + str(np.int(np.round(time_hour, 1)))
    # my_dict = {'ocean_time': time, 'temp': t, 'salt': s, 'u': u, 'v': v, 'lon_rho': lon_rho, 'lat_rho': lat_rho,
    #            'lon_u': lon_u, 'lat_u': lat_u, 'lon_v': lon_v, 'lat_v': lat_v, 'z': z}
    # output = open('/Users/jake/Documents/baroclinic_modes/Model/n_s_extraction/pickle_protocol_2/N_S_extraction_' +
    #               date_str_out + '.pkl', 'wb')
    # pickle.dump(my_dict, output, protocol=2)
    # output.close()

    # -- INTERPOLATE TO REGULAR Y
    y = 1852. * 60. * (lat_rho - ref_lat)
    # y = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_rho - ref_lon)

    # -- GSW
    sig0_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    u_g_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    t_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    s_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    # loop over each column and interpolate to z
    p = gsw.p_from_z(z, ref_lat * np.ones(np.shape(z)))
    SA = gsw.SA_from_SP(s, p, lon_rho[0] * np.ones(np.shape(s)), np.tile(lat_rho, (np.shape(p)[0], 1)))
    CT = gsw.CT_from_t(s, t, p)
    sig0_0 = gsw.sigma0(SA, CT)
    sig0_raw[:, :, m] = sig0_0.copy()
    z_raw[:, :, m] = z.copy()
    for i in range(len(lat_rho)):
        # p = gsw.p_from_z(z[:, i], ref_lat * np.ones(len(z)))
        # SA = gsw.SA_from_SP(s[:, i], p, lon_rho[0] * np.ones(len(s[:, i])), lat_rho[i] * np.ones(len(s[:, i])))
        # CT = gsw.CT_from_t(s[:, i], t[:, i], p)
        # sig0_0 = gsw.sigma0(SA, CT)
        # -- INTERPOLATE TO REGULAR Z GRID
        t_p[:, i] = np.interp(z_grid, z[:, i], t[:, i])
        s_p[:, i] = np.interp(z_grid, z[:, i], s[:, i])
        sig0_p[:, i] = np.interp(z_grid, z[:, i], sig0_0[:, i])
        u_g_p[:, i] = np.interp(z_grid, z[:, i], vel[:, i])

    # loop over each row and interpolate to Y grid
    tt = np.nan * np.ones((len(z_grid), len(y_grid)))
    ss = np.nan * np.ones((len(z_grid), len(y_grid)))
    sig0 = np.nan * np.ones((len(z_grid), len(y_grid)))
    u_g = np.nan * np.ones((len(z_grid), len(y_grid)))
    yg_low = np.where(y_grid > np.round(np.nanmin(y)))[0][0] - 1
    yg_up = np.where(y_grid < np.round(np.nanmax(y)))[0][-1] + 1
    for i in range(len(z_grid)):
        tt[i, yg_low:yg_up] = np.interp(y_grid[yg_low:yg_up], y, t_p[i, :])
        ss[i, yg_low:yg_up] = np.interp(y_grid[yg_low:yg_up], y, s_p[i, :])
        sig0[i, yg_low:yg_up] = np.interp(y_grid[yg_low:yg_up], y, sig0_p[i, :])
        u_g[i, yg_low:yg_up] = np.interp(y_grid[yg_low:yg_up], y, u_g_p[i, :])

    t_out[:, :, m] = tt.copy()
    s_out[:, :, m] = ss.copy()
    sig0_out[:, :, m] = sig0.copy()
    u_out[:, :, m] = u_g.copy()
    time_ord[m] = time_cor.copy()
    time_out[m] = time.copy()
    date_out[m, :] = np.array([np.int(time_cor), time_hour])
    print('loaded, gridded, computed density of file = ' + str(m))
    # -- END LOOP COLLECTING RESULT OF EACH TIME STEP

# SORT TIME STEPS BY TIME
sig0_raw = sig0_raw[:, :, np.argsort(time_out)]
z_raw = z_raw[:, :, np.argsort(time_out)]
t_out_s = t_out[:, :, np.argsort(time_out)]
s_out_s = s_out[:, :, np.argsort(time_out)]
sig0_out_s = sig0_out[:, :, np.argsort(time_out)]
u_out_s = u_out[:, :, np.argsort(time_out)]
time_out_s = time_out[np.argsort(time_out)]
time_ord_s = time_ord[np.argsort(time_out)]
date_out_s = date_out[np.argsort(time_out), :]

# --- convert in matlab
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/')
eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/library/')
gamma = np.nan * np.ones(np.shape(sig0_out_s))
print('Opened Matlab')
for j in range(2):
    tic = TT.clock()
    for i in range(len(y_grid)):
        gamma[:, i, j] = np.squeeze(np.array(eng.eos80_legacy_gamma_n(matlab.double(s_out_s[:, i, j].tolist()),
                                                                      matlab.double(t_out_s[:, i, j].tolist()),
                                                                      matlab.double(p_out.tolist()),
                                                                      matlab.double([lon_yy[0]]),
                                                                      matlab.double([lat_yy[i]]))))
    toc = TT.clock()
    print('Time step = ' + str(j) + ' = '+ str(toc - tic) + 's')
eng.quit()

my_dict = {'z': z_grid, 'gamma': gamma, 'sig0': sig0_out_s, 'u': u_out_s, 'time': time_out_s, 'time_ord': time_ord_s,
           'date': date_out_s, 'raw_sig0': sig0_raw, 'raw_z': z_raw}
output = open('/Users/jake/Documents/baroclinic_modes/Model/extracted_gridded_gamma.pkl', 'wb')
pickle.dump(my_dict, output)
output.close()

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# CREATE LOCATIONS OF FAKE GLIDER DIVE
# approximate glider as flying with vertical velocity of 0.08 m/s, glide slope of 1:3
# assume glider measurements are made every 10, 15, 30, 45, 120 second intervals
# time between each model time step is 1hr.

dg_vertical_speed = .14  # m/s
dg_glide_slope = 3
num_dives = 7

y_dg_s = 0
z_dg_s = 0
dg_z = np.flipud(z_grid.copy())
dg_z_g = np.tile(np.concatenate((np.flipud(z_grid.copy())[:, None], np.flipud(z_grid.copy())[:, None]), axis=1),
                 (1, num_dives))
dg_y = np.nan * np.ones((len(dg_z), num_dives*2))
dg_y[0, 0] = y_dg_s
for i in range(1, len(dg_z)):
    dg_y[i, 0] =  dg_glide_slope * (dg_z[i - 1] - dg_z[i]) + dg_y[i - 1, 0]
dg_y[-1, 1] = dg_y[-1, 0]
for i in range(len(dg_z) - 2, 0, -1):
    dg_y[i, 1] = dg_glide_slope * (dg_z[i] - dg_z[i + 1]) + dg_y[i + 1, 1]
dg_y[0, 1] = dg_glide_slope * (dg_z[0] - dg_z[1]) + dg_y[1, 1]

for i in range(2, num_dives*2, 2):
    dg_y[:, i] = (dg_y[:, 0] - dg_y[0, 0]) + dg_y[0, i - 1] + 10
    dg_y[:, i + 1] = (dg_y[:, 1] - dg_y[-1, 1]) + dg_y[-1, i]

# DG time
dg_t = np.nan * np.ones(dg_y.shape)
dg_t[0, 0] = np.nanmin(time_ord_s)
for j in range(np.shape(dg_y)[1]):
    # climb portion
    if np.mod(j, 2):
        # loop over z's as the glider climbs
        dg_t[-1, j] = dg_t[-1, j - 1] + 10 * (1/(60*60*24))
        for i in range(len(z_grid) - 2, 0, -1):
            dg_t[i, j] = dg_t[i + 1, j] + (np.abs(dg_z_g[i, j] - dg_z_g[i - 1, j]) / dg_vertical_speed) / (60 * 60 * 24)
        dg_t[0, j] = dg_t[1, j] + (np.abs(dg_z_g[0, j] - dg_z_g[1, j]) / dg_vertical_speed) / (60 * 60 * 24)
    # dive portion
    else:
        if j < 1:
            dg_t[0, j] = dg_t[0, 0]
        else:
            dg_t[0, j] = dg_t[0, j - 1]
        # loop over z's as the glider dives
        for i in range(1, len(z_grid)):
            dg_t[i, j] = dg_t[i - 1, j] + (np.abs(dg_z_g[i, j] - dg_z_g[i - 1, j]) / dg_vertical_speed) / (60 * 60 * 24)

# f, ax = plt.subplots()
# ax.scatter(np.arange(0, len(z_grid)), dg_t[:, 2], color='r')
# ax.scatter(np.arange(len(z_grid), 2*len(z_grid)), np.flipud(dg_t[:, 3]), color='b')
# plot_pro(ax)

# --- Interpolation of Model to each glider measurements
dg_sig0 = np.nan * np.ones(np.shape(dg_t))
for i in range(np.shape(dg_y)[1]):
    for j in range(np.shape(dg_y)[0]):
        if (i < 1) & (j < 1):
            dg_sig0[j, i] = np.interp(dg_y[j, i], y_grid, sig0_out_s[j, :, 0])
        else:
            this_t = dg_t[j, i]
            # find time bounding model runs
            nearest_model_t_over = np.where(time_ord_s > this_t)[0][0]
            # print(this_t - time_ord_s[nearest_model_t_over])
            # interpolate to hor position of glider dive for time before and after
            sig_t_before = np.interp(dg_y[j, i], y_grid, sig0_out_s[j, :, nearest_model_t_over - 1])
            sig_t_after = np.interp(dg_y[j, i], y_grid, sig0_out_s[j, :, nearest_model_t_over])
            # interpolate across time
            dg_sig0[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                   [sig_t_before, sig_t_after])

# --- Estimate of DAC
dg_dac = np.nan * np.ones(np.shape(dg_sig0)[1] - 1)
for i in range(np.shape(dg_y)[1] - 1):
    min_t = np.nanmin(dg_t[:, i:i + 2])
    max_t = np.nanmax(dg_t[:, i:i + 2])
    # mto = np.where((time_ord_s >= min_t) & (time_ord_s <= max_t))[0]
    mtun = np.where(time_ord_s <= min_t)[0][-1]
    mtov = np.where(time_ord_s > max_t)[0][0]

    this_ys = np.nanmin(dg_y[:, i:i + 2])
    this_ye = np.nanmax(dg_y[:, i:i + 2])
    this_ygs = np.where(y_grid <= this_ys)[0][-1]
    this_yge = np.where(y_grid >= this_ye)[0][0]
    dg_dac[i] = -1 * np.nanmean(np.nanmean(u_out_s[:, this_ygs:this_yge, mtun:mtov], axis=2))

# ---------------------------------------------------------------------------------------------------------------------
# --- M/W estimation
g = 9.81
rho0 = 1025.0
ff = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)  # Coriolis parameter [s^-1]
sigth_levels = np.array([26, 26.2, 26.4, 26.6, 26.8, 27, 27.2, 27.4, 27.6, 27.7])
num_profs = np.shape(dg_sig0)[1]
order_set = np.arange(0, np.shape(dg_sig0)[1], 2)

mw_y = np.nan * np.zeros(num_profs - 1)
sigma_theta_out = np.nan * np.zeros((np.size(z_grid), num_profs - 1))
shear = np.nan * np.zeros((np.size(z_grid), num_profs - 1))
avg_sig_pd = np.nan * np.zeros((np.size(z_grid), num_profs - 1))
isopycdep = np.nan * np.zeros((np.size(sigth_levels), num_profs))
isopycx = np.nan * np.zeros((np.size(sigth_levels), num_profs))
for i in order_set:
    sigma_theta_pa_M = np.nan * np.zeros(np.size(z_grid))
    sigma_theta_pa_W = np.nan * np.zeros(np.size(z_grid))
    shearM = np.nan * np.zeros(np.size(z_grid))
    shearW = np.nan * np.zeros(np.size(z_grid))
    p_avg_sig_M = np.nan * np.zeros(np.size(z_grid))
    p_avg_sig_W = np.nan * np.zeros(np.size(z_grid))
    yy_M = np.nan * np.zeros(np.size(z_grid))
    yy_W = np.nan * np.zeros(np.size(z_grid))
    # LOOP OVER EACH BIN_DEPTH
    for j in range(np.size(z_grid)):
        # find array of indices for M / W sampling
        if i < 2:
            c_i_m = np.arange(i, i + 3)
            # c_i_m = []  # omit partial "M" estimate
            c_i_w = np.arange(i, i + 4)
        elif (i >= 2) and (i < num_profs - 2):
            c_i_m = np.arange(i - 1, i + 3)
            c_i_w = np.arange(i, i + 4)
        elif i >= num_profs - 2:
            c_i_m = np.arange(i - 1, num_profs)
            # c_i_m = []  # omit partial "M" estimated
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

    # OUTPUT FOR EACH TRANSECT (at least 2 DIVES)
    # because this is M/W profiling, for a 3 dive transect, only 5 profiles of shear and eta are compiled
    sigma_theta_out[:, i] = sigma_theta_pa_M
    shear[:, i] = shearM
    avg_sig_pd[:, i] = p_avg_sig_M
    mw_y[i] = dg_y[-1, i]

    if i < num_profs - 2:
        sigma_theta_out[:, i + 1] = sigma_theta_pa_W
        shear[:, i + 1] = shearW
        avg_sig_pd[:, i + 1] = p_avg_sig_W
        mw_y[i + 1] = dg_y[0, i + 1]

    # ISOPYCNAL DEPTHS ON PROFILES ALONG EACH TRANSECT
    sigthmin = np.nanmin(np.array(dg_sig0[:, i]))
    sigthmax = np.nanmax(np.array(dg_sig0[:, i]))
    isigth = (sigth_levels > sigthmin) & (sigth_levels < sigthmax)
    isopycdep[isigth, i] = np.interp(sigth_levels[isigth], np.flipud(dg_sig0[:, i]), np.flipud(z_grid))
    isopycx[isigth, i] = np.interp(sigth_levels[isigth], np.flipud(dg_sig0[:, i]), dg_y[:, i])

    sigthmin = np.nanmin(np.array(dg_sig0[:, i + 1]))
    sigthmax = np.nanmax(np.array(dg_sig0[:, i + 1]))
    isigth = (sigth_levels > sigthmin) & (sigth_levels < sigthmax)
    isopycdep[isigth, i + 1] = np.interp(sigth_levels[isigth], np.flipud(dg_sig0[:, i + 1]), np.flipud(z_grid))
    isopycx[isigth, i + 1] = np.interp(sigth_levels[isigth], np.flipud(dg_sig0[:, i + 1]), dg_y[:, i + 1])

# FOR EACH TRANSECT COMPUTE GEOSTROPHIC VELOCITY
vbc_g = np.nan * np.zeros(np.shape(shear))
v_g = np.nan * np.zeros((np.size(z_grid), num_profs))
for m in range(num_profs - 1):
    iq = np.where(~np.isnan(shear[:, m]))
    if np.size(iq) > 10:
        z2 = z_grid[iq]
        vrel = cumtrapz(shear[iq, m], x=z2, initial=0)
        vrel_av = np.trapz(vrel / (z2[-1] - z2[0]), x=z2)
        vbc = vrel - vrel_av
        vbc_g[iq, m] = vbc
        v_g[iq, m] = dg_dac[m] + vbc
    else:
        vbc_g[iq, m] = np.nan
        v_g[iq, m] = np.nan

v_g = v_g[:, 0:-1]

# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --- PLOTTING
# f, ax = plt.subplots()
# ax.contourf(dg_y, np.flipud(np.tile(z_grid[:, None], (1, np.shape(dg_y)[1]))), np.flipud(dg_sig0),levels=sigth_levels)
# plot_pro(ax)

h_max = 125  # horizontal domain limit
u_levels = np.array([-.4, -.35, -.3, -.25, - .2, -.15, -.125, -.1, -.075, -.05, -0.025, 0,
                     0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
u_mod_at_mw_y = np.nan * np.ones(np.shape(v_g))
mw_time = np.nan * np.ones(len(mw_y))
for i in range(len(mw_y)):
    mean_t = np.nanmean(dg_t[:, i:i + 2])
    nearest_t = np.where(time_ord_s < mean_t)[0][-1]
    mw_time[i] = np.nanmean(time_ord_s[nearest_t:nearest_t + 2])
    avg_u_at = np.nanmean(u_out_s[:, :, nearest_t:nearest_t + 2], axis=2)
    for j in range(len(z_grid)):
        u_mod_at_mw_y[j, i] = np.interp(mw_y[i], y_grid, avg_u_at[j, :])

plot0 = 1
if plot0 > 0:
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.contourf(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                 np.nanmean(u_out_s, axis=2), levels=u_levels)
    uvc = ax1.contour(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                      np.nanmean(u_out_s, axis=2), levels=u_levels, colors='k', linewidth=0.75)
    ax1.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
    rhoc = ax1.contour(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                       np.nanmean(sig0_out_s, axis=2), levels=sigth_levels, colors='#A9A9A9', linewidth=0.35)
    ax1.scatter(dg_y/1000, dg_z_g, 4, color='k')
    for r in range(np.shape(isopycdep)[0]):
        ax1.plot(isopycx[r, :]/1000, isopycdep[r, :], color='r', linewidth=0.45)
    ax1.set_title(str(datetime.date.fromordinal(np.int(time_ord_s[0]))) +
                  ' -- ' + str(datetime.date.fromordinal(np.int(time_ord_s[-2]))) + ', 72hr. Avg.')
    ax1.set_ylim([-2650, 0])
    ax1.set_xlim([0, h_max])
    ax1.set_xlabel('Km')
    ax1.set_ylabel('Z [m]')
    ax1.grid()

    ax2.contourf(np.tile(mw_y/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(mw_y))),
                 -1 * v_g, levels=u_levels)
    uvc = ax2.contour(np.tile(mw_y/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(mw_y))), -1 * v_g,
                      levels=u_levels, colors='k', linewidth=0.75)
    ax2.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
    ax2.scatter(dg_y/1000, dg_z_g, 4, color='k')
    for r in range(np.shape(isopycdep)[0]):
        ax2.plot(isopycx[r, :]/1000, isopycdep[r, :], color='r')
    ax2.set_xlim([0, h_max])
    ax2.set_title('DG Estimated Cross-Track Vel.')
    ax2.set_xlabel('Km')
    ax2.grid()

    ax3.pcolor(np.tile(mw_y/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(mw_y))),
               u_mod_at_mw_y - (-1 * v_g), vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
    uvc = ax3.contour(np.tile(mw_y/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(mw_y))),
                      u_mod_at_mw_y - (-1 * v_g), levels=u_levels, colors='k', linewidth=0.75)
    ax3.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
    ax3.set_title('Velocity Difference')
    ax3.set_xlabel('Km')
    ax3.set_xlim([0, h_max])
    plot_pro(ax3)

plot1 = 1
if plot1 > 0:
    u_anom = u_mod_at_mw_y - (-1 * v_g)
    f, ax = plt.subplots()
    ua_mean = np.nan * np.ones(len(z_grid))
    ua_std = np.nan * np.ones(len(z_grid))
    for i in range(0, len(z_grid), 2):
        ua_mean[i] = np.nanmean(u_anom[i, :])
        ua_std[i] = np.nanstd(u_anom[i, :])
    ax.errorbar(ua_mean, z_grid, xerr=ua_std)
    ax.plot(np.nanmean(u_anom, axis=1), z_grid, color='k', linewidth=1.5)
    ax.set_ylim([-2650, 0])
    ax.set_xlim([-0.15, 0.15])
    ax.set_title('Model V. - DG V.')
    ax.set_xlabel('Velocity Error [m/s]')
    ax.set_ylabel('Z [m]')
    plot_pro(ax)

plot2 = 0
if plot2 > 0:
    ti1 = 23
    ti2 = 46
    ti3 = 69
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    ax1.pcolor(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
               u_out_s[:, :, ti1], vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
    uvc = ax1.contour(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                       u_out_s[:, :, ti1], levels=u_levels, colors='k', linewidth=0.75)
    rhoc = ax1.contour(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                       sig0_out_s[: ,:, ti1], levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

    ax2.pcolor(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
               u_out_s[:, :, ti2], vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
    uvc = ax2.contour(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                       u_out_s[:, :, ti2], levels=u_levels, colors='k', linewidth=0.75)
    rhoc = ax2.contour(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                       sig0_out_s[: ,:, ti2], levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

    ax3.pcolor(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
               u_out_s[:, :, ti3], vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
    uvc = ax3.contour(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                       u_out_s[:, :, ti3], levels=u_levels, colors='k', linewidth=0.75)
    rhoc = ax3.contour(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                       sig0_out_s[: ,:, ti3], levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

    ax4.pcolor(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
               np.nanmean(u_out_s, 2), vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
    uvc = ax4.contour(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                       np.nanmean(u_out_s, axis=2), levels=u_levels, colors='k', linewidth=0.75)
    rhoc = ax4.contour(np.tile(y_grid/1000, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                       np.nanmean(sig0_out_s, axis=2), levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

    ax1.set_xlim([0, h_max])
    ax1.set_ylim([-2650, 0])
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
# --- Background density
# file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/time_avg/*.pkl')

# --- TOGGLE TO RECOMPUTE, OTHERWISE LOAD BELOW
# file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/time_avg/pickle_protocol_2/*.pkl')
# # file_name = open('/Users/jake/Documents/baroclinic_modes/Model/test_extraction_1541203200.pkl', 'rb')
#
# sig0_bck_out = np.nan * np.ones((len(z_grid), len(y_grid), len(file_list)))
# t_bck_out = np.nan * np.ones((len(z_grid), len(y_grid), len(file_list)))
# s_bck_out = np.nan * np.ones((len(z_grid), len(y_grid), len(file_list)))
# N2_bck_out = np.nan * np.ones((len(z_grid), len(y_grid), len(file_list)))
# date_out = np.nan * np.ones((num_files, 2))
# for m in range(len(file_list)):
#     file_name = open(file_list[m], 'rb')
#     D = pickle.load(file_name)
#     file_name.close()
#
#     time = D['ocean_time']
#     t = D['temp']
#     s = D['salt']
#     lon_rho = D['lon_rho'][:]
#     lat_rho = D['lat_rho'][:]
#     z = D['z'][:]
#
#     # -- INTERPOLATE TO REGULAR Y
#     y = 1852 * 60 * (lat_rho - ref_lat)
#     # y = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_rho - ref_lon)
#
#     # -- TIME
#     time_cor = time/(60*60*24) + 719163  # correct to start (1970)
#     time_hour = 24*(time_cor - np.int(time_cor))
#     date_time = datetime.date.fromordinal(np.int(time_cor))
#
#     # # ---- SAVE OUTPUT for old PYTHON/PICKLE ------------------------------------------------------------------------
#     # date_str_out = str(date_time.year) + '_' + str(date_time.month) + '_' + str(date_time.day) + \
#     #                '_' + str(np.int(np.round(time_hour, 1)))
#     # my_dict = {'ocean_time': time, 'temp': t, 'salt': s, 'u': u, 'v': v, 'lon_rho': lon_rho, 'lat_rho': lat_rho,
#     #            'lon_u': lon_u, 'lat_u': lat_u, 'lon_v': lon_v, 'lat_v': lat_v, 'z': z}
#     # output = open('/Users/jake/Documents/baroclinic_modes/Model/time_avg/pickle_protocol_2/LT_N_S_extraction_' +
#     #               date_str_out + '.pkl', 'wb')
#     # pickle.dump(my_dict, output, protocol=2)
#     # output.close()
#
#     # -- GSW
#     sig0_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
#     N2_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
#     t_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
#     s_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
#     # loop over each column and interpolate to z
#     p = gsw.p_from_z(z, ref_lat * np.ones(np.shape(z)))
#     SA = gsw.SA_from_SP(s, p, lon_rho[0] * np.ones(np.shape(s)), np.tile(lat_rho, (np.shape(p)[0], 1)))
#     CT = gsw.CT_from_t(s, t, p)
#     sig0_0 = gsw.sigma0(SA, CT)
#     N2_0 = gsw.Nsquared(SA, CT, p, lat=ref_lat)[0]
#     for i in range(len(lat_rho)):
#         # p = gsw.p_from_z(z[:, i], ref_lat * np.ones(len(z)))
#         # SA = gsw.SA_from_SP(s[:, i], p, lon_rho[0] * np.ones(len(s[:, i])), lat_rho[i] * np.ones(len(s[:, i])))
#         # CT = gsw.CT_from_t(s[:, i], t[:, i], p)
#         # sig0_0 = gsw.sigma0(SA, CT)
#         # N2_0 = gsw.Nsquared(SA[:, i], CT[:, i], p[:, i], lat=ref_lat)[0]
#         # -- INTERPOLATE TO REGULAR Z GRID
#         t_p[:, i] = np.interp(z_grid, z[:, i], t[:, i])
#         s_p[:, i] = np.interp(z_grid, z[:, i], s[:, i])
#         sig0_p[:, i] = np.interp(z_grid, z[:, i], sig0_0[:, i])
#         N2_p[:, i] = np.interp(z_grid, z[0:-1, i], N2_0[:, i])
#
#     # loop over each row and interpolate to Y grid
#     sig0 = np.nan * np.ones((len(z_grid), len(y_grid)))
#     N2 = np.nan * np.ones((len(z_grid), len(y_grid)))
#     yg_low = np.where(y_grid > np.round(np.nanmin(y)))[0][0] - 1
#     yg_up = np.where(y_grid < np.round(np.nanmax(y)))[0][-1] + 1
#     tt = np.nan * np.ones((len(z_grid), len(y_grid)))
#     ss = np.nan * np.ones((len(z_grid), len(y_grid)))
#     for i in range(len(z_grid)):
#         sig0[i, yg_low:yg_up] = np.interp(y_grid[yg_low:yg_up], y, sig0_p[i, :])
#         N2[i, yg_low:yg_up] = np.interp(y_grid[yg_low:yg_up], y, N2_p[i, :])
#         tt[i, yg_low:yg_up] = np.interp(y_grid[yg_low:yg_up], y, t_p[i, :])
#         ss[i, yg_low:yg_up] = np.interp(y_grid[yg_low:yg_up], y, s_p[i, :])
#
#     sig0_bck_out[:, :, m] = sig0.copy()
#     N2_bck_out[:, :, m] = N2.copy()
#     date_out[m, :] = np.array([np.int(time_cor), time_hour])
#     t_bck_out[:, :, m] = tt.copy()
#     s_bck_out[:, :, m] = ss.copy()
#
# # # --- convert in matlab
# eng = matlab.engine.start_matlab()
# eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/')
# eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/library/')
# gamma_bck = np.nan * np.ones(np.shape(sig0_bck_out))
# for j in range(len(file_list)):
#     tic = TT.clock()
#     for i in range(len(y_grid)):
#         gamma_bck[:, i, j] = np.squeeze(np.array(eng.eos80_legacy_gamma_n(matlab.double(s_bck_out[:, i, j].tolist()),
#                                                                           matlab.double(t_bck_out[:, i, j].tolist()),
#                                                                           matlab.double(p_out.tolist()),
#                                                                           matlab.double([lon_yy[0]]),
#                                                                           matlab.double([lat_yy[0]]))))
#     toc = TT.clock()
#     print('Time step = ' + str(j) + ' = '+ str(toc - tic) + 's')
# eng.quit()
# avg_gamma = np.nanmean(np.nanmean(gamma_bck, axis=2), axis=1)
# my_dict = {'z': z_grid, 'gamma': avg_gamma, 'sig0_back': sig0_bck_out, 'N2_back': N2_bck_out, 'time': date_out}
# output = open('/Users/jake/Documents/baroclinic_modes/Model/extracted_gridded_bck_gamma.pkl', 'wb')
# pickle.dump(my_dict, output)
# output.close()

pkl_file = open('/Users/jake/Documents/baroclinic_modes/Model/extracted_gridded_bck_gamma.pkl', 'rb')
BG = pickle.load(pkl_file)
pkl_file.close()
avg_gamma = BG['gamma'][:]
sig0_bck_out = BG['sig0_back'][:]
N2_bck_out = BG['N2_back'][:]

# avg_sig0 = avg_gamma.copy()  #
avg_sig0 = np.nanmean(np.nanmean(sig0_bck_out, axis=2), axis=1)
ddz_avg_sig0 = np.nan * np.ones(np.shape(avg_sig0))
ddz_avg_sig0[0] = (avg_sig0[1] - avg_sig0[0])/(z_grid[1] - z_grid[0])
ddz_avg_sig0[-1] = (avg_sig0[-1] - avg_sig0[-2])/(z_grid[-1] - z_grid[-2])
for i in range(1, len(z_grid) - 1):
    ddz_avg_sig0[i] = (avg_sig0[i+1] - avg_sig0[i-1])/(z_grid[i+1] - z_grid[i-1])
# ddz_avg_sig0 = np.gradient(avg_sig0, z_grid_s)
avg_N2 = np.nanmean(np.nanmean(N2_bck_out, axis=2), axis=1)

# ---------------------------------------------------------------------------------------------------------------------
# -- MODES ------------------------------------------------------------------------------------------------------------
# --- MODE PARAMETERS
# frequency zeroed for geostrophic modes
omega = 0
# highest baroclinic mode to be calculated
mmax = 25
nmodes = mmax + 1
# maximum allowed deep shear [m/s/km]
deep_shr_max = 0.1
# minimum depth for which shear is limited [m]
deep_shr_max_dep = 3500.0
# fit limits
eta_fit_dep_min = 50.0
eta_fit_dep_max = 2200.0

# --- compute vertical mode shapes
G, Gz, c, epsilon = vertical_modes(np.flipud(avg_N2), np.flipud(-1.0 * z_grid), omega, mmax)  # N2

# --- fit eta
# DG
eta = np.nan * np.ones(np.shape(dg_sig0))
for i in range(num_profs):
    eta[:, i] = (np.flipud(dg_sig0[:, i]) - np.flipud(avg_sig0)) / np.flipud(ddz_avg_sig0)
AG_dg, eta_m_dg, Neta_m_dg, PE_per_mass_dg = eta_fit(num_profs, -1.0 * np.flipud(z_grid), nmodes, np.flipud(avg_N2), G, c,
                                                     eta, eta_fit_dep_min, eta_fit_dep_max)
eta_sm = np.nan * np.ones(np.shape(avg_sig_pd))
for i in range(np.shape(avg_sig_pd)[1]):
    eta_sm[:, i] = (np.flipud(avg_sig_pd[:, i]) - np.flipud(avg_sig0)) / np.flipud(ddz_avg_sig0)
AG_dg_sm, eta_m_dg_sm, Neta_m_dg_sm, PE_per_mass_dg_sm = eta_fit(np.shape(avg_sig_pd)[1], -1.0 * np.flipud(z_grid),
                                                                 nmodes, np.flipud(avg_N2), G, c, eta_sm,
                                                                 eta_fit_dep_min, eta_fit_dep_max)
# Model
eta_model = np.nan * np.ones(np.shape(sig0_out_s[:, :, 0]))
for i in range(np.shape(eta_model)[1]):
    eta_model[:, i] = (np.flipud(sig0_out_s[:, i, 0]) - np.flipud(avg_sig0)) / np.flipud(ddz_avg_sig0)
AG_model, eta_m_model, Neta_m_model, PE_per_mass_model = eta_fit(len(y_grid), -1.0 * np.flipud(z_grid), nmodes,
                                                                 np.flipud(avg_N2), G, c, eta_model,
                                                                 eta_fit_dep_min, eta_fit_dep_max)

# --- fit vel
# DG
HKE_per_mass_dg = np.nan * np.zeros([nmodes, num_profs - 1])
modest = np.arange(11, nmodes)
good_ke_prof = np.ones(num_profs - 1)
AGz = np.zeros([nmodes, num_profs - 1])
HKE_noise_threshold = 1e-5  # 1e-5
V_m = np.nan * np.zeros([np.size(z_grid), num_profs - 1])
for i in range(num_profs - 1):
    # fit to velocity profiles
    this_V = -1 * v_g[:, i].copy()
    iv = np.where(~np.isnan(this_V))
    if iv[0].size > 1:
        AGz[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(Gz[iv, :]), np.transpose(np.atleast_2d(this_V[iv])))[0])
        # Gz(iv,:)\V_g(iv,ip)
        V_m[:, i] = np.squeeze(np.matrix(Gz) * np.transpose(np.matrix(AGz[:, i])))
        # Gz*AGz[:,i];
        HKE_per_mass_dg[:, i] = 0.5 * (AGz[:, i] * AGz[:, i])
        ival = np.where(HKE_per_mass_dg[modest, i] >= HKE_noise_threshold)
        if np.size(ival) > 0:
            good_ke_prof[i] = 0  # flag profile as noisy
    else:
        good_ke_prof[i] = 0  # flag empty profile as noisy as well

# Model
HKE_per_mass_model = np.nan * np.zeros([nmodes, len(y_grid)])
modest = np.arange(11, nmodes)
good_ke_prof = np.ones(len(y_grid))
AGz_model = np.zeros([nmodes, len(y_grid)])
HKE_noise_threshold = 1e-5  # 1e-5
V_m_model = np.nan * np.zeros([np.size(z_grid), len(y_grid)])
for i in range(len(y_grid)):
    # fit to velocity profiles
    this_V = u_out_s[:, i, 0].copy()
    iv = np.where(~np.isnan(this_V))
    if iv[0].size > 1:
        AGz_model[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(Gz[iv, :]), np.transpose(np.atleast_2d(this_V[iv])))[0])
        # Gz(iv,:)\V_g(iv,ip)
        V_m_model[:, i] = np.squeeze(np.matrix(Gz) * np.transpose(np.matrix(AGz_model[:, i])))
        # Gz*AGz[:,i];
        HKE_per_mass_model[:, i] = 0.5 * (AGz_model[:, i] * AGz_model[:, i])
        ival = np.where(HKE_per_mass_model[modest, i] >= HKE_noise_threshold)
        if np.size(ival) > 0:
            good_ke_prof[i] = 0  # flag profile as noisy
    else:
        good_ke_prof[i] = 0  # flag empty profile as noisy as well

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
z_grid_f = np.flipud(z_grid)
dg_sig0_f = np.flipud(dg_sig0)

# --- PLOT DENSITY ANOMALIES, ETA, AND VELOCITIES
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
for i in range(np.shape(dg_sig0)[1]):
    ax1.plot(dg_sig0[:, i] - avg_sig0, z_grid)
ax1.grid()
ax1.set_xlim([-.4, .4])
ax1.set_ylabel('Z [m]')
ax1.set_xlabel('Density Anomaly')
ax1.set_title('Density Anomaly')
for i in range(np.shape(eta)[1]):
    ax2.plot(eta[:, i], z_grid_f, color='#4682B4', linewidth=1.25)
    ax2.plot(eta_m_dg[:, i], z_grid_f, color='k', linestyle='--', linewidth=.75)
ax2.grid()
ax2.set_xlabel('Isopycnal Displacement [m]')
ax2.set_title('Vertical Displacement')

avg_mod_u = np.nanmean(u_out_s, axis=2)
for i in range(np.shape(avg_mod_u)[1]):
    ax3.plot(avg_mod_u[:, i], z_grid, linewidth=0.75, color='#DCDCDC')
for i in range(np.shape(v_g)[1]):
    ax3.plot(-1 * v_g[:, i], z_grid)
    ax3.plot(V_m[:, i], z_grid, color='k', linestyle='--', linewidth=.75)
ax2.set_xlim([-200, 200])
ax3.set_xlim([-.4, .4])
ax3.set_title('Model and DG Velocity (u)')
ax3.set_xlabel('Velocity [m/s]')
plot_pro(ax3)

# ---------------------------------------------------------------------------------------------------------------------
# --- ENERGY
avg_PE = np.nanmean(PE_per_mass_dg, axis=1)
avg_PE_smooth = np.nanmean(PE_per_mass_dg_sm, axis=1)
avg_KE = np.nanmean(HKE_per_mass_dg, axis=1)
avg_PE_model = np.nanmean(PE_per_mass_model, axis=1)
avg_KE_model = np.nanmean(HKE_per_mass_model, axis=1)
dk = ff / c[1]
sc_x = 1000 * ff / c[1:]
PE_SD, PE_GM, GMPE, GMKE = PE_Tide_GM(rho0, -1.0 * z_grid_f, nmodes, np.flipud(np.transpose(np.atleast_2d(avg_N2))), ff)

# --- PLOT ENERGY SPECTRA
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
l_lim = 3 * 10 ** -2
# DG
ax1.plot(sc_x, avg_PE[1:] / dk, color='#B22222', label='APE$_{DG}$', linewidth=3)
ax1.scatter(sc_x, avg_PE[1:] / dk, color='#B22222', s=20)
ax1.plot(sc_x, avg_PE_smooth[1:] / dk, color='#FF8C00', label='APE$_{DG}$', linewidth=3)
ax1.scatter(sc_x, avg_PE_smooth[1:] / dk, color='#FF8C00', s=20)

ax2.plot(1000 * ff / c[1:], avg_KE[1:] / dk, 'g', label='KE$_{DG}$', linewidth=3)
ax2.scatter(sc_x, avg_KE[1:] / dk, color='g', s=20)  # DG KE
ax2.plot([l_lim, 1000 * ff / c[1]], avg_KE[0:2] / dk, 'g', linewidth=3)  # DG KE_0
ax2.scatter(l_lim, avg_KE[0] / dk, color='g', s=25, facecolors='none')  # DG KE_0
# Model
ax1.plot(sc_x, avg_PE_model[1:] / dk, color='#B22222', label='APE$_{DG}$', linewidth=2, linestyle='--')
ax1.scatter(sc_x, avg_PE_model[1:] / dk, color='#B22222', s=20)
ax2.plot(1000 * ff / c[1:], avg_KE_model[1:] / dk, 'g', label='KE$_{DG}$', linewidth=2, linestyle='--')
ax2.scatter(sc_x, avg_KE_model[1:] / dk, color='g', s=20)
ax2.plot([l_lim, 1000 * ff / c[1]], avg_KE_model[0:2] / dk, 'g', linewidth=2, linestyle='--')
ax2.scatter(l_lim, avg_KE_model[0] / dk, color='g', s=25, facecolors='none')

# test
ax2.plot(1000 * ff / c[1:4], (avg_KE_model[1:4] - GMKE[0:3]) / dk, 'b', linewidth=2, linestyle='--')

# GM
ax1.plot(sc_x, 0.25 * PE_GM / dk, linestyle='--', color='k', linewidth=0.75)
ax1.plot(sc_x, 0.25 * GMPE / dk, color='k', linewidth=0.75)
ax1.text(sc_x[0] - .01, 0.5 * PE_GM[1] / dk, r'$1/4 PE_{GM}$', fontsize=10)
ax2.plot(sc_x, 0.25 * GMKE / dk, color='k', linewidth=0.75)
ax2.text(sc_x[0] - .01, 0.5 * GMKE[1] / dk, r'$1/4 KE_{GM}$', fontsize=10)

ax1.set_xlim([l_lim, 2 * 10 ** 0])
ax2.set_xlim([l_lim, 2 * 10 ** 0])
ax2.set_ylim([10 ** (-4), 2 * 10 ** 2])
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.grid()
ax2.set_xscale('log')
plot_pro(ax2)

# ---------------------------------------------------------------------------------------------------------------------
# --- DEPTH OF ISOPYCNALS IN TIME ----------------------------------------------
# isopycnals i care about
rho1 = 26.6
rho2 = 27
rho3 = 27.6

d_time_per_prof = np.nanmean(dg_t, axis=0)
d_time_per_prof_date = []
d_dep_rho1 = np.nan * np.ones((3, len(d_time_per_prof)))
for i in range(len(d_time_per_prof)):
    d_time_per_prof_date.append(datetime.date.fromordinal(np.int(d_time_per_prof[i])))
    d_dep_rho1[0, i] = np.interp(rho1, dg_sig0_f[:, i], z_grid_f)
    d_dep_rho1[1, i] = np.interp(rho2, dg_sig0_f[:, i], z_grid_f)
    d_dep_rho1[2, i] = np.interp(rho3, dg_sig0_f[:, i], z_grid_f)
mw_time_ordered = mw_time
mw_sig_ordered = np.flipud(avg_sig_pd)
mw_time_date = []
mw_dep_rho1 = np.nan * np.ones((3, len(mw_time_ordered)))
for i in range(len(mw_time)):
    mw_time_date.append(datetime.date.fromordinal(np.int(np.round(mw_time_ordered[i]))))
    mw_dep_rho1[0, i] = np.interp(rho1, mw_sig_ordered[:, i], z_grid_f)
    mw_dep_rho1[1, i] = np.interp(rho2, mw_sig_ordered[:, i], z_grid_f)
    mw_dep_rho1[2, i] = np.interp(rho3, mw_sig_ordered[:, i], z_grid_f)


f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.scatter(d_time_per_prof, d_dep_rho1[0, :], color='g', s=15, label=r'DG$_{ind}$')
ax2.scatter(d_time_per_prof, d_dep_rho1[1, :], color='g', s=15)
ax3.scatter(d_time_per_prof, d_dep_rho1[2, :], color='g', s=15)

ax1.plot(mw_time, mw_dep_rho1[0, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')
ax2.plot(mw_time, mw_dep_rho1[1, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')
ax3.plot(mw_time, mw_dep_rho1[2, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')

ax1.set_title('Depth of $\gamma^{n}$ = ' + str(rho1))
ax2.set_title('Depth of $\gamma^{n}$ = ' + str(rho2))
ax3.set_title('Depth of $\gamma^{n}$ = ' + str(rho3))
ax1.set_ylabel('Depth [m]')
ax2.set_ylabel('Depth [m]')
ax3.set_ylabel('Depth [m]')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=10)
# ax1.set_ylim([500, 850])
# ax2.set_ylim([1050, 1400])
# ax3.set_ylim([2600, 2950])
ax1.grid()
ax2.grid()
plot_pro(ax3)
import numpy as np
import pickle
import glob
import datetime
import gsw
from scipy.integrate import cumtrapz
# -- plotting
import matplotlib.pyplot as plt
from toolkit import plot_pro

# ---- LOAD PROCESSED FILES ---------------------------------------------------------------------------------------
file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/n_s_extraction/*.pkl')
# file_name = open('/Users/jake/Documents/baroclinic_modes/Model/test_extraction_1541203200.pkl', 'rb')

file_name = open(file_list[0], 'rb')
D = pickle.load(file_name)
file_name.close()
y_grid = np.arange(0, 125000, 250)
z_grid = np.concatenate((np.arange(-2400, -1000, 20), np.arange(-1000, 0, 10)))
ref_lat = np.nanmin(D['lat_rho'][:])
ref_lon = np.nanmin(D['lon_rho'][:])

# -- LOOP OVER FILES to collect
num_files = len(file_list)
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

    # -- INTERPOLATE TO REGULAR Y
    y = 1852 * 60 * (lat_rho - ref_lat)
    # y = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_rho - ref_lon)

    # -- TIME
    time_cor = time/(60*60*24) + 719163  # correct to start (1970)
    time_hour = 24*(time_cor - np.int(time_cor))
    date_time = datetime.date.fromordinal(np.int(time_cor))

    # -- GSW
    sig0_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    u_g_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    # loop over each column and interpolate to z
    for i in range(len(lat_rho)):
        p = gsw.p_from_z(z[:, i], ref_lat * np.ones(len(z)))
        SA = gsw.SA_from_SP(s[:, i], p, lon_rho[0] * np.ones(len(s[:, i])), lat_rho[i] * np.ones(len(s[:, i])))
        CT = gsw.CT_from_t(s[:, i], t[:, i], p)
        sig0_0 = gsw.sigma0(SA, CT)
        # -- INTERPOLATE TO REGULAR Z GRID
        sig0_p[:, i] = np.interp(z_grid, z[:, i], sig0_0)
        u_g_p[:, i] = np.interp(z_grid, z[:, i], vel[:, i])

    # loop over each row and interpolate to Y grid
    sig0 = np.nan * np.ones((len(z_grid), len(y_grid)))
    u_g = np.nan * np.ones((len(z_grid), len(y_grid)))
    yg_low = np.where(y_grid > np.round(np.nanmin(y)))[0][0] - 1
    yg_up = np.where(y_grid < np.round(np.nanmax(y)))[0][-1] + 1
    for i in range(len(z_grid)):
        sig0[i, yg_low:yg_up] = np.interp(y_grid[yg_low:yg_up], y, sig0_p[i, :])
        u_g[i, yg_low:yg_up] = np.interp(y_grid[yg_low:yg_up], y, u_g_p[i, :])


    sig0_out[:, :, m] = sig0.copy()
    u_out[:, :, m] = u_g.copy()
    time_ord[m] = time_cor.copy()
    time_out[m] = time.copy()
    date_out[m, :] = np.array([np.int(time_cor), time_hour])
    # -- END LOOP COLLECTING RESULT OF EACH TIME STEP

# SORT TIME STEPS BY TIME
sig0_out_s = sig0_out[:, :, np.argsort(time_out)]
u_out_s = u_out[:, :, np.argsort(time_out)]
time_out_s = time_out[np.argsort(time_out)]
time_ord_s = time_ord[np.argsort(time_out)]
date_out_s = date_out[np.argsort(time_out), :]

# -------------------------------------------------------------------------------------------------------
# CREATE LOCATIONS OF FAKE GLIDER DIVE
# approximate glider as flying with vertical velocity of 0.08 m/s, glide slope of 1:3
# assume glider measurements are made every 10, 15, 30, 45, 120 second intervals
# time between each model time step is 1hr.

dg_vertical_speed = .2  # m/s
dg_glide_slope = 3
num_dives = 8

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
            print(this_t - time_ord_s[nearest_model_t_over])
            # interpolate to hor position of glider dive for time before and after
            sig_t_before = np.interp(dg_y[j, i], y_grid, sig0_out_s[j, :, nearest_model_t_over - 1])
            sig_t_after = np.interp(dg_y[j, i], y_grid, sig0_out_s[j, :, nearest_model_t_over])
            # interpolate across time
            dg_sig0[j, i] = np.interp(this_t, [time_ord_s[nearest_model_t_over - 1], time_ord_s[nearest_model_t_over]],
                                   [sig_t_before, sig_t_after])

# f, ax = plt.subplots()
# ax.plot(sig0_out_s[:, 250, 10], z_grid)
# ax.plot(sig0_out_s[:, 250, 11], z_grid)
# ax.plot(sig0_out_s[:, 250, 12], z_grid)
# ax.plot(sig0_out_s[:, 250, 13], z_grid)
# plot_pro(ax)

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


# --- M/W estimation
g = 9.81
rho0 = 1025
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

# --------------------------------------------------------------------------------------------------------------------
# --- PLOTTING
# f, ax = plt.subplots()
# ax.contourf(dg_y, np.flipud(np.tile(z_grid[:, None], (1, np.shape(dg_y)[1]))), np.flipud(dg_sig0), levels=sigth_levels)
# plot_pro(ax)

h_max = 125000  # horizontal domain limit
u_levels = np.array([-.4, -.35, -.3, -.25, - .2, -.15, -.125, -.1, -.075, -.05, -0.025, 0,
                     0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.contourf(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
           np.nanmean(u_out_s, axis=2), levels=u_levels)
uvc = ax1.contour(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                  np.nanmean(u_out_s, axis=2), levels=u_levels, colors='k', linewidth=0.75)
ax1.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
rhoc = ax1.contour(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                  np.nanmean(sig0_out_s, axis=2), levels=sigth_levels, colors='#A9A9A9', linewidth=0.35)
ax1.scatter(dg_y, dg_z_g, 5, color='k')
for r in range(np.shape(isopycdep)[0]):
    ax1.plot(isopycx[r, :], isopycdep[r, :], color='r', linewidth=0.35)
ax1.set_title(str(datetime.date.fromordinal(np.int(time_ord_s[0]))) +
              '  ' + str(datetime.date.fromordinal(np.int(time_ord_s[-2]))) + ' 72hr. Avg.')
ax1.set_ylim([-2550, 0])
ax1.set_xlim([0, h_max])
ax1.grid()

ax2.contourf(np.tile(mw_y, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(mw_y))),
           -1 * v_g, levels=u_levels)
uvc = ax2.contour(np.tile(mw_y, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(mw_y))), -1 * v_g,
                 levels=u_levels, colors='k', linewidth=0.75)
ax2.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
ax2.scatter(dg_y, dg_z_g, 5, color='k')
for r in range(np.shape(isopycdep)[0]):
    ax2.plot(isopycx[r, :], isopycdep[r, :], color='r')
ax2.set_xlim([0, h_max])
ax2.set_title('DG Estimated Cross-Track Vel.')
ax2.grid()

# ax3.pcolor() --- difference in velocities
plot_pro(ax3)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
for i in range(np.shape(dg_sig0)[1]):
    ax1.plot(dg_sig0[:, i], z_grid)

avg_mod_u = np.nanmean(u_out_s, axis=2)
for i in range(np.shape(avg_mod_u)[1]):
    ax2.plot(avg_mod_u[:, i], z_grid, linewidth=0.75, color='#DCDCDC')
for i in range(np.shape(v_g)[1]):
    ax2.plot(v_g[:, i], z_grid)
ax1.grid()
ax2.set_xlim([-.4, .4])
plot_pro(ax2)

ti1 = 23
ti2 = 46
ti3 = 69
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
ax1.pcolor(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
           u_out_s[:, :, ti1], vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
uvc = ax1.contour(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))), u_out_s[:, :, ti1],
                 levels=u_levels, colors='k', linewidth=0.75)
rhoc = ax1.contour(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                  sig0_out_s[: ,:, ti1], levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

ax2.pcolor(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
           u_out_s[:, :, ti2], vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
uvc = ax2.contour(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))), u_out_s[:, :, ti2],
                 levels=u_levels, colors='k', linewidth=0.75)
rhoc = ax2.contour(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                  sig0_out_s[: ,:, ti2], levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

ax3.pcolor(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
           u_out_s[:, :, ti3], vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
uvc = ax3.contour(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))), u_out_s[:, :, ti3],
                 levels=u_levels, colors='k', linewidth=0.75)
rhoc = ax3.contour(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                  sig0_out_s[: ,:, ti3], levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

ax4.pcolor(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
           np.nanmean(u_out_s, 2), vmin=np.nanmin(u_levels), vmax=np.nanmax(u_levels))
uvc = ax4.contour(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                  np.nanmean(u_out_s, axis=2), levels=u_levels, colors='k', linewidth=0.75)
rhoc = ax4.contour(np.tile(y_grid, (len(z_grid), 1)), np.tile(z_grid[:, None], (1, len(y_grid))),
                  np.nanmean(sig0_out_s, axis=2), levels=sigth_levels, colors='#A9A9A9', linewidth=0.5)

ax4.set_title('72hr avg.')
ax1.set_xlim([0, h_max])
ax2.set_xlim([0, h_max])
ax3.set_xlim([0, h_max])
ax4.set_xlim([0, h_max])
plot_pro(ax4)

# f, ax = plt.subplots()
# ax.pcolor(lat_rho, z, t, vmin=1, vmax=13)
# uvc = ax.contour(np.tile(lat_u, (len(z), 1)), z, u,
#                  levels=[-.2, -.1, -.05, -.025, -0.01, 0, 0.01, 0.025, 0.05, 0.1, 0.2], colors='k', linewidth=0.75)
# ax.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
# rhoc = ax.contour(np.tile(lat_u, (len(z), 1)), z, sig0, levels=[26, 26.2, 26.4, 26.6, 26.8, 27, 27.2, 27.4, 27.6],
#                   colors='#A9A9A9', linewidth=0.5)
# ax.set_title(str(date_time) + '  ' + str(np.int(np.round(time_hour, 1))) + 'hr')
# ax.set_ylim([-2750, 0])
# ax.grid()
# plot_pro(ax)


# TODO
# figure out timing of fake glider dive
# - if dive measurements are made every few seconds, will have to interpolate between relevant mode estimates (1hr apart)
# - I have 72 hours of model output. this equals = (259200s) * (0.08 m / s) / max(abs(z))
# - this corresponds to 8 glider profiles or 4 glider dive-cycles
# - need to interpolate density and velocity from model to each datapoint
# - did all of the above, now have glider profiles of density and velocity (as they would be sampled by a glider)

# - need to now run m/w estimation on output to get cross-track velocity
# - need longer duration of model output to compute 'background' average
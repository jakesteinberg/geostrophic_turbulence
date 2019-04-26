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


# ---- LOAD model extraction FILES ------------------------------------------------------------------------------------
file_path = 'e_w_extraction_oct01_oct03_offshore'  # 'n_s_extraction_eddy_nov1_nov3'  #

# file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/' + file_path + '/*.pkl')
# first run will save output as old pickle format (above)
# second run will load these pkl files with old format so that gamma can be estimated (below)
# when gamma is computed need to use different conda environment (older version of ipython) -- USE glider

file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/' + file_path + '/pickle_protocol_2/*.pkl')

file_name = open(file_list[0], 'rb')
D_0 = pickle.load(file_name)
file_name.close()

z_bottom_levels = np.arange(-4000, -1000, 100)
z_lower_limit = z_bottom_levels[np.where(z_bottom_levels < np.nanmin(D_0['z']))[0][-1]]

# for x,y to turn to dist grid, 0 is either min lat or min lon
ref_lat = np.nanmin(D_0['lat_rho'][:])
ref_lon = np.nanmin(D_0['lon_rho'][:])
x_0 = 1852. * 60. * np.cos(np.deg2rad(ref_lat)) * (D_0['lon_rho'][:] - ref_lon)
y_0 = 1852. * 60. * (D_0['lat_rho'][:] - ref_lat)
# test for n/s or e/w orientation
x_test = x_0[-1] - x_0[0]
N_S = 0
E_W = 0
if x_test > 20000:
    E_W = 1
else:
    N_S = 1

xy_possible_max = np.arange(75000, 200000, 1000)
if E_W:
    xy_max = xy_possible_max[np.where(xy_possible_max > np.nanmax(x_0))[0][0]]
else:
    xy_max = xy_possible_max[np.where(xy_possible_max > np.nanmax(y_0))[0][0]]

xy_grid = np.arange(0, xy_max, 1000) # make 1 km (either x or y)
z_grid = np.concatenate((np.arange(z_lower_limit, -1000, 20), np.arange(-1000, 0, 20)))
p_out = gsw.p_from_z(z_grid, ref_lat)

# -- set for X or Y
if N_S:  # Y
    lat_xy = np.interp(xy_grid, y_0, D_0['lat_rho'][:])
    lon_xy = D_0['lon_rho'][:][0] * np.ones(len(xy_grid))
else:  # X
    lat_xy = D_0['lat_rho'][:][0] * np.ones(len(xy_grid))
    lon_xy = np.interp(xy_grid, x_0, D_0['lon_rho'][:])

# --- TOGGLE TO REPROCESS MODEL OUTPUT
# -- LOOP OVER FILES to collect
num_files = len(file_list)
if N_S > 0:
    sig0_raw = np.nan * np.ones((len(D_0['z']), len(y_0), num_files))
    z_raw = np.nan * np.ones((len(D_0['z']), len(y_0), num_files))
else:
    sig0_raw = np.nan * np.ones((len(D_0['z']), len(x_0), num_files))
    z_raw = np.nan * np.ones((len(D_0['z']), len(x_0), num_files))

t_out = np.nan * np.ones((len(z_grid), len(xy_grid), num_files))
s_out = np.nan * np.ones((len(z_grid), len(xy_grid), num_files))
ct_out = np.nan * np.ones((len(z_grid), len(xy_grid), num_files))
sa_out = np.nan * np.ones((len(z_grid), len(xy_grid), num_files))
sig0_out = np.nan * np.ones((len(z_grid), len(xy_grid), num_files))
u_out = np.nan * np.ones((len(z_grid), len(xy_grid), num_files))
u_off_out = np.nan * np.ones((len(z_grid), len(xy_grid), num_files))
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
    if N_S:
        vel = u.copy()
        vel_off = v.copy()
    else:
        vel = v.copy()
        vel_off = u.copy()

    # -- TIME
    time_cor = time/(60*60*24) + 719163  # correct to start (1970)
    time_hour = 24. * (time_cor - np.int(time_cor))
    date_time = datetime.date.fromordinal(np.int(time_cor))

    # # ---- SAVE OUTPUT for old PYTHON/PICKLE ------------------------------------------------------------------------
    # so that we can use matlab engine to compute gamma
    # date_str_out = str(date_time.year) + '_' + str(date_time.month) + '_' + str(date_time.day) + \
    #                '_' + str(np.int(np.round(time_hour, 1)))
    # my_dict = {'ocean_time': time, 'temp': t, 'salt': s, 'u': u, 'v': v, 'lon_rho': lon_rho, 'lat_rho': lat_rho,
    #            'lon_u': lon_u, 'lat_u': lat_u, 'lon_v': lon_v, 'lat_v': lat_v, 'z': z}
    # output = open('/Users/jake/Documents/baroclinic_modes/Model/' + file_path +
    #               '/pickle_protocol_2/E_W_extraction_' + date_str_out + '.pkl', 'wb')
    # pickle.dump(my_dict, output, protocol=2)
    # output.close()

    # comment / uncomment here

    # -- INTERPOLATE TO REGULAR X, Y
    x = 1852. * 60. * np.cos(np.deg2rad(ref_lat)) * (lon_rho - ref_lon)
    y = 1852. * 60. * (lat_rho - ref_lat)
    # y = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_rho - ref_lon)

    # -- GSW
    sig0_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    u_g_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    u_g_off_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    t_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    s_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    ct_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
    sa_p = np.nan * np.ones((len(z_grid), len(lat_rho)))
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

        max_dep = np.nanmin(z[:, i])
        interp_index = np.where(z_grid > max_dep)[0]

        # -- INTERPOLATE TO REGULAR Z GRID
        t_p[interp_index, i] = np.interp(z_grid[interp_index], z[:, i], t[:, i])
        s_p[interp_index, i] = np.interp(z_grid[interp_index], z[:, i], s[:, i])
        ct_p[interp_index, i] = np.interp(z_grid[interp_index], z[:, i], CT[:, i])
        sa_p[interp_index, i] = np.interp(z_grid[interp_index], z[:, i], SA[:, i])
        sig0_p[interp_index, i] = np.interp(z_grid[interp_index], z[:, i], sig0_0[:, i])
        u_g_p[interp_index, i] = np.interp(z_grid[interp_index], z[:, i], vel[:, i])
        u_g_off_p[interp_index, i] = np.interp(z_grid[interp_index], z[:, i], vel_off[:, i])

    # loop over each row and interpolate to X or Y grid
    tt = np.nan * np.ones((len(z_grid), len(xy_grid)))
    ss = np.nan * np.ones((len(z_grid), len(xy_grid)))
    ctt = np.nan * np.ones((len(z_grid), len(xy_grid)))
    ssa = np.nan * np.ones((len(z_grid), len(xy_grid)))
    sig0 = np.nan * np.ones((len(z_grid), len(xy_grid)))
    u_g = np.nan * np.ones((len(z_grid), len(xy_grid)))
    u_g_off = np.nan * np.ones((len(z_grid), len(xy_grid)))

    # set to X or Y
    if N_S:
        yy = y
    else:
        yy = x
    dist_g_low = np.where(xy_grid > np.round(np.nanmin(yy)))[0][0] - 1
    dist_g_up = np.where(xy_grid < np.round(np.nanmax(yy)))[0][-1] + 1
    for i in range(len(z_grid)):
        tt[i, dist_g_low:dist_g_up] = np.interp(xy_grid[dist_g_low:dist_g_up], yy, t_p[i, :])
        ss[i, dist_g_low:dist_g_up] = np.interp(xy_grid[dist_g_low:dist_g_up], yy, s_p[i, :])
        ctt[i, dist_g_low:dist_g_up] = np.interp(xy_grid[dist_g_low:dist_g_up], yy, ct_p[i, :])
        ssa[i, dist_g_low:dist_g_up] = np.interp(xy_grid[dist_g_low:dist_g_up], yy, sa_p[i, :])
        sig0[i, dist_g_low:dist_g_up] = np.interp(xy_grid[dist_g_low:dist_g_up], yy, sig0_p[i, :])
        u_g[i, dist_g_low:dist_g_up] = np.interp(xy_grid[dist_g_low:dist_g_up], yy, u_g_p[i, :])
        u_g_off[i, dist_g_low:dist_g_up] = np.interp(xy_grid[dist_g_low:dist_g_up], yy, u_g_off_p[i, :])

    t_out[:, :, m] = tt.copy()
    s_out[:, :, m] = ss.copy()
    ct_out[:, :, m] = ctt.copy()
    sa_out[:, :, m] = ssa.copy()
    sig0_out[:, :, m] = sig0.copy()
    u_out[:, :, m] = u_g.copy()
    u_off_out[:, :, m] = u_g_off.copy()
    time_ord[m] = time_cor.copy()
    time_out[m] = time.copy()
    date_out[m, :] = np.array([np.int(time_cor), time_hour])
    print('loaded, gridded, computed density of file = ' + str(m))
    # -- END LOOP COLLECTING RESULT OF EACH TIME STEP

# SORT TIME STEPS BY TIME
sig0_raw_s = sig0_raw[:, :, np.argsort(time_out)]
z_raw = z_raw[:, :, np.argsort(time_out)]
t_out_s = t_out[:, :, np.argsort(time_out)]
s_out_s = s_out[:, :, np.argsort(time_out)]
ct_out_s = ct_out[:, :, np.argsort(time_out)]
sa_out_s = sa_out[:, :, np.argsort(time_out)]
sig0_out_s = sig0_out[:, :, np.argsort(time_out)]
u_out_s = u_out[:, :, np.argsort(time_out)]
u_off_out_s = u_off_out[:, :, np.argsort(time_out)]
time_out_s = time_out[np.argsort(time_out)]
time_ord_s = time_ord[np.argsort(time_out)]
date_out_s = date_out[np.argsort(time_out), :]

# # --- convert in matlab
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/')
eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/library/')
gamma = np.nan * np.ones(np.shape(sig0_out_s))
print('Opened Matlab')
for j in range(len(time_ord_s)):
    tic = TT.clock()
    for i in range(len(xy_grid)):
        good_ind = np.where(~np.isnan(s_out_s[:, i, j]))[0]
        gamma[good_ind, i, j] = np.squeeze(np.array(eng.eos80_legacy_gamma_n(
            matlab.double(s_out_s[good_ind, i, j].tolist()), matlab.double(t_out_s[good_ind, i, j].tolist()),
            matlab.double(p_out[good_ind].tolist()), matlab.double([lon_xy[i]]), matlab.double([lat_xy[0]]))))
    toc = TT.clock()
    print('Time step = ' + str(j) + ' = '+ str(toc - tic) + 's')
eng.quit()
print('Closed Matlab')
my_dict = {'z': z_grid, 'gamma': gamma, 'sig0': sig0_out_s, 'CT': t_out_s, 'SA': s_out_s,
           'vel_cross_transect': u_out_s, 'vel_along_transect': u_off_out_s,
           'time': time_out_s, 'time_ord': time_ord_s,
           'date': date_out_s, 'raw_sig0': sig0_raw_s, 'raw_z': z_raw,
           'dist_grid': xy_grid, 'lon_grid': lon_xy, 'lat_grid': lat_xy}
output = open('/Users/jake/Documents/baroclinic_modes/Model/' + file_path + '/gamma_output'
              '/extracted_gridded_gamma.pkl', 'wb')
pickle.dump(my_dict, output)
output.close()
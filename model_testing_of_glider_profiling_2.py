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
# file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/n_s_extraction_nov1_nov3_offshore/*.pkl')
# first run will save output as old pickle format (above)
# second run will load these pkl files with old format so that gamma can be estimated (below)
# when gamma is computed need to use different conda environment (older version of ipython) -- USE glider
file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/n_s_extraction_nov1_nov3_offshore/'
                      'pickle_protocol_2/*.pkl')


file_name = open(file_list[0], 'rb')
D = pickle.load(file_name)
file_name.close()
y_grid = np.arange(0, 150000, 500)
z_grid = np.concatenate((np.arange(-2800, -1000, 20), np.arange(-1000, 0, 10))) # check max depths, run again
ref_lat = np.nanmin(D['lat_rho'][:])
ref_lon = np.nanmin(D['lon_rho'][:])
p_out = gsw.p_from_z(z_grid, ref_lat)
y = 1852. * 60. * (D['lat_rho'][:] - ref_lat)
lat_yy = np.interp(y_grid, y, D['lat_rho'][:])
lon_yy = D['lon_rho'][:][0] * np.ones(len(y_grid))

# --- TOGGLE TO REPROCESS MODEL OUTPUT
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
    # so that we can use matlab engine to compute gamma
    # date_str_out = str(date_time.year) + '_' + str(date_time.month) + '_' + str(date_time.day) + \
    #                '_' + str(np.int(np.round(time_hour, 1)))
    # my_dict = {'ocean_time': time, 'temp': t, 'salt': s, 'u': u, 'v': v, 'lon_rho': lon_rho, 'lat_rho': lat_rho,
    #            'lon_u': lon_u, 'lat_u': lat_u, 'lon_v': lon_v, 'lat_v': lat_v, 'z': z}
    # output = open('/Users/jake/Documents/baroclinic_modes/Model/n_s_extraction_nov1_nov3_offshore/'
    #               'pickle_protocol_2/N_S_extraction_' + date_str_out + '.pkl', 'wb')
    # pickle.dump(my_dict, output, protocol=2)
    # output.close()

    # comment / uncomment here

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

# # --- convert in matlab
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/')
eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/library/')
gamma = np.nan * np.ones(np.shape(sig0_out_s))
print('Opened Matlab')
for j in range(len(time_ord_s)):
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
print('Closed Matlab')
my_dict = {'z': z_grid, 'gamma': gamma, 'sig0': sig0_out_s, 'u': u_out_s, 'time': time_out_s, 'time_ord': time_ord_s,
           'date': date_out_s, 'raw_sig0': sig0_raw, 'raw_z': z_raw}
output = open('/Users/jake/Documents/baroclinic_modes/Model/n_s_extraction_nov1_nov3_offshore/gamma_output'
              '/extracted_gridded_gamma.pkl', 'wb')
pickle.dump(my_dict, output)
output.close()
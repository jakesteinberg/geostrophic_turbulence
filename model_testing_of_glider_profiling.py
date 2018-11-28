from netCDF4 import Dataset
import numpy as np
import pickle
import glob
import datetime
from zrfun import get_basic_info, get_z

# --- LOAD .NC ---------------------------------------------------------------------------------------------------
# - path for testing on my machine
# --- '/Users/jake/Documents/baroclinic_modes/Model/LiveOcean_11_01_18/ocean_his_0001.nc'
# - path on fjord to output
# --- '/boildat1/parker/LiveOcean_roms/output/cas4_v2_lo6biom/f2018.11.01/'
# --- LOAD
file_list = glob.glob('/boildat1/parker/LiveOcean_roms/output/cas4_v2_lo6biom/f2018.11.15/ocean_his_0001.nc')
# file_list = glob.glob('/boildat1/parker/LiveOcean_roms/output/cas4_v2_lo6biom/f2018.11.01/*ocean_his_007*.nc')
for i in range(len(file_list)):
    file_name = file_list[i]
    D = Dataset(file_name, 'r')
    time = D['ocean_time'][0]
    u = D['u'][0]
    v = D['v'][0]
    t = D['temp'][0]
    s = D['salt'][0]
    lon_rho = D['lon_rho'][:]
    lat_rho = D['lat_rho'][:]
    lon_u = D['lon_u'][:]
    lat_u = D['lat_u'][:]
    lon_v = D['lon_v'][:]
    lat_v = D['lat_v'][:]
    s_rho = D['s_rho'][0]          # sigma coordinates
    zeta = D['zeta'][0]            # SSH
    h = D['h'][:]                  # bathymetry

    G, S, T = get_basic_info(file_name, only_G=False, only_S=False, only_T=False)
    z = get_z(h, zeta, S)[0]

    # --- CHOOSE SLICE
    # north/south
    lon_select_i = 103
    lat_select_i = range(275, 475)
    # east/west
    # lon_select_i = range(50, 130)
    # lat_select_i = 350

    # --- TIME COR
    time_cor = time / (60 * 60 * 24) + 719163  # correct to start (1970)
    time_hour = 24 * (time_cor - np.int(time_cor))
    date_time = datetime.date.fromordinal(np.int(time_cor))
    date_str_out = str(date_time.year) + '_' + str(date_time.month) + '_' + str(date_time.day) + \
                   '_' + str(np.int(np.round(time_hour, 1)))

    # ---- SAVE OUTPUT ---------------------------------------------------------------------------------------
    my_dict = {'ocean_time': time, 'temp': t[:, lat_select_i, lon_select_i], 'salt': s[:, lat_select_i, lon_select_i],
               'u': u[:, lat_select_i, lon_select_i], 'v': v[:, lat_select_i, lon_select_i],
               'lon_rho': lon_rho[lat_select_i, lon_select_i], 'lat_rho': lat_rho[lat_select_i, lon_select_i],
               'lon_u': lon_u[lat_select_i, lon_select_i], 'lat_u': lat_u[lat_select_i, lon_select_i],
               'lon_v': lon_v[lat_select_i, lon_select_i], 'lat_v': lat_v[lat_select_i, lon_select_i],
               'z': z[:, lat_select_i, lon_select_i]}
    # optional for plan view plotting
    # 'lon_all': lon_rho[350:650, 65:95], 'lat_all': lat_rho[350:650, 65:95],
    # 'z_bath': z[0, 350:650, 65:95]}
    output = open('/home/jstein/LT_N_S_extraction_' + date_str_out + '.pkl', 'wb')
    # output = open('/Users/jake/Documents/baroclinic_modes/Model/test_extraction_' + str(np.int(time)) + '.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()


# ---------------------------------------------------------------------------------------------------------------------
# --- RUN MY DOWNLOADED FILE FOR SLICE SELECTION AND CONTEXT PLOT
# file_name = '/Users/jake/Documents/baroclinic_modes/Model/LiveOcean_11_01_18/ocean_his_0001.nc'
# D = Dataset(file_name, 'r')
# time = D['ocean_time'][0]
# u = D['u'][0]
# v = D['v'][0]
# t = D['temp'][0]
# s = D['salt'][0]
# lon_rho = D['lon_rho'][:]
# lat_rho = D['lat_rho'][:]
# lon_u = D['lon_u'][:]
# lat_u = D['lat_u'][:]
# lon_v = D['lon_v'][:]
# lat_v = D['lat_v'][:]
# s_rho = D['s_rho'][0]  # sigma coordinates
# zeta = D['zeta'][0]  # SSH
# h = D['h'][:]  # bathymetry
#
# G, S, T = get_basic_info(file_name, only_G=False, only_S=False, only_T=False)
# z = get_z(h, zeta, S)[0]
#
# # --- CHOOSE SLICE
# # N/S slice
# lon_select_i = 103
# lat_select_i = range(275, 475)
#
# # E/W slice
# # lon_select_i = range(50, 130)
# # lat_select_i = 350
#
# # --- PLOT
# import matplotlib.pyplot as plt
# from toolkit import plot_pro
# f, ax = plt.subplots()
# ax.contour(lon_rho, lat_rho, z[0, :, :], levels=[-15, -10, -5], colors='k')
# ax.contour(lon_rho, lat_rho, z[0, :, :], levels=[-2000, -1000], colors='b')
# ax.plot(lon_rho[lat_select_i, lon_select_i], lat_rho[lat_select_i, lon_select_i], color='r')
# ax.axis([-128, -122, 42, 50])
# plot_pro(ax)

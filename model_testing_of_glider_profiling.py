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
file_list = glob.glob('/boildat1/parker/LiveOcean_roms/output/cas4_v2_lo6biom/f2018.11.01/ocean_his_0*.nc')
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
    # lat_select_i = 350  # old = 160
    # lon_select_i = range(20, 145)
    # lat_select_i = 355

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
    output = open('/home/jstein/NS_nov1_3_eddy/N_S_ext_' + date_str_out + '.pkl', 'wb')
    # output = open('/Users/jake/Documents/baroclinic_modes/Model/test_extraction_' + str(np.int(time)) + '.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()


# ---------------------------------------------------------------------------------------------------------------------
# Instructions for extracting and processes Parker's model data
# sftp jstein@fjord.ocean.washington.edu
# password: musiCman0915

# path = /boildat1/parker/LiveOcean_roms/output/cas4_v2_lo6biom/f2018.11.01/ocean_his_*.nc'  (sample start date)
# - this list of files (for a three day forecast) should be 72 elements long and represents hourly output
# - run code to extract a specified slice of output from each file
# -- put /Users/jake/Documents/geostrophic_turbulence/model_testing_of_glider_profiling.py /home/jstein
# - save this output in /jstein/ directory on fjord
# - get pkl files

# should choose horizontal slice (more geostrophic)
# ---------------------------------------------------------------------------------------------------------------------
# --- RUN MY DOWNLOADED FILE FOR SLICE SELECTION AND CONTEXT PLOT (toggle on when running on my computer)
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
# # lon_select_i = 103
# # lat_select_i = range(275, 475)
# # nov 1 - 3 (eddy) lon = 103, lat = 275-475
# lon_select_i = 103
# lat_select_i = range(275, 475)
#
# # E/W slice (w/ eddy)
# # lon_select_i = range(20, 145)
# # lat_select_i = 355
#
# # resolution test
# ref_lat = 45.5
# ref_lon = -125
# x = 1852. * 60. * np.cos(np.deg2rad(ref_lat)) * (lon_rho[lat_select_i, lon_select_i] - ref_lon)
# y = 1852. * 60. * (lat_rho[lat_select_i, lon_select_i] - ref_lat)
#
# # --- PLOT
# import matplotlib.pyplot as plt
# from toolkit import plot_pro
# f, ax = plt.subplots()
# ax.contour(lon_rho, lat_rho, z[0, :, :], levels=[-25, -20, -15, -10, -5], colors='k')
# bc = ax.contour(lon_rho, lat_rho, z[0, :, :], levels=[-3000, -2900, -2800, -2700, -2600, -2500, -2000, -1000], colors='b')
# ax.clabel(bc, inline_spacing=-3, fmt='%.4g', colors='b')
# ax.plot(lon_rho[lat_select_i, lon_select_i], lat_rho[lat_select_i, lon_select_i], color='r')
# ax.axis([-128, -122, 42, 48])
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('Slice Extracted from Model')
# w = 1 / np.cos(np.deg2rad(46))
# ax.set_aspect(w)
# plot_pro(ax)

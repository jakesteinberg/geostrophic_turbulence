from netCDF4 import Dataset
import numpy as np
import pickle
import glob
from datetime import datetime
import xarray
import time
from tqdm import tqdm
from zrfun import get_basic_info, get_z

# extract model output at a specific lat/lon and create time series (for spectral analysis)

# --- LOAD .NC ---------------------------------------------------------------------------------------------------
# - path for testing on my machine
# --- '/Users/jake/Documents/baroclinic_modes/Model/LiveOcean_11_01_18/ocean_his_0001.nc'
# - path on fjord to output
# --- '/boildat1/parker/LiveOcean_roms/output/cas4_v2_lo6biom/f2018.11.01/'

# single station (mooring)
lon_select_i = 50
lat_select_i = 165

# --- LOAD
folder_list0 = glob.glob('/pgdat1/parker/LiveOcean_roms/output/cas6_v3_lo8b/f2018.10.*')
folder_list1 = glob.glob('/pgdat1/parker/LiveOcean_roms/output/cas6_v3_lo8b/f2018.11.*')
folder_list2 = glob.glob('/pgdat1/parker/LiveOcean_roms/output/cas6_v3_lo8b/f2018.12.*')
folder_list = np.concatenate((folder_list0, folder_list1, folder_list2))
for j in tqdm(range(0, len(folder_list)), ncols=50):  # range(len(folder_list)):
    file_list = glob.glob(folder_list[j] + '/ocean_his_0*.nc')
    file_list_sort = np.sort(file_list)
    # LOOP over files
    for i in tqdm(range(0, len(file_list_sort)-1), ncols=50):  
        file_name = file_list_sort[i]

        xx = xarray.open_dataset(file_name)
        temp_xx = xx.temp.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data[0]
        salt_xx = xx.salt.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data[0]
        zeta_xx = xx.zeta.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
        h_xx = xx.h.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
        lon_xx = xx.lon_rho.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
        lat_xx = xx.lat_rho.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
        time_xx = xx.ocean_time.data[0]
        s_rho_xx = xx.s_rho.data

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
        # s_rho = D['s_rho'][0]          # sigma coordinates
        # zeta = D['zeta'][0]            # SSH
        # h = D['h'][:]                  # bathymetry

        # if (i < 1) & (j < 1):
        G, S, T = get_basic_info(file_name, only_G=False, only_S=False, only_T=False)
        z = get_z(h_xx, zeta_xx, S)[0]

        # print(file_name)

        # --- TIME CORRECTION
        ts = (time_xx - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        # time_cor = time / (60 * 60 * 24) + 719163  # correct to start (1970)
        # time_hour = 24 * (time_cor - np.int(time_cor))
        # date_time = datetime.date.fromordinal(np.int(time_cor))
        # date_time_full = time_cor + (time_hour/24.0)
        # date_str_out = str(date_time.year) + '_' + str(date_time.month) + '_' + str(date_time.day) + '_' + str(np.int(np.round(time_hour, 1)))

        lon_rho_out = lon_xx
        lat_rho_out = lat_xx
        # append to array
        if (i < 1) & (j < 1):
            ocean_time = np.array([ts])[:, None]
            z_out = z.data[:, None]
            temp_out = temp_xx[:, None]
            salin_out = salt_xx[:, None]
        else:
            ocean_time = np.concatenate((ocean_time, np.array([ts])[:, None]))
            z_out = np.concatenate((z_out, z.data[:, None]), axis=1)
            temp_out = np.concatenate((temp_out, temp_xx[:, None]), axis=1)
            salin_out = np.concatenate((salin_out, salt_xx[:, None]), axis=1)

# ---- SAVE OUTPUT ---------------------------------------------------------------------------------------
# create netcdf
save_net = 1
if save_net > 0:
    dataset = Dataset('/home/jstein/E_W_mooring_nov/mooring_3mo.nc', 'w', format='NETCDF4_CLASSIC')
    # -- create dimensions
    dep_dim = dataset.createDimension('dep_dim', len(z_out[:, 0]))
    # individual profiles
    profile_dim = dataset.createDimension('prof_dim', np.shape(temp_out)[1])
    # assign variables
    zz_out = dataset.createVariable('depth', np.float64, ('dep_dim', 'prof_dim'))
    zz_out[:] = z_out
    tt_out = dataset.createVariable('temperature', np.float64, ('dep_dim', 'prof_dim'))
    tt_out[:] = temp_out
    ss_out = dataset.createVariable('salinity', np.float64, ('dep_dim', 'prof_dim'))
    ss_out[:] = salin_out
    time_time_out = dataset.createVariable('time', np.float64, ('prof_dim'))
    time_time_out[:] = ocean_time
    dataset.close()
                   

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
# file_name = '/Users/jake/Documents/baroclinic_modes/Model/misc/LiveOcean_11_01_18/ocean_his_0001.nc'
# G, S, T = get_basic_info(file_name, only_G=False, only_S=False, only_T=False)
#
# # Mooring
# lon_select_i = 50
# lat_select_i = 165  # 355
# depth_stations = [2, 12, 18, 22, 26]  # first index is the bottom
#
# ttt = time.time()
# xx = xarray.open_dataset(file_name)
# temp_xx = xx.temp.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data[0]
# zeta_xx = xx.zeta.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
# h_xx = xx.h.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
# lon_xx = xx.lon_rho.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
# lat_xx = xx.lat_rho.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
#
# z_xx = get_z(h_xx, zeta_xx, S)[0]
#
# print(time.time() - ttt)
#
# print(temp_xx[depth_stations])
#
# ttt = time.time()
# D = Dataset(file_name, 'r')
# time_D = D['ocean_time'][0]
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
# print(time.time() - ttt)
#
# z = get_z(h, zeta, S)[0]
#
# # --- CHOOSE SLICE
# # N/S slice
# # lon_select_i = 103
# # lat_select_i = range(275, 475)
# # nov 1 - 3 (eddy) lon = 103, lat = 275-475
# # lon_select_i = 103
# # lat_select_i = range(275, 475)
#
# # E/W slice (w/ eddy)
# # lon_select_i = range(20, 125)
# # lat_select_i = 165  # 355
#
# # resolution test
# ref_lat = 45.5
# ref_lon = -125
# x = 1852. * 60. * np.cos(np.deg2rad(ref_lat)) * (lon_rho[lat_select_i, lon_select_i] - ref_lon)
# y = 1852. * 60. * (lat_rho[lat_select_i, lon_select_i] - ref_lat)
#
# t_test = t[depth_stations, lat_select_i, lon_select_i]
# z_test = z[depth_stations, lat_select_i, lon_select_i]
#
# print(t_test.data)
#
# # time_cor = time_D / (60 * 60 * 24) + 719163  # correct to start (1970)
# # time_hour = 24 * (time_cor - np.int(time_cor))
# # date_time = datetime.date.fromordinal(np.int(time_cor))
# # date_str_out = str(date_time.year) + '_' + str(date_time.month) + '_' + str(date_time.day) + \
# #     '_' + str(np.int(np.round(time_hour, 1)))
#
# # --- PLOT
# import matplotlib.pyplot as plt
# from toolkit import plot_pro
# f, ax = plt.subplots()
# ax.contour(lon_rho, lat_rho, z[0, :, :], levels=[-25, -20, -15, -10, -5], colors='k')
# bc = ax.contour(lon_rho, lat_rho, z[0, :, :], levels=[-3000, -2900, -2800, -2700, -2600, -2500, -2000, -1000], colors='b')
# ax.clabel(bc, inline_spacing=-3, fmt='%.4g', colors='b')
# # ax.plot(lon_rho[lat_select_i, lon_select_i], lat_rho[lat_select_i, lon_select_i], color='r')
# ax.scatter(lon_rho[lat_select_i, lon_select_i], lat_rho[lat_select_i, lon_select_i], color='r', s=10)
# ax.axis([-128, -122, 42, 48])
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('Slice Extracted from Model')
# w = 1 / np.cos(np.deg2rad(46))
# ax.set_aspect(w)
# plot_pro(ax)
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
lon_select_i = range(15, 220)
lat_select_i = 165

# --- LOAD
# folder_list0 = glob.glob('/pgdat1/parker/LiveOcean_roms/output/cas6_v3_lo8b/f2018.10.*')
folder_list1 = glob.glob('/pgdat1/parker/LiveOcean_roms/output/cas6_v3_lo8b/f2018.11.*')
# folder_list2 = glob.glob('/pgdat1/parker/LiveOcean_roms/output/cas6_v3_lo8b/f2018.12.*')
# folder_list = np.concatenate((folder_list0, folder_list1, folder_list2))
folder_list = folder_list1
# LOOP over folders (1 folder per day)
for j in range(0):  # tqdm(range(0, len(folder_list)), ncols=50):  # range(len(folder_list)):
    file_list = glob.glob(folder_list[j] + '/ocean_his_0*.nc')
    file_list_sort = np.sort(file_list)
    # LOOP over files (1 file per hour)
    for i in range(0):  # tqdm(range(0, len(file_list_sort) - 1), ncols=50):  # range(len(file_list) - 1):
        file_name = file_list_sort[i]

        xx = xarray.open_dataset(file_name)
        temp_xx = xx.temp.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data[0]
        salt_xx = xx.salt.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data[0]
        zeta_xx = xx.zeta.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
        h_xx = xx.h.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
        lon_xx = xx.lon_rho.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
        lat_xx = xx.lat_rho.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
        dist_x = 1852. * 60. * np.cos(np.deg2rad(np.nanmean(lat_xx))) * (lon_xx - np.nanmin(lon_xx))
        time_xx = xx.ocean_time.data[0]
        s_rho_xx = xx.s_rho.data

        G, S, T = get_basic_info(file_name, only_G=False, only_S=False, only_T=False)
        z = get_z(h_xx, zeta_xx, S)[0]

        z0 = np.array([-50])
        z1 = np.array([-200])
        z2 = np.array([-1000])
        z3 = np.array([-2000])
        temp_z = np.nan * np.ones((4, len(temp_xx[10, :])))
        salt_z = np.nan * np.ones((4, len(temp_xx[10, :])))
        for k in range(len(temp_xx[10, :])):
            temp_z[0, k] = np.interp(z0, z[:, k].data, temp_xx[:, k])
            temp_z[1, k] = np.interp(z1, z[:, k].data, temp_xx[:, k])
            temp_z[2, k] = np.interp(z2, z[:, k].data, temp_xx[:, k])
            temp_z[3, k] = np.interp(z3, z[:, k].data, temp_xx[:, k])
            salt_z[0, k] = np.interp(z0, z[:, k].data, salt_xx[:, k])
            salt_z[1, k] = np.interp(z1, z[:, k].data, salt_xx[:, k])
            salt_z[2, k] = np.interp(z2, z[:, k].data, salt_xx[:, k])
            salt_z[3, k] = np.interp(z3, z[:, k].data, salt_xx[:, k])  

        # --- TIME CORRECTION
        ts = (time_xx - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        # time_cor = time / (60 * 60 * 24) + 719163  # correct to start (1970)
        # time_hour = 24 * (time_cor - np.int(time_cor))
        # date_time = datetime.date.fromordinal(np.int(time_cor))
        # date_time_full = time_cor + (time_hour/24.0)
        # date_str_out = str(date_time.year) + '_' + str(date_time.month) + '_' + str(date_time.day) + '_' + str(np.int(np.round(time_hour, 1)))

        lon_rho_out = lon_xx
        lat_rho_out = lat_xx
        dist_x_out = dist_x
        z_out = np.concatenate((z0[:, None], z1[:, None], z2[:, None], z3[:, None]), axis=0)
        # append to array
        if (i < 1) & (j < 1):
            ocean_time = np.array([ts])[:, None]
            temp_z_out = temp_z[:, :, None]
            salt_z_out = temp_z[:, :, None]
            # z_out = z.data[:, :, None]
            # temp_out = temp_xx[:, :, None]
            # salin_out = salt_xx[:, :, None]
        else:
            ocean_time = np.concatenate((ocean_time, np.array([ts])[:, None]))
            temp_z_out = np.concatenate((temp_z_out, temp_z[:, :, None]), axis=2)
            salt_z_out = np.concatenate((salt_z_out, salt_z[:, :, None]), axis=2)
            # z_out = np.concatenate((z_out, z.data[:, None]), axis=2)
            # temp_out = np.concatenate((temp_out, temp_xx[:, None]), axis=2)
            # salin_out = np.concatenate((salin_out, salt_xx[:, None]), axis=2)

# ---- SAVE OUTPUT ---------------------------------------------------------------------------------------
# create netcdf
save_net = 0
if save_net > 0:
    dataset = Dataset('/home/jstein/E_W_mooring_nov/kh_spectra.nc', 'w', format='NETCDF4_CLASSIC')
    # -- create dimensions
    dep_dim = dataset.createDimension('dep_dim', np.shape(z_out)[0])
    # individual profiles (across lon)
    profile_dim = dataset.createDimension('prof_dim', np.shape(temp_z_out)[1])
    # individual profiles (in time)
    time_dim = dataset.createDimension('time_dim', np.shape(temp_z_out)[2])
    # assign variables
    zz_out = dataset.createVariable('depth', np.float64, ('dep_dim'))
    zz_out[:] = z_out
    
    tt_out = dataset.createVariable('temperature', np.float64, ('dep_dim', 'prof_dim', 'time_dim'))
    tt_out[:] = temp_z_out   
    ss_out = dataset.createVariable('salinity', np.float64, ('dep_dim', 'prof_dim', 'time_dim'))
    ss_out[:] = salt_z_out
    dist_dist_out = dataset.createVariable('x_grid', np.float64, ('prof_dim'))
    dist_dist_out[:] = dist_x_out
    time_time_out = dataset.createVariable('time', np.float64, ('time_dim'))
    time_time_out[:] = ocean_time
    dataset.close()
    
save_vel_slice = 1    
if save_vel_slice > 0:
    file_list = glob.glob(folder_list[10] + '/ocean_his_0*.nc')
    file_list_sort = np.sort(file_list)
    file_name = file_list_sort[10]
    
    lon_select_i = range(1, 300)
    lat_select_i = 165
    xx = xarray.open_dataset(file_name)
    lon_xx = xx.lon_v.isel(eta_v=lat_select_i, xi_v=lon_select_i).data
    lat_xx = xx.lat_v.isel(eta_v=lat_select_i, xi_v=lon_select_i).data
    vel_xx = xx.v.isel(eta_v=lat_select_i, xi_v=lon_select_i).data     
    zeta_xx = xx.zeta.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
    h_xx = xx.h.isel(eta_rho=lat_select_i, xi_rho=lon_select_i).data
    dist_x = 1852. * 60. * np.cos(np.deg2rad(np.nanmean(lat_xx))) * (lon_xx - np.nanmin(lon_xx))
    time_xx = xx.ocean_time.data[0]
    
    G, S, T = get_basic_info(file_name, only_G=False, only_S=False, only_T=False)
    z = get_z(h_xx, zeta_xx, S)[0]
    
    z_bot = get_z(xx.h.data, xx.zeta.data, S)[0]
    
    save_slice = 1
    if save_slice > 0:
        dataset = Dataset('/home/jstein/E_W_mooring_nov/v_slice.nc', 'w', format='NETCDF4_CLASSIC')
        # -- create dimensions
        dep_dim = dataset.createDimension('dep_dim', np.shape(z)[0])
        # time dim 
        time_dim = dataset.createDimension('time_dim', 1)
        # individual profiles (across lon)
        profile_dim = dataset.createDimension('prof_dim', np.shape(z)[1])
        # lat_tot dim 
        lon_t_dim = dataset.createDimension('lon_dim', np.shape(xx.lat_rho.data)[1])
        lat_t_dim = dataset.createDimension('lat_dim', np.shape(xx.lat_rho.data)[0])
        bot_z_dim = dataset.createDimension('bot_dim', 1)
        
        # assign variables
        zz_out = dataset.createVariable('depth', np.float64, ('dep_dim', 'prof_dim'))
        zz_out[:] = z    
        vv_out = dataset.createVariable('v_vel', np.float64, ('dep_dim', 'prof_dim'))
        vv_out[:] = vel_xx   
        dist_dist_out = dataset.createVariable('lon_grid', np.float64, ('prof_dim'))
        dist_dist_out[:] = lon_xx
        
        time_time_out = dataset.createVariable('time', np.float64, ('time_dim'))
        time_time_out[:] = time_xx
        bathy_out = dataset.createVariable('bottom_val', np.float64, ('bot_dim', 'lat_dim', 'lon_dim'))
        bathy_out[:] = z_bot[0, :, :]  
        bathy_lon = dataset.createVariable('bottom_lon', np.float64, ('lat_dim', 'lon_dim'))
        bathy_lon[:] = xx.lon_rho.data
        bathy_lat = dataset.createVariable('bottom_lat', np.float64, ('lat_dim', 'lon_dim'))    
        bathy_lat[:] = xx.lat_rho.data
        dataset.close()
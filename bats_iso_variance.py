# BATS

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import datetime 
import glob
import seawater as sw 
import pandas as pd 
import scipy.io as si
from scipy.io import netcdf
# functions I've written 
from grids import make_bin
from mode_decompositions import vertical_modes, PE_Tide_GM

############ Plot plan view of station BATS and glider sampling pattern for 2015

## bathymetry 
bath = '/Users/jake/Documents/baroclinic_modes/DG/sg035_BATS_2014/OceanWatch_smith_sandwell.nc'
bath_fid = netcdf.netcdf_file(bath,'r',mmap=False if sys.platform == 'darwin' else mmap, version=1)
bath_lon = bath_fid.variables['longitude'][:] - 360
bath_lat = bath_fid.variables['latitude'][:]
bath_z = bath_fid.variables['ROSE'][:]

## gliders 
dg_list = glob.glob('/Users/jake/Documents/baroclinic_modes/DG/sg035_BATS_2015/p*.nc')

# physical parameters 
bin_depth = np.concatenate([np.arange(0,150,10), np.arange(150,300,10), np.arange(300,5000,20)])
ref_lat = 31.8
lat_in = 31.7
lon_in = 64.2
grid = bin_depth[1:-1]
grid_p = sw.pres(grid,lat_in)
secs_per_day = 86400.0
datenum_start = 719163 # jan 1 1970 

plot_bath = 1
# initial arrays and dataframes 
df_t = pd.DataFrame()
df_s = pd.DataFrame()
df_den = pd.DataFrame()
df_lon = pd.DataFrame()
df_lat = pd.DataFrame()
heading_rec = []
time_rec = []
time_rec_2 = np.zeros([np.size(dg_list), 2])

## loop over each dive 
count = 0
for i in dg_list[10:]:
    dive_nc_file = netcdf.netcdf_file(i,'r',mmap=False if sys.platform == 'darwin' else mmap, version=1)
    # initial dive information 
    glid_num = dive_nc_file.glider 
    dive_num = dive_nc_file.dive_number 
    heading_ind = dive_nc_file.variables['log_MHEAD_RNG_PITCHd_Wd']   
    
    # extract heading 
    h1 = heading_ind.data[0]
    h_test_0 = heading_ind.data[1]
    h_test_1 = heading_ind.data[2]
    if h_test_0.isdigit() == True:
        if h_test_1.isdigit() == True:
            h2 = h_test_0
            h3 = h_test_1
            h = int(h1+h2+h3)
        else:
            h = int(h1+h2) 
    else:
        h = int(h1)   
        
    if np.sum(heading_rec) < 1:
        heading_rec = np.concatenate([ [h], [h] ])
    else:
        heading_rec = np.concatenate([ heading_rec, [h], [h] ])                  
    
    # dive position                                       
    lat = dive_nc_file.variables['latitude'][:]
    lon = dive_nc_file.variables['longitude'][:]
    x = (lon - lon_in)*(1852*60*np.cos(np.deg2rad(ref_lat)))
    y = (lat - lat_in)*(1852*60)    
    dac_u = dive_nc_file.variables['depth_avg_curr_east'].data
    dac_v = dive_nc_file.variables['depth_avg_curr_north'].data 
    
    # eng     
    ctd_epoch_time = dive_nc_file.variables['ctd_time'][:]
    pitch_ang = dive_nc_file.variables['eng_pitchAng'][:]
    
    # science 
    press = dive_nc_file.variables['ctd_pressure'][:]
    temp = dive_nc_file.variables['temperature'][:]
    salin = dive_nc_file.variables['salinity'][:]    
    theta = sw.ptmp(salin, temp, press,0)
    depth = sw.dpth(press,ref_lat)
    
    # time conversion 
    dive_start_time = dive_nc_file.start_time
    ctd_time = ctd_epoch_time - dive_start_time    
    
    # put on vertical grid (1 column per profile)
    max_d = np.where(depth == depth.max())
    max_d_ind = max_d[0]
    pitch_ang_sub1 = pitch_ang[0:max_d_ind[0]]
    pitch_ang_sub2 = pitch_ang[max_d_ind[0]:]
    dive_mask_i= np.where(pitch_ang_sub1 < 0)
    dive_mask = dive_mask_i[0][:]
    climb_mask_i = np.where(pitch_ang_sub2 > 0)
    climb_mask = climb_mask_i[0][:] + max_d_ind[0] - 1      
    
    # dive/climb time midpoints
    time_dive = ctd_time[dive_mask]
    time_climb = ctd_time[climb_mask]
    
    serial_date_time_dive = datenum_start + dive_start_time/(60*60*24) + np.median(time_dive)/secs_per_day
    serial_date_time_climb = datenum_start + dive_start_time/(60*60*24) + np.median(time_climb)/secs_per_day
    time_rec_2[count,:] = np.array([ serial_date_time_dive, serial_date_time_climb ])
    if np.sum(time_rec) < 1:
        time_rec = np.concatenate([ [serial_date_time_dive], [serial_date_time_climb] ])
    else:
        time_rec = np.concatenate([ time_rec, [serial_date_time_dive], [serial_date_time_climb] ])
                
    # interpolate (bin_average) to smooth and place T/S on vertical depth grid 
    theta_grid_dive, theta_grid_climb, salin_grid_dive, salin_grid_climb, lon_grid_dive, lon_grid_climb, lat_grid_dive, lat_grid_climb = make_bin(bin_depth,depth[dive_mask],
        depth[climb_mask],theta[dive_mask],theta[climb_mask],salin[dive_mask],salin[climb_mask], lon[dive_mask]*1000,lon[climb_mask]*1000,lat[dive_mask]*1000,lat[climb_mask]*1000)
    
    den_grid_dive = sw.pden(salin_grid_dive, theta_grid_dive, grid_p) - 1000
    den_grid_climb = sw.pden(salin_grid_climb, theta_grid_climb, grid_p) - 1000
    
    # create dataframes where each column is a profile 
    t_data_d = pd.DataFrame(theta_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
    t_data_c = pd.DataFrame(theta_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
    s_data_d = pd.DataFrame(salin_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
    s_data_c = pd.DataFrame(salin_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
    den_data_d = pd.DataFrame(den_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
    den_data_c = pd.DataFrame(den_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
    lon_data_d = pd.DataFrame(lon_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
    lon_data_c = pd.DataFrame(lon_grid_climb,index=grid,columns=[glid_num*1000 + dive_num])
    lat_data_d = pd.DataFrame(lat_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
    lat_data_c = pd.DataFrame(lat_grid_climb,index=grid,columns=[glid_num*1000 + dive_num])
    
    if df_t.size < 1:
        df_t = pd.concat([t_data_d,t_data_c],axis=1)
        df_s = pd.concat([s_data_d,s_data_c],axis=1)
        df_den = pd.concat([den_data_d,den_data_c],axis=1)
        df_lon = pd.concat([lon_data_d,lon_data_c],axis=1)
        df_lat = pd.concat([lat_data_d,lat_data_c],axis=1)
    else:
        df_t = pd.concat([df_t,t_data_d,t_data_c],axis=1)
        df_s = pd.concat([df_s,s_data_d,s_data_c],axis=1)
        df_den = pd.concat([df_den,den_data_d,den_data_c],axis=1)
        df_lon = pd.concat([df_lon,lon_data_d,lon_data_c],axis=1)
        df_lat = pd.concat([df_lat,lat_data_d,lat_data_c],axis=1)
 
    count = count+1
# plan view plot     
if plot_bath > 0:
    levels = [ -5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500 , 0]
    fig0, ax0 = plt.subplots(figsize=(7.5,4))
    bc = ax0.contourf(bath_lon,bath_lat,bath_z,levels,cmap='PuBu_r')
    # ax0.contourf(bath_lon,bath_lat,bath_z,[-5, -4, -3, -2, -1, 0, 100, 1000], cmap = 'YlGn_r')
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    bcl = ax0.contour(bath_lon,bath_lat,bath_z,[-4500, -4000],colors='k')
    ml = [(-65,31.5),(-64.4, 32.435)]
    ax0.clabel(bcl,manual = ml, inline_spacing=-3, fmt='%1.0f',colors='k')  
    
    heading_mask = np.where( (heading_rec>200) & (heading_rec <300) ) 
    ax0.plot(df_lon.iloc[:,heading_mask[0]],df_lat.iloc[:,heading_mask[0]],color='r',linewidth=1) 
    ax0.scatter(np.nanmean(df_lon.iloc[:,heading_mask[0]],0),np.nanmean(df_lat.iloc[:,heading_mask[0]],0),s=20,color='k')  
     
    w = 1/np.cos(np.deg2rad(ref_lat))
    ax0.axis([-65.75, -63.25, 31.2, 32.8])
    ax0.set_aspect(w)
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig0.colorbar(bc, cax=cax, label='[m]')
    ax0.set_xlabel('Longitude')
    ax0.set_ylabel('Latitude')    
    plt.show()


############ compute vertical displacements for both station and glider profiles 

############ compute KE from separate transects made using butterfly pattern  




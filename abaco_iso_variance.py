# DG along track isopycnal variance 

import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import glob
import seawater as sw 
import pandas as pd 

dg037_list = glob.glob('/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/sg037/p*.nc')

# grid parameters  
lat_in = 26.5
lon_in = -76.75
den_grid = np.arange(24.5, 28 , 0.02)
dist_grid = np.arange(20,130,2.5)
depth_grid = np.arange(0,5000,5)

grid = depth_grid 

# output dataframe 
df_t = pd.DataFrame()
df_d = pd.DataFrame()
df_den = pd.DataFrame()

plot_plan = 1 

for i in dg037_list:
    nc_fid = Dataset(i,'r')
    dive_num = nc_fid.dive_number 
    
    if dive_num > 39:                                                # exclude dives carried out getting to transect 
        lat = nc_fid.variables['latitude'][:]
        lon = nc_fid.variables['longitude'][:]
        press = nc_fid.variables['ctd_pressure'][:]
        time = nc_fid.variables['ctd_time'][:]
        temp = nc_fid.variables['temperature'][:]
        salin = nc_fid.variables['salinity'][:]
        pitch_ang = nc_fid.variables['eng_pitchAng'][:]
        
        theta = sw.ptmp(salin, temp, press,0)
        sigma = sw.pden(salin, theta, press) - 1000
        depth = sw.dpth(press,26.5)
        
        # put on distance/density grid (1 column per profile)
        dive_mask = pitch_ang < 0
        climb_mask = pitch_ang > 0 
        
        # toggle interpolation (depth/density)  
        dg_interp = depth      
        temp_grid_dive = np.interp(grid,dg_interp[dive_mask],theta[dive_mask])
        temp_grid_climb = np.interp(grid,np.flipud(dg_interp[climb_mask]),np.flipud(theta[climb_mask]))        
        lon_grid_dive = np.interp(grid,dg_interp[dive_mask],lon[dive_mask])
        lat_grid_dive = np.interp(grid,dg_interp[dive_mask],lat[dive_mask])       
        lon_grid_climb = np.interp(grid,dg_interp[climb_mask],lon[climb_mask])
        lat_grid_climb = np.interp(grid,dg_interp[climb_mask],lat[climb_mask])
        
        # have to address this flipping 
        
        if plot_plan > 0:
            plt.scatter(lon_grid_dive,lat_grid_dive,s=1)
            plt.scatter(lon_grid_climb,lat_grid_climb,s=1)
        
        x = (lon_grid_dive - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))
        y = (lat_grid_dive - lat_in)*(1852*60)
        dist_dive = np.sqrt(x**2 + y**2)/1000
        x = (lon_grid_climb - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))
        y = (lat_grid_climb - lat_in)*(1852*60)
        dist_climb = np.sqrt(x**2 + y**2)/1000
        
        # create dataframes where each column is a profile 
        t_data_d = pd.DataFrame(temp_grid_dive,index=grid,columns=[dive_num])
        t_data_c = pd.DataFrame(temp_grid_climb,index=grid,columns=[dive_num+.5])
        d_data_d = pd.DataFrame(dist_dive,index=grid,columns=[dive_num])
        d_data_c = pd.DataFrame(np.flipud(dist_climb),index=grid,columns=[dive_num+.5])
        
        if df_t.size < 1:
            df_t = pd.concat([t_data_d,t_data_c],axis=1)
            df_d = pd.concat([d_data_d,d_data_c],axis=1)
        else:
            df_t = pd.concat([df_t,t_data_d,t_data_c],axis=1)
            df_d = pd.concat([df_d,d_data_d,d_data_c],axis=1)
        
        # if interpolating on a depth grid, interpolate density 
        if grid[10]-grid[9] > 1: 
            den_grid_dive = np.interp(grid,dg_interp[dive_mask],sigma[dive_mask])
            den_grid_climb = np.interp(grid,np.flipud(dg_interp[climb_mask]),np.flipud(sigma[climb_mask]))
            den_data_d = pd.DataFrame(den_grid_dive,index=grid,columns=[dive_num])
            den_data_c = pd.DataFrame(den_grid_climb,index=grid,columns=[dive_num+.5])
            if df_t.size < 1:
                df_den = pd.concat([den_data_d,den_data_c],axis=1)
            else:
                df_den = pd.concat([df_den,den_data_d,den_data_c],axis=1)

if plot_plan > 0:
    plt.show()
        
        
# compute average density/temperature as a function of distance offshore       
count = 0  
mean_dist = np.nanmean(df_d,0)  
t_avg_grid = np.zeros([np.size(grid),np.size(dist_grid)]) 
sigma_avg_grid = np.zeros([np.size(grid),np.size(dist_grid)])  
for i in dist_grid:
    mask = (mean_dist > i-15) & (mean_dist < i+15)
    t_avg_grid[:,count] = np.nanmean(df_t[df_t.columns[mask]],1)
    sigma_avg_grid[:,count] = np.nanmean(df_den[df_den.columns[mask]],1)            
    count = count + 1

plot1 = 0
if plot1 > 0:
    fig = plt.figure()
    plt.pcolor(dist_grid,grid,t_avg_grid, vmin=2, vmax=18)  
    # pc = plt.contour(dist_grid,grid,t_avg_grid,colors='k',levels=[2,3,4,5,6,7,8,10,12,16,20])
    pc = plt.contour(dist_grid,grid,sigma_avg_grid,colors='k',levels=np.arange(25.5,28,0.1))
    plt.gca().invert_yaxis()
    fig.savefig('/Users/jake/Desktop/dg037_depth_den.png',dpi = 300)    
    
# compute eta 
# instead of removing a mean, remove the linear trend 
# create average density profile that is a function of distance 
ddz_avg_sigma = np.zeros([np.size(grid),np.size(dist_grid)]) 
for i in range(np.size(dist_grid)):
     for j in range(5,np.size(grid)-7):
        if np.sum(np.isnan(sigma_avg_grid[j-5:j+5,i])) < 1:
            pfit = np.polyfit(grid[j-5:j+5],sigma_avg_grid[j-5:j+5,i],1)
            ddz_avg_sigma[j,i] = pfit[0] # (sigma_avg_grid[2:,i] - sigma_avg_grid[0:-2,i])/(grid[2]-grid[0])
    
# find closest average profile to subtract to find eta     
eta = np.zeros([np.size(grid),np.size(dist_grid)]) 
for i in range(np.size(dist_grid)):
    dist_test = mean_dist[i] - dist_grid
    closest_i = np.where(dist_test == dist_test.min())
    eta[:,i] = (df_den[df_den.columns[i]] - np.squeeze(sigma_avg_grid[:,closest_i[0]]))/np.squeeze(ddz_avg_sigma[:,closest_i[0]])

plot2 = 1
if plot2 > 0: 
    fig = plt.figure() 
    for i in range(np.size(dist_grid)):
        plt.plot(eta[:,i],grid)
    plt.axis([-1000, 1000, 0, 5000])
    plt.gca().invert_yaxis()    
    fig.savefig('/Users/jake/Desktop/dg037_eta.png',dpi = 300)  
    

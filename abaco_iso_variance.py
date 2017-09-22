# DG along track isopycnal variance 

import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import glob
import seawater as sw 
import pandas as pd 

#### bathymetry 
bath = '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/OceanWatch_smith_sandwell.nc'
bath_fid = Dataset(bath,'r')
bath_lon = bath_fid['longitude'][:] - 360
bath_lat = bath_fid['latitude'][:]
bath_z = bath_fid['ROSE'][:]

#### gliders 
dg037_list = glob.glob('/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/sg037/p*.nc')
dg038_list = glob.glob('/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/sg038/p*.nc') # 50-72
dg_list = np.concatenate([dg037_list[45:],dg038_list[50:72]])
# dg_list = dg037_list[45:]
# dg_list = dg038_list[50:72]

##### grid parameters  
lat_in = 26.5
lon_in = -76.75
bin_depth = np.concatenate([np.arange(0,150,5), np.arange(150,300,10), np.arange(300,5000,20)])
grid = bin_depth[1:-1]
bin_press = sw.pres(grid,lat_in)
den_grid = np.arange(24.5, 28 , 0.02)
dist_grid_s = np.arange(0,125,0.005)
dist_grid = np.arange(0,125,1)
# depth_grid = np.arange(0,5000,5)
# grid = depth_grid 

# output dataframe 
df_t = pd.DataFrame()
df_d = pd.DataFrame()
df_den = pd.DataFrame()

# plot controls 
plot_plan = 1 
plot_cross = 0
p_eta = 0

def make_bin(bin_depth,depth_d,depth_c,temp_d,temp_c,salin_d,salin_c, x_g_d, x_g_c, y_g_d, y_g_c):
    bin_up = bin_depth[0:-2]
    bin_down = bin_depth[2:]
    bin_cen = bin_depth[1:-1] 
    temp_g_dive = np.empty(np.size(bin_cen))
    temp_g_climb = np.empty(np.size(bin_cen))
    salin_g_dive = np.empty(np.size(bin_cen))
    salin_g_climb = np.empty(np.size(bin_cen))
    x_g_dive = np.empty(np.size(bin_cen))
    x_g_climb = np.empty(np.size(bin_cen))
    y_g_dive = np.empty(np.size(bin_cen))
    y_g_climb = np.empty(np.size(bin_cen))
    for i in range(np.size(bin_cen)):
        dp_in_d = (depth_d > bin_up[i]) & (depth_d < bin_down[i])
        dp_in_c = (depth_c > bin_up[i]) & (depth_c < bin_down[i])
        temp_g_dive[i] = np.nanmean(temp_d[dp_in_d])
        temp_g_climb[i] = np.nanmean(temp_c[dp_in_c])
        salin_g_dive[i] = np.nanmean(salin_d[dp_in_d])
        salin_g_climb[i] = np.nanmean(salin_c[dp_in_c])
        x_g_dive[i] = np.nanmean(x_g_d[dp_in_d])/1000
        x_g_climb[i] = np.nanmean(x_g_c[dp_in_c])/1000
        y_g_dive[i] = np.nanmean(y_g_d[dp_in_d])/1000
        y_g_climb[i] = np.nanmean(y_g_c[dp_in_c])/1000
    return(temp_g_dive, temp_g_climb, salin_g_dive, salin_g_climb, x_g_dive, x_g_climb, y_g_dive, y_g_climb)

####################################################################
##### iterate for each dive cycle ######

fig0, ax0 = plt.subplots()
ax0.contour(bath_lon,bath_lat,bath_z)
w = 1/np.cos(np.deg2rad(26.5))
ax0.set_aspect(w)
for i in dg_list:
    nc_fid = Dataset(i,'r')
    glid_num = nc_fid.glider 
    dive_num = nc_fid.dive_number 
                                            
    lat = nc_fid.variables['latitude'][:]
    lon = nc_fid.variables['longitude'][:]
    x = (lon - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))
    y = (lat - lat_in)*(1852*60)
        
    press = nc_fid.variables['ctd_pressure'][:]
    time = nc_fid.variables['ctd_time'][:]
    temp = nc_fid.variables['temperature'][:]
    salin = nc_fid.variables['salinity'][:]
    pitch_ang = nc_fid.variables['eng_pitchAng'][:]
    
    theta = sw.ptmp(salin, temp, press,0)
    # sigma = sw.pden(salin, theta, press) - 1000
    depth = sw.dpth(press,26.5)
        
    # put on distance/density grid (1 column per profile)
    max_d = np.where(depth == depth.max())
    max_d_ind = max_d[0]
    pitch_ang_sub1 = pitch_ang[0:max_d_ind[0]]
    pitch_ang_sub2 = pitch_ang[max_d_ind[0]:]
    dive_mask_i= np.where(pitch_ang_sub1 < 0)
    dive_mask = dive_mask_i[0][:]
    climb_mask_i = np.where(pitch_ang_sub2 > 0)
    climb_mask = climb_mask_i[0][:] + max_d_ind[0] - 1       
    
    # interpolate (bin_average) to smooth and place T/S on vertical depth grid 
    theta_grid_dive, theta_grid_climb, salin_grid_dive, salin_grid_climb, x_grid_dive, x_grid_climb, y_grid_dive, y_grid_climb = make_bin(bin_depth,depth[dive_mask],depth[climb_mask],theta[dive_mask],theta[climb_mask],salin[dive_mask],salin[climb_mask], x[dive_mask],x[climb_mask],y[dive_mask],y[climb_mask])
    
    # compute distance to closest point on transect 
    dist_dive = np.zeros(np.size(x_grid_dive))
    dist_climb = np.zeros(np.size(y_grid_dive))
    for i in range(np.size(x_grid_dive)):
        all_dist_dive = np.sqrt( ( x_grid_dive[i] - dist_grid_s )**2 + ( y_grid_dive[i] - 0 )**2 )
        all_dist_climb = np.sqrt( ( x_grid_climb[i] - dist_grid_s )**2 + ( y_grid_climb[i] - 0 )**2 )
        # dive
        if np.isnan(x_grid_dive[i]):
            dist_dive[i] = float('nan')
        else: 
            closest_dist_dive_i = np.where(all_dist_dive == all_dist_dive.min())
            dist_dive[i] = dist_grid_s[closest_dist_dive_i[0]]
        # climb 
        if np.isnan(x_grid_climb[i]):
            dist_climb[i] = float('nan')
        else:
            closest_dist_climb_i = np.where(all_dist_climb == all_dist_climb.min())
            dist_climb[i] = dist_grid_s[closest_dist_climb_i[0]]      
        
    # create dataframes where each column is a profile 
    t_data_d = pd.DataFrame(theta_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
    t_data_c = pd.DataFrame(theta_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
    d_data_d = pd.DataFrame(dist_dive,index=grid,columns=[glid_num*1000 + dive_num])
    d_data_c = pd.DataFrame(np.flipud(dist_climb),index=grid,columns=[glid_num*1000 + dive_num+.5])
        
    if df_t.size < 1:
        df_t = pd.concat([t_data_d,t_data_c],axis=1)
        df_d = pd.concat([d_data_d,d_data_c],axis=1)
    else:
        df_t = pd.concat([df_t,t_data_d,t_data_c],axis=1)
        df_d = pd.concat([df_d,d_data_d,d_data_c],axis=1)
        
    # if interpolating on a depth grid, interpolate density 
    if grid[10]-grid[9] > 1: 
        den_grid_dive = sw.pden(salin_grid_dive, theta_grid_dive, bin_press) - 1000
        den_grid_climb = sw.pden(salin_grid_climb, theta_grid_climb, bin_press) - 1000
        den_data_d = pd.DataFrame(den_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
        den_data_c = pd.DataFrame(den_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
        if df_t.size < 1:
            df_den = pd.concat([den_data_d,den_data_c],axis=1)
        else:
            df_den = pd.concat([df_den,den_data_d,den_data_c],axis=1)            
            
    # plot plan view action if needed     
    if plot_plan > 0:
        plt.scatter(dist_dive,np.zeros(np.size(dist_dive)),s=0.5,color='k')
        plt.scatter(dist_climb,np.zeros(np.size(dist_climb)),s=0.5,color='k')
        # plt.scatter(x_grid_dive,y_grid_dive,s=2,)
        # plt.scatter(x_grid_climb,y_grid_climb,s=2)
        
        if glid_num > 37:
            plt.scatter(x_grid_dive,y_grid_dive,s=2,color='#B22222')
            plt.scatter(x_grid_climb,y_grid_climb,s=2,color='#B22222')
        else:
            plt.scatter(x_grid_dive,y_grid_dive,s=2,color='#48D1CC')
            plt.scatter(x_grid_climb,y_grid_climb,s=2,color='#48D1CC')
        
                
# end of for loop running over each dive 
if plot_plan > 0:
    plt.show()
        

#######################################################################
        
# compute average density/temperature as a function of distance offshore       
count = 0  
mean_dist = np.nanmean(df_d,0)  
theta_avg_grid = np.zeros([np.size(grid),np.size(dist_grid)]) 
sigma_avg_grid = np.zeros([np.size(grid),np.size(dist_grid)])  
for i in dist_grid:
    mask = (mean_dist > i-5) & (mean_dist < i+5)
    theta_avg_grid[:,count] = np.nanmean(df_t[df_t.columns[mask]],1)
    sigma_avg_grid[:,count] = np.nanmean(df_den[df_den.columns[mask]],1)            
    count = count + 1

if plot_cross > 0:
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.pcolor(dist_grid,grid,sigma_avg_grid, vmin=25, vmax=28)  
    # ax1.contour(dist_grid,grid,t_avg_grid,colors='k',levels=[2,3,4,5,6,7,8,10,12,16,20])
    ax1.contour(dist_grid,grid,sigma_avg_grid,colors='k',
        levels=[25,26,26.2,26.4,26.8,27,27.1,27.2,27.3,27.4,27.5,27.6,27.7,27.75,27.8,27.85,27.9])
    ax1.invert_yaxis()
    
    c2s = plt.cm.jet(np.linspace(0,1,125)) 
    for i in range(np.size(dist_grid)):
        ax2.plot(sigma_avg_grid[:,i],grid,color=c2s[i,:])
    ax2.axis([25,28,0,5000])    
    ax2.invert_yaxis()        
    # plt.show()
    f.savefig('/Users/jake/Desktop/dg037_8_bin_depth_den.png',dpi = 300)    

### plot mean density profile as a function of distance offshore to see how the avg profile changes (and represents the linear trend)   
    
# compute eta 
# instead of removing a mean, remove the linear trend 
# create average density profile that is a function of distance 
if p_eta > 0: 
    ddz_avg_sigma = np.zeros([np.size(grid),np.size(dist_grid)]) 
    ddz_avg_theta = np.zeros([np.size(grid),np.size(dist_grid)]) 
    for i in range(np.size(dist_grid)):
         for j in range(6,np.size(grid)-8):
             if np.sum(np.isnan(sigma_avg_grid[j-6:j+6,i])) < 1:
                 pfit = np.polyfit(grid[j-6:j+6],sigma_avg_grid[j-6:j+6,i],1)
                 ddz_avg_sigma[j,i] = pfit[0] # (sigma_avg_grid[2:,i] - sigma_avg_grid[0:-2,i])/(grid[2]-grid[0])
                 
                 pfit2 = np.polyfit(grid[j-6:j+6],theta_avg_grid[j-6:j+6,i],1)
                 ddz_avg_theta[j,i] = pfit2[0] # (sigma_avg_grid[2:,i] - sigma_avg_grid[0:-2,i])/(grid[2]-grid[0])
    
    # find closest average profile to subtract to find eta     
    eta = np.zeros([np.size(grid),np.size(dist_grid)]) 
    df_eta = pd.DataFrame()
    eta_theta = np.zeros([np.size(grid),np.size(dist_grid)]) 
    df_eta_theta = pd.DataFrame()
    
    # subset = np.where((df_den.columns > 37057) & (df_den.columns < 37064)) # 58-63
    count = 0
    for i in range(np.size(mean_dist)):
        dist_test = np.abs(mean_dist[i] - dist_grid) # distance between this profile and every other on dist_grid 
        closest_i = np.where(dist_test == dist_test.min()) # find closest dist_grid station to this profile 
        eta = (df_den[df_den.columns[i]] - np.squeeze(sigma_avg_grid[:,closest_i[0]]))/np.squeeze(ddz_avg_sigma[:,closest_i[0]])
        eta_theta = (df_t[df_t.columns[i]] - np.squeeze(theta_avg_grid[:,closest_i[0]]))/np.squeeze(ddz_avg_theta[:,closest_i[0]])
        if count < 1:
            df_eta = pd.DataFrame(eta,index=grid,columns=[df_den.columns[i]])
            df_eta_theta = pd.DataFrame(eta_theta,index=grid,columns=[df_den.columns[i]])
        else:
            eta2 = pd.DataFrame(eta,index=grid,columns=[df_t.columns[i]])
            df_eta = pd.concat([df_eta,eta2],axis=1)
            eta3 = pd.DataFrame(eta_theta,index=grid,columns=[df_t.columns[i]])
            df_eta_theta = pd.concat([df_eta_theta,eta3],axis=1)
        count = count+1   
        # eta[:,i] = (df_den[df_den.columns[i]] - np.squeeze(sigma_avg_grid[:,closest_i[0]]))/np.squeeze(ddz_avg_sigma[:,closest_i[0]])

    plot2 = 1
    if plot2 > 0: 
        # fig = plt.figure(num=None, figsize=(5.75, 7.5), dpi=100, facecolor='w', edgecolor='k') 
        f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
        for i in range(np.size(mean_dist)): # range(np.size(subset[0])): #
            if df_eta.columns[i] > 38000:
                p38 = ax1.plot(df_eta_theta.iloc[:,i],grid,color='#B22222',linewidth=.5,label='DG038')
                p38_2 = ax2.plot(df_eta.iloc[:,i],grid,color='#B22222',linewidth=.5,label='DG038')
            else:
                p37 = ax1.plot(df_eta_theta.iloc[:,i],grid,color='#48D1CC',linewidth=.5,label='DG037')
                p37_2 = ax2.plot(df_eta.iloc[:,i],grid,color='#48D1CC',linewidth=.5,label='DG037')
        ax1.plot([0, 0],[0, 5000],'--k')
        ax1.axis([-600, 600, 0, 4800])
        ax1.invert_yaxis()    
        ax2.plot([0, 0],[0, 4800],'--k')
        ax2.axis([-600, 600, 0, 5000])
        ax2.invert_yaxis()
        ax2.set_xlabel(r'$\eta_{\sigma_{\theta}}$')
        ax1.set_xlabel(r'$\eta_{\theta}$')
        ax1.set_ylabel('Depth [m]')
        ax1.set_title(r'ABACO Vertical $\theta$ Disp.')
        ax2.set_title('Vertical Isopycnal Disp.')
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend([handles[1], handles[-1]],[labels[1], labels[-1]],fontsize=10)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend([handles[1], handles[-1]],[labels[1], labels[-1]],fontsize=10)
        ax1.grid()
        ax2.grid()
        f.savefig('/Users/jake/Desktop/dg037_8_eta_theta.png',dpi = 300)
        plt.close()
    

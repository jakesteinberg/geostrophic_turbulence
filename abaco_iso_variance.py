# DG along track isopycnal variance 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import datetime 
from netCDF4 import Dataset
import glob
import seawater as sw 
import pandas as pd 
import scipy.io as si
# functions I've written 
from grids import make_bin
from mode_decompositions import vertical_modes, PE_Tide_GM

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
# dg_list = np.array(dg037_list[45:])
# dg_list = dg038_list[50:72]

#### Deep Argo (nearby)
nc_DA_fid = Dataset('/Users/jake/Documents/baroclinic_modes/Deep_Argo/4902326_prof.nc','r') 
da_press = nc_DA_fid['PRES']
da_t = nc_DA_fid['TEMP_QC']

##### grid parameters  
lat_in = 26.5
lon_in = -76.75
bin_depth = np.concatenate([np.arange(0,150,10), np.arange(150,300,10), np.arange(300,5000,20)])
grid = bin_depth[1:-1]
grid_p = sw.pres(grid,lat_in)
den_grid = np.arange(24.5, 28 , 0.02)
dist_grid_s = np.arange(2,125,0.005)
dist_grid = np.arange(2,125,2)

# output dataframe 
df_t = pd.DataFrame()
df_s = pd.DataFrame()
df_d = pd.DataFrame()
df_den = pd.DataFrame()
time_rec = []
time_rec_2 = np.zeros([dg_list.shape[0], 2])

# plot controls 
plot_plan = 0 
plot_cross = 0
p_eta = 1
plot_eta = 1
plot_eng = 0

####################################################################
##### iterate for each dive cycle ######

# prep plan view plot of dive locations 
if plot_plan > 0: 
    levels = [ -5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500 , 0]
    fig0, ax0 = plt.subplots(figsize=(7.5,4))
    bc = ax0.contourf(bath_lon,bath_lat,bath_z,levels,cmap='PuBu_r')
    ax0.contourf(bath_lon,bath_lat,bath_z,[0, 100, 200], cmap = 'YlGn_r')
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    bcl = ax0.contour(bath_lon,bath_lat,bath_z,[-4500, -1000],colors='k')
    ml = [(-76.75,26.9),(-77.4, 26.8)]
    ax0.clabel(bcl,manual = ml, inline_spacing=-3, fmt='%1.0f',colors='k')
    w = 1/np.cos(np.deg2rad(26.5))
    ax0.axis([-77.5, -75, 25.75, 27.25])
    ax0.set_aspect(w)
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig0.colorbar(bc, cax=cax, label='[m]')
    ax0.set_xlabel('Longitude')
    ax0.set_ylabel('Latitude')

# loop over each dive 
count = 0    
for i in dg_list:
    nc_fid = Dataset(i,'r')
    glid_num = nc_fid.glider 
    dive_num = nc_fid.dive_number 
                                            
    lat = nc_fid.variables['latitude'][:]
    lon = nc_fid.variables['longitude'][:]
    x = (lon - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))
    y = (lat - lat_in)*(1852*60)
        
    press = nc_fid.variables['ctd_pressure'][:]
    ctd_epoch_time = nc_fid.variables['ctd_time'][:]
    temp = nc_fid.variables['temperature'][:]
    salin = nc_fid.variables['salinity'][:]
    pitch_ang = nc_fid.variables['eng_pitchAng'][:]    
    theta = sw.ptmp(salin, temp, press,0)
    depth = sw.dpth(press,26.5)
    
    # time conversion 
    secs_per_day = 86400.0
    dive_start_time = nc_fid.start_time
    ctd_time = ctd_epoch_time - dive_start_time 
    datenum_start = 719163 # jan 1 1970
        
    # put on distance/density grid (1 column per profile)
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
    s_data_d = pd.DataFrame(salin_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
    s_data_c = pd.DataFrame(salin_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
    d_data_d = pd.DataFrame(dist_dive,index=grid,columns=[glid_num*1000 + dive_num])
    d_data_c = pd.DataFrame(np.flipud(dist_climb),index=grid,columns=[glid_num*1000 + dive_num+.5])
        
    if df_t.size < 1:
        df_t = pd.concat([t_data_d,t_data_c],axis=1)
        df_s = pd.concat([s_data_d,s_data_c],axis=1)
        df_d = pd.concat([d_data_d,d_data_c],axis=1)
    else:
        df_t = pd.concat([df_t,t_data_d,t_data_c],axis=1)
        df_s = pd.concat([df_s,s_data_d,s_data_c],axis=1)
        df_d = pd.concat([df_d,d_data_d,d_data_c],axis=1)
        
    # if interpolating on a depth grid, interpolate density 
    if grid[10]-grid[9] > 1: 
        den_grid_dive = sw.pden(salin_grid_dive, theta_grid_dive, grid_p) - 1000
        den_grid_climb = sw.pden(salin_grid_climb, theta_grid_climb, grid_p) - 1000
        den_data_d = pd.DataFrame(den_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
        den_data_c = pd.DataFrame(den_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
        if df_t.size < 1:
            df_den = pd.concat([den_data_d,den_data_c],axis=1)
        else:
            df_den = pd.concat([df_den,den_data_d,den_data_c],axis=1)            
            
    # plot plan view action if needed     
    if plot_plan > 0:
        if glid_num > 37:
            ax0.scatter(1000*x_grid_dive/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, 1000*y_grid_dive/(1852*60)+lat_in, s=2, color='#FFD700')
            ax0.scatter(1000*x_grid_climb/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, 1000*y_grid_climb/(1852*60)+lat_in ,s=2, color='#FFD700')
        else:
            ax0.scatter(1000*x_grid_dive/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, 1000*y_grid_dive/(1852*60)+lat_in,s=2,color='#B22222')
            ax0.scatter(1000*x_grid_climb/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, 1000*y_grid_climb/(1852*60)+lat_in,s=2,color='#B22222')
        
        ax0.scatter(1000*dist_dive/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, np.zeros(np.size(dist_dive))+lat_in,s=0.75,color='k')
        ax0.scatter(1000*dist_climb/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, np.zeros(np.size(dist_climb))+lat_in,s=0.75,color='k')
        ax0.text(1000*(np.median(dist_grid)-20)/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in,lat_in - .3,'~ 125km',fontsize=12,color='w') 
    
    count = count + 1

##### end of for loop running over each dive 
if plot_plan > 0:
    t_s = datetime.date.fromordinal(np.int( np.min(time_rec[:,0]) ))
    t_e = datetime.date.fromordinal(np.int( np.max(time_rec[:,1]) ))
    ax0.set_title('Nine ABACO Transects (DG37,38 - 57 dive-cycles): ' + np.str(t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day))
    fig0.savefig('/Users/jake/Desktop/abaco/plan_view.png',dpi = 300)
    plt.close()
        

#######################################################################     
sz = df_den.shape 
num_profs = sz[1]
time_min = np.min(time_rec)
time_max = np.max(time_rec)
time_mid = 10.5*(time_max-time_min) + time_min
        
# compute average density/temperature as a function of distance offshore       
count = 0  
mean_dist = np.nanmean(df_d,0)  
profs_per_avg = np.zeros(np.shape(dist_grid))
theta_avg_grid = np.zeros([np.size(grid),np.size(dist_grid)])
salin_avg_grid = np.zeros([np.size(grid),np.size(dist_grid)]) 
sigma_avg_grid = np.zeros([np.size(grid),np.size(dist_grid)])  
for i in dist_grid:
    mask = (mean_dist > i-10) & (mean_dist < i+10)
    profs_per_avg[count] = np.sum(mask)
    theta_avg_grid[:,count] = np.nanmean(df_t[df_t.columns[mask]],1)
    salin_avg_grid[:,count] = np.nanmean(df_s[df_s.columns[mask]],1)
    sigma_avg_grid[:,count] = np.nanmean(df_den[df_den.columns[mask]],1)            
    count = count + 1
    
#######################################################################    
### plot mean density profile as a function of distance offshore to see how the avg profile changes (and represents the linear trend)   

if plot_cross > 0:
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.pcolor(dist_grid,grid,sigma_avg_grid, vmin=25, vmax=28)  
    # ax1.contour(dist_grid,grid,t_avg_grid,colors='k',levels=[2,3,4,5,6,7,8,10,12,16,20])
    den_c = ax1.contour(dist_grid,grid,sigma_avg_grid,colors='k',
        levels=[25,26,26.5,27,27.2,27.4,27.6,27.7,27.75,27.8,27.85,27.9])
    ax1.clabel(den_c, fontsize=6, inline=1,fmt='%.4g',spacing=10)  
    ax1.invert_yaxis()
    ax1.set_xlabel('Distance Offshore [km]')
    ax1.set_ylabel('Depth [m]')
    ax1.set_title('Avg. Density Cross-Section (4/21 - 6/15)')
    
    c2s = plt.cm.jet(np.linspace(0,1,62)) 
    for i in range(np.size(dist_grid)):
        ax2.plot(sigma_avg_grid[:,i],grid,color=c2s[i,:])
    ax2.axis([25,28,0,5000])    
    ax2.invert_yaxis()  
    ax2.set_xlabel(r'$\sigma_{\theta}$')
    ax2.set_title('Avg. Density per Dist.')    
    ax2.grid()  
    f.savefig('/Users/jake/Desktop/abaco/dg037_8_bin_depth_den.png',dpi = 300)    
    
#######################################################################        
# compute eta 
# instead of removing a mean, remove the linear trend 
# create average density profile that is a function of distance 
# if compute eta 
if p_eta > 0: 
    z = -1*grid
    ddz_avg_sigma = np.zeros([np.size(grid),np.size(dist_grid)]) 
    ddz_avg_theta = np.zeros([np.size(grid),np.size(dist_grid)]) 
    for i in range(np.size(dist_grid)):
        ddz_avg_sigma[:,i] = np.gradient(sigma_avg_grid[:,i],z)
        ddz_avg_theta[:,i] = np.gradient(theta_avg_grid[:,i],z)
        
         # for j in range(6,np.size(grid)-8):
         #     if np.sum(np.isnan(sigma_avg_grid[j-6:j+6,i])) < 1:
         #         pfit = np.polyfit(grid[j-6:j+6],sigma_avg_grid[j-6:j+6,i],1)
         #         ddz_avg_sigma[j,i] = pfit[0] # (sigma_avg_grid[2:,i] - sigma_avg_grid[0:-2,i])/(grid[2]-grid[0])
         #         
         #         pfit2 = np.polyfit(grid[j-6:j+6],theta_avg_grid[j-6:j+6,i],1)
         #         ddz_avg_theta[j,i] = pfit2[0] # (sigma_avg_grid[2:,i] - sigma_avg_grid[0:-2,i])/(grid[2]-grid[0])         
    
    # compute background N
    N2 = np.zeros(np.shape(df_den))
    for i in range(np.size(dist_grid)):
        N2[1:,i] = np.squeeze(sw.bfrq(salin_avg_grid[:,i], theta_avg_grid[:,i], grid_p, lat=26.5)[0])  
    lz = np.where(N2 < 0)   
    lnan = np.isnan(N2)
    N2[lz] = 0 
    N2[lnan] = 0
    N = np.sqrt(N2)    
    
    # find closest average profile to subtract to find eta     
    eta = np.zeros([np.size(grid),np.size(dist_grid)]) 
    df_theta_anom = pd.DataFrame()
    df_salin_anom = pd.DataFrame()
    df_sigma_anom = pd.DataFrame()
    df_eta = pd.DataFrame()
    eta_theta = np.zeros([np.size(grid),np.size(dist_grid)]) 
    df_eta_theta = pd.DataFrame()
    closest_rec = np.nan*np.zeros([np.size(mean_dist)])
    
    # subset = np.where((df_den.columns > 37057) & (df_den.columns < 37064)) # 58-63
    ### test for deep salinity offset
    f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
    count = 0
    for i in range(np.size(mean_dist)):
        dist_test = np.abs(mean_dist[i] - dist_grid) # distance between this profile and every other on dist_grid 
        closest_i = np.where(dist_test == dist_test.min()) # find closest dist_grid station to this profile 
        closest_rec[i] = np.int(closest_i[0])
        
        theta_anom = (df_t[df_t.columns[i]] - np.squeeze(theta_avg_grid[:,closest_i[0]]))
        salin_anom = (df_s[df_s.columns[i]] - np.squeeze(salin_avg_grid[:,closest_i[0]]))
        sigma_anom = (df_den[df_den.columns[i]] - np.squeeze(sigma_avg_grid[:,closest_i[0]]))       
        eta = (df_den[df_den.columns[i]] - np.squeeze(sigma_avg_grid[:,closest_i[0]]))/np.squeeze(ddz_avg_sigma[:,closest_i[0]])
        eta_theta = (df_t[df_t.columns[i]] - np.squeeze(theta_avg_grid[:,closest_i[0]]))/np.squeeze(ddz_avg_theta[:,closest_i[0]])
        if count < 1:
            df_theta_anom = pd.DataFrame(theta_anom,index=grid,columns=[df_t.columns[i]])
            df_salin_anom = pd.DataFrame(salin_anom,index=grid,columns=[df_s.columns[i]])
            df_sigma_anom = pd.DataFrame(sigma_anom,index=grid,columns=[df_t.columns[i]])
            df_eta = pd.DataFrame(eta,index=grid,columns=[df_den.columns[i]])
            df_eta_theta = pd.DataFrame(eta_theta,index=grid,columns=[df_den.columns[i]])
        else:
            df_theta_anom2 = pd.DataFrame(theta_anom,index=grid,columns=[df_t.columns[i]])
            df_salin_anom2 = pd.DataFrame(salin_anom,index=grid,columns=[df_s.columns[i]])
            df_sigma_anom2 = pd.DataFrame(sigma_anom,index=grid,columns=[df_t.columns[i]])
            eta2 = pd.DataFrame(eta,index=grid,columns=[df_t.columns[i]])
            eta3 = pd.DataFrame(eta_theta,index=grid,columns=[df_t.columns[i]])
            
            df_theta_anom = pd.concat([df_theta_anom,df_theta_anom2],axis=1)
            df_salin_anom = pd.concat([df_salin_anom,df_salin_anom2],axis=1)
            df_sigma_anom = pd.concat([df_sigma_anom,df_sigma_anom2],axis=1)
            df_eta = pd.concat([df_eta,eta2],axis=1)
            df_eta_theta = pd.concat([df_eta_theta,eta3],axis=1)            

        if df_s.columns[i] > 38000:
            if time_rec[i] < time_mid:
                ax1.plot(df_theta_anom.iloc[:,i], grid,'r',linewidth=.5,label='38')
                ax2.plot(df_salin_anom.iloc[:,i], grid,'r',linewidth=.5,label='38')
        else:
            if time_rec[i] < time_mid:
                ax1.plot(df_theta_anom.iloc[:,i], grid,'b',linewidth=.5,label='37')
                ax2.plot(df_salin_anom.iloc[:,i], grid,'b',linewidth=.5,label='37')
            
        count = count+1   
    ax1.set_xlabel('Deg. C')    
    ax1.set_title(r'$\theta - \overline{\theta}$')
    ax1.axis([-1.5,1.5,0,5000])
    ax1.grid()     
    ax1.invert_yaxis()
    ax1.set_ylabel('Depth [m]')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[0],handles[-1]],[labels[0], labels[-1]],fontsize=10)
    ax2.set_title('Salinity Anomaly')   
    ax2.set_xlabel('PSU')
    ax2.axis([-.25,.25,0,5000]) 
    ax2.grid()      
    ax2.invert_yaxis()
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend([handles[0],handles[-1]],[labels[0], labels[-1]],fontsize=10)
    f.savefig('/Users/jake/Desktop/abaco/dg037_8_anoms.png',dpi = 300)
                  

    
    ############# Eta_fit / Mode Decomposition #############
    
    # define G grid 
    omega = 0  # frequency zeroed for geostrophic modes
    mmax = 60  # highest baroclinic mode to be calculated
    nmodes = mmax + 1
    
    G, Gz, c = vertical_modes(N2,grid,omega,mmax)
    
    # first taper fit above and below min/max limits
    # Project modes onto each eta (find fitted eta)
    # Compute PE 
    eta_fit_depth_min = 50
    eta_fit_depth_max = 4250
    AG = np.zeros([nmodes, num_profs])
    AG_theta = np.zeros([nmodes, num_profs])
    Eta_m = np.nan*np.zeros([np.size(grid), num_profs])
    Neta = np.nan*np.zeros([np.size(grid), num_profs])
    NEta_m = np.nan*np.zeros([np.size(grid), num_profs])
    Eta_theta_m = np.nan*np.zeros([np.size(grid), num_profs])
    PE_per_mass = np.nan*np.zeros([nmodes, num_profs])
    PE_theta_per_mass = np.nan*np.zeros([nmodes, num_profs])
    for i in range(num_profs):
        this_eta = df_eta.iloc[:,i][:]
        # obtain matrix of NEta
        Neta[:,i] = N[:,np.int(closest_rec[i])]*this_eta
        this_eta_theta = df_eta_theta.iloc[:,i][:]
        iw = np.where((grid>=eta_fit_depth_min) & (grid<=eta_fit_depth_max))
        if iw[0].size > 1:
            eta_fs = df_eta.iloc[:,i][:] # ETA
            eta_theta_fs = df_eta_theta.iloc[:,i][:]
        
            i_sh = np.where( (bin_depth < eta_fit_depth_min))
            eta_fs.iloc[i_sh[0]] = bin_depth[i_sh]*this_eta.iloc[iw[0][0]]/bin_depth[iw[0][0]]
            eta_theta_fs.iloc[i_sh[0]] = bin_depth[i_sh]*this_eta_theta.iloc[iw[0][0]]/bin_depth[iw[0][0]]
        
            i_dp = np.where( (grid > eta_fit_depth_max) )
            eta_fs.iloc[i_dp[0]] = (grid[i_dp] - grid[-1])*this_eta.iloc[iw[0][-1]]/(grid[iw[0][-1]]-grid[-1])
            eta_theta_fs.iloc[i_dp[0]] = (grid[i_dp] - grid[-1])*this_eta_theta.iloc[iw[0][-1]]/(grid[iw[0][-1]]-grid[-1])
            
            AG[1:,i] = np.squeeze(np.linalg.lstsq(G[:,1:],np.transpose(np.atleast_2d(eta_fs)))[0])
            AG_theta[1:,i] = np.squeeze(np.linalg.lstsq(G[:,1:],np.transpose(np.atleast_2d(eta_theta_fs)))[0])
            Eta_m[:,i] = np.squeeze(np.matrix(G)*np.transpose(np.matrix(AG[:,i])))
            NEta_m[:,i] = N[:,np.int(closest_rec[i])]*np.array(np.squeeze(np.matrix(G)*np.transpose(np.matrix(AG[:,i]))))
            Eta_theta_m[:,i] = np.squeeze(np.matrix(G)*np.transpose(np.matrix(AG_theta[:,i])))
            PE_per_mass[:,i] = (1/2)*AG[:,i]*AG[:,i]*c*c
            PE_theta_per_mass[:,i] = (1/2)*AG_theta[:,i]*AG_theta[:,i]*c*c 
    
    # add fitted Eta to this ......
    if plot_eta > 0: 
        f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
        for i in range(np.size(mean_dist)): # range(np.size(subset[0])): #
            if df_eta.columns[i] > 38000:
                p38 = ax1.plot(df_eta_theta.iloc[:,i],grid,color='#B22222',linewidth=.4,label='DG038')
                p38_f = ax1.plot(Eta_theta_m[:,i],grid,'k--',linewidth=.3,label='DG038')
                
                p38_2 = ax2.plot(df_eta.iloc[:,i],grid,color='#B22222',linewidth=.4,label='DG038')
                p38_f = ax2.plot(Eta_m[:,i],grid,'k--',linewidth=.3)
                # p37_2 = ax2.plot(Neta[:,i],grid,color='#48D1CC',linewidth=.4,label='DG037')
                # p37_f = ax2.plot(NEta_m[:,i],grid,'k--',linewidth=.3)
            else:
                p37 = ax1.plot(df_eta_theta.iloc[:,i],grid,color='#48D1CC',linewidth=.4,label='DG037')
                p37_f = ax1.plot(Eta_theta_m[:,i],grid,'k--',linewidth=.3,label='DG037')
                
                p37_2 = ax2.plot(df_eta.iloc[:,i],grid,color='#48D1CC',linewidth=.4,label='DG037')
                p37_f = ax2.plot(Eta_m[:,i],grid,'k--',linewidth=.3)
                # p37_2 = ax2.plot(Neta[:,i],grid,color='#48D1CC',linewidth=.4,label='DG037')
                # p37_f = ax2.plot(NEta_m[:,i],grid,'k--',linewidth=.3)
        ax1.plot([0, 0],[0, 5000],'--k')
        ax1.axis([-600, 600, 0, 4800])
        ax1.invert_yaxis()    
        ax2.plot([0, 0],[0, 4800],'--k')
        ax2.axis([-600, 600, 0, 5000])
        # ax2.axis([-.5, .5, 0, 5000])
        ax2.invert_yaxis()
        ax2.set_xlabel(r'$\eta_{\sigma_{\theta}}$ [m]')
        ax1.set_xlabel(r'$\eta_{\theta}$ [m]')
        ax1.set_ylabel('Depth [m]')
        ax1.set_title(r'ABACO Vertical $\theta$ Disp.')
        ax2.set_title(r'$\eta$ Vertical Isopycnal Disp.')
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend([handles[0], handles[-2]],[labels[1], labels[-1]],fontsize=8)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend([handles[1], handles[-1]],[labels[1], labels[-1]],fontsize=8)
        ax1.grid()
        ax2.grid()
        f.savefig('/Users/jake/Desktop/abaco/dg037_8_eta.png',dpi = 300)
        plt.close()
    
            
    avg_PE = np.nanmean(PE_per_mass,1)
    avg_PE_theta = np.nanmean(PE_theta_per_mass,1)
    f_ref = np.pi*np.sin(np.deg2rad(26.5))/(12*1800)
    rho0 = 1025
    dk = f_ref/c[1]
    sc_x = (1000)*f_ref/c[1:]
    
    PE_SD, PE_GM = PE_Tide_GM(rho0,grid,nmodes,N2,f_ref)
    
    # KE from MATLAB transects
    import_ke = si.loadmat('/Users/jake/Documents/geostrophic_turbulence/ABACO_KE.mat')
    ke_data = import_ke['out']
    ke = ke_data['KE'][0][0]
    dk_ke = 1000*ke_data['f_ref'][0][0][0]/ke_data['c'][0][0][1]
    
    k_h = 1e3*(f_ref/c[1:])*np.sqrt( ke[1:,0]/avg_PE[1:])
    # text(1.1e3*f_ref./c(end), mean(k_h(end-5:end)), 'k_h [km^-^1]', 'color', 'k')
    
    if plot_eng > 0:
        fig0, ax0 = plt.subplots()
        PE_p = ax0.plot(sc_x,avg_PE[1:]/dk,'r',label='PE')
        ax0.plot( [10**-1, 10**0], [1.5*10**1, 1.5*10**-2],color='k',linestyle='--',linewidth=0.8)
        ax0.text(0.8*10**-1,1.3*10**1,'-3',fontsize=8)
        ax0.scatter(sc_x,avg_PE[1:]/dk,color='r',s=6)
        ax0.plot(sc_x,PE_GM/dk,linestyle='--',color='#DAA520')
        ax0.text(sc_x[0]-.009,PE_GM[0]/dk,r'$PE_{GM}$')
        KE_p = ax0.plot(1000*ke_data['f_ref'][0][0][0]/ke_data['c'][0][0][1:],ke[1:]/(dk_ke/1000),color='b',label='KE')
        ax0.scatter(1000*ke_data['f_ref'][0][0][0]/ke_data['c'][0][0][1:],ke[1:]/(dk_ke/1000),color='b',s=6)
        ax0.plot( [1000*f_ref/c[1], 1000*f_ref/c[-2]],[1000*f_ref/c[1], 1000*f_ref/c[-2]],linestyle='--',color='k',linewidth=0.8)
        ax0.text( 1000*f_ref/c[-2]+.1, 1000*f_ref/c[-2], r'f/c$_m$',fontsize=8)
        ax0.plot(sc_x,k_h,color='k')
        ax0.text(sc_x[0]-.008,k_h[0]+.01,r'$k_{h}$ [km$^{-1}$]',fontsize=8)
        
        ax0.set_yscale('log')
        ax0.set_xscale('log')
        # ax0.axis('square')
        ax0.axis([10**-2, 1.5*10**1, 10**(-4), 10**(3)])
        ax0.grid()
        ax0.set_xlabel(r'Vertical Wavenumber = Inverse Rossby Radius = $\frac{f}{c}$ [$km^{-1}$]',fontsize=13)
        ax0.set_ylabel('Spectral Density (and Hor. Wavenumber)')
        ax0.set_title('ABACO')
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend([handles[0],handles[-1]],[labels[0], labels[-1]],fontsize=10)
        fig0.savefig('/Users/jake/Desktop/abaco/dg037_8_PE.png',dpi = 300)
        plt.close()
        
        ######### Analysis
        
        # avgerage displacements as a function of distance and/or time
        fig0, ax0 = plt.subplots()
        time_wins = np.linspace(np.floor(time_min),np.floor(time_max),5)
        dist_wins = np.linspace(dist_grid[0],dist_grid[-1],5)
        c3s = plt.cm.jet(np.linspace(0,1,4)) 
        for i in range(np.size(time_wins)-1):
            # in_win = np.where((time_rec < time_wins[i+1]) & (time_rec > time_wins[i]) )
            in_win = np.where((mean_dist < dist_wins[i+1]) & (mean_dist > dist_wins[i]) )
            avg_prof = np.nanmean(df_eta.iloc[:,in_win[0]],1)
            plt.plot(avg_prof,grid,color=c3s[i,:])
            
        ax0.axis([-100, 100, 0, 5000])
        ax0.invert_yaxis()
        plt.show()
        
        # dynamic mode amplitude with time
        time_ord = np.argsort(time_rec)
        # time_label = datetime.date.fromordinal(np.int(time_rec))
        myFmt = matplotlib.dates.DateFormatter('%m/%d')
        mode_range = [1,2,3,4]
        fig0, ax0 = plt.subplots()
        plt.plot([time_min, time_max],[0, 0],'k',linewidth=1.5)
        for i in mode_range:
            ax_i = ax0.plot(time_rec,AG[i,:],label='Baroclinic Mode ' + np.str(i) )
            ax0.scatter(time_rec,AG[i,:])
        plt.axes([0, time_max, -.07, .07])
        ax0.xaxis.set_major_formatter(myFmt)
        handles, labels = ax0.get_legend_handles_labels()
        ax0.set_xlabel('Date')
        ax0.set_ylabel('Scaled Dynamic Mode (cG) Amplitude')
        ax0.legend(handles,labels,fontsize=10)
        ax0.grid()
        fig0.savefig('/Users/jake/Desktop/abaco/dg037_8_mode_amp.png',dpi = 300)
            
        
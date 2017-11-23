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
import pickle
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec
# functions I've written 
from grids import make_bin
from mode_decompositions import vertical_modes, PE_Tide_GM
from toolkit import plot_pro, find_nearest 

#### bathymetry 
bath = '/Users/jake/Desktop/abaco/abaco_bathymetry/GEBCO_2014_2D_-79.275_22.25_-67.975_29.1.nc'
bath_fid = Dataset(bath,'r')
bath_lon = bath_fid['lon'][:] 
bath_lat = bath_fid['lat'][:]
bath_z = bath_fid['elevation'][:]

#### gliders 
dg037_list = glob.glob('/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/sg037/p*.nc')
dg038_list = glob.glob('/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/sg038/p*.nc') # 50-72
# dg_list = np.concatenate([dg037_list[45:],dg038_list[50:72]])
dg_list = np.array(dg037_list[45:])
# dg_list = dg038_list[50:72]

#######################################################################        
# LOAD ABACO SHIPBOARD CTD DATA 
pkl_file = open('/Users/jake/Desktop/abaco/ship_adcp.pkl', 'rb')
abaco_ship = pickle.load(pkl_file)
pkl_file.close() 

# LOAD NEARBY ARGO CTD DATA 
pkl_file = open('/Users/jake/Desktop/argo/deep_argo_nwa.pkl', 'rb')
abaco_argo = pickle.load(pkl_file)
pkl_file.close()   

##### grid parameters  
lat_in = 26.5
lon_in = -77
bin_depth = np.concatenate([np.arange(0,150,10), np.arange(150,300,10), np.arange(300,5000,20)])
grid = bin_depth[1:-1]
grid_p = sw.pres(grid,lat_in)
den_grid = np.arange(24.5, 28 , 0.02)
dist_grid_s = np.arange(2,125,0.005)
dist_grid = np.arange(2,125,4)

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
plot_eta = 0
plot_eng = 0

####################################################################
##### iterate for each dive cycle ######

# prep plan view plot of dive locations 
if plot_plan > 0: 
    levels = [ -5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500 , 0]
    fig0, ax0 = plt.subplots()
    cmap = plt.cm.get_cmap("Blues_r")
    cmap.set_over('#808000') # ('#E6E6E6')
    bc = ax0.contourf(bath_lon,bath_lat,bath_z,levels,cmap='Blues_r',extend='both',zorder=0)
    # ax0.contourf(bath_lon,bath_lat,bath_z,[0, 100, 200], cmap = 'YlGn_r')
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    bcl = ax0.contour(bath_lon,bath_lat,bath_z,[-4500, -1000],colors='k',zorder=0)
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
    temp_grid_dive, temp_grid_climb, salin_grid_dive, salin_grid_climb, x_grid_dive, x_grid_climb, y_grid_dive, y_grid_climb = make_bin(bin_depth,
        depth[dive_mask],depth[climb_mask],temp[dive_mask],temp[climb_mask],salin[dive_mask],salin[climb_mask],
        x[dive_mask],x[climb_mask],y[dive_mask],y[climb_mask])
    
    theta_grid_dive = sw.ptmp(salin_grid_dive, temp_grid_dive, grid_p, pr=0)
    theta_grid_climb = sw.ptmp(salin_grid_climb, temp_grid_climb, grid_p, pr=0)
    
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
        den_grid_dive = sw.pden(salin_grid_dive, temp_grid_dive, grid_p,pr=0) - 1000
        den_grid_climb = sw.pden(salin_grid_climb, temp_grid_climb, grid_p,pr=0) - 1000
        den_data_d = pd.DataFrame(den_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
        den_data_c = pd.DataFrame(den_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
        if df_t.size < 1:
            df_den = pd.concat([den_data_d,den_data_c],axis=1)
        else:
            df_den = pd.concat([df_den,den_data_d,den_data_c],axis=1)            
            
    # plot plan view action if needed     
    if plot_plan > 0:
        if glid_num > 37:
            dg1 = ax0.scatter(1000*x_grid_dive/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, 1000*y_grid_dive/(1852*60)+lat_in, s=2, color='#FFD700',zorder=1,label='DG38')
            ax0.scatter(1000*x_grid_climb/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, 1000*y_grid_climb/(1852*60)+lat_in ,s=2, color='#FFD700',zorder=1)
        else:
            dg2 = ax0.scatter(1000*x_grid_dive/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, 1000*y_grid_dive/(1852*60)+lat_in,s=2,color='#B22222',zorder=1,label='DG37')
            ax0.scatter(1000*x_grid_climb/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, 1000*y_grid_climb/(1852*60)+lat_in,s=2,color='#B22222',zorder=1)        
        ax0.scatter(1000*dist_dive/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, np.zeros(np.size(dist_dive))+lat_in,s=0.75,color='k')
        ax0.scatter(1000*dist_climb/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in, np.zeros(np.size(dist_climb))+lat_in,s=0.75,color='k')
        ax0.text(1000*(np.median(dist_grid)-20)/(1852*60*np.cos(np.deg2rad(26.5)))+lon_in,lat_in - .3,'~ 125km',fontsize=12,color='w')     
    count = count + 1

##### end of for loop running over each dive 
if plot_plan > 0:    
    sp = ax0.scatter(abaco_ship['cast_lon'],abaco_ship['cast_lat'],s=4,color='#7CFC00',label='Ship')    
    t_s = datetime.date.fromordinal(np.int( np.min(time_rec[0]) ))
    t_e = datetime.date.fromordinal(np.int( np.max(time_rec[-1]) ))
    ax0.set_title('Nine ABACO Transects (DG37,38 - 57 dive-cycles): ' + np.str(t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day))
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles=[dg1,dg2,sp]) # ,[np.unique(labels)],fontsize=10)
    plt.tight_layout()
    fig0.savefig('/Users/jake/Desktop/abaco/plan_2.png',dpi = 200)
    plt.close()
        

#######################################################################     
sz = df_den.shape 
num_profs = sz[1]
time_min = np.min(time_rec)
time_max = np.max(time_rec)
time_mid = 10.5*(time_max-time_min) + time_min
        
# compute average density/temperature as a function of distance offshore       

# compare to shipboard data 
ship_den = abaco_ship['den_grid']
ship_den_2 = abaco_ship['den_grid_2']
ship_theta = abaco_ship['theta_grid']
ship_theta_2 = abaco_ship['theta_grid_2']
ship_salin = abaco_ship['salin_grid']
ship_salin_2 = abaco_ship['salin_grid_2']
ship_dist_0 = abaco_ship['den_dist']
ship_dist = np.nanmean(ship_dist_0,axis=0)
ship_depth_0 = abaco_ship['bin_depth']
ship_depth = np.repeat(np.transpose(np.array([abaco_ship['bin_depth']])),np.shape(ship_den)[1],axis=1)

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
c2s = plt.cm.jet(np.linspace(0,1,31)) 
c2_ship = plt.cm.jet(np.linspace(0,1,11)) 
rho_comp = 0
rho_z_comp = 0
if plot_cross > 0:
    # plan view and comparison across platforms 
    gs = gridspec.GridSpec(3,4)
    ax1 = plt.subplot(gs[0,:])
    ax2 = plt.subplot(gs[1:,0:2])
    ax3 = plt.subplot(gs[1:,2:])    
    levels = [ -5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500 , 0]    
    cmap = plt.cm.get_cmap("Blues_r")
    cmap.set_over('#808000') # ('#E6E6E6')
    bc = ax1.contourf(bath_lon,bath_lat,bath_z,levels,cmap='Blues_r',extend='both',zorder=0)
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    bcl = ax1.contour(bath_lon,bath_lat,bath_z,[-4500, -1000],colors='k',zorder=0)
    ml = [(-76.75,26.9),(-77.4, 26.8)]
    ax1.clabel(bcl,manual = ml, inline_spacing=-3, fmt='%1.0f',colors='k')
    w = 1/np.cos(np.deg2rad(26.5))
    ax1.axis([-77.25, -74, 26.25, 26.75])
    ax1.set_aspect(w)
    for i in dg_list:
        nc_fid = Dataset(i,'r')
        glid_num = nc_fid.glider 
        dive_num = nc_fid.dive_number                                         
        lat = nc_fid.variables['latitude'][:]
        lon = nc_fid.variables['longitude'][:]
        ax1.scatter(lon,lat,s=1,color='#A0522D')
    import_dg = si.loadmat('/Users/jake/Documents/geostrophic_turbulence/ABACO_dg_transect_9.mat')
    dg_data = import_dg['out_t']
    ax1.plot(dg_data['dac_lon'][0][0],dg_data['dac_lat'][0][0],color='y',linewidth=2) 
    ax1.text(-75.28,26.66,'DG38: 67-77',fontsize=8,color='y')
    ax1.scatter(dg_data['dac_lon'][0][0],dg_data['dac_lat'][0][0],s=15,color='y')  
    ax1.quiver(dg_data['dac_lon'][0][0],dg_data['dac_lat'][0][0],dg_data['dac_u'][0][0],dg_data['dac_v'][0][0],color='y',scale=2,headwidth=2,headlength=3,width=.005) 
    ax1.scatter(abaco_ship['adcp_lon'],abaco_ship['adcp_lat'],s=10,color='#7CFC00')   
    ax1.quiver(abaco_ship['adcp_lon'],abaco_ship['adcp_lat'],np.nanmean(abaco_ship['adcp_u']/100,axis=0),np.nanmean(abaco_ship['adcp_v']/100,axis=0),color='#7CFC00',scale=2,headwidth=2,headlength=3,width=.005)  
    ax1.scatter(abaco_ship['cast_lon'],abaco_ship['cast_lat'],s=10,color='r') 
    ax1.quiver(-74.5,26.29,0.1,0,color='w',scale=2,headwidth=2,headlength=3,width=.005)
    ax1.text(-74.3,26.29,'0.1 m/s',color='w',fontsize=8)
    ax1.text(-74.2,26.66,'ADCP',fontsize=8,color='#7CFC00')
    ax1.text(-74.2,26.58,'RV CTD',fontsize=8,color='r')
    ax1.scatter(-77,26.5,s=15,color='m')
    ax1.text(-77,26.44,'Trans. St.',fontsize=6)
    ax1.set_title('RV Endeavour (5/8 - 5/15) CTD/ADCP and DG (5/9 - 5/25)')
    
    lv = np.arange(-.6,.6,.05)
    lv2 = np.arange(-.6,.6,.1)
    ad = ax2.contourf(abaco_ship['adcp_dist'],abaco_ship['adcp_depth'],abaco_ship['adcp_v']/100,levels=lv)
    va = ax2.contour(abaco_ship['adcp_dist'],abaco_ship['adcp_depth'],abaco_ship['adcp_v']/100,levels=lv2,colors='k')
    ax2.plot(35*np.ones(10),np.linspace(0,5000,10),color='r',linewidth=.75,linestyle='--')
    ax2.plot(265*np.ones(10),np.linspace(0,5000,10),color='r',linewidth=.75,linestyle='--')
    ax2.clabel(va, fontsize=6, inline=1,spacing=10,fmt='%1.2g') 
    ax2.set_ylabel('Depth [m]')
    ax2.set_xlabel('Distance Offshore [km]')
    ax2.text(225,4800,'ADCP',fontsize=12)
    ax2.axis([0,300,0,5000])
    ax2.invert_yaxis()
    
    dg_bin = dg_data['bin_depth'][0][0]
    dg_Ds = dg_data['dist'][0][0]
    dg_V = dg_data['V'][0][0]
    # 26.475, -76.648  ==== transect starting position 
    ax3.contourf(np.squeeze(dg_Ds)+35,np.squeeze(dg_bin),dg_V,levels=lv) # shift transect to match inshore distance point with adcp 
    vc = ax3.contour(np.squeeze(dg_Ds)+35,np.squeeze(dg_bin),dg_V,levels=lv2,colors='k')
    ax3.clabel(vc, fontsize=6, inline=1,spacing=10,fmt='%1.2g') 
    ax3.set_xlabel('Distance Offshore [km]')
    ax3.text(225,4800,'DG',fontsize=12)
    ax3.axis([0,300,0,5000])
    ax3.invert_yaxis()
    ax3.axes.get_yaxis().set_visible(False)
    ax3.grid()
    plot_pro(ax3)

if rho_comp > 0:    
    gs = gridspec.GridSpec(4,4)
    ax1 = plt.subplot(gs[0:2,:])
    ax2 = plt.subplot(gs[2:,0:2])
    ax3 = plt.subplot(gs[2:,2:])
    # cross section density T/S profile comparisons 
    ax1.pcolor(dist_grid,grid,sigma_avg_grid, vmin=25, vmax=28)  
    # ax1.contour(dist_grid,grid,t_avg_grid,colors='k',levels=[2,3,4,5,6,7,8,10,12,16,20])
    den_c = ax1.contour(dist_grid,grid,sigma_avg_grid,colors='k', levels=np.append(np.arange(25,27.78,.4),np.arange(27.78,27.9,.02)),linewidth=.35) #[25,26,26.5,27,27.2,27.4,27.6,27.7,27.75,27.8,27.85,27.875,27.9])
    # ax1.clabel(den_c, fontsize=6, inline=1,fmt='%.4g',spacing=10)  
    den_ship = ax1.contour(ship_dist,ship_depth_0,ship_den,colors='r', levels=np.append(np.arange(25,27.78,.4),np.arange(27.78,27.9,.02)),linewidth=.75)
    ax1.clabel(den_ship, fontsize=6, inline=1,fmt='%.4g',spacing=-50)  
    for i in range(28,34):
        ax1.plot(dg_data['Isopyc_x'][0][0][i,:], dg_data['Isopyc_dep'][0][0][i,:],color='k',linewidth=0.75)
        ax1.text(dg_data['Isopyc_x'][0][0][i,-1]+7,dg_data['Isopyc_dep'][0][0][i,-1],str(np.round(np.squeeze(dg_data['sig_th_lev'][0][0][i]),decimals=2)),fontsize=7)
    ax1.text(200,3500,'DG38 - Long Transect')    
    ax1.axis([0, 250, 0, 5000])
    # ax1.set_xlabel('Distance Offshore [km]')
    ax1.set_ylabel('Depth [m]')
    ax1.set_title(r'$\overline{\rho_{\theta}}$ Cross-Section from DG37,38 (4/21 - 6/15) Comp. to RV CTD (Red)',fontsize=12)
    ax1.invert_yaxis()
    
    ### close up of T/S differences 
    TT,SS = np.meshgrid(np.arange(0,25,.1),np.arange(34,38,.1))
    DD = sw.pden(SS,TT,np.zeros(np.shape(TT)),0)-1000
    DD_l = np.arange(25,28.2,0.2)
    DD_c = ax2.contour(SS,TT,DD,colors='k',linewidths=0.25,levels=DD_l)
    ax2.clabel(DD_c, fontsize=8,fmt='%.4g',inline_spacing=-1) #, inline=1,fmt='%.4g',spacing=5)  
    for i in range(np.size(dist_grid)):
        ax2.plot(salin_avg_grid[:,i],theta_avg_grid[:,i],color=c2s[i,:],linewidth=0.75)   
    for i in range(10):
        ax2.plot(ship_salin[:,i],ship_theta[:,i],color=c2_ship[i,:],linestyle='--',linewidth=0.75)   
    # ax2.plot(np.nanmean(abaco_argo['salin'][1:-1],axis=1),np.nanmean(abaco_argo['theta'][1:-1],axis=1),color='k')    
    ax2.axis([34.8,36.75,1.5,22]) 
    ax2.set_ylabel('Potential Temp.',fontsize=12)
    ax2.set_xlabel('Salinity',fontsize=12)
    ax2.grid()
    
    TT,SS = np.meshgrid(np.arange(0,2.8,.001),np.arange(34.4,35,.001))
    DD = sw.pden(SS,TT,np.zeros(np.shape(TT)),0)-1000
    DD_l3 = np.arange(27.7,28,0.005)
    DD_c3 = ax3.contour(SS,TT,DD,colors='k',linewidths=0.25,levels=DD_l3)
    # ax3.clabel(DD_c3,inline=1,fontsize=8) #,inline_spacing=-5,fmt='%.4g') #, inline=1,fmt='%.4g',spacing=5)  
    ml = [(34.87,2.1),(34.88,2.13),(34.89, 2.15),(34.9,1.9)]
    ax3.clabel(DD_c3,manual = ml, inline_spacing=-10, fmt='%.3f',colors='k')
    for i in range(np.size(dist_grid)):
        ts_D = ax3.plot(salin_avg_grid[:,i],theta_avg_grid[:,i],linewidth=0.75,color='b',label='DG') #c2s[i,:])   
    for i in range(10):
        ts_s = ax3.plot(ship_salin[:,i],ship_theta[:,i],linestyle='--',linewidth=0.75,color='r',label='Ship') #c2_ship[i,:])  
    # ts_a = ax3.plot(np.nanmean(abaco_argo['salin'][1:-1],axis=1),np.nanmean(abaco_argo['theta'][1:-1],axis=1),color='k',label='Argo')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend([handles[0],handles[-1]],[labels[0], labels[-1]],fontsize=12)
    ax3.set_xlabel('Salinity',fontsize=12)
    ax3.axis([34.86,34.93,1.7,2.52])
    plot_pro(ax3)

if rho_z_comp > 0:     
    # argo noodling
    argo_press = abaco_argo['bin_press']
    argo_dep = sw.dpth(argo_press,26)
    argo_sigma_interp = np.interp(grid,argo_dep,np.nanmean(abaco_argo['sigma_theta'],axis=1))
    argo_theta_interp = np.interp(grid,argo_dep,np.nanmean(abaco_argo['theta'],axis=1))
    argo_salin_interp = np.interp(grid,argo_dep,np.nanmean(abaco_argo['salin'],axis=1))
    f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)
    for i in range(np.size(dist_grid)):
        if np.sum(np.isnan(sigma_avg_grid[:,1])) > 200:
            ix,iv = find_nearest(ship_dist,dist_grid[i]) 
            # ship noodling
            ship_den_interp = np.interp(grid,ship_depth_0,ship_den[:,ix])
            ship_salin_interp = np.interp(grid,ship_depth_0,ship_salin[:,ix])
            ship_theta_interp = np.interp(grid,ship_depth_0,ship_theta[:,ix])
            shp = ax1.plot( (salin_avg_grid[:,i]-ship_salin_interp),grid,label='Shipboard',color='#48D1CC')
            shp = ax2.plot( (theta_avg_grid[:,i]-ship_theta_interp),grid,label='Shipboard',color='#48D1CC')
            shp = ax3.plot( (sigma_avg_grid[:,i]-ship_den_interp),grid,label='Shipboard',color='#48D1CC')  
            # shp = ax1.plot( (salin_avg_grid[:,i]-argo_salin_interp ),grid,label='Argo',color='r') 
            # shp = ax2.plot( (theta_avg_grid[:,i]-argo_theta_interp ),grid,label='Argo',color='r') 
            # shp = ax3.plot( (sigma_avg_grid[:,i]-argo_sigma_interp ),grid,label='Argo',color='r')    
    ax1.axis([-0.02,0.02,0,5000])  
    ax1.grid()
    ax1.set_title('Salinity Offset')
    ax1.set_xlabel(r'$S_{DG} - S_{ship}$') 
    ax2.axis([-.15,.15,0,5000])  
    ax2.grid()
    ax2.set_title('Pot Temp. Offset')
    ax2.set_xlabel(r'$\theta_{DG} - \theta_{ship}$')
    ax3.axis([-.03,.03,0,5000])  
    ax3.set_xlabel(r'$\sigma_{\theta_{DG}} - \sigma_{\theta_{ship}}$')
    ax3.set_title('Density Offset')       
    ax3.invert_yaxis()
    plot_pro(ax3)
    
    f,ax = plt.subplots()
    ax.plot(np.nanmean(salin_avg_grid,axis=1),grid)
    ax.invert_yaxis()
    plot_pro(ax)
    # ax3.grid()
    # f.savefig('/Users/jake/Desktop/abaco/platform_comp.png',dpi = 300)    
    # plt.close()      
    
#######################################################################        
# compute eta 
# instead of removing a mean, remove the linear trend 
# create average density profile that is a function of distance 
z = -1*grid
ddz_avg_sigma = np.zeros([np.size(grid),np.size(dist_grid)]) 
ddz_avg_theta = np.zeros([np.size(grid),np.size(dist_grid)]) 
for i in range(np.size(dist_grid)):
    ddz_avg_sigma[:,i] = np.gradient(sigma_avg_grid[:,i],z)
    ddz_avg_theta[:,i] = np.gradient(theta_avg_grid[:,i],z)         
    
# compute background N
N2 = np.zeros(np.shape(sigma_avg_grid))
for i in range(np.size(dist_grid)):
    N2[1:,i] = np.squeeze(sw.bfrq(salin_avg_grid[:,i], theta_avg_grid[:,i], grid_p, lat=26.5)[0])  
# f,ax = plt.subplots()
# ax.plot(N2[:,5],grid_p)
# ax.plot(N2[:,7],grid_p)
# plot_pro(ax)    
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
plot_pro(ax2)
# f.savefig('/Users/jake/Desktop/abaco/dg037_8_anoms.png',dpi = 300)
                     
############# Eta_fit / Mode Decomposition #############    
# define G grid 
omega = 0  # frequency zeroed for geostrophic modes
mmax = 60  # highest baroclinic mode to be calculated
nmodes = mmax + 1
    
G, Gz, c = vertical_modes(N2,grid,omega,mmax)

# sample plots of G and Gz
sam_pl = 0
if sam_pl > 0:
    f, (ax1,ax2) = plt.subplots(1,2,sharey=True)
    colors = plt.cm.tab10(np.arange(0,5,1))
    for i in range(5):
        gp = ax1.plot(G[:,i]/np.max(grid),grid,label='Mode # = ' + str(i),color=colors[i,:])
        ax2.plot(Gz[:,i],grid)
    n2p = ax1.plot((np.sqrt(np.nanmean(N2,axis=1))*(1800/np.pi))/10,grid,color='k',label='N(z) [10 cph]')    
    ax1.grid()
    ax1.axis([-1, 1, 0, 5000])
    ax1.set_ylabel('Depth [m]')
    ax1.set_xlabel('Vert. Displacement Stucture')
    ax1.set_title(r"G(z) Vert. Displacement $\eta$ Mode Shapes")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[-1],handles[0],handles[1],handles[2],handles[3],handles[4]],[labels[-1],labels[0],labels[1],labels[2],labels[3],labels[4]],fontsize=10)
    ax2.axis([-4, 4, 0, 5000])
    ax2.set_xlabel('Vert. Structure of Hor. Velocity')
    ax2.set_title("G'(z) Horizontal Velocity Mode Shapes")
    ax2.invert_yaxis()    
    plot_pro(ax2)

# first taper fit above and below min/max limits
# Project modes onto each eta (find fitted eta)
# Compute PE 
eta_fit_depth_min = 50
eta_fit_depth_max = 3800
eta_th_fit_depth_min = 50
eta_th_fit_depth_max = 4200
AG = np.zeros([nmodes, num_profs])
AG_theta = np.zeros([nmodes, num_profs])
Eta_m = np.nan*np.zeros([np.size(grid), num_profs])
Neta = np.nan*np.zeros([np.size(grid), num_profs])
NEta_m = np.nan*np.zeros([np.size(grid), num_profs])
Eta_theta_m = np.nan*np.zeros([np.size(grid), num_profs])
PE_per_mass = np.nan*np.zeros([nmodes, num_profs])
PE_theta_per_mass = np.nan*np.zeros([nmodes, num_profs])
for i in range(num_profs):
    this_eta = df_eta.iloc[:,i][:].copy()
    # obtain matrix of NEta
    Neta[:,i] = N[:,np.int(closest_rec[i])]*this_eta
    this_eta_theta = df_eta_theta.iloc[:,i][:].copy()
    iw = np.where((grid>=eta_fit_depth_min) & (grid<=eta_fit_depth_max))
    if iw[0].size > 1:
        eta_fs = df_eta.iloc[:,i][:].copy() # ETA
        eta_theta_fs = df_eta_theta.iloc[:,i][:].copy()
    
        i_sh = np.where( (bin_depth < eta_fit_depth_min))
        eta_fs.iloc[i_sh[0]] = bin_depth[i_sh]*this_eta.iloc[iw[0][0]]/bin_depth[iw[0][0]]
        i_sh = np.where( (bin_depth < eta_th_fit_depth_min))
        eta_theta_fs.iloc[i_sh[0]] = bin_depth[i_sh]*this_eta_theta.iloc[iw[0][0]]/bin_depth[iw[0][0]]
        
        i_dp = np.where( (grid > eta_fit_depth_max) )
        eta_fs.iloc[i_dp[0]] = (grid[i_dp] - grid[-1])*this_eta.iloc[iw[0][-1]]/(grid[iw[0][-1]]-grid[-1])
        i_dp = np.where( (grid > eta_th_fit_depth_max) )
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
            p38 = ax1.plot(df_eta_theta.iloc[:,i],grid,color='#CD853F',linewidth=.4,label='DG038')
            p38_f = ax1.plot(Eta_theta_m[:,i],grid,'k--',linewidth=.3,label='DG038')
            
            p38_2 = ax2.plot(df_eta.iloc[:,i],grid,color='#CD853F',linewidth=.4,label='DG038')
            p38_f = ax2.plot(Eta_m[:,i],grid,'k--',linewidth=.3)
            # p37_2 = ax2.plot(Neta[:,i],grid,color='#48D1CC',linewidth=.4,label='DG037')
            # p37_f = ax2.plot(NEta_m[:,i],grid,'k--',linewidth=.3) 
        else:
            p37 = ax1.plot(df_eta_theta.iloc[:,i],grid,color='#008080',linewidth=.4,label='DG037')
            p37_f = ax1.plot(Eta_theta_m[:,i],grid,'k--',linewidth=.3,label='DG037')
                
            p37_2 = ax2.plot(df_eta.iloc[:,i],grid,color='#008080',linewidth=.4,label='DG037')
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
    ax2.set_xlabel(r'$\eta_{\sigma_{\theta}}$ [m]',fontsize=14)
    ax1.set_xlabel(r'$\eta_{\theta}$ [m]',fontsize=14)
    ax1.set_ylabel('Depth [m]',fontsize=14)
    ax1.set_title(r'ABACO Vertical $\theta$ Disp.',fontsize=14)
    ax2.set_title(r'$\eta$ Vertical Isopycnal Disp.',fontsize=14)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[0], handles[-2]],[labels[1], labels[-1]],fontsize=12)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend([handles[1], handles[-1]],[labels[1], labels[-1]],fontsize=12)
    ax1.grid()
    plot_pro(ax2)
    # f.savefig('/Users/jake/Desktop/abaco/dg037_8_eta.png',dpi = 300)
    # plt.close()


################### DG Geostrophic Velocity Profiles 
import_ke = si.loadmat('/Users/jake/Documents/geostrophic_turbulence/ABACO_V_energy.mat')
ke_data = import_ke['out']
ke = ke_data['KE'][0][0]
dg_v_0 = ke_data['V_g'][0][0]
dg_v = np.concatenate( (dg_v_0[:,0:28],dg_v_0[:,29:]),axis=1)
# dg_v_m = ke_data['V_m'][0][0]
dg_G = ke_data['G'][0][0]
dg_Gz = ke_data['Gz'][0][0]
dg_bin = ke_data['bin_depth'][0][0]
dg_dep = ke_data['Depth'][0][0]
# dg_c = _data['c'][0][0]
# dg_quiet = ke_data['quiet_prof'][0][0]
dg_np = np.shape(dg_v)[1]

HKE_noise_threshold = 1e-4 # 1e-5
### DG V - HKE est.
dg_AGz = np.zeros([nmodes, dg_np])
dg_v_m_2 = np.nan*np.zeros([np.size(grid), dg_np])
dg_HKE_per_mass = np.nan*np.zeros([nmodes, dg_np])
modest = np.arange(11,nmodes)
dg_good_prof = np.zeros(dg_np)
for i in range(dg_np):    
    # fit to velocity profiles
    this_V = np.interp(grid,np.squeeze(dg_bin),dg_v[:,i])
    iv = np.where( ~np.isnan(this_V) )
    if iv[0].size > 1:
        dg_AGz[:,i] =  np.squeeze(np.linalg.lstsq( np.squeeze(Gz[iv,:]),np.transpose(np.atleast_2d(this_V[iv])))[0]) # Gz(iv,:)\V_g(iv,ip)  
        dg_v_m_2[:,i] =  np.squeeze(np.matrix(Gz)*np.transpose(np.matrix(dg_AGz[:,i])))  #Gz*AGz[:,i];
        dg_HKE_per_mass[:,i] = dg_AGz[:,i]*dg_AGz[:,i]
        ival = np.where( dg_HKE_per_mass[modest,i] >= HKE_noise_threshold )
        if np.size(ival) > 0:
            dg_good_prof[i] = 1 # flag profile as noisy
    else: 
        dg_good_prof[i] = 1 # flag empty profile as noisy as well
        
HKE_noise_threshold_strict = 1e-5 # 1e-5
### DG V - HKE est.
dg_AGz_s = np.zeros([nmodes, dg_np])
dg_v_m_2_s = np.nan*np.zeros([np.size(grid), dg_np])
dg_HKE_per_mass_s = np.nan*np.zeros([nmodes, dg_np])
dg_good_prof_s = np.zeros(dg_np)
for i in range(dg_np):    
    # fit to velocity profiles
    this_V = np.interp(grid,np.squeeze(dg_bin),dg_v[:,i])
    iv = np.where( ~np.isnan(this_V) )
    if iv[0].size > 1:
        dg_AGz_s[:,i] =  np.squeeze(np.linalg.lstsq( np.squeeze(Gz[iv,:]),np.transpose(np.atleast_2d(this_V[iv])))[0]) # Gz(iv,:)\V_g(iv,ip)  
        dg_v_m_2_s[:,i] =  np.squeeze(np.matrix(Gz)*np.transpose(np.matrix(dg_AGz[:,i])))  #Gz*AGz[:,i];
        dg_HKE_per_mass_s[:,i] = dg_AGz[:,i]*dg_AGz[:,i]
        ival = np.where( dg_HKE_per_mass_s[modest,i] >= HKE_noise_threshold_strict)
        if np.size(ival) > 0:
            dg_good_prof_s[i] = 1 # flag profile as noisy
    else: 
        dg_good_prof_s[i] = 1 # flag empty profile as noisy as well        

### SHIP ADCP HKE est. 
# find adcp profiles that are deep enough and fit baroclinic modes to these 
HKE_noise_threshold_adcp = 1e-5
adcp_dist = abaco_ship['adcp_dist']
adcp_depth = abaco_ship['adcp_depth']
adcp_v = abaco_ship['adcp_v']
check = np.zeros(np.size(adcp_dist))
for i in range(np.size(adcp_dist)):
    check[i] = adcp_depth[np.where(~np.isnan(adcp_v[:,i]))[0][-1]]
adcp_in = np.where(check >= 4750)
V = adcp_v[:,adcp_in[0]]/100
V_dist = adcp_dist[adcp_in[0]]      
adcp_np = np.size(V_dist)
AGz = np.zeros([nmodes, adcp_np])
V_m = np.nan*np.zeros([np.size(grid), adcp_np])
HKE_per_mass = np.nan*np.zeros([nmodes, adcp_np])
modest = np.arange(11,nmodes)
good_prof = np.zeros(adcp_np)
for i in range(adcp_np):    
    # fit to velocity profiles
    this_V_0= V[:,i].copy()
    this_V = np.interp(grid,adcp_depth,this_V_0)
    iv = np.where( ~np.isnan(this_V) )
    if iv[0].size > 1:
        AGz[:,i] =  np.squeeze(np.linalg.lstsq( np.squeeze(Gz[iv,:]),np.transpose(np.atleast_2d(this_V[iv])))[0]) # Gz(iv,:)\V_g(iv,ip)  
        V_m[:,i] =  np.squeeze(np.matrix(Gz)*np.transpose(np.matrix(AGz[:,i])))  #Gz*AGz[:,i];
        HKE_per_mass[:,i] = AGz[:,i]*AGz[:,i]
        ival = np.where( HKE_per_mass[modest,i] >= HKE_noise_threshold_adcp )
        if np.size(ival) > 0:
            good_prof[i] = 1 # flag profile as noisy
    else:
        good_prof[i] = 1 # flag empty profile as noisy as well

##### ______ PLOT VELOCITY PROFILES AND COMPARE _________
f, (ax1,ax2) = plt.subplots(1,2)
for i in range(adcp_np):    
    if good_prof[i] < 1:
        ad1 = ax1.plot(V[:,i],adcp_depth,color='#CD853F')
        ax1.plot(V_m[:,i],grid,color='k',linewidth=0.75)
    else:
        ax1.plot(V[:,i],adcp_depth,color='r')  
for i in range(dg_np):
    if dg_good_prof_s[i] < 1:    # dg_quiet[0][i] < 1:
        ad1 = ax2.plot(dg_v[:,i],dg_bin,color='#008080')
        ax2.plot(dg_v_m_2[:,i],grid,color='k',linestyle='--',linewidth=0.75)   
    else:
        ax2.plot(dg_v[:,i],dg_bin,color='r')     
ax1.axis([-.6,.6,0,5500])
ax1.set_xlabel('Meridional Velocity [m/s]')
ax1.set_ylabel('Depth [m]')
ax1.set_title('Shipboard ADCP (' + str(adcp_np) + ' profiles)')
ax1.text(.25,5300,'Noisy = '+ str(np.sum(good_prof)),fontsize=8)         
ax1.invert_yaxis()
ax1.grid()
ax2.axis([-.6,.6,0,5500])
ax2.set_xlabel('Meridional Velocity [m/s]')
ax2.set_title('DG Geostrophic Vel. (' + str(dg_np) + ' profiles)')
ax2.text(-.25,5150,'Noise Thresh. = '+ str(HKE_noise_threshold_strict),fontsize=8) 
ax2.text(-.25,5300,'Noisy = '+ str(np.sum(dg_good_prof_s)),fontsize=8)         
ax2.invert_yaxis()
plot_pro(ax2)
       
################### ENERGY SPECTRA             
avg_PE = np.nanmean(PE_per_mass,1)
avg_PE_theta = np.nanmean(PE_theta_per_mass,1)
avg_KE_adcp = np.nanmean(HKE_per_mass,1)
# avg_KE_adcp = np.nanmean(np.squeeze(HKE_per_mass[:,np.where(good_prof>0)]),1)
avg_KE_dg_strict = np.nanmean(np.squeeze(dg_HKE_per_mass_s[:,np.where(dg_good_prof_s<1)]),1)
avg_KE_dg = np.nanmean(np.squeeze(dg_HKE_per_mass[:,np.where(dg_good_prof<1)]),1)  
avg_KE_dg_all = np.nanmean(dg_HKE_per_mass,1)
f_ref = np.pi*np.sin(np.deg2rad(26.5))/(12*1800)
rho0 = 1025
dk = f_ref/c[1]
sc_x = (1000)*f_ref/c[1:]
PE_SD, PE_GM = PE_Tide_GM(rho0,grid,nmodes,N2,f_ref)
k_h = 1e3*(f_ref/c[1:])*np.sqrt( avg_KE_dg[1:]/avg_PE[1:])
    
# TESTING ADCP DG/KE DIFFERENCES     
f1,ax = plt.subplots()
# for i in range(dg_np):
#     if dg_good_prof[i] < 1:
#         ax.plot(sc_x,dg_HKE_per_mass[1:,i]/dk,color='k')
#     else:
#         ax.plot(sc_x,dg_HKE_per_mass[1:,i]/dk,color='r')
# for i in range(adcp_np):
#     if good_prof[i] < 1:
#         ax.plot(sc_x,HKE_per_mass[1:,i]/dk,color='#8FBC8F')
#     else:
#         ax.plot(sc_x,HKE_per_mass[1:,i]/dk,color='g') 
        
PE_ref = ax.plot(sc_x,avg_PE[1:]/dk,color='k',label=r'PE$_{dg}$',linewidth=1)   
KE_adcp = ax.plot(sc_x,avg_KE_adcp[1:]/dk,color='g',label=r'KE$_{adcp}$',linewidth=2)           
KE_dg_strict = ax.plot(sc_x,avg_KE_dg_strict[1:]/dk,color='#B22222',label=r'KE$_{1e-5}$',linewidth=2)   
KE_dg = ax.plot(sc_x,avg_KE_dg[1:]/dk,color='#FF8C00',label=r'KE$_{1e-4}$',linewidth=2)   
KE_dg_all = ax.plot(sc_x,avg_KE_dg_all[1:]/dk,color='#DAA520',label=r'KE$_{all}$',linewidth=2)  
ax.set_title('KE Noise Threshold Comparison')     
ax.set_xlabel('vertical wavenumber')
ax.set_ylabel('variance per vert. wavenumber')
ax.set_yscale('log')
ax.set_xscale('log')  
ax.axis([10**-2, 1.5*10**1, 10**(-6), 10**(3)])
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[0],handles[1],handles[2],handles[3],handles[4]],[labels[0],labels[1],labels[2],labels[3],labels[4]],fontsize=10)
ax.axis([10**-2, 1.5*10**1, 10**(-4), 10**(3)])
plot_pro(ax)    
    
if plot_eng > 0:
    fig0, ax0 = plt.subplots()
    PE_p = ax0.plot(sc_x,avg_PE[1:]/dk,color='#FF8C00',label=r'PE$_{dg}$')
    ax0.scatter(sc_x,avg_PE[1:]/dk,color='#FF8C00',s=6)
    # limits/scales 
    ax0.plot( [3*10**-1, 3*10**0], [1.5*10**1, 1.5*10**-2],color='k',linewidth=0.75)
    ax0.plot([3*10**-2, 3*10**-1],[7*10**2, ((5/3)*(np.log10(2*10**-1) - np.log10(2*10**-2) ) +  np.log10(7*10**2) )] ,color='k',linewidth=0.75)
    ax0.text(3.3*10**-1,1.3*10**1,'-3',fontsize=8)
    ax0.text(3.3*10**-2,6*10**2,'-5/3',fontsize=8)
    ax0.plot( [1000*f_ref/c[1], 1000*f_ref/c[-2]],[1000*f_ref/c[1], 1000*f_ref/c[-2]],linestyle='--',color='k',linewidth=0.8)
    ax0.text( 1000*f_ref/c[1]-.001, 1000*f_ref/c[1]-0.01, r'f/c$_m$',fontsize=10)
    ax0.plot(sc_x,PE_GM/dk,linestyle='--',color='#808000')
    ax0.text(sc_x[0]-.009,PE_GM[0]/dk,r'$PE_{GM}$')
    # KE 
    # KE_p = ax0.plot(1000*ke_data['f_ref'][0][0][0]/ke_data['c'][0][0][1:],ke[1:]/(dk_ke/1000),color='#483D8B',label='KE')
    # ax0.scatter(1000*ke_data['f_ref'][0][0][0]/ke_data['c'][0][0][1:],ke[1:]/(dk_ke/1000),color='#483D8B',s=6)
    KE_dg = ax0.plot(sc_x,avg_KE_dg[1:]/dk,color='#8B0000',label=r'KE$_{dg}$')
    KE_adcp = ax0.plot(sc_x,avg_KE_adcp[1:]/dk,color='g',label=r'KE$_{adcp}$')
    # ke/pe ratio 
    k_h_p = ax0.plot(sc_x,k_h,color='k',label=r'DG$_{k_h}$')
    ax0.text(sc_x[0]-.008,k_h[0]+.01,r'$k_{h}$',fontsize=10)
    # plot tailoring    
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.axis([10**-2, 1.5*10**1, 10**(-4), 10**(3)])
    ax0.set_xlabel(r'Vertical Wavenumber = Inverse Rossby Radius = $\frac{f}{c}$ [$km^{-1}$]',fontsize=13)
    ax0.set_ylabel('Spectral Density (and Hor. Wavenumber)')
    ax0.set_title('ABACO')
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend([handles[0],handles[1],handles[-2],handles[-1]],[labels[0],labels[1],labels[-2],labels[-1]],fontsize=10)
    plot_pro(ax0)
    # ax0.grid()
    # fig0.savefig('/Users/jake/Desktop/abaco/abaco_energy_all.png',dpi = 200)
    # plt.close()        
        
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
    
    
####### EOF MODE SHAPESSSSS
## find EOFs of dynamic horizontal current (v) mode amplitudes _____DG_____
AGzq = dg_AGz_s #(:,quiet_prof)
nq = np.size(dg_good_prof_s) # good_prof and dg_good_prof
avg_AGzq = np.nanmean(np.transpose(AGzq),axis=0)
AGzqa = AGzq - np.transpose(np.tile(avg_AGzq,[nq,1])) # mode amplitude anomaly matrix
cov_AGzqa = (1/nq)*np.matrix(AGzqa)*np.matrix(np.transpose(AGzqa)) # nmodes X nmodes covariance matrix
var_AGzqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa))))*np.matrix(np.sqrt(np.diag(cov_AGzqa)))
cor_AGzqa = cov_AGzqa/var_AGzqa # nmodes X nmodes correlation matrix

D_AGzqa,V_AGzqa = np.linalg.eig(cov_AGzqa) # columns of V_AGzqa are eigenvectors 
 
EOFseries = np.transpose(V_AGzqa)*np.matrix(AGzqa) # EOF "timeseries' [nmodes X nq]
EOFshape_dg = np.matrix(Gz)*V_AGzqa # depth shape of eigenfunctions [ndepths X nmodes]
EOFshape1_BTpBC1 = np.matrix(Gz[:,0:2])*V_AGzqa[0:2,0] # truncated 2 mode shape of EOF#1
EOFshape2_BTpBC1 = np.matrix(Gz[:,0:2])*V_AGzqa[0:2,1] # truncated 2 mode shape of EOF#2

## find EOFs of dynamic horizontal current (v) mode amplitudes _____ADCP_____
AGzq = AGz #(:,quiet_prof)
nq = np.size(good_prof) # good_prof and dg_good_prof
avg_AGzq = np.nanmean(np.transpose(AGzq),axis=0)
AGzqa = AGzq - np.transpose(np.tile(avg_AGzq,[nq,1])) # mode amplitude anomaly matrix
cov_AGzqa = (1/nq)*np.matrix(AGzqa)*np.matrix(np.transpose(AGzqa)) # nmodes X nmodes covariance matrix
var_AGzqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa))))*np.matrix(np.sqrt(np.diag(cov_AGzqa)))
cor_AGzqa = cov_AGzqa/var_AGzqa # nmodes X nmodes correlation matrix

D_AGzqa,V_AGzqa = np.linalg.eig(cov_AGzqa) # columns of V_AGzqa are eigenvectors 
 
EOFseries = np.transpose(V_AGzqa)*np.matrix(AGzqa) # EOF "timeseries' [nmodes X nq]
EOFshape_adcp = np.matrix(Gz)*V_AGzqa # depth shape of eigenfunctions [ndepths X nmodes]
EOFshape1_BTpBC1 = np.matrix(Gz[:,0:2])*V_AGzqa[0:2,0] # truncated 2 mode shape of EOF#1
EOFshape2_BTpBC1 = np.matrix(Gz[:,0:2])*V_AGzqa[0:2,1] # truncated 2 mode shape of EOF#2  
  
## find EOFs of dynamic vertical displacement (eta) mode amplitudes
# extract noisy/bad profiles 
good_prof = np.where(~np.isnan(AG[2,:]))
num_profs_2 = np.size(good_prof)
AG2 = AG[:,good_prof[0]]
C = np.transpose(np.tile(c,(num_profs_2,1)))
AGs = C*AG2
AGq = AGs[1:,:] # ignores barotropic mode
nqd = num_profs_2
avg_AGq = np.nanmean(AGq,axis=1)
AGqa = AGq - np.transpose(np.tile(avg_AGq,[nqd,1])) # mode amplitude anomaly matrix
cov_AGqa = (1/nqd)*np.matrix(AGqa)*np.matrix(np.transpose(AGqa)) # nmodes X nmodes covariance matrix
var_AGqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGqa))))*np.matrix(np.sqrt(np.diag(cov_AGqa)))
cor_AGqa = cov_AGqa/var_AGqa # nmodes X nmodes correlation matrix
 
D_AGqa,V_AGqa = np.linalg.eig(cov_AGqa) # columns of V_AGzqa are eigenvectors 

EOFetaseries = np.transpose(V_AGqa)*np.matrix(AGqa) # EOF "timeseries' [nmodes X nq]
EOFetashape = np.matrix(G[:,1:])*V_AGqa # depth shape of eigenfunctions [ndepths X nmodes]
EOFetashape1_BTpBC1 = G[:,1:3]*V_AGqa[0:2,0] # truncated 2 mode shape of EOF#1
EOFetashape2_BTpBC1 = G[:,1:3]*V_AGqa[0:2,1] # truncated 2 mode shape of EOF#2    

#### plot mode shapes 
max_plot = 3 
f, (ax1,ax2,ax3) = plt.subplots(1,3) 
n2p = ax1.plot((np.sqrt(np.nanmean(N2,axis=1))*(1800/np.pi)),grid,color='k',label='N(z) [cph]') 
colors = plt.cm.Dark2(np.arange(0,4,1))

for ii in range(max_plot):
    ax1.plot(Gz[:,ii], grid,color='#2F4F4F',linestyle='--')
    p_eof=ax1.plot(-EOFshape_adcp[:,ii], grid,color=colors[ii,:],label='EOF # = ' + str(ii+1),linewidth=2)
handles, labels = ax1.get_legend_handles_labels()    
ax1.legend(handles,labels,fontsize=10)    
ax1.axis([-4,4,0,5000])    
ax1.invert_yaxis()
ax1.set_title('ABACO EOF Mode Shapes (ADCP)')
ax1.set_ylabel('Depth [m]')
ax1.set_xlabel('Hor. Vel. Mode Shapes (ADCP)')
ax1.grid()

for ii in range(max_plot):
    ax2.plot(Gz[:,ii], grid,color='#2F4F4F',linestyle='--')
    p_eof=ax2.plot(-EOFshape_dg[:,ii], grid,color=colors[ii,:],label='EOF # = ' + str(ii+1),linewidth=2)
handles, labels = ax2.get_legend_handles_labels()    
ax2.legend(handles,labels,fontsize=10)    
ax2.axis([-4,4,0,5000])    
ax2.invert_yaxis()
ax2.set_title('EOF Mode Shapes (DG)')
ax2.set_xlabel('Hor. Vel. Mode Shapes (DG)')
ax2.grid()

for ii in range(max_plot):
    ax3.plot(G[:,ii+1]/np.max(grid),grid,color='#2F4F4F',linestyle='--')
    p_eof_eta=ax3.plot(-EOFetashape[:,ii]/np.max(grid), grid,color=colors[ii,:],label='EOF # = ' + str(ii+1),linewidth=2)
handles, labels = ax3.get_legend_handles_labels()    
ax3.legend(handles,labels,fontsize=10)    
ax3.axis([-.7,.7,0,5000])    
ax3.set_title('EOF Mode Shapes (DG)')
ax3.set_xlabel('Vert. Disp. Mode Shapes')
ax3.invert_yaxis()
plot_pro(ax3)

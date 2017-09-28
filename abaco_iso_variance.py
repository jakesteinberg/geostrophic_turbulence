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
# functions I've written 
from grids import make_bin
from mode_decompositions import vertical_modes

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
grid_p = sw.pres(grid,lat_in)
den_grid = np.arange(24.5, 28 , 0.02)
dist_grid_s = np.arange(5,125,0.005)
dist_grid = np.arange(5,125,2)
# depth_grid = np.arange(0,5000,5)
# grid = depth_grid 

# output dataframe 
df_t = pd.DataFrame()
df_s = pd.DataFrame()
df_d = pd.DataFrame()
df_den = pd.DataFrame()
time_rec = np.zeros([dg_list.shape[0], 2])

# plot controls 
plot_plan = 0 
plot_cross = 0
p_eta = 1

####################################################################
##### iterate for each dive cycle ######

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
    time_rec[count,:] = np.array([ serial_date_time_dive, serial_date_time_climb ])
        
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
        
# compute average density/temperature as a function of distance offshore       
count = 0  
mean_dist = np.nanmean(df_d,0)  
theta_avg_grid = np.zeros([np.size(grid),np.size(dist_grid)])
salin_avg_grid = np.zeros([np.size(grid),np.size(dist_grid)]) 
sigma_avg_grid = np.zeros([np.size(grid),np.size(dist_grid)])  
for i in dist_grid:
    mask = (mean_dist > i-7.5) & (mean_dist < i+7.5)
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
    ax1.contour(dist_grid,grid,sigma_avg_grid,colors='k',
        levels=[25,26,26.2,26.4,26.8,27,27.1,27.2,27.3,27.4,27.5,27.6,27.7,27.75,27.8,27.85,27.9])
    ax1.invert_yaxis()
    
    c2s = plt.cm.jet(np.linspace(0,1,125)) 
    for i in range(np.size(dist_grid)):
        ax2.plot(sigma_avg_grid[:,i],grid,color=c2s[i,:])
    ax2.axis([25,28,0,5000])    
    ax2.invert_yaxis()        
    # plt.show()
    f.savefig('/Users/jake/Desktop/abaco/dg037_8_bin_depth_den.png',dpi = 300)    
    
#######################################################################        
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

    plot2 = 0
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
        ax2.set_xlabel(r'$\eta_{\sigma_{\theta}}$ [m]')
        ax1.set_xlabel(r'$\eta_{\theta}$ [m]')
        ax1.set_ylabel('Depth [m]')
        ax1.set_title(r'ABACO Vertical $\theta$ Disp.')
        ax2.set_title('Vertical Isopycnal Disp.')
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend([handles[1], handles[-1]],[labels[1], labels[-1]],fontsize=8)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend([handles[1], handles[-1]],[labels[1], labels[-1]],fontsize=8)
        ax1.grid()
        ax2.grid()
        f.savefig('/Users/jake/Desktop/abaco/dg037_8_eta_theta.png',dpi = 300)
        plt.close()
    
    ############# Eta_fit / Mode Decomposition #############
    
    # compute N
    N2 = np.zeros(np.shape(df_eta))
    for i in range(np.size(dist_grid)):
        N2[1:,i] = np.squeeze(sw.bfrq(salin_avg_grid[:,i], theta_avg_grid[:,i], grid_p, lat=26.5)[0])  
    lz = np.where(N2 < 0)   
    lnan = np.isnan(N2)
    N2[lz] = 0 
    N2[lnan] = 0
    N = np.sqrt(N2)    
    # define G grid 
    omega = 0  # frequency zeroed for geostrophic modes
    mmax = 40  # highest baroclinic mode to be calculated
    nmodes = mmax + 1
    
    G, Gz, c = vertical_modes(N2,grid,omega,mmax)
    
    # first taper fit above and below min/max limits
    sz = df_eta.shape 
    num_profs = sz[1]
    eta_fit_depth_min = 50
    eta_fit_depth_max = 4250
    AG = np.zeros([nmodes, num_profs])
    Eta_m = np.nan*np.zeros([np.size(grid), num_profs])
    PE_per_mass = np.nan*np.zeros([nmodes, num_profs])
    for i in range(num_profs):
        this_eta = df_eta.iloc[:,i][:]
        iw = np.where((grid>=eta_fit_depth_min) & (grid<=eta_fit_depth_max))
        if iw[0].size > 1:
            eta_fs = df_eta.iloc[:,i][:]
        
            i_sh = np.where( (bin_depth < eta_fit_depth_min))
            eta_fs.iloc[i_sh[0]] = bin_depth[i_sh]*this_eta.iloc[iw[0][0]]/bin_depth[iw[0][0]]
        
            i_dp = np.where( (grid > eta_fit_depth_max) )
            eta_fs.iloc[i_dp[0]] = (grid[i_dp] - grid[-1])*this_eta.iloc[iw[0][-1]]/(grid[iw[0][-1]]-grid[-1])
            
            AG[1:,i] = np.squeeze(np.linalg.lstsq(G[:,1:],np.transpose(np.atleast_2d(eta_fs)))[0])
            Eta_m[:,i] = np.squeeze(np.matrix(G)*np.transpose(np.matrix(AG[:,i])))
            PE_per_mass[:,i] = (1/2)*AG[:,i]*AG[:,i]*c*c
            # plt.plot(eta_fs,grid,'k',linewidth=2)    
            # plt.plot(Eta_m[:,i],grid,'r--',linewidth=1)    
             
    # plt.axis([-600, 600, 0, 5000])        
    # plt.show()    
    
    avg_PE = np.nanmean(PE_per_mass,1)
    f_ref = np.pi*np.sin(np.deg2rad(26.5))/(12*1800)
    dk = f_ref/c[1]
    sc_x = (1000)*f_ref/c[1:]
    plt.plot(sc_x,avg_PE[1:]/dk)
    plt.yscale('log')
    plt.xscale('log')
    plt.axis('square')
    plt.axis([10**-3, 10**1, 10**(-2), 10**(2)])
    plt.grid()
    plt.xlabel(r'Vertical Wavenumber = Inverse Rossby Radius = $\frac{f}{c}$ [$km**(-1)$]',fontsize=13)
    plt.ylabel('Spectral Density of Potential Energy')
    plt.title('ABACO')
    plt.show()
    
    # solves G''(z) + (N^2(z) - omega^2)G(z)/c^2 = 0 
    #   subject to G'(0) = gG(0)/c^2 (free surface) & G(-D) = 0 (flat bottom)
    # G(z) is normalized so that the vertical integral of (G'(z))^2 is D
    # G' is dimensionless, G has dimensions of length
    
    # - N is buoyancy frequency [s^-1] (nX1 vector)
    # - depth [m] (maximum depth is considered the sea floor) (nX1 vector)
    # - omega is frequency [s^-1] (scalar)
    # - mmax is the highest baroclinic mode calculated
    # - m=0 is the barotropic mode
    # - 0 < m <= mmax are the baroclinic modes
    # - Modes are calculated by expressing in finite difference form 1) the
    #  governing equation for interior depths (rows 2 through n-1) and 2) the
    #  boundary conditions at the surface (1st row) and the bottome (last row).
    # - Solution is found by solving the eigenvalue system A*x = lambda*B*x

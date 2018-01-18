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
from scipy.integrate import cumtrapz
import seaborn as sns
import pickle
# functions I've written 
from grids import make_bin, collect_dives
from mode_decompositions import vertical_modes, PE_Tide_GM
from toolkit import cart2pol, pol2cart, plot_pro

############ Plot plan view of station BATS and glider sampling pattern for 2015

## bathymetry 
bath = '/Users/jake/Desktop/bats/bats_bathymetry/GEBCO_2014_2D_-67.7_29.8_-59.9_34.8.nc'
bath_fid = netcdf.netcdf_file(bath,'r',mmap=False if sys.platform == 'darwin' else mmap, version=1)
bath_lon = bath_fid.variables['lon'][:]
bath_lat = bath_fid.variables['lat'][:]
bath_z = bath_fid.variables['elevation'][:]

## gliders 
dg_list = glob.glob('/Users/jake/Documents/baroclinic_modes/DG/sg035_BATS_2015/p*.nc')

# physical parameters 
g = 9.81
rho0 = 1027
bin_depth = np.concatenate([np.arange(0,150,10), np.arange(150,300,10), np.arange(300,5000,20)])
ref_lat = 31.8
lat_in = 31.7
lon_in = 64.2
grid = bin_depth[1:-1]
grid_p = sw.pres(grid,lat_in)
z = -1*grid
# mode parameters 
omega = 0  # frequency zeroed for geostrophic modes
mmax = 60  # highest baroclinic mode to be calculated
nmodes = mmax + 1  
deep_shr_max = 0.1 # maximum allowed deep shear [m/s/km]
deep_shr_max_dep = 3500 # minimum depth for which shear is limited [m]

# PLOTTING SWITCHES 
plot_bath = 0
plot_cross = 0

# LOAD DATA 
GD = netcdf.netcdf_file('BATs_2015_gridded_2.nc','r')
df_den = pd.DataFrame(np.float64(GD.variables['Density'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_t = pd.DataFrame(np.float64(GD.variables['Temperature'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_s = pd.DataFrame(np.float64(GD.variables['Salinity'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_lon = pd.DataFrame(np.float64(GD.variables['Longitude'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_lat = pd.DataFrame(np.float64(GD.variables['Latitude'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_avg_lon = np.nanmean(df_lon,axis=0)
df_avg_lat = np.nanmean(df_lat,axis=0)
dac_u = GD.variables['DAC_u'][:]
dac_v = GD.variables['DAC_v'][:]
time_rec = GD.variables['time_rec'][:]
time_sta_sto = GD.variables['time_start_stop'][:]
heading_m_rec = GD.variables['heading_record'][:]
target_rec = GD.variables['target_record'][:]
gps_rec = GD.variables['gps_record'][:]/100 
profile_list = np.float64(GD.variables['dive_list'][:]) - 35000

##################################       
##################################

# look at headings per each arm of the bowtie pattern  ORDERING: 6, 2, 12, 8 
heading_test1 = np.where( (target_rec > 7) & (target_rec < 9) )  # 8
heading_test2 = np.where( (target_rec > 11) & (target_rec < 13) ) # 12
heading_test3 = np.where( (target_rec > 5) & (target_rec < 7) )  # 6
heading_test4 = np.where( (target_rec > 1) & (target_rec < 3) )  # 2
# the nan's in the target_rec represent times when the glider target was station BATS (the center of the sample pattern)
fig0, ax0 = plt.subplots()
ax0.plot(df_avg_lon[heading_test1],df_avg_lat[heading_test1])
ax0.plot(df_avg_lon[heading_test2],df_avg_lat[heading_test2])
ax0.plot(df_avg_lon[heading_test3],df_avg_lat[heading_test3])
ax0.plot(df_avg_lon[heading_test4],df_avg_lat[heading_test4])
ax0.grid()
plt.gca().set_aspect('equal')
plot_pro(ax0)

# compute heading for each profile (or really dive)
heading_rec = np.nan*np.zeros(len(profile_list))
for jj in range(0,len(profile_list),2):
    if jj < (len(profile_list)-2):
        lon_a = gps_rec[jj,3]
        lat_a = gps_rec[jj,2]
        lon_b = gps_rec[jj+2,1]
        lat_b = gps_rec[jj+2,0]
        A = np.sin(np.deg2rad(lon_b) - np.deg2rad(lon_a))*np.cos(np.deg2rad(lat_b))
        B = np.cos(np.deg2rad(lat_a))*np.sin(np.deg2rad(lat_b))-np.sin(np.deg2rad(lat_a))*np.cos(np.deg2rad(lat_b))*np.cos(np.deg2rad(lon_b)-np.deg2rad(lon_a))
        C = np.arctan2(A,B)
        D = np.rad2deg(np.mod(C,2*np.pi))
        heading_rec[jj] = D
        heading_rec[jj+1] = D
    else:  
        lon_a = gps_rec[jj,3]
        lat_a = gps_rec[jj,2]
        lon_b = df_lon.iloc[0,jj+1]
        lat_b = df_lat.iloc[0,jj+1]
        A = np.sin(np.deg2rad(lon_b) - np.deg2rad(lon_a))*np.cos(np.deg2rad(lat_b))
        B = np.cos(np.deg2rad(lat_a))*np.sin(np.deg2rad(lat_b))-np.sin(np.deg2rad(lat_a))*np.cos(np.deg2rad(lat_b))*np.cos(np.deg2rad(lon_b)-np.deg2rad(lon_a))
        C = np.arctan2(A,B)
        D = np.rad2deg(np.mod(C,2*np.pi))
        heading_rec[jj] = D
        heading_rec[jj+1] = D  
heading_rec[np.where(heading_rec == 0)] = np.nan

# fig0, ax0 = plt.subplots()
# ax0.scatter(gps_rec[100,3],gps_rec[100,2],color='r')
# ax0.scatter(gps_rec[100+2,3],gps_rec[100+2,2],color='b')
# ax0.grid()
# w = 1/np.cos(np.deg2rad(ref_lat))
# ax0.axis([-65.5, -63.35, 31.2, 32.7])
# ax0.set_aspect(w)
# plot_pro(ax0)

fig0, ax0 = plt.subplots()
# ax0.scatter(df_lon.iloc[:,0:23],df_lat.iloc[:,0:23])
# ax0.scatter(df_lon.iloc[:,24:30],df_lat.iloc[:,24:30],color='r')
ax0.scatter(df_lon.iloc[:,128:134],df_lat.iloc[:,128:134])
w = 1/np.cos(np.deg2rad(ref_lat))
ax0.axis([-65.5, -63.35, 31.2, 32.7])
ax0.set_aspect(w)
plot_pro(ax0)

# need to loop over each of the 4 arms of the sampling pattern. 
# once cross track velocity is found, need to reference this to the heading associated with each profile (sample heading for two profiles in a dive-cycle)
target = np.array([ [5,7], [1,2], [11,13], [7,9] ]) 
heading = np.array([[200,300],[100,200]])
# outputs from all loops 
N2_out = np.ones((np.size(grid),2))
Eta = []
Eta_theta = []
V = [] 
Time = []
heading_out = []
target_out = []
for main in range(2):
    head_low = heading[main,0]
    head_high = heading[main,1]
    targ_low = target[main,0]
    targ_high = target[main,1]
    # plan view plot (for select heading)   
    if plot_bath > 0:
        heading_mask = np.where( (heading_rec > head_low) & (heading_rec < head_high) ) 
        heading_mask_out = np.where( (heading_rec < head_low) | (heading_rec > head_high) )
        target_mask = np.where( (target_rec > targ_low) & (target_rec < targ_high) )
        target_mask_out = np.where( (target_rec < targ_low) | (target_rec > targ_high) )
        
        levels = [ -5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500 , 0]
        fig0, ax0 = plt.subplots()
        cmap = plt.cm.get_cmap("Blues_r")
        cmap.set_over('#808000') # ('#E6E6E6')
        bc = ax0.contourf(bath_lon,bath_lat,bath_z,levels,cmap='Blues_r',extend='both',zorder=0)
        # ax0.contourf(bath_lon,bath_lat,bath_z,[-5, -4, -3, -2, -1, 0, 100, 1000], cmap = 'YlGn_r')
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        bcl = ax0.contour(bath_lon,bath_lat,bath_z,[-4500, -4000],colors='k',zorder=0)
        ml = [(-65,31.5),(-64.4, 32.435)]
        ax0.clabel(bcl,manual = ml, inline_spacing=-3, fmt='%1.0f',colors='k')      
        dg_a = ax0.plot(df_lon.iloc[:,target_mask_out[0]],df_lat.iloc[:,target_mask_out[0]],color='#8B0000',linewidth=1.5,
            label='All Dives (' + str(int(profile_list[0])) + '-' + str(int(profile_list[-2])) + ')',zorder=1) 
        dg_s = ax0.plot(df_lon.iloc[:,target_mask[0]],df_lat.iloc[:,target_mask[0]],color='#FF4500',linewidth=2, label = 'Target:' + str(np.mean([targ_low,targ_high])),zorder=1) 
        sta_b = ax0.scatter(-(64+(10/60)), 31 + (40/60),s=40,color='#E6E6FA',zorder=2,edgecolors='w')
        ax0.text(-(64+(10/60)) + .1, 31 + (40/60)-.07,'BATS',color='w')
        # ax0.scatter(np.nanmean(df_lon.iloc[:,heading_mask[0]],0),np.nanmean(df_lat.iloc[:,heading_mask[0]],0),s=20,color='g')       
        w = 1/np.cos(np.deg2rad(ref_lat))
        ax0.axis([-65.5, -63.35, 31.2, 32.7])
        ax0.set_aspect(w)
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig0.colorbar(bc, cax=cax, label='[m]')
        ax0.set_xlabel('Longitude')
        ax0.set_ylabel('Latitude')   
        t_s = datetime.date.fromordinal(np.int( np.min(time_rec) ))
        t_e = datetime.date.fromordinal(np.int( np.max(time_rec) ))
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend([handles[0],handles[-1]],[labels[0], labels[-1]],fontsize=10)
        ax0.set_title('Select BATS Transects (DG35): ' + np.str(t_s.month) + '/' + np.str(t_s.day) + '/' + np.str(t_s.year) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day) + '/' + np.str(t_e.year))
        plt.tight_layout()
        ax0.grid()
        plot_pro(ax0)
        # plt.show()
        # fig0.savefig('/Users/jake/Desktop/bats/plan_200_300.png',dpi = 200)


    ############ SELECT ALL TRANSECTS ALONG A HEADING (or heading for the same target) AND COMPUTE VERTICAL DISPLACEMENT AND HORIZONTAL VELOCITY 
    # select only dives along desired heading
    heading_mask = np.where( (heading_rec > head_low) & (heading_rec < head_high) ) 
    target_mask = np.where( (target_rec > targ_low) & (target_rec < targ_high) )
    df_den_in = df_den.iloc[:,target_mask[0]]
    df_t_in = df_t.iloc[:,target_mask[0]]
    df_s_in = df_s.iloc[:,target_mask[0]]
    df_lon_in = df_lon.iloc[:,target_mask[0]]
    df_lat_in = df_lat.iloc[:,target_mask[0]]
    time_in = time_sta_sto[target_mask[0]]
    head_in = heading_rec[target_mask[0]]

    # average background properties of profiles along these transects (average for all dives along this transect )
    sigma_theta_avg = np.array(np.nanmean(df_den_in,1))
    theta_avg = np.array(np.nanmean(df_t_in,1))
    salin_avg = np.array(np.nanmean(df_s_in,1))
    ddz_avg_sigma = np.gradient(sigma_theta_avg,z)
    ddz_avg_theta = np.gradient(theta_avg,z)
    # next few lines only need for mode fitting (done in other .py file)
    # N2 = np.nan*np.zeros(np.size(sigma_theta_avg))
    # N2[1:] = np.squeeze(sw.bfrq(salin_avg, theta_avg, grid_p, lat=ref_lat)[0])  
    # lz = np.where(N2 < 0)   
    # lnan = np.isnan(N2)
    # N2[lz] = 0 
    # N2[lnan] = 0
    # N = np.sqrt(N2)   
    # N2_out[:,main] = N2
    # computer vertical mode shapes 
    # G, Gz, c = vertical_modes(N2,grid,omega,mmax)

    # PREPARE FOR TRANSECT ANALYSIS 
    # dive list to consider
    dives = np.array(df_den_in.columns) - 35000

    # section out dives into continuous transect groups 
    if main < 1:
        to_consider = 14
    else:
        to_consider = 16
    dive_iter = np.array(dives[0])
    time_iter = np.array(time_in[0])
    head_iter = np.array(head_in[0])
    dive_out = {}
    time_out = {}
    head_out = {}
    for i in range(to_consider):
        if i < 1:
            this_dive = dives[0]
            this_time = time_in[0]
            this_head = head_in[0]
        else:
            this_dive = dive_iter[i]   
            this_time = time_iter[i]   
            this_head = head_iter[i]           
    
        dive_group = np.array([this_dive]) 
        time_group = np.array([this_time]) 
        head_group = np.array([this_head])
        up_o = np.where(dives==this_dive)[0]
        for j in dives[up_o[0]+1:]:
            if j - dive_group[-1] < 1:
                dive_group = np.append(dive_group,j)  
                t_coor = np.where(dives==j)[0][0]
                time_group = np.append(time_group,time_in[t_coor])   
                head_group = np.append(head_group,head_in[t_coor])
             
        dive_out[i] = dive_group
        time_out[i] = time_group
        head_out[i] = head_group
        up_n = np.where(dives==dive_group[-1])[0]
        dive_iter = np.array(np.append(dive_iter,dives[up_n[0]+1])) # starting dive of each transect 
        time_iter = np.array(np.append(time_iter,time_in[up_n[0]+1])) 
        head_iter = np.array(np.append(head_iter,head_in[up_n[0]+1]))   
    
    # loop over each dive_group (looking for transects that are long enough to carry out m/w shear computation and that headings are consistent enough)
    # want to select transects that are at least 2 dive-cycles long
    ndives_in_trans = np.nan*np.zeros(to_consider)
    std_head_in_trans = np.nan*np.zeros(to_consider)
    for l in range(to_consider):
        ndives_in_trans[l] = np.size(dive_out[l])/2
        std_head_in_trans[l] = np.nanstd(head_out[l])
    # choose only transects that have two or more dives   
    good = np.where( (ndives_in_trans >= 2) & (std_head_in_trans < 40)  )

    ######### MAIN LOOP OVER EACH TRANSECT (EACH TRANSECT CONTAINS AT LEAST 3 DIVE CYCLES)
    for master in range(np.size(good)): 
        ii = good[0][master]
        this_set = dive_out[ii] + 35000
        this_set_time = time_out[ii]
        this_set_head = head_out[ii]
        df_den_set = df_den_in[this_set] 
        df_theta_set = df_t_in[this_set] 
        df_lon_set = df_lon_in[this_set]
        df_lat_set = df_lat_in[this_set]  

        # total number of dive cycles within this transect 
        dive_cycs = np.unique(np.floor(this_set))
        # pair dac_u,v with each M/W center
        dive_list = np.array(df_den_in.columns)
        inn = np.where( (dive_list >= this_set[0]) & (dive_list <= this_set[-1]) )
        dac_u_inn = dac_u[inn]
        dac_v_inn = dac_v[inn]

        info = np.nan*np.zeros( (3,np.size(this_set)-1))
        sigma_theta_out = np.nan*np.zeros( (np.size(grid),np.size(this_set)-1))
        shear = np.nan*np.zeros( (np.size(grid),np.size(this_set)-1))
        eta = np.nan*np.zeros( (np.size(grid),np.size(this_set)-1))
        eta_theta = np.nan*np.zeros( (np.size(grid),np.size(this_set)-1) )
        sigth_levels = np.concatenate([ np.arange(23,27.5,0.5), np.arange(27.2,27.8,0.2), np.arange(27.7,27.9,0.02)])
        isopycdep = np.nan*np.zeros( (np.size(sigth_levels), np.size(this_set)))
        isopycx = np.nan*np.zeros( (np.size(sigth_levels), np.size(this_set)))
        Vbt = np.nan*np.zeros( np.size(this_set) )
        Ds = np.nan*np.zeros( np.size(this_set) )
        dist = np.nan*np.zeros( np.shape(df_lon_set) )
        dist_st = 0
        distance = 0 
        #### LOOP OVER EACH DIVE CYCLE PROFILE AND COMPUTE SHEAR AND ETA (M/W PROFILING)  
        if np.size(this_set) < 5:
            order_set = [0,2] # go from 0,2 (because each transect only has 2 dives)
        elif (np.size(this_set) > 5) & (np.size(this_set) < 7):
            order_set = [0,2,4] # go from 0,2,4 (because each transect only has 3 dives)
        else:
            order_set = [0,2,4,6] # go from 0,2,4,6 (because each transect only has 4 dives)    
        
        for i in order_set:
            # M 
            lon_start = df_lon_set.iloc[0,i]
            lat_start = df_lat_set.iloc[0,i]
            lon_finish = df_lon_set.iloc[0,i+1]
            lat_finish = df_lat_set.iloc[0,i+1]
            lat_ref = 0.5*(lat_start + lat_finish);
            f = np.pi*np.sin(np.deg2rad(lat_ref))/(12*1800);    # Coriolis parameter [s^-1]
            dxs_m = 1.852*60*np.cos(np.deg2rad(lat_ref))*(lon_finish - lon_start);	# zonal sfc disp [km]
            dys_m = 1.852*60*(lat_finish - lat_start);    # meridional sfc disp [km]    
            ds, ang_sfc_m = cart2pol(dxs_m, dys_m) 
    
            dx = 1.852*60*np.cos(np.deg2rad(lat_ref))*( np.concatenate([np.array(df_lon_set.iloc[:,i]),np.flipud(np.array(df_lon_set.iloc[:,i+1]))]) - df_lon_set.iloc[0,i] )
            dy = 1.852*60*(np.concatenate([np.array(df_lat_set.iloc[:,i]),np.flipud(np.array(df_lat_set.iloc[:,i+1]))]) - df_lat_set.iloc[0,i] )
            ss,ang = cart2pol(dx, dy)
            xx, yy = pol2cart(ss,ang - ang_sfc_m)
            length1 = np.size(np.array(df_lon_set.iloc[:,i]))
            dist[:,i] = dist_st + xx[0:length1]
            dist[:,i+1] = dist_st + np.flipud(xx[length1:])
            dist_st = dist_st + ds
    
            distance = distance + np.nanmedian(xx) # 0.5*ds # distance for each velocity estimate    # np.nanmedian(xx) #
            Ds[i] = distance 
            DACe = dac_u_inn[i] # zonal depth averaged current [m/s]
            DACn = dac_v_inn[i] # meridional depth averaged current [m/s]
            mag_DAC, ang_DAC = cart2pol(DACe, DACn);
            DACat, DACpot = pol2cart(mag_DAC,ang_DAC - ang_sfc_m);
            Vbt[i] = DACpot; # across-track barotropic current comp (>0 to left)

            # W 
            lon_start = df_lon_set.iloc[0,i]
            lat_start = df_lat_set.iloc[0,i]
            if i >= order_set[-1]:
                lon_finish = df_lon_set.iloc[0,-1]
                lat_finish = df_lat_set.iloc[0,-1]
                DACe = np.nanmean( [[dac_u_inn[i]], [dac_u_inn[-1]]] ) # zonal depth averaged current [m/s]
                DACn = np.nanmean( [[dac_v_inn[i]], [dac_v_inn[-1]]] ) # meridional depth averaged current [m/s]
            else:
                lon_finish = df_lon_set.iloc[0,i+3]
                lat_finish = df_lat_set.iloc[0,i+3]
                DACe = np.nanmean( [[dac_u_inn[i]], [dac_u_inn[i+2]]] ) # zonal depth averaged current [m/s]
                DACn = np.nanmean( [[dac_v_inn[i]], [dac_v_inn[i+2]]] ) # meridional depth averaged current [m/s]
            lat_ref = 0.5*(lat_start + lat_finish);
            f = np.pi*np.sin(np.deg2rad(lat_ref))/(12*1800);    # Coriolis parameter [s^-1]
            dxs = 1.852*60*np.cos(np.deg2rad(lat_ref))*(lon_finish - lon_start);	# zonal sfc disp [km]
            dys = 1.852*60*(lat_finish - lat_start)   # meridional sfc disp [km]
            ds_w, ang_sfc_w = cart2pol(dxs, dys)
            distance = distance + (ds-np.nanmedian(xx)) # distance for each velocity estimate 
            Ds[i+1] = distance
            mag_DAC, ang_DAC = cart2pol(DACe, DACn)
            DACat, DACpot = pol2cart(mag_DAC,ang_DAC - ang_sfc_w)
            Vbt[i+1] = DACpot; # across-track barotropic current comp (>0 to left)
        
            shearM = np.nan*np.zeros(np.size(grid))
            shearW = np.nan*np.zeros(np.size(grid))
            etaM = np.nan*np.zeros(np.size(grid))
            etaW = np.nan*np.zeros(np.size(grid))
            eta_thetaM = np.nan*np.zeros(np.size(grid))
            eta_thetaW = np.nan*np.zeros(np.size(grid))
            sigma_theta_pa_M = np.nan*np.zeros(np.size(grid))
            sigma_theta_pa_W = np.nan*np.zeros(np.size(grid))
            lon_pa_M = np.nan*np.zeros(np.size(grid))
            lon_pa_W = np.nan*np.zeros(np.size(grid))
            lat_pa_M = np.nan*np.zeros(np.size(grid))
            lat_pa_W = np.nan*np.zeros(np.size(grid))
            # LOOP OVER EACH BIN_DEPTH
            for j in range(np.size(grid)):
                # find array of indices for M / W sampling 
                if i < 2:      
                    c_i_m = np.arange(i,i+3) 
                    # c_i_m = []; % omit partial "M" estimate
                    c_i_w = np.arange(i,i+4) 
                elif (i >= 2) and (i < this_set.size-2):
                    c_i_m = np.arange(i-1,i+3) 
                    c_i_w = np.arange(i,i+4) 
                elif i >= this_set.size-2:
                    c_i_m = np.arange(i-1,this_set.size) 
                    # c_i_m = []; % omit partial "M" estimated
                    c_i_w = []
                nm = np.size(c_i_m); nw = np.size(c_i_w)

                # for M profile compute shear and eta 
                if nm > 2 and np.size(df_den_set.iloc[j,c_i_m]) > 2:
                    sigmathetaM = df_den_set.iloc[j,c_i_m]
                    sigma_theta_pa_M[j] = np.nanmean(sigmathetaM) # average density across 4 profiles 
                    lon_pa_M[j] = np.nanmean( df_lon_set.iloc[j,c_i_m] ) # avg lat/lon across M/W profiles
                    lat_pa_M[j] = np.nanmean( df_lat_set.iloc[j,c_i_m] )
                    thetaM = df_theta_set.iloc[j,c_i_m]
                    imv = ~np.isnan(np.array(df_den_set.iloc[j,c_i_m]))
                    c_i_m_in = c_i_m[imv]
        
                if np.size(c_i_m_in) > 1:
                    xM = 1.852*60*np.cos(np.deg2rad(lat_ref))*( df_lon_set.iloc[j,c_i_m_in] - df_lon_set.iloc[ j,c_i_m_in[0] ] ) # E loc [km]
                    yM = 1.852*60*(df_lat_set.iloc[j,c_i_m_in] - df_lat_set.iloc[j,c_i_m_in[0] ] ) # N location [km]
                    XXM = np.concatenate( [ np.ones( (np.size(sigmathetaM[imv]),1)), np.transpose(np.atleast_2d(np.array(xM))), np.transpose(np.atleast_2d(np.array(yM))) ],axis=1)
                    d_anom0M = sigmathetaM[imv] - np.nanmean(sigmathetaM[imv])
                    ADM = np.squeeze( np.linalg.lstsq( XXM, np.transpose(np.atleast_2d(np.array(d_anom0M))) )[0] )
                    drhodxM = ADM[1]   # [zonal gradient [kg/m^3/km]
                    drhodyM = ADM[2]   # [meridional gradient [kg/m^3km]
                    drhodsM, ang_drhoM = cart2pol(drhodxM, drhodyM);
                    drhodatM, drhodpotM = pol2cart(drhodsM, ang_drhoM - ang_sfc_m)
                    shearM[j] = -g*drhodatM/(rho0*f) # shear to port of track [m/s/km]
                    if (np.abs(shearM[j]) > deep_shr_max) and grid[j] >= deep_shr_max_dep: 
                        shearM[j] = np.sign(shearM[j])*deep_shr_max
                    etaM[j] = (sigma_theta_avg[j] - np.nanmean(sigmathetaM[imv]) )/ddz_avg_sigma[j] 
                    eta_thetaM[j] = (theta_avg[j] - np.nanmean(thetaM[imv]) )/ddz_avg_theta[j]   
        
                # for W profile compute shear and eta 
                if nw > 2 and np.size(df_den_set.iloc[j,c_i_w]) > 2:
                    sigmathetaW = df_den_set.iloc[j,c_i_w]
                    sigma_theta_pa_W[j] = np.nanmean(sigmathetaW) # average density across 4 profiles 
                    lon_pa_W[j] = np.nanmean( df_lon_set.iloc[j,c_i_w] ) # avg lat/lon across M/W profiles
                    lat_pa_W[j] = np.nanmean( df_lat_set.iloc[j,c_i_w] )
                    thetaW = df_theta_set.iloc[j,c_i_w]
                    iwv = ~np.isnan(np.array(df_den_set.iloc[j,c_i_w]))
                    c_i_w_in = c_i_w[iwv]
        
                if np.sum(c_i_w_in) > 1:
                    xW = 1.852*60*np.cos(np.deg2rad(lat_ref))*( df_lon_set.iloc[j,c_i_w_in] - df_lon_set.iloc[ j,c_i_w_in[0] ] ) # E loc [km]
                    yW = 1.852*60*(df_lat_set.iloc[j,c_i_w_in] - df_lat_set.iloc[j,c_i_w_in[0] ] ) # N location [km]
                    XXW = np.concatenate( [ np.ones( (np.size(sigmathetaW[iwv]),1)), np.transpose(np.atleast_2d(np.array(xW))), np.transpose(np.atleast_2d(np.array(yW))) ],axis=1)
                    d_anom0W = sigmathetaW[iwv] - np.nanmean(sigmathetaW[iwv])
                    ADW = np.squeeze( np.linalg.lstsq( XXW, np.transpose(np.atleast_2d(np.array(d_anom0W))) )[0] )
                    drhodxW = ADW[1]   # [zonal gradient [kg/m^3/km]
                    drhodyW = ADW[2]   # [meridional gradient [kg/m^3km]
                    drhodsW, ang_drhoW = cart2pol(drhodxW, drhodyW);
                    drhodatW, drhodpotW = pol2cart(drhodsW,ang_drhoW - ang_sfc_w)
                    shearW[j] = -g*drhodatW/(rho0*f) # shear to port of track [m/s/km]
                    if (np.abs(shearW[j]) > deep_shr_max) and grid[j] >= deep_shr_max_dep: 
                        shearW[j] = np.sign(shearW[j])*deep_shr_max
                    etaW[j] = (sigma_theta_avg[j] - np.nanmean(sigmathetaW[iwv]) )/ddz_avg_sigma[j]
                    eta_thetaW[j] = (theta_avg[j] - np.nanmean(thetaW[iwv]) )/ddz_avg_theta[j]          

            # END LOOP OVER EACH BIN_DEPTH 
                
            # OUTPUT FOR EACH TRANSECT (3 DIVES)
            # because this is M/W profiling, for a 3 dive transect, only 5 profiles of shear and eta are compiled 
            sigma_theta_out[:,i] = sigma_theta_pa_M
            shear[:,i] = shearM
            eta[:,i] = etaM
            eta_theta[:,i] = eta_thetaM
            info[0,i] = this_set[i] 
            info[1,i] = np.nanmean( lon_pa_M )
            info[2,i] = np.nanmean( lat_pa_M )
            if i < np.size(this_set)-2:
                sigma_theta_out[:,i+1] = sigma_theta_pa_W
                shear[:,i+1] = shearW
                eta[:,i+1] = etaW
                eta_theta[:,i+1] = eta_thetaW
                info[0,i] = this_set[i] 
                info[1,i] = np.nanmean( lon_pa_W )
                info[2,i] = np.nanmean( lat_pa_W )
            
            # ISOPYCNAL DEPTHS ON PROFILES ALONG EACH TRANSECT
            sigthmin = np.nanmin( np.array(df_den_set.iloc[:,i]) )
            sigthmax = np.nanmax( np.array(df_den_set.iloc[:,i]) )
            isigth = np.where( (sigth_levels > sigthmin) & (sigth_levels < sigthmax) )
            isopycdep[isigth,i] = np.interp( sigth_levels[isigth], np.array(df_den_set.iloc[:,i]), grid)
            isopycx[isigth,i] = np.interp( sigth_levels[isigth], np.array(df_den_set.iloc[:,i]), dist[:,i])
        
            sigthmin = np.nanmin( np.array(df_den_set.iloc[:,i+1]) )
            sigthmax = np.nanmax( np.array(df_den_set.iloc[:,i+1]) )
            isigth = np.where( (sigth_levels > sigthmin) & (sigth_levels < sigthmax) )
            isopycdep[isigth,i+1] = np.interp( sigth_levels[isigth], np.array(df_den_set.iloc[:,i+1]), grid)
            isopycx[isigth,i+1] = np.interp( sigth_levels[isigth], np.array(df_den_set.iloc[:,i+1]), dist[:,i+1])
        
            #### END LOOP OVER EACH DIVE IN TRANSECT 
    
        # FOR EACH TRANSECT COMPUTE GEOSTROPHIC VELOCITY 
        Vbc_g = np.nan*np.zeros(np.shape(shear))
        V_g = np.nan*np.zeros( (np.size(grid),np.size(this_set)) )
        for m in range(np.size(this_set)-1):
            iq = np.where( ~np.isnan(shear[:,m]) ) 
            if np.size(iq) > 10:
                z2 = -grid[iq]
                vrel = cumtrapz(0.001*shear[iq,m],x=z2,initial=0)
                vrel_av = np.trapz(vrel/(z2[-1] - z2[0]), x=z2)
                vbc = vrel - vrel_av
                Vbc_g[iq,m] = vbc
                V_g[iq,m] = Vbt[m] + vbc
            else:
                Vbc_g[iq,m] = np.nan
                V_g[iq,m] = np.nan
               
        
        # AVERAGE HEADING FOR EACH DIVE (from difference of GPS points)
        # shear is to port of track (90 deg to the left of the glider heading)    
        u_head = 90
        v_head = 0 
        u_g = np.nan*np.zeros(np.shape(V_g))
        v_g = np.nan*np.zeros(np.shape(V_g))
        for m in range(np.size(this_set)-1):
            this_V_g = V_g[:,m]
            this_heading = this_set_head[m]
            pos_v = np.where(this_V_g > 0)
            neg_v = np.where(this_V_g < 0) 
            pos_head = this_heading-90
            neg_head = this_heading+90
            # consider 0-90, 90-180, 180-270, 270-360
            if (pos_head > 0) & (pos_head < 180):    
                u_g[pos_v,m] = np.cos(np.deg2rad(90-pos_head))*this_V_g[pos_v] 
                v_g[pos_v,m] = np.sin(np.deg2rad(90-pos_head))*this_V_g[pos_v]   
            else:
                u_g[pos_v,m] = np.cos(np.deg2rad(360+90-pos_head))*this_V_g[pos_v]
                v_g[pos_v,m] = np.sin(np.deg2rad(360+90-pos_head))*this_V_g[pos_v]  
            if (neg_head > 0) & (neg_head < 180):    
                u_g[neg_v,m] = np.cos(np.deg2rad(90-neg_head))*this_V_g[neg_v] 
                v_g[neg_v,m] = np.sin(np.deg2rad(90-neg_head))*this_V_g[neg_v]   
            else:
                u_g[neg_v,m] = np.cos(np.deg2rad(360+90-neg_head))*this_V_g[neg_v]
                v_g[neg_v,m] = np.sin(np.deg2rad(360+90-neg_head))*this_V_g[neg_v]     
        
        # fig0, ax0 = plt.subplots()
        # ax0.scatter(df_lon_set.iloc[:,m],df_lat_set.iloc[:,m],color='b')
        # ax0.scatter(df_lon_set.iloc[:,m+1],df_lat_set.iloc[:,m+1],color='r')
        # ax0.quiver(np.nanmean(df_lon_set.iloc[:,m]),np.nanmean(df_lat_set.iloc[:,m]),dac_u_inn[m],dac_v_inn[m],color='b')
        # ax0.quiver(np.nanmean(df_lon_set.iloc[:,m+1]),np.nanmean(df_lat_set.iloc[:,m+1]),dac_u_inn[m+1],dac_v_inn[m+1],color='r')
        # ax0.quiver(np.nanmean(df_lon_set.iloc[:,m]),np.nanmean(df_lat_set.iloc[:,m]),np.nanmean(u_g[:,m]),np.nanmean(v_g[:,m]))
        # ax0.quiver(np.nanmean(df_lon_set.iloc[:,m+1]),np.nanmean(df_lat_set.iloc[:,m+1]),np.nanmean(u_g[:,m]),np.nanmean(v_g[:,m]))
        # plot_pro(ax0)         

        if plot_cross > 0: 
            sns.set(context="notebook", style="whitegrid", rc={"axes.axisbelow": False})
               
            fig0, ax0 = plt.subplots()
            matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
            levels = np.arange(-.26,.26,.02)
            vc = ax0.contourf(Ds,grid,V_g,levels=levels,cmap=plt.cm.coolwarm)
            vcc = ax0.contour(Ds,grid,V_g,levels=levels,colors='k',linewidth=1)
            for p in range(np.size(this_set)):
                ax0.scatter(dist[:,p],grid,s=.75,color='k') 
            dive_label = np.arange(0,np.size(this_set),2)    
            for pp in range(np.size(dive_label)):
                p = dive_label[pp]
                ax0.text(np.nanmax(dist[:,p])-1,np.max(grid[~np.isnan(dist[:,p])])+200, str(int(this_set[p]-35000)))   
            sig_good = np.where(~np.isnan(isopycdep[:,0]) )   
            for p in range(np.size(sig_good[0])):
                ax0.plot(isopycx[sig_good[0][p],:],isopycdep[sig_good[0][p],:],color='#708090',linewidth=.75)              
                ax0.text(np.nanmax(isopycx[sig_good[0][p],:])+2, np.nanmean(isopycdep[sig_good[0][p],:]), str(sigth_levels[sig_good[0][p]]),fontsize=6)    
            ax0.clabel(vcc,inline=1,fontsize=8,fmt='%1.2f',color='k')
            ax0.axis([0, np.max(Ds)+4, 0, 4750]) 
            ax0.invert_yaxis()
            ax0.set_xlabel('Distance along transect [km]')
            ax0.set_ylabel('Depth [m]')
            t_s = datetime.date.fromordinal(np.int( this_set_time[0] ))
            t_e = datetime.date.fromordinal(np.int( this_set_time[-1] ))
            ax0.set_title('DG35 BATS Transects 2015: ' + np.str(t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day))        
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(vc, label='[m/s]')
            plt.tight_layout()
            ax0.grid()
            # plot_pro(ax0)
            fig0.savefig( ('/Users/jake/Desktop/BATS/bats_transects/dg035_BATS_H' + str(main) + '_T' + str(ii) + '.png'),dpi = 300)
            plt.close()    
    
        # OUTPUT V_g AND Eta from each transect collection so that it PE and KE can be computed 
        # size (m,n) is gridded depths and number of profiles 
        if np.size(Eta) < 1:
            Eta = eta
            Eta_theta = eta_theta
            V = V_g[:,:-1]
            UU = u_g[:,:-1]
            VV = v_g[:,:-1]
            Time = this_set_time[0:-1]
            heading_out = np.array( [ np.nanmean( heading_rec[heading_mask]) ] ) # average heading of said transect 
            Info = info # need dive number and lat/lon 
            Sigma_Theta_f = sigma_theta_out
        else:
            Eta = np.concatenate( (Eta, eta), axis=1 )
            Eta_theta = np.concatenate( (Eta_theta, eta_theta), axis=1 )
            V = np.concatenate( (V, V_g[:,:-1]), axis=1 )
            UU = np.concatenate( (UU, u_g[:,:-1]), axis=1 )
            VV = np.concatenate( (VV, v_g[:,:-1]), axis=1 )
            Time = np.concatenate( (Time, this_set_time[0:-1]) )
            heading_out = np.concatenate( (heading_out, np.array([ np.nanmean( heading_rec[heading_mask]) ])) )
            Info = np.concatenate( (Info, info), axis=1 )
            Sigma_Theta_f = np.concatenate( (Sigma_Theta_f, sigma_theta_out), axis=1 )

# END LOOPING OVER ALL TRANSECTS   
# END LOOPING OVER THE TWO HEADING CHOICES    

### SAVE 
# write python dict to a file
sa = 0
if sa > 0:
    mydict = {'bin_depth': grid,'Sigma_Theta': Sigma_Theta_f, 'Eta': Eta, 'Eta_theta': Eta_theta, 'V': V, 'Time': Time, 'Heading': heading_out, 'Info': Info }
    output = open('/Users/jake/Desktop/bats/transect_profiles.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()
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
plot_eta = 0

# LOAD EACH DIVE AND COLLECT IN MAIN DATAFRAME ###### GRIDDING! -- ONLY DO ONCE 
# df_t, df_s, df_den, df_lon, df_lat, dac_u, dac_v, time_rec, time_sta_sto, heading_rec = collect_dives(dg_list, bin_depth, grid, grid_p, ref_lat)
# dive_list = np.array(df_t.columns)
# f = netcdf.netcdf_file('BATs_2015_gridded.nc', 'w')    
# f.history = 'DG 2015 dives; have been gridded vertically and separated into dive and climb cycles'
# f.createDimension('grid',np.size(grid))
# f.createDimension('dive_list',np.size(dive_list))
# b_d = f.createVariable('grid',  np.float64, ('grid',) )
# b_d[:] = grid
# b_l = f.createVariable('dive_list',  np.float64, ('dive_list',) )
# b_l[:] = dive_list
# b_t = f.createVariable('Temperature',  np.float64, ('grid','dive_list') )
# b_t[:] = df_t
# b_s = f.createVariable('Salinity',  np.float64, ('grid','dive_list') )
# b_s[:] = df_s
# b_den = f.createVariable('Density',  np.float64, ('grid','dive_list') )
# b_den[:] = df_den
# b_lon = f.createVariable('Longitude',  np.float64, ('grid','dive_list') )
# b_lon[:] = df_lon
# b_lat = f.createVariable('Latitude',  np.float64, ('grid','dive_list') )
# b_lat[:] = df_lat
# b_u = f.createVariable('DAC_u',  np.float64, ('dive_list',) )
# b_u[:] = dac_u
# b_v = f.createVariable('DAC_v',  np.float64, ('dive_list',) )
# b_v[:] = dac_v
# b_time = f.createVariable('time_rec',  np.float64, ('dive_list',) )
# b_time[:] = time_rec
# b_t_ss = f.createVariable('time_start_stop',  np.float64, ('dive_list',) )
# b_t_ss[:] = time_sta_sto
# b_h = f.createVariable('heading_record',  np.float64, ('dive_list',) )
# b_h[:] = heading_rec
# f.close()

# LOAD DATA 
GD = netcdf.netcdf_file('BATs_2015_gridded.nc','r')
df_den = pd.DataFrame(np.float64(GD.variables['Density'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_t = pd.DataFrame(np.float64(GD.variables['Temperature'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_s = pd.DataFrame(np.float64(GD.variables['Salinity'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_lon = pd.DataFrame(np.float64(GD.variables['Longitude'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_lat = pd.DataFrame(np.float64(GD.variables['Latitude'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
dac_u = GD.variables['DAC_u'][:]
dac_v = GD.variables['DAC_v'][:]
time_rec = GD.variables['time_rec'][:]
time_sta_sto = GD.variables['time_start_stop'][:]
heading_rec = GD.variables['heading_record'][:]
profile_list = np.float64(GD.variables['dive_list'][:]) - 35000

# HEADING CHOICE AND BEGIN 
heading = np.array([[200,300],[100,200]])
# outputs from all loops 
N2_out = np.ones((np.size(grid),2))
Eta = []
Eta_theta = []
V = [] 
Time = []
heading_out = []
for main in range(2):
    head_low = heading[main,0]
    head_high = heading[main,1]
    # plan view plot (for select heading)   
    if plot_bath > 0:
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
        heading_mask = np.where( (heading_rec > head_low) & (heading_rec < head_high) ) 
        heading_mask_out = np.where( (heading_rec < head_low) | (heading_rec > head_high) ) 
        dg_a = ax0.plot(df_lon.iloc[:,heading_mask_out[0]],df_lat.iloc[:,heading_mask_out[0]],color='#8B0000',linewidth=1.5,
            label='All Dives (' + str(int(profile_list[0])) + '-' + str(int(profile_list[-2])) + ')',zorder=1) 
        dg_s = ax0.plot(df_lon.iloc[:,heading_mask[0]],df_lat.iloc[:,heading_mask[0]],color='#FF4500',linewidth=2, label = 'Dives Along Select Heading',zorder=1) 
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
        # ax0.grid()
        # plot_pro(ax0)
        # plt.show()
        fig0.savefig('/Users/jake/Desktop/bats/plan_200_300.png',dpi = 200)


    ############ SELECT ALL TRANSECTS ALONG A HEADING AND COMPUTE VERTICAL DISPLACEMENT AND HORIZONTAL VELOCITY 
    # select only dives along desired heading
    heading_mask = np.where( (heading_rec > head_low) & (heading_rec < head_high) ) 
    df_den_in = df_den.iloc[:,heading_mask[0]]
    df_t_in = df_t.iloc[:,heading_mask[0]]
    df_s_in = df_s.iloc[:,heading_mask[0]]
    df_lon_in = df_lon.iloc[:,heading_mask[0]]
    df_lat_in = df_lat.iloc[:,heading_mask[0]]
    time_in = time_sta_sto[heading_mask[0]]

    # average background properties of profiles along these transects 
    sigma_theta_avg = np.array(np.nanmean(df_den_in,1))
    theta_avg = np.array(np.nanmean(df_t_in,1))
    salin_avg = np.array(np.nanmean(df_s_in,1))
    ddz_avg_sigma = np.gradient(sigma_theta_avg,z)
    ddz_avg_theta = np.gradient(theta_avg,z)
    N2 = np.nan*np.zeros(np.size(sigma_theta_avg))
    N2[1:] = np.squeeze(sw.bfrq(salin_avg, theta_avg, grid_p, lat=ref_lat)[0])  
    lz = np.where(N2 < 0)   
    lnan = np.isnan(N2)
    N2[lz] = 0 
    N2[lnan] = 0
    N = np.sqrt(N2)   
    N2_out[:,main] = N2

    # computer vertical mode shapes 
    G, Gz, c = vertical_modes(N2,grid,omega,mmax)

    # PREPARE FOR TRANSECT ANALYSIS 
    # dive list to consider
    dives = np.array(df_den_in.columns) - 35000

    # section out dives into continuous transect groups 
    if main < 1:
        to_consider = 13
    else:
        to_consider = 16
    dive_iter = np.array(dives[0])
    time_iter = np.array(time_in[0])
    dive_out = {}
    time_out = {}
    for i in range(to_consider):
        if i < 1:
            this_dive = dives[0]
            this_time = time_in[0]
        else:
            this_dive = dive_iter[i]   
            this_time = time_iter[i]              
    
        dive_group = np.array([this_dive]) 
        time_group = np.array([this_time]) 
        up_o = np.where(dives==this_dive)[0]
        for j in dives[up_o[0]+1:]:
            if j - dive_group[-1] < 1:
                dive_group = np.append(dive_group,j)  
                t_coor = np.where(dives==j)[0][0]
                time_group = np.append(time_group,time_in[t_coor])    
             
        dive_out[i] = dive_group
        time_out[i] = time_group
        up_n = np.where(dives==dive_group[-1])[0]
        dive_iter = np.array(np.append(dive_iter,dives[up_n[0]+1]))
        time_iter = np.array(np.append(time_iter,time_in[up_n[0]+1]))    
    
    # loop over each dive_group
    ndives_in_trans = np.nan*np.zeros(to_consider)
    for l in range(to_consider):
        ndives_in_trans[l] = np.size(dive_out[l])/2
        # choose only transects that have three dives     
        good = np.where(ndives_in_trans > 2)

    ######### MAIN LOOP OVER EACH TRANSECT (EACH TRANSECT CONTAINS AT LEAST 3 DIVE CYCLES)
    for master in range(np.size(good)): 
        ii = good[0][master]
        this_set = dive_out[ii] + 35000
        this_set_time = time_out[ii]
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
        if np.size(this_set) <= 6:
            order_set = [0,2,4] # go from 0,2,4 (because each transect only has 3 dives)
        else:
            order_set = [0,2,4,6] # go from 0,2,4 (because each transect only has 3 dives)    
        
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
            shear[:,i] = shearM
            eta[:,i] = etaM
            eta_theta[:,i] = eta_thetaM
            if i < np.size(this_set)-2:
                shear[:,i+1] = shearW
                eta[:,i+1] = etaW
                eta_theta[:,i+1] = eta_thetaW
            
            # ISOPYCNAL DEPTHS ON PROFILES ALONG EACH TRANSECT
            sigthmin = np.nanmin( df_den_set.iloc[:,i] )
            sigthmax = np.nanmax( df_den_set.iloc[:,i] )
            isigth = np.where( (sigth_levels > sigthmin) & (sigth_levels < sigthmax) )
            isopycdep[isigth,i] = np.interp( sigth_levels[isigth], np.array(df_den_set.iloc[:,i]), grid)
            isopycx[isigth,i] = np.interp( sigth_levels[isigth], np.array(df_den_set.iloc[:,i]), dist[:,i])
        
            sigthmin = np.nanmin( df_den_set.iloc[:,i+1] )
            sigthmax = np.nanmax( df_den_set.iloc[:,i+1] )
            isigth = np.where( (sigth_levels > sigthmin) & (sigth_levels < sigthmax) )
            isopycdep[isigth,i+1] = np.interp( sigth_levels[isigth], np.array(df_den_set.iloc[:,i+1]), grid)
            isopycx[isigth,i+1] = np.interp( sigth_levels[isigth], np.array(df_den_set.iloc[:,i+1]), dist[:,i+1])
        
            # END LOOP OVER EACH DIVE IN TRANSECT 
    
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

        if plot_cross > 0: 
            # sns.set(style="darkgrid")
            sns.set(context="notebook", style="whitegrid", rc={"axes.axisbelow": False})
            # sns.set_color_codes(palette='muted')        
               
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
            ax0.grid()
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
            # plt.show()
            # fig0.savefig( ('/Users/jake/Desktop/BATS/dg035_BATS_15b_' + str(ii) + '_test.png'),dpi = 300)
            plt.close()    
    
        # OUTPUT V_g AND Eta from each transect collection so that it PE and KE can be computed 
        if np.size(Eta) < 1:
            Eta = eta
            Eta_theta = eta_theta
            V = V_g[:,:-1]
            Time = this_set_time[0:-1]
            heading_out = np.array([main])
        else:
            Eta = np.concatenate( (Eta, eta), axis=1 )
            Eta_theta = np.concatenate( (Eta_theta, eta_theta), axis=1 )
            V = np.concatenate( (V, V_g[:,:-1]), axis=1 )
            Time = np.concatenate( (Time, this_set_time[0:-1]) )
            heading_out = np.concatenate( (heading_out, np.array([main])) )

# END LOOPING OVER ALL TRANSECTS                 
# END LOOPING OVER ALL TRANSECTS 
# END LOOPING OVER ALL TRANSECTS 

# END LOOPING OVER THE TWO HEADING CHOICES    

# first taper fit above and below min/max limits
# Project modes onto each eta (find fitted eta)
# Compute PE 

# presort V 
good_v = np.zeros(np.size(Time))
for i in range(np.size(Time)):
    v_dz = np.gradient(V[10:,i])
    if np.nanmax(np.abs(v_dz)) < 0.075:
        good_v[i] = 1        
good0 = np.intersect1d(np.where((np.abs(V[-45,:]) < 0.2))[0],np.where((np.abs(V[10,:]) < 0.4))[0])
good = np.intersect1d(np.where(good_v > 0),good0)
V2 = V[:,good]
Eta2 = Eta[:,good]
Eta_theta2 = Eta_theta[:,good]

sz = np.shape(Eta2)
num_profs = sz[1]
eta_fit_depth_min = 50
eta_fit_depth_max = 3800
eta_theta_fit_depth_max = 4200
AG = np.zeros([nmodes, num_profs])
AGz = np.zeros([nmodes, num_profs])
AG_theta = np.zeros([nmodes, num_profs])
Eta_m = np.nan*np.zeros([np.size(grid), num_profs])
V_m = np.nan*np.zeros([np.size(grid), num_profs])
Neta = np.nan*np.zeros([np.size(grid), num_profs])
NEta_m = np.nan*np.zeros([np.size(grid), num_profs])
Eta_theta_m = np.nan*np.zeros([np.size(grid), num_profs])
PE_per_mass = np.nan*np.zeros([nmodes, num_profs])
HKE_per_mass = np.nan*np.zeros([nmodes, num_profs])
PE_theta_per_mass = np.nan*np.zeros([nmodes, num_profs])
modest = np.arange(11,nmodes)
good_prof = np.ones(num_profs)
HKE_noise_threshold = 1e-4 # 1e-5
for i in range(num_profs):    
    # fit to velocity profiles
    this_V = V2[:,i].copy()
    iv = np.where( ~np.isnan(this_V) )
    if iv[0].size > 1:
        AGz[:,i] =  np.squeeze(np.linalg.lstsq( np.squeeze(Gz[iv,:]),np.transpose(np.atleast_2d(this_V[iv])))[0]) # Gz(iv,:)\V_g(iv,ip)                
        V_m[:,i] =  np.squeeze(np.matrix(Gz)*np.transpose(np.matrix(AGz[:,i])))  #Gz*AGz[:,i];
        HKE_per_mass[:,i] = AGz[:,i]*AGz[:,i]
        ival = np.where( HKE_per_mass[modest,i] >= HKE_noise_threshold )
        if np.size(ival) > 0:
            good_prof[i] = 0 # flag profile as noisy
    else:
        good_prof[i] = 0 # flag empty profile as noisy as well
  
    # fit to eta profiles
    this_eta = Eta2[:,i].copy()
    # obtain matrix of NEta
    Neta[:,i] = N*this_eta
    this_eta_theta = Eta_theta2[:,i].copy()
    iw = np.where((grid>=eta_fit_depth_min) & (grid<=eta_fit_depth_max))
    iw_theta = np.where((grid>=eta_fit_depth_min) & (grid<=eta_theta_fit_depth_max))
    if iw[0].size > 1:
        eta_fs = Eta2[:,i].copy() # ETA
        eta_theta_fs = Eta_theta2[:,i].copy() # ETA THETA
    
        i_sh = np.where( (grid < eta_fit_depth_min))
        eta_fs[i_sh[0]] = grid[i_sh]*this_eta[iw[0][0]]/grid[iw[0][0]]
        eta_theta_fs[i_sh[0]] = grid[i_sh]*this_eta_theta[iw[0][0]]/grid[iw[0][0]]
    
        i_dp = np.where( (grid > eta_fit_depth_max) )
        eta_fs[i_dp[0]] = (grid[i_dp] - grid[-1])*this_eta[iw[0][-1]]/(grid[iw[0][-1]]-grid[-1])
        
        i_dp_theta = np.where( (grid > eta_theta_fit_depth_max) )
        eta_theta_fs[i_dp_theta[0]] = (grid[i_dp_theta] - grid[-1])*this_eta_theta[iw_theta[0][-1]]/(grid[iw_theta[0][-1]]-grid[-1])
        
        AG[1:,i] = np.squeeze(np.linalg.lstsq(G[:,1:],np.transpose(np.atleast_2d(eta_fs)))[0])
        AG_theta[1:,i] = np.squeeze(np.linalg.lstsq(G[:,1:],np.transpose(np.atleast_2d(eta_theta_fs)))[0])
        Eta_m[:,i] = np.squeeze(np.matrix(G)*np.transpose(np.matrix(AG[:,i])))
        NEta_m[:,i] = N*np.array(np.squeeze(np.matrix(G)*np.transpose(np.matrix(AG[:,i]))))
        Eta_theta_m[:,i] = np.squeeze(np.matrix(G)*np.transpose(np.matrix(AG_theta[:,i])))
        PE_per_mass[:,i] = (1/2)*AG[:,i]*AG[:,i]*c*c
        PE_theta_per_mass[:,i] = (1/2)*AG_theta[:,i]*AG_theta[:,i]*c*c 

# output density structure for comparison 
sa = 0
if sa > 0:
    mydict = {'bin_depth': grid,'eta': Eta2,'dg_v': V2}
    output = open('/Users/jake/Desktop/bats/den_v_profs.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()        

### COMPUTE EOF SHAPES AND COMPARE TO ASSUMED STRUCTURE 
### EOF SHAPES 
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
EOFshape = np.matrix(Gz)*V_AGzqa # depth shape of eigenfunctions [ndepths X nmodes]
EOFshape1_BTpBC1 = np.matrix(Gz[:,0:2])*V_AGzqa[0:2,0] # truncated 2 mode shape of EOF#1
EOFshape2_BTpBC1 = np.matrix(Gz[:,0:2])*V_AGzqa[0:2,1] # truncated 2 mode shape of EOF#2  

## find EOFs of dynamic vertical displacement (eta) mode amplitudes
# extract noisy/bad profiles 
good_prof_eof = np.where(~np.isnan(AG[2,:]))
num_profs_2 = np.size(good_prof_eof)
AG2 = AG[:,good_prof_eof[0]]
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

## PLOT ETA / EOF 
if plot_eta > 0: 
    f, (ax0,ax1) = plt.subplots(1, 2, sharey=True)
    for j in range(num_profs):
        ax1.plot(Eta2[:,j],grid,color='#CD853F',linewidth=1.25)  
        ax1.plot(Eta_m[:,j],grid,color='k',linestyle='--',linewidth=.75)    
        # ax0.plot(Eta_theta2[:,j],grid,color='#CD853F',linewidth=.65) 
        # ax0.plot(Eta_theta_m[:,j],grid,color='k',linestyle='--',linewidth=.75)
        ax0.plot(V2[:,j],grid,color='#CD853F',linewidth=1.25)  
        ax0.plot(V_m[:,j],grid,color='k',linestyle='--',linewidth=.75)
    ax1.axis([-600, 600, 0, 5000]) 
    ax0.text(190,800,str(num_profs)+' Profiles')
    ax1.set_xlabel(r'$\xi_{\sigma_{\theta}}$ [m]')
    ax1.set_title(r'$\xi$ Vertical Isopycnal Disp.') # + '(' + str(Time[0]) + '-' )
    ax0.axis([-.4, .4, 0, 5000]) 
    ax0.set_title("BATS '15 DG Cross-track u")
    ax0.set_ylabel('Depth [m]',fontsize=14)
    ax0.set_xlabel('u [m/s]',fontsize=14)
    ax0.invert_yaxis() 
    ax0.grid()    
    plot_pro(ax1)
    # f.savefig('/Users/jake/Desktop/bats/dg035_15_Eta_a.png',dpi = 300)
    # plt.show()    
    
    max_plot = 3 
    f, (ax1,ax2) = plt.subplots(1,2,sharey=True) 
    n2p = ax1.plot((np.sqrt(N2)*(1800/np.pi)),grid,color='k',label='N(z) [cph]') 
    colors = plt.cm.Dark2(np.arange(0,4,1))
    for ii in range(max_plot):
        ax1.plot(Gz[:,ii], grid,color='#2F4F4F',linestyle='--')
        p_eof=ax1.plot(-EOFshape[:,ii], grid,color=colors[ii,:],label='EOF # = ' + str(ii+1),linewidth=2)
    handles, labels = ax1.get_legend_handles_labels()    
    ax1.legend(handles,labels,fontsize=10)    
    ax1.axis([-4,4,0,5000])    
    ax1.invert_yaxis()
    ax1.set_title('BATS DG EOF Mode Shapes (DG)')
    ax1.set_ylabel('Depth [m]')
    ax1.set_xlabel('Hor. Vel. Mode Shapes (DG)')
    ax1.grid()    
    for ii in range(max_plot):
        ax2.plot(G[:,ii+1]/np.max(grid),grid,color='#2F4F4F',linestyle='--')
        p_eof_eta=ax2.plot(-EOFetashape[:,ii]/np.max(grid), grid,color=colors[ii,:],label='EOF # = ' + str(ii+1),linewidth=2)
    handles, labels = ax2.get_legend_handles_labels()    
    ax2.legend(handles,labels,fontsize=10)    
    ax2.axis([-.7,.7,0,5000])    
    ax2.set_title('EOF Mode Shapes (DG)')
    ax2.set_xlabel('Vert. Disp. Mode Shapes')
    ax2.invert_yaxis()    
    plot_pro(ax2)

avg_PE = np.nanmean(PE_per_mass,1)
good_prof_i = good_prof #np.where(good_prof > 0)
avg_KE = np.nanmean(HKE_per_mass[:,np.where(good_prof>0)[0]],1)
fig0, ax0 = plt.subplots()
for i in range(np.size(good_prof)):
    if good_prof[i] > 0:
        ax0.plot(np.arange(0,61,1),HKE_per_mass[:,i])
ax0.set_xscale('log')      
ax0.set_yscale('log')    
plot_pro(ax0)    
# avg_PE_theta = np.nanmean(PE_theta_per_mass,1)
f_ref = np.pi*np.sin(np.deg2rad(ref_lat))/(12*1800)
rho0 = 1025
dk = f_ref/c[1]
sc_x = (1000)*f_ref/c[1:]
vert_wavenumber = f_ref/c[1:]
    
PE_SD, PE_GM = PE_Tide_GM(rho0,grid,nmodes,np.transpose(np.atleast_2d(N2)),f_ref)

# KE parameters
dk_ke = 1000*f_ref/c[1]   
k_h = 1e3*(f_ref/c[1:])*np.sqrt( avg_KE[1:]/avg_PE[1:])

# load in Station BATs PE Comparison
SB = si.loadmat('/Users/jake/Desktop/bats/station_bats_pe.mat')
sta_bats_pe = SB['out'][0][0][0]
sta_bats_c = SB['out'][0][0][3]
sta_bats_f = SB['out'][0][0][2]
sta_bats_dk = SB['out'][0][0][1]

plot_eng = 0
plot_spec = 0
plot_comp = 0
if plot_eng > 0:    
    if plot_spec > 0:
        fig0, ax0 = plt.subplots()
        PE_p = ax0.plot(sc_x,avg_PE[1:]/dk,color='#B22222',label='PE',linewidth=1.5)
        # PE_sta_p = ax0.plot((1000)*sta_bats_f/sta_bats_c[1:],sta_bats_pe[1:]/sta_bats_dk,color='#FF8C00',label='$PE_{ship}$',linewidth=1.5)
        KE_p = ax0.plot(sc_x,avg_KE[1:]/dk,'g',label='KE',linewidth=1.5)        
        ax0.scatter(sc_x,avg_PE[1:]/dk,color='#B22222',s=10) # DG PE
        # ax0.scatter((1000)*sta_bats_f/sta_bats_c[1:],sta_bats_pe[1:]/sta_bats_dk,color='#FF8C00',s=10) # BATS PE
        ax0.scatter(sc_x,avg_KE[1:]/dk,color='g',s=10) # DG KE
        
        ax0.plot(sc_x,0.25*PE_GM/dk,linestyle='--',color='#B22222',linewidth=0.75)
        # ax0.plot(sc_x,PE_GM/dk,linestyle='--',color='#FF8C00',linewidth=0.75)
        ax0.text(sc_x[0]-.014,.25*PE_GM[1]/dk,r'$\frac{1}{4}PE_{GM}$',fontsize=12)
        # ax0.text(sc_x[0]-.011,PE_GM[1]/dk,r'$PE_{GM}$')
        ## ax0.plot( [1000*f_ref/c[1], 1000*f_ref/c[-2]],[1000*f_ref/c[1], 1000*f_ref/c[-2]],linestyle='--',color='k',linewidth=0.8)
        ax0.text( 1000*f_ref/c[-2]+.1, 1000*f_ref/c[-2], r'f/c$_m$',fontsize=10)
        ax0.plot(sc_x,k_h,color='k',linewidth=.9,label=r'$k_h$')
        ax0.text(sc_x[0]-.008,k_h[0]-.011,r'$k_{h}$ [km$^{-1}$]',fontsize=10)       
        
        # limits/scales 
        ax0.plot( [3*10**-1, 3*10**0], [1.5*10**1, 1.5*10**-2],color='k',linewidth=0.75)
        ax0.plot([3*10**-2, 3*10**-1],[7*10**2, ((5/3)*(np.log10(2*10**-1) - np.log10(2*10**-2) ) +  np.log10(7*10**2) )] ,color='k',linewidth=0.75)
        ax0.text(3.3*10**-1,1.3*10**1,'-3',fontsize=10)
        ax0.text(3.3*10**-2,6*10**2,'-5/3',fontsize=10)
        ax0.plot( [1000*f_ref/c[1], 1000*f_ref/c[-2]],[1000*f_ref/c[1], 1000*f_ref/c[-2]],linestyle='--',color='k',linewidth=0.8)
         
        ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax0.axis([10**-2, 10**1, 3*10**(-4), 2*10**(3)])
        ax0.set_xlabel(r'Scaled Vertical Wavenumber = (Rossby Radius)$^{-1}$ = $\frac{f}{c_m}$ [$km^{-1}$]',fontsize=14)
        ax0.set_ylabel('Spectral Density, Hor. Wavenumber',fontsize=14) # ' (and Hor. Wavenumber)')
        ax0.set_title('DG 2015 BATS Deployment (Energy Spectra)',fontsize=14)
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend([handles[0],handles[1],handles[2]],[labels[0], labels[1], labels[2]],fontsize=12)
        plt.tight_layout()
        plot_pro(ax0)
        # fig0.savefig('/Users/jake/Desktop/bats/dg035_15_PE_b.png',dpi = 300)
        # plt.close()
        # plt.show()

    if plot_comp > 0: 
        ############## ABACO BATS COMPARISON ################
        # load ABACO spectra
        AB_f_pe = np.load('/Users/jake/Desktop/abaco/f_ref_pe.npy')
        AB_c_pe = np.load('/Users/jake/Desktop/abaco/c_pe.npy')
        AB_f = np.load('/Users/jake/Desktop/abaco/f_ref.npy')
        AB_c = np.load('/Users/jake/Desktop/abaco/c.npy')
        AB_avg_PE = np.load('/Users/jake/Desktop/abaco/avg_PE.npy')
        AB_avg_KE = np.load('/Users/jake/Desktop/abaco/avg_KE.npy')
        AB_sc_x = (1000)*AB_f_pe/AB_c_pe[1:]
        AB_dk = AB_f_pe/AB_c_pe[1]
        AB_dk_ke = AB_f/AB_c[1]

        # PLOT ABACO AND BATS 
        fig0, (ax0,ax1) = plt.subplots(1, 2, sharey=True)
        PE_p = ax0.plot(sc_x,avg_PE[1:]/dk,'r',label='PE')
        KE_p = ax0.plot(sc_x,avg_KE[1:]/dk,'b',label='KE')
        ax0.plot( [10**-1, 10**0], [1.5*10**1, 1.5*10**-2],color='k',linestyle='--',linewidth=0.8)
        ax0.text(0.8*10**-1,1.3*10**1,'-3',fontsize=8)
        ax0.scatter(sc_x,avg_PE[1:]/dk,color='r',s=6)
        ax0.plot(sc_x,0.5*PE_GM/dk,linestyle='--',color='#DAA520')
        ax0.text(sc_x[0]-.009,PE_GM[0]/dk,r'$PE_{GM}$')
        ax0.plot( [1000*f_ref/c[1], 1000*f_ref/c[-2]],[1000*f_ref/c[1], 1000*f_ref/c[-2]],linestyle='--',color='k',linewidth=0.8)
        ax0.text( 1000*f_ref/c[-2]+.1, 1000*f_ref/c[-2], r'f/c$_m$',fontsize=8)
        ax0.plot(sc_x,k_h,color='k')
        ax0.text(sc_x[0]-.008,k_h[0]-.008,r'$k_{h}$ [km$^{-1}$]',fontsize=8)        
        ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax0.axis([10**-2, 1.5*10**1, 10**(-4), 10**(3)])
        ax0.grid()
        ax0.set_xlabel(r'Scaled Vert. Wavenumber = $\frac{f}{c}$ [$km^{-1}$]',fontsize=10)
        ax0.set_ylabel('Spectral Density (and Hor. Wavenumber)')
        ax0.set_title('BATS (2015) 42 Profiles')
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend([handles[0],handles[-1]],[labels[0], labels[-1]],fontsize=10)
        
        PE_p = ax1.plot(AB_sc_x,AB_avg_PE[1:]/AB_dk,'r',label='PE')
        ax1.scatter(AB_sc_x,AB_avg_PE[1:]/AB_dk,color='r',s=6)
        KE_p = ax1.plot(AB_sc_x,AB_avg_KE[1:]/AB_dk_ke,'b',label='KE')    
        AB_k_h = AB_sc_x*np.sqrt( np.squeeze(AB_avg_KE[1:])/AB_avg_PE[1:])     
        ax1.plot( [1000*AB_f_pe/AB_c_pe[1], 1000*AB_f_pe/AB_c_pe[-2]], [1000*AB_f_pe/AB_c_pe[1], 1000*AB_f_pe/AB_c_pe[-2]],linestyle='--',color='k',linewidth=0.8)
        ax1.text( 1000*AB_f_pe/AB_c_pe[-2]+.1, 1000*AB_f_pe/AB_c_pe[-2], r'f/c$_m$',fontsize=8)
        ax1.plot(AB_sc_x,AB_k_h,color='k')
        ax1.plot( [10**-1, 10**0], [1.5*10**1, 1.5*10**-2],color='k',linestyle='--',linewidth=0.8)
        ax1.plot(AB_sc_x,PE_GM/AB_dk,linestyle='--',color='#DAA520')
        ax1.text(AB_sc_x[0]-.009,PE_GM[0]/AB_dk,r'$PE_{GM}$')
        ax1.text(0.8*10**-1,1.3*10**1,'-3',fontsize=8)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.axis([10**-2, 1.5*10**1, 10**(-4), 10**(3)])
        ax1.set_xlabel(r'Scaled Vert. Wavenumber = $\frac{f}{c}$ [$km^{-1}$]',fontsize=10)
        ax1.set_title('ABACO (2017) 57 Profiles')
        ax1.grid()        
        fig0.savefig('/Users/jake/Desktop/bats/dg035_15_PE_comp_b_test.png',dpi = 300)   
        plt.close()
    
        # PLOT ON SAME X AXIS 
        fig0, ax0 = plt.subplots()
        PE_BATS = ax0.plot(np.arange(0,60),avg_PE[1:]/dk,'r',label='BATS PE')    
        PE_ABACO = ax0.plot(np.arange(0,60),AB_avg_PE[1:]/AB_dk,'b',label='ABACO PE')
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend([handles[0],handles[-1]],[labels[0], labels[-1]],fontsize=10)
        ax0.set_yscale('log')
        ax0.set_xlabel('Vertical Mode Number')
        ax0.set_ylabel('Energy (variance per vert. wave number)')
        ax0.set_title('PE Comparison')
        ax0.grid()  
        fig0.savefig('/Users/jake/Desktop/bats/dg035_15_PE_mode_comp_b_test.png',dpi = 300)
        plt.close()
    
####### 
# PE COMPARISON BETWEEN HOTS, BATS_SHIP, AND BATS_DG
# load in Station BATs PE Comparison
SH = si.loadmat('/Users/jake/Desktop/bats/station_hots_pe.mat')
sta_hots_pe = SH['out'][0][0][0]
sta_hots_c = SH['out'][0][0][3]
sta_hots_f = SH['out'][0][0][2]
sta_hots_dk = SH['out'][0][0][1]

fig0, ax0 = plt.subplots()
ax0.plot( [3*10**-1, 3*10**0], [1.5*10**1, 1.5*10**-2],color='k',linewidth=0.75)
ax0.plot([3*10**-2, 3*10**-1],[7*10**2, ((5/3)*(np.log10(2*10**-1) - np.log10(2*10**-2) ) +  np.log10(7*10**2) )] ,color='k',linewidth=0.75)
ax0.text(3.3*10**-1,1.3*10**1,'-3',fontsize=8)
ax0.text(3.3*10**-2,6*10**2,'-5/3',fontsize=8)
ax0.plot(sc_x,PE_GM/dk,linestyle='--',color='k')
ax0.text(sc_x[0]-.009,PE_GM[1]/dk,r'$PE_{GM}$')
PE_p = ax0.plot(sc_x,avg_PE[1:]/dk,color='#B22222',label=r'$PE_{BATS_{DG}}$')
PE_sta_p = ax0.plot((1000)*sta_bats_f/sta_bats_c[1:],sta_bats_pe[1:]/sta_bats_dk,color='#FF8C00',label=r'$PE_{BATS_{ship}}$')
# PE_sta_hots = ax0.plot((1000)*sta_hots_f/sta_hots_c[1:],sta_hots_pe[1:]/sta_hots_dk,color='g',label=r'$PE_{HOTS_{ship}}$')
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.set_xlabel(r'Vertical Wavenumber = Inverse Rossby Radius = $\frac{f}{c}$ [$km^{-1}$]',fontsize=13)
ax0.set_ylabel('Spectral Density (and Hor. Wavenumber)')
ax0.set_title('Potential Energy Spectra (Site/Platform Comparison)')
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles,labels,fontsize=12)
plot_pro(ax0)


# LOAD NEARBY ABACO 
pkl_file = open('/Users/jake/Desktop/abaco/abaco_outputs.pkl', 'rb')
abaco_energies = pickle.load(pkl_file)
pkl_file.close()   

fig0, ax0 = plt.subplots()
mode_num = np.arange(1,61,1)
PE_p = ax0.plot(mode_num,avg_PE[1:]/dk,label=r'$BATS_{DG}$')
PE_sta_p = ax0.plot(mode_num,sta_bats_pe[1:]/sta_bats_dk,label=r'$BATS_{ship}$')
PE_ab = ax0.plot(mode_num,abaco_energies['avg_PE'][1:]/(abaco_energies['f_ref']/abaco_energies['c'][1]),label=r'$ABACO_{DG}$')
# PE_sta_hots = ax0.plot(mode_num,sta_hots_pe[1:]/sta_hots_dk,label=r'$HOTS_{ship}$')
ax0.set_xlabel('Mode Number',fontsize=13)
ax0.set_ylabel('Spectral Density',fontsize=13)
ax0.set_title('Potential Energy Spectra (Site/Platform Comparison)',fontsize=14)
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.axis([8*10**-1, 10**2, 3*10**(-4), 2*10**(3)])
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles,labels,fontsize=12)
plot_pro(ax0)
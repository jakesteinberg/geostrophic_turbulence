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
# functions I've written 
from grids import make_bin
from mode_decompositions import vertical_modes, PE_Tide_GM

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)    

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
z = -1*grid
grid_p = sw.pres(grid,lat_in)
secs_per_day = 86400.0
datenum_start = 719163 # jan 1 1970 

plot_bath = 0
# initial arrays and dataframes 
df_t = pd.DataFrame()
df_s = pd.DataFrame()
df_den = pd.DataFrame()
df_lon = pd.DataFrame()
df_lat = pd.DataFrame()
heading_rec = []
time_rec = []
dac_u = []
dac_v = []
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
        dac_u = np.concatenate([ [dive_nc_file.variables['depth_avg_curr_east'].data], [dive_nc_file.variables['depth_avg_curr_east'].data] ])
        dac_v = np.concatenate([ [dive_nc_file.variables['depth_avg_curr_north'].data], [dive_nc_file.variables['depth_avg_curr_north'].data] ])
    else:
        time_rec = np.concatenate([ time_rec, [serial_date_time_dive], [serial_date_time_climb] ])
        dac_u = np.concatenate([ dac_u,[dive_nc_file.variables['depth_avg_curr_east'].data],[dive_nc_file.variables['depth_avg_curr_east'].data] ])
        dac_v = np.concatenate([ dac_v,[dive_nc_file.variables['depth_avg_curr_north'].data],[dive_nc_file.variables['depth_avg_curr_north'].data] ])
                
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
    lon_data_c = pd.DataFrame(lon_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
    lat_data_d = pd.DataFrame(lat_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
    lat_data_c = pd.DataFrame(lat_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
    
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
    fig0, ax0 = plt.subplots()
    bc = ax0.contourf(bath_lon,bath_lat,bath_z,levels,cmap='PuBu_r')
    # ax0.contourf(bath_lon,bath_lat,bath_z,[-5, -4, -3, -2, -1, 0, 100, 1000], cmap = 'YlGn_r')
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    bcl = ax0.contour(bath_lon,bath_lat,bath_z,[-4500, -4000],colors='k')
    ml = [(-65,31.5),(-64.4, 32.435)]
    ax0.clabel(bcl,manual = ml, inline_spacing=-3, fmt='%1.0f',colors='k')  
    
    heading_mask = np.where( (heading_rec>200) & (heading_rec <300) ) 
    heading_mask_out = np.where( (heading_rec<200) | (heading_rec >300) ) 
    ax0.plot(df_lon.iloc[:,heading_mask_out[0]],df_lat.iloc[:,heading_mask_out[0]],color='k',linewidth=1) 
    ax0.plot(df_lon.iloc[:,heading_mask[0]],df_lat.iloc[:,heading_mask[0]],color='r',linewidth=1) 
    ax0.scatter(np.nanmean(df_lon.iloc[:,heading_mask[0]],0),np.nanmean(df_lat.iloc[:,heading_mask[0]],0),s=20,color='g')  
     
    w = 1/np.cos(np.deg2rad(ref_lat))
    ax0.axis([-65.6, -63.25, 31.2, 32.8])
    ax0.set_aspect(w)
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig0.colorbar(bc, cax=cax, label='[m]')
    ax0.set_xlabel('Longitude')
    ax0.set_ylabel('Latitude')   
    t_s = datetime.date.fromordinal(np.int( np.min(time_rec) ))
    t_e = datetime.date.fromordinal(np.int( np.max(time_rec) ))
    ax0.set_title('Select BATS Transects (DG35): ' + np.str(t_s.month) + '/' + np.str(t_s.day) + '/' + np.str(t_s.year) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day) + '/' + np.str(t_e.year))
    plt.tight_layout()
    fig0.savefig('/Users/jake/Desktop/bats/plan_view.png',dpi = 200)


############ compute vertical displacements for both station and glider profiles 
# select only dives along desired heading
heading_mask = np.where( (heading_rec>200) & (heading_rec <300) ) 
df_den_in = df_den.iloc[:,heading_mask[0]]
df_lon_in = df_lon.iloc[:,heading_mask[0]]
df_lat_in = df_lat.iloc[:,heading_mask[0]]
# average properties of profiles along these transects 
sigma_theta_avg = np.array(np.nanmean(df_den_in,1))
ddz_avg_sigma = np.gradient(sigma_theta_avg,z)

# dive list to consider
dives = np.array(df_den_in.columns) - 35000

# section out dives into continuous transects 
dive_iter = np.array(dives[0])
dive_out = {}
for i in range(5):
    if i < 1:
        this_dive = dives[0]
    else:
        this_dive = dive_iter[i] 
        
    dive_group = this_dive    
    up_o = np.where(dives==this_dive)[0]
    for j in dives[up_o[0]+1:]:
        if j - this_dive < 3:
            dive_group = np.append(dive_group,j)
    
    dive_out[i] = dive_group
    up_n = np.where(dives==dive_group[-1])[0]
    dive_iter = np.array(np.append(dive_iter,dives[up_n[0]+1]))    
    
# loop over each dive_group
i = 3
this_set = dive_out[i] + 35000
df_den_set = df_den_in[this_set] 
df_lon_set = df_lon_in[this_set]
df_lat_set = df_lat_in[this_set]  

# total number of dive cycles
dive_cycs = np.unique(np.floor(this_set))
# pair dac_u,v with each M/W center
dive_list = np.array(df_den_in.columns)
inn = np.where( (dive_list >= this_set[0]) & (dive_list <= this_set[-1]) )
dac_u_inn = dac_u[inn]
dac_v_inn = dac_v[inn]

# parameters 
g = 9.81
rho0 = 1027
deep_shr_max = 0.1 # maximum allowed deep shear [m/s/km]
deep_shr_max_dep = 3500 # minimum depth for which shear is limited [m]



shear = np.nan*np.zeros( (np.size(grid),np.size(this_set)-1))
eta = np.nan*np.zeros( (np.size(grid),np.size(this_set)-1))
# loop over dive-cycles in this group to carry out M/W sampling and produce shear and eta profiles 
order_set = [0,2,4] # go from 0,2,4
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
    DACe = dac_u_inn[i] # zonal depth averaged current [m/s]
    DACn = dac_v_inn[i] # meridional depth averaged current [m/s]
    mag_DAC, ang_DAC = cart2pol(DACe, DACn);
    DACat, DACpot = pol2cart(mag_DAC,ang_DAC - ang_sfc_m);
    Vbt_m = DACpot; # across-track barotropic current comp (>0 to left)

    # W 
    lon_start = df_lon_set.iloc[0,i]
    lat_start = df_lat_set.iloc[0,i]
    if i >= 4:
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
    ds, ang_sfc_w = cart2pol(dxs, dys)
    mag_DAC, ang_DAC = cart2pol(DACe, DACn)
    DACat, DACpot = pol2cart(mag_DAC,ang_DAC - ang_sfc_w)
    Vbt_w = DACpot; # across-track barotropic current comp (>0 to left)

    # loop over all bin_depths 
    shearM = np.nan*np.zeros(np.size(grid))
    shearW = np.nan*np.zeros(np.size(grid))
    etaM = np.nan*np.zeros(np.size(grid))
    etaW = np.nan*np.zeros(np.size(grid))
    for j in range(np.size(grid)):
        # find array of indices for M / W sampling 
        if i < 2:      
            c_i_m = np.arange(i,i+3) 
            # im = []; % omit partial "M" estimate
            c_i_w = np.arange(i,i+4) 
        elif (i >= 2) and (i < this_set.size-2):
            c_i_m = np.arange(i-1,i+3) 
            c_i_w = np.arange(i,i+4) 
        elif i >= this_set.size-2:
            c_i_m = np.arange(i-1,this_set.size) 
            c_i_w = []
            # im = []; % omit partial "M" estimate
        nm = np.size(c_i_m); nw = np.size(c_i_w)

        # for M profile compute shear and eta 
        if nm > 2 and np.size(df_den_set.iloc[j,c_i_m]) > 2:
            sigmathetaM = df_den_set.iloc[j,c_i_m]
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
        
        # for W profile compute shear and eta 
        if nw > 2 and np.size(df_den_set.iloc[j,c_i_w]) > 2:
            sigmathetaW = df_den_set.iloc[j,c_i_w]
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

        # end looping on each bin
                
    # output
    shear[:,i] = shearM
    shear[:,i+1] = shearW
    eta[:,i] = etaM
    eta[:,i+1] = etaW

for m in range(np.size(this_set)-1):
    iq = np.where( ~np.isnan(shear[:,m]) ) 
    if np.size(iq) > 10:
        z = -grid[iq]
        vrel = cumtrapz(0.001*shear[iq,m],x=z )
        vrel_av = trapz(z, vrel)/(z(end) - z(1));
        vbc = vrel - vrel_av;
        Vbc_g(iq,ip) = vbc;
        V_g(iq,ip) = Vbt(ip) + vbc;
    else
        Vbc(:,ip) = NaN;
        V_g(:,ip) = NaN;
    end
end

for l in range(6):
    plt.scatter(df_lon_set.iloc[:,l],grid)
plt.grid()
plt.show()    

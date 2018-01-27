# BATS STATISTICAL INTERPOLATION ON DEPTH LEVELS 

# Procedue:
# 1. collect data and define lat/lon window within which we want to grid
# 2. define grid
# 3. determine a spatial mean and remove it
#  - need a function to determine a vector pointing in the direction of greatest change (do I want to remove a linear mean that is a function of some combination of lat/lon?)
# 4. determine spatial and temporal decorrelcation scales 
# 4. objectively map 


import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import netcdf
from scipy.integrate import cumtrapz
from scipy.linalg import solve
import seawater as sw 
from toolkit import cart2pol, pol2cart, plot_pro, nanseg_interp, data_covariance

def trend_fit(lon,lat,data):
    A = np.transpose([lon,lat,lon/lon])
    b = data
    C = np.linalg.lstsq(A,b)
    return C[0][0],C[0][1],C[0][2]
    
def createCorrelationMatrices(lon,lat,data_anom,data_sig,noise_sig,Lx,Ly):
# lon: vector of data longitudes
# lat: vector of data latitudes
# data_anom: vector of data anomalies from fitted large-scale trend
# data_sig: signal standard deviation, in data units
# noise_sig: noise standard deviation, in data units
# Lx: Zonal Gaussian covariance lengthscale, in km
# Ly: Meridional Gaussian covariance lengthscale, in km
    npts = len(lon)
    C = np.zeros((npts,npts),dtype=np.float)
    A = np.zeros((npts,npts),dtype=np.float)
    for j in range(npts):
        # xscale = np.cos(lat[j]*3.1415926/180.)
        for i in range(npts):
            # dxij = (lon[j]-lon[i])*111.0*xscale
            # dyij = (lat[j]-lat[i])*111.0
            dxij = lon[j] - lon[i]
            dyij = lat[j] - lat[i]
            C[j,i]=data_sig*data_sig*np.exp(-(dxij*dxij)/(Lx*Lx)-(dyij*dyij)/(Ly*Ly))
            if (i==j):
                A[j,i]=C[j,i]+noise_sig*noise_sig
            else:
                A[j,i]=C[j,i]
    Ainv = np.linalg.inv(A)
    return Ainv    
    
# START
# physical parameters 
g = 9.81
rho0 = 1027
bin_depth = np.concatenate([np.arange(0,150,10), np.arange(150,300,10), np.arange(300,5000,20)])
ref_lat = 31.7
ref_lon = -64.2
lat_in = 31.7
lon_in = 64.2
grid = bin_depth[1:-1]
grid_p = sw.pres(grid,lat_in)
z = -1*grid
grid_2 = grid[0:214]

# LOAD DATA (gridded dives)
GD = netcdf.netcdf_file('BATs_2015_gridded_2.nc','r')
profile_list = np.float64(GD.variables['dive_list'][:]) - 35000  
df_den = pd.DataFrame(np.float64(GD.variables['Density'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))


# select only dives with depths greater than 4000m 
grid_test = np.nan*np.zeros(len(profile_list))
for i in range(len(profile_list)):
    grid_test[i] = grid[np.where( np.array(df_den.iloc[:,i]) == np.nanmax(np.array(df_den.iloc[:,i])) )[0][0]]
good = np.where(grid_test > 4000)[0]   

# load select profiles
df_den = pd.DataFrame(np.float64(GD.variables['Density'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:])).iloc[:,good]
df_t = pd.DataFrame(np.float64(GD.variables['Temperature'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:])).iloc[:,good]
df_s = pd.DataFrame(np.float64(GD.variables['Salinity'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:])).iloc[:,good]
df_lon = pd.DataFrame(np.float64(GD.variables['Longitude'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:])).iloc[:,good]
df_lat = pd.DataFrame(np.float64(GD.variables['Latitude'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:])).iloc[:,good]
dac_u = GD.variables['DAC_u'][good]
dac_v = GD.variables['DAC_v'][good]
time_rec = GD.variables['time_start_stop'][good]
heading_rec = GD.variables['heading_record'][good]
profile_list = np.float64(GD.variables['dive_list'][good]) - 35000    
    
# interpolate nans that populate density profiles     
for i in range(len(profile_list)):    
    fix = nanseg_interp(grid,np.array(df_den.iloc[:,i]))
    df_den.iloc[:,i] = fix 
    fix_lon = nanseg_interp(grid,np.array(df_lon.iloc[:,i]))
    df_lon.iloc[:,i] = fix_lon
    fix_lat = nanseg_interp(grid,np.array(df_lat.iloc[:,i]))
    df_lat.iloc[:,i] = fix_lat

# extract DAC_U/V and lat/lon pos 
ev_oth = range(0,int(len(dac_u)),2)
count = 0
dac_lon = np.nan*np.zeros(int(len(dac_u)/2))
dac_lat = np.nan*np.zeros(int(len(dac_u)/2))
d_u = np.nan*np.zeros(int(len(dac_u)/2))
d_v = np.nan*np.zeros(int(len(dac_u)/2))
d_time = np.nan*np.zeros(int(len(dac_u)/2))
for p in ev_oth:
    dac_lon[count] = np.nanmean( [df_lon.iloc[:,p],df_lon.iloc[:,p+1]] )
    dac_lat[count] = np.nanmean( [df_lat.iloc[:,p],df_lat.iloc[:,p+1]] )
    d_u[count] = dac_u[p]
    d_v[count] = dac_v[p]
    d_time[count] = time_rec[p]
    count = count+1

############################################################################    
#### compute correlations and covariance ####
# estimate covariance function from data
# for all pairs of points di and dj compute & store
# 1) distance between them (km)
# 2) time lag between them (days)
# 3) their product: di*dj    

# pick density values on a depth level
dl = 100
t_int = np.where( (time_rec < np.nanmedian(time_rec) ))[0]
dc = np.array(df_den.iloc[dl,t_int])
d_lo = np.array(df_lon.iloc[dl,t_int])
d_la = np.array(df_lat.iloc[dl,t_int])
d_x = 1852*60*np.cos(np.deg2rad(ref_lat))*(d_lo - ref_lon)	
d_y = 1852*60*(d_la - ref_lat)  
d_t = time_rec[t_int].copy()

cx,cy,c0 = trend_fit(d_x,d_y,dc)
den_anom_o = dc-(cx*d_x + cy*d_y + c0)

dt = 10 # separation in time of points (up to 271 days )
ds = 10000 # separation in space (up to 100km)
Lt = 25
Ls = 50000
# den_var, cov_est = data_covariance(den_anom_o,d_x,d_y,d_t,dt,ds,Ls,Lt)

############################################################################    

# Parameters for objective mapping
Lx = 30000
Ly = 30000
lon_grid=np.arange(-64.7,-63.55,.05,dtype=np.float)
lat_grid=np.arange(31.3,32.0,.05,dtype=np.float)
x_grid = 1852*60*np.cos(np.deg2rad(ref_lat))*(lon_grid - ref_lon)	
y_grid = 1852*60*(lat_grid - ref_lat)     

# select time window from which to extract subset for initial objective mapping 
win_size = 18
t_windows = np.arange(np.nanmin(time_rec),np.nanmax(time_rec),win_size)
t_bin = np.nan*np.zeros((len(time_rec)-win_size,2))
for i in range(len(time_rec)-win_size):
    t_bin[i,:] = [time_rec[i], time_rec[i]+win_size]
    
k = 24
k_out = k
time_in = np.where( (time_rec > t_bin[k,0]) & (time_rec < t_bin[k,1]) )[0] # data
time_in_2 = np.where( (d_time > t_bin[k,0]) & (d_time < t_bin[k,1]) )[0] # DAC 

### LOOPING ### over depth layers 
sigma_theta = np.nan*np.zeros((len(lat_grid),len(lon_grid),len(grid)))
error = np.nan*np.zeros((len(lat_grid),len(lon_grid),len(grid)))
d_sigma_dx = np.nan*np.zeros((len(lat_grid),len(lon_grid),len(grid)))
d_sigma_dy = np.nan*np.zeros((len(lat_grid),len(lon_grid),len(grid)))
for k in range(len(grid_2)):
    depth = np.where(grid==grid[k])[0][0]
    lon_in = np.array(df_lon.iloc[depth,time_in])
    lat_in = np.array(df_lat.iloc[depth,time_in])
    den_in = np.array(df_den.iloc[depth,time_in])
    
    # attempt to correct for nan's 
    if len(np.where(np.isnan(den_in))[0]) > 0:
        den_up = np.array(df_den.iloc[depth-1,time_in]) 
        den_down = np.array(df_den.iloc[depth+1,time_in])
        lon_up = np.array(df_lon.iloc[depth-1,time_in]) 
        lon_down = np.array(df_lon.iloc[depth+1,time_in])
        lat_up = np.array(df_lat.iloc[depth-1,time_in]) 
        lat_down = np.array(df_lat.iloc[depth+1,time_in])
        bad = np.where(np.isnan(den_in))[0]
        for l in range(len(bad)):
            den_in[bad[l]] = np.interp(grid[depth],[grid[depth-1], grid[depth+1]],[den_up[bad[l]],den_down[bad[l]]])
            lon_in[bad[l]] = np.interp(grid[depth],[grid[depth-1], grid[depth+1]],[lon_up[bad[l]],lon_down[bad[l]]])
            lat_in[bad[l]] = np.interp(grid[depth],[grid[depth-1], grid[depth+1]],[lat_up[bad[l]],lat_down[bad[l]]])
            
    # convert to x,y distance
    x = 1852*60*np.cos(np.deg2rad(ref_lat))*(lon_in - ref_lon)	
    y = 1852*60*(lat_in - ref_lat)        

    # Fit a trend to the data 
    cx,cy,c0 = trend_fit(x,y,den_in)
    den_anom = den_in-(cx*x+cy*y+c0)
    den_sig = np.nanstd(den_anom)
    den_var = np.nanvar(den_anom)
    errsq = 0
    noise_sig = (np.max(den_in)-np.min(den_in))/6 # 0.01
    
    # data-data covariance (loop over nXn data points) 
    npts = len(x)
    C = np.zeros((npts,npts),dtype=np.float)
    data_data = np.zeros((npts,npts),dtype=np.float)
    for l in range(npts):
        for k0 in range(npts):
            dxij = (x[l]-x[k0])
            dyij = (y[l]-y[k0])
            C[l,k0]=den_var*np.exp(- (dxij*dxij)/(Lx*Lx) - (dyij*dyij)/(Ly*Ly) )
            if (k0==l):
                data_data[l,k0]=C[l,k0]+noise_sig*noise_sig
            else:
                data_data[l,k0]=C[l,k0]
    
    # loop over each grid point 
    Dmap = np.zeros((len(lat_grid),len(lon_grid)),dtype=np.float)
    Emap = np.zeros((len(lat_grid),len(lon_grid)),dtype=np.float)
    for j in range(len(lat_grid)):
        for i in range(len(lon_grid)):
            x_g = x_grid[i]
            y_g = y_grid[j]
            # data-grid
            data_grid=(den_var-errsq)*np.exp( -((x_g-x)/Lx)**2- ((y_g-y)/Ly)**2 )
                        
            alpha = solve(data_data,data_grid)        
            Dmap[j,i] = np.sum( [den_anom*alpha] ) + (cx*x_grid[i]+cy*y_grid[j]+c0)
            Emap[j,i] = np.sqrt(den_sig*den_sig - np.dot(data_grid,alpha))
            
    sigma_theta[:,:,k] = Dmap 
    error[:,:,k] = Emap    

    # # Create correlation matrix 
    # Ainv = createCorrelationMatrices(x,y,den_anom,data_sig,noise_sig,Lx,Ly)
    # # Map
    # Dmap = np.zeros((len(lat_grid),len(lon_grid)),dtype=np.float)
    # for j in range(len(lat_grid)):
    #     for i in range(len(lon_grid)):
    #         for n in range(len(lon_in)):
    #             dx0i = (x_grid[i]-x[n])   # (lon_grid[i]-lon_in[n])*111.0*xscale
    #             dy0i = (y_grid[j]-y[n])         # (lat_grid[i]-lat_in[n])*111.0
    #             C0i=data_sig*data_sig*np.exp(-(dx0i*dx0i)/(Lx*Lx)-(dy0i*dy0i)/(Ly*Ly))
    #             Dmap[j,i] = Dmap[j,i] + C0i*np.sum(Ainv[n,:]*den_anom)
    #         Dmap[j,i]=Dmap[j,i]+(cx*x_grid[i]+cy*y_grid[j]+c0)

    # sigma_theta[:,:,k] = Dmap
    d_sigma_dy[:,:,k], d_sigma_dx[:,:,k] = np.gradient(sigma_theta[:,:,k], y_grid[2]-y_grid[1], x_grid[2]-x_grid[1] )
    
    ## FINISH LOOPING OVER ALL DEPTH LAYERS 
    
# mapping for DAC vectors 
d_lon_in = dac_lon[time_in_2]
d_lat_in = dac_lat[time_in_2]
d_u_in = d_u[time_in_2]
d_v_in = d_v[time_in_2]  
d_x = 1852*60*np.cos(np.deg2rad(ref_lat))*(d_lon_in - ref_lon)	
d_y = 1852*60*(d_lat_in - ref_lat)
# Fit a trend to the data (average DAC)
mean_u = np.nanmean(d_u_in)
mean_v = np.nanmean(d_v_in)
u_anom = d_u_in - mean_u
v_anom = d_v_in - mean_v
d_u_sig = np.nanstd(u_anom)
d_v_sig = np.nanstd(v_anom)
noise_sig = 0.01
# Create correlation matrix 
Ainv_u = createCorrelationMatrices(d_x,d_y,u_anom,d_u_sig,noise_sig,Lx,Ly)
Ainv_v = createCorrelationMatrices(d_x,d_y,v_anom,d_v_sig,noise_sig,Lx,Ly)
# Map
DACU_map = np.zeros((len(lat_grid),len(lon_grid)),dtype=np.float)
DACV_map = np.zeros((len(lat_grid),len(lon_grid)),dtype=np.float)
for j in range(len(lat_grid)):
    for i in range(len(lon_grid)):
        for n in range(len(d_lon_in)):
            dx0i = (x_grid[i]-d_x[n])   # (lon_grid[i]-lon_in[n])*111.0*xscale
            dy0i = (y_grid[j]-d_y[n])         # (lat_grid[i]-lat_in[n])*111.0
            C0i=d_v_sig*d_u_sig*np.exp(-(dx0i*dx0i)/(Lx*Lx)-(dy0i*dy0i)/(Ly*Ly))
            DACU_map[j,i] = DACU_map[j,i] + C0i*np.sum(Ainv_u[n,:]*u_anom)
            DACV_map[j,i] = DACV_map[j,i] + C0i*np.sum(Ainv_v[n,:]*v_anom)
        DACU_map[j,i]=DACU_map[j,i]+mean_u
        DACV_map[j,i]=DACV_map[j,i]+mean_v

# fig, ax = plt.subplots()
# x1 = lon_in
# y1 = lat_in
# im = ax.plot(x1, den_in, '.r')
# plt.plot(x1,cx*lon_in+cy*lat_in+c0,'.k')
# plt.title('Plane Fit vs Latitude')
# plot_pro(ax)

# clr = matplotlib.cm.get_cmap('jet',30)
# f, ax = plt.subplots()
# ax.scatter(lon_in,lat_in,c=den_in,cmap=plt.cm.coolwarm)
# plot_pro(ax)

# GEOSTROPHIC SHEAR to geostrophic velocity 
ff = np.pi*np.sin(np.deg2rad(ref_lat))/(12*1800)
du_dz = np.nan*np.zeros(np.shape(d_sigma_dy))
dv_dz = np.nan*np.zeros(np.shape(d_sigma_dy))
Ubc_g = np.nan*np.zeros(np.shape(d_sigma_dy))
Vbc_g = np.nan*np.zeros(np.shape(d_sigma_dy))
U_g = np.nan*np.zeros(np.shape(d_sigma_dy))
V_g = np.nan*np.zeros(np.shape(d_sigma_dy))
for m in range(len(lat_grid)):
    for n in range(len(lon_grid)):
        for o in range(len(grid_2)):
            du_dz[m,n,o] = (g/(rho0*ff))*d_sigma_dy[m,n,o]
            dv_dz[m,n,o] = (-g/(rho0*ff))*d_sigma_dx[m,n,o]
            
        iq = np.where( ~np.isnan(du_dz[m,n,:]) )    
        if np.size(iq) > 10:
            z2 = -grid_2[iq]    
            # u
            urel = cumtrapz(du_dz[m,n,iq],x=z2,initial=0)
            urel_av = np.trapz(urel/(z2[-1] - z2[0]), x=z2)
            ubc = urel - urel_av
            Ubc_g[m,n,iq] = ubc    
            # v
            vrel = cumtrapz(dv_dz[m,n,iq],x=z2,initial=0)
            vrel_av = np.trapz(vrel/(z2[-1] - z2[0]), x=z2)
            vbc = vrel - vrel_av
            Vbc_g[m,n,iq] = vbc
                     
            U_g[m,n,iq] = DACU_map[m,n] + ubc
            V_g[m,n,iq] = DACV_map[m,n] + vbc 


k1 = 50
k2 = 100
k3 = 150
lim = 50
fig, ax_ar = plt.subplots(2,3,sharey=True,sharex=True)
x1,y1 = np.meshgrid(x_grid,y_grid)

cont_data = np.array(df_den.iloc[np.where(grid==grid[k1])[0][0],time_in])
this_x = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lon.iloc[np.where(grid==grid[k1])[0][0],time_in]) - ref_lon )
this_y = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lat.iloc[np.where(grid==grid[k1])[0][0],time_in]) - ref_lat )
ax_ar[0,0].scatter(this_x/1000, this_y/1000, c=np.array(df_den.iloc[np.where(grid==grid[k1])[0][0],time_in]),s=60,cmap=plt.cm.coolwarm,zorder=2,vmin=np.min(cont_data),vmax=np.max(cont_data))
im = ax_ar[0,0].pcolor(x1/1000, y1/1000, sigma_theta[:,:,k1], cmap=plt.cm.coolwarm,zorder=0,vmin=np.min(cont_data),vmax=np.max(cont_data))
ax_ar[0,0].quiver(d_x/1000,d_y/1000,d_u_in,d_v_in, color='k',angles='xy', scale_units='xy', scale=.005,zorder=1)
ax_ar[0,0].quiver(x1/1000, y1/1000, U_g[:,:,k1], V_g[:,:,k1], color='g',angles='xy', scale_units='xy', scale=.005,zorder=1)
ax_ar[0,0].set_title('Objectively mapped grid at ' + str(grid[k1]) + 'm')
ax_ar[0,0].axis([-lim,lim,-lim,lim])
ax_ar[0,0].grid()
plt.colorbar(im,ax=ax_ar[0,0],orientation='horizontal')

cont_data = np.array(df_den.iloc[np.where(grid==grid[k2])[0][0],time_in])
this_x = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lon.iloc[np.where(grid==grid[k2])[0][0],time_in]) - ref_lon )
this_y = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lat.iloc[np.where(grid==grid[k2])[0][0],time_in]) - ref_lat )
ax_ar[0,1].scatter(this_x/1000, this_y/1000, c=np.array(df_den.iloc[np.where(grid==grid[k2])[0][0],time_in]),s=60,cmap=plt.cm.coolwarm,zorder=2,vmin=np.min(cont_data),vmax=np.max(cont_data))
im = ax_ar[0,1].pcolor(x1/1000, y1/1000, sigma_theta[:,:,k2], cmap=plt.cm.coolwarm,zorder=0,vmin=np.min(cont_data),vmax=np.max(cont_data))
ax_ar[0,1].quiver(d_x/1000,d_y/1000,d_u_in,d_v_in, color='k',angles='xy', scale_units='xy', scale=.005,zorder=1)
ax_ar[0,1].quiver(x1/1000, y1/1000, U_g[:,:,k2], V_g[:,:,k2], color='g',angles='xy', scale_units='xy', scale=.005,zorder=1)
ax_ar[0,1].set_title('Objectively mapped grid at ' + str(grid[k2]) + 'm')
ax_ar[0,1].axis([-lim,lim,-lim,lim])
ax_ar[0,1].grid()
plt.colorbar(im,ax=ax_ar[0,1],orientation='horizontal')

cont_data = np.array(df_den.iloc[np.where(grid==grid[k3])[0][0],time_in])
this_x = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lon.iloc[np.where(grid==grid[k3])[0][0],time_in]) - ref_lon )
this_y = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lat.iloc[np.where(grid==grid[k3])[0][0],time_in]) - ref_lat )
ax_ar[0,2].scatter(this_x/1000, this_y/1000, c=np.array(df_den.iloc[np.where(grid==grid[k3])[0][0],time_in]),s=60,cmap=plt.cm.coolwarm,zorder=2,vmin=np.min(cont_data),vmax=np.max(cont_data))
im = ax_ar[0,2].pcolor(x1/1000, y1/1000, sigma_theta[:,:,k3], cmap=plt.cm.coolwarm,zorder=0,vmin=np.min(cont_data),vmax=np.max(cont_data))
ax_ar[0,2].quiver(d_x/1000,d_y/1000,d_u_in,d_v_in, color='k',angles='xy', scale_units='xy', scale=.005,zorder=1)
ax_ar[0,2].quiver(x1/1000, y1/1000, U_g[:,:,k3], V_g[:,:,k3], color='g',angles='xy', scale_units='xy', scale=.005,zorder=1)
ax_ar[0,2].set_title('Objectively mapped grid at ' + str(grid[k3]) + 'm')
ax_ar[0,2].axis([-lim,lim,-lim,lim])
ax_ar[0,2].grid()
plt.colorbar(im,ax=ax_ar[0,2],orientation='horizontal')

# im = ax_ar[1,0].quiver(x1/1000, y1/1000, d_sigma_dx[:,:,k1], d_sigma_dy[:,:,k1], color='g',angles='xy', scale_units='xy', scale=.0000001)
im = ax_ar[1,0].pcolor(x1/1000, y1/1000, error[:,:,k1], cmap=plt.cm.plasma,zorder=0) # ,vmin=0.0005,vmax=.02)
plt.colorbar(im,ax=ax_ar[1,0],orientation='horizontal')
ax_ar[1,0].set_title('Error Map')
ax_ar[1,0].axis([-lim,lim,-lim,lim])
ax_ar[1,0].grid()

im = ax_ar[1,1].pcolor(x1/1000, y1/1000, error[:,:,k2], cmap=plt.cm.plasma,zorder=0) # ,vmin=0.0005,vmax=.02)
plt.colorbar(im,ax=ax_ar[1,1],orientation='horizontal')
ax_ar[1,1].axis([-lim,lim,-lim,lim])
ax_ar[1,1].set_title('Error Map')
ax_ar[1,1].grid()

im = ax_ar[1,2].pcolor(x1/1000, y1/1000, error[:,:,k3], cmap=plt.cm.plasma,zorder=0) #,vmin=0.0005,vmax=.02)
plt.colorbar(im,ax=ax_ar[1,2],orientation='horizontal')
ax_ar[1,2].set_title('Error Map')
ax_ar[1,2].axis([-lim,lim,-lim,lim])
plot_pro(ax_ar[1,2])


# MASK consider only profiles that are within a low error region 
# error values are plus/minus in units of the signal (in this case it density)
# include only u/v profiles with density error less than 0.01 
# - average error on the vertical 
error_mask = np.zeros( (len(lat_grid),len(lon_grid),len(grid_2)) )
for i in range(len(grid_2)):
    sig_range = (np.nanmax(sigma_theta[:,:,i]) - np.nanmin(sigma_theta[:,:,i]))/5
    good = np.where(error[:,:,i] < sig_range)
    error_mask[good[0],good[1],i] = 1
error_test = np.sum(error_mask,axis=2)    
good_prof = np.where(error_test > 180)

fig,ax = plt.subplots()
ax.scatter(x1[good_prof[0],good_prof[1]]/1000,y1[good_prof[0],good_prof[1]]/1000)
ax.axis([-lim,lim,-lim,lim])
ax.set_title('lat/lon positions with high confidence estimates')
plot_pro(ax)

error_av = np.nanmean(error,axis=2)
np.ma.masked_where()

min_err

### plot U,V 
fig, (ax0,ax1) = plt.subplots(1,2,sharey=True)
for i in range(len(good_prof[0])):
    ax0.plot(U_g[good_prof[0][i],good_prof[1][i],:],grid,color='r',linewidth=0.25)
    ax1.plot(V_g[good_prof[0][i],good_prof[1][i],:],grid,color='b',linewidth=0.25)
ax0.set_title('U')
ax0.set_ylabel('Depth [m]')
ax0.set_xlabel('m/s')
ax0.axis([-.3,.3,0,4250])
ax1.set_title('V')
ax1.set_xlabel('m/s')
ax1.axis([-.3,.3,0,4250])
ax0.grid()
ax0.invert_yaxis() 
plot_pro(ax1)


### OUTPUT
U_out = U_g[good_prof[0],good_prof[1],:]
V_out = V_g[good_prof[0],good_prof[1],:]
sigma_theta_out = sigma_theta[good_prof[0],good_prof[1],:]
lon_out = lon_grid[good_prof[1]]
lat_out = lat_grid[good_prof[0]]
time_out = t_bin[k_out,:]

### SAVE 
# write python dict to a file
sa = 0
if sa > 0:
    mydict = {'depth': grid,'Sigma_Theta': sigma_theta,'U_g': U_g, 'V_g': V_g, 'lon_grid': lon_grid, 'lat_grid': lat_grid, 'x_grid': x_grid, 'y_grid': y_grid}
    output = open('/Users/jake/Documents/geostrophic_turbulence/BATS_obj_map.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()
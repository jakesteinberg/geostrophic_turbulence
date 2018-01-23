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
import seawater as sw 
from toolkit import cart2pol, pol2cart, plot_pro

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
        xscale = np.cos(lat[j]*3.1415926/180.)
        for i in range(npts):
            dxij = (lon[j]-lon[i])*111.0*xscale
            dyij = (lat[j]-lat[i])*111.0
            C[j,i]=data_sig*data_sig*np.exp(-(dxij*dxij)/(Lx*Lx)-(dyij*dyij)/(Ly*Ly))
            if (i==j):
                A[j,i]=C[j,i]+noise_sig*noise_sig
            else:
                A[j,i]=C[j,i]
    Ainv = np.linalg.inv(A)
    return Ainv    
    

# LOAD DATA (gridded dives)
GD = netcdf.netcdf_file('BATs_2015_gridded_2.nc','r')
df_den = pd.DataFrame(np.float64(GD.variables['Density'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_t = pd.DataFrame(np.float64(GD.variables['Temperature'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_s = pd.DataFrame(np.float64(GD.variables['Salinity'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_lon = pd.DataFrame(np.float64(GD.variables['Longitude'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_lat = pd.DataFrame(np.float64(GD.variables['Latitude'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
dac_u = GD.variables['DAC_u'][:]
dac_v = GD.variables['DAC_v'][:]
time_rec = GD.variables['time_start_stop'][:]
heading_rec = GD.variables['heading_record'][:]
profile_list = np.float64(GD.variables['dive_list'][:]) - 35000    

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
grid_2 = grid[0:228]

# Parameters for objective mapping
Lx = 40000
Ly = 40000
data_sig = 2.0
noise_sig = 2.0
lon_grid=np.arange(-64.8,-63.6,.05,dtype=np.float)
lat_grid=np.arange(31.3,32.1,.05,dtype=np.float)
x_grid = 1852*60*np.cos(np.deg2rad(ref_lat))*(lon_grid - ref_lon)	
y_grid = 1852*60*(lat_grid - ref_lat)     

# select time window from which to extract subset for initial objective mapping 
t_windows = np.arange(np.nanmin(time_rec),np.nanmax(time_rec),25)
k = 2
time_in = np.where( (time_rec > t_windows[k]) & (time_rec < t_windows[k+1]) )[0] # data
time_in_2 = np.where( (d_time > t_windows[k]) & (d_time < t_windows[k+1]) )[0] # DAC 
# loop over depth layers 
sigma_theta = np.nan*np.zeros((len(lat_grid),len(lon_grid),len(grid)))
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
    # Create correlation matrix 
    Ainv = createCorrelationMatrices(x,y,den_anom,data_sig,noise_sig,Lx,Ly)
    # Map
    Dmap = np.zeros((len(lat_grid),len(lon_grid)),dtype=np.float)
    for j in range(len(lat_grid)):
        # xscale = np.cos(lat_grid[j]*3.1415926/180.)
        for i in range(len(lon_grid)):
            for n in range(len(lon_in)):
                dx0i = (x_grid[i]-x[n])   # (lon_grid[i]-lon_in[n])*111.0*xscale
                dy0i = (y_grid[j]-y[n])         # (lat_grid[i]-lat_in[n])*111.0
                C0i=data_sig*data_sig*np.exp(-(dx0i*dx0i)/(Lx*Lx)-(dy0i*dy0i)/(Ly*Ly))
                Dmap[j,i] = Dmap[j,i] + C0i*np.sum(Ainv[n,:]*den_anom)
            Dmap[j,i]=Dmap[j,i]+(cx*x_grid[i]+cy*y_grid[j]+c0)

    sigma_theta[:,:,k] = Dmap
    d_sigma_dy[:,:,k], d_sigma_dx[:,:,k] = np.gradient(sigma_theta[:,:,k], y_grid[2]-y_grid[1], x_grid[2]-x_grid[1] )
    
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
# Create correlation matrix 
Ainv_u = createCorrelationMatrices(d_x,d_y,u_anom,data_sig,noise_sig,Lx,Ly)
Ainv_v = createCorrelationMatrices(d_x,d_y,v_anom,data_sig,noise_sig,Lx,Ly)
# Map
DACU_map = np.zeros((len(lat_grid),len(lon_grid)),dtype=np.float)
DACV_map = np.zeros((len(lat_grid),len(lon_grid)),dtype=np.float)
for j in range(len(lat_grid)):
    for i in range(len(lon_grid)):
        for n in range(len(d_lon_in)):
            dx0i = (x_grid[i]-d_x[n])   # (lon_grid[i]-lon_in[n])*111.0*xscale
            dy0i = (y_grid[j]-d_y[n])         # (lat_grid[i]-lat_in[n])*111.0
            C0i=data_sig*data_sig*np.exp(-(dx0i*dx0i)/(Lx*Lx)-(dy0i*dy0i)/(Ly*Ly))
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


kk = 50
lim = 60000
fig, ax_ar = plt.subplots(2,2,sharey=True)
x1,y1 = np.meshgrid(x_grid,y_grid)
this_x = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lon.iloc[np.where(grid==grid[kk])[0][0],time_in]) - ref_lon )
this_y = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lat.iloc[np.where(grid==grid[kk])[0][0],time_in]) - ref_lat )
ax_ar[0,0].scatter(this_x, this_y, c=np.array(df_den.iloc[np.where(grid==grid[kk])[0][0],time_in]),cmap=plt.cm.coolwarm,zorder=1)
im = ax_ar[0,0].pcolor(x1, y1, sigma_theta[:,:,kk], cmap=plt.cm.coolwarm,zorder=0)
plt.colorbar(im,ax=ax_ar[0,0])
ax_ar[0,0].set_title('Objectively mapped grid')
ax_ar[0,0].axis([-lim,lim,-lim,lim])
ax_ar[0,0].grid()

im = ax_ar[0,1].quiver(x1, y1, d_sigma_dx[:,:,kk], d_sigma_dy[:,:,kk], color='g',angles='xy', scale_units='xy', scale=.00000000001)
ax_ar[0,1].set_title('mapped density gradient grid')
ax_ar[0,1].axis([-lim,lim,-lim,lim])
ax_ar[0,1].grid()

im = ax_ar[1,0].quiver(x1, y1, DACU_map, DACV_map, color='g',angles='xy', scale_units='xy', scale=.000005)
ax_ar[1,0].quiver(d_x,d_y,d_u_in,d_v_in, color='r',angles='xy', scale_units='xy', scale=.000005)
ax_ar[1,0].axis([-lim,lim,-lim,lim])
ax_ar[1,0].grid()

im = ax_ar[1,1].quiver(x1, y1, U_g[:,:,kk], V_g[:,:,kk], color='g',angles='xy', scale_units='xy', scale=.000005)
plot_pro(ax_ar[1,0])
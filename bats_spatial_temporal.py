# BATS isotropy, spatial variabilty, temporal changes of T/S
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import datetime
import altair as alt
from scipy.io import netcdf
import sys
import seawater as sw 
from toolkit import cart2pol, pol2cart, plot_pro

## bathymetry 
bath = '/Users/jake/Desktop/bats/bats_bathymetry/GEBCO_2014_2D_-67.7_29.8_-59.9_34.8.nc'
bath_fid = netcdf.netcdf_file(bath,'r') # , mmap=False if sys.platform == 'darwin' else mmap, version=1)
bath_lon = bath_fid.variables['lon'][:]
bath_lat = bath_fid.variables['lat'][:]
bath_z = bath_fid.variables['elevation'][:]
levels_b = [ -5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500 , 0]

# LOAD DATA (gridded dives)
GD = netcdf.netcdf_file('BATs_2015_gridded_2.nc','r')
df_den = pd.DataFrame(np.float64(GD.variables['Density'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_t = pd.DataFrame(np.float64(GD.variables['Temperature'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_s = pd.DataFrame(np.float64(GD.variables['Salinity'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_lon = pd.DataFrame(np.float64(GD.variables['Longitude'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_lat = pd.DataFrame(np.float64(GD.variables['Latitude'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
dac_u = GD.variables['DAC_u'][:]
dac_v = GD.variables['DAC_v'][:]
# time_rec = GD.variables['time_rec'][:]
time_rec = GD.variables['time_start_stop'][:]
heading_rec = GD.variables['heading_record'][:]
profile_list = np.float64(GD.variables['dive_list'][:]) - 35000

# LOAD TRANSECT TO PROFILE DATA COMPILED IN BATS_TRANSECTS.PY
pkl_file = open('/Users/jake/Desktop/bats/transect_profiles_jan18.pkl', 'rb')
bats_trans = pickle.load(pkl_file)
pkl_file.close() 
Time = bats_trans['Time']
Info = bats_trans['Info']
Sigma_Theta = bats_trans['Sigma_Theta']
Eta = bats_trans['Eta']
Eta_theta = bats_trans['Eta_theta']
V = bats_trans['V']
U_g = bats_trans['UU']
V_g = bats_trans['VV']
UV_lon = bats_trans['V_lon']
UV_lat = bats_trans['V_lat']
Heading = bats_trans['Heading']

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

# goal: look at spatial variability of T/S fields on varied time scales at various depths 
# goal: look at temporal variability of DAC field 

### TIME WINDOWS 
lon_per_prof = np.nanmean(df_lon,axis=0)
lat_per_prof = np.nanmean(df_lat,axis=0)
t_win = (time_rec[-1] - time_rec[0])/12 # is just shy of 20 days ... this is the amount of time it took to execute one butterfly pattern
t_windows = np.arange(time_rec[0],time_rec[-1],20)
avg_u = np.zeros(np.size(t_windows)-1)
avg_v = np.zeros(np.size(t_windows)-1)
for i in range(np.size(t_windows)-1):
    avg_u[i] = np.nanmean( dac_u[ np.where((time_rec > t_windows[i]) & (time_rec < t_windows[i+1]))[0] ] )
    avg_v[i] = np.nanmean( dac_v[ np.where((time_rec > t_windows[i]) & (time_rec < t_windows[i+1]))[0] ] )
t_s = datetime.date.fromordinal(np.int( np.min(time_rec[0]) ))
t_e = datetime.date.fromordinal(np.int( np.max(time_rec[-1]) ))


### U/V ON DEPTH LAYERS 
dp = 50
k = 3
time_in = np.where( (Time > t_windows[k]) & (Time < t_windows[k+1]) )
track_in = np.where( (time_rec > t_windows[k]) & (time_rec < t_windows[k+1]) )[0]
### plotting 
fig0, ax0 = plt.subplots()
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
ax0.contour(bath_lon,bath_lat,bath_z,levels=levels_b,colors='k',linewidths=0.5,zorder=0)
ax0.scatter(df_lon.iloc[:,track_in],df_lat.iloc[:,track_in],s=2,color='r')
ax0.quiver(UV_lon[time_in],UV_lat[time_in],U_g[dp,time_in],V_g[dp,time_in],color='b',angles='xy', scale_units='xy', scale=.8, width=0.004)
ax0.axis([-65.5, -63.35, 31.2, 32.7])
plt.tight_layout()
ax0.grid()
plot_pro(ax0)

### U/V AVERAGE PROFILES
# presort V 
good_u = np.zeros(np.size(Time))
good_v = np.zeros(np.size(Time))
for i in range(np.size(Time)):
    u_dz = np.gradient(U_g[10:,i])
    v_dz = np.gradient(V_g[10:,i])
    if np.nanmax(np.abs(u_dz)) < 0.075: 
        good_u[i] = 1     
    if np.nanmax(np.abs(v_dz)) < 0.075: 
        good_v[i] = 1  
good0 = np.intersect1d(np.where((np.abs(V[-45,:]) < 0.2))[0],np.where((np.abs(V[10,:]) < 0.4))[0])
good = np.intersect1d(np.where(good_u > 0),np.where(good_v > 0)) #,good0)
U_g_2 = U_g[:,good]
V_g_2 = V_g[:,good]

num_prof = np.size(good)
fig0, ax0 = plt.subplots()
for i in range(5,10):
    ax0.plot(U_g_2[:,i],grid,color='r')
    ax0.plot(V_g_2[:,i],grid,color='b')
ax0.plot(np.nanmean(U_g_2,axis=1),grid,color='k',linewidth=2) 
ax0.plot(np.nanmean(V_g_2,axis=1),grid,color='k',linewidth=2)    
plot_pro(ax0)

### DAC AND ITS AVERAGE 
fig0, ax0 = plt.subplots()
ax0.quiver(np.zeros(np.size(dac_u)),np.zeros(np.size(dac_u)),dac_u,dac_v,color='r',angles='xy', scale_units='xy', scale=1)
ax0.quiver(0,0,np.nanmean(dac_u),np.nanmean(dac_v),color='b',angles='xy', scale_units='xy', scale=1)
for i in range(np.size(avg_u)):
    ax0.quiver(0,0,avg_u[i],avg_v[i],color='g',angles='xy', scale_units='xy', scale=1)
ax0.axis([-.2, .2, -.2, .2])
plt.tight_layout()
plt.gca().set_aspect('equal')
plot_pro(ax0)

### TEMPERATURE on depth surfaces with time 
# depth = 3000
# t_map = np.linspace(2.2,2.5,30)
# depth = 1000
# t_map = np.linspace(5,8,30)
depth = 4000
t_map = np.linspace(1.75,1.95,30)

i = 0
this_depth_i = np.where(grid == depth)[0][0]
color = matplotlib.cm.get_cmap('jet',30)

# Using contourf to provide my colorbar info, then clearing the figure
Z = [[0,0],[0,0]]
levels = t_map
CS3 = plt.contourf(Z, levels, cmap=color)
plt.clf()

fig0, axes = plt.subplots(nrows=2,ncols=5)
count = 0 
for ax_j in axes.flat:
    this_time_i = np.where( (time_rec >= t_windows[count]) & (time_rec <= t_windows[count+1]) )[0]
    this_theta = df_t.iloc[this_depth_i,this_time_i]

    for k in range(np.size(this_theta)):
        idx = (np.abs(t_map-this_theta.iloc[k])).argmin()
        axx = ax_j.scatter(df_lon.iloc[this_depth_i,this_time_i[k]],df_lat.iloc[this_depth_i,this_time_i[k]],s=40,c=color(idx))
    count = count+1   
ax_j.grid()
fig0.subplots_adjust(right=0.8)
cbar_ax = fig0.add_axes([0.85, 0.15, 0.05, 0.7])
fig0.colorbar(CS3, cax=cbar_ax)
plot_pro(ax_j)
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
levels = [ -5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500 , 0]


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


lon_per_prof = np.nanmean(df_lon,axis=0)
lat_per_prof = np.nanmean(df_lat,axis=0)
t_win = (time_rec[-1] - time_rec[0])/14 # is just shy of 20 days ... this is the amount of time it took to execute one butterfly pattern
t_windows = np.arange(time_rec[0],time_rec[-1],20)
avg_u = np.zeros(np.size(t_windows)-1)
avg_v = np.zeros(np.size(t_windows)-1)
for i in range(np.size(t_windows)-1):
    avg_u[i] = np.nanmean( dac_u[ np.where((time_rec > t_windows[i]) & (time_rec < t_windows[i+1]))[0] ] )
    avg_v[i] = np.nanmean( dac_v[ np.where((time_rec > t_windows[i]) & (time_rec < t_windows[i+1]))[0] ] )
t_s = datetime.date.fromordinal(np.int( np.min(time_rec[0]) ))
t_e = datetime.date.fromordinal(np.int( np.max(time_rec[-1]) ))


### plotting 
fig0, ax0 = plt.subplots()
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
ax0.contour(bath_lon,bath_lat,bath_z,levels,colors='k',zorder=0)
ax0.scatter(lon_per_prof[0:27],lat_per_prof[0:27],s=15,color='r')
ax0.axis([-65.5, -63.35, 31.2, 32.7])
plt.tight_layout()
ax0.grid()
plot_pro(ax0)

fig0, ax0 = plt.subplots()
ax0.quiver(np.zeros(np.size(dac_u)),np.zeros(np.size(dac_u)),dac_u,dac_v,color='r',angles='xy', scale_units='xy', scale=1)
ax0.quiver(0,0,np.nanmean(dac_u),np.nanmean(dac_v),color='b',angles='xy', scale_units='xy', scale=1)
for i in range(np.size(avg_u)):
    ax0.quiver(0,0,avg_u[i],avg_v[i],color='g',angles='xy', scale_units='xy', scale=1)
ax0.axis([-.2, .2, -.2, .2])
plt.tight_layout()
plt.gca().set_aspect('equal')
plot_pro(ax0)

### temperature on depth surfaces with time 
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
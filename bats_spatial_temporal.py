# BATS isotropy, spatial variabilty, temporal changes of T/S
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import datetime
import altair as alt
from scipy.io import netcdf
import sys
import pickle
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
U_g = bats_trans['UU'] # this is wrong
V_g = bats_trans['VV'] # this is wrong 
UV_lon = bats_trans['V_lon']
UV_lat = bats_trans['V_lat']
Heading = bats_trans['Heading']
# the ordering that these lists are compiled is by transect no., not by time

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

### U/V AVERAGE PROFILES / NOT CORRECT 
# exclude u,v profiles with large gradients (some issue with m/w profiling)
# good_u = np.zeros(np.size(Time))
# good_v = np.zeros(np.size(Time))
# for i in range(np.size(Time)):
#     u_dz = np.gradient(U_g[10:,i])
#     v_dz = np.gradient(V_g[10:,i])
#     if np.nanmax(np.abs(u_dz)) < 0.075: 
#         good_u[i] = 1     
#     if np.nanmax(np.abs(v_dz)) < 0.075: 
#         good_v[i] = 1  
# good0 = np.intersect1d(np.where((np.abs(V[-45,:]) < 0.2))[0],np.where((np.abs(V[10,:]) < 0.4))[0])
# good = np.intersect1d(np.where(good_u > 0),np.where(good_v > 0)) #,good0) 
# U_g_2 = U_g[:,good]
# V_g_2 = V_g[:,good]
# Time_2 = Time[good]
# num_prof = np.size(good) # lose roughly 10 out of 179 


### U/V ON DEPTH LAYERS 
# dp = 50 # depth layer
# k = 6 # time window number (each is 20 days, the time it takes to carry out a bowtie pattern)
# time_in = np.where( (Time > t_windows[k]) & (Time < t_windows[k+1]) ) # for U,V vectors 
# u_v_in = Info[0,time_in] - 35000 
# track_in = np.where( (time_rec > t_windows[k]) & (time_rec < t_windows[k+1]) )[0] # for dive position (from gridded profiles)
# dives_in = df_lon.columns[track_in] - 35000

### plot plan view dives and associated u,v profiles 
# f, (ax0, ax1) = plt.subplots(1, 2)
# matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
# ax0.contour(bath_lon,bath_lat,bath_z,levels=levels_b,colors='k',linewidths=0.5,zorder=0)
# ax0.scatter(df_lon.iloc[:,track_in],df_lat.iloc[:,track_in],s=2,color='r')
# ax0.quiver(UV_lon[time_in],UV_lat[time_in],U_g[dp,time_in],V_g[dp,time_in],color='b',angles='xy', scale_units='xy', scale=.7, width=0.004)
# ax0.set_title('Dives: ' + str(int(np.nanmin(u_v_in))) + ':' + str(int(np.nanmax(u_v_in) )) )
# ax0.axis([-65.5, -63.35, 31.2, 32.7])
# w = 1/np.cos(np.deg2rad(ref_lat))
# ax0.set_aspect(w)
# plt.tight_layout()
# ax0.grid()

# for i in time_in:
#     uu = ax1.plot(U_g_2[:,i],grid,color='r',label='u')
#     vv = ax1.plot(V_g_2[:,i],grid,color='b',label='v')
# ax1.plot(np.nanmean(U_g_2,axis=1),grid,color='k',linewidth=2) 
# ax1.plot(np.nanmean(V_g_2,axis=1),grid,color='k',linewidth=2)  
# ax1.set_title('Dive Profiles: ' + str(int(np.nanmin(dives_in))) + ':' + str(int(np.nanmax(dives_in) )) )  
# plt.legend(handles=[uu[0], vv[0]])
# ax1.invert_yaxis() 
# plot_pro(ax1)

###### PARTITION OF U/V WITH DEPTh
# depth_levels = np.arange(100,5000,100)
# time_s = np.nanmin(Time_2)
# time_e = np.nanmin(Time_2) + (np.nanmax(Time_2) - np.nanmin(Time_2))/2
# long_time_win = np.where( (Time >= time_s) & (Time <= time_e) )[0]
# long_wins = np.linspace(np.nanmin(Time_2),np.nanmax(Time_2),9)
# color = matplotlib.cm.get_cmap('viridis',np.size(depth_levels))
# f, ax_ar = plt.subplots(2,4,sharey=True)
# ax_ind = np.array( [[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3]] )
# for j in range(8):
#     long_time_win = np.where( (Time_2 >= long_wins[j]) & (Time_2 <= long_wins[j+1]) )[0]
#     ax_ar[ax_ind[j][0],ax_ind[j][1]].plot([-.1,.1],[0,0],color='k',linewidth=1.5)
#     ax_ar[ax_ind[j][0],ax_ind[j][1]].plot([0,0],[-.1,.1],color='k',linewidth=1.5)
#     for i in range(np.size(depth_levels)-1):
#         dp_in = np.where( (grid >= depth_levels[i]) & (grid <= depth_levels[i+1]) )[0]
#         ax_ar[ax_ind[j][0],ax_ind[j][1]].scatter(np.nanmean(U_g_2[dp_in[:,None],long_time_win]),np.nanmean(V_g_2[dp_in[:,None],long_time_win]),c=color(i))
#         ax_ar[ax_ind[j][0],ax_ind[j][1]].axis([-.075, .075,-.075, .075])
#     ax_ar[ax_ind[j][0],ax_ind[j][1]].grid()
        
# ax_ar[ax_ind[j][0],ax_ind[j][1]].grid()        
# plot_pro(ax_ar[ax_ind[j][0],ax_ind[j][1]])    

###### DAC AND ITS AVERAGE ######
# fig0, ax0 = plt.subplots()
# ax0.quiver(np.zeros(np.size(dac_u)),np.zeros(np.size(dac_u)),dac_u,dac_v,color='r',angles='xy', scale_units='xy', scale=1)
# ax0.quiver(0,0,np.nanmean(dac_u),np.nanmean(dac_v),color='b',angles='xy', scale_units='xy', scale=1)
# for i in range(np.size(avg_u)):
#     ax0.quiver(0,0,avg_u[i],avg_v[i],color='g',angles='xy', scale_units='xy', scale=1)
# ax0.axis([-.2, .2, -.2, .2])
# plt.tight_layout()
# plt.gca().set_aspect('equal')
# plot_pro(ax0)

###### TEMPERATURE on depth surfaces with time ######
# depth = 3000
# t_map = np.linspace(2.2,2.5,30)
# depth = 1000
# t_map = np.linspace(5,8,30)
# depth = 4000
# t_map = np.linspace(1.75,1.95,30)

# i = 0
# this_depth_i = np.where(grid == depth)[0][0]
# color = matplotlib.cm.get_cmap('jet',30)

# Using contourf to provide my colorbar info, then clearing the figure
# Z = [[0,0],[0,0]]
# levels = t_map
# CS3 = plt.contourf(Z, levels, cmap=color)
# plt.clf()

# fig0, axes = plt.subplots(nrows=2,ncols=5)
# count = 0 
# for ax_j in axes.flat:
#     this_time_i = np.where( (time_rec >= t_windows[count]) & (time_rec <= t_windows[count+1]) )[0]
#     this_theta = df_t.iloc[this_depth_i,this_time_i]

#     for k in range(np.size(this_theta)):
#         idx = (np.abs(t_map-this_theta.iloc[k])).argmin()
#         axx = ax_j.scatter(df_lon.iloc[this_depth_i,this_time_i[k]],df_lat.iloc[this_depth_i,this_time_i[k]],s=40,c=color(idx))
#     count = count+1   
# ax_j.grid()
# fig0.subplots_adjust(right=0.8)
# cbar_ax = fig0.add_axes([0.85, 0.15, 0.05, 0.7])
# fig0.colorbar(CS3, cax=cbar_ax)
# plot_pro(ax_j)
# read ABACO shipboard ctd 

import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io as si
import datetime 
import seawater as sw
import pickle
from toolkit import unq_searchsorted, plot_pro

file_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_may_2017/ab*.cnv')

# name 0 = scan: Scan Count
# name 1 = timeJ: Julian Days
# name 2 = timeS: Time, Elapsed [seconds]
# name 3 = prDM: Pressure, Digiquartz [db]
# name 4 = t090C: Temperature [ITS-90, deg C]
# name 5 = t190C: Temperature, 2 [ITS-90, deg C]
# name 6 = c0S/m: Conductivity [S/m]
# name 7 = c1S/m: Conductivity, 2 [S/m]
# name 8 = sal00: Salinity, Practical [PSU]
# name 9 = sal11: Salinity, Practical, 2 [PSU]
# name 10 = latitude: Latitude [deg]
# name 11 = longitude: Longitude [deg]
# name 12 = sbeox0V: Oxygen raw, SBE 43 [V]
# name 13 = sbeox1V: Oxygen raw, SBE 43, 2 [V]
# name 14 = sbeox0ML/L: Oxygen, SBE 43 [ml/l]
# name 15 = sbeox1ML/L: Oxygen, SBE 43, 2 [ml/l]
# name 16 = flag: flag

# bin_press = np.arange(0,5200,5)
# bin_press = np.concatenate([np.arange(0,150,5), np.arange(150,300,10), np.arange(300,5200,20)])
bin_depth = np.concatenate([np.arange(0,150,10), np.arange(150,300,10), np.arange(300,5000,20)])
bin_press = sw.pres(bin_depth,26.5)
T1 = np.nan*np.ones( (np.size(bin_press),np.size(file_list)-10) )
S1 = np.nan*np.ones( (np.size(bin_press),np.size(file_list)-10) )
T2 = np.nan*np.ones( (np.size(bin_press),np.size(file_list)-10) )
S2 = np.nan*np.ones( (np.size(bin_press),np.size(file_list)-10) )
lat = np.nan*np.ones( (np.size(bin_press),np.size(file_list)-10) )
lon = np.nan*np.ones( (np.size(bin_press),np.size(file_list)-10) )
for i in range(np.size(file_list)-10):
    load_dat = np.transpose(np.loadtxt(file_list[i],dtype='float64',skiprows=330,usecols=(np.arange(0,16)),unpack='True'))
    dep = sw.dpth(load_dat[:,3],26.5)
    for j in range(2,np.size(bin_depth)-1):
        p_in = np.where( (dep >= bin_press[j-1]) & (dep <= bin_press[j+1])  )
        
        this_S = load_dat[p_in,8:10]
        S_bad = np.where(this_S[0] < 32)
        this_S[0][S_bad[0],S_bad[1]] = np.nan
        
        this_T = load_dat[p_in,4:6]
        T_bad = np.where(this_T[0] < 0)
        this_T[0][T_bad[0],T_bad[1]] = np.nan
        
        T1[j,i] = np.nanmean(this_T[0][:,0] )
        S1[j,i] = np.nanmean(this_S[0][:,0] )
        T2[j,i] = np.nanmean(this_T[0][:,1] )
        S2[j,i] = np.nanmean(this_S[0][:,1] )
        lat[j,i] = np.nanmean(load_dat[p_in,10])
        lon[j,i] = np.nanmean(load_dat[p_in,11])
        

plot_plan = 0
if plot_plan > 0:
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)        
    for k in range(np.size(file_list)-20):    
        ax1.scatter(T1[:,k],bin_depth,s=1)
        ax2.scatter(S1[:,k],bin_depth,s=1)        
    ax1.axis([0,25, 0, 5000]) 
    ax1.grid()
    ax2.axis([34,37, 0, 5000])  
    ax2.invert_yaxis()  
    ax2.grid()   
    plt.show()
    # fig.savefig('/Users/jake/Desktop/abaco/abaco_ship_may_2017/T1_profiles.png',dpi = 300)   
    # plt.close()

### compute distance to closest point on transect
dist_grid_s = np.arange(2,800,0.005)
lat_in = 26.5
lon_in = -77  
x_grid = (lon - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))/1000
y_grid = (lat - lat_in)*(1852*60)/1000
sz = np.shape(lon)
number_profiles = sz[1]
dist_dive = np.zeros(np.shape(x_grid))
for i in range(number_profiles):
    for j in range(np.size(bin_press)):
        all_dist = np.sqrt( ( x_grid[j,i] - dist_grid_s )**2 + ( y_grid[j,i] - 0 )**2 )
        if np.isnan(x_grid[j,i]):
            dist_dive[j,i] = float('nan')
        else: 
            closest_dist_dive_i = np.where(all_dist == all_dist.min())
            dist_dive[j,i] = dist_grid_s[closest_dist_dive_i[0]]

### compute theta and potential density
bin_press_grid = np.repeat(  np.transpose(np.array([bin_press])),number_profiles,axis=1 )
bin_depth_grid = np.repeat(  np.transpose(np.array([bin_depth])),number_profiles,axis=1 )
theta_grid = sw.ptmp(S1, T1, bin_press_grid, pr=0)
den_grid = sw.pden(S1, T1, bin_press_grid, pr=1) - 1000
theta_grid_2 = sw.ptmp(S2, T2, bin_press_grid, pr=0)
den_grid_2 = sw.pden(S2, T2, bin_press_grid, pr=1) - 1000
 
######### velocity estimates 
#   depth_units      =  meters 
#   velocity_units   =  cm_per_sec   
#   data_column_1    =  z_depth 
#   data_column_2    =  u_water_velocity_component 
#   data_column_3    =  v_water_velocity_component 
#   data_column_4    =  error_velocity 

adcp_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_may_2017/AB*.vel')
dac_bin_dep = np.float64(np.arange(0,5500,10))
u = np.nan*np.ones( (np.size(dac_bin_dep),np.size(adcp_list)-10) )
v = np.nan*np.ones( (np.size(dac_bin_dep),np.size(adcp_list)-10) )
lat_uv = np.nan*np.ones( np.size(adcp_list)-10 )
lon_uv = np.nan*np.ones( np.size(adcp_list)-10 )
time_uv = np.nan*np.ones( np.size(adcp_list)-10 )
for i in range(np.size(adcp_list)-10):
    test3 = open(adcp_list[i], 'r') 
    line1 = test3.readline()
    line2 = test3.readline()
    line3 = test3.readline()
    line_x = test3.read(1400)
    line4 = test3.readline()
    test3.close()
    load_dat = np.genfromtxt(adcp_list[i],skip_header=75,usecols=(0,1,2,3))
    a,b = unq_searchsorted(load_dat[:,0],dac_bin_dep)
    u[np.where(b)[0],i] = load_dat[:,1]
    v[np.where(b)[0],i] = load_dat[:,2]
    lat_uv[i] = int(line2[4:6]) + np.float64( line2[7:14] )/60
    lon_uv[i] = -1*(int(line3[4:6]) + np.float64( line3[7:14] )/60)
    time_uv[i] = np.float64(line4[5:16])
     
# fig, ax = plt.subplots()
# ax.scatter(lon,lat,s=2,color='r')
# ax.scatter(lon_uv,lat_uv,s=2,color='b')
# ax.grid()
# plt.show()
      
# compute cross-shore distance and plot geostrophic velocity         
cast_lon = np.nanmean(lon,axis=0) 
cast_lat = np.nanmean(lat,axis=0) 
ctd_x = (cast_lon - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))
ctd_y = (cast_lat - lat_in)*(1852*60)
adcp_in = np.where(lat_uv>26.45)
V = v[:,adcp_in[0]]
# closest pos on line to adcp site 
adcp_x = (lon_uv[adcp_in] - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))
adcp_y = (lat_uv[adcp_in] - lat_in)*(1852*60)
for i in range(np.size(adcp_in[0])):
    all_dist = np.sqrt( ( adcp_x[i]/1000 - dist_grid_s )**2 + ( adcp_y[i]/1000 - 0 )**2 )
    if np.isnan(adcp_x[i]):
        adcp_dist[i] = float('nan')
    else: 
        closest_dist_dive_i = np.where(all_dist == all_dist.min())
        adcp_dist[i] = dist_grid_s[closest_dist_dive_i[0]]

fig,ax = plt.subplots()
# ax.scatter(lon_uv[adcp_in],lat_uv[adcp_in])
ax.scatter(adcp_x/1000,adcp_y/1000)
ax.scatter(adcp_dist,np.zeros(np.size(adcp_dist)))
plot_pro(ax)

matlab_datenum = 731965.04835648148
t_s = datetime.date.fromordinal(int( np.min(time_uv) )) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
t_e = datetime.date.fromordinal(int( np.max(time_uv) )) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)


fig, (ax1,ax2) = plt.subplots(1,2) 
ax1.scatter(cast_lon,cast_lat,s=2,color='r')
ax1.scatter(lon_uv[adcp_in],lat_uv[adcp_in],s=2,color='k')
ax1.grid()
lv = np.arange(-60,60,5)
lv2 = np.arange(-60,60,10)
dn_lv = np.concatenate( (np.arange(25,27.8,0.2),np.arange(27.8,28,0.025)))
order = np.argsort(adcp_dist)
ad = ax2.contourf(adcp_dist[order],dac_bin_dep,V[:,order],levels=lv)
ax2.contour(adcp_dist[order],dac_bin_dep,V[:,order],levels=lv2,colors='k')
den_c = ax2.contour(dist_dive,bin_depth_grid, den_grid,levels=dn_lv,colors='r' ,linewidth=.75)
ax2.clabel(den_c, fontsize=6, inline=1,fmt='%.4g',spacing=10) 
ax2.set_title('ABACO SHIP: ' + np.str(t_s.month) + '/' + np.str(t_s.day) + '/' + np.str(t_s.year) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day) + '/' + np.str(t_e.year))
ax2.axis([0, 500, 0, 5000]) 
ax2.invert_yaxis() 
plt.colorbar(ad, label='[cm/s]')
plot_pro(ax2)
# plt.close()

### SAVE 
# write python dict to a file
sa = 1
if sa > 0:
    mydict = {'bin_depth': bin_depth,'adcp_depth': dac_bin_dep, 'adcp_v': V,
        'adcp_lon': lon_uv[adcp_in], 'adcp_lat': lat_uv[adcp_in], 'adcp_dist': adcp_dist,
        'den_grid': den_grid, 'den_grid_2': den_grid_2, 'den_dist': dist_dive, 
        'theta_grid': theta_grid, 'theta_grid_2': theta_grid_2,
        'adcp_lon': lon_uv[adcp_in], 'adcp_lat': lat_uv[adcp_in], 'cast_lon': cast_lon, 'cast_lat': cast_lat }
    output = open('/Users/jake/Desktop/abaco/ship_adcp.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()

      

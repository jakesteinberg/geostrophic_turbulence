# read ABACO shipboard ctd 

import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io as si
import datetime 

plot_plan = 0

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
bin_press = np.concatenate([np.arange(0,150,5), np.arange(150,300,10), np.arange(300,5200,20)])
T = np.nan*np.ones( (np.size(bin_press),np.size(file_list)-10) )
S = np.nan*np.ones( (np.size(bin_press),np.size(file_list)-10) )
lat = np.nan*np.ones( (np.size(bin_press),np.size(file_list)-10) )
lon = np.nan*np.ones( (np.size(bin_press),np.size(file_list)-10) )
for i in range(np.size(file_list)-10):
    load_dat = np.transpose(np.loadtxt(file_list[i],dtype='float64',skiprows=330,usecols=(np.arange(0,16)),unpack='True'))
    for j in range(2,np.size(bin_press)-1):
        p_in = np.where( (load_dat[:,3] >= bin_press[j-1]) & (load_dat[:,3] <= bin_press[j+1])  )
        
        this_S = load_dat[p_in,8:10]
        S_bad = np.where(this_S < 32)
        this_S[S_bad] = np.nan
        
        T[j,i] = np.nanmean( np.nanmean( load_dat[p_in,4:6], axis=0 ) )
        S[j,i] = np.nanmean( np.nanmean( this_S, axis=0 ) )
        lat[j,i] = np.nanmean(load_dat[p_in,10])
        lon[j,i] = np.nanmean(load_dat[p_in,11])
        

if plot_plan > 0:
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)        
    for k in range(np.size(file_list)-20):    
        ax1.scatter(T[:,k],bin_press,s=1)
        ax2.scatter(S[:,k],bin_press,s=1)        
    ax1.axis([0,25, 0, 5000]) 
    ax1.grid()
    ax2.axis([34,37, 0, 5000])  
    ax2.invert_yaxis()  
    ax2.grid()   
    plt.show()
# fig.savefig('/Users/jake/Desktop/abaco/abaco_ship_may_2017/T1_profiles.png',dpi = 300)   
# plt.close()

######### velocity estimates 
def unq_searchsorted(A,B):
    # Get unique elements of A and B and the indices based on the uniqueness
    unqA,idx1 = np.unique(A,return_inverse=True)
    unqB,idx2 = np.unique(B,return_inverse=True)
    # Create mask equivalent to np.in1d(A,B) and np.in1d(B,A) for unique elements
    mask1 = (np.searchsorted(unqB,unqA,'right') - np.searchsorted(unqB,unqA,'left'))==1
    mask2 = (np.searchsorted(unqA,unqB,'right') - np.searchsorted(unqA,unqB,'left'))==1
    # Map back to all non-unique indices to get equivalent of np.in1d(A,B), 
    # np.in1d(B,A) results for non-unique elements
    return mask1[idx1],mask2[idx2]

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
    lon_uv[i] = -1*int(line3[4:6]) + np.float64( line3[7:14] )/60
    time_uv[i] = np.float64(line4[5:16])
     

# fig, ax = plt.subplots()
# ax.scatter(lon,lat,s=2,color='r')
# ax.scatter(lon_uv,lat_uv,s=2,color='b')
# ax.grid()
# plt.show()
      
# compute cross-shore distance and plot geostrophic velocity      
lat_in = 26.5
lon_in = -76.75     
cast_lon = np.nanmean(lon,axis=0) 
cast_lat = np.nanmean(lat,axis=0) 
ctd_x = (cast_lon - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))
ctd_y = (cast_lat - lat_in)*(1852*60)
adcp_x = (lon_uv - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))
adcp_y = (lat_uv - lat_in)*(1852*60)
adcp_dist = np.sqrt(adcp_x**2 + adcp_y**2)

matlab_datenum = 731965.04835648148
t_s = datetime.date.fromordinal(int( np.min(time_uv) )) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
t_e = datetime.date.fromordinal(int( np.max(time_uv) )) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)


fig, ax = plt.subplots()
lv = np.arange(-60,60,5)
order = np.argsort(adcp_dist)
ad = ax.contourf(adcp_dist[order]/1000,dac_bin_dep,v[:,order],levels=lv)
ax.contour(adcp_dist[order]/1000,dac_bin_dep,v[:,order],levels=lv,colors='k')
ax.set_title('ABACO SHIP: ' + np.str(t_s.month) + '/' + np.str(t_s.day) + '/' + np.str(t_s.year) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day) + '/' + np.str(t_e.year))
ax.axis([0, 300, 0, 5000]) 
ax.grid()
ax.invert_yaxis() 
plt.colorbar(ad, label='[m/s]')
plt.show()

      

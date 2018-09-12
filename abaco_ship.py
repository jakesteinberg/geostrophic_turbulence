# read ABACO shipboard ctd 

import numpy as np
from scipy.io import netcdf
import matplotlib.pyplot as plt
import glob
import datetime 
import gsw
import pickle
from toolkit import unq_searchsorted, plot_pro
# import scipy.io as si

# --- ctd
# file_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_may_2017/ctd/ab*.cal')
# file_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_feb_2016/ctd/*.cal')
# file_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_feb_2015/ctd/*.cal')
# file_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_oct_2015/ctd/*.cnv')
# file_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_mar_2014/ctd/*.cal')
# file_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_feb_2013/ctd/*.cal')
# --- ladcp
adcp_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_feb_2018/ladcp/AB*.vel')
# adcp_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_may_2017/ladcp/AB*.vel')
# adcp_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_feb_2016/ladcp/AB*d.vel')
# adcp_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_feb_2015/ladcp/AB*d.vel')
# adcp_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_oct_2015/ladcp/AB*d.vel')
# adcp_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_mar_2014/ladcp/AB*d.vel')
# adcp_list = glob.glob('/Users/jake/Desktop/abaco/abaco_ship_feb_2013/ladcp/AB*d.vel')

# ---- seemingly all except 2017
# name 0 = scan: Scan Count
# name 1 = timeS: Time, Elapsed [seconds]
# name 2 = depSM: Depth [salt water, m]
# name 3 = prDM: Pressure, Digiquartz [db]
# name 4 = t090C: Temperature [ITS-90, deg C]
# name 5 = t190C: Temperature, 2 [ITS-90, deg C]
# name 6 = c0S/m: Conductivity [S/m]
# name 7 = c1S/m: Conductivity, 2 [S/m]
# name 8 = sbeox0V: Oxygen raw, SBE 43 [V]
# name 9 = sbeox1V: Oxygen raw, SBE 43, 2 [V]
# name 10 = latitude: Latitude [deg]
# name 11 = longitude: Longitude [deg]
# name 12 = sbeox0dOV/dT: Oxygen, SBE 43 [dov/dt]
# name 13 = sbeox1dOV/dT: Oxygen, SBE 43, 2 [dov/dt]
# name 14 = sbeox0ML/L: Oxygen, SBE 43 [ml/l]
# name 15 = sbeox1ML/L: Oxygen, SBE 43, 2 [ml/l]
# name 16 = potemp090C: Potential Temperature [ITS-90, deg C]
# name 17 = potemp190C: Potential Temperature, 2 [ITS-90, deg C]
# name 18 = sal00: Salinity, Practical [PSU]
# name 19 = sal11: Salinity, Practical, 2 [PSU]

# -------------------------------------------------------------------------------------------------------

# Data = []
# Lon = np.nan * np.ones(len(file_list))
# Lat = np.nan * np.ones(len(file_list))
# for m in range(len(file_list)):
#     this_file = file_list[m]
#     count_r = 0
#     f = open(this_file, encoding="ISO-8859-1")
#     initial = f.readlines()
#     # extract cast lat/lon
#     line_2 = initial[1].strip().split("\t")
#     item_2 = line_2[0].split()
#     Lon[m] = np.float(item_2[2])
#     Lat[m] = np.float(item_2[1])
#
#     # loops over each row
#     for line in initial:  # loops over each row
#         by_line = line.strip().split("\t")
#         by_item = by_line[0].split()
#         if len(by_item) > 1:
#             item_test0 = by_item[0]
#             item_test1 = by_item[1]
#             item_test2 = by_item[-1]
#             if item_test0[0].isdigit() & item_test1[0].isdigit() & item_test2[0].isdigit() & (len(by_item) > 6):
#                 count = 0  # count for each column value
#                 data = np.nan * np.zeros((1, len(by_item)))  # one row's worth of data
#                 for i in by_item:  # each element in the row
#                     data[0, count] = np.float(i)  # data = one row's worth of data
#                     count = count + 1
#                 if count_r < 1:  # deal with first element of storage vs. all others
#                     data_out = data
#                 data_out = np.concatenate((data_out, data), axis=0)
#                 count_r = count_r + 1
#     Data.append(data_out)

# -------------------------------------------------------------------------------------------------------

# # bin_press = np.arange(0,5200,5)
# # bin_press = np.concatenate([np.arange(0,150,5), np.arange(150,300,10), np.arange(300,5200,20)])
# bin_depth = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 5500, 20)])
# bin_press = gsw.p_from_z(-1 * bin_depth, 26.5*np.ones(len(bin_depth)))
# T1 = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# S1 = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# T2 = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# S2 = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# O1 = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# O2 = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# SA = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# CT = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# sig0 = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# # lat = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# # lon = np.nan*np.ones((np.size(bin_press), np.size(file_list)))
# # --- load in each profile and bin average following bin_press spacing
# for i in range(np.size(file_list)):
#     # load_dat = np.transpose(np.loadtxt(file_list[i], dtype='float64', skiprows=330,
#     #                                    usecols=(np.arange(0, 16)), unpack='True'))
#     load_dat = Data[i]
#     dep = -1 * gsw.z_from_p(load_dat[:, 0], Lat[i] * np.ones(len(load_dat[:, 0])))
#     # dep = -1 * gsw.z_from_p(load_dat[:, 3], load_dat[:, 10])  # all others?
#     # dep = -1 * gsw.z_from_p(load_dat[:, 2], load_dat[:, 9])     # oct 2015
#     for j in range(1, np.size(bin_depth)-1):
#         p_in = np.where((dep >= bin_press[j-1]) & (dep <= bin_press[j+1]))[0]
#
#         this_S = load_dat[p_in, 3]
#         # this_S = load_dat[p_in, 18:20]  # --- all others?
#         # this_S = load_dat[p_in, 8:10]   # --- may 2017
#         # this_S = load_dat[p_in, 7:9]      # --- oct 2015
#         S_bad = np.where(this_S < 32)
#         if np.size(S_bad) > 0:
#             this_S[S_bad[0]] = np.nan
#             # this_S[S_bad[0], S_bad[1]] = np.nan
#
#         this_T = load_dat[p_in, 1]
#         # this_T = load_dat[p_in, 4:6]  # --- all others?
#         # this_T = load_dat[p_in, 3:5]    # --- oct 2015
#         T_bad = np.where(this_T < 0)
#         if np.size(T_bad) > 0:
#             this_T[T_bad[0]] = np.nan
#             # this_T[T_bad[0], T_bad[1]] = np.nan
#
#         this_O = load_dat[p_in, 6]
#         # this_O = load_dat[p_in, 23:25]  # --- all others?
#         # this_O = load_dat[p_in, 14:16]    # --- may 2017 (in units of ml/l)
#         # this_O = load_dat[p_in, 3:5]    # --- oct 2015
#         O_bad = np.where(this_O < 0)
#         if np.size(O_bad) > 0:
#             this_O[O_bad[0]] = np.nan
#             # this_O[O_bad[0], O_bad[1]] = np.nan
#
#         # this_lon = load_dat[p_in, 11]  # --- all others?
#         # # this_lon = load_dat[p_in, 10]    # --- oct 2015
#         # lon_bad = np.where((np.abs(this_lon) < 50) | (np.abs(this_lon) > 82))
#         # if np.size(lon_bad) > 0:
#         #     this_lon[lon_bad[0]] = np.nan
#         #
#         # this_lat = load_dat[p_in, 10]  # --- all others?
#         # # this_lat = load_dat[p_in, 9]     # --- oct 2015
#         # lat_bad = np.where((np.abs(this_lat) < 20) | (np.abs(this_lat) > 30))
#         # if np.size(lat_bad) > 0:
#         #     this_lat[lat_bad[0]] = np.nan
#
#         # two CTDs
#         T1[j, i] = np.nanmean(this_T)
#         S1[j, i] = np.nanmean(this_S)
#         O1[j, i] = np.nanmean(this_O)
#         # T2[j, i] = np.nanmean(this_T[:, 1])
#         # S2[j, i] = np.nanmean(this_S[:, 1])
#         # O2[j, i] = np.nanmean(this_O[:, 1])
#         # lat[j, i] = np.nanmean(this_lat)
#         # lon[j, i] = np.nanmean(this_lon)
#
#     SA[:, i] = gsw.SA_from_SP(S1[:, i], bin_press, Lon[i] * np.ones(len(S1[:, i])), Lat[i] * np.ones(len(S1[:, i])))
#     CT[:, i] = gsw.CT_from_t(S1[:, i], T1[:, i], bin_press)
#     sig0[:, i] = gsw.sigma0(SA[:, i], CT[:, i])

# -------------------------------------------------------------------------------------------------------

plot_plan = 0
if plot_plan > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    for k in range(np.size(file_list)-20):    
        ax1.scatter(CT[:, k], bin_depth, s=1)
        ax2.scatter(SA[:, k], bin_depth, s=1)
    ax1.axis([0, 25, 0, 5500])
    ax1.set_xlabel('Conservative Temperature')
    ax1.set_ylabel('Depth')
    ax1.grid()
    ax2.axis([34, 37, 0, 5500])
    ax2.set_xlabel('Absolute Salinity')
    ax2.set_ylabel('Depth')
    ax2.invert_yaxis()
    plot_pro(ax2)
    # fig.savefig('/Users/jake/Desktop/abaco/abaco_ship_may_2017/T1_profiles.png',dpi = 300)   
    # plt.close()

# -------------------------------------------------------------------------------------------------------
dist_grid_s = np.arange(2, 800, 0.005)
lat_in = 26.5
lon_in = -77
# -------------------------------------------------------------------------------------------------------
# --- compute distance to closest point on transect ---
# x_grid = (Lon - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))/1000
# y_grid = (Lat - lat_in)*(1852*60)/1000
# number_profiles = len(Lon)
# dist_dive = np.zeros(np.shape(x_grid))
# for i in range(number_profiles):
#     all_dist = np.sqrt((x_grid[i] - dist_grid_s)**2 + (y_grid[i] - 0)**2)
#     if np.isnan(x_grid[i]) | np.isnan(y_grid[i]):
#         dist_dive[i] = float('nan')
#     else:
#         closest_dist_dive_i = np.where(all_dist == all_dist.min())
#         dist_dive[i] = dist_grid_s[closest_dist_dive_i[0]]
#
# sig0 = sig0[:, dist_dive > 5]
# Lat_1 = Lat[dist_dive > 5]
# Lon_1 = Lon[dist_dive > 5]
# SA = SA[:, dist_dive > 5]
# CT = CT[:, dist_dive > 5]
# O2 = O1[:, dist_dive > 5]
# dist_dive_2 = dist_dive[dist_dive > 5]
# number_profiles = len(dist_dive_2)
#
# ordered = np.argsort(dist_dive_2)
# sig0 = sig0[:, ordered]
# Lat_2 = Lat[ordered]
# Lon_2 = Lon[ordered]
# SA = SA[:, ordered]
# CT = CT[:, ordered]
# O2 = O1[:, ordered]
# dist_dive_3 = dist_dive_2[ordered]

# -------------------------------------------------------------------------------------------------------
# --- velocity estimates
#   depth_units      =  meters 
#   velocity_units   =  cm_per_sec   
#   data_column_1    =  z_depth 
#   data_column_2    =  u_water_velocity_component 
#   data_column_3    =  v_water_velocity_component 
#   data_column_4    =  error_velocity 

dac_bin_dep = np.float64(np.arange(0, 5500, 10))
u = np.nan*np.ones((np.size(dac_bin_dep), np.size(adcp_list)))
v = np.nan*np.ones((np.size(dac_bin_dep), np.size(adcp_list)))
lat_uv = np.nan*np.ones(np.size(adcp_list))
lon_uv = np.nan*np.ones(np.size(adcp_list))
time_uv = np.nan*np.ones(np.size(adcp_list))
for i in range(np.size(adcp_list)):
    # test3 = open(adcp_list[i], 'r')  # for 2017
    test3 = open(adcp_list[i], 'r', encoding="ISO-8859-1")  # for all others
    line1 = test3.readline()
    line2 = test3.readline()
    line3 = test3.readline()
    line_x = test3.read(1346) # for 2018
    # line_x = test3.read(1400)
    line4 = test3.readline()
    test3.close()
    load_dat = np.genfromtxt(adcp_list[i], skip_header=75, usecols=(0, 1, 2, 3))
    a, b = unq_searchsorted(load_dat[:, 0], dac_bin_dep)
    u[np.where(b)[0], i] = load_dat[:, 1]
    v[np.where(b)[0], i] = load_dat[:, 2]
    lat_uv[i] = int(line2[4:6]) + np.float64(line2[7:14])/60  # 7:14, except 2017 = 8:14
    lon_uv[i] = -1*(int(line3[4:6]) + np.float64(line3[7:14])/60)  # 7:14, except 2017 = 8:14
    # 24:33 for 2013,  11:19 for 2014,  7:16 for 2017,  42:51 for 2018, 8:16 for others
    time_uv[i] = np.float64(line4[42:51])
      
# compute cross-shore distance and plot (non)-geostrophic velocity

adcp_in = np.where((lat_uv > 26.45) & (lon_uv > -77))
V = v[:, adcp_in[0]]
U = u[:, adcp_in[0]]
# closest pos on line to adcp site 
adcp_x = (lon_uv[adcp_in] - lon_in)*(1852*60*np.cos(np.deg2rad(26.5)))
adcp_y = (lat_uv[adcp_in] - lat_in)*(1852*60)
adcp_dist = np.zeros(np.size(adcp_x))
for i in range(np.size(adcp_in[0])):
    all_dist = np.sqrt((adcp_x[i]/1000 - dist_grid_s)**2 + (adcp_y[i]/1000 - 0)**2)
    if np.isnan(adcp_x[i]):
        adcp_dist[i] = float('nan')
    else: 
        closest_dist_dive_i = np.where(all_dist == all_dist.min())
        adcp_dist[i] = dist_grid_s[closest_dist_dive_i[0]]

matlab_datenum = 731965.04835648148
t_s = datetime.date.fromordinal(int(np.min(time_uv))) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days=366)
t_e = datetime.date.fromordinal(int(np.max(time_uv))) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days=366)


# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.scatter(Lon, Lat, s=2, color='r')
# ax1.scatter(lon_uv[adcp_in], lat_uv[adcp_in], s=2, color='k')
# ax1.grid()
# lv = np.arange(-60, 60, 5)
# lv2 = np.arange(-60, 60, 10)
# dn_lv = np.concatenate((np.arange(25, 27.8, 0.2), np.arange(27.8, 28, 0.025)))
# order = np.argsort(adcp_dist)
# ad = ax2.contourf(adcp_dist[order], dac_bin_dep, V[:, order], levels=lv)
# ax2.contour(adcp_dist[order], dac_bin_dep, V[:, order], levels=lv2, colors='k')
# den_c = ax2.contour(np.tile(dist_dive_3[None, :], (len(bin_depth), 1)), np.tile(bin_depth[:, None], (1, number_profiles)), sig0,
#                     levels=dn_lv, colors='r', linewidth=.75)
# ax2.clabel(den_c, fontsize=6, inline=1, fmt='%.4g', spacing=10)
# ax2.set_title('ABACO SHIP: ' + np.str(t_s.month) + '/' +
#               np.str(t_s.day) + '/' + np.str(t_s.year) + ' - ' + np.str(t_e.month) +
#               '/' + np.str(t_e.day) + '/' + np.str(t_e.year))
# ax1.axis([-77.5, -68, 26.4, 26.6])
# ax1.set_xlabel('Lon')
# ax1.set_ylabel('Lat')
# ax2.axis([0, np.nanmax(dist_dive), 0, 5500])
# ax2.set_xlabel('Km offshore')
# ax2.set_ylabel('Depth')
# ax2.invert_yaxis()
# plt.colorbar(ad, label='[cm/s]')
# plot_pro(ax2)

# --- SAVE ---
# write python dict to a file
sa = 1
if sa > 0:
    mydict = {'adcp_depth': dac_bin_dep, 'adcp_u': U, 'adcp_v': V,
              'adcp_lon': lon_uv[adcp_in], 'adcp_lat': lat_uv[adcp_in], 'adcp_dist': adcp_dist,
              'time_uv': time_uv[adcp_in]}
    output = open('/Users/jake/Documents/baroclinic_modes/Shipboard/ABACO/ship_ladcp_' + str(t_s) + '.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()
    # mydict = {'bin_depth': bin_depth, 'adcp_depth': dac_bin_dep, 'adcp_u': U, 'adcp_v': V,
    #           'adcp_lon': lon_uv[adcp_in], 'adcp_lat': lat_uv[adcp_in], 'adcp_dist': adcp_dist,
    #           'den_grid': sig0, 'den_dist': dist_dive_3, 'SA': SA, 'CT': CT, 'cast_lon': Lon_2, 'cast_lat': Lat_2,
    #           'time_uv': time_uv[adcp_in], 'oxygen': O2}
    # output = open('/Users/jake/Documents/baroclinic_modes/Shipboard/ABACO/ship_ladcp_' + str(t_s) + '.pkl', 'wb')
    # pickle.dump(mydict, output)
    # output.close()


# to change
# salinity indices
# encoding specification on adcp files
# time columns for adcp
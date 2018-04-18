# BATS isotropy, spatial variability, temporal changes of T/S
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from netCDF4 import Dataset
import pickle
import seawater as sw
import glob
from toolkit import plot_pro

# physical parameters
g = 9.81
rho0 = 1027
bin_depth = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 5000, 20)])
ref_lat = 31.7
ref_lon = -64.2
grid = bin_depth[1:-1]
grid_p = sw.pres(grid, ref_lat)
z = -1 * grid

# --- LOAD bathymetry
bath = '/Users/jake/Desktop/bats/bats_bathymetry/GEBCO_2014_2D_-67.7_29.8_-59.9_34.8.nc'
bath_fid = Dataset(bath, 'r')  # , mmap=False if sys.platform == 'darwin' else mmap, version=1)
bath_lon = bath_fid.variables['lon'][:]
bath_lat = bath_fid.variables['lat'][:]
bath_z = bath_fid.variables['elevation'][:]
levels_b = [-5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0]

# --- LOAD gridded dives (gridded dives)
GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
df_den = pd.DataFrame(GD['Density'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_theta = pd.DataFrame(GD['Theta'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_s = pd.DataFrame(GD['Absolute Salinity'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lon = pd.DataFrame(GD['Longitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lat = pd.DataFrame(GD['Latitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
time_rec_all = GD.variables['time_start_stop'][:]
dac_u = GD.variables['DAC_u'][:]
dac_v = GD.variables['DAC_v'][:]

df_den[df_den < 0] = np.nan
df_theta[df_theta < 0] = np.nan
df_s[df_s < 0] = np.nan
df_lon[df_lon < -500] = np.nan
df_lat[df_lat < -500] = np.nan
dac_u[dac_u < -100] = np.nan
dac_v[dac_v < -100] = np.nan

time_rec = GD.variables['time_start_stop'][:]
profile_list = np.float64(GD.variables['dive_list'][:]) - 35000

# --- LOAD TRANSECT TO PROFILE DATA COMPILED IN BATS_TRANSECTS.PY
pkl_file = open('/Users/jake/Desktop/bats/dep15_transect_profiles_apr04.pkl', 'rb')
bats_trans = pickle.load(pkl_file)
pkl_file.close()
Time = bats_trans['Time']
Info = bats_trans['Info']
Sigma_Theta = bats_trans['Sigma_Theta']
Eta = bats_trans['Eta']
Eta_theta = bats_trans['Eta_theta']
V = bats_trans['V']
UV_lon = bats_trans['V_lon']
UV_lat = bats_trans['V_lat']
# the ordering that these lists are compiled is by transect no., not by time

# --- LOAD SATELLITE ALTIMETRY
Jtime = []
Jlon = []
Jlat = []
Jtrack = []
Jsla = []
dx = []
l_lat = 28
u_lat = 36
l_lon = 292
u_lon = 298
count = 0
f, ax = plt.subplots()
sat_days = glob.glob('/users/Jake/Documents/baroclinic_modes/Altimetry/altika_along_track_2015/dt*.nc')
for i in range(len(sat_days[0:40])):
    J2 = Dataset(sat_days[i], 'r')
    in_it = np.where((J2.variables['latitude'][:] > l_lat) & (J2.variables['latitude'][:] < u_lat) & (
            J2.variables['longitude'][:] > l_lon) & (J2.variables['longitude'][:] < u_lon))[0]
    # only take satellite paths within defined window
    if in_it.size > 2:
        Jlon_i = J2.variables['longitude'][in_it]
        Jlat_i = J2.variables['latitude'][in_it]
        Jsla_i = J2.variables['sla_unfiltered'][in_it]
        Jtime_i = J2.variables['time'][in_it]
        Jtrack_i = J2.variables['track'][in_it]
        tracks = np.unique(Jtrack_i)
        # separate individual passes
        for j in range(len(tracks)):
            Jlon.append(Jlon_i[Jtrack_i == tracks[j]])
            Jlat.append(Jlat_i[Jtrack_i == tracks[j]])
            Jsla.append(Jsla_i[Jtrack_i == tracks[j]])
            Jtime.append(Jtime_i[Jtrack_i == tracks[j]])
            Jtrack.append(Jtrack_i[Jtrack_i == tracks[j]])

            this_lon = np.deg2rad(Jlon[count])
            this_lat = np.deg2rad(Jlat[count])
            # haversine formula
            dlon = this_lon - this_lon[0]
            dlat = this_lat - this_lat[0]
            a = (np.sin(dlat / 2) ** 2) + np.cos(this_lat[0]) * np.cos(this_lat) * (np.sin(dlon / 2) ** 2)
            # a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            dx.append(6371 * c)  # Radius of earth in kilometers. Use 3956 for miles

            ax.scatter(Jlon[count] - 360, Jlat[count], s=2)
            count = count + 1

ax.scatter(ref_lon, ref_lat)
ax.scatter(df_lon, df_lat, color='k', s=0.1)
ax.set_xlabel('Lon')
ax.set_xlabel('Lat')
ax.set_title('Altika Track and DG BATS15 sampling pattern')
# ax.axis([l_lon, u_lon, l_lat, u_lat])
w = 1 / np.cos(np.deg2rad(ref_lat))
ax.set_aspect(w)
plot_pro(ax)

f, ax = plt.subplots()
for k in range(len(dx)):
    ax.plot(dx[k], Jsla[k], linewidth=0.75)
    ax.scatter(dx[k], Jsla[k], s=1)
ax.set_xlabel('SLA [m]')
ax.set_xlabel('Distance [km]')
ax.set_title('Along-Track SLA near BATS')
plot_pro(ax)

# goal: look at spatial variability of T/S fields on varied time scales at various depths 
# goal: look at temporal variability of DAC field 

# ---  TIME WINDOWS
# lon_per_prof = np.nanmean(df_lon, axis=0)
# lat_per_prof = np.nanmean(df_lat, axis=0)
# t_win = (time_rec[-1] - time_rec[
#     0]) / 12  # is just shy of 20 days ... this is the amount of time it took to execute one butterfly pattern
# t_windows = np.arange(time_rec[0], time_rec[-1], 20)
# avg_u = np.zeros(np.size(t_windows) - 1)
# avg_v = np.zeros(np.size(t_windows) - 1)
# for i in range(np.size(t_windows) - 1):
#     avg_u[i] = np.nanmean(dac_u[np.where((time_rec > t_windows[i]) & (time_rec < t_windows[i + 1]))[0]])
#     avg_v[i] = np.nanmean(dac_v[np.where((time_rec > t_windows[i]) & (time_rec < t_windows[i + 1]))[0]])
# t_s = datetime.date.fromordinal(np.int(np.min(time_rec[0])))
# t_e = datetime.date.fromordinal(np.int(np.max(time_rec[-1])))


Time_dt = []
for i in range(len(time_rec_all)):
    Time_dt.append(datetime.date.fromordinal(np.int(time_rec_all[i])))

u_pos = np.where(dac_u > 0)[0]
v_pos = np.where(dac_v > 0)[0]
days_pos_u = 0
days_pos_v = 0
for i in range(1, len(u_pos)):
    if (u_pos[i] - u_pos[i-1]) < 1.1:
        days_pos_u = days_pos_u + (time_rec_all[u_pos[i]] - time_rec_all[u_pos[i-1]])
for i in range(1, len(v_pos)):
    if (v_pos[i] - v_pos[i-1]) < 1.1:
        days_pos_v = days_pos_v + (time_rec_all[v_pos[i]] - time_rec_all[v_pos[i-1]])

# ---- DAC AND ITS AVERAGE ----
fig0, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
ax0.plot(Time_dt, np.zeros(len(time_rec_all)), color='k')
ax0.plot(Time_dt[0:-1:2], dac_u[0:-1:2])
ax0.set_title('DAC$_u$')
ax1.plot(Time_dt, np.zeros(len(time_rec_all)), color='k')
ax1.plot(Time_dt[0:-1:2], dac_v[0:-1:2])
ax1.set_title('DAC$_v$')
ax0.set_ylim([-.2, .2])
ax1.set_ylim([-.2, .2])
ax0.grid()
plot_pro(ax1)

# ---- TEMPERATURE on depth surfaces with time ----
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

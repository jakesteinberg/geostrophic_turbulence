import numpy as np
import scipy.fftpack as sci_fft
import matplotlib
import matplotlib.pyplot as plt
import datetime
import glob
from netCDF4 import Dataset
from toolkit import plot_pro

GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
dg_lon = GD['Longitude'][:]
dg_lat = GD['Latitude'][:]
dg_lon[dg_lon < -500] = np.nan
dg_lat[dg_lat < -500] = np.nan

l_lon = -75
u_lon = -55
l_lat = 25
u_lat = 45

Time = []
Lon = []
Lat = []
Track = []
SLA = []
dx = []
main_path = '/Users/jake/Documents/baroclinic_modes/Altimetry/nrt_j3_july_2018/*.nc'
files = glob.glob(main_path)
count = 0

for i in range(len(files)):
    data = Dataset(files[i], 'r')
    lon_0 = data['longitude'][:]
    lon_0[lon_0 > 180] = lon_0[lon_0 > 180] - 360
    in_it = np.where((data['latitude'][:] > l_lat) & (data['latitude'][:] < u_lat) & (
            lon_0 > l_lon) & (lon_0 < u_lon))[0]
    if i < 1:
        lon_i = lon_0[in_it]
        lat_i = data['latitude'][in_it]
        time_i = data['time'][in_it]
        adt_i = data['adt_filtered'][in_it]
        sla_i = data['sla_filtered'][in_it]
        tracks_i = data['track'][in_it]
    else:
        lon_i = np.concatenate((lon_i, lon_0[in_it]))
        lat_i = np.concatenate((lat_i, data['latitude'][in_it]))
        time_i = np.concatenate((time_i, data['time'][in_it]))
        adt_i = np.concatenate((adt_i, data['adt_filtered'][in_it]))
        sla_i = np.concatenate((sla_i, data['sla_filtered'][in_it]))
        tracks_i = np.concatenate((tracks_i, data['track'][in_it]))

    # tracks = np.unique(data['track'][in_it])
    # print(tracks)

    lon_j = data['longitude'][in_it]
    lat_j = data['latitude'][in_it]
    sla_j = data['sla_filtered'][in_it]
    time_j = data['time'][in_it]
    track_step = data['track'][in_it]
    tracks = []
    if len(track_step) > 0:
        tracks.append(track_step[0])
        for j in range(1,len(track_step)):
            this_track = track_step[j]
            if this_track - track_step[j - 1] > 1:
                tracks.append(this_track)

    # separate individual passes
    for j in range(len(tracks)):
        Lon.append(lon_j[track_step == tracks[j]])
        Lat.append(lat_j[track_step == tracks[j]])
        SLA.append(sla_j[track_step == tracks[j]])
        Time.append(time_j[track_step == tracks[j]])
        Track.append(track_step[track_step == tracks[j]])

        this_lon = np.deg2rad(Lon[count])
        this_lat = np.deg2rad(Lat[count])
        # haversine formula
        dlon = this_lon - this_lon[0]
        dlat = this_lat - this_lat[0]
        a = (np.sin(dlat / 2) ** 2) + np.cos(this_lat[0]) * np.cos(this_lat) * (np.sin(dlon / 2) ** 2)
        # a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        dx.append(6371 * c)  # Radius of earth in kilometers. Use 3956 for miles

        count = count + 1

# SG041 targets
dg41_lon = np.array([-64.1666, -64.1666, -65, -65, -65, -65, -65, -65.4866, -65, -64.5034,
                    -65, -65, -65, -65.4866, -65, -64.5034])
dg41_lat = np.array([31.6666, 33, 33, 35.587, 35.587, 36, 36.403, 36, 36, 36,
                    35.587, 36, 36.403, 36, 36, 36])

time_adj = time_i + 711858
t_min = time_adj[0]
t_max = time_adj[0] + 15
t_s = datetime.date.fromordinal(np.int(time_adj[0]))
t_e = datetime.date.fromordinal(np.int(time_adj[-1]))

# JASON-3 Sept 2018
# ind = np.where((time_adj > time_adj[0]) & (time_adj < (time_adj[0] + 1)))[0]
# time1_s = datetime.date.fromordinal(np.int(time_adj[ind[0]]))
# time1_e = datetime.date.fromordinal(np.int(time_adj[ind[-1]]))
# ind2 = np.where((time_adj > (time_adj[0] + 9.5)) & (time_adj < (time_adj[0] + 11)))[0]
# time2_s = datetime.date.fromordinal(np.int(time_adj[ind2[0]]))
# time2_e = datetime.date.fromordinal(np.int(time_adj[ind2[-1]]))

# ind = np.where((time_adj > time_adj[0] + 3) & (time_adj < (time_adj[0] + 5)))[0]
# time1_s = datetime.date.fromordinal(np.int(time_adj[ind[0]]))
# time1_e = datetime.date.fromordinal(np.int(time_adj[ind[-1]]))
# ind2 = np.where((time_adj > (time_adj[0] + 13)) & (time_adj < (time_adj[0] + 14)))[0]
# time2_s = datetime.date.fromordinal(np.int(time_adj[ind2[0]]))
# time2_e = datetime.date.fromordinal(np.int(time_adj[ind2[-1]]))

# ind = np.where((time_adj > time_adj[0] + 6) & (time_adj < (time_adj[0] + 9)))[0]
# time1_s = datetime.date.fromordinal(np.int(time_adj[ind[0]]))
# time1_e = datetime.date.fromordinal(np.int(time_adj[ind[-1]]))
# ind2 = np.where((time_adj > (time_adj[0] + 16)) & (time_adj < (time_adj[0] + 18)))[0]
# time2_s = datetime.date.fromordinal(np.int(time_adj[ind2[0]]))
# time2_e = datetime.date.fromordinal(np.int(time_adj[ind2[-1]]))

ind = np.where((time_adj > time_adj[0]) & (time_adj < (time_adj[0] + 16)))[0]
time1_s = datetime.date.fromordinal(np.int(time_adj[ind[0]]))
time1_e = datetime.date.fromordinal(np.int(time_adj[ind[-1]]))
ind2 = np.where((time_adj > (time_adj[0] + 16)) & (time_adj < (time_adj[0] + 20)))[0]
time2_s = datetime.date.fromordinal(np.int(time_adj[ind2[0]]))
time2_e = datetime.date.fromordinal(np.int(time_adj[ind2[-1]]))

plott = 0
if plott > 0:
    cmap = matplotlib.cm.get_cmap('plasma')
    # cmap((np.nanmean(time_adj[ind]) - t_min)/(t_max - t_min))

    count = 0
    f, (ax0, ax1) = plt.subplots(1, 2)
    for i in range(len(files)):
        ax0.scatter(Lon[count], Lat[count], s=2)
        count = count + 1
    ax0.set_xlabel('Lon')
    ax0.set_ylabel('Lat')
    ax0.set_title('Altika Track and DG BATS15 sampling pattern')
    # ax.axis([l_lon, u_lon, l_lat, u_lat])
    w = 1 / np.cos(np.deg2rad(34))
    ax0.set_aspect(w)

    for k in range(len(dx)):
        ax1.plot(dx[k], SLA[k], linewidth=0.75)
        ax1.scatter(dx[k], SLA[k], s=1)
    ax1.set_ylabel('SLA [m]')
    ax1.set_xlabel('Distance [km]')
    ax1.set_title('Along-Track SLA near BATS')
    plot_pro(ax1)

    f, ax =plt.subplots()
    # ax.scatter(lon, lat, s=10, color='r')
    ax.scatter(lon_i[ind], lat_i[ind], s=40, color='r', label=str(time1_s) + ' - ' + str(time1_e))
    ax.scatter(lon_i[ind2], lat_i[ind2], s=10, color='g', label=str(time2_s) + ' - ' + str(time2_e))
    # ax.scatter(lon[ind3], lat[ind3], s=5, color='b')
    # ax.scatter(lon[ind4], lat[ind4], s=5, color='y')
    # ax.scatter(dg_lon, dg_lat, s=0.2)
    ax.plot(dg41_lon, dg41_lat, color='b', label='SG041')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Altika tracks Sept 2018')
    ax.axis([-70, -60, 31, 37])
    w = 1 / np.cos(np.deg2rad(34))
    ax.set_aspect(w)
    plot_pro(ax)


# ---------------------------------------------------------------------------------
# --- SPECTRAL POWER
power = []
index = []
yhat_out = []
M_out = []
L_out = []
for m in range(len(dx)):
    this_x = dx[m]
    this_y = SLA[m] - np.nanmean(SLA[m])
    # put onto regular grid
    x_grid = np.arange(0, np.nanmax(this_x), 5)
    y_grid_0 = np.interp(x_grid, this_x, this_y)
    # remove linear trend (and square the result to get variance [m^2])
    fit_coef = np.polyfit(x_grid, y_grid_0, 1)
    fit_y = fit_coef[0] * x_grid + fit_coef[1]
    y_grid = y_grid_0 - fit_y

    # number of km in dx
    L = np.nanmax(this_x)

    yhat = sci_fft.fft(y_grid)
    N = len(y_grid)  # number of samples
    if np.mod(N, 2):
        M = np.concatenate((np.arange(0, N/2), np.arange(-N/2 + 1, 0)))
    else:
        M = np.concatenate((np.arange(0, N / 2), np.arange(-N / 2, 0)))
    power_0 = np.abs(yhat/N)**2
    # selecting the first number to plot
    index = (M <= 20) & (M > 0)
    power.append(power_0[index])
    yhat_out.append(yhat)
    M_out.append(M[index])
    L_out.append(L)

fac = 2 * np.pi
k_grid = np.arange(7*10**-3, 10**-1, 10**-3)

fig = plt.figure()
ax = fig.add_subplot(111)
power_grid = np.nan*np.ones((len(dx), len(k_grid)))
for o in range(len(dx)):
    ax.plot(fac * M_out[o] / L_out[o], power[o], color='#D3D3D3')
    power_grid[o, :] = np.interp(k_grid, fac * M_out[o] / L_out[o], power[o])
ax.plot(k_grid, np.nanmean(power_grid, axis=0), 'k', linewidth=2)
ax.set_xlim([10**-3, 10**0])
ax.set_ylim([10**-7, 10**-1])
ax.set_yscale('log')
ax.set_xscale('log')

ax2 = ax.twiny()
ax2.set_xlim(10**-3, 10**0)
ax2.set_xscale('log')
ax2.set_xticks(np.array([5*10**-3, 10**-2, 10**-1]))
ax2.set_xticklabels([str(1 / (5*10**-3)) + 'km', str(1 / (10**-2)) + 'km', str(1 / (10**-1)) + 'km'], fontsize=9)
plot_pro(ax2)

# index2 = ((M / L) > 0.004) | ((M / L) < -0.004)
# yshat = yhat.copy()
# yshat[index2] = 0
# ys = np.real(sci_fft.ifft(yshat))
# f, ax = plt.subplots()
# ax.plot(this_x, this_y + np.nanmean(SLA[m]), 'r')
# ax.plot(x_grid, (ys + fit_y) + np.nanmean(SLA[m]), 'k', linewidth=0.5, linestyle='--')
# plot_pro(ax)
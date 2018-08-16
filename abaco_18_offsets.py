from glider_cross_section import Glider
import numpy as np
import datetime
import gsw
import matplotlib
import matplotlib.pyplot as plt
from toolkit import plot_pro, cart2pol, pol2cart, nanseg_interp

x7 = Glider(37, np.arange(8, 62), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2018/sg037')
x9 = Glider(39, np.arange(8, 58), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2018/sg039')

bin_depth = np.concatenate((np.arange(0, 300, 5), np.arange(300, 1000, 10), np.arange(1000, 4710, 20)))  # shallower

ref_lat = 26.5
d_time_7, lon_7, lat_7, t_7, s_7, dac_u_7, dac_v_7, profile_tags_7 = x7.make_bin(bin_depth)
sa_7, ct_7, sig0_7, N2_7 = x7.density(bin_depth, ref_lat, t_7, s_7, lon_7, lat_7)

d_time_9, lon_9, lat_9, t_9, s_9, dac_u_9, dac_v_9, profile_tags_9 = x9.make_bin(bin_depth)
sa_9, ct_9, sig0_9, N2_9 = x9.density(bin_depth, ref_lat, t_9, s_9, lon_9, lat_9)

dist = np.nan*np.ones((len(profile_tags_7), len(profile_tags_9)))
den_anom = np.nan*np.ones((len(profile_tags_7), len(profile_tags_9)))
t_anom = np.nan*np.ones((len(profile_tags_7), len(profile_tags_9)))
s_anom = np.nan*np.ones((len(profile_tags_7), len(profile_tags_9)))
time_ij = np.nan*np.ones((len(profile_tags_7), len(profile_tags_9)))
dep_ind = 280
for i in range(len(profile_tags_7)):
    for j in range(len(profile_tags_9)):
        dx = 1.852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_7[dep_ind, i] - lon_9[dep_ind, j])
        dy = 1.852 * 60 * (lat_7[dep_ind, i] - lat_9[dep_ind, j])
        dist[i, j] = np.sqrt(dx**2 + dy**2)
        den_anom[i, j] = sig0_7[dep_ind, i] - sig0_9[dep_ind, j]
        t_anom[i, j] = ct_7[dep_ind, i] - ct_9[dep_ind, j]
        s_anom[i, j] = sa_7[dep_ind, i] - sa_9[dep_ind, j]
        time_ij[i, j] = np.nanmean([d_time_7[dep_ind, i], d_time_9[dep_ind, j]])

t_s = datetime.date.fromordinal(np.int(np.nanmin(time_ij)))
t_e = datetime.date.fromordinal(np.int(np.nanmax(time_ij)))

cmap = matplotlib.cm.get_cmap('Spectral')
t_min = np.nanmin(time_ij)
t_max = np.nanmax(time_ij)
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
for i in range(len(profile_tags_7)):
    for j in range(len(profile_tags_9)):
        da = ax1.scatter(dist[i, j], den_anom[i, j], s=4.5, color=cmap((time_ij[i, j] - t_min)/(t_max - t_min)))
        ax2.scatter(dist[i, j], t_anom[i, j], s=4.5, color=cmap((time_ij[i, j] - t_min)/(t_max - t_min)))
        ax3.scatter(dist[i, j], s_anom[i, j], s=4.5, color=cmap((time_ij[i, j] - t_min)/(t_max - t_min)))

c_map_ax = f.add_axes([0.933, 0.1, 0.02, 0.8])
norm = matplotlib.colors.Normalize(vmin=0, vmax=t_max - t_min)
cb1 = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=cmap, norm=norm, orientation='vertical')
cb1.set_label('Days Since Start')

ax1.set_ylim([-0.02, 0.05])
ax2.set_ylim([-0.15, 0.1])
ax3.set_ylim([-0.03, 0.05])
ax3.set_xlabel('Distance apart [km]')
ax1.set_ylabel('Potential Density Anom.')
ax2.set_ylabel('Conservative Temp. Anom.')
ax3.set_ylabel('Absolute Sal. Anom.')
ax1.set_title('Density Anomaly at 4000m')
ax2.set_title('Temperature Anomaly at 4000m')
ax3.set_title('Salinity Anomaly at 4000m')
ax1.grid()
ax2.grid()
plot_pro(ax3)

sa_gr = np.arange(32, 38, 0.1)
ct_gr = np.arange(1, 30, 0.5)
X, Y = np.meshgrid(sa_gr, ct_gr)
sig0_gr = gsw.sigma0(X, Y)
levs = np.arange(np.round(np.min(sig0_gr), 1), np.max(sig0_gr), 0.2)

f, ax = plt.subplots()
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
dc = ax.contour(sa_gr, ct_gr, sig0_gr, levs, colors='k', linewidth=0.25)
ax.clabel(dc, inline=1, fontsize=7, fmt='%1.2f', color='k')
for i in range(len(profile_tags_7)):
    ax.scatter(sa_7[:, i], ct_7[:, i], s=2, color=cmap((np.nanmean(d_time_7[:, i]) - t_min)/(t_max - t_min)),
               marker='v')
    # ax1.plot(sig0_7[:, i], bin_depth, color=cmap((np.nanmean(d_time_7[:, i]) - t_min)/(t_max - t_min)))
    # ax2.plot(ct_7[:, i], bin_depth, color=cmap((np.nanmean(d_time_7[:, i]) - t_min)/(t_max - t_min)))
    # ax3.plot(sa_7[:, i], bin_depth, color=cmap((np.nanmean(d_time_7[:, i]) - t_min)/(t_max - t_min)))
for i in range(len(profile_tags_9)):
    ax.scatter(sa_9[:, i], ct_9[:, i], s=6, color=cmap((np.nanmean(d_time_9[:, i]) - t_min)/(t_max - t_min)),
               marker='x')
    # ax1.plot(sig0_9[:, i], bin_depth, color=cmap((np.nanmean(d_time_9[:, i]) - t_min)/(t_max - t_min)))
    # ax2.plot(ct_9[:, i], bin_depth, color=cmap((np.nanmean(d_time_9[:, i]) - t_min)/(t_max - t_min)), linestyle='--')
    # ax3.plot(sa_9[:, i], bin_depth, color=cmap((np.nanmean(d_time_9[:, i]) - t_min)/(t_max - t_min)), linestyle='--')
# ax1.invert_yaxis()
# ax1.grid()
# ax2.grid()
# ax.axis([np.round(np.nanmin(sa_9), 1) - .2, np.round(np.nanmax(sa_9), 1) + .2,
#          np.round(np.nanmin(ct_9), 1) - .5, np.round(np.nanmax(ct_9), 1) + .5])
ax.axis([35, 35.2, 1.6, 3.5])
plot_pro(ax)
import numpy as np
import matplotlib.pyplot as plt
import datetime
import gsw
import pandas as pd
from netCDF4 import Dataset
import pickle
from mpl_toolkits.mplot3d import axes3d
from toolkit import plot_pro

# LOAD DATA (gridded dives)
GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
profile_list = GD['dive_list'][:] - 35000
df_den = pd.DataFrame(GD['Density'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lon = pd.DataFrame(GD['Longitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lat = pd.DataFrame(GD['Latitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lon[df_lon < -500] = np.nan
df_lat[df_lat < -500] = np.nan

# physical parameters
g = 9.81
rho0 = 1027
bin_depth = GD.variables['grid'][:]
ref_lon = np.nanmean(df_lon)
ref_lat = np.nanmean(df_lat)
grid = bin_depth
grid_p = gsw.p_from_z(-1 * grid, ref_lat)
z = -1 * grid
sz_g = grid.shape[0]

# select only dives with depths greater than 4000m 
grid_test = np.nan * np.zeros(len(profile_list))
for i in range(len(profile_list)):
    grid_test[i] = grid[np.where(np.array(df_den.iloc[:, i]) == np.nanmax(np.array(df_den.iloc[:, i])))[0][0]]
good = np.where(grid_test >= 4000)[0]

# --- LOAD gridded dives (gridded dives)
df_den = pd.DataFrame(GD['Density'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_theta = pd.DataFrame(GD['Theta'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_s = pd.DataFrame(GD['Absolute Salinity'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lon = pd.DataFrame(GD['Longitude'][:, good], index=GD['grid'][:], columns=GD['dive_list'][good])
df_lat = pd.DataFrame(GD['Latitude'][:, good], index=GD['grid'][:], columns=GD['dive_list'][good])
df_lon_all = pd.DataFrame(GD['Longitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lat_all = pd.DataFrame(GD['Latitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
dac_u = GD.variables['DAC_u'][:]
dac_v = GD.variables['DAC_v'][:]
time_rec = GD.variables['time_start_stop'][good]
time_rec_all = GD.variables['time_start_stop'][:]
time_rec_all_dt = []
for i in range(len(time_rec_all)):
    time_rec_all_dt.append(datetime.date.fromordinal(np.int(time_rec_all[i])))
profile_list = np.float64(GD.variables['dive_list'][good]) - 35000
profile_list_all = np.float64(GD.variables['dive_list'][:]) - 35000

df_den[df_den < 0] = np.nan
df_theta[df_theta < 0] = np.nan
df_s[df_s < 0] = np.nan
df_lon_all[df_lon_all < -500] = np.nan
df_lat_all[df_lat_all < -500] = np.nan
dac_u[dac_u < -500] = np.nan
dac_v[dac_v < -500] = np.nan

# -------------- LOAD IN MAPPED U/V, AND DENSITY
pkl_file = open('/Users/jake/Documents/geostrophic_turbulence/BATS_obj_map_L35_W4_apr19.pkl', 'rb')  # _2
bats_map = pickle.load(pkl_file)
pkl_file.close()
Time = bats_map['time']
sigma_theta = np.array(bats_map['Sigma_Theta'])
U = np.array(bats_map['U_g'])
V = np.array(bats_map['V_g'])
sigma_theta_all = np.array(bats_map['Sigma_Theta_All'])
d_dx_sig = np.array(bats_map['d_sigma_dx'])
d_dy_sig = np.array(bats_map['d_sigma_dy'])
dac_u_map = np.array(bats_map['dac_u_map'])
dac_v_map = np.array(bats_map['dac_v_map'])
U_all = np.array(bats_map['U_g_All'])
V_all = np.array(bats_map['V_g_All'])
lat_grid_good = np.array(bats_map['lat_grid'])
lon_grid_good = np.array(bats_map['lon_grid'])
lat_grid_all = np.array(bats_map['lat_grid_All'])
lon_grid_all = np.array(bats_map['lon_grid_All'])
mask = bats_map['mask']
Lon_map, Lat_map = np.meshgrid(lon_grid_all[0, :], lat_grid_all[0, :])
# todo compute potential vorticity (need to check where Q goes to zero or becomes negative)

# ---- LOAD IN TRANSECT TO PROFILE DATA COMPILED IN BATS_TRANSECTS.PY
pkl_file = open('/Users/jake/Desktop/bats/dep15_transect_profiles_apr04.pkl', 'rb')
bats_trans = pickle.load(pkl_file)
pkl_file.close()
Time_t = bats_trans['Time']
Info = bats_trans['Info']
Sigma_Theta = bats_trans['Sigma_Theta'][0:sz_g, :]
Eta = bats_trans['Eta'][0:sz_g, :]
Eta_theta = bats_trans['Eta_theta'][0:sz_g, :]
V_t = bats_trans['V'][0:sz_g, :]
prof_lon = bats_trans['V_lon']
prof_lat = bats_trans['V_lat']

# ---- LOAD GRIDDED ALTIMETRY AND SURFACE VELOCITIES
SA = Dataset('/Users/jake/Desktop/bats/dataset-duacs-rep-global-merged-allsat-phy-l4-v3_1524083289217.nc', 'r')
lol = -65.2
lom = -63.3
lal = 31.1
lam = 32.4
sa_lon = SA.variables['longitude'][:] - 360
sa_lon_2 = sa_lon[(sa_lon >= lol) & (sa_lon <= lom)]
sa_lat = SA.variables['latitude'][:]
sa_lat_2 = sa_lat[(sa_lat >= lal) & (sa_lat <= lam)]
SA_lon, SA_lat = np.meshgrid(sa_lon_2, sa_lat_2)
# SA time
sa_time = SA.variables['time'][:] + 711858  # 1950-01-01      datetime.date.fromordinal(np.int(Time[i]))
sa_time_dt = []
for i in range(len(sa_time)):
    sa_time_dt.append(datetime.date.fromordinal(np.int(sa_time[i])))
# SA fields
sa_u = SA.variables['ugos'][:, (sa_lat >= lal) & (sa_lat <= lam), (sa_lon >= lol) & (sa_lon <= lom)]
sa_v = SA.variables['vgos'][:, (sa_lat >= lal) & (sa_lat <= lam), (sa_lon >= lol) & (sa_lon <= lom)]
sa_adt = SA.variables['adt'][:, (sa_lat >= lal) & (sa_lat <= lam), (sa_lon >= lol) & (sa_lon <= lom)]
sa_mean_u = np.nanmean(np.nanmean(sa_u, axis=1), axis=1)
sa_mean_v = np.nanmean(np.nanmean(sa_v, axis=1), axis=1)

# ------------- PLOTTING POSSIBILITIES
# ----- U, V (good) in time
U_avg = np.nan * np.zeros((len(grid), len(Time)))
V_avg = np.nan * np.zeros((len(grid), len(Time)))
T_mid = []
for i in range(len(Time)):
    U_avg[:, i] = np.transpose(np.nanmean(U[i], axis=0))
    V_avg[:, i] = np.transpose(np.nanmean(V[i], axis=0))
    T_mid.append(datetime.date.fromordinal(np.int(np.nanmean(Time[i]))))

# --- eddy
ed_in = np.where((profile_list_all >= 62) & (profile_list_all <= 64.5))[0]
ed_time = 735712

# --- ALTIMETRY and mapping consistency
this_time = 735725
sa_t_choice = (np.abs(sa_time - this_time)).argmin()
map_t_choice = (np.abs(np.mean(Time, axis=1) - this_time)).argmin()
dive_t_low = np.abs(time_rec_all - (this_time - 7)).argmin()
dive_t_up = np.abs(time_rec_all - (this_time + 7)).argmin()

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.pcolor(SA_lon, SA_lat, sa_adt[sa_t_choice, :, :])
ax1.quiver(SA_lon, SA_lat, sa_u[sa_t_choice, :, :], sa_v[sa_t_choice, :, :], scale_units='xy')
ax1.quiver(sa_lon_2[0] + .1, sa_lat_2[-1] - .1, 0.1, 0, scale_units='xy', color='m')
ax1.text(sa_lon_2[0] + .1, sa_lat_2[-1] - .13, '0.1 m/s', color='m', fontsize=9)
ax1.grid()
ax1.set_title('Altimetry ' + str(sa_time_dt[sa_t_choice]))
ax1.axis([lol, lom, lal, lam])
ax2.quiver(Lon_map, Lat_map, U_all[map_t_choice, :, :, 0], V_all[map_t_choice, :, :, 0], scale_units='xy', label='map')
ax2.quiver(df_lon_all.iloc[:, dive_t_low:dive_t_up].mean(), df_lat_all.iloc[:, dive_t_low:dive_t_up].mean(),
           dac_u[dive_t_low:dive_t_up], dac_v[dive_t_low:dive_t_up], color='r', scale_units='xy', label='DAC')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
if np.abs(this_time - ed_time) < 5:
    ax2.scatter(df_lon_all.iloc[0, ed_in[0]], df_lat_all.iloc[0, ed_in[0]], s=30, color='c')
    ax2.scatter(df_lon_all.iloc[:, ed_in], df_lat_all.iloc[:, ed_in], s=0.5, color='b')
ax2.quiver(sa_lon_2[0] + .1, sa_lat_2[-1] - .1, 0.1, 0, scale_units='xy', color='m')
ax2.text(sa_lon_2[0] + .1, sa_lat_2[-1] - .13, '0.1 m/s', color='m', fontsize=9)
ax2.set_title('DG Mapping ' + str(time_rec_all_dt[dive_t_low]) + ' - ' + str(time_rec_all_dt[dive_t_up]))
ax2.axis([lol, lom, lal, lam])
ax2.set_xlabel('Longitude')
ax2.text(-65, 32.1, 'Surface Velocity')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=10)
w = 1 / np.cos(np.deg2rad(ref_lat))
ax1.set_aspect(w)
ax2.set_aspect(w)
plot_pro(ax2)

f, ax = plt.subplots(6, 1, sharex=True)
indi = [0, 24, 64, 114, 164, 209]
for i in range(6):
    if i < 1:
        # ax[i].plot(time_rec_all[0::2], dac_u[0::2], color='r', linestyle=':')
        # ax[i].plot(time_rec_all[0::2], dac_v[0::2], color='k', linestyle=':')
        ax[i].plot(sa_time, sa_mean_u, color='r', linestyle='--')
        ax[i].plot(sa_time, sa_mean_v, color='k', linestyle='--')
        ax[i].plot([sa_time_dt[sa_t_choice], sa_time_dt[sa_t_choice]], [-.3, .3], color='b', linestyle='--')
    ax[i].plot(T_mid, U_avg[indi[i], :], color='r')
    ax[i].plot(T_mid, V_avg[indi[i], :], color='k')
    ax[i].set_ylim([-.3, .3])
    ax[i].set_title('Mapped Average U/V at ' + str(grid[indi[i]]) + 'm', fontsize=10)
    ax[i].plot([time_rec_all[ed_in[0]], time_rec_all[ed_in[0]]], [-.3, .3], color='k', linestyle='--')
    ax[i].plot([time_rec_all[ed_in[-1]], time_rec_all[ed_in[-1]]], [-.3, .3], color='k', linestyle='--')
    ax[i].set_ylabel('m/s')
    ax[i].grid()
ax[i].grid()
ax[i].set_xlim([T_mid[0], T_mid[-1]])
ax[i].set_xlabel('Date')
plot_pro(ax[i])

# ----- DENSITY CROSS-SECTIONS
# fixed_lat = 7
# fixed_lon = 10
# levels_rho = np.concatenate((np.array([26,26.2,26.4,26.8,27,27.2,27.6]), np.arange(27.68,28.2,0.02)))
# levels_v = np.arange(-.26,.26,.02)
# fig,(ax,ax1) = plt.subplots(1,2,sharey=True)
# ax.pcolor(x_grid,grid,sigma_theta_all[:,:,fixed_lat,iter],vmin=25.5,vmax=28)
# ax.contour(x_grid,grid,sigma_theta_all[:,:,fixed_lat,iter],levels=levels_rho,colors='k')
# vcc = ax.contour(x_grid,grid,V_all[:,:,fixed_lat,iter],levels=levels_v,colors='r')
# ax.clabel(vcc,inline=1,fontsize=8,fmt='%1.2f',color='r')
# ax.set_ylabel('Depth [m]')
# ax.set_xlabel('Zonal Distance (looking N)')
# ax.set_title('Zonal Cross Section (density,velocity)')

# ax1.pcolor(y_grid,grid,sigma_theta_all[:,fixed_lon,:,iter],vmin=25.5,vmax=28)
# ax1.contour(y_grid,grid,sigma_theta_all[:,fixed_lon,:,iter],levels=levels_rho,colors='k')
# vcc = ax1.contour(y_grid,grid,U_all[:,fixed_lon,:,iter],levels=levels_v,colors='r')
# ax1.clabel(vcc,inline=1,fontsize=8,fmt='%1.2f',color='r')
# ax1.set_xlabel('Meridional Distance (Looking E)')
# ax1.set_title('Meridional Cross Section (density,velocity)')
# ax1.invert_yaxis() 
# plot_pro(ax1)
# ax1.grid()
# fig.savefig( ('/Users/jake/Desktop/BATS/bats_mapping/cross_map_' + str(k_out) + '.png'),dpi = 300)
# plt.close()


# ------------ PLAN VIEW ------------------------------
test_lev = 65
lim = 50
time_t = 0  # 15  # map_t_choice  # 10
x_grid = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_grid_all[time_t, :] - ref_lon)
y_grid = 1852 * 60 * (lat_grid_all[time_t, :] - ref_lat)
x1, y1 = np.meshgrid(x_grid / 1000, y_grid / 1000)

f, ax = plt.subplots(3, 3)
axl1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
axl2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
cmp = plt.cm.get_cmap("viridis")
cmp.set_over('w')  # ('#E6E6E6')
cmp.set_under('w')
tt = np.arange(time_t, time_t + 20)
den_min = 27.38
den_max = 27.58
# den_min = 27.41
# den_max = 27.61

for i in range(9):
    this_t = tt[i]  # tt[i]
    t_s = datetime.date.fromordinal(np.int(Time[this_t][0]))
    t_e = datetime.date.fromordinal(np.int(Time[this_t][1]))
    t_in_all = np.where((time_rec_all > Time[this_t][0]) & (time_rec_all < Time[this_t][1]))[0]
    profs = profile_list_all[t_in_all]
    this_x_all = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (np.array(df_lon_all.iloc[:, t_in_all]) - ref_lon) / 1000
    this_y_all = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (np.array(df_lat_all.iloc[:, t_in_all]) - ref_lat) / 1000
    dive_pos_x = np.nanmean(this_x_all, axis=0)
    dive_pos_y = np.nanmean(this_y_all, axis=0)
    # mask grid for plotting
    sig_th_i = np.nan * np.zeros(np.shape(sigma_theta_all[:, :, :, test_lev]))
    sig_th_i[this_t, mask[this_t][0], mask[this_t][1]] = sigma_theta_all[
        this_t, mask[this_t][0], mask[this_t][1], test_lev]
    U_i = np.nan * np.zeros(np.shape(U_all[:, :, :, test_lev]))
    V_i = np.nan * np.zeros(np.shape(V_all[:, :, :, test_lev]))
    U_i[this_t, mask[this_t][0], mask[this_t][1]] = U_all[this_t, mask[this_t][0], mask[this_t][1], test_lev]
    V_i[this_t, mask[this_t][0], mask[this_t][1]] = V_all[this_t, mask[this_t][0], mask[this_t][1], test_lev]

    # -- old density limits = np.nanmin(sig_th_i[this_t, :, :])
    # -- pcolor mapped density field
    im = ax[axl1[i], axl2[i]].pcolor(x1, y1, sig_th_i[this_t, :, :], vmin=den_min, vmax=den_max, cmap=cmp, zorder=0)
    ax[axl1[i], axl2[i]].contour(x1, y1, sig_th_i[this_t, :, :], colors='k', zorder=1)
    ax[axl1[i], axl2[i]].quiver(x1, y1, U_i[this_t, :, :], V_i[this_t, :, :], color='w',
                                label='Mapped Velocity', scale=1.2, zorder=2)
    # -- add path of DG during mapping window
    ax[axl1[i], axl2[i]].scatter(this_x_all, this_y_all, s=5, color='k', label='DG dives', zorder=3)
    # -- color density values of DG profiles in this window
    ax[axl1[i], axl2[i]].scatter(this_x_all[test_lev, :], this_y_all[test_lev, :],
                                 c=np.array(df_den.iloc[test_lev, t_in_all]),
                                 s=35, edgecolor='k', cmap=cmp, zorder=4, vmin=den_min, vmax=den_max)
    # -- add DG DACS during window
    this_prof = profile_list[t_in_all]
    for l in range(len(np.unique(np.floor(profile_list[t_in_all])))):
        di = np.unique(np.floor(profile_list[t_in_all]))[l]
        ind_in = np.where((this_prof >= di) & (this_prof <= di+1))[0]
        dau = dac_u[t_in_all]
        dav = dac_v[t_in_all]
        ax[axl1[i], axl2[i]].quiver(this_x_all[-50, ind_in[0]], this_y_all[-50, ind_in[0]],
                                    dau[l], dav[l], color='r', label='DAC', scale=1.4, zorder=3)

    for j in range(len(profs)):
        ax[axl1[i], axl2[i]].text(dive_pos_x[j] + 2, dive_pos_y[j], str(profs[j]), color='r',
                                  fontsize=6, fontweight='bold')
    ax[axl1[i], axl2[i]].set_title(
        str(grid[test_lev]) + 'm ' + '(' + str(t_s) + ' - ' + str(t_e) + ')', fontsize=10)
    if i > 7.5:
        handles, labels = ax[axl1[i], axl2[i]].get_legend_handles_labels()
        ax[axl1[i], axl2[i]].legend(handles[0:3], labels[0:3], fontsize=10)
    if i < 1:
        ax[axl1[i], axl2[i]].set_ylabel('Y distance [km]', fontsize=10)
    if (i > 2) & (i < 4):
        ax[axl1[i], axl2[i]].set_ylabel('Y distance [km]', fontsize=10)
    if (i > 5) & (i < 7):
        ax[axl1[i], axl2[i]].set_ylabel('Y distance [km]', fontsize=10)
    if i >= 6:
        ax[axl1[i], axl2[i]].set_xlabel('X distance [km]', fontsize=10)
    # plt.colorbar(im, ax=ax[axl1[i], axl2[i]], orientation='horizontal')
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.03, 0.7])
    f.colorbar(im, cax=cbar_ax)

    ax[axl1[i], axl2[i]].grid()
    ax[axl1[i], axl2[i]].grid()
    plot_pro(ax[axl1[i], axl2[i]])


# ---- DAC / integral check
# xc = mask[time_t][0][0]
# yc = mask[time_t][1][0]
# dac_u_c = dac_u_map[time_t]
# dac_v_c = dac_v_map[time_t]
# f, ax = plt.subplots()
# ax.plot(U_all[time_t, xc, yc, :], grid)
# ax.plot(U[time_t][0, :], grid, linestyle=':')
# ax.plot([dac_u_c[xc, yc], dac_u_c[xc, yc]], [0, 5000])
# iq = np.where(~np.isnan(U_all[time_t, xc, yc, :]))
# z2 = -grid[iq]
#  urel_av = np.trapz(U_all[time_t, xc, yc, iq] / (z2[-1] - z2[0]), x=z2)
# ubc = U_all[time_t, xc, yc, :] - urel_av
# ax.invert_yaxis()
# plot_pro(ax)

# ---- gradients
# f, ax_ar = plt.subplots(3, 3)
# colors = plt.cm.Spectral(np.linspace(0, 1, 13))
# count1 = 0
# count2 = 0
# lim = 0.01
# time_m = range(0, 9, 1)
# for m in range(9):
#     count = 0
#     time_t = time_m[m]
#     for i in range(0, grid.size-10, 20):
#         ax_ar[count1, count2].quiver(0, 0, np.nanmean(d_dx_sig[time_t, mask[time_t][0], mask[time_t][1], i]),
#                 np.nanmean(d_dy_sig[time_t, mask[time_t][0], mask[time_t][1], i]), scale=.000009, color=colors[count, :])
#         ax_ar[count1, count2].axis([-lim, lim, -lim, lim])
#         ax_ar[count1, count2].grid()
#         count = count + 1
#     if count1 <= 1:
#         count1 = count1 + 1
#     else:
#         count1 = 0
#         count2 = count2 + 1
#
# ax_ar[2, 2].grid()
# plot_pro(ax_ar[2, 2])
# --------------------------------------------------------

# cont_data = np.array(df_den.iloc[np.where(grid == grid[k1])[0][0], time_in])
# this_x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (
#             np.array(df_lon.iloc[np.where(grid == grid[k1])[0][0], time_in]) - ref_lon)
# this_y = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (
#             np.array(df_lat.iloc[np.where(grid == grid[k1])[0][0], time_in]) - ref_lat)
# ax_ar[0, 0].scatter(this_x / 1000, this_y / 1000, c=np.array(df_den.iloc[np.where(grid == grid[k1])[0][0], time_in]),
#                     s=60, cmap=plt.cm.coolwarm, zorder=2, vmin=np.min(cont_data), vmax=np.max(cont_data)
# im = ax_ar[0, 0].pcolor(x1 / 1000, y1 / 1000, sigma_theta[:, :, k1], cmap=plt.cm.coolwarm, zorder=0,
#                         vmin=np.min(cont_data), vmax=np.max(cont_data))
# ax_ar[0, 0].quiver(d_x / 1000, d_y / 1000, d_u_in, d_v_in, color='k', angles='xy', scale_units='xy', scale=.005,
#                    zorder=1)
# ax_ar[0, 0].quiver(x1 / 1000, y1 / 1000, U_g[:, :, k1], V_g[:, :, k1], color='g', angles='xy', scale_units='xy',
#                    scale=.005, zorder=1)
# ax_ar[0, 0].plot(np.array([x_grid[0], x_grid[-1]]) / 1000, np.array([y_grid[fixed_lat], y_grid[fixed_lat]]) / 1000,
#                  color='k', linestyle='--', zorder=3, linewidth=0.5)
# ax_ar[0, 0].plot(np.array([x_grid[fixed_lon], x_grid[fixed_lon]]) / 1000, np.array([y_grid[0], y_grid[-1]]) / 1000,
#                  color='k', linestyle='--', zorder=3, linewidth=0.5)
# ax_ar[0, 0].set_title('Obj. map at ' + str(grid[k1]) + 'm')
# ax.axis([-lim, lim, -lim, lim])
# ax.grid()
# plt.colorbar(im, ax=ax_ar[0, 0], orientation='horizontal')
#
# cont_data = np.array(df_den.iloc[np.where(grid == grid[k2])[0][0], time_in])
# this_x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (
#             np.array(df_lon.iloc[np.where(grid == grid[k2])[0][0], time_in]) - ref_lon)
# this_y = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (
#             np.array(df_lat.iloc[np.where(grid == grid[k2])[0][0], time_in]) - ref_lat)
# ax_ar[0, 1].scatter(this_x / 1000, this_y / 1000, c=np.array(df_den.iloc[np.where(grid == grid[k2])[0][0], time_in]),
#                     s=60, cmap=plt.cm.coolwarm, zorder=2, vmin=np.min(cont_data), vmax=np.max(cont_data)
# im = ax_ar[0, 1].pcolor(x1 / 1000, y1 / 1000, sigma_theta[:, :, k2], cmap=plt.cm.coolwarm, zorder=0,
#                         vmin=np.min(cont_data), vmax=np.max(cont_data))
# ax_ar[0, 1].quiver(d_x / 1000, d_y / 1000, d_u_in, d_v_in, color='k', angles='xy', scale_units='xy', scale=.005,
#                    zorder=1)
# ax_ar[0, 1].quiver(x1 / 1000, y1 / 1000, U_g[:, :, k2], V_g[:, :, k2], color='g', angles='xy', scale_units='xy',
#                    scale=.005, zorder=1)
# ax_ar[0, 1].set_title('Obj. map at ' + str(grid[k2]) + 'm')
# ax_ar[0, 1].axis([-lim, lim, -lim, lim])
# ax_ar[0, 1].grid()
# plt.colorbar(im, ax=ax_ar[0, 1], orientation='horizontal')
#
# cont_data = np.array(df_den.iloc[np.where(grid == grid[k3])[0][0], time_in])
# this_x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (
#             np.array(df_lon.iloc[np.where(grid == grid[k3])[0][0], time_in]) - ref_lon)
# this_y = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (
#             np.array(df_lat.iloc[np.where(grid == grid[k3])[0][0], time_in]) - ref_lat)
# ax_ar[0, 2].scatter(this_x / 1000, this_y / 1000, c=np.array(df_den.iloc[np.where(grid == grid[k3])[0][0], time_in]),
#                     s=60, cmap=plt.cm.coolwarm, zorder=2, vmin=np.min(cont_data), vmax=np.max(cont_data)
# im = ax_ar[0, 2].pcolor(x1 / 1000, y1 / 1000, sigma_theta[:, :, k3], cmap=plt.cm.coolwarm, zorder=0,
#                         vmin=np.min(cont_data), vmax=np.max(cont_data))
# ax_ar[0, 2].quiver(d_x / 1000, d_y / 1000, d_u_in, d_v_in, color='k', angles='xy', scale_units='xy', scale=.005,
#                    zorder=1)
# ax_ar[0, 2].quiver(x1 / 1000, y1 / 1000, U_g[:, :, k3], V_g[:, :, k3], color='g', angles='xy', scale_units='xy',
#                    scale=.005, zorder=1)
# ax_ar[0, 2].set_title('Obj. map at ' + str(grid[k3]) + 'm')
# ax_ar[0, 2].axis([-lim, lim, -lim, lim])
# ax_ar[0, 2].grid()
# plt.colorbar(im, ax=ax_ar[0, 2], orientation='horizontal')
#
# im = ax_ar[1,0].pcolor(x1/1000, y1/1000, error[:,:,k1], cmap=plt.cm.plasma,zorder=0) # ,vmin=0.0005,vmax=.02)
# ax_ar[1,0].scatter(x1[good_prof[0],good_prof[1]]/1000,y1[good_prof[0],good_prof[1]]/1000,c='k',s=7,zorder=1)
# plt.colorbar(im,ax=ax_ar[1,0],orientation='horizontal')
# ax_ar[1,0].set_title('Error Map')
# ax_ar[1,0].axis([-lim,lim,-lim,lim])
# ax_ar[1,0].grid()
#
# im = ax_ar[1,1].pcolor(x1/1000, y1/1000, error[:,:,k2], cmap=plt.cm.plasma,zorder=0) # ,vmin=0.0005,vmax=.02)
# ax_ar[1,1].scatter(x1[good_prof[0],good_prof[1]]/1000,y1[good_prof[0],good_prof[1]]/1000,c='k',s=7,zorder=1)
# plt.colorbar(im,ax=ax_ar[1,1],orientation='horizontal')
# ax_ar[1,1].axis([-lim,lim,-lim,lim])
# ax_ar[1,1].set_title('Error Map')
# ax_ar[1,1].grid()
#
# im = ax_ar[1,2].pcolor(x1/1000, y1/1000, error[:,:,k3], cmap=plt.cm.plasma,zorder=0) #,vmin=0.0005,vmax=.02)
# ax_ar[1,2].scatter(x1[good_prof[0],good_prof[1]]/1000,y1[good_prof[0],good_prof[1]]/1000,c='k',s=7,zorder=1)
# plt.colorbar(im,ax=ax_ar[1,2],orientation='horizontal')
# ax_ar[1,2].set_title('Error Map')
# ax_ar[1,2].axis([-lim,lim,-lim,lim])
# ax_ar[1,2].grid()
# plot_pro(ax_ar[1,2])
# fig.savefig( ('/Users/jake/Desktop/BATS/bats_mapping/map_' + str(k_out) + '.png'),dpi = 300)
# plt.close()

# ------- 3-D PLOT AND DEN AND VELOCITY
movie_making = 0
if movie_making > 0:
    k1 = 30
    k3 = 112
    k4 = 200

    # LOOP OVER TIMES
    # relevant glider dives
    wins = range(0, np.shape(Time)[0], 1)  # [0, 1, 2, 3, 4, 5, 6, 7]
    for time_i in wins:
        lvls = np.linspace(np.float(np.nanmin(sigma_theta_all[:, :, :, k1])),
                           np.float(np.nanmax(sigma_theta_all[:, :, :, k1])), 20)
    lvls3 = np.linspace(np.float(np.nanmin(sigma_theta_all[:, :, :, k3])),
                        np.float(np.nanmax(sigma_theta_all[:, :, :, k3])), 20)
    lvls4 = np.linspace(np.float(np.nanmin(sigma_theta_all[:, :, :, k4])),
                        np.float(np.nanmax(sigma_theta_all[:, :, :, k4])), 20)

    x_grid = (1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_grid_all[time_i, :] - ref_lon)) / 10000
    y_grid = (1852 * 60 * (lat_grid_all[time_i, :] - ref_lat)) / 10000
    x1, y1 = np.meshgrid(x_grid, y_grid)
    grid2 = grid.copy() / 1000

    t_s = datetime.date.fromordinal(np.int(Time[time_i][0]))
    t_e = datetime.date.fromordinal(np.int(Time[time_i][1]))
    time_in = np.where((time_rec > Time[time_i][0]) & (time_rec < Time[time_i][1]))[0]
    time_in_all = np.where((time_rec_all > Time[time_i][0]) & (time_rec_all < Time[time_i][1]))[0]
    uint = 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(x1, y1, (grid2[k1] + sigma_theta_all[time_i, :, :, k1] / 1000), cmap="autumn_r", alpha=0.5)
    ax.contour(x1, y1, (grid2[k1] + sigma_theta_all[time_i, :, :, k1] / 1000), zdir='z', cmap="RdBu_r",
               levels=(grid2[k1] + lvls / 1000), zorder=1, linewidth=0.75)
    ax.quiver(x1[0:-1:uint, 0:-1:uint], y1[0:-1:uint, 0:-1:uint],
              grid2[k1] * np.ones(np.shape(x1[0:-1:uint, 0:-1:uint])),
              U_all[time_i, 0:-1:uint, 0:-1:uint, k1], V_all[time_i, 0:-1:uint, 0:-1:uint, k1],
              np.zeros(np.shape(x1[0:-1:uint, 0:-1:uint])), color='k', length=7)

    ax.plot_surface(x1, y1, grid2[k3] + sigma_theta_all[time_i, :, :, k3] / 1000, cmap="autumn_r", alpha=0.5)
    ax.contour(x1, y1, grid2[k3] + sigma_theta_all[time_i, :, :, k3] / 1000, zdir='z', cmap="RdBu_r",
               levels=grid2[k3] + lvls3 / 1000, zorder=0, linewidth=0.75)
    ax.quiver(x1[0:-1:uint, 0:-1:uint], y1[0:-1:uint, 0:-1:uint],
              grid2[k3] * np.ones(np.shape(x1[0:-1:uint, 0:-1:uint])),
              U_all[time_i, 0:-1:uint, 0:-1:uint, k3], V_all[time_i, 0:-1:uint, 0:-1:uint, k3],
              np.zeros(np.shape(x1[0:-1:uint, 0:-1:uint])), color='k', length=7)

    ax.plot_surface(x1, y1, grid2[k4] + sigma_theta_all[time_i, :, :, k4] / 1000, cmap="autumn_r", alpha=0.5)
    ax.contour(x1, y1, grid2[k4] + sigma_theta_all[time_i, :, :, k4] / 1000, zdir='z', cmap="RdBu_r",
               levels=grid2[k4] + lvls4 / 1000, zorder=0, linewidth=0.75)
    ax.quiver(x1[0:-1:uint, 0:-1:uint], y1[0:-1:uint, 0:-1:uint],
              grid[k4] * np.ones(np.shape(x1[0:-1:uint, 0:-1:uint])) / 1000,
              U_all[time_i, 0:-1:uint, 0:-1:uint, k4], V_all[time_i, 0:-1:uint, 0:-1:uint, k4],
              np.zeros(np.shape(x1[0:-1:uint, 0:-1:uint])), color='k', length=7)

    this_x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (np.array(df_lon.iloc[:, time_in]) - ref_lon)
    this_y = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (np.array(df_lat.iloc[:, time_in]) - ref_lat)
    this_x_all = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (np.array(df_lon_all.iloc[:, time_in_all]) - ref_lon)
    this_y_all = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (np.array(df_lat_all.iloc[:, time_in_all]) - ref_lat)
    ax.scatter(this_x / 10000, this_y / 10000, 5500 / 1000, s=7, color='b')
    ax.scatter(this_x_all / 10000, this_y_all / 10000, 5500 / 1000, s=1, color='k')
    ax.scatter(x1[mask[time_i][0], mask[time_i][1]], y1[mask[time_i][0], mask[time_i][1]], 5500 / 1000, s=10,
               color='b')

    ax.quiver(x_grid[0], y_grid[0], -.15, 0, 0.1, 0, color='k', length=10)
    ax.text(x_grid[0], y_grid[0], -.45, '0.1m/s', fontsize=6)

    ax.set_xlim([x_grid[0], x_grid[-1]])
    ax.set_ylim([y_grid[0], y_grid[-1]])
    ax.set_zlim([0, 5.5])
    ax.view_init(elev=15, azim=-60)
    ax.invert_zaxis()
    ax.set_xlabel('X [10km]')
    ax.set_ylabel('Y [10km]')
    ax.set_zlabel('Depth [1000m]')
    ax.set_title(
        'DG35 2015: ' + np.str(t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(
            t_e.day))
    ax.grid()
    # plot_pro(ax)
    fig.savefig(('/Users/jake/Desktop/BATS/bats_mapping/3d/map3d_' + str(time_i) + '.png'), dpi=300)
    plt.close()

    # in a directory with a bunch of png’s like plot_0000.png I did this:
    # ffmpeg -r 8 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p -crf 25 movie.mp4
    #           8 = frame rate (switched to 1 (fps))

    #

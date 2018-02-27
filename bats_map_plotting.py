import numpy as np
import matplotlib.pyplot as plt
import datetime
import seawater as sw
import pandas as pd
from scipy.io import netcdf
import pickle
from mpl_toolkits.mplot3d import axes3d
from toolkit import plot_pro

# LOAD DATA (gridded dives)
GD = netcdf.netcdf_file('BATs_2015_gridded_3.nc', 'r')
profile_list = np.float64(GD.variables['dive_list'][:]) - 35000
df_den = pd.DataFrame(np.float64(GD.variables['Density'][:]), index=np.float64(GD.variables['grid'][:]),
                      columns=np.float64(GD.variables['dive_list'][:]))

# physical parameters
g = 9.81
rho0 = 1027
bin_depth = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 5000, 20)])
ref_lat = 31.8
ref_lon = -64.2
lat_in = 31.7
lon_in = 64.2
grid = bin_depth[1:-1]
grid_p = sw.pres(grid, lat_in)
z = -1 * grid

# select only dives with depths greater than 4000m 
grid_test = np.nan * np.zeros(len(profile_list))
for i in range(len(profile_list)):
    grid_test[i] = grid[np.where(np.array(df_den.iloc[:, i]) == np.nanmax(np.array(df_den.iloc[:, i])))[0][0]]
good = np.where(grid_test >= 4000)[0]

# load select profiles
df_den = pd.DataFrame(np.float64(GD.variables['Density'][:]), index=np.float64(GD.variables['grid'][:]),
                      columns=np.float64(GD.variables['dive_list'][:])).iloc[:, good]
df_t = pd.DataFrame(np.float64(GD.variables['Temperature'][:]), index=np.float64(GD.variables['grid'][:]),
                    columns=np.float64(GD.variables['dive_list'][:])).iloc[:, good]
df_s = pd.DataFrame(np.float64(GD.variables['Salinity'][:]), index=np.float64(GD.variables['grid'][:]),
                    columns=np.float64(GD.variables['dive_list'][:])).iloc[:, good]
df_lon = pd.DataFrame(np.float64(GD.variables['Longitude'][:]), index=np.float64(GD.variables['grid'][:]),
                      columns=np.float64(GD.variables['dive_list'][:])).iloc[:, good]
df_lat = pd.DataFrame(np.float64(GD.variables['Latitude'][:]), index=np.float64(GD.variables['grid'][:]),
                      columns=np.float64(GD.variables['dive_list'][:])).iloc[:, good]
df_lon_all = pd.DataFrame(np.float64(GD.variables['Longitude'][:]), index=np.float64(GD.variables['grid'][:]),
                          columns=np.float64(GD.variables['dive_list'][:]))
df_lat_all = pd.DataFrame(np.float64(GD.variables['Latitude'][:]), index=np.float64(GD.variables['grid'][:]),
                          columns=np.float64(GD.variables['dive_list'][:]))
dac_u = GD.variables['DAC_u'][good]
dac_v = GD.variables['DAC_v'][good]
time_rec = GD.variables['time_start_stop'][good]
time_rec_all = GD.variables['time_start_stop'][:]
profile_list = np.float64(GD.variables['dive_list'][good]) - 35000

# -------------- LOAD IN TRANSECT TO PROFILE DATA COMPILED IN BATS_TRANSECTS.PY
pkl_file = open('/Users/jake/Documents/geostrophic_turbulence/BATS_obj_map_3.pkl', 'rb')
bats_trans = pickle.load(pkl_file)
pkl_file.close()
Time = bats_trans['time']

sigma_theta = np.array(bats_trans['Sigma_Theta'])
U = np.array(bats_trans['U_g'])
V = np.array(bats_trans['V_g'])
sigma_theta_all = np.array(bats_trans['Sigma_Theta_All'])
d_dx_sig = np.array(bats_trans['d_sigma_dx'])
d_dy_sig = np.array(bats_trans['d_sigma_dy'])
U_all = np.array(bats_trans['U_g_All'])
V_all = np.array(bats_trans['V_g_All'])
lat_grid_good = np.array(bats_trans['lat_grid'])
lon_grid_good = np.array(bats_trans['lon_grid'])
lat_grid_all = np.array(bats_trans['lat_grid_All'])
lon_grid_all = np.array(bats_trans['lon_grid_All'])
mask = bats_trans['mask']

# ------------- PLOTTING POSSIBILITIES
# --------LOOK AT DENSITY CROSS-SECTIONS
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

# ------------ PLAN VIEW
test_lev = 100
lim = 50
time_t = 0
time_in_all = np.where((time_rec_all > Time[time_t][0]) & (time_rec_all < Time[time_t][1]))[0]
this_x_all = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (np.array(df_lon_all.iloc[:, time_in_all]) - ref_lon)
this_y_all = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (np.array(df_lat_all.iloc[:, time_in_all]) - ref_lat)
x_grid = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_grid_all[time_t, :] - ref_lon)
y_grid = 1852 * 60 * (lat_grid_all[time_t, :] - ref_lat)
x1, y1 = np.meshgrid(x_grid, y_grid)

f, ax = plt.subplots()
im = ax.pcolor(x1, y1, sigma_theta_all[time_t, :, :, test_lev])
ax.contour(x1, y1, sigma_theta_all[time_t, :, :, test_lev], colors='k')
ax.quiver(x1, y1, d_dx_sig[time_t, :, :, test_lev], d_dy_sig[time_t, :, :, test_lev], color='r')
ax.quiver(x1, y1, U_all[time_t, :, :, test_lev], V_all[time_t, :, :, test_lev], color='w')
ax.scatter(this_x_all, this_y_all, s=10, color='k')
ax.scatter(x1[mask[time_t][0], mask[time_t][1]], y1[mask[time_t][0], mask[time_t][1]], s=15, color='m')
f.colorbar(im, orientation='horizontal')
plot_pro(ax)

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

# --------- PLOT U,V
# fig, (ax0,ax1) = plt.subplots(1,2,sharey=True)
# for i in range(len(good_prof[0])):
#     ax0.plot(U_g[good_prof[0][i],good_prof[1][i],:],grid,color='r',linewidth=0.25)
#     ax1.plot(V_g[good_prof[0][i],good_prof[1][i],:],grid,color='b',linewidth=0.25)
# ax0.set_title('U')
# ax0.set_ylabel('Depth [m]')
# ax0.set_xlabel('m/s')
# ax0.axis([-.3,.3,0,4250])
# ax1.set_title('V')
# ax1.set_xlabel('m/s')
# ax1.axis([-.3,.3,0,4250])
# ax0.grid()
# ax0.invert_yaxis() 
# plot_pro(ax1)

# fig.savefig( ('/Users/jake/Desktop/BATS/bats_mapping/u_v_' + str(k_out) + '.png'),dpi = 300)
# plt.close()

# ------- 3-D PLOT AND DEN AND VELOCITY
k1 = 30
k3 = 112
k4 = 200

# LOOP OVER TIMES 
# relevant glider dives 
wins = [0, 1, 2, 3, 4, 5, 6, 7]
for time_i in wins:
    lvls = np.linspace(np.float(np.nanmin(sigma_theta_all[time_i, :, :, k1])),
                       np.float(np.nanmax(sigma_theta_all[time_i, :, :, k1])), 20)
    lvls3 = np.linspace(np.float(np.nanmin(sigma_theta_all[time_i, :, :, k3])),
                        np.float(np.nanmax(sigma_theta_all[time_i, :, :, k3])), 20)
    lvls4 = np.linspace(np.float(np.nanmin(sigma_theta_all[time_i, :, :, k4])),
                        np.float(np.nanmax(sigma_theta_all[time_i, :, :, k4])), 20)

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
    ax.scatter(x1[mask[time_i][0], mask[time_i][1]], y1[mask[time_i][0], mask[time_i][1]], 5500 / 1000, s=10, color='b')

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
        'DG35 2015: ' + np.str(t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day))
    ax.grid()
    # plot_pro(ax)
    fig.savefig(('/Users/jake/Desktop/BATS/bats_mapping/3d/map3d_' + str(time_i) + '.png'), dpi=300)
    plt.close()

# in a directory with a bunch of png’s like plot_0000.png I did this:
# ffmpeg -r 8 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p -crf 25 movie.mp4
# and it worked perfectly!!!  Also the movie it made was 5 MB, about half the size of the same movie I had made using Quicktime 7 Pro for PECS.

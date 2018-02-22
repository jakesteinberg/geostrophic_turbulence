import numpy as np
import matplotlib.pyplot as plt
import datetime
import seawater as sw
import pandas as pd
from scipy.io import netcdf
import pickle
from matplotlib import cm
# functions I've written
from toolkit import cart2pol, pol2cart, plot_pro

# LOAD DATA (gridded dives)
GD = netcdf.netcdf_file('BATs_2015_gridded_2.nc', 'r')
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
good = np.where(grid_test > 4000)[0]

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
dac_u = GD.variables['DAC_u'][good]
dac_v = GD.variables['DAC_v'][good]
time_rec = GD.variables['time_start_stop'][good]
heading_rec = GD.variables['heading_record'][good]
profile_list = np.float64(GD.variables['dive_list'][good]) - 35000

#### LOAD IN TRANSECT TO PROFILE DATA COMPILED IN BATS_TRANSECTS.PY
pkl_file = open('/Users/jake/Documents/geostrophic_turbulence/BATS_obj_map_2.pkl', 'rb')
bats_trans = pickle.load(pkl_file)
pkl_file.close()
Time = bats_trans['time']
sigma_theta = np.transpose(bats_trans['Sigma_Theta'])
U = np.transpose(bats_trans['U_g'])
V = np.transpose(bats_trans['V_g'])
sigma_theta_all = np.transpose(bats_trans['Sigma_Theta_All'])
U_all = np.transpose(bats_trans['U_g_All'])
V_all = np.transpose(bats_trans['V_g_All'])
lat_grid_good = np.transpose(bats_trans['lat_grid'])
lon_grid_good = np.transpose(bats_trans['lon_grid'])
lat_grid_all = np.transpose(bats_trans['lat_grid_All'])
lon_grid_all = np.transpose(bats_trans['lon_grid_All'])
mask = bats_trans['mask']

x_grid = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_grid_all - ref_lon)
y_grid = 1852 * 60 * (lat_grid_all - ref_lat)
x1, y1 = np.meshgrid(x_grid, y_grid)

######################################## PLOTTING POSSIBILITIES 
###### LOOK AT DENSITY CROSS-SECTIONS #######
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

###### PLAN VIEW #######
# k1 = 50
# k2 = 100
# k3 = 150
# lim = 50
# fig, ax_ar = plt.subplots(2,3,sharey=True,sharex=True)

# cont_data = np.array(df_den.iloc[np.where(grid==grid[k1])[0][0],time_in])
# this_x = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lon.iloc[np.where(grid==grid[k1])[0][0],time_in]) - ref_lon )
# this_y = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lat.iloc[np.where(grid==grid[k1])[0][0],time_in]) - ref_lat )
# ax_ar[0,0].scatter(this_x/1000,this_y/1000,c=np.array(df_den.iloc[np.where(grid==grid[k1])[0][0],time_in]),s=60,cmap=plt.cm.coolwarm,zorder=2,vmin=np.min(cont_data),vmax=np.max(cont_data)
# im = ax_ar[0,0].pcolor(x1/1000, y1/1000, sigma_theta[:,:,k1], cmap=plt.cm.coolwarm,zorder=0,vmin=np.min(cont_data),vmax=np.max(cont_data))
# ax_ar[0,0].quiver(d_x/1000,d_y/1000,d_u_in,d_v_in, color='k',angles='xy', scale_units='xy', scale=.005,zorder=1)
# ax_ar[0,0].quiver(x1/1000, y1/1000, U_g[:,:,k1], V_g[:,:,k1], color='g',angles='xy', scale_units='xy', scale=.005,zorder=1)
# ax_ar[0,0].plot( np.array([x_grid[0],x_grid[-1]])/1000,np.array([y_grid[fixed_lat],y_grid[fixed_lat]])/1000,color='k',linestyle='--',zorder=3,linewidth=0.5)
# ax_ar[0,0].plot( np.array([x_grid[fixed_lon],x_grid[fixed_lon]])/1000,np.array([y_grid[0],y_grid[-1]])/1000,color='k',linestyle='--',zorder=3,linewidth=0.5)
# ax_ar[0,0].set_title('Obj. map at ' + str(grid[k1]) + 'm')
# ax_ar[0,0].axis([-lim,lim,-lim,lim])
# ax_ar[0,0].grid()
# plt.colorbar(im,ax=ax_ar[0,0],orientation='horizontal')

# cont_data = np.array(df_den.iloc[np.where(grid==grid[k2])[0][0],time_in])
# this_x = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lon.iloc[np.where(grid==grid[k2])[0][0],time_in]) - ref_lon )
# this_y = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lat.iloc[np.where(grid==grid[k2])[0][0],time_in]) - ref_lat )
# ax_ar[0,1].scatter(this_x/1000,this_y/1000,c=np.array(df_den.iloc[np.where(grid==grid[k2])[0][0],time_in]),s=60,cmap=plt.cm.coolwarm,zorder=2,vmin=np.min(cont_data),vmax=np.max(cont_data)
# im = ax_ar[0,1].pcolor(x1/1000, y1/1000, sigma_theta[:,:,k2], cmap=plt.cm.coolwarm,zorder=0,vmin=np.min(cont_data),vmax=np.max(cont_data))
# ax_ar[0,1].quiver(d_x/1000,d_y/1000,d_u_in,d_v_in, color='k',angles='xy', scale_units='xy', scale=.005,zorder=1)
# ax_ar[0,1].quiver(x1/1000, y1/1000, U_g[:,:,k2], V_g[:,:,k2], color='g',angles='xy', scale_units='xy', scale=.005,zorder=1)
# ax_ar[0,1].set_title('Obj. map at ' + str(grid[k2]) + 'm')
# ax_ar[0,1].axis([-lim,lim,-lim,lim])
# ax_ar[0,1].grid()
# plt.colorbar(im,ax=ax_ar[0,1],orientation='horizontal')

# cont_data = np.array(df_den.iloc[np.where(grid==grid[k3])[0][0],time_in])
# this_x = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lon.iloc[np.where(grid==grid[k3])[0][0],time_in]) - ref_lon )
# this_y = 1852*60*np.cos(np.deg2rad(ref_lat))*( np.array(df_lat.iloc[np.where(grid==grid[k3])[0][0],time_in]) - ref_lat )
# ax_ar[0,2].scatter(this_x/1000,this_y/1000,c=np.array(df_den.iloc[np.where(grid==grid[k3])[0][0],time_in]),s=60,cmap=plt.cm.coolwarm,zorder=2,vmin=np.min(cont_data),vmax=np.max(cont_data)
# im = ax_ar[0,2].pcolor(x1/1000, y1/1000, sigma_theta[:,:,k3], cmap=plt.cm.coolwarm,zorder=0,vmin=np.min(cont_data),vmax=np.max(cont_data))
# ax_ar[0,2].quiver(d_x/1000,d_y/1000,d_u_in,d_v_in, color='k',angles='xy', scale_units='xy', scale=.005,zorder=1)
# ax_ar[0,2].quiver(x1/1000, y1/1000, U_g[:,:,k3], V_g[:,:,k3], color='g',angles='xy', scale_units='xy', scale=.005,zorder=1)
# ax_ar[0,2].set_title('Obj. map at ' + str(grid[k3]) + 'm')
# ax_ar[0,2].axis([-lim,lim,-lim,lim])
# ax_ar[0,2].grid()
# plt.colorbar(im,ax=ax_ar[0,2],orientation='horizontal')
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

##### plot U,V #####
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

### 3-D PLOT AND DEN AND VELOCITY 
k1 = 30
lvls = np.linspace(np.int(np.nanmin(sigma_theta_all[k1, :, :, iter])),
                   np.int(np.nanmax(sigma_theta_all[k1, :, :, iter])), 20)
k3 = 112
lvls3 = np.linspace(np.int(np.nanmin(sigma_theta_all[k3, :, :, iter])),
                    np.int(np.nanmax(sigma_theta_all[k3, :, :, iter])), 20)
k4 = 200
lvls4 = np.linspace(np.int(np.nanmin(sigma_theta_all[k4, :, :, iter])),
                    np.int(np.nanmax(sigma_theta_all[k4, :, :, iter])), 20)

# LOOP OVER TIMES 
# relevant glider dives 
wins = [0, 1, 2, 3]
for iter in wins:
    t_s = datetime.date.fromordinal(np.int(Time[iter][0]))
    t_e = datetime.date.fromordinal(np.int(Time[iter][1]))
    time_in = np.where((time_rec > Time[iter][0]) & (time_rec < Time[iter][1]))[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(x1 / 10000, y1 / 10000, (grid[k1] + np.transpose(sigma_theta_all[k1, :, :, iter])) / 1000,
                    cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
    ax.contour(x1 / 10000, y1 / 10000, (grid[k1] + np.transpose(sigma_theta_all[k1, :, :, iter])) / 1000, zdir='z',
               cmap=cm.RdBu_r, levels=(grid[k1] + lvls) / 1000, zorder=1, linewidth=1)
    ax.quiver(x1[0:-1:2, 0:-1:2] / 10000, y1[0:-1:2, 0:-1:2] / 10000,
              grid[k1] * np.ones(np.shape(x1[0:-1:2, 0:-1:2])) / 1000, np.transpose(U_all[k1, 0:-1:2, 0:-1:2, iter]),
              np.transpose(V_all[k1, 0:-1:2, 0:-1:2, iter]), np.zeros(np.shape(x1[0:-1:2, 0:-1:2])), color='k',
              length=10)
    ax.plot_surface((x1 / 1000) / 10, (y1 / 1000) / 10,
                    (grid[k3] + np.transpose(sigma_theta_all[k3, :, :, iter])) / 1000, cmap="autumn_r",
                    alpha=0.5)  # , lw=0.5, rstride=1, cstride=1,
    ax.contour((x1 / 1000) / 10, (y1 / 1000) / 10, (grid[k3] + np.transpose(sigma_theta_all[k3, :, :, iter])) / 1000,
               zdir='z', cmap=cm.RdBu_r, levels=(grid[k3] + lvls3) / 1000, zorder=0, linewidth=1)
    ax.quiver((x1[0:-1:2, 0:-1:2] / 1000) / 10, (y1[0:-1:2, 0:-1:2] / 1000) / 10,
              grid[k3] * np.ones(np.shape(x1[0:-1:2, 0:-1:2])) / 1000, np.transpose(U_all[k3, 0:-1:2, 0:-1:2, iter]),
              np.transpose(V_all[k3, 0:-1:2, 0:-1:2, iter]), np.zeros(np.shape(x1[0:-1:2, 0:-1:2])), color='k',
              length=10)
    ax.plot_surface(x1 / 10000, y1 / 10000, (grid[k4] + np.transpose(sigma_theta_all[k4, :, :, iter])) / 1000,
                    cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
    ax.contour(x1 / 10000, y1 / 10000, (grid[k4] + np.transpose(sigma_theta_all[k4, :, :, iter])) / 1000, zdir='z',
               cmap=cm.RdBu_r, levels=(grid[k4] + lvls4) / 1000, zorder=0, linewidth=1)
    ax.quiver(x1[0:-1:2, 0:-1:2] / 10000, y1[0:-1:2, 0:-1:2] / 10000,
              grid[k4] * np.ones(np.shape(x1[0:-1:2, 0:-1:2])) / 1000, np.transpose(U_all[k4, 0:-1:2, 0:-1:2, iter]),
              np.transpose(V_all[k4, 0:-1:2, 0:-1:2, iter]), np.zeros(np.shape(x1[0:-1:2, 0:-1:2])), color='k',
              length=10)

    # cont_data = np.array(df_den.iloc[np.where(grid == grid[k1])[0][0], time_in])
    # this_x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (
    #             np.array(df_lon.iloc[np.where(grid == grid[k1])[0][0], time_in]) - ref_lon)
    # this_y = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (
    #             np.array(df_lat.iloc[np.where(grid == grid[k1])[0][0], time_in]) - ref_lat)
    # ax.scatter(this_x / 10000, this_y / 10000, 5200 / 1000,
    #            c=np.array(df_den.iloc[np.where(grid == grid[k1])[0][0], time_in]), s=30,
    #            cmap=plt.cm.coolwarm, zorder=3, vmin=np.min(cont_data), vmax=np.max(cont_data))
    this_x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (np.array(df_lon.iloc[:, time_in]) - ref_lon)
    this_y = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (np.array(df_lat.iloc[:, time_in]) - ref_lat)
    ax.scatter(this_x / 10000, this_y / 10000, 5500 / 1000, s=7, color='k')
    ax.scatter(x1[mask[iter][0], mask[iter][1]] / 10000, y1[mask[iter][0], mask[iter][1]] / 10000, 5500 / 1000, s=10,
               color='b')

    ax.set_xlim([x_grid[0] / 10000, x_grid[-1] / 10000])
    ax.set_ylim([y_grid[0] / 10000, y_grid[-1] / 10000])
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
    fig.savefig(('/Users/jake/Desktop/BATS/bats_mapping/3d/map3d_' + str(k_out) + '.png'), dpi=300)
    plt.close()

# in a directory with a bunch of png’s like plot_0000.png I did this:
# ffmpeg -r 8 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p -crf 25 movie.mp4
# and it worked perfectly!!!  Also the movie it made was 5 MB, about half the size of the same movie I had made using Quicktime 7 Pro for PECS.

# Map of locations to be used in research 

import numpy as np
import cartopy
import cartopy.crs as ccrs
import pickle
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from toolkit import plot_pro
from glider_cross_section import Glider

# fig, ax = plt.subplots()
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cartopy.feature.LAND)
# ax.add_feature(cartopy.feature.OCEAN)
# ax.set_extent([-160, -45, 10, 50])
# ax.gridlines(draw_labels=True, linewidth=.5, color='#808080', alpha=.75, linestyle='-', zorder=2)
# # ABACO ATL
# ax.plot([-77, -75], [26.5, 26.5], color='g', linestyle='-', transform=ccrs.PlateCarree(), linewidth=3)
# ax.text(-74, 25.5, 'ABACO', color='k', fontsize=10, transform=ccrs.PlateCarree())
# # # # ARGO
# # ax.scatter([-59], [25], color='r', s=7, transform=ccrs.PlateCarree(), zorder=3)
# # ax.text(-60, 18, 'Deep Argo', color='k', fontsize=10, transform=ccrs.PlateCarree())
# # # # ARGO NZ
# # ax.scatter([-147], [-44], color='r', s=7, transform=ccrs.PlateCarree(), zorder=3)
# # ax.text(-145, -44, 'Deep Argo', color='k', fontsize=10, transform=ccrs.PlateCarree())
# # # ARGO
# # ax.scatter([-59], [26], color='r', s=7, transform=ccrs.PlateCarree(), zorder=3)
# # BATS
# ax.scatter([-64.2], [31.6], color='r', s=9, transform=ccrs.PlateCarree(), zorder=3)
# ax.text(-62, 31.6, 'BATS', color='k', fontsize=10, transform=ccrs.PlateCarree())
# # 36
# ax.scatter([-64.2], [36], color='r', s=9, transform=ccrs.PlateCarree(), zorder=3)
# ax.text(-62, 36, '36N', color='k', fontsize=10, transform=ccrs.PlateCarree())
# # PAPA
# ax.scatter([-145], [50], color='r', s=7, transform=ccrs.PlateCarree(), zorder=3)
# ax.text(-144, 49, 'PAPA', color='k', fontsize=10, transform=ccrs.PlateCarree())
# # HOTS
# ax.scatter([-158], [22.75], color='r', s=9, transform=ccrs.PlateCarree(), zorder=3)
# ax.text(-157, 21.75, 'ALOHA', color='k', fontsize=10, transform=ccrs.PlateCarree())
#
# ax.text(-0.07, 0.55, 'latitude', va='bottom', ha='center',
#         rotation='vertical', rotation_mode='anchor',
#         transform=ax.transAxes)
# ax.text(0.5, -0.2, 'longitude', va='bottom', ha='center',
#         rotation='horizontal', rotation_mode='anchor',
#         transform=ax.transAxes)
#
# plot_pro(ax)
# fig.savefig('/Users/jake/Desktop/abaco/map_2.png', dpi=300)

# BATS 2015
pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg035_2015_initial_processing.pkl', 'rb')
IP = pickle.load(pkl_file)
pkl_file.close()
mw_lon = IP['mw_lon']
mw_lat = IP['mw_lat']

# 36N 2018
pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg041_2018_initial_processing.pkl', 'rb')
IP = pickle.load(pkl_file)
pkl_file.close()
mw_lon_36 = IP['mw_lon']
mw_lat_36 = IP['mw_lat']

# ABACO
x = Glider(39, np.concatenate((np.arange(18, 62), np.arange(63, 94))),
           r'/Users/jake/Documents/baroclinic_modes/DG/ABACO_2018/sg039')
GD = Dataset('/Users/jake/Documents/geostrophic_turbulence/BATs_2015_gridded_apr04.nc', 'r')
grid = np.concatenate([GD.variables['grid'][:], np.arange(GD.variables['grid'][:][-1] + 20, 4700, 20)])
Binned = x.make_bin(grid)
d_time = Binned['time']
lon_ab = Binned['lon']
lat_ab = Binned['lat']

# bath = '/Users/jake/Desktop/bats/bats_bathymetry/GEBCO_2014_2D_-67.7_29.8_-59.9_34.8.nc'
bath = '/Users/jake/Desktop/bats/bats_bathymetry/etopo180_3e57_6032_17e3.nc'
bath_fid = Dataset(bath, 'r')
bath_lon = bath_fid.variables['longitude'][:]
bath_lat = bath_fid.variables['latitude'][:]
bath_z = bath_fid.variables['altitude'][:]
levels = [-5200, -5100, -5000, -4900, -4800, -4700, -4600,
          -4500, -4400, -4300, -4200, -4100, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0]
fig0, ax0 = plt.subplots()
cmap = plt.cm.get_cmap("Blues_r")
cmap.set_over('#808000')  # ('#E6E6E6')
bc = ax0.contourf(bath_lon, bath_lat, bath_z, levels, cmap='Blues_r', extend='both', zorder=0)
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
bcl = ax0.contour(bath_lon, bath_lat, bath_z, [-5000, -4000, -3000, -2000], colors='k', zorder=0, linewidths=0.35)

ax0.scatter(np.nanmean(mw_lon), np.nanmean(mw_lat), color='r', s=2, label='BATS 2014-15')
ax0.scatter(np.nanmean(mw_lon_36), np.nanmean(mw_lat_36), color='#FF00FF', s=2, label='36$^{\circ}$N 2018')
ax0.scatter(np.nanmean(lon_ab), np.nanmean(lat_ab), color='#FF8C00', s=2, label='ABACO 2017-18')
ax0.scatter(mw_lon, mw_lat, color='r', s=0.5)
ax0.scatter(mw_lon_36, mw_lat_36, color='#FF00FF', s=0.5)
ax0.scatter(lon_ab, lat_ab, color='#FF8C00', s=0.5)

handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles, labels, fontsize=10, fancybox=True, framealpha=1, loc='lower right')
w = 1 / np.cos(np.deg2rad(30))
ax0.axis([-80, -58, 23.5, 40])

divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig0.colorbar(bc, cax=cax, label='[m]')

ax0.set_aspect(w)
ax0.set_xlabel('Longitude')
ax0.set_ylabel('Latitude')
ax0.set_title('North Atlantic Deepglider Deployment Sites')
ax0.grid()
plot_pro(ax0)
fig0.savefig('/Users/jake/Documents/baroclinic_modes/Meetings/meeting_19_04_18/dg_deployments.png', dpi=350)
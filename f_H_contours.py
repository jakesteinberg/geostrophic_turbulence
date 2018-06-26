import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from toolkit import plot_pro

limits = [-128.5, -123.75, 46, 50]
ref_lat = 48.5

bathy_path = '/Users/jake/Documents/Cuddy_tailored/DG_wa_coast/smith_sandwell_wa_coast.nc'
bath_fid = Dataset(bathy_path, 'r')
bath_lon = bath_fid.variables['longitude'][:] - 360
bath_lat = bath_fid.variables['latitude'][:]
bath_z = np.int32(bath_fid.variables['ROSE'][:])
levels = [-3000, -2500, -2000, -1500, -1000, -500, 0]
cmap = plt.cm.get_cmap("Blues_r")
cmap.set_over('#808000')  # ('#E6E6E6')

# cuddy path
mat = scipy.io.loadmat(
    '/Users/jake/Documents/Cuddy_tailored/state_store_restart_11_20_to_2_25_gaussian_with_108.mat')
state_hist = mat['state_store']['x_smooth'][0][0]
# look at Cuddy average speed when moving SW
period = [139, 200]
klon = state_hist[period[0]:period[1], 0]
klat = state_hist[period[0]:period[1], 1]
ktime = mat['state_store']['time'][0][0][period[0]:period[1]]
kbu = state_hist[period[0]:period[1], 4]
kbv = state_hist[period[0]:period[1], 5]
kdist = np.nan * np.ones(len(klon) - 1)
kspan = np.nan * np.ones(len(klon) - 1)
for i in range(len(klon) - 1):
    dxs = 1000 * 1.852 * 60 * np.cos(np.deg2rad(ref_lat)) * (klon[i+1] - klon[i])  # zonal sfc disp [km]
    dys = 1000 * 1.852 * 60 * (klat[i+1] - klat[i])  # meridional sfc disp [km]
    kdist[i] = np.sqrt(dxs**2 + dys**2)
    kspan[i] = (ktime[i + 1] - ktime[i]) * (24 * 60 * 60)
speed = kdist / kspan
speed[speed > 0.3] = np.nan
back_flow = np.sqrt(kbu**2 + kbv**2)
print('eddy speed = ' + str(np.nanmean(speed)) + ' m/s')
print('background flow speed = ' + str(np.nanmean(back_flow)) + ' m/s')

ff = np.pi * np.sin(np.deg2rad(bath_lat)) / (12 * 1800)  # Coriolis parameter [s^-1]
fff = np.tile(ff[:, None], (1, len(bath_lon)))
bath_z[bath_z >= -40] = 100
f_h = fff / (-1 * bath_z)
f_h[f_h <= 0] = np.nan
# levels_f_h = np.arange(np.nanmin(f_h), np.nanmax(f_h), (np.nanmax(f_h) - np.nanmin(f_h))/50)
levels_f_h = np.concatenate([np.arange(2*10**-8, 1*10**-7, 2*10**-8), np.arange(1*10**-7, 1*10**-6, 2*10**-7)])

f, ax = plt.subplots()
bc = ax.contourf(bath_lon, bath_lat, bath_z, levels, cmap='Blues_r', extend='both', zorder=0)
ax.contour(bath_lon, bath_lat, f_h, levels_f_h, colors='k')
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
ax.plot(state_hist[:, 0], state_hist[:, 1], color='#FFD700')
w = 1 / np.cos(np.deg2rad(ref_lat))
ax.axis(limits)
ax.set_aspect(w)
plot_pro(ax)
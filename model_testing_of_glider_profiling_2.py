import numpy as np
import pickle
import glob
import datetime
import gsw
# -- plotting
import matplotlib.pyplot as plt
from toolkit import plot_pro

# ---- LOAD PROCESSED FILES ---------------------------------------------------------------------------------------
file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/ex*.pkl')
# file_name = open('/Users/jake/Documents/baroclinic_modes/Model/test_extraction_1541203200.pkl', 'rb')
file_name = open(file_list[39], 'rb')
D = pickle.load(file_name)
file_name.close()

time = D['ocean_time']
u = D['u']
v = D['v']
t = D['temp']
s = D['salt']
lon_rho = D['lon_rho'][:]
lat_rho = D['lat_rho'][:]
lon_u = D['lon_u'][:]
lat_u = D['lat_u'][:]
lon_v = D['lon_v'][:]
lat_v = D['lat_v'][:]
# lon_all = D['lon_all'][:]
# lat_all = D['lat_all'][:]
# z_bath = D['z_bath'][:]
z = D['z'][:]
ref_lat = np.nanmean(lat_rho)

print(lon_rho[0])
print(lat_rho[0])

# -- TIME
time_cor = time/(60*60*24) + 719163  # correct to start (1970)
time_hour = 24*(time_cor - np.int(time_cor))
date_time = datetime.date.fromordinal(np.int(time_cor))

# -- GSW
p = np.nan * np.ones(np.shape(s))
SA = np.nan * np.ones(np.shape(s))
CT = np.nan * np.ones(np.shape(s))
sig0 = np.nan * np.ones(np.shape(s))
for i in range(len(lat_rho)):
    p[:, i] = gsw.p_from_z(z[:, i], ref_lat * np.ones(len(z)))
    SA[:, i] = gsw.SA_from_SP(s[:, i], p[:, i], lon_rho[0] * np.ones(len(s[:, i])), lat_rho[i] * np.ones(len(s[:, i])))
    CT[:, i] = gsw.CT_from_t(s[:, i], t[:, i], p[:, i])
    sig0[:, i] = gsw.sigma0(SA[:, i], CT[:, i])


# --- PLOT
f, ax = plt.subplots()
ax.pcolor(lat_rho, z, t, vmin=1, vmax=13)
uvc = ax.contour(np.tile(lat_u, (len(z), 1)), z, u,
                 levels=[-.2, -.1, -.05, -.025, -0.01, 0, 0.01, 0.025, 0.05, 0.1, 0.2], colors='k', linewidth=0.75)
ax.clabel(uvc, inline_spacing=-3, fmt='%.4g', colors='k')
rhoc = ax.contour(np.tile(lat_u, (len(z), 1)), z, sig0, levels=[26, 26.2, 26.4, 26.6, 26.8, 27, 27.2, 27.4, 27.6],
                  colors='#A9A9A9', linewidth=0.5)
ax.set_title(str(date_time) + '  ' + str(np.int(np.round(time_hour, 1))) + 'hr')
ax.set_ylim([-2750, 0])
ax.grid()
plot_pro(ax)

# f, ax = plt.subplots()
# ax.contour(lon_all, lat_all, z_bath, levels=[-2500, -2250, -2000], colors='k')
# ax.plot(lon_all, lat_all, color='r')
# ax.axis([-128, -122, 42, 50])
# plot_pro(ax)

# TODO
# compute density
# figure out timing of fake glider dive
# if dive measurements are made every few seconds, will have to interpolate between relevant mode estimates (1hr apart)
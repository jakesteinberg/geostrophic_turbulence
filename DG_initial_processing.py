from grids import collect_dives
import numpy as np
from scipy.io import netcdf
from netCDF4 import Dataset
import glob
import seawater as sw

# --- gliders
dg_list = glob.glob('/Users/jake/Documents/baroclinic_modes/DG/sg035_BATS_2014/p*.nc')

# choose reference latitude
ref_lat = 31.8

# choose bin_depth increment
bin_depth = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 5000, 20)])
grid = bin_depth[1:-1]
grid_p = sw.pres(grid, ref_lat)

theta_g, salin_g, den_g, lon_g, lat_g, dac_u, dac_v, time_rec, time_sta_sto, targ_rec, gps_rec, dive_list = collect_dives(
    dg_list, bin_depth, grid_p, ref_lat)

theta_g[np.isnan(theta_g)] = -999
salin_g[np.isnan(salin_g)] = -999
den_g[np.isnan(den_g)] = -999
lon_g[np.isnan(lon_g)] = -999
lat_g[np.isnan(lat_g)] = -999
dac_u[np.isnan(dac_u)] = -999
dac_v[np.isnan(dac_v)] = -999


def is_number(a):
    # will be True also for 'NaN'
    try:
        number = float(a)
        return True
    except ValueError:
        return False


targ_rec_o = np.zeros(len(targ_rec))
for i in range(len(targ_rec)):
    test = is_number(targ_rec[i])
    if test:
        targ_rec_o[i] = float(targ_rec[i])
    else:
        targ_rec_o[i] = np.nan

f = netcdf.netcdf_file('BATs_2014_gridded.nc', 'w')
f.history = 'DG 2015 dives; have been gridded vertically and separated into dive and climb cycles'
f.createDimension('grid', np.size(grid))
f.createDimension('lat_lon', 4)
f.createDimension('dive_list', np.size(dive_list))
b_d = f.createVariable('grid', np.float64, ('grid',))
b_d[:] = grid
b_l = f.createVariable('dive_list', np.float64, ('dive_list',))
b_l[:] = dive_list
b_t = f.createVariable('Theta', np.float64, ('grid', 'dive_list'))
b_t[:] = theta_g
b_s = f.createVariable('Salinity', np.float64, ('grid', 'dive_list'))
b_s[:] = salin_g
b_den = f.createVariable('Density', np.float64, ('grid', 'dive_list'))
b_den[:] = den_g
b_lon = f.createVariable('Longitude', np.float64, ('grid', 'dive_list'))
b_lon[:] = lon_g
b_lat = f.createVariable('Latitude', np.float64, ('grid', 'dive_list'))
b_lat[:] = lat_g
b_u = f.createVariable('DAC_u', np.float64, ('dive_list',))
b_u[:] = dac_u
b_v = f.createVariable('DAC_v', np.float64, ('dive_list',))
b_v[:] = dac_v
b_time = f.createVariable('time_rec', np.float64, ('dive_list',))
b_time[:] = time_rec
b_t_ss = f.createVariable('time_start_stop', np.float64, ('dive_list',))
b_t_ss[:] = time_sta_sto
b_targ = f.createVariable('target_record', np.float64, ('dive_list',))
b_targ[:] = targ_rec_o
b_gps = f.createVariable('gps_record', np.float64, ('dive_list', 'lat_lon'))
b_gps[:] = gps_rec
f.close()

# time_in_test = np.where((time_sta_sto > Time[time_i][0]) & (time_sta_sto < Time[time_i][1]))[0]
# plt.figure()
# plt.scatter(df_lon.iloc[:, time_in_test], df_lat.iloc[:, time_in_test])
# plt.show()

from grids import collect_dives
import numpy as np
from scipy.io import netcdf
from toolkit import is_number
import glob
import gsw

# --- gliders
dg_list = glob.glob('/Users/jake/Documents/baroclinic_modes/DG/sg035_BATS_2015/p*.nc')

# choose reference latitude
ref_lat = 31.8

# choose bin_depth increment
bin_depth = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4500, 20)])
bin_depth[0] = 5
grid = bin_depth
grid_p = gsw.p_from_z(-1*grid, ref_lat)

theta_g, ct_g, sa_g, sig0_g, lon_g, lat_g, dac_u, dac_v, time_rec, time_sta_sto, targ_rec, gps_rec, dive_list = collect_dives(
    dg_list, bin_depth, grid_p, ref_lat)

" note: time_rec is the median time for each profile while time_sta_sto is the earliest time on the dive profile" \
"and the latest time on the climb profile"

theta_g[theta_g < 0] = -999
theta_g[np.isnan(theta_g)] = -999
ct_g[ct_g < 0] = -999
ct_g[np.isnan(ct_g)] = -999
sa_g[sa_g < 0] = -999
sa_g[np.isnan(sa_g)] = -999
sig0_g[sig0_g < 0] = -999
sig0_g[np.isnan(sig0_g)] = -999
lon_g[lon_g < -360] = -999
lon_g[np.isnan(lon_g)] = -999
lat_g[lat_g < 0] = -999
lat_g[np.isnan(lat_g)] = -999
dac_u[np.isnan(dac_u)] = -999
dac_v[np.isnan(dac_v)] = -999

# -- check targets and make sure that they are numbered (if not == nan)
targ_rec_o = np.zeros(len(targ_rec))
for i in range(len(targ_rec)):
    test = is_number(targ_rec[i])
    if test:
        targ_rec_o[i] = float(targ_rec[i])
    else:
        targ_rec_o[i] = np.nan

f = netcdf.netcdf_file('BATs_2015_gridded_apr04.nc', 'w')
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
b_t = f.createVariable('Conservative Temperature', np.float64, ('grid', 'dive_list'))
b_t[:] = ct_g
b_s = f.createVariable('Absolute Salinity', np.float64, ('grid', 'dive_list'))
b_s[:] = sa_g
b_den = f.createVariable('Density', np.float64, ('grid', 'dive_list'))
b_den[:] = sig0_g
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

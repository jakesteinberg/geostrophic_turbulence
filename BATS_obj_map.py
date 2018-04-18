# BATS STATISTICAL INTERPOLATION ON DEPTH LEVELS 

# Procedue:
# 1. collect data and define lat/lon window within which we want to grid
# 2. define grid
# 3. determine a spatial mean and remove it
#  - need a function to determine a vector pointing in the direction of greatest change
#   (do I want to remove a linear mean that is a function of some combination of lat/lon?)
# 4. determine spatial and temporal decorrelation scales
# 4. objectively map 

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.integrate import cumtrapz
from scipy.linalg import solve
import seawater as sw
import pickle
import datetime
from toolkit import cart2pol, pol2cart, plot_pro, nanseg_interp, data_covariance, trend_fit

def createCorrelationMatrices(lon, lat, data_anom, data_sig, noise_sig, Lx, Ly):
    # lon: vector of data longitudes
    # lat: vector of data latitudes
    # data_anom: vector of data anomalies from fitted large-scale trend
    # data_sig: signal standard deviation, in data units
    # noise_sig: noise standard deviation, in data units
    # Lx: Zonal Gaussian covariance lengthscale, in km
    # Ly: Meridional Gaussian covariance lengthscale, in km
    npts = len(lon)
    C = np.zeros((npts, npts), dtype=np.float)
    A = np.zeros((npts, npts), dtype=np.float)
    for j in range(npts):
        # xscale = np.cos(lat[j]*3.1415926/180.)
        for i in range(npts):
            # dxij = (lon[j]-lon[i])*111.0*xscale
            # dyij = (lat[j]-lat[i])*111.0
            dxij = lon[j] - lon[i]
            dyij = lat[j] - lat[i]
            C[j, i] = data_sig * data_sig * np.exp(-(dxij * dxij) / (Lx * Lx) - (dyij * dyij) / (Ly * Ly))
            if (i == j):
                A[j, i] = C[j, i] + noise_sig * noise_sig
            else:
                A[j, i] = C[j, i]
    Ainv = np.linalg.inv(A)
    return Ainv


# START

# LOAD DATA (gridded dives)
GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
bin_depth = GD.variables['grid'][:]
profile_list = GD['dive_list'][:] - 35000
df_den = pd.DataFrame(GD['Density'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lon = pd.DataFrame(GD['Longitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lat = pd.DataFrame(GD['Latitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lon[df_lon < -500] = np.nan
df_lat[df_lat < -500] = np.nan

# physical parameters 
g = 9.81
rho0 = 1027
#  bin_depth = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 5000, 20)])
ref_lon = np.nanmean(df_lon)
ref_lat = np.nanmean(df_lat)
grid = bin_depth
grid_p = sw.pres(grid, ref_lat)
z = -1 * grid
grid_2 = grid[0:214]

# select only dives with depths greater than 4000m 
grid_test = np.nan * np.zeros(len(profile_list))
for i in range(len(profile_list)):
    grid_test[i] = grid[np.where(np.array(df_den.iloc[:, i]) == np.nanmax(np.array(df_den.iloc[:, i])))[0][0]]
good = np.where(grid_test > 4000)[0]

# load select profiles
df_den = pd.DataFrame(GD['Density'][:, good], index=GD['grid'][:], columns=GD['dive_list'][good])
df_theta = pd.DataFrame(GD['Theta'][:, good], index=GD['grid'][:], columns=GD['dive_list'][good])
df_ct = pd.DataFrame(GD['Conservative Temperature'][:, good], index=GD['grid'][:], columns=GD['dive_list'][good])
df_s = pd.DataFrame(GD['Absolute Salinity'][:, good], index=GD['grid'][:], columns=GD['dive_list'][good])
df_lon = pd.DataFrame(GD['Longitude'][:, good], index=GD['grid'][:], columns=GD['dive_list'][good])
df_lat = pd.DataFrame(GD['Latitude'][:, good], index=GD['grid'][:], columns=GD['dive_list'][good])
dac_u = GD.variables['DAC_u'][good]
dac_v = GD.variables['DAC_v'][good]
time_rec = GD.variables['time_start_stop'][good]
profile_list = np.float64(GD.variables['dive_list'][good]) - 35000
df_den[df_den < 0] = np.nan
df_theta[df_theta < 0] = np.nan
df_ct[df_ct < 0] = np.nan
df_s[df_s < 0] = np.nan
dac_u[dac_u < -500] = np.nan
dac_v[dac_v < -500] = np.nan


# interpolate nans that populate density profiles     
for i in range(len(profile_list)):
    fix = nanseg_interp(grid, np.array(df_den.iloc[:, i]))
    df_den.iloc[:, i] = fix
    fix_lon = nanseg_interp(grid, np.array(df_lon.iloc[:, i]))
    df_lon.iloc[:, i] = fix_lon
    fix_lat = nanseg_interp(grid, np.array(df_lat.iloc[:, i]))
    df_lat.iloc[:, i] = fix_lat

# extract DAC_U/V and lat/lon pos 
ev_oth = range(0, int(len(dac_u)-1), 2)
count = 0
dac_lon = np.nan * np.zeros(int(len(dac_u) / 2))
dac_lat = np.nan * np.zeros(int(len(dac_u) / 2))
d_u = np.nan * np.zeros(int(len(dac_u) / 2))
d_v = np.nan * np.zeros(int(len(dac_u) / 2))
d_time = np.nan * np.zeros(int(len(dac_u) / 2))
for p in ev_oth:
    dac_lon[count] = np.nanmean([df_lon.iloc[:, p], df_lon.iloc[:, p + 1]])
    dac_lat[count] = np.nanmean([df_lat.iloc[:, p], df_lat.iloc[:, p + 1]])
    d_u[count] = dac_u[p]
    d_v[count] = dac_v[p]
    d_time[count] = time_rec[p]
    count = count + 1

# -------------------------------------------------------------------------------
# ------------ compute correlations and covariance ####
# estimate covariance function from data
# for all pairs of points di and dj compute & store
# 1) distance between them (km)
# 2) time lag between them (days)
# 3) their product: di*dj    

compute_cor = 0
if compute_cor > 0:
    # pick density values on a depth level
    dl = 100
    t_int = np.where((time_rec < np.nanmedian(time_rec)))[0]
    dc = np.array(df_den.iloc[dl, t_int])
    d_lo = np.array(df_lon.iloc[dl, t_int])
    d_la = np.array(df_lat.iloc[dl, t_int])
    d_x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (d_lo - ref_lon)
    d_y = 1852 * 60 * (d_la - ref_lat)
    d_t = time_rec[t_int].copy()

    cx, cy, c0 = trend_fit(d_x, d_y, dc)
    den_anom_o = dc - (cx * d_x + cy * d_y + c0)
    dt = 10  # separation in time of points (up to 271 days )
    ds = 10000  # separation in space (up to 100km)
    Lt = 25
    Ls = 50000
    den_var, cov_est = data_covariance(den_anom_o, d_x, d_y, d_t, dt, ds, Ls, Lt)

# --------------------------------------------------

# Parameters for objective mapping
Lx = 35000
Ly = 35000
lon_grid = np.arange(-64.7, -63.55, .05, dtype=np.float)
lat_grid = np.arange(31.3, 32.0, .05, dtype=np.float)
x_grid = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_grid - ref_lon)
y_grid = 1852 * 60 * (lat_grid - ref_lat)

# select time window from which to extract subset for initial objective mapping 
win_size = 16
time_step = 4
t_windows = np.arange(np.floor(np.nanmin(time_rec)), np.floor(np.nanmax(time_rec)), time_step)
t_bin = np.nan * np.zeros((len(t_windows) - win_size, 2))
for i in range(len(t_windows) - win_size):
    t_bin[i, :] = [t_windows[i], t_windows[i] + win_size]

# --- LOOPING ###
# --- LOOPING ### over time windows
n_error = []
U_out = []
V_out = []
sigma_theta_out = []
sigma_theta_limits_out = []
sigma_theta_all = []
d_dx_sigma = []
d_dy_sigma = []
lon_out = []
lat_out = []
lon_all = []
lat_all = []
time_out = []
U_all = []
V_all = []
DAC_U_M = []
DAC_V_M = []
good_mask = []
sample_win = np.arange(18, 45, 1)   # started at 20
for k0 in range(np.size(sample_win)):
    k = sample_win[k0]
    k_out = k
    time_in = np.where((time_rec > t_bin[k, 0]) & (time_rec < t_bin[k, 1]))[0]  # data
    time_in_2 = np.where((d_time > t_bin[k, 0]) & (d_time < t_bin[k, 1]))[0]  # DAC

    # --- LOOPING
    # --- LOOPING  over depth layers
    sigma_theta = np.nan * np.zeros((len(lat_grid), len(lon_grid), len(grid)))
    sigma_theta_obs_limits = np.nan*np.zeros((len(grid), 2))
    error = np.nan * np.zeros((len(lat_grid), len(lon_grid), len(grid)))
    d_sigma_dx = np.nan * np.zeros((len(lat_grid), len(lon_grid), len(grid)))
    d_sigma_dy = np.nan * np.zeros((len(lat_grid), len(lon_grid), len(grid)))
    error_mask = np.zeros((len(lat_grid), len(lon_grid), len(grid_2)))
    norm_error = np.zeros((len(lat_grid), len(lon_grid), len(grid_2)))
    for l in range(len(grid_2)):
        depth = np.where(grid == grid[l])[0][0]
        lon_in = np.array(df_lon.iloc[depth, time_in])
        lat_in = np.array(df_lat.iloc[depth, time_in])
        den_in = np.array(df_den.iloc[depth, time_in])
        den_in_max = np.nanmax(den_in)
        den_in_min = np.nanmin(den_in)

        # attempt to correct for nan's 
        if len(np.where(np.isnan(den_in))[0]) > 0:
            den_up = np.array(df_den.iloc[depth - 1, time_in])
            den_down = np.array(df_den.iloc[depth + 1, time_in])
            lon_up = np.array(df_lon.iloc[depth - 1, time_in])
            lon_down = np.array(df_lon.iloc[depth + 1, time_in])
            lat_up = np.array(df_lat.iloc[depth - 1, time_in])
            lat_down = np.array(df_lat.iloc[depth + 1, time_in])
            bad = np.where(np.isnan(den_in))[0]
            for l0 in range(len(bad)):
                den_in[bad[l0]] = np.interp(grid[depth], [grid[depth - 1], grid[depth + 1]],
                                           [den_up[bad[l0]], den_down[bad[l0]]])
                lon_in[bad[l0]] = np.interp(grid[depth], [grid[depth - 1], grid[depth + 1]],
                                           [lon_up[bad[l0]], lon_down[bad[l0]]])
                lat_in[bad[l0]] = np.interp(grid[depth], [grid[depth - 1], grid[depth + 1]],
                                           [lat_up[bad[l0]], lat_down[bad[l0]]])

        # convert to x,y distance
        x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_in - ref_lon)
        y = 1852 * 60 * (lat_in - ref_lat)

        # Fit a trend to the data 
        cx, cy, c0 = trend_fit(x, y, den_in)
        den_anom = den_in - (cx * x + cy * y + c0)
        den_sig = np.nanstd(den_anom)
        den_var = np.nanvar(den_anom)
        errsq = 0
        noise_sig = (np.max(den_in) - np.min(den_in)) / 7.5  # 0.01

        # data-data covariance (loop over nXn data points) 
        npts = len(x)
        C = np.zeros((npts, npts), dtype=np.float)
        data_data = np.zeros((npts, npts), dtype=np.float)
        for l2 in range(npts):
            for k2 in range(npts):
                dxij = (x[l2] - x[k2])
                dyij = (y[l2] - y[k2])
                C[l2, k2] = den_var * np.exp(- (dxij * dxij) / (Lx * Lx) - (dyij * dyij) / (Ly * Ly))
                if k2 == l2:
                    data_data[l2, k2] = C[l2, k2] + noise_sig * noise_sig
                else:
                    data_data[l2, k2] = C[l2, k2]

        # loop over each grid point 
        Dmap = np.zeros((len(lat_grid), len(lon_grid)), dtype=np.float)
        Emap = np.zeros((len(lat_grid), len(lon_grid)), dtype=np.float)
        for j in range(len(lat_grid)):
            for i in range(len(lon_grid)):
                x_g = x_grid[i]
                y_g = y_grid[j]
                # data-grid
                data_grid = (den_var - errsq) * np.exp(-((x_g - x) / Lx) ** 2 - ((y_g - y) / Ly) ** 2)

                alpha = solve(data_data, data_grid)
                Dmap[j, i] = np.sum([den_anom * alpha]) + (cx * x_grid[i] + cy * y_grid[j] + c0)
                Emap[j, i] = np.sqrt(den_sig * den_sig - np.dot(data_grid, alpha))

        sigma_theta_obs_limits[l, :] = np.array([den_in_min, den_in_max])
        sigma_theta[:, :, l] = Dmap
        error[:, :, l] = Emap

        # # Create correlation matrix 
        # Ainv = createCorrelationMatrices(x,y,den_anom,data_sig,noise_sig,Lx,Ly)
        # # Map
        # Dmap = np.zeros((len(lat_grid),len(lon_grid)),dtype=np.float)
        # for j in range(len(lat_grid)):
        #     for i in range(len(lon_grid)):
        #         for n in range(len(lon_in)):
        #             dx0i = (x_grid[i]-x[n])   # (lon_grid[i]-lon_in[n])*111.0*xscale
        #             dy0i = (y_grid[j]-y[n])         # (lat_grid[i]-lat_in[n])*111.0
        #             C0i=data_sig*data_sig*np.exp(-(dx0i*dx0i)/(Lx*Lx)-(dy0i*dy0i)/(Ly*Ly))
        #             Dmap[j,i] = Dmap[j,i] + C0i*np.sum(Ainv[n,:]*den_anom)
        #         Dmap[j,i]=Dmap[j,i]+(cx*x_grid[i]+cy*y_grid[j]+c0)
        # sigma_theta[:,:,k] = Dmap

        d_sigma_dy[:, :, l], d_sigma_dx[:, :, l] = np.gradient(sigma_theta[:, :, l], y_grid[2] - y_grid[1],
                                                               x_grid[2] - x_grid[1])

        # MASK consider only profiles that are within a low error region
        # error values are plus/minus in units of the signal (in this case it density)
        # include only u/v profiles with density error less than 0.01
        # --- NORMALIZED ERROR
        # norm_error[:, :, l] = error[:, :, l] / (np.nanmax(sigma_theta[:, :, l]) - np.nanmin(sigma_theta[:, :, l]))
        norm_error[:, :, l] = error[:, :, l] / (sigma_theta_obs_limits[l, 1] - sigma_theta_obs_limits[l, 0])
        # sig_range = (np.nanmax(sigma_theta[:, :, l]) - np.nanmin(sigma_theta[:, :, l])) / 7
        # sig_range2 = 1 * (sigma_theta[:, :, l].std())
        # good = np.where(error[:, :, l] < sig_range)
        # error_mask[good[0], good[1], l] = 1

        # --- FINISH LOOPING OVER ALL DEPTH LAYERS
        # -------------------------------------------------------------------

    # normalized error averaged over the grid points to give one value per grid cell
    error_test = np.nanmean(norm_error, axis=2)
    good_prof = np.where(error_test <= 0.1)

    # mapping for DAC vectors 
    d_lon_in = dac_lon[time_in_2]
    d_lat_in = dac_lat[time_in_2]
    d_u_in = d_u[time_in_2]
    d_v_in = d_v[time_in_2]
    d_x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (d_lon_in - ref_lon)
    d_y = 1852 * 60 * (d_lat_in - ref_lat)
    # Fit a trend to the data (average DAC)
    mean_u = np.nanmean(d_u_in)
    mean_v = np.nanmean(d_v_in)
    u_anom = d_u_in - mean_u
    v_anom = d_v_in - mean_v
    d_u_sig = np.nanstd(u_anom)
    d_v_sig = np.nanstd(v_anom)
    noise_sig = 0.01
    # Create correlation matrix 
    Ainv_u = createCorrelationMatrices(d_x, d_y, u_anom, d_u_sig, noise_sig, Lx, Ly)
    Ainv_v = createCorrelationMatrices(d_x, d_y, v_anom, d_v_sig, noise_sig, Lx, Ly)
    # Map
    DACU_map = np.zeros((len(lat_grid), len(lon_grid)), dtype=np.float)
    DACV_map = np.zeros((len(lat_grid), len(lon_grid)), dtype=np.float)
    for j in range(len(lat_grid)):
        for i in range(len(lon_grid)):
            for n in range(len(d_lon_in)):
                dx0i = (x_grid[i] - d_x[n])  # (lon_grid[i]-lon_in[n])*111.0*xscale
                dy0i = (y_grid[j] - d_y[n])  # (lat_grid[i]-lat_in[n])*111.0
                C0i = d_v_sig * d_u_sig * np.exp(-(dx0i * dx0i) / (Lx * Lx) - (dy0i * dy0i) / (Ly * Ly))
                DACU_map[j, i] = DACU_map[j, i] + C0i * np.sum(Ainv_u[n, :] * u_anom)
                DACV_map[j, i] = DACV_map[j, i] + C0i * np.sum(Ainv_v[n, :] * v_anom)
            DACU_map[j, i] = DACU_map[j, i] + mean_u
            DACV_map[j, i] = DACV_map[j, i] + mean_v

    # GEOSTROPHIC SHEAR to geostrophic velocity 
    ff = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)
    du_dz = np.nan * np.zeros(np.shape(d_sigma_dy))
    dv_dz = np.nan * np.zeros(np.shape(d_sigma_dy))
    Ubc_g = np.nan * np.zeros(np.shape(d_sigma_dy))
    Vbc_g = np.nan * np.zeros(np.shape(d_sigma_dy))
    U_g = np.nan * np.zeros(np.shape(d_sigma_dy))
    V_g = np.nan * np.zeros(np.shape(d_sigma_dy))
    for m in range(len(lat_grid)):
        for n in range(len(lon_grid)):
            for o in range(len(grid_2)):
                du_dz[m, n, o] = (g / (rho0 * ff)) * d_sigma_dy[m, n, o]
                dv_dz[m, n, o] = (-g / (rho0 * ff)) * d_sigma_dx[m, n, o]

            iq = np.where(~np.isnan(du_dz[m, n, :]))
            if np.size(iq) > 10:
                z2 = -grid_2[iq]
                # u
                urel = cumtrapz(du_dz[m, n, iq], x=z2, initial=0)
                urel_av = np.trapz(urel / (z2[-1] - z2[0]), x=z2)
                ubc = urel - urel_av
                Ubc_g[m, n, iq] = ubc
                # v
                vrel = cumtrapz(dv_dz[m, n, iq], x=z2, initial=0)
                vrel_av = np.trapz(vrel / (z2[-1] - z2[0]), x=z2)
                vbc = vrel - vrel_av
                Vbc_g[m, n, iq] = vbc

                U_g[m, n, iq] = DACU_map[m, n] + ubc
                V_g[m, n, iq] = DACV_map[m, n] + vbc

    t_s = datetime.date.fromordinal(np.int(t_bin[k_out, 0]))
    t_e = datetime.date.fromordinal(np.int(t_bin[k_out, 1]))

    # --- OUTPUT
    # indices (profile,z_grid,)
    n_error.append(norm_error)
    U_out.append(U_g[good_prof[0], good_prof[1], :])
    V_out.append(V_g[good_prof[0], good_prof[1], :])
    sigma_theta_out.append(sigma_theta[good_prof[0], good_prof[1], :])
    sigma_theta_limits_out.append(sigma_theta_obs_limits)
    lon_out.append(lon_grid[good_prof[1]])
    lat_out.append(lat_grid[good_prof[0]])

    lon_all.append(lon_grid)
    lat_all.append(lat_grid)
    sigma_theta_all.append(sigma_theta)
    d_dx_sigma.append(d_sigma_dx)
    d_dy_sigma.append(d_sigma_dy)
    U_all.append(U_g)
    V_all.append(V_g)
    DAC_U_M.append(DACU_map)
    DAC_V_M.append(DACV_map)

    time_out.append(t_bin[k_out, :])
    good_mask.append(good_prof)
    print(str(k0))

# --- END LOOPING OVER EACH TIME WINDOW


# --- SAVE
# write python dict to a file
sa = 1
if sa > 0:
    mydict = {'depth': grid, 'Sigma_Theta': sigma_theta_out, 'U_g': U_out, 'V_g': V_out, 'lon_grid': lon_out,
              'lat_grid': lat_out, 'time': time_out,
              'Sigma_Theta_All': sigma_theta_all, 'U_g_All': U_all, 'V_g_All': V_all, 'lon_grid_All': lon_all,
              'lat_grid_All': lat_all, 'mask': good_mask, 'dac_u_map': DAC_U_M, 'dac_v_map': DAC_V_M,
              'd_sigma_dx': d_dx_sigma, 'd_sigma_dy': d_dy_sigma}
    output = open('/Users/jake/Documents/geostrophic_turbulence/BATS_obj_map_L35_W4_apr18.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()



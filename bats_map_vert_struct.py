# BATS (OBJECTIVE MAP OUTPUT)

import numpy as np
import matplotlib.pyplot as plt
import datetime
from netCDF4 import Dataset
import pandas as pd
import gsw
import pickle
# functions I've written 
from mode_decompositions import vertical_modes, PE_Tide_GM
from toolkit import plot_pro, nanseg_interp

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
f_ref = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)

# ---- gridded dive T/S
df_ct = pd.DataFrame(GD['Conservative Temperature'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
df_s = pd.DataFrame(GD['Absolute Salinity'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
df_ct[df_ct < 0] = np.nan
df_s[df_s < 0] = np.nan

# -------- mode parameters
omega = 0  # frequency zeroed for geostrophic modes
mmax = 60  # highest baroclinic mode to be calculated
nmodes = mmax + 1
deep_shr_max = 0.1  # maximum allowed deep shear [m/s/km]
deep_shr_max_dep = 3500  # minimum depth for which shear is limited [m]

# ---- LOAD IN TRANSECT TO PROFILE DATA COMPILED IN BATS_TRANSECTS.PY
pkl_file = open('/Users/jake/Desktop/bats/dep15_transect_profiles_apr04.pkl', 'rb')
bats_trans = pickle.load(pkl_file)
pkl_file.close()
Time_t = bats_trans['Time']
Info = bats_trans['Info']
Sigma_Theta_t = bats_trans['Sigma_Theta'][0:sz_g, :]
Eta = bats_trans['Eta'][0:sz_g, :]
Eta_theta = bats_trans['Eta_theta'][0:sz_g, :]
V_t = bats_trans['V'][0:sz_g, :]
prof_lon = bats_trans['V_lon']
prof_lat = bats_trans['V_lat']

# -- SOME VELOCITY PROFILES ARE TOO NOISY AND DEEMED UNTRUSTWORTHY
# select only velocity profiles that seem reasonable
# criteria are slope of v (dont want kinks)
# criteria: limit surface velocity to greater that 40cm/s
good_v = np.zeros(np.size(Time_t))
v_dz = np.zeros(np.shape(V_t))
for i in range(np.size(Time_t)):
    v_dz[5:-20, i] = np.gradient(V_t[5:-20, i], z[5:-20])
    if np.nanmax(np.abs(v_dz[:, i])) < 0.0015:  # 0.075
        good_v[i] = 1
good_v[191] = 1
good_ex = np.where(np.abs(V_t[5, :]) < 0.4)[0]
good_der = np.where(good_v > 0)[0]
good = np.intersect1d(good_der, good_ex)

Sigma_Theta_t2 = Sigma_Theta_t[:, good]
V_t2 = V_t[:, good].copy()
Eta2 = Eta[:, good].copy()
Eta_theta2 = Eta_theta[:, good].copy()
Time_t2 = Time_t[good].copy()
Info2 = Info[:, good].copy()
prof_lon2 = prof_lon[good].copy()
prof_lat2 = prof_lat[good].copy()
for i in range(len(Time_t2)):
    y_i = Eta2[:, i]
    if np.sum(np.isnan(y_i)) > 0:
        Eta2[:, i] = nanseg_interp(grid, y_i)

# -------- AVG background properties of profiles along these transects
sigma_theta_avg = df_den.mean(axis=1)
ct_avg = df_ct.mean(axis=1)
salin_avg = df_s.mean(axis=1)
ddz_avg_sigma = np.gradient(sigma_theta_avg, z)
N2 = np.nan * np.zeros(sigma_theta_avg.size)
N2[0:-1] = gsw.Nsquared(salin_avg, ct_avg, grid_p, lat=ref_lat)[0]
N2[-2:] = N2[-3]
N2[N2 < 0] = np.nan
N2 = nanseg_interp(grid, N2)
N = np.sqrt(N2)

# # - MODEL AVERAGE PROFILES
# sigma_theta_avg_map = np.nanmean(Sigma_Theta, axis=1)
# ddz_avg_sigma_map = np.gradient(sigma_theta_avg_map, z)

# N2_map = (-g / rho0) * ddz_avg_sigma_map
# lz = np.where(N2_map < 0)
# lnan = np.isnan(N2_map)
# N2_map[lz] = 0
# N2_map[lnan] = 0
# N_map = np.sqrt(N2_map)

# --- TRANSECT AVERAGE M/W profiles (FROM ALL PROFILES)
sigma_theta_avg_t = np.nanmean(Sigma_Theta_t2, axis=1)
ddz_avg_sigma_t = np.gradient(sigma_theta_avg_t, z)

N2_t = (-g / rho0) * ddz_avg_sigma_t
lz = np.where(N2_t < 0)
lnan = np.isnan(N2_t)
N2_t[lz] = 0
N2_t[lnan] = 0
N_t = np.sqrt(N2_t)

# ---- VERTICAL MODES
G, Gz, c = vertical_modes(N2, grid, omega, mmax)
G_t, Gz_t, c_t = vertical_modes(N2_t, grid, omega, mmax)

# presort V 
# good_v = np.zeros(np.size(Time))
# for i in range(np.size(Time)):
#     v_dz = np.gradient(V[10:,i])
#     if np.nanmax(np.abs(v_dz)) < 0.075:
#         good_v[i] = 1        
# good0 = np.intersect1d(np.where((np.abs(V[-45,:]) < 0.2))[0],np.where((np.abs(V[10,:]) < 0.4))[0])
# good = np.intersect1d(np.where(good_v > 0),good0)
# V2 = V[:,good]
# Eta2 = Eta[:,good]
# Eta_theta2 = Eta_theta[:,good]

# --------------------------------------------------------------------------------------------

# --- SATELLITE SURFACE TKE
SA = Dataset('/Users/jake/Desktop/bats/dataset-duacs-rep-global-merged-allsat-phy-l4-v3_1521132997576.nc', 'r')
lol = -65.2
lom = -63.3
lal = 31.1
lam = 32.4
sa_lon = SA.variables['longitude'][:] - 360
sa_lon_2 = sa_lon[(sa_lon >= lol) & (sa_lon <= lom)]
sa_lat = SA.variables['latitude'][:]
sa_lat_2 = sa_lat[(sa_lat >= lal) & (sa_lat <= lam)]
SA_x_grid = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (sa_lon_2 - ref_lon)
SA_y_grid = 1852 * 60 * (sa_lat_2 - ref_lat)
SA_x1, SA_y1 = np.meshgrid(SA_x_grid / 1000, SA_y_grid / 1000)
SA_lon, SA_lat = np.meshgrid(sa_lon_2, sa_lat_2)
# SA time
sa_time = SA.variables['time'][:] + 711858  # 1950-01-01      datetime.date.fromordinal(np.int(Time[i]))
sa_time_dt = []
for i in range(len(sa_time)):
    sa_time_dt.append(datetime.date.fromordinal(np.int(sa_time[i])))

# ---- LOAD PE/KE from DG 2015 estimates
pkl_file = open('/Users/jake/Documents/geostrophic_turbulence/BATs_DG_2015_energy.pkl', 'rb')
bats_map = pickle.load(pkl_file)
pkl_file.close()
bdg_ke = bats_map['KE']
bdg_pe = bats_map['PE']
bdg_c = bats_map['c'][:]
bdg_f = bats_map['f']

# -------- LOAD IN MAPPING
pkl_file = open('/Users/jake/Documents/geostrophic_turbulence/BATS_obj_map_L35_apr05.pkl', 'rb')
bats_map = pickle.load(pkl_file)
pkl_file.close()

# --- iterate over time windows (looking into KE per window (how does energy vary in time?)
AGz_U_out = []
AGz_V_out = []
AGz_t_out = []
good_prof_u_out = []
good_prof_v_out = []
good_prof_vt_out = []
KE_i_out = []
lr = range(15, 16)
avg_KE_U = np.nan*np.zeros((len(lr), nmodes))
avg_KE_V = np.nan*np.zeros((len(lr), nmodes))
avg_KE = np.nan*np.zeros((len(lr), nmodes))
avg_KE_V_t = np.nan*np.zeros((len(lr), nmodes))
TKE_surface_map_out = []
TKE_surface_out = []
SAT_STKE_out = []
avg_surf_KE_m0 = []
avg_surf_KE_m1 = []
avg_surf_KE_m2 = []
prof_x3_out = []
prof_y3_out = []
x1_out = []
y1_out = []
for m in range(len(lr)):
    map_t_itera = lr[m]
    Time = bats_map['time'][map_t_itera]
    t_s = datetime.date.fromordinal(np.int(Time[0]))
    t_e = datetime.date.fromordinal(np.int(Time[1]))
    Sigma_Theta = np.transpose(bats_map['Sigma_Theta'][map_t_itera][:, 0:sz_g])
    sigma_theta_all = np.transpose(bats_map['Sigma_Theta_All'][map_t_itera][:, 0:sz_g])
    U = np.transpose(bats_map['U_g'][map_t_itera][:, 0:sz_g])
    V = np.transpose(bats_map['V_g'][map_t_itera][:, 0:sz_g])
    U_all = np.array(bats_map['U_g_All'])
    V_all = np.array(bats_map['V_g_All'])
    lat_grid_all = np.array(bats_map['lat_grid_All'])
    lon_grid_all = np.array(bats_map['lon_grid_All'])
    mask = bats_map['mask']

    # --- select transect profiles that fall within this time window
    V_t3 = V_t2[:, ((Time_t2 > Time[0]) & (Time_t2 < Time[1]))]
    Sigma_Theta_t3 = Sigma_Theta_t2[:, ((Time_t2 > Time[0]) & (Time_t2 < Time[1]))]
    prof_x3 = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (prof_lon2[(Time_t2 > Time[0]) & (Time_t2 < Time[1])] - ref_lon)
    prof_y3 = 1852 * 60 * (prof_lat2[(Time_t2 > Time[0]) & (Time_t2 < Time[1])] - ref_lat)

    # --- FIND X,Y POSITION OF MAPPED GRID POINTS
    x_grid = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon_grid_all[map_t_itera, :] - ref_lon)
    y_grid = 1852 * 60 * (lat_grid_all[map_t_itera, :] - ref_lat)
    x1, y1 = np.meshgrid(x_grid / 1000, y_grid / 1000)

    sz = np.shape(V)
    num_profs = sz[1]
    eta_fit_depth_min = 50
    eta_fit_depth_max = 3800
    eta_theta_fit_depth_max = 4200
    AG = np.zeros([nmodes, num_profs])
    AGz_U = np.zeros([nmodes, num_profs])
    AGz_V = np.zeros([nmodes, num_profs])
    U_m = np.nan * np.zeros([np.size(grid), num_profs])
    V_m = np.nan * np.zeros([np.size(grid), num_profs])
    HKE_U_per_mass = np.nan * np.zeros([nmodes, num_profs])
    HKE_V_per_mass = np.nan * np.zeros([nmodes, num_profs])
    # PE_theta_per_mass = np.nan*np.zeros([nmodes, num_profs])
    modest = np.arange(11, nmodes)
    good_prof_u = np.ones(num_profs)
    good_prof_v = np.ones(num_profs)
    HKE_noise_threshold = 1e-4  # 1e-5
    for i in range(num_profs):
        # fit to velocity profiles
        this_U = U[:, i].copy()
        this_V = V[:, i].copy()
        iu = np.where(~np.isnan(this_U))
        iv = np.where(~np.isnan(this_V))
        # U PROFILES
        if iu[0].size > 1:
            # AGz = Gz(iv,:)\V_g(iv,ip)
            AGz_U[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(Gz[iu, :]), np.transpose(np.atleast_2d(this_U[iu])))[0])
            # U_m =  Gz*AGz[:,i];
            U_m[:, i] = np.squeeze(np.matrix(Gz) * np.transpose(np.matrix(AGz_U[:, i])))
            HKE_U_per_mass[:, i] = AGz_U[:, i] * AGz_U[:, i]
            ival = np.where(HKE_U_per_mass[modest, i] >= HKE_noise_threshold)
            if np.size(ival) > 0:
                good_prof_u[i] = 0  # flag profile as noisy
        else:
            good_prof_u[i] = 0  # flag empty profile as noisy as well
        # V PROFILES
        if iv[0].size > 1:
            AGz_V[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(Gz[iv, :]), np.transpose(np.atleast_2d(this_V[iv])))[0])
            V_m[:, i] = np.squeeze(np.matrix(Gz) * np.transpose(np.matrix(AGz_V[:, i])))
            HKE_V_per_mass[:, i] = AGz_V[:, i] * AGz_V[:, i]
            ival = np.where(HKE_V_per_mass[modest, i] >= HKE_noise_threshold)
            if np.size(ival) > 0:
                good_prof_v[i] = 0  # flag profile as noisy
        else:
            good_prof_v[i] = 0  # flag empty profile as noisy as well

    # --- TOTAL ENERGY
    # --- total water column KE for each profile
    TKE_per_mass = (1 / 2) * (HKE_U_per_mass + HKE_V_per_mass)
    KE_i = np.nan*np.zeros((num_profs, len(grid)))
    for i in range(num_profs):
        um = np.nan*np.zeros((len(grid), nmodes))
        vm = np.nan * np.zeros((len(grid), nmodes))
        for j in range(len(grid)):
            # u^2 at each dpth with contribution of each mode at that dpth
            um[j, :] = (AGz_U[:, i] ** 2) * (Gz[j, :] ** 2)
            # v^2 at each dpth with contribution of each mode at that dpth
            vm[j, :] = (AGz_V[:, i] ** 2) * (Gz[j, :] ** 2)
        # total KE for each profile broken down by contribution from each mode at each depth
        KE_i[i, :] = (1 / 2) * np.sum(um + vm, axis=1)
    TKE = np.trapz(KE_i, grid, axis=1)  # TKE per profile
    # TKE = (1 / 2) * np.sum(HKE_U_per_mass + HKE_V_per_mass, axis=0)

    # --- SURFACE KE
    z_surf = 0
    u_sum = np.nan * np.zeros(num_profs)
    v_sum = np.nan * np.zeros(num_profs)
    # sum the KE from each mode at each profile location
    for i in range(num_profs):
        u_sum[i] = np.sum((AGz_U[:, i] ** 2) * (Gz[z_surf, :] ** 2))
        v_sum[i] = np.sum((AGz_V[:, i] ** 2) * (Gz[z_surf, :] ** 2))
    # --- AVGERAGE TKE at z_surf at each location
    dk = f_ref / c[1]
    TKE_surface = (1 / 2) * (u_sum + v_sum)
    # -- fraction of TKE
    TKE_frac = TKE_surface * (grid[1] - grid[0]) / TKE
    # -- fractions in modes (at surface)
    mode0 = 0
    TKE_surface_mode0 = (1 / 2) * (
            (AGz_U[mode0, :] ** 2) * (Gz[z_surf, mode0] ** 2) + (AGz_V[mode0, :] ** 2) * (Gz[z_surf, mode0] ** 2))
    mode1 = 1
    TKE_surface_mode1 = (1 / 2) * (
            (AGz_U[mode1, :] ** 2) * (Gz[z_surf, mode1] ** 2) + (AGz_V[mode1, :] ** 2) * (Gz[z_surf, mode1] ** 2))
    mode2 = 2
    TKE_surface_mode2 = (1 / 2) * (
            (AGz_U[mode2, :] ** 2) * (Gz[z_surf, mode2] ** 2) + (AGz_V[mode2, :] ** 2) * (Gz[z_surf, mode2] ** 2))

    # --- ALTIMETERY
    sa_u = SA.variables['ugos'][:, (sa_lat >= lal) & (sa_lat <= lam), (sa_lon >= lol) & (sa_lon <= lom)]
    sa_v = SA.variables['vgos'][:, (sa_lat >= lal) & (sa_lat <= lam), (sa_lon >= lol) & (sa_lon <= lom)]
    this_time = np.mean(Time)  # this is the avg of the interpolating window within which we map a map of density (15 days)
    sa_t_choice = (np.abs(sa_time - this_time)).argmin()
    SA_U_in = sa_u[sa_t_choice, :, :]
    SA_V_in = sa_v[sa_t_choice, :, :]
    SA_TKE = (1 / 2) * (SA_U_in ** 2 + SA_V_in ** 2)
    SA_TKE = (1 / 2) * (SA_U_in ** 2 + SA_V_in ** 2)

    # mask grid for plotting MAP
    test_lev = 0
    U_i = np.nan * np.zeros(np.shape(U_all[:, :, :, test_lev]))
    V_i = np.nan * np.zeros(np.shape(V_all[:, :, :, test_lev]))
    U_i[map_t_itera, mask[map_t_itera][0], mask[map_t_itera][1]] = U_all[
        map_t_itera, mask[map_t_itera][0], mask[map_t_itera][1], test_lev]
    V_i[map_t_itera, mask[map_t_itera][0], mask[map_t_itera][1]] = V_all[
        map_t_itera, mask[map_t_itera][0], mask[map_t_itera][1], test_lev]
    TKE_surface_map = (1 / 2) * (U_i[map_t_itera, :, :] ** 2 + V_i[map_t_itera, :, :] ** 2)

    # --- VERTICAL STRUCTURE OF UN-MAPPED TRANSECT VELOCITY PROFILES
    AGz_t = np.zeros([nmodes, V_t3.shape[1]])
    V_t_m = np.nan * np.zeros([np.size(grid), V_t3.shape[1]])
    HKE_V_t_per_mass = np.nan * np.zeros([nmodes, V_t3.shape[1]])
    good_prof_vt = np.ones(V_t3.shape[1])
    for i in range(V_t3.shape[1]):
        # fit to velocity profiles
        this_vv = V_t3[:, i].copy()
        ivv = np.where(~np.isnan(this_vv))
        if ivv[0].size > 1:
            AGz_t[:, i] = np.squeeze(
                np.linalg.lstsq(np.squeeze(Gz_t[ivv, :]), np.transpose(np.atleast_2d(this_vv[ivv])))[0])
            V_t_m[:, i] = np.squeeze(np.matrix(Gz_t) * np.transpose(np.matrix(AGz_t[:, i])))
            HKE_V_t_per_mass[:, i] = AGz_t[:, i] * AGz_t[:, i]
            ival = np.where(HKE_V_t_per_mass[modest, i] >= HKE_noise_threshold)
            if np.size(ival) > 0:
                good_prof_vt[i] = 0  # flag profile as noisy
        else:
            good_prof_vt[i] = 0  # flag empty profile as noisy as well

    KE_i_out.append(KE_i)
    SAT_STKE_out.append(SA_TKE)                                                           # satellite
    avg_KE_U[m, :] = np.nanmean(HKE_U_per_mass[:, np.where(good_prof_u > 0)[0]], 1)       # map U
    avg_KE_V[m, :] = np.nanmean(HKE_V_per_mass[:, np.where(good_prof_v > 0)[0]], 1)       # map V
    avg_KE[m, :] = np.nanmean(TKE_per_mass[:, np.where(good_prof_v > 0)[0]], 1)           # total map
    avg_KE_V_t[m, :] = np.nanmean(HKE_V_t_per_mass[:, np.where(good_prof_vt > 0)[0]], 1)  # DG V
    avg_surf_KE_m0.append(TKE_surface_mode0)
    avg_surf_KE_m1.append(TKE_surface_mode1)
    avg_surf_KE_m2.append(TKE_surface_mode2)
    TKE_surface_map_out.append(TKE_surface_map)
    TKE_surface_out.append(TKE_surface)
    AGz_U_out.append(AGz_U)
    AGz_V_out.append(AGz_V)
    AGz_t_out.append(AGz_t)
    good_prof_u_out.append(good_prof_u)
    good_prof_v_out.append(good_prof_v)
    good_prof_vt_out.append(good_prof_vt)
    prof_x3_out.append(prof_x3)
    prof_y3_out.append(prof_y3)
    x1_out.append(x1)
    y1_out.append(y1)

# from time windows select one to plot/evaluate
inn = 0
AGz_U = AGz_U_out[inn]
AGz_V = AGz_V_out[inn]
AGz_t = AGz_t_out[inn]
good_prof_u = good_prof_u_out[inn]
good_prof_v = good_prof_v_out[inn]
good_prof_vt = good_prof_vt_out[inn]
KE_i = KE_i_out[inn]
TKE = np.trapz(KE_i, grid, axis=1)  # integrate KE in depth
SA_TKE = SAT_STKE_out[inn]
TKE_surface = TKE_surface_out[inn]
TKE_surface_map = TKE_surface_map_out[inn]
TKE_surface_mode0 = avg_surf_KE_m0[inn]
TKE_surface_mode1 = avg_surf_KE_m1[inn]
TKE_surface_mode2 = avg_surf_KE_m2[inn]
prof_x3 = prof_x3_out[inn]
prof_y3 = prof_y3_out[inn]
x1 = x1_out[inn]
y1 = y1_out[inn]

# --- prep for energy spectra
dk = f_ref / c[1]
sc_x = 1000 * f_ref / c[1:]
dk_t = f_ref / c_t[1]
sc_x_t = 1000 * f_ref / c_t[1:]
vert_wavenumber = f_ref / c[1:]

PE_SD, PE_GM = PE_Tide_GM(rho0, grid, nmodes, np.transpose(np.atleast_2d(N2)), f_ref)

# ------------ EOF SHAPES ------------------
# ----- EOFs of dynamic horizontal current (U) mode amplitudes
AGzq = AGz_U  # (:,quiet_prof)
nq = np.size(good_prof_u)  # good_prof and dg_good_prof
avg_AGzq = np.nanmean(np.transpose(AGzq), axis=0)
AGzqa = AGzq - np.transpose(np.tile(avg_AGzq, [nq, 1]))  # mode amplitude anomaly matrix
cov_AGzqa = (1 / nq) * np.matrix(AGzqa) * np.matrix(np.transpose(AGzqa))  # nmodes X nmodes covariance matrix
var_AGzqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa)))) * np.matrix(np.sqrt(np.diag(cov_AGzqa)))
cor_AGzqa = cov_AGzqa / var_AGzqa  # nmodes X nmodes correlation matrix
D_AGzqa, V_AGzqa = np.linalg.eig(cov_AGzqa)  # columns of V_AGzqa are eigenvectors
EOFseries_U = np.transpose(V_AGzqa) * np.matrix(AGzqa)  # EOF "timeseries' [nmodes X nq]
EOFshape_U = np.matrix(Gz) * V_AGzqa  # depth shape of eigenfunctions [ndepths X nmodes]

# ----- EOFs of dynamic horizontal current (V) mode amplitudes
AGzq = AGz_V  # (:,quiet_prof)
nq = np.size(good_prof_v)  # good_prof and dg_good_prof
avg_AGzq = np.nanmean(np.transpose(AGzq), axis=0)
AGzqa = AGzq - np.transpose(np.tile(avg_AGzq, [nq, 1]))  # mode amplitude anomaly matrix
cov_AGzqa = (1 / nq) * np.matrix(AGzqa) * np.matrix(np.transpose(AGzqa))  # nmodes X nmodes covariance matrix
var_AGzqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa)))) * np.matrix(np.sqrt(np.diag(cov_AGzqa)))
cor_AGzqa = cov_AGzqa / var_AGzqa  # nmodes X nmodes correlation matrix
D_AGzqa, V_AGzqa = np.linalg.eig(cov_AGzqa)  # columns of V_AGzqa are eigenvectors
EOFseries_V = np.transpose(V_AGzqa) * np.matrix(AGzqa)  # EOF "timeseries' [nmodes X nq]
EOFshape_V = np.matrix(Gz) * V_AGzqa  # depth shape of eigenfunctions [ndepths X nmodes]
EOFshape1_BTpBC1 = np.matrix(Gz[:, 0:2]) * V_AGzqa[0:2, 0]  # truncated 2 mode shape of EOF#1
EOFshape2_BTpBC1 = np.matrix(Gz[:, 0:2]) * V_AGzqa[0:2, 1]  # truncated 2 mode shape of EOF#2

# ----- EOFs of dynamic horizontal current (V_t) mode amplitudes from DG M/W transects
AGzq_t = AGz_t  # (:,quiet_prof)
nq = np.size(good_prof_vt)  # good_prof and dg_good_prof
avg_AGzq_t = np.nanmean(np.transpose(AGzq_t), axis=0)
AGzqa_t = AGzq_t - np.transpose(np.tile(avg_AGzq_t, [nq, 1]))  # mode amplitude anomaly matrix
cov_AGzqa_t = (1 / nq) * np.matrix(AGzqa_t) * np.matrix(np.transpose(AGzqa_t))  # nmodes X nmodes covariance matrix
var_AGzqa_t = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa_t)))) * np.matrix(np.sqrt(np.diag(cov_AGzqa_t)))
cor_AGzqa_t = cov_AGzqa_t / var_AGzqa_t  # nmodes X nmodes correlation matrix
D_AGzqa_t, V_AGzqa_t = np.linalg.eig(cov_AGzqa_t)  # columns of V_AGzqa are eigenvectors
EOFseries_t = np.transpose(V_AGzqa) * np.matrix(AGzqa)  # EOF "timeseries' [nmodes X nq]
EOFshape_Vt = np.matrix(Gz_t) * V_AGzqa_t  # depth shape of eigenfunctions [ndepths X nmodes]

# --- PLOT fraction of TKE throughout the water column
plot_tke_frac = 0
if plot_tke_frac > 0:
    dp_w = [0, 40, 65, 90, 115, 140, 165, 190, 215, (len(grid)-1)]
    KE_per = np.nan*np.zeros((num_profs, len(grid)))
    f, ax = plt.subplots()
    for m in range(num_profs):
        for i in range(2, len(grid)):  # range(2, len(dp_w)):
            KE_per[m, i] = np.trapz(KE_i[m, 0:i], grid[0:i])/TKE[m]
            # KE_per[i] = 100*np.trapz(KE_i[0, dp_w[i-1]:dp_w[i]], grid[dp_w[i-1]:dp_w[i]])/TKE[0]
        ax.plot(KE_per[m, :], grid)
    ax.invert_yaxis()
    ax.set_ylabel('Depth')
    ax.set_xlabel('Percent of TKE at Depths < This Depth')
    ax.set_title('Cumulative Percentage of TKE')
    plot_pro(ax)

# ---- PLOT SURFACE KE AND COMPARE TO SATELLITE
plot_TKE_surf = 1
if plot_TKE_surf > 0:
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    cmp = plt.cm.get_cmap("viridis")
    cmp.set_over('w')  # ('#E6E6E6')
    cmp.set_under('w')
    TKE_low = 0
    TKE_high = .09
    im1 = ax1.pcolor(SA_x1, SA_y1, SA_TKE, vmin=TKE_low, vmax=TKE_high, cmap=cmp)
    im2 = ax2.pcolor(x1, y1, TKE_surface_map, vmin=TKE_low, vmax=TKE_high, cmap=cmp)
    ax2.scatter(prof_x3/1000, prof_y3/1000, s=5, color='r')
    ax2.text(-25, 50, str(t_s) + ' -- ' + str(t_e))
    ax2.text(-60, -40, 'Mode 0 Fraction: ' + str(np.round(100*TKE_surface_mode0.mean()/TKE_surface.mean(), 2)) + '%')
    ax2.text(-60, -50, 'Mode 1 Fraction: ' + str(np.round(100*TKE_surface_mode1.mean()/TKE_surface.mean(), 2)) + '%')
    ax2.text(-60, -60, 'Mode 2 Fraction: ' + str(np.round(100*TKE_surface_mode2.mean()/TKE_surface.mean(), 2)) + '%')
    ax1.set_title('Altimetric Surface Kinetic Energy ')
    ax2.set_title('Mapped DG Surface Kinetic Energy ')
    c1 = plt.colorbar(im1, ax=ax1, orientation='horizontal')
    c1.set_label(r'Surface KE [m$^2$/s$^2$]')
    c2 = plt.colorbar(im2, ax=ax2, orientation='horizontal')
    c2.set_label(r'Surface KE [m$^2$/s$^2$]')
    ax1.set_xlabel('X [km]')
    ax2.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_xlim([-75, 75])
    ax1.set_ylim([-70, 70])
    ax2.set_xlim([-75, 75])
    ax2.set_ylim([-70, 70])
    plot_pro(ax2)

# ---- PLOT ETA / EOF OF PROFILES FROM TRANSECTS AND FROM MAPPING THAT FALL WITHIN THIS TIME WINDOW
plot_eta = 0
if plot_eta > 0:
    f, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, sharey=True)
    for j in range(num_profs):
        ax0.plot(U[:, j], grid, color='#CD853F', linewidth=1.25)
        ax0.plot(U_m[:, j], grid, color='k', linestyle='--', linewidth=.75)
        ax1.plot(V[:, j], grid, color='#CD853F', linewidth=1.25)
        ax1.plot(V_m[:, j], grid, color='k', linestyle='--', linewidth=.75)
    ax0.text(190, 800, str(num_profs) + ' Profiles')
    ax0.axis([-.4, .4, 0, 5000])
    ax1.axis([-.4, .4, 0, 5000])
    ax0.set_title(
        'Map U (' + np.str(t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day) + ')')
    ax1.set_title("Map V")
    ax0.set_ylabel('Depth [m]', fontsize=14)
    ax0.set_xlabel('u [m/s]', fontsize=14)
    ax1.set_xlabel('v [m/s]', fontsize=14)
    ax0.invert_yaxis()
    ax0.grid()
    ax1.grid()
    for l in range(V_t3.shape[1]):
        ax2.plot(V_t3[:, l], grid, color='k')
    ax2.axis([-.4, .4, 0, 5000])
    ax2.invert_yaxis()
    ax2.set_title('DG Transect V')
    ax2.set_xlabel('v [m/s]', fontsize=14)
    ax2.grid()
    # plot_pro(ax2)
    # f.savefig('/Users/jake/Desktop/bats/dg035_15_Eta_a.png',dpi = 300)
    # plt.show()    

    max_plot = 3
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # n2p = ax3.plot((np.sqrt(N2) * (1800 / np.pi)), grid, color='k', label='N(z) [cph]')
    colors = plt.cm.Dark2(np.arange(0, 4, 1))
    for ii in range(max_plot):
        # ax3.plot(Gz[:, ii], grid, color='#2F4F4F', linestyle='--')
        p_eof_u = ax3.plot(-EOFshape_U[:, ii], grid, color=colors[ii, :], label='EOF # = ' + str(ii + 1), linewidth=2)
        if ii > 0:
            p_eof_v = ax3.plot(EOFshape_V[:, ii], grid, color=colors[ii, :],
                                linewidth=2, linestyle='--')  # , label='EOF # = ' + str(ii + 1))
        else:
            p_eof_v = ax3.plot(-EOFshape_V[:, ii], grid, color=colors[ii, :],
                                linewidth=2, linestyle='--')  # , label='EOF # = ' + str(ii + 1))
        ax4.plot(Gz[:, ii], grid, color='#2F4F4F', linestyle='--')
        p_eof_vt = ax4.plot(-EOFshape_Vt[:, ii], grid, color=colors[ii, :], label='EOF # = ' + str(ii + 1), linewidth=2)

    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, fontsize=10)
    ax3.axis([-4, 4, 0, 5000])
    ax3.invert_yaxis()
    ax3.grid()
    ax3.set_title('Map U,V Mode Shapes')
    ax3.set_xlabel('U,V Mode Shapes (Map)')
    ax4.set_title('DG Mode Shapes')
    ax4.set_xlabel('V_t Mode Shapes (DG)')
    plot_pro(ax4)
    # END PLOTTING

# -- PLOT ENERGY SPECTRA
plot_eng = 0
if plot_eng > 0:
    fig0, ax0 = plt.subplots()
    # PE_p = ax0.plot(sc_x,avg_PE[1:]/dk,color='#B22222',label='PE',linewidth=1.5)
    KE_p = ax0.plot(1000 * f_ref / c, avg_KE / dk, 'k', label=r'KE$_{u_{map}}$', linewidth=2)
    ax0.scatter(1000 * f_ref / c, avg_KE / dk, color='k', s=20)  # map KE
    # KE_p = ax0.plot(sc_x, avg_KE_U[1:] / dk, 'g', label='KE_u_map', linewidth=1.5)
    # ax0.scatter(sc_x, avg_KE_U[1:] / dk, color='g', s=10)  # map KE
    # KE_p = ax0.plot(sc_x, avg_KE_V[1:] / dk, 'r', label='KE_v_map', linewidth=1.5)
    # ax0.scatter(sc_x, avg_KE_V[1:] / dk, color='r', s=10)  # map KE
    # KE_p = ax0.plot(sc_x_t, avg_KE_V_t[1:] / dk_t, 'k', label='KE_trans', linewidth=1.5)
    # ax0.scatter(sc_x_t, avg_KE_V_t[1:] / dk_t, color='k', s=10)  # DG KE
    KE_p = ax0.plot(1000 * bdg_f / bdg_c, bdg_ke / (bdg_f / bdg_c[1]), 'g', label='KE$_{trans}$', linewidth=2)
    ax0.scatter(1000 * bdg_f / bdg_c, bdg_ke / (bdg_f / bdg_c[1]), color='g', s=20)  # DG KE
    # PE_p = ax0.plot(1000 * bdg_f / bdg_c[1:], bdg_pe[1:] / (bdg_f / bdg_c[1]), 'b', label='PE$_{trans}$', linewidth=1.5)
    # ax0.scatter(1000 * bdg_f / bdg_c[1:], bdg_pe[1:] / (bdg_f / bdg_c[1]), color='b', s=10)  # DG KE

    # limits/scales
    ax0.plot([3 * 10 ** -1, 3 * 10 ** 0], [1.5 * 10 ** 1, 1.5 * 10 ** -2], color='k', linewidth=0.75)
    # ax0.plot([3 * 10 ** -2, 3 * 10 ** -1],
    #          [7 * 10 ** 2, ((5 / 3) * (np.log10(2 * 10 ** -1) - np.log10(2 * 10 ** -2)) + np.log10(7 * 10 ** 2))],
    #          color='k', linewidth=0.75)
    ax0.text(3.3 * 10 ** -1, 1.3 * 10 ** 1, '-3', fontsize=10)
    # ax0.text(3.3 * 10 ** -2, 6 * 10 ** 2, '-5/3', fontsize=10)
    # ax0.plot([1000 * f_ref / c[1], 1000 * f_ref / c[-2]], [1000 * f_ref / c[1], 1000 * f_ref / c[-2]], linestyle='--',
    #          color='k', linewidth=0.8)

    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.axis([10 ** -2, 10 ** 1, 1 * 10 ** (-4), 2 * 10 ** 3])
    ax0.set_xlabel(r'Scaled Vertical Wavenumber = (Rossby Radius)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=14)
    ax0.set_ylabel('Spectral Density, Hor. Wavenumber', fontsize=14)  # ' (and Hor. Wavenumber)')
    ax0.set_title(
        'Energy Map, Transect (' + np.str(t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(
            t_e.day) + ')', fontsize=14)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, fontsize=12)
    plt.tight_layout()
    plot_pro(ax0)
    # fig0.savefig('/Users/jake/Desktop/bats/dg035_15_PE_b.png',dpi = 300)
    # plt.close()
    # plt.show()

    # --- SAVE
    # write python dict to a file
    sa = 0
    if sa > 0:
        mydict = {'sc_x': sc_x, 'avg_ke_u': avg_KE_U, 'avg_ke_v': avg_KE_V, 'dk': dk}
        output = open('/Users/jake/Documents/geostrophic_turbulence/BATS_OM_KE.pkl', 'wb')
        pickle.dump(mydict, output)
        output.close()


# todo read wunsch comparisons with altimetry and compare his results to time (do satellites agree with my data?)
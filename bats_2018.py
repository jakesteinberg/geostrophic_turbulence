import numpy as np
import scipy.io as si
import matplotlib
import matplotlib.pyplot as plt
import gsw
import scipy
from netCDF4 import Dataset
import pickle
import datetime
from scipy.optimize import fmin
from scipy.signal import savgol_filter
# functions I've written
from glider_cross_section import Glider
from mode_decompositions import eta_fit, vertical_modes, PE_Tide_GM, vertical_modes_f
from toolkit import spectrum_fit, nanseg_interp, plot_pro, find_nearest


def functi_1(p, xe, xb):
    #  This is the target function that needs to be minimized
    fsq = (xe - p*xb)**2
    return fsq.sum()


def functi_2(p, xe, xb, xs):
    #  This is the target function that needs to be minimized
    fsq = (xe - (p[0] * xb + p[1] * xs)) ** 2
    # fsq = (xe - p*xb)**2
    return fsq.sum()


# --- PHYSICAL PARAMETERS
g = 9.81
rho0 = 1025  # - 1027
ref_lat = 31.7
ref_lon = 64.2

# --- MODE PARAMETERS
# frequency zeroed for geostrophic modes
omega = 0
# highest baroclinic mode to be calculated
mmax = 40
nmodes = mmax + 1
# maximum allowed deep shear [m/s/km]
deep_shr_max = 0.1
# minimum depth for which shear is limited [m]
deep_shr_max_dep = 3500
# ----------------------------------------------------------------------------------------------------------------------
# ---- PROCESSING USING GLIDER PACKAGE
gs = 65
ge = 149
x = Glider(41, np.arange(gs, ge), '/Users/jake/Documents/baroclinic_modes/DG/BATS_2018/sg041')
# ----------------------------------------------------------------------------------------------------------------------
import_dg = si.loadmat('/Users/jake/Documents/baroclinic_modes/sg041_2018_neutral_density_bin.mat')
dg_data = import_dg['out']
limm = 471
profile_tags = dg_data['prof_number'][0][0][0]
if profile_tags[0] == gs:
    first = 0
else:
    first = np.where(profile_tags < gs)[0][-1] + 1
if profile_tags[-1] == ge - 0.5:
    last = len(profile_tags)
else:
    last = np.where(profile_tags > ge)[0][0] - 1
d_in = range(first, last)
profile_tags = profile_tags[d_in]
neutral_density = dg_data['Neut_den'][0][0][0:limm, d_in]
t = dg_data['Temp'][0][0][0:limm, d_in]
s = dg_data['Sal'][0][0][0:limm, d_in]
lon = dg_data['Lon'][0][0][0:limm, d_in]
lat = dg_data['Lat'][0][0][0:limm, d_in]
dac_u = dg_data['Dac_u'][0][0][0][d_in]
dac_v = dg_data['Dac_v'][0][0][0][d_in]
d_time = dg_data['Time'][0][0][0:limm, d_in] - 366
ref_lat = np.nanmean(lat)
time_rec_bin = np.nanmean(d_time, axis=0)

grid = dg_data['Depth'][0][0][0:limm, 0]
grid_p = gsw.p_from_z(-1 * grid, ref_lat)
z = -1 * grid
sz_g = grid.shape[0]
# ----------------------------------------------------------------------------------------------------------------------
# -- Compute density
sa, ct, theta, sig0, sig2, dg_N2 = x.density(grid, ref_lat, t, s, lon, lat)
# ----------------------------------------------------------------------------------------------------------------------
t_s = datetime.date.fromordinal(np.int(np.nanmin(d_time)))
t_e = datetime.date.fromordinal(np.int(np.nanmax(d_time)))
salin_avg = np.nanmean(sa, axis=1)
cons_t_avg = np.nanmean(ct, axis=1)
theta_avg = np.nanmean(theta, axis=1)
sigma_theta_avg = np.nanmean(neutral_density, axis=1)
ddz_avg_sigma = np.gradient(sigma_theta_avg, z)
N2 = np.nan * np.zeros(sigma_theta_avg.shape)
go = ~np.isnan(salin_avg)
N2[np.where(go)[0][0:-1]] = gsw.Nsquared(salin_avg[go], cons_t_avg[go], grid_p[go], lat=ref_lat)[0]
N2[N2 < 0] = np.nan
N2 = nanseg_interp(grid, N2)
N2_all = np.nan * np.zeros(len(grid))
N2_all[0:-1] = gsw.Nsquared(salin_avg, cons_t_avg, grid_p, lat=ref_lat)[0]
N2_all[-2:] = N2_all[-3]
N2_all[N2_all < 0] = np.nan
N2_all = nanseg_interp(grid, N2_all)
N_all = np.sqrt(N2_all)
N2_all = savgol_filter(N2_all, 5, 3)

this_n2 = np.nan * np.ones(np.shape(sa))
for i in range(np.shape(ct)[1]):
    go = ~np.isnan(sa[:, i])
    this_n2[np.where(go)[0][0:-1], i] = gsw.Nsquared(sa[go, i], ct[go, i], grid_p[go], lat=ref_lat)[0]

# f, ax = plt.subplots()
# colls = 'r', 'g', 'b', 'm'
# inn = [0, 50, 100, 158]
# for i in range(1, 4):
#     ax.plot(np.nanmean(this_n2[:, inn[i - 1]:inn[i]], axis=1), grid, color=colls[i - 1])
# ax.invert_yaxis()
# ax.set_title(r'36N N$^2$ (Oct - Feb)')
# ax.set_ylabel('Depth [m]')
# plot_pro(ax)
# -------------------------------------------------------------------------------------------------
# -- compute M/W sections and compute velocity
# -- USING X.TRANSECT_CROSS_SECTION_1 (THIS WILL SEPARATE TRANSECTS BY TARGET OF EACH DIVE)
sigth_levels = np.concatenate(
    [np.arange(23, 26.5, 0.5), np.arange(26.2, 27.2, 0.2),
     np.arange(27.2, 27.7, 0.2), np.arange(27.7, 28, 0.02), np.arange(28, 28.15, 0.01)])
# sigth_levels = np.concatenate([np.aranger(32, 36.6, 0.2), np.arange(36.6, 36.8, 0.05), np.arange(36.8, 37.4, 0.02)])

# --- SAVE so that we don't have to run transects every time
savee = 0
if savee > 0:
    ds, dist, avg_ct_per_dep_0, avg_sa_per_dep_0, avg_sig0_per_dep_0, v_g, vbt, isopycdep, isopycx, mwe_lon, mwe_lat,\
    DACe_MW, DACn_MW, profile_tags_per, shear, v_g_east, v_g_north = x.transect_cross_section_1(grid, neutral_density,
                                                                                                ct, sa, lon, lat,
                                                                                                dac_u, dac_v,
                                                                                                profile_tags,
                                                                                                sigth_levels)
    my_dict = {'ds': ds, 'dist': dist, 'avg_ct_per_dep_0': avg_ct_per_dep_0,
               'avg_sa_per_dep_0': avg_sa_per_dep_0, 'avg_sig0_per_dep_0': avg_sig0_per_dep_0, 'v_g': v_g, 'vbt': vbt,
               'isopycdep': isopycdep, 'isopycx': isopycx, 'mwe_lon': mwe_lon, 'mwe_lat': mwe_lat, 'DACe_MW': DACe_MW,
               'DACn_MW': DACn_MW, 'profile_tags_per': profile_tags_per, 'v_g_east': v_g_east, 'v_g_north': v_g_north}
    output = open('/Users/jake/Documents/baroclinic_modes/DG/sg041_2018_transects_gamma.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()
else:
    pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg041_2018_transects_gamma.pkl', 'rb')
    B15 = pickle.load(pkl_file)
    pkl_file.close()
    ds = B15['ds']
    dist = B15['dist']
    avg_ct_per_dep_0 = B15['avg_ct_per_dep_0']
    avg_sa_per_dep_0 = B15['avg_sa_per_dep_0']
    avg_sig0_per_dep_0 = B15['avg_sig0_per_dep_0']
    v_g = B15['v_g']
    v_g_east = B15['v_g_east']
    v_g_north = B15['v_g_north']
    vbt = B15['vbt']
    isopycdep = B15['isopycdep']
    isopycx = B15['isopycx']
    mwe_lon = B15['mwe_lon']
    mwe_lat = B15['mwe_lat']
    DACe_MW = B15['DACe_MW']
    DACn_MW = B15['DACn_MW']
    profile_tags_per = B15['profile_tags_per']

# unpack velocity profiles from transect analysis
dace_mw_0 = DACe_MW[0][0:-1].copy()
dacn_mw_0 = DACn_MW[0][0:-1].copy()
dg_v_0 = v_g[0][:, 0:-1].copy()
dg_v_e_0 = v_g_east[0][:, 0:-1].copy()
dg_v_n_0 = v_g_north[0][:, 0:-1].copy()
avg_sig0_per_dep = avg_sig0_per_dep_0[0].copy()
avg_ct_per_dep = avg_ct_per_dep_0[0].copy()
avg_sa_per_dep = avg_sa_per_dep_0[0].copy()
dg_v_lon_0 = mwe_lon[0][0:-1].copy()
dg_v_lat_0 = mwe_lat[0][0:-1].copy()
dg_v_dive_no_0 = profile_tags_per[0][0:-1].copy()
for i in range(1, len(v_g)):
    dace_mw_0 = np.concatenate((dace_mw_0, DACe_MW[i][0:-1]), axis=0)
    dacn_mw_0 = np.concatenate((dacn_mw_0, DACn_MW[i][0:-1]), axis=0)
    dg_v_0 = np.concatenate((dg_v_0, v_g[i][:, 0:-1]), axis=1)
    dg_v_e_0 = np.concatenate((dg_v_e_0, v_g_east[i][:, 0:-1]), axis=1)
    dg_v_n_0 = np.concatenate((dg_v_n_0, v_g_north[i][:, 0:-1]), axis=1)
    avg_ct_per_dep = np.concatenate((avg_ct_per_dep, avg_ct_per_dep_0[i]), axis=1)
    avg_sa_per_dep = np.concatenate((avg_sa_per_dep, avg_sa_per_dep_0[i]), axis=1)
    avg_sig0_per_dep = np.concatenate((avg_sig0_per_dep, avg_sig0_per_dep_0[i]), axis=1)
    dg_v_lon_0 = np.concatenate((dg_v_lon_0, mwe_lon[i][0:-1]))
    dg_v_lat_0 = np.concatenate((dg_v_lat_0, mwe_lat[i][0:-1]))
    dg_v_dive_no_0 = np.concatenate((dg_v_dive_no_0, profile_tags_per[i][0:-1]))

# Time matching to eta/v profiles
count = 0
for i in range(0, len(profile_tags_per)):
    these_dives = profile_tags_per[i]
    for j in range(len(these_dives) - 1):
        tin = time_rec_bin[np.in1d(profile_tags, these_dives[j:j+2])]
        if count < 1:
            dg_mw_time = np.array([np.nanmean(tin)])
        else:
            dg_mw_time = np.concatenate((dg_mw_time, np.array([np.nanmean(tin)])))
        count = count + 1

# ----------------------------------------------------------------------------------------------------------------------
# ----- Eta compute from M/W method, which produces an average density per set of profiles
eta_alt = np.nan * np.ones(np.shape(avg_sig0_per_dep))
eta_alt_2 = np.nan * np.ones(np.shape(avg_sig0_per_dep))
eta_alt_3 = np.nan * np.ones(np.shape(avg_sig0_per_dep))
d_anom_alt = np.nan * np.ones(np.shape(avg_sig0_per_dep))
gradient_alt = np.nan * np.ones(np.shape(avg_sig0_per_dep))
for i in range(np.shape(avg_sig0_per_dep)[1]):  # loop over each profile
    # (average of four profiles) - (total long term average, that is seasonal)
    this_time = dg_mw_time[i]
    avg_a_salin = salin_avg
    avg_c_temp = cons_t_avg

    # loop over each bin depth
    for j in range(1, len(grid) - 1):
        # profile density at depth j with local
        this_sigma = gsw.rho(avg_sa_per_dep[j, i], avg_ct_per_dep[j, i], grid_p[j]) - 1000      # profile density
        # background density with local reference pressure
        this_sigma_avg = gsw.rho(avg_a_salin[j-1:j+2], avg_c_temp[j-1:j+2], grid_p[j]) - 1000
        d_anom_alt[j, i] = this_sigma - this_sigma_avg[1]
        gradient_alt[j, i] = np.nanmean(np.gradient(this_sigma_avg, z[j-1:j+2]))
        eta_alt_2[j, i] = d_anom_alt[j, i] / gradient_alt[j, i]

    eta_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg) / np.squeeze(ddz_avg_sigma)
    d_anom_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg)

    # ETA ALT 3
    # try a new way to compute vertical displacement
    for j in range(len(grid)):
        # find this profile density at j along avg profile
        idx, rho_idx = find_nearest(sigma_theta_avg, avg_sig0_per_dep[j, i])
        if idx <= 2:
            z_rho_1 = grid[0:idx + 3]
            eta_alt_3[j, i] = np.interp(avg_sig0_per_dep[j, i], sigma_theta_avg[0:idx + 3], z_rho_1) - grid[j]
        else:
            z_rho_1 = grid[idx - 2:idx + 3]
            eta_alt_3[j, i] = np.interp(avg_sig0_per_dep[j, i], sigma_theta_avg[idx - 2:idx + 3], z_rho_1) - grid[j]

eta_alt_0 = eta_alt.copy()
# ----------------------------------------------------------------------------------------------------------------------
# FILTER VELOCITY PROFILES IF THEY ARE TOO NOISY / BAD -- ALSO HAVE TO REMOVE EQUIVALENT ETA PROFILE
good_v = np.zeros(np.shape(dg_v_0)[1], dtype=bool)
for i in range(np.shape(dg_v_0)[1]):
    dv_dz = np.gradient(dg_v_0[:, i], -1 * grid)
    if (np.nanmax(np.abs(dv_dz)) < 0.0075) & (np.abs(dg_v_0[200, i]) < 0.2):
        good_v[i] = True

avg_sig = avg_sig0_per_dep[:, good_v]
eta_alt = eta_alt_3[:, good_v]
dace_mw = dace_mw_0[good_v]
dacn_mw = dacn_mw_0[good_v]
dg_v_lon = dg_v_lon_0[good_v]
dg_v_lat = dg_v_lat_0[good_v]
dg_v = dg_v_0[:, good_v]
dg_v_e = dg_v_e_0[:, good_v]
dg_v_n = dg_v_n_0[:, good_v]
dg_mw_time = dg_mw_time[good_v]
dg_v_dive_no = dg_v_dive_no_0[good_v]
num_mw_profs = np.shape(eta_alt)[1]

# Smooth DG N2 profiles
dg_avg_N2_coarse = N2_all.copy()
dg_avg_N2_coarse[np.isnan(dg_avg_N2_coarse)] = dg_avg_N2_coarse[~np.isnan(dg_avg_N2_coarse)][0] - 1*10**(-5)
dg_avg_N2 = savgol_filter(dg_avg_N2_coarse, 15, 3)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING cross section (CHECK)
# choose which transect
# u_levels = np.arange(-.7, .72, .04)
# transect_no = 8   # 29 = labby
# x.plot_cross_section(grid, ds[transect_no], v_g[transect_no], dist[transect_no],
#                      profile_tags_per[transect_no], isopycdep[transect_no], isopycx[transect_no],
#                      sigth_levels, d_time, u_levels)
# bathy_path = '/Users/jake/Desktop/bats/bats_bathymetry/bathymetry_b38e_27c7_f8c3_f3d6_790d_30c7.nc'
# plan_window = [-66, -63, 32, 37]
# plan_in = np.where((profile_tags >= profile_tags_per[transect_no][0]) & (profile_tags <=
#                                                                          profile_tags_per[transect_no][-1]))[0]
# # x.plot_plan_view(lon[:, plan_in], lat[:, plan_in], mwe_lon[transect_no], mwe_lat[transect_no],
# #                  DACe_MW[transect_no], DACn_MW[transect_no],
# #                  ref_lat, profile_tags_per[transect_no], d_time[:, plan_in], plan_window, bathy_path)
# x.plot_plan_view(lon, lat, mwe_lon[transect_no], mwe_lat[transect_no],
#                  DACe_MW[transect_no], DACn_MW[transect_no],
#                  ref_lat, profile_tags_per[transect_no], d_time, plan_window, bathy_path)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# - Vertical Modes

# --- compute vertical mode shapes
G, Gz, c, epsilon = vertical_modes(dg_avg_N2, grid, omega, mmax)  # N2

# --- compute alternate vertical modes
bc_bot = 2  # 1 = flat, 2 = rough
grid2 = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4500, 10)])
n2_interp = np.interp(grid2, grid, dg_avg_N2)
F_int_g2, F_g2, c_ff, norm_constant, epsilon2 = vertical_modes_f(n2_interp, grid2, omega, mmax, bc_bot)
F = np.nan * np.ones((np.size(grid), mmax + 1))
F_int = np.nan * np.ones((np.size(grid), mmax + 1))
for i in range(mmax + 1):
    F[:, i] = np.interp(grid, grid2, F_g2[:, i])
    F_int[:, i] = np.interp(grid, grid2, F_int_g2[:, i])
# -----------------------------------------------------------------------------------
# ----- SOME VELOCITY PROFILES ARE TOO NOISY AND DEEMED UNTRUSTWORTHY --------------------------------------
# select only velocity profiles that seem reasonable
# criteria are slope of v (dont want kinks)
# criteria: limit surface velocity to greater that 40cm/s
Avg_sig = avg_sig.copy()
Time2 = dg_mw_time.copy()
V2 = dg_v.copy()
Eta2 = eta_alt.copy()
Eta2_c = eta_alt.copy()
Info2 = dg_v_dive_no.copy()
prof_lon2 = dg_v_lon.copy()
prof_lat2 = dg_v_lat.copy()
# ---------------------------------------------------------------------------------------------------------------
# ---- PROJECT MODES ONTO EACH PROFILE -------
# ---- Velocity and Eta
sz = np.shape(Eta2)
num_profs = sz[1]
eta_fit_depth_min = 250
eta_fit_depth_max = 3750  # 3900
AG = np.zeros([nmodes, num_profs])
AGz = np.zeros([nmodes, num_profs])
Eta_m = np.nan * np.zeros([np.size(grid), num_profs])
V_m = np.nan * np.zeros([np.size(grid), num_profs])
Neta = np.nan * np.zeros([np.size(grid), num_profs])
NEta_m = np.nan * np.zeros([np.size(grid), num_profs])
PE_per_mass = np.nan * np.zeros([nmodes, num_profs])
HKE_per_mass = np.nan * np.zeros([nmodes, num_profs])
modest = np.arange(11, nmodes)
good_ke_prof = np.ones(num_profs)
good_pe_prof = np.ones(num_profs)
HKE_noise_threshold = 1e-5  # 1e-5
PE_noise_threshold = 1e5
for i in range(num_profs):
    # fit to velocity profiles
    this_V = V2[:, i].copy()
    iv = np.where(~np.isnan(this_V))
    if iv[0].size > 1:
        AGz[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(Gz[iv, :]), np.transpose(np.atleast_2d(this_V[iv])))[0])
        # Gz(iv,:)\V_g(iv,ip)
        V_m[:, i] = np.squeeze(np.matrix(Gz) * np.transpose(np.matrix(AGz[:, i])))
        # Gz*AGz[:,i];
        HKE_per_mass[:, i] = (1 / 2) * (AGz[:, i] * AGz[:, i])
        ival = np.where(HKE_per_mass[modest, i] >= HKE_noise_threshold)
        if np.size(ival) > 0:
            good_ke_prof[i] = 0  # flag profile as noisy
    else:
        good_ke_prof[i] = 0  # flag empty profile as noisy as well

    # fit to eta profiles
    this_eta = Eta2[:, i].copy()
    # obtain matrix of NEta
    Neta[:, i] = N_all * this_eta
    iw = np.where((grid >= eta_fit_depth_min) & (grid <= eta_fit_depth_max))
    if len(iw[0]) > 1:
        eta_fs = Eta2[:, i].copy()  # ETA

        # -- taper fit as z approaches 0
        i_sh = np.where((grid < eta_fit_depth_min))
        eta_fs[i_sh[0]] = grid[i_sh] * this_eta[iw[0][0]] / grid[iw[0][0]]
        # -- taper fit as z approaches -H
        i_dp = np.where((grid > eta_fit_depth_max))
        eta_fs[i_dp[0]] = (grid[i_dp] - grid[-1]) * this_eta[iw[0][-1]] / (grid[iw[0][-1]] - grid[-1])

        # -- solve matrix problem
        AG[1:, i] = np.linalg.lstsq(G[:, 1:], eta_fs[:, np.newaxis])[0][:, 0]
        # AG[1:, i] = np.linalg.lstsq(F_int[:, 1:], eta_fs[:, np.newaxis])[0][:, 0]

        Eta_m[:, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG[:, i])))
        # Eta_m[:, i] = np.squeeze(np.matrix(F_int) * np.transpose(np.matrix(AG[:, i])))
        NEta_m[:, i] = N_all * np.array(np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG[:, i]))))
        PE_per_mass[:, i] = (1 / 2) * AG[:, i] * AG[:, i] * c * c

        np.where(PE_per_mass[modest, i] >= PE_noise_threshold)
        iwal = np.where(PE_per_mass[modest, i] >= PE_noise_threshold)
        if np.size(iwal) > 0:
            good_pe_prof[i] = 0  # flag profile as noisy
# end loop over each v and eta for fitting

sa = 0
if sa > 0:
    mydict = {'bin_depth': grid, 'eta': Eta2, 'dg_v': V2}
    output = open('/Users/jake/Desktop/bats/den_v_profs.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()

# ---------------------------------------------------------------------------------------------------------------------
# EAST/NORTH VELOCITY PROFILES
dg_v_e_avg = np.nanmean(dg_v_e[:, good_ke_prof > 0], axis=1)
dg_v_n_avg = np.nanmean(dg_v_n[:, good_ke_prof > 0], axis=1)
dz_dg_v_e_avg = np.gradient(savgol_filter(dg_v_e_avg, 15, 7), z)
dz_dg_v_n_avg = np.gradient(savgol_filter(dg_v_n_avg, 15, 7), z)
# PLOT (non-noisy) EAST/NORTH VELOCITY PROFILES
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
dz_dg_v = np.nan * np.ones(np.shape(dg_v))
for i in range(np.shape(dg_v_e)[1]):
    if good_ke_prof[i] > 0:
        # ax1.plot(dg_v_e[:, i], grid, color='#D3D3D3')
        # ax2.plot(dg_v_n[:, i], grid, color='#D3D3D3')
        ax3.plot(dg_v[:, i], grid)
        dz_dg_v[:, i] = np.gradient(savgol_filter(dg_v[:, i], 13, 5), z)
ax1.plot(np.nanmean(dg_v_e[:, good_ke_prof > 0], axis=1), grid, color='b', linewidth=2)
ax2.plot(np.nanmean(dg_v_n[:, good_ke_prof > 0], axis=1), grid, color='b', linewidth=2)
ax1.plot(np.nanmean(dace_mw[good_ke_prof > 0]) * np.ones(10), np.linspace(0, 5000, 10), color='k', linewidth=1)
ax2.plot(np.nanmean(dacn_mw[good_ke_prof > 0]) * np.ones(10), np.linspace(0, 5000, 10), color='k', linewidth=1)
ax1.set_ylim([0, 5000])
ax1.set_xlim([-.1, .05])
ax2.set_xlim([-.1, .05])
ax3.set_xlim([-.75, .75])
ax1.invert_yaxis()
ax1.grid()
ax2.grid()
ax1.set_title('Mean Zonal Vel')
ax1.set_ylabel('Depth [m]')
ax1.set_xlabel('[m/s]')
ax2.set_title('Mean Meridional Vel')
ax2.set_xlabel('[m/s]')
ax3.set_title('Cross-Track Vel')
ax3.set_xlabel('[m/s]')
plot_pro(ax3)

mw_time_ordered_i = np.argsort(Time2)
dg_v_lon[good_ke_prof < 0] = np.nan
dg_v_lat[good_ke_prof < 0] = np.nan
dace_mw[good_ke_prof < 0] = np.nan
dacn_mw[good_ke_prof < 0] = np.nan
dg_v_lon_1 = dg_v_lon[mw_time_ordered_i]
dg_v_lat_1 = dg_v_lat[mw_time_ordered_i]
dace_mw_1 = dace_mw[mw_time_ordered_i]
dacn_mw_1 = dacn_mw[mw_time_ordered_i]

f, ax = plt.subplots()
ax.plot(dg_v_lon_1, dg_v_lat_1, color='k', linewidth=0.5)
ax.scatter(dg_v_lon_1, dg_v_lat_1, color='k', s=3)
ax.quiver(dg_v_lon_1, dg_v_lat_1, dace_mw_1, dacn_mw_1, color='r', scale=0.8)
w = 1 / np.cos(np.deg2rad(ref_lat))
ax.set_aspect(w)
ax.set_title('Bermuda 2018: DAC')
plot_pro(ax)
# ---------------------------------------------------------------------------------------------------------------------
# --- ETA COMPUTED FROM INDIVIDUAL DENSITY PROFILES
# --- compute vertical mode shapes
G_all, Gz_all, c_all, epsilon_all = vertical_modes(N2_all, grid, omega, mmax)
# --- need to compute eta from individual profiles (new disp. mode amplitudes) to compared to averaging eta technique
# eta_per_prof = np.nan * np.ones(np.shape(df_den))
# for i in range(len(prof_lon_i)):
#     # (density from each profile) - (total long term average)
#     eta_per_prof[:, i] = (df_den.iloc[:, i] - sigma_theta_avg)/ddz_avg_sigma
# --- by season background profile
eta_per_prof = np.nan * np.ones(sig2.shape)
eta_per_prof_3 = np.nan * np.ones(sig2.shape)
d_anom_prof = np.nan * np.ones(sig2.shape)
for i in range(lon.shape[1]):
    eta_per_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg) / np.squeeze(ddz_avg_sigma)
    d_anom_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg)

    # ETA ALT 3
    # try a new way to compute vertical displacement
    for j in range(len(grid)):
        # find this profile density at j along avg profile
        idx, rho_idx = find_nearest(sigma_theta_avg, neutral_density[j, i])
        if idx <= 2:
            z_rho_1 = grid[0:idx + 3]
            eta_per_prof_3[j, i] = np.interp(neutral_density[j, i], sigma_theta_avg[0:idx + 3],
                                             z_rho_1) - grid[j]
        else:
            z_rho_1 = grid[idx - 2:idx + 3]
            eta_per_prof_3[j, i] = np.interp(neutral_density[j, i], sigma_theta_avg[idx - 2:idx + 3],
                                             z_rho_1) - grid[j]


AG_all, eta_m_all, Neta_m_all, PE_per_mass_all = eta_fit(lon.shape[1], grid, nmodes, N2_all, G_all, c_all,
                                                         eta_per_prof, eta_fit_depth_min, eta_fit_depth_max)
PE_per_mass_all = PE_per_mass_all[:, np.abs(AG_all[1, :]) > 1*10**-4]

# --- check on mode amplitudes from averaging or individual profiles
mw_time_ordered_i = np.argsort(Time2)
AG_ordered = AG[:, mw_time_ordered_i]
AGz_ordered = AGz[:, mw_time_ordered_i]

# -- load other data
pkl_file = open('/Users/jake/Desktop/bats/station_bats_pe_nov05.pkl', 'rb')
SB = pickle.load(pkl_file)
pkl_file.close()
bats_time = SB['time'][SB['time'] > 2015]
bats_sig2 = SB['Sigma2'][:, SB['time'] > 2015]
bats_time_ord = np.nan * np.ones(len(bats_time))
bats_time_date = []
for i in range(len(bats_time)):
    bats_year = np.floor(bats_time[i])
    bats_month = np.floor(12*(bats_time[i] - bats_year))
    bats_day = np.floor(30*(12*(bats_time[i] - bats_year) - bats_month))
    bats_time_ord[i] = datetime.date.toordinal(datetime.date(np.int(bats_year), np.int(bats_month), np.int(bats_day)))
    bats_time_date.append(datetime.date.fromordinal(np.int(bats_time_ord[i])))

# ----------------------------------------------------------------------------------------------------------------------
# ISOPYCNAL DEPTH IN TIME
# isopycnals I care about
rho1 = 27.0
rho2 = 27.8
rho3 = 28.05

d_time_per_prof = np.nanmean(d_time, axis=0)
d_time_per_prof_date = []
d_dep_rho1 = np.nan * np.ones((3, len(d_time_per_prof)))
for i in range(len(d_time_per_prof)):
    d_time_per_prof_date.append(datetime.date.fromordinal(np.int(d_time_per_prof[i])))
    d_dep_rho1[0, i] = np.interp(rho1, neutral_density[:, i], grid)
    d_dep_rho1[1, i] = np.interp(rho2, neutral_density[:, i], grid)
    d_dep_rho1[2, i] = np.interp(rho3, neutral_density[:, i], grid)
mw_time_ordered = Time2[mw_time_ordered_i]
mw_sig_ordered = Avg_sig[:, mw_time_ordered_i]
mw_time_date = []
mw_dep_rho1 = np.nan * np.ones((3, len(mw_time_ordered)))
for i in range(len(Time2)):
    mw_time_date.append(datetime.date.fromordinal(np.int(np.round(mw_time_ordered[i]))))
    mw_dep_rho1[0, i] = np.interp(rho1, mw_sig_ordered[:, i], grid)
    mw_dep_rho1[1, i] = np.interp(rho2, mw_sig_ordered[:, i], grid)
    mw_dep_rho1[2, i] = np.interp(rho3, mw_sig_ordered[:, i], grid)

# ------
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.scatter(d_time_per_prof_date, d_dep_rho1[0, :], color='g', s=15, label=r'DG$_{ind}$')
ax2.scatter(d_time_per_prof_date, d_dep_rho1[1, :], color='g', s=15)
ax3.scatter(d_time_per_prof_date, d_dep_rho1[2, :], color='g', s=15)

ax1.plot(mw_time_date, mw_dep_rho1[0, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')
ax2.plot(mw_time_date, mw_dep_rho1[1, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')
ax3.plot(mw_time_date, mw_dep_rho1[2, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')

ax1.set_title(x.project + str(r': Depth of $\gamma^{n}$ = ') + str(rho1))
ax2.set_title('Depth of $\gamma^{n}$ = ' + str(rho2))
ax3.set_title('Depth of $\gamma^{n}$ = ' + str(rho3))
ax1.set_ylabel('Depth [m]')
ax2.set_ylabel('Depth [m]')
ax3.set_ylabel('Depth [m]')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=10)
# ax1.set_ylim([np.nanmean(Avg_sig[90, mw_time_ordered_i]) - 0.05, np.nanmean(Avg_sig[90, mw_time_ordered_i]) + 0.05])
# ax2.set_ylim([np.nanmean(Avg_sig[165, mw_time_ordered_i]) - 0.05, np.nanmean(Avg_sig[165, mw_time_ordered_i]) + 0.05])
ax1.set_ylim([500, 900])
ax2.set_ylim([1000, 1400])
ax3.set_ylim([2850, 3250])
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax1.grid()
ax2.grid()
ax3.set_xlim([datetime.date(2018, 10, 1), datetime.date(2019, 3, 10)])
plot_pro(ax3)

f, (ax, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
ax.scatter(d_time_per_prof_date, AG_all[1, :], color='g', s=15)
ax.plot(mw_time_date, -AG_ordered[1, :], color='b', linewidth=0.75)
ax.set_title('Displacement Mode 1 Amplitude')
ax2.scatter(d_time_per_prof_date, AG_all[2, :], color='g', s=15)
ax2.plot(mw_time_date, -AG_ordered[2, :], color='b', linewidth=0.75)
ax2.set_title('Displacement Mode 2 Amplitude')
ax3.scatter(d_time_per_prof_date, AG_all[3, :], color='g', s=15)
ax3.plot(mw_time_date, -AG_ordered[3, :], color='b', linewidth=0.75)
ax3.set_title('Displacement Mode 3 Amplitude')
ax4.scatter(d_time_per_prof_date, AG_all[4, :], color='g', s=15)
ax4.plot(mw_time_date, -AG_ordered[4, :], color='b', linewidth=0.75)
ax4.set_title('Displacement Mode 4 Amplitude')
ax5.scatter(d_time_per_prof_date, AG_all[5, :], color='g', s=15)
ax5.plot(mw_time_date, -AG_ordered[5, :], color='b', linewidth=0.75)
ax5.set_title('Displacement Mode 5 Amplitude')
ax.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.set_xlim([datetime.date(2018, 10, 1), datetime.date(2019, 3, 10)])
plot_pro(ax5)

f, (ax, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
ax.plot(mw_time_date, AGz_ordered[0, :], color='b', linewidth=0.75)
ax.set_title('Velocity Mode 0 Amplitude')
ax2.plot(mw_time_date, AGz_ordered[1, :], color='b', linewidth=0.75)
ax2.set_title('Velocity Mode 1 Amplitude')
ax3.plot(mw_time_date, AGz_ordered[2, :], color='b', linewidth=0.75)
ax3.set_title('Velocity Mode 2 Amplitude')
ax4.plot(mw_time_date, AGz_ordered[3, :], color='b', linewidth=0.75)
ax4.set_title('Velocity Mode 3 Amplitude')
ax5.plot(mw_time_date, AGz_ordered[4, :], color='b', linewidth=0.75)
ax5.set_title('Velocity Mode 4 Amplitude')
ax.set_ylim([-.2, .2])
ax.grid()
ax2.grid()
ax2.set_ylim([-.2, .2])
ax3.grid()
ax3.set_ylim([-.12, .12])
ax4.grid()
ax4.set_ylim([-.07, .07])
ax5.set_xlabel('Date')
ax5.set_ylim([-.07, .07])
ax5.set_xlim([datetime.date(2018, 10, 1), datetime.date(2019, 3, 10)])
plot_pro(ax5)
# ----------------------------------------------------------------------------------------------------------------------
# --- Eta per select profiles
# these_profiles = np.array([80, 81, 82, 83, 84])
# these_profiles = np.array([72, 74, 74.5, 75, 76])
these_profiles = np.array([86, 86.5, 87, 87.5, 88])
# these_profiles = np.array([50, 75, 100, 125])  # dive numbers of profiles to compare (individual dives)
# these_profiles = np.array([60, 85, 110, 135])  # dive numbers of profiles to compare (individual dives)
# these_profiles = np.array([62, 62.5, 63, 63.5, 64])  # dive numbers of profiles to compare (individual dives)
# these_profiles = np.array([67, 68, 69, 70, 71])  # dive numbers of profiles to compare (individual dives)
# these_profiles = np.array([72, 72.5, 73, 73.5, 74])  # dive numbers of profiles to compare (individual dives)
f, ax = plt.subplots(1, 5, sharey=True)
for i in range(5):
    ind_rel = profile_tags == these_profiles[i]
    avg_rel = dg_v_dive_no_0 == these_profiles[i]
    ax[i].plot(eta_per_prof_3[:, ind_rel], grid, color='k', linewidth=0.75, label=r'$\gamma$ Ind,Dir')  # individual profiles direct search, gamma
    ax[i].plot(eta_alt_3[:, avg_rel], grid, color='r', linewidth=0.75, label=r'$\gamma$ Avg,Dir')  # avg direct search, gamma
    ax[i].plot(-1 * eta_alt_0[:, avg_rel], grid, color='b', linewidth=0.75, label=r'$\gamma$ Avg,ddz')  # avg divide by ddz, gamma
    ax[i].plot(-1 * eta_alt_2[:, avg_rel], grid, color='g', linewidth=0.75, label=r'$\sigma_{\theta0}$ Avg, ddz')  # avg divide by ddz, pot den, local pref
    ax[i].set_xlim([-380, 380])
    ax[i].set_title('Dive-Cycle = ' + str(these_profiles[i]))

handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles, labels, fontsize=7)
ax[0].set_ylabel('Depth [m]')
ax[0].set_xlabel('Vertical Disp. [m]', fontsize=10)
ax[1].set_xlabel('Vertical Disp. [m]', fontsize=10)
ax[2].set_xlabel('Vertical Disp. [m]', fontsize=10)
ax[3].set_xlabel('Vertical Disp. [m]', fontsize=10)
ax[4].set_xlabel('Vertical Disp. [m]', fontsize=10)
ax[4].invert_yaxis()
ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[3].grid()
plot_pro(ax[4])
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ---- COMPUTE EOF SHAPES AND COMPARE TO ASSUMED STRUCTURE
# --- V ---
# --- find EOFs of dynamic horizontal current (V mode amplitudes)
AGzq = AGz[:, good_ke_prof > 0]
nq = np.sum(good_ke_prof > 0)  # good_prof and dg_good_prof
avg_AGzq = np.nanmean(np.transpose(AGzq), axis=0)
# -- mode amplitude anomaly matrix
AGzqa = AGzq - np.transpose(np.tile(avg_AGzq, [nq, 1]))
# -- nmodes X nmodes covariance matrix (squared mode amplitude anomaly / number of profiles) = covariance
cov_AGzqa = (1 / nq) * np.matrix(AGzqa) * np.matrix(np.transpose(AGzqa))
# -- sqrt(cov)*sqrt(cov) = variance
var_AGzqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa)))) * np.matrix(np.sqrt(np.diag(cov_AGzqa)))
# -- nmodes X nmodes correlation matrix (cov/var) = correlation
cor_AGzqa = cov_AGzqa / var_AGzqa

# (look at how mode amplitude anomalies are correlated) =>
# to look at shape of eigenfunctions of the correlation matrix
# (project the shape of the eigenfunctions onto the vertical structure G, Gz )
D_AGzqa, V_AGzqa = np.linalg.eig(cov_AGzqa)  # columns of V_AGzqa are eigenvectors

EOFseries = np.transpose(V_AGzqa) * np.matrix(AGzqa)  # EOF "timeseries' [nmodes X nq]
EOFshape = np.matrix(Gz) * V_AGzqa  # depth shape of eigenfunctions [ndepths X nmodes]
EOFshape1_BTpBC1 = np.matrix(Gz[:, 0:2]) * V_AGzqa[0:2, 0]  # truncated 2 mode shape of EOF#1
EOFshape2_BTpBC1 = np.matrix(Gz[:, 0:2]) * V_AGzqa[0:2, 1]  # truncated 2 mode shape of EOF#2
# --------------------------------------------------------------------------------------------------
# --- ETA ---
# --- find EOFs of dynamic vertical displacement (Eta mode amplitudes)
# extract noisy/bad profiles
good_prof_eof = np.where(~np.isnan(AG[2, :]))
num_profs_2 = np.size(good_prof_eof)
AG2 = AG[:, good_prof_eof[0]]
C = np.transpose(np.tile(c, (num_profs_2, 1)))
AGs = C * AG2
AGq = AGs[1:, :]  # ignores barotropic mode
nqd = num_profs_2
avg_AGq = np.nanmean(AGq, axis=1)
AGqa = AGq - np.transpose(np.tile(avg_AGq, [nqd, 1]))  # mode amplitude anomaly matrix
cov_AGqa = (1 / nqd) * np.matrix(AGqa) * np.matrix(np.transpose(AGqa))  # nmodes X nmodes covariance matrix
var_AGqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGqa)))) * np.matrix(np.sqrt(np.diag(cov_AGqa)))
cor_AGqa = cov_AGqa / var_AGqa  # nmodes X nmodes correlation matrix
D_AGqa, V_AGqa = np.linalg.eig(cov_AGqa)  # columns of V_AGzqa are eigenvectors
EOFetaseries = np.transpose(V_AGqa) * np.matrix(AGqa)  # EOF "timeseries' [nmodes X nq]
EOFetashape = np.matrix(G[:, 1:]) * V_AGqa  # depth shape of eigenfunctions [ndepths X nmodes]
EOFetashape1_BTpBC1 = G[:, 1:3] * V_AGqa[0:2, 0]  # truncated 2 mode shape of EOF#1
EOFetashape2_BTpBC1 = G[:, 1:3] * V_AGqa[0:2, 1]  # truncated 2 mode shape of EOF#2
# ----------------------------------------------------------------------------------------------------
# --- EOF of velocity profiles ----------------------------------------
not_shallow = np.isfinite(V2[-15, :])  # & (Time2 > 735750)
V3 = V2[:, (good_ke_prof > 0) & not_shallow]
Time3 = Time2[(good_ke_prof > 0) & not_shallow]
check1 = 3      # upper index to include in eof computation
check2 = -14     # lower index to include in eof computation
grid_check = grid[check1:check2]
Uzq = V3[check1:check2, :].copy()

# loop over every two weeks (to obtain statistics)
T_week = np.arange(Time3.min(), Time3.max(), 14)
PEV_per = np.nan * np.ones((len(T_week) - 1, 10))
fvu1_per = np.nan * np.ones(len(T_week) - 1)
fvu2_per = np.nan * np.ones(len(T_week) - 1)
bc1 = Gz[check1:check2, 0]  # flat bottom (modes 0)
bc11 = + Gz[check1:check2, 1]  # flat bottom (modes 1)
bc2 = F[check1:check2, 0]  # sloping bottom  (modes 1 + 2 ... there is no barotropic mode)
f, (ax, ax1) = plt.subplots(1, 2)
AGz_eof = np.nan * np.ones((nmodes, len(T_week) - 1))
AF_eof = np.nan * np.ones((nmodes, len(T_week) - 1))
for i in range(len(T_week) - 1):
    V4 = Uzq[:, (Time3 > T_week[i]) & (Time3 < T_week[i + 1])].copy()
    nq = np.size(V4[0, :])
    avg_Uzq = np.nanmean(np.transpose(Uzq), axis=0)
    Uzqa = V4 - np.transpose(np.tile(avg_Uzq, [nq, 1]))
    cov_Uzqa = (1 / nq) * np.matrix(Uzqa) * np.matrix(np.transpose(Uzqa))
    D_Uzqa, V_Uzqa = np.linalg.eig(cov_Uzqa)
    t1 = np.real(D_Uzqa[0:10])
    # percent explained variance by each eof
    PEV_per[i, :] = (t1 / np.sum(t1))
    # percent variance explained of each eof by baroclinic modes with either bottom boundary condition
    eof1 = np.array(np.real(V_Uzqa[:, 0]))
    # -- minimize mode shapes onto eof shape
    p = np.array([0.8 * eof1.min() / np.max(np.abs(F[:, 0])), 0.8 * eof1.min() / np.max(np.abs(F[:, 0]))])
    p2 = 0.8 * eof1.min() / np.max(np.abs(F[:, 0]))
    ins1 = np.transpose(np.concatenate([eof1, bc1[:, np.newaxis], bc11[:, np.newaxis]], axis=1))
    ins2 = np.transpose(np.concatenate([eof1, bc2[:, np.newaxis]], axis=1))
    min_p1 = fmin(functi_2, p, args=(tuple(ins1)))
    min_p2 = fmin(functi_1, p2, args=(tuple(ins2)))
    fvu1_per[i] = 1 - (np.sum((eof1[:, 0] - (bc1 * min_p1[0] + bc11 * min_p1[1])) ** 2) /
                       np.sum((eof1 - np.mean(eof1)) ** 2))
    fvu2_per[i] = 1 - (np.sum((eof1[:, 0] - bc2 * min_p2) ** 2) / np.sum((eof1 - np.mean(eof1)) ** 2))

    if np.nanmean(eof1[0:20] < 0):
        if fvu1_per[i] > 0:
            ax.plot(-1 * eof1, grid_check, color='k', linewidth=1.25, label='EOF1')
            ax.plot(-1 * (bc1 * min_p1[0] + bc11 * min_p1[1]), grid_check, color='r', linewidth=0.75, label='B1')
        if fvu2_per[i] > 0:
            ax.plot(-1 * bc2 * min_p2, grid_check, color='b', linewidth=0.75, label='B2')
    else:
        if fvu1_per[i] > 0:
            ax.plot(eof1, grid_check, color='k', linewidth=1.25)
            ax.plot((bc1 * min_p1[0] + bc11 * min_p1[1]), grid_check, color='r', linewidth=0.75)
        if fvu2_per[i] > 0:
            ax.plot(bc2 * min_p2, grid_check, color='b', linewidth=0.75)

handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[0], handles[1], handles[2]], [labels[0], labels[1], labels[2]], fontsize=10)
ax.set_title('Velocity EOF1 (per two week interval)')
ax.set_ylabel('Depth [m]')
ax.set_xlabel('EOF magnitude')
ax.invert_yaxis()

fvu1_per[fvu1_per < 0] = np.nan
fvu2_per[fvu2_per < 0] = np.nan
data1 = np.array([np.nanstd(PEV_per[:, 0]), np.nanmean(PEV_per[:, 0]), np.nanmin(PEV_per[:, 0]),
                       np.nanmax(PEV_per[:, 0])])
data2 = np.array([np.nanstd(PEV_per[:, 1]), np.nanmean(PEV_per[:, 1]), np.nanmin(PEV_per[:, 1]),
                       np.nanmax(PEV_per[:, 1])])
data3 = np.array([np.nanstd(fvu1_per), np.nanmean(fvu1_per), np.nanmin(fvu1_per), np.nanmax(fvu1_per)])
data4 = np.array([np.nanstd(fvu2_per), np.nanmean(fvu2_per), np.nanmin(fvu2_per), np.nanmax(fvu2_per)])

# ax1.boxplot([data1, data2])
# np.concatenate((fvu1_per[:, None], fvu2_per[:, None]), axis=1)
ax1.boxplot([fvu1_per[:, None], fvu2_per[~np.isnan(fvu2_per)]])
ax1.set_ylim([0, 1])
ax1.set_title('Frac. Var. Explained of EOF1 by Baroclinic Modes')
ax1.set_xlabel('Bottom Boundary Condition')
ax1.set_ylabel('Frac Var. Explained')
ax1.set_xticklabels(['Flat Bottom (0 + 1)', 'Sloping Bottom (0)'])
# ax2.set_title('Bottom Boundary Condition')
# ax2.set_ylabel('Frac Var. Explained by Mode Shapes')
# ax2.set_ylim([0, 1])
ax.grid()
plot_pro(ax1)

# Old method computing EOFs from all V profiles
nq = np.size(V3[0, :])
avg_Uzq = np.nanmean(np.transpose(Uzq), axis=0)
Uzqa = Uzq - np.transpose(np.tile(avg_Uzq, [nq, 1]))
cov_Uzqa = (1 / nq) * np.matrix(Uzqa) * np.matrix(np.transpose(Uzqa))
D_Uzqa, V_Uzqa = np.linalg.eig(cov_Uzqa)

t1 = np.real(D_Uzqa[0:10])
PEV = t1 / np.sum(t1)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --- PLOT ETA / EOF
# --- time series
mission_start = datetime.date.fromordinal(np.int(Time2.min()))
mission_end = datetime.date.fromordinal(np.int(Time2.max()))
plot_eta = 1
if plot_eta > 0:
    f, (ax2, ax1, ax15, ax0) = plt.subplots(1, 4, sharey=True)
    for j in range(len(Time2)):
        ax2.plot(d_anom_alt[:, j], grid, linewidth=0.75)
    ax2.set_xlim([-.5, .5])
    ax2.set_xlabel(r'$\gamma^n - \overline{\gamma^n}$', fontsize=12)
    ax2.set_title("DG41: " + str(mission_start) + ' - ' + str(mission_end))
    ax2.text(0.1, 4000, str(len(Time2)) + ' profiles', fontsize=10)
    for j in range(num_profs):
        ax1.plot(Eta2[:, j], grid, color='#4682B4', linewidth=1.25)
        ax1.plot(Eta_m[:, j], grid, color='k', linestyle='--', linewidth=.75)
        if good_ke_prof[j] > 0:
            ax0.plot(V2[:, j], grid, color='#4682B4', linewidth=1.25)
            ax0.plot(V_m[:, j], grid, color='k', linestyle='--', linewidth=.75)
    for j in range(eta_per_prof.shape[1]):
        ax15.plot(-1 * eta_per_prof[:, j], grid, color='#4682B4', linewidth=1.25)
        ax15.plot(-1 * eta_m_all[:, j], grid, color='k', linestyle='--', linewidth=.75)
    ax15.set_title(r'Isopycnal Disp. (Ind.)', fontsize=11)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=10)
    ax1.set_xlim([-400, 400])
    ax15.set_xlim([-400, 400])
    ax0.text(190, 800, str(num_profs) + ' Profiles')
    ax1.set_xlabel(r'Vertical Isopycnal Displacement, $\xi_{\gamma}$ [m]', fontsize=11)
    ax1.set_title(r'Isopycnal Disp. (Avg.)', fontsize=11)  # + '(' + str(Time[0]) + '-' )
    ax0.axis([-1, 1, 0, 5000])
    ax0.set_title("Geostrophic Velocity", fontsize=11)  # (" + str(num_profs) + 'profiles)' )
    ax2.set_ylabel('Depth [m]', fontsize=11)
    ax0.set_xlabel('Cross-Track Velocity, U [m/s]', fontsize=11)
    ax0.invert_yaxis()
    ax2.grid()
    ax1.grid()
    ax15.grid()
    plot_pro(ax0)

    max_plot = 3
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    n2p = ax1.plot((np.sqrt(N2_all) * (1800 / np.pi)), grid, color='k', label='N(z) [cph]')
    colors = plt.cm.Dark2(np.arange(0, 4, 1))
    for ii in range(max_plot):
        ax1.plot(Gz[:, ii], grid, color='#2F4F4F', linestyle='--')
        p_eof = ax1.plot(-EOFshape[:, ii], grid, color=colors[ii, :], label='EOF # = ' + str(ii + 1), linewidth=2.5)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=10)
    ax1.axis([-4, 4, 0, 5000])
    ax1.invert_yaxis()
    ax1.set_title('EOF Velocity Mode Shapes', fontsize=18)
    ax1.set_ylabel('Depth [m]', fontsize=16)
    ax1.set_xlabel('Normalized Mode Amp.', fontsize=14)
    ax1.grid()
    for ii in range(max_plot):
        ax2.plot(G[:, ii + 1] / np.max(grid), grid, color='#2F4F4F', linestyle='--')
        p_eof_eta = ax2.plot(EOFetashape[:, ii] / np.max(grid), grid, color=colors[ii, :],
                             label='EOF # = ' + str(ii + 1), linewidth=2.5)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=10)
    ax2.axis([-.7, .7, 0, 5000])
    ax2.set_title('EOF Displacement Mode Shapes', fontsize=18)
    ax2.set_xlabel('Normalized Mode Amp.', fontsize=14)
    ax2.invert_yaxis()
    plot_pro(ax2)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --- ENERGY SPECTRA
# ----------------------------------------------------------------------------------------------------------------------

# --- AVERAGE ENERGY
HKE_per_mass = HKE_per_mass[:, np.where(good_ke_prof > 0)[0]]
PE_per_mass = PE_per_mass[:, np.where(good_ke_prof > 0)[0]]
used_profiles = dg_v_dive_no[good_ke_prof > 0]
# calmer = np.concatenate((np.arange(0, 16), np.arange(18, 102), np.arange(106, PE_per_mass.shape[1])))  # can exclude the labby
calmer = np.arange(0, np.int(np.sum(good_ke_prof)))
avg_PE = np.nanmean(PE_per_mass[:, calmer], 1)
avg_KE = np.nanmean(HKE_per_mass[:, calmer], 1)
# --- eddy kinetic and potential energy
# PE_ed = np.nanmean(PE_per_mass[:, ed_in[0]:ed_in[-1]], axis=1)
# KE_ed = np.nanmean(HKE_per_mass[:, ed_in[0]:ed_in[-1]], axis=1)

# f, ax = plt.subplots()
# for i in range(len(calmer)):
#     ax.plot(np.arange(0, mmax+1), HKE_per_mass, linewidth=0.5)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plot_pro(ax)

# ----- ENERGY parameters ------
f_ref = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)
dk = f_ref / c[1]
sc_x = 1000 * f_ref / c[1:]
vert_wavenumber = f_ref / c[1:]
dk_ke = 1000 * f_ref / c[1]
k_h = 1e3 * (f_ref / c[1:]) * np.sqrt(avg_KE[1:] / avg_PE[1:])
PE_SD, PE_GM, GMPE, GMKE = PE_Tide_GM(rho0, grid, nmodes, np.transpose(np.atleast_2d(N2_all)), f_ref)
vert_wave = sc_x / 1000
alpha = 10
mu = 1.88e-3 / (1 + 0.03222 * theta_avg + 0.002377 * theta_avg * theta_avg)
nu = mu / gsw.rho(salin_avg, cons_t_avg, grid_p)
avg_nu = np.nanmean(nu)

# ----------------------------------------------------------------------------------------------------------------------
# --- CURVE FITTING TO FIND BREAK IN SLOPES, WHERE WE MIGHT SEE -5/3 AND THEN -3
xx = sc_x.copy()
yy = avg_PE[1:] / dk
yy2 = avg_KE[1:] / dk
ipoint = 30  # 8  # 11
# fit slopes to PE and KE spectra (with a break point that I determine)
x_53 = np.log10(xx[0:ipoint+1])
y_53 = np.log10(yy[0:ipoint+1])
slope1 = np.polyfit(x_53, y_53, 1)
x_3 = np.log10(xx[ipoint-1:])
y_3 = np.log10(yy[ipoint-1:])
slope2 = np.polyfit(x_3, y_3, 1)
x_3_2 = np.log10(xx[0:55])
y_3_2 = np.log10(yy2[0:55])
slope_ke = np.polyfit(x_3_2, y_3_2, 1)
y_g_53 = np.polyval(slope1, x_53)
y_g_3 = np.polyval(slope2, x_3)
y_g_ke = np.polyval(slope_ke, x_3_2)

# --- Use function and iterate over each energy profile
TE_spectrum = (avg_PE / dk) + (avg_KE / dk)
TE_spectrum_per = (PE_per_mass / dk) + (HKE_per_mass / dk)
# find break for every profile
start_g = sc_x[5]
min_sp = np.nan * np.ones(PE_per_mass.shape[1])
enst_xfer_per = np.nan * np.ones(PE_per_mass.shape[1])
ener_xfer_per = np.nan * np.ones(PE_per_mass.shape[1])
enst_diss_per = np.nan * np.ones(PE_per_mass.shape[1])
rms_vort_per = np.nan * np.ones(PE_per_mass.shape[1])
# f, ax = plt.subplots()
for i in range(PE_per_mass.shape[1]):
    in_sp = np.transpose(np.concatenate([sc_x[:, np.newaxis], TE_spectrum_per[1:, i][:, np.newaxis]], axis=1))
    min_sp[i] = fmin(spectrum_fit, start_g, args=(tuple(in_sp)))

    this_TE = TE_spectrum_per[1:, i]
    x = np.log10(sc_x)
    pe = np.log10(this_TE)
    mid_p = np.log10(min_sp[i])
    l_b = np.nanmin(x)
    r_b = np.nanmax(x)
    x_grid = np.arange(l_b, r_b, 0.01)
    pe_grid = np.interp(x_grid, x, pe)
    first_over = np.where(x_grid > mid_p)[0][0]
    s1 = -5/3
    b1 = pe_grid[first_over] - s1 * x_grid[first_over]
    fit_53 = np.polyval(np.array([s1, b1]), x_grid[0:first_over + 1])
    s2 = -3
    b2 = pe_grid[first_over] - s2 * x_grid[first_over]
    fit_3 = np.polyval(np.array([s2, b2]), x_grid[first_over:])
    fit = np.concatenate((fit_53[0:-1], fit_3))

    ak0 = min_sp[i] / 1000  # xx[ipoint] / 1000
    E0 = np.interp(ak0 * 1000, sc_x, this_TE)  # np.mean(yy_tot[ipoint - 3:ipoint + 4])
    ak = vert_wave / ak0
    one = E0 * ((ak ** (5 * alpha / 3)) * (1 + ak ** (4 * alpha / 3))) ** (-1 / alpha)
    # -  enstrophy/energy transfers
    enst_xfer_per[i] = (E0 * (ak0 ** 3)) ** (3 / 2)
    ener_xfer_per[i] = (E0 * ak0 ** (5 / 3)) ** (3 / 2)
    enst_diss_per[i] = np.sqrt(avg_nu) / (enst_xfer_per[i] ** (1 / 6))
    rms_vort_per[i] = E0 * (ak0 ** 3) * (0.75 * (1 - (sc_x[0] / 1000) / ak0) ** (4/3) + np.log(enst_diss_per[i] / ak0))

# find break for average profile (total)
in_sp = np.transpose(np.concatenate([sc_x[:, np.newaxis], TE_spectrum[1:][:, np.newaxis]], axis=1))
min_sp_avg = fmin(spectrum_fit, start_g, args=(tuple(in_sp)))
this_TE = TE_spectrum[1:]
x = np.log10(sc_x)
pe = np.log10(this_TE)
mid_p = np.log10(min_sp_avg)
l_b = np.nanmin(x)
r_b = np.nanmax(x)
x_grid = np.arange(l_b, r_b, 0.01)
pe_grid = np.interp(x_grid, x, pe)
first_over = np.where(x_grid > mid_p)[0][0]
s1 = -5 / 3
b1 = pe_grid[first_over] - s1 * x_grid[first_over]
fit_53 = np.polyval(np.array([s1, b1]), x_grid[0:first_over + 1])
s2 = -3
b2 = pe_grid[first_over] - s2 * x_grid[first_over]
fit_3 = np.polyval(np.array([s2, b2]), x_grid[first_over:])
fit_total = np.concatenate((fit_53[0:-1], fit_3))

# closest mode number to ak0
sc_x_break_i = np.where(sc_x < min_sp_avg)[0][-1]

# --- cascade rates (for average TE spectrum)
ak0 = min_sp_avg / 1000  # xx[ipoint] / 1000
E0 = np.interp(ak0 * 1000, sc_x, TE_spectrum[1:])  # np.mean(yy_tot[ipoint - 3:ipoint + 4])
ak = vert_wave / ak0
one = E0 * ((ak ** (5 * alpha / 3)) * (1 + ak ** (4 * alpha / 3))) ** (-1 / alpha)
# ---  enstrophy/energy transfers
enst_xfer = (E0 * ak0 ** 3) ** (3 / 2)
ener_xfer = (E0 * ak0 ** (5 / 3)) ** (3 / 2)
enst_diss = np.sqrt(avg_nu) / (enst_xfer ** (1 / 6))
rms_vort = E0 * (ak0 **3) * (0.75*(1 - (sc_x[0] / 1000)/ak0)**(4/3) + np.log(enst_diss / ak0))
rms_ener = E0 * (ak0) * ( -3/2 + 3/2*( (ak0 ** (2/3))*((sc_x[0] / 1000) ** (-2/3))) -
                          0.5 * (ak0 ** 2) * (enst_diss ** -2) + 0.5 * ak0 ** 4)

# --- rhines scale
r_earth = 6371e3  # earth radius [m]
beta_ref = f_ref / (np.tan(np.deg2rad(ref_lat)) * r_earth)
# K_beta = 1 / np.sqrt(np.sqrt(np.sum(avg_KE)) / beta_ref)
K_beta = 1 / np.sqrt(np.sqrt(rms_ener) / beta_ref)
K_beta_2 = 1 / np.sqrt(np.sqrt(np.nanmean(V3**2)) / beta_ref)
non_linearity = np.sqrt(rms_ener) / (beta_ref * ((c[1] / f_ref) ** 2))
non_linearity_2 = np.sqrt(np.nanmean(V3**2)) / (beta_ref * ((c[1] / f_ref) ** 2))

# ----- LOAD in Comparison DATA
# -- load in Station BATs PE Comparison
pkl_file = open('/Users/jake/Desktop/bats/station_bats_pe_jan30.pkl', 'rb')
SB = pickle.load(pkl_file)
pkl_file.close()
sta_bats_pe = SB['PE_by_season']
sta_bats_c = SB['c']
sta_bats_f = np.pi * np.sin(np.deg2rad(31.6)) / (12 * 1800)
sta_bats_dk = sta_bats_f / sta_bats_c[1]

# seasonal and variable spread at bats station for each mode
sta_max = np.nan * np.ones(len(sc_x))
sta_min = np.nan * np.ones(len(sc_x))
dg_per_max = np.nan * np.ones(len(sc_x))
dg_per_min = np.nan * np.ones(len(sc_x))
for i in range(1, mmax+1):
    test1 = np.nanmean(sta_bats_pe[0][i, :])
    test2 = np.nanmean(sta_bats_pe[1][i, :])
    test3 = np.nanmean(sta_bats_pe[2][i, :])
    # test4 = np.nanmean(sta_bats_pe[3][i, :])
    sta_max[i - 1] = np.max([test1, test2, test3])
    sta_min[i - 1] = np.min([test1, test2, test3])
    dg_per_max[i - 1] = np.nanmax(PE_per_mass_all[i, :])
    dg_per_min[i - 1] = np.nanmin(PE_per_mass_all[i, :])

# ------------------------------------------------------------------------------------------------------------------
plot_eng = 1
if plot_eng > 0:
    fig0, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
    # Station (by season)
    ax0.fill_between(1000 * sta_bats_f / sta_bats_c[1:mmax+1], sta_min / sta_bats_dk, sta_max / sta_bats_dk,
                     label='APE$_{sta. BATS}$', color='#D3D3D3')
    # PE_sta_p = ax0.plot(1000 * sta_bats_f / sta_bats_c[1:], np.nanmean(sta_bats_pe[0][1:, :], axis=1) / sta_bats_dk,
    #                     color='#7B68EE', label='APE$_{ship_{spr}}$', linewidth=1)
    # ax0.plot(1000 * sta_bats_f / sta_bats_c[1:], np.nanmean(sta_bats_pe[1][1:, :], axis=1) / sta_bats_dk,
    #                     color='#6495ED', label='APE$_{ship_{sum}}$', linewidth=1)
    # ax0.plot(1000 * sta_bats_f / sta_bats_c[1:], np.nanmean(sta_bats_pe[2][1:, :], axis=1) / sta_bats_dk,
    #                     color='b', label='APE$_{ship_{f}}$', linewidth=1)
    # ax0.plot(1000 * sta_bats_f / sta_bats_c[1:], np.nanmean(sta_bats_pe[3][1:, :], axis=1) / sta_bats_dk,
    #                     color='#000080', label='APE$_{ship_{w}}$', linewidth=1)

    # DG PE averaging to find eta
    PE_p = ax0.plot(sc_x, avg_PE[1:] / dk, color='#B22222', label='APE$_{DG}$', linewidth=3)
    ax0.scatter(sc_x, avg_PE[1:] / dk, color='#B22222', s=20)
    # ax0.plot(sc_x, PE_per_mass[1:, PE_i_max] / dk, color='#B22222', linewidth=1)
    # ax0.plot(sc_x, PE_per_mass[1:, PE_i_min] / dk, color='#B22222', linewidth=1)
    # DG PE individual profiles
    PE_p = ax0.plot(1000 * f_ref / c_all[1:], np.nanmean(PE_per_mass_all[1:, :], axis=1) / (f_ref / c_all[1]),
                    color='#B22222', label='APE$_{DG_{ind.}}$', linewidth=1.5, linestyle='--')
    # ax0.fill_between(1000 * f_ref / c_all[1:], dg_per_min / (f_ref / c_all[1]), dg_per_max / (f_ref / c_all[1]),
    #                  label='APE$_{DG_{ind.}}$', color='y')

    # DG KE
    KE_p = ax1.plot(1000 * f_ref / c[1:], avg_KE[1:] / dk, 'g', label='KE$_{DG}$', linewidth=3)
    ax1.scatter(sc_x, avg_KE[1:] / dk, color='g', s=20)  # DG KE
    KE_p = ax1.plot([10**-2, 1000 * f_ref / c[1]], avg_KE[0:2] / dk, 'g', linewidth=3) # DG KE_0
    ax1.scatter(10**-2, avg_KE[0] / dk, color='g', s=25, facecolors='none')  # DG KE_0
    # -- max / min
    # ax1.plot(1000 * f_ref / c[1:], KE_i[1:, KE_i_max] / dk, 'g', label='KE$_{DG}$', linewidth=2)
    # ax1.plot(1000 * f_ref / c[1:], KE_i[1:, KE_i_min] / dk, 'g', label='KE$_{DG}$', linewidth=2)

    # -- Eddy energies
    # PE_e = ax0.plot(sc_x, PE_ed[1:] / dk, color='c', label='eddy PE', linewidth=2)
    # KE_e = ax0.plot(sc_x, KE_ed[1:] / dk, color='y', label='eddy KE', linewidth=2)
    # KE_e = ax0.plot([10**-2, 1000 * f_ref / c[1]], KE_ed[0:2] / dk, color='y', label='eddy KE', linewidth=2)

    # -- Slope fits
    # ax0.plot(10 ** x_grid, 10 ** fit_total, color='#FF8C00', label=r'APE$_{fit}$')
    # PE
    ax0.plot(10 ** x_53, 10 ** y_g_53, color='b', linewidth=1.25)
    # ax0.plot(10 ** x_3, 10 ** y_g_3, color='b', linewidth=1.25)
    ax0.text(10 ** x_53[0] - .012, 10 ** y_g_53[0], str(float("{0:.2f}".format(slope1[0]))), fontsize=10)
    # ax0.text(10 ** x_3[0] - .11, 10 ** y_g_3[0], str(float("{0:.2f}".format(slope2[0]))), fontsize=10)
    # KE
    ax1.plot(10 ** x_3_2, 10 ** y_g_ke, color='b', linewidth=1.5, linestyle='--')
    ax1.text(10 ** x_3_2[1] + .01, 10 ** y_g_ke[1], str(float("{0:.2f}".format(slope_ke[0]))), fontsize=12)

    # ax0.scatter(vert_wave[ipoint] * 1000, one[ipoint], color='b', s=7)
    # ax0.plot([xx[ipoint], xx[ipoint]], [10 ** (-4), 4 * 10 ** (-4)], color='k', linewidth=2)
    # ax0.text(xx[ipoint + 1], 2 * 10 ** (-4),
    #          str('Break at ') + str(float("{0:.1f}".format(1 / xx[ipoint]))) + 'km')

    # -- Rossby Radii
    ax0.plot([sc_x[0], sc_x[0]], [10 ** (-4), 3 * 10 ** (-4)], color='k', linewidth=2)
    ax0.text(sc_x[0] - .6 * 10 ** -2, 5 * 10 ** (-4),
             str(r'$c_1/f$ = ') + str(float("{0:.1f}".format(1 / sc_x[0]))) + 'km', fontsize=8)
    ax0.plot([sc_x[sc_x_break_i], sc_x[sc_x_break_i]], [10 ** (-4), 3 * 10 ** (-4)], color='k', linewidth=2)
    ax0.text(sc_x[sc_x_break_i] - .4 * 10 ** -2, 5 * 10 ** (-4),
             'Break at Mode ' + str(sc_x_break_i + 1) + ' = ' + str(float("{0:.1f}".format(1 / sc_x[sc_x_break_i]))) + 'km',
             fontsize=8)
    # ax1.plot([sc_x[0], sc_x[0]], [10 ** (-4), 3 * 10 ** (-4)], color='k', linewidth=2)
    # ax1.text(sc_x[0] - .6 * 10 ** -2, 5 * 10 ** (-4),
    #          str(r'$c_1/f$ = ') + str(float("{0:.1f}".format(1 / sc_x[0]))) + 'km', fontsize=8)
    # ax1.plot([sc_x[sc_x_break_i], sc_x[sc_x_break_i]], [10 ** (-4), 3 * 10 ** (-4)], color='k', linewidth=2)
    # ax1.text(sc_x[sc_x_break_i] - .4 * 10 ** -2, 5 * 10 ** (-4),
    #          'Break at Mode ' + str(sc_x_break_i) + ' = ' + str(float("{0:.1f}".format(1 / sc_x[sc_x_break_i]))) + 'km',
    #          fontsize=8)

    # GM
    # ax0.plot(sc_x, 0.25 * PE_GM / dk, linestyle='--', color='k', linewidth=0.75)
    ax0.plot(sc_x, 0.25 * GMPE / dk, color='k', linewidth=0.75)
    ax0.text(sc_x[0] - .01, 0.5 * PE_GM[1] / dk, r'$1/4 PE_{GM}$', fontsize=10)
    # ax0.plot(np.array([10**-2, 10]), [PE_SD / dk, PE_SD / dk], linestyle='--', color='k', linewidth=0.75)
    ax1.plot(sc_x, 0.25 * GMKE / dk, color='k', linewidth=0.75)
    ax1.text(sc_x[0] - .01, 0.5 * GMKE[1] / dk, r'$1/4 KE_{GM}$', fontsize=10)

    # ax0.axis([10 ** -2, 10 ** 1, 10 ** (-4), 2 * 10 ** 3])
    ax0.set_xlim([10 ** -2, 2 * 10 ** 0])
    ax0.set_ylim([10 ** (-4), 1 * 10 ** 3])
    ax0.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=14)
    ax0.set_ylabel('Spectral Density', fontsize=14)  # ' (and Hor. Wavenumber)')
    ax0.set_title('Potential Energy Spectrum', fontsize=14)
    ax0.set_yscale('log')
    ax0.set_xscale('log')

    ax1.set_xlim([10 ** -2, 2 * 10 ** 0])
    ax1.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=14)
    ax1.set_ylabel('Spectral Density', fontsize=14)  # ' (and Hor. Wavenumber)')
    ax1.set_title('Kinetic Energy Spectrum', fontsize=14)
    ax1.set_xscale('log')
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, fontsize=12)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=12)
    ax0.grid()
    plot_pro(ax1)
    # fig0.savefig('/Users/jake/Desktop/bats/dg035_15_PE_b.png',dpi = 300)
    # plt.close()
    # plt.show()

    # -------------------------------
    # -----------------------------------------------------------------------------------------------------------
    # additional plot to highlight the ratio of KE to APE to predict the scale of motion
    fig0, (ax0, ax1) = plt.subplots(1, 2)
    # Limits/scales
    ax0.plot([1000 * f_ref / c[1], 1000 * f_ref / c[-2]], [1000 * f_ref / c[1], 1000 * f_ref / c[-2]], linestyle='--',
             color='k', linewidth=1.5, zorder=2, label=r'$L_{d_n}^{-1}$')
    ax0.text(1000 * f_ref / c[-2] + .1, 1000 * f_ref / c[-2], r'f/c$_n$', fontsize=14)
    ax0.plot(sc_x, k_h, color='k', label=r'$k_h$', linewidth=1.5)
    xx_fill = 1000 * f_ref / c[1:]
    yy_fill = 1000 * f_ref / c[1:]
    # ax0.fill_between(xx_fill, yy_fill, k_h, color='b',interpolate=True)
    ax0.fill_between(xx_fill, yy_fill, k_h, where=yy_fill >= k_h, facecolor='#FAEBD7', interpolate=True, alpha=0.75)
    ax0.fill_between(xx_fill, yy_fill, k_h, where=yy_fill <= k_h, facecolor='#6B8E23', interpolate=True, alpha=0.75)
    # ax0.plot(sc_x, k_h_max, color='k', label=r'$k_h_{max}$', linewidth=0.5)
    # ax0.plot(sc_x, k_h_min, color='k', label=r'$k_h_{min}$', linewidth=0.5)
    ax0.plot([10**-2, 10**1], 1e3 * np.array([K_beta_2, K_beta_2]), color='k', linestyle='-.')
    ax0.text(1.1, 0.025, r'k$_{Rhines}$', fontsize=12)

    ax0.set_yscale('log')
    ax0.set_xscale('log')
    # ax0.axis([10**-2, 10**1, 3*10**(-4), 2*10**(3)])
    ax0.axis([10 ** -2, 10 ** 1, 10 ** (-3), 10 ** 3])
    ax0.set_title('Predicted Horizontal Length Scale (KE/APE)', fontsize=12)
    ax0.set_xlabel(r'Inverse Deformation Radius ($L_{d_n}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=11)
    ax0.set_ylabel(r'Horizontal Wavenumber [$km^{-1}$]', fontsize=12)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend([handles[0], handles[1]], [labels[0], labels[1]], fontsize=14)
    ax0.set_aspect('equal')
    ax0.grid()

    k_xx = k_h.copy()
    yy = avg_PE[1:] / dk
    yy2 = avg_KE[1:] / dk
    x_3h = np.log10(k_xx[0:55])
    y_3p = np.log10(yy[0:55])
    y_3h = np.log10(yy2[0:55])
    slope_pe_h = np.polyfit(x_3h, y_3p, 1)
    y_g_pe_h = np.polyval(slope_pe_h, x_3h)
    slope_ke_h = np.polyfit(x_3h, y_3h, 1)
    y_g_ke_h = np.polyval(slope_ke_h, x_3h)
    ax1.scatter(k_h, avg_PE[1:] / dk, 5, color='#B22222')
    for i in range(6):
        ax1.text(k_h[i] + k_h[i]/5, avg_PE[i + 1] / dk, str(i + 1), color='#B22222', fontsize=6)
        ax1.text(k_h[i] - k_h[i] / 8, avg_KE[i + 1] / dk, str(i + 1), color='g', fontsize=6)
    ax1.scatter(k_h, avg_KE[1:] / dk, 5, color='g')
    # ax1.scatter(k_h, ke_om_tot[1:] / dk_om, 5, color='b')
    ax1.plot(10 ** x_3h, 10 ** y_g_pe_h, color='#B22222', linewidth=1.5, linestyle='--', label='PE')
    ax1.text(10 ** x_3h[3] + .02, 10 ** y_g_pe_h[3], str(float("{0:.2f}".format(slope_pe_h[0]))), fontsize=12)
    ax1.plot(10 ** x_3h, 10 ** y_g_ke_h, color='g', linewidth=1.5, linestyle='--', label='KE')
    ax1.text(10 ** x_3h[3] + .01, 10 ** y_g_ke_h[3], str(float("{0:.2f}".format(slope_ke_h[0]))), fontsize=12)
    ax1.axis([10 ** -2, 10 ** 0, 10 ** (-4), 10 ** 3])
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=12)
    ax1.set_xlabel(r'Horizontal Wavenumber [$km^{-1}$]', fontsize=11)
    ax1.set_title('Energy per Implied Hor. Scale')
    ax1.set_ylabel('Energy')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    plot_pro(ax1)

# --- SAVE 36N ENERGIES DG TRANSECTS
# write python dict to a file
sa = 0
if sa > 0:
    my_dict = {'depth': grid, 'KE': avg_KE, 'PE': avg_PE, 'c': c, 'f': f_ref, 'N2': N2_all}
    output = open('/Users/jake/Documents/baroclinic_modes/DG/sg041_energy.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()

# - load BATS
pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg035_2015_energy.pkl', 'rb')
DGB = pickle.load(pkl_file)
pkl_file.close()
bats_dg_KE = DGB['KE']
bats_dg_PE = DGB['PE']
bats_dg_bckgrds = DGB['background_eddy_indicies_for_energy'][0]  # winter index
bats_dg_KE_all = np.nanmean(DGB['KE_all'][:, bats_dg_bckgrds], axis=1)
bats_dg_PE_all = np.nanmean(DGB['PE_all'][:, bats_dg_bckgrds], axis=1)
bats_dg_c = DGB['c']
bats_dg_f = DGB['f']
bats_dg_depth = DGB['depth']
bats_dg_GMKE = DGB['GMKE']
bats_dg_GMPE = DGB['GMPE']
dk_bats = bats_dg_f / bats_dg_c[1]
sc_x_bats = 1000 * bats_dg_f / bats_dg_c[1:]
# ----------------------------------------------------------------------------------------------------------------------
# - comparison plot of 36N and BATS
fig0, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

# Station (by season)
ax0.fill_between(1000 * sta_bats_f / sta_bats_c[1:mmax + 1], sta_min / sta_bats_dk, sta_max / sta_bats_dk,
                 label='PE$_{ship}$', color='#D3D3D3')
scols = ['#00BFFF', '#6B8E23', '#800000']
# DG PE avg. (36N)
PE_p = ax1.plot(sc_x, avg_PE[1:] / dk, color=scols[0], label='PE', linewidth=3)
ax1.scatter(sc_x, avg_PE[1:] / dk, color=scols[0], s=20)
# DG KE (36N)
KE_p = ax1.plot(1000 * f_ref / c[1:], avg_KE[1:] / dk, color=scols[2], label='KE', linewidth=3)
ax1.scatter(sc_x, avg_KE[1:] / dk, color=scols[2], s=20)                                         # DG KE
KE_p = ax1.plot([10 ** -2, 1000 * f_ref / c[1]], avg_KE[0:2] / dk, color=scols[2], linewidth=3)        # DG KE_0
ax1.scatter(10 ** -2, avg_KE[0] / dk, color=scols[2], s=20, facecolors='none')                   # DG KE_0

# DG PE avg. (BATS)
ax0.plot(sc_x_bats, bats_dg_PE_all[1:] / dk_bats, color=scols[0], label='PE', linewidth=3)
ax0.scatter(sc_x_bats, bats_dg_PE_all[1:] / dk_bats, color=scols[0], s=20)
# DG KE (BATS)
ax0.plot(1000 * bats_dg_f / bats_dg_c[1:], bats_dg_KE_all[1:] / dk_bats, color=scols[2], label=r'KE', linewidth=3)
ax0.scatter(sc_x_bats, bats_dg_KE_all[1:] / dk_bats, color=scols[2], s=20)                                         # DG KE
ax0.plot([10 ** -2, 1000 * bats_dg_f / bats_dg_c[1]], bats_dg_KE_all[0:2] / dk_bats, color=scols[2], linewidth=3)        # DG KE_0
ax0.scatter(10 ** -2, bats_dg_KE_all[0] / dk_bats, color=scols[2], s=20, facecolors='none')                   # DG KE_0
nums = '1', '2', '3', '4', '5', '6', '7', '8'
for i in range(1, 8):
    ax0.text( (1000 * bats_dg_f / bats_dg_c[i]) - (1/10)*(1000 * bats_dg_f / bats_dg_c[i]),
              (bats_dg_KE_all[i] / dk_bats) - (1/4)*(bats_dg_KE_all[i] / dk_bats), nums[i - 1], fontsize=7, color='k')
    ax0.text( (1000 * bats_dg_f / bats_dg_c[i]),
              (bats_dg_PE_all[i] / dk_bats) + (1/7)*(bats_dg_PE_all[i] / dk_bats), nums[i - 1], fontsize=7, color='k')
            #  + (1/15)*(1000 * bats_dg_f / bats_dg_c[i])

    ax1.text((1000 * f_ref / c[i]) - (1/10)*(1000 * f_ref / c[i]),
              (avg_KE[i] / dk) - (1/4)*(avg_KE[i] / dk), nums[i - 1], fontsize=7, color='k')
    ax1.text((1000 * f_ref / c[i]),
              (avg_PE[i] / dk) + (1/7)*(avg_PE[i] / dk), nums[i - 1], fontsize=7, color='k')
              #  + (1/15)*(1000 * f_ref / c[i])
ax0.text(9 * 10 ** -3, (bats_dg_KE_all[0] / dk_bats) - 5, '0', color='k', fontsize=7)
ax1.text(9 * 10 ** -3, (avg_KE[0] / dk) - 5, '0', color='k', fontsize=7)

# GM
# ax0.plot(sc_x, 0.25 * PE_GM / dk, linestyle='--', color='k', linewidth=0.75)
ax1.plot(sc_x, 0.25 * GMPE / dk, color=scols[0], linewidth=0.75, linestyle='--')
ax1.text(sc_x[0] - .01, 0.4 * GMPE[1] / dk, r'$\frac{1}{4}$PE$_{GM}$', fontsize=8)
# ax0.plot(np.array([10**-2, 10]), [PE_SD / dk, PE_SD / dk], linestyle='--', color='k', linewidth=0.75)
ax1.plot(sc_x, 0.25 * GMKE / dk, color=scols[2], linewidth=0.75, linestyle='--')
ax1.text(sc_x[0] - .01, 0.4 * GMKE[1] / dk, r'$\frac{1}{4}$KE$_{GM}$', fontsize=8)

# ax0.plot(sc_x, 0.25 * PE_GM / dk, linestyle='--', color='k', linewidth=0.75)
ax0.plot(sc_x_bats, 0.25 * bats_dg_GMPE / dk_bats, color=scols[0], linewidth=0.75, linestyle='--')
ax0.text(sc_x_bats[0] - .01, 0.4 * bats_dg_GMPE[1] / dk_bats, r'$\frac{1}{4}$PE$_{GM}$', fontsize=8)
# ax0.plot(np.array([10**-2, 10]), [PE_SD / dk, PE_SD / dk], linestyle='--', color='k', linewidth=0.75)
ax0.plot(sc_x_bats, 0.25 * bats_dg_GMKE / dk_bats, color=scols[2], linewidth=0.75, linestyle='--')
ax0.text(sc_x_bats[0] - .01, 0.4 * bats_dg_GMKE[1] / dk_bats, r'$\frac{1}{4}$KE$_{GM}$', fontsize=8)

# slopess
ax0.plot([10**-2, 10**0], [10**3, 10**-3], color='k', linewidth=0.5)
ax0.plot([10**-2, 10**1], [3*10**2, 3*10**-4], color='k', linewidth=0.5)
ax0.text(2*10**0, 3*10**-3, '-2', fontsize=10)
ax0.text(4.5*10**-1, 2*10**-3, '-3', fontsize=10)
ax1.plot([10**-2, 10**0], [10**3, 10**-3], color='k', linewidth=0.5)
ax1.plot([10**-2, 10**1], [3*10**2, 3*10**-4], color='k', linewidth=0.5)
ax1.text(2*10**0, 3*10**-3, '-2', fontsize=10)
ax1.text(4.5*10**-1, 2*10**-3, '-3', fontsize=10)

handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles, labels, fontsize=12)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=12)
ax0.set_xlim([10 ** -2, 3 * 10 ** 0])
ax0.set_ylim([1 * 10 ** (-3), 1 * 10 ** 3])
ax1.set_xlim([10 ** -2, 3 * 10 ** 0])
ax0.set_yscale('log')
ax0.set_xscale('log')
ax1.set_xscale('log')
ax0.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
ax0.set_ylabel('Variance per Vertical Wavenumber', fontsize=12)  # ' (and Hor. Wavenumber)')
ax0.set_title('BATS Winter (DG035, 2015)', fontsize=14)
ax1.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
ax1.set_title(r'36$^{\circ}$N (DG041 dive-cycles 50:110, 2018-)', fontsize=14)
ax0.grid()
plot_pro(ax1)

# fig0.savefig('/Users/jake/Documents/baroclinic_modes/Meetings/meeting_end_of_18/bats_36_comparison.pdf')

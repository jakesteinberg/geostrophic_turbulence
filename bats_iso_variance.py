# BATS
# take velocity and displacement profiles and compute energy spectra / explore mode amplitude variability
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gsw
import scipy
import scipy.io as si
from scipy.optimize import fmin
from scipy.signal import savgol_filter
from netCDF4 import Dataset
import pickle
import datetime
# functions I've written
from glider_cross_section import Glider
from mode_decompositions import eta_fit, vertical_modes, PE_Tide_GM, vertical_modes_f
from toolkit import spectrum_fit, nanseg_interp, plot_pro


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
mmax = 50
nmodes = mmax + 1
# maximum allowed deep shear [m/s/km]
deep_shr_max = 0.1
# minimum depth for which shear is limited [m]
deep_shr_max_dep = 3500

# --- BIN PARAMETERS
GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
bin_depth = GD.variables['grid'][:]
grid = bin_depth
grid_p = gsw.p_from_z(-1 * grid, ref_lat)
z = -1 * grid
sz_g = grid.shape[0]

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ---- PROCESSING USING GLIDER PACKAGE
x = Glider(35, np.arange(30, 165), '/Users/jake/Documents/baroclinic_modes/DG/BATS_2015/sg035')
# -------------------------------------------------------------------------------------------------
# Vertically Bin
# Binned = x.make_bin(grid)
# profile_tags_0 = Binned['profs']
# d_time_0 = Binned['time']
# lon = Binned['lon']
# lat = Binned['lat']
# t = Binned['temp']
# s = Binned['sal']
# dac_u = Binned['dac_u']
# dac_v = Binned['dac_v']
# if 'o2' in Binned.keys():
#     o2 = Binned['o2']
# ref_lat = np.nanmean(lat)
# time_rec_bin = np.nanmean(d_time, axis=0)
# -------------------------------------------------------------------------------------------------
# Test alternate density computation
import_dg = si.loadmat('/Users/jake/Documents/baroclinic_modes/sg035_2015_neutral_density_bin.mat')
dg_data = import_dg['out']
neutral_density = dg_data['Neut_den'][0][0][0:240, :]
t = dg_data['Temp'][0][0][0:240, :]
s = dg_data['Sal'][0][0][0:240, :]
lon = dg_data['Lon'][0][0][0:240, :]
lat = dg_data['Lat'][0][0][0:240, :]
dac_u = dg_data['Dac_u'][0][0][0]
dac_v = dg_data['Dac_v'][0][0][0]
d_time = dg_data['Time'][0][0][0:240, :] - 366
profile_tags = dg_data['prof_number'][0][0][0]
ref_lat = np.nanmean(lat)
time_rec_bin = np.nanmean(d_time, axis=0)
# -------------------------------------------------------------------------------------------------
# -- Compute density
sa, ct, theta, sig0, sig2, dg_N2 = x.density(grid, ref_lat, t, s, lon, lat)
# -------------------------------------------------------------------------------------------------
# -- use to compute background profiles
t_s = datetime.date.fromordinal(np.int(np.nanmin(d_time)))
t_e = datetime.date.fromordinal(np.int(np.nanmax(d_time)))
# --- compute 4 seasonal averages (as with station bats)
# -- construct four background profiles to represent seasons
d_spring = np.where((d_time > 735658) & (d_time < 735750))[0]       # Mar 1 - June 1
d_summer = np.where((d_time > 735750) & (d_time < 735842))[0]       # June 1 - Sept 1
d_fall = np.where((d_time > 735842) & (d_time < 735903))[0]         # Sept 1 - Nov 1
d_winter = np.where((d_time > 735903) | (d_time < 735658))[0]       # Nov 1 - Mar 1
bckgrds = [d_spring, d_summer, d_fall, d_winter]
bckgrds_wins = np.array([735658, 735750, 735842, 735903])
salin_avg = np.nan * np.zeros((len(grid), 4))
cons_t_avg = np.nan * np.zeros((len(grid), 4))
theta_avg = np.nan * np.zeros((len(grid), 4))
sigma_theta_avg = np.nan * np.zeros((len(grid), 4))
ddz_avg_sigma = np.nan * np.zeros((len(grid), 4))
N2 = np.nan * np.zeros(sigma_theta_avg.shape)
N = np.nan * np.zeros(sigma_theta_avg.shape)
for i in range(4):
    inn = bckgrds[i]
    salin_avg[:, i] = np.nanmean(sa[:, inn], axis=1)
    cons_t_avg[:, i] = np.nanmean(ct[:, inn], axis=1)
    theta_avg[:, i] = np.nanmean(theta[:, inn], axis=1)
    sigma_theta_avg[:, i] = np.nanmean(neutral_density[:, inn], axis=1)
    ddz_avg_sigma[:, i] = np.gradient(sigma_theta_avg[:, i], z)
    go = ~np.isnan(salin_avg[:, i])
    N2[np.where(go)[0][0:-1], i] = gsw.Nsquared(salin_avg[go, i], cons_t_avg[go, i], grid_p[go], lat=ref_lat)[0]
    N2[N2[:, i] < 0] = np.nan
    N2[:, i] = nanseg_interp(grid, N2[:, i])
N2_all = np.nan * np.zeros(len(grid))
N2_all[0:-1] = gsw.Nsquared(np.nanmean(salin_avg, axis=1), np.nanmean(cons_t_avg, axis=1), grid_p, lat=ref_lat)[0]
N2_all[-2:] = N2_all[-3]
N2_all[N2_all < 0] = np.nan
N2_all = nanseg_interp(grid, N2_all)
N_all = np.sqrt(N2_all)
N2_all = savgol_filter(N2_all, 5, 3)
# -------------------------------------------------------------------------------------------------
# -- compute M/W sections and compute velocity
# -- USING X.TRANSECT_CROSS_SECTION_1 (THIS WILL SEPARATE TRANSECTS BY TARGET OF EACH DIVE)
sigth_levels = np.concatenate(
    [np.arange(23, 26.5, 0.5), np.arange(26.2, 27.2, 0.2),
     np.arange(27.2, 27.7, 0.2), np.arange(27.7, 28, 0.02), np.arange(28, 28.15, 0.01)])
# sigth_levels = np.concatenate([np.aranger(32, 36.6, 0.2), np.arange(36.6, 36.8, 0.05), np.arange(36.8, 37.4, 0.02)])

# --- SAVE so that we dont have to run transects every time
savee = 1
if savee > 0:
    ds, dist, avg_ct_per_dep_0, avg_sa_per_dep_0, avg_sig0_per_dep_0, v_g, vbt, isopycdep, isopycx, mwe_lon, mwe_lat,\
    DACe_MW, DACn_MW, profile_tags_per, shear = x.transect_cross_section_1(grid, neutral_density, ct, sa, lon, lat,
                                                                           dac_u, dac_v, profile_tags, sigth_levels)
    my_dict = {'ds': ds, 'dist': dist, 'avg_ct_per_dep_0': avg_ct_per_dep_0,
               'avg_sa_per_dep_0': avg_sa_per_dep_0, 'avg_sig0_per_dep_0': avg_sig0_per_dep_0, 'v_g': v_g, 'vbt': vbt,
               'isopycdep': isopycdep, 'isopycx': isopycx, 'mwe_lon': mwe_lon, 'mwe_lat': mwe_lat, 'DACe_MW': DACe_MW,
               'DACn_MW': DACn_MW, 'profile_tags_per': profile_tags_per}
    output = open('/Users/jake/Documents/baroclinic_modes/DG/sg035_2015_transects_sig2_test.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()
else:
    pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg035_2015_transects_sig2.pkl', 'rb')
    B15 = pickle.load(pkl_file)
    pkl_file.close()
    ds = B15['ds']
    dist = B15['dist']
    avg_ct_per_dep_0 = B15['avg_ct_per_dep_0']
    avg_sa_per_dep_0 = B15['avg_sa_per_dep_0']
    avg_sig0_per_dep_0 = B15['avg_sig0_per_dep_0']
    v_g = B15['v_g']
    vbt = B15['vbt']
    isopycdep = B15['isopycdep']
    isopycx = B15['isopycx']
    mwe_lon = B15['mwe_lon']
    mwe_lat = B15['mwe_lat']
    DACe_MW = B15['DACe_MW']
    DACn_MW = B15['DACn_MW']
    profile_tags_per = B15['profile_tags_per']

# unpack velocity profiles from transect analysis
dg_v_0 = v_g[0][:, 0:-1].copy()
avg_sig0_per_dep = avg_sig0_per_dep_0[0].copy()
avg_ct_per_dep = avg_ct_per_dep_0[0].copy()
avg_sa_per_dep = avg_sa_per_dep_0[0].copy()
dg_v_lon = mwe_lon[0][0:-1].copy()
dg_v_lat = mwe_lat[0][0:-1].copy()
dg_v_dive_no = profile_tags_per[0][0:-1].copy()
for i in range(1, len(v_g)):
    dg_v_0 = np.concatenate((dg_v_0, v_g[i][:, 0:-1]), axis=1)
    avg_ct_per_dep = np.concatenate((avg_ct_per_dep, avg_ct_per_dep_0[i]), axis=1)
    avg_sa_per_dep = np.concatenate((avg_sa_per_dep, avg_sa_per_dep_0[i]), axis=1)
    avg_sig0_per_dep = np.concatenate((avg_sig0_per_dep, avg_sig0_per_dep_0[i]), axis=1)
    dg_v_lon = np.concatenate((dg_v_lon, mwe_lon[i][0:-1]))
    dg_v_lat = np.concatenate((dg_v_lat, mwe_lat[i][0:-1]))
    dg_v_dive_no = np.concatenate((dg_v_dive_no, profile_tags_per[i][0:-1]))

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

# ----- Eta compute from M/W method, which produces an average density per set of profiles
eta_alt = np.nan * np.ones(np.shape(avg_sig0_per_dep))
eta_alt_2 = np.nan * np.ones(np.shape(avg_sig0_per_dep))
d_anom_alt = np.nan * np.ones(np.shape(avg_sig0_per_dep))
gradient_alt = np.nan * np.ones(np.shape(avg_sig0_per_dep))
for i in range(np.shape(avg_sig0_per_dep)[1]):  # loop over each profile
    # (average of four profiles) - (total long term average, that is seasonal)
    this_time = dg_mw_time[i]
    t_over = np.where(bckgrds_wins > this_time)[0]
    # find appropriate average background profiles
    if len(t_over) > 1:
        avg_a_salin = salin_avg[:, t_over[0]]
        avg_c_temp = cons_t_avg[:, t_over[0]]
    else:
        avg_a_salin = salin_avg[:, 3]
        avg_c_temp = cons_t_avg[:, 3]

    # loop over each bin depth
    for j in range(1, len(grid) - 1):
        # profile density at depth j with local
        this_sigma = gsw.rho(avg_sa_per_dep[j, i], avg_ct_per_dep[j, i], grid_p[j]) - 1000      # profile density
        # background density with local reference pressure
        this_sigma_avg = gsw.rho(avg_a_salin[j-1:j+2], avg_c_temp[j-1:j+2], grid_p[j]) - 1000
        d_anom_alt[j, i] = this_sigma - this_sigma_avg[1]
        gradient_alt[j, i] = np.nanmean(np.gradient(this_sigma_avg, z[j-1:j+2]))
        eta_alt_2[j, i] = d_anom_alt[j, i] / gradient_alt[j, i]

    if len(t_over) > 1:
        if t_over[0] == 1:
            eta_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg[:, 0]) / np.squeeze(ddz_avg_sigma[:, 0])
            d_anom_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg[:, 0])
        elif t_over[0] == 2:
            eta_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg[:, 1]) / np.squeeze(ddz_avg_sigma[:, 1])
            d_anom_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg[:, 1])
        elif t_over[0] == 3:
            eta_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg[:, 2]) / np.squeeze(ddz_avg_sigma[:, 2])
            d_anom_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg[:, 2])
        else:
            eta_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg[:, 3]) / np.squeeze(ddz_avg_sigma[:, 3])
            d_anom_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg[:, 3])
    else:
        eta_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_theta_avg[:, 3]) / np.squeeze(ddz_avg_sigma[:, 3])

# FILTER VELOCITY PROFILES IF THEY ARE TOO NOISY / BAD -- ALSO HAVE TO REMOVE EQUIVALENT ETA PROFILE
good_v = np.zeros(np.shape(dg_v_0)[1], dtype=bool)
for i in range(np.shape(dg_v_0)[1]):
    dv_dz = np.gradient(dg_v_0[:, i], -1 * grid)
    if (np.nanmax(np.abs(dv_dz)) < 0.02):  # & ((i < 28) | (i > 28)):
        good_v[i] = True

avg_sig = avg_sig0_per_dep[:, good_v]
eta_alt = eta_alt[:, good_v]
dg_v = dg_v_0[:, good_v]
dg_mw_time = dg_mw_time[good_v]
dg_v_dive_no = dg_v_dive_no[good_v]
num_mw_profs = np.shape(eta_alt)[1]

# Smooth DG N2 profiles
dg_avg_N2_coarse = np.nanmean(dg_N2, axis=1)
dg_avg_N2_coarse[np.isnan(dg_avg_N2_coarse)] = dg_avg_N2_coarse[~np.isnan(dg_avg_N2_coarse)][0] - 1*10**(-5)
dg_avg_N2 = savgol_filter(dg_avg_N2_coarse, 15, 3)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING cross section (CHECK)
# choose which transect
# transect_no = 29    # 29 = labby
# x.plot_cross_section(grid, ds[transect_no], v_g[transect_no], dist[transect_no],
#                      profile_tags_per[transect_no], isopycdep[transect_no], isopycx[transect_no],
#                      sigth_levels, d_time)
# bathy_path = '/Users/jake/Desktop/bats/bats_bathymetry/bathymetry_b38e_27c7_f8c3_f3d6_790d_30c7.nc'
# plan_window = [-66, -63, 31, 33]
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
d_anom_prof = np.nan * np.ones(sig2.shape)
for i in range(lon.shape[1]):
    this_time = np.nanmean(d_time[:, i])
    t_over = np.where(bckgrds_wins > this_time)[0]
    if len(t_over) > 1:
        if t_over[0] == 1:
            eta_per_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg[:, 0]) / np.squeeze(ddz_avg_sigma[:, 0])
            d_anom_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg[:, 0])
        elif t_over[0] == 2:
            eta_per_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg[:, 1]) / np.squeeze(ddz_avg_sigma[:, 1])
            d_anom_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg[:, 1])
        elif t_over[0] == 3:
            eta_per_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg[:, 2]) / np.squeeze(ddz_avg_sigma[:, 2])
            d_anom_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg[:, 2])
        else:
            eta_per_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg[:, 3]) / np.squeeze(ddz_avg_sigma[:, 3])
            d_anom_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg[:, 3])
    else:
        eta_per_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg[:, 3]) / np.squeeze(ddz_avg_sigma[:, 3])
        d_anom_prof[:, i] = (neutral_density[:, i] - sigma_theta_avg[:, 3])

AG_all, eta_m_all, Neta_m_all, PE_per_mass_all = eta_fit(lon.shape[1], grid, nmodes, N2_all, G_all, c_all,
                                                         eta_per_prof, eta_fit_depth_min, eta_fit_depth_max)
PE_per_mass_all = PE_per_mass_all[:, np.abs(AG_all[1, :]) > 1*10**-4]

# --- check on mode amplitudes from averaging or individual profiles
mw_time_ordered_i = np.argsort(Time2)
AG_ordered = AG[:, mw_time_ordered_i]

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


# isopycnals i care about
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


f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# ax1.scatter(d_time_per_prof_date, neutral_density[90, :], color='g', s=15, label=r'DG$_{ind}$')
# ax2.scatter(d_time_per_prof_date, neutral_density[165, :], color='g', s=15)
ax1.scatter(d_time_per_prof_date, d_dep_rho1[0, :], color='g', s=15, label=r'DG$_{ind}$')
ax2.scatter(d_time_per_prof_date, d_dep_rho1[1, :], color='g', s=15)
ax3.scatter(d_time_per_prof_date, d_dep_rho1[2, :], color='g', s=15)

# ax1.plot(mw_time_date, Avg_sig[90, mw_time_ordered_i], color='b', linewidth=0.75, label=r'DG$_{avg}$')
# ax2.plot(mw_time_date, Avg_sig[165, mw_time_ordered_i], color='b', linewidth=0.75)
ax1.plot(mw_time_date, mw_dep_rho1[0, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')
ax2.plot(mw_time_date, mw_dep_rho1[1, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')
ax3.plot(mw_time_date, mw_dep_rho1[2, :], color='b', linewidth=0.75, label=r'DG$_{avg}$')

# ax1.scatter(bats_time_date, bats_sig2[90, :], color='m', s=25, label='Ship')
# ax2.scatter(bats_time_date, bats_sig2[165, :], color='m', s=25)
eddy_disps = np.where((profile_tags >= 62) & (profile_tags <=74.5))[0]
ax1.plot([d_time_per_prof_date[eddy_disps[0]], d_time_per_prof_date[eddy_disps[0]]], [400, 900],
         color='k', linestyle='--', linewidth=0.75)
ax1.plot([d_time_per_prof_date[eddy_disps[-1]], d_time_per_prof_date[eddy_disps[-1]]], [400, 900],
         color='k', linestyle='--', linewidth=0.75)
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
ax1.set_ylim([500, 850])
ax2.set_ylim([1050, 1400])
ax3.set_ylim([2600, 2950])
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax1.grid()
ax2.grid()
plot_pro(ax3)

f, (ax, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
ax.scatter(d_time_per_prof_date, AG_all[1, :], color='g', s=15)
ax.plot(mw_time_date, AG_ordered[1, :], color='b', linewidth=0.75)
ax.set_title('Displacement Mode 1 Amplitude')
ax2.scatter(d_time_per_prof_date, AG_all[2, :], color='g', s=15)
ax2.plot(mw_time_date, AG_ordered[2, :], color='b', linewidth=0.75)
ax2.set_title('Displacement Mode 2 Amplitude')
ax3.scatter(d_time_per_prof_date, AG_all[3, :], color='g', s=15)
ax3.plot(mw_time_date, AG_ordered[3, :], color='b', linewidth=0.75)
ax3.set_title('Displacement Mode 3 Amplitude')
ax4.scatter(d_time_per_prof_date, AG_all[4, :], color='g', s=15)
ax4.plot(mw_time_date, AG_ordered[4, :], color='b', linewidth=0.75)
ax4.set_title('Displacement Mode 4 Amplitude')
ax5.scatter(d_time_per_prof_date, AG_all[5, :], color='g', s=15)
ax5.plot(mw_time_date, AG_ordered[5, :], color='b', linewidth=0.75)
ax5.set_title('Displacement Mode 5 Amplitude')
ax.grid()
ax2.grid()
ax3.grid()
ax4.grid()
plot_pro(ax5)

# ------- END OF ITERATION ON EACH PROFILE TO COMPUTE MODE FITS
# --------------------------------------------------------------------------------------------------
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

    # this_V = eof1[:, 0].copy()
    # Gz_rep = Gz[check1:check2, :].copy()
    # F_rep = F[check1:check2, :].copy()
    # iv = np.where(~np.isnan(this_V))
    # if iv[0].size > 1:
    #     AGz_eof[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(Gz_rep[iv, :]), np.transpose(np.atleast_2d(this_V[iv])))[0])
    #     V_m_eof_Gz = np.squeeze(np.matrix(Gz_rep) * np.transpose(np.matrix(AGz_eof[:, i])))
    #     AF_eof[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(F_rep[iv, :]), np.transpose(np.atleast_2d(this_V[iv])))[0])
    #     V_m_eof_F = np.squeeze(np.matrix(F_rep) * np.transpose(np.matrix(AF_eof[:, i])))

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
    # ax1.plot(np.transpose(V_m_eof_Gz)[:,0], grid_check, color='r', linewidth=0.75, linestyle='--')
    # if fvu1_per[i] > 0:
    #     ax1.plot(bc1 * min_p1, grid_check, color='r', linewidth=0.75)
    # ax2.plot(eof1, grid_check, color='k', linewidth=0.75)
    # ax2.plot(np.transpose(V_m_eof_F)[:,0], grid_check, color='r', linewidth=0.75, linestyle='--')
    # if fvu2_per[i] > 0:
    #     ax2.plot(bc2 * min_p2, grid_check, color='r', linewidth=0.75)

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
# ---------------------------------------------------------------------------------------------------------------------
# ------ VARIANCE EXPLAINED BY BAROCLINIC MODES ------------------------
eof1 = np.array(np.real(V_Uzqa[:, 0]))
eof1_sc = (1/2)*(eof1.max() - eof1.min()) + eof1.min()
bc1 = Gz[check1:check2, 1]  # flat bottom
bc2 = F[check1:check2, 0]   # sloping bottom

# -- minimize mode shapes onto eof shape
p = 0.8*eof1.min()/np.max(np.abs(F[:, 0]))
ins1 = np.transpose(np.concatenate([eof1, bc1[:, np.newaxis]], axis=1))
ins2 = np.transpose(np.concatenate([eof1, bc2[:, np.newaxis]], axis=1))
min_p1 = fmin(functi_1, p, args=(tuple(ins1)))
min_p2 = fmin(functi_1, p, args=(tuple(ins2)))

# -- plot inspect minimization of mode shapes
fvu1 = np.sum((eof1[:, 0] - bc1*min_p1)**2)/np.sum((eof1 - np.mean(eof1))**2)
fvu2 = np.sum((eof1[:, 0] - bc2*min_p2)**2)/np.sum((eof1 - np.mean(eof1))**2)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --- Isolate eddy dives
# 2015 - dives 62, 63 ,64
ed_prof_in = np.where(((x.dives) >= 62) & ((x.dives) <= 64))[0]
ed_in = np.where((Info2 >= 62) & (Info2 <= 64))[0]
ed_in_2 = np.where((Info2 > 61) & (Info2 < 63))[0]
ed_time_s = datetime.date.fromordinal(np.int(Time2[ed_in[0]]))
ed_time_e = datetime.date.fromordinal(np.int(Time2[ed_in[-1] + 1]))
# --- time series
mission_start = datetime.date.fromordinal(np.int(Time2.min()))
mission_end = datetime.date.fromordinal(np.int(Time2.max()))
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --- PLOT ETA / EOF
plot_eta = 1
if plot_eta > 0:
    f, (ax2, ax1, ax15, ax0) = plt.subplots(1, 4, sharey=True)
    for j in range(len(Time2)):
        ax2.plot(d_anom_alt[:, j], grid, linewidth=0.75)
    ax2.set_xlim([-.5, .5])
    ax2.set_xlabel(r'$\gamma^n - \overline{\gamma^n}$', fontsize=12)
    ax2.set_title("DG35 BATS: " + str(mission_start) + ' - ' + str(mission_end))
    ax2.text(0.1, 4000, str(len(Time2)) + ' profiles', fontsize=10)
    for j in range(num_profs):
        ax1.plot(Eta2[:, j], grid, color='#4682B4', linewidth=1.25)
        ax1.plot(Eta_m[:, j], grid, color='k', linestyle='--', linewidth=.75)
        if good_ke_prof[j] > 0:
            ax0.plot(V2[:, j], grid, color='#4682B4', linewidth=1.25)
            ax0.plot(V_m[:, j], grid, color='k', linestyle='--', linewidth=.75)
    for j in range(eta_per_prof.shape[1]):
        ax15.plot(eta_per_prof[:, j], grid, color='#4682B4', linewidth=1.25)
        ax15.plot(eta_m_all[:, j], grid, color='k', linestyle='--', linewidth=.75)
    ax15.set_title(r'Isopycnal Disp. (Ind.)', fontsize=11)
    # -- plot eddy profiles
    for k in range(ed_in[0], ed_in[-1] + 2):
        ax1.plot(Eta2[:, k], grid, color='m', linewidth=1.5, label='eddy')
        ax0.plot(V2[:, k], grid, color='m', linewidth=1.5)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[0]], [labels[0]], fontsize=10)
    ax1.set_xlim([-400, 400])
    ax15.set_xlim([-400, 400])
    ax0.text(190, 800, str(num_profs) + ' Profiles')
    ax1.set_xlabel(r'Vertical Isopycnal Displacement, $\xi_{\sigma_{\theta}}$ [m]', fontsize=11)
    ax1.set_title(r'Isopycnal Disp. (Avg.)', fontsize=11)  # + '(' + str(Time[0]) + '-' )
    ax0.axis([-.5, .5, 0, 4750])
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
    ax1.axis([-4, 4, 0, 4750])
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
    ax2.axis([-.7, .7, 0, 4750])
    ax2.set_title('EOF Displacement Mode Shapes', fontsize=18)
    ax2.set_xlabel('Normalized Mode Amp.', fontsize=14)
    ax2.invert_yaxis()
    plot_pro(ax2)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --- PLOT V STRUCTURE
# --- bottom boundary conditions
plot_v_struct = 1
if plot_v_struct > 0:
    f, (ax, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    for i in range(nq):
        ax.plot(V3[:, i], grid, color='#5F9EA0', linewidth=0.75)
    ax.plot(np.nanmean(np.abs(V3), axis=1), grid, color='k', label='Average |V|')
    ax.set_xlim([-.3, .3])
    ax.set_ylim([0, 4750])
    ax.set_title('Cross-Track Velocity [V]', fontsize=12)
    ax.set_xlabel('m/s', fontsize=16)
    ax.set_ylabel('Depth [m]', fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10)

    ax2.plot(np.zeros(10), np.arange(0, 5000, 500), color='k', linewidth=0.5)
    ax3.plot(np.zeros(10), np.arange(0, 5000, 500), color='k', linewidth=0.5)
    for i in range(4):
        ax2.plot(V_Uzqa[:, i], grid_check, label=r'PEV$_{' + str(i + 1) + '}$ = ' + str(100 * np.round(PEV[i], 3)),
                 linewidth=2)
        ax3.plot(Gz[:, i], grid, label='Mode' + str(i), linewidth=2)
        ax4.plot(F[:, i], grid, label='Mode' + str(i), linewidth=2)

        # if i < 1:
        #     ax4.plot(F_int[:, i] + np.nanmax(np.abs(F_int[:, i])), grid)
        # else:
        #     ax4.plot(F_int[:, i], grid)
        # ax4.plot(G[:, i], grid, c='k', linestyle='--', linewidth=0.5)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=12)
    ax2.set_xlim([-.2, .2])
    ax2.set_title('Principle EOFs of V', fontsize=12)
    ax2.set_xlabel('m/s', fontsize=16)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, fontsize=10)
    ax3.set_xlim([-5, 5])
    ax3.set_title('Flat Bottom', fontsize=12)
    ax3.set_xlabel('Mode Amplitude', fontsize=10)
    ax4.set_title('Sloping Bottom', fontsize=12)
    ax4.set_xlabel('Mode Amplitude', fontsize=10)
    ax.grid()
    ax2.grid()
    ax3.grid()
    ax.invert_yaxis()
    plot_pro(ax4)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --- MODE AMPLITUDE IN TIME AND SPACE ----------
# averaging eta estimation
Time3 = Time2[np.argsort(Time2)]
Time2_dt = []
for i in range(len(Time3)):
    Time2_dt.append(datetime.date.fromordinal(np.int(Time3[i])))
# individual eta estimation
Time_all = time_rec_bin[np.argsort(time_rec_bin)]
Time2_all_dt = []
for i in range(len(Time_all)):
    Time2_all_dt.append(datetime.date.fromordinal(np.int(Time_all[i])))

# load in Station BATs mode amplitude Comparison
pkl_file = open('/Users/jake/Desktop/bats/station_bats_pe.pkl', 'rb')
SB = pickle.load(pkl_file)
pkl_file.close()
sta_bats_ag = SB['AG']
sta_bats_time = SB['time']
sba_in = sta_bats_ag[:, sta_bats_time > 2015]
sbt_in = sta_bats_time[sta_bats_time > 2015]
sb_dt = []
for i in range(len(sbt_in)):
    sb_dt.append(datetime.datetime(np.int(sbt_in[i]), np.int((sbt_in[i] - np.int(sbt_in[i])) * 12), np.int(
        ((sbt_in[i] - np.int(sbt_in[i])) * 12 - np.int((sbt_in[i] - np.int(sbt_in[i])) * 12)) * 30)))

# PLOTTING
plot_mode = 0
if plot_mode > 0:
    # DISPLACEMENT MODE AMPLITUDES
    window_size, poly_order = 9, 2
    fm, (ax, ax2) = plt.subplots(2, 1)
    colors = ['#8B0000', '#FF8C00', '#808000', '#5F9EA0', 'g', 'c']
    ax.plot(sb_dt, c[1] * sba_in[1, :], color='m', label='Hydrography Mode 1')
    for mo in range(1, 4):
        # orderAG = AG[mo, np.argsort(Time2)]
        # y_sg = savgol_filter(orderAG, window_size, poly_order)
        ax.plot(Time2_dt, c[mo] * AG[mo, np.argsort(Time2)], color=colors[mo], linewidth=1, label=('Mode ' + str(mo)))
        # ax.plot(Time2_dt, c[mo] * y_sg, color=colors[mo], linewidth=2, label=('Mode ' + str(mo)))
        ax.plot(Time2_all_dt, c[mo] * AG_all[mo, np.argsort(time_rec_bin)], color=colors[mo],
                linewidth=0.75, linestyle='--')
    ax.set_xlim([Time2_dt[0], Time2_dt[-1]])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12)
    ax.set_title(r'Scaled Displacement Mode Amplitude (c$_{n}\beta_{n}$)')
    # ax.set_xlabel('Date')
    ax.grid()

    # -- attempt fft to find period of oscillation
    Time_grid = np.arange(np.round(Time2.min()), np.round(Time2.max()), 1)
    order_0_AGz = AGz[0, np.argsort(Time2)]
    y_AGz_0 = savgol_filter(order_0_AGz, window_size, poly_order)
    order_0_AGz_grid = np.interp(Time_grid, Time2, y_AGz_0)
    order_1_AGz = AGz[1, np.argsort(Time2)]
    y_AGz_1 = savgol_filter(order_1_AGz, window_size, poly_order)
    order_1_AGz_grid = np.interp(Time_grid, Time2, y_AGz_1)

    N = len(order_0_AGz_grid)
    T = Time_grid[1] - Time_grid[0]
    yf_0 = scipy.fftpack.fft(order_0_AGz_grid)
    yf_1 = scipy.fftpack.fft(order_1_AGz_grid)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
    # f, ax = plt.subplots()
    # ax.plot(xf, 2.0/N * np.abs(yf_0[:N//2]), 'r')
    # ax.plot(xf, 2.0 / N * np.abs(yf_1[:N // 2]), 'b')
    # plot_pro(ax)

    # VELOCITY MODE AMPLITUDES
    for mo in range(3):
        orderAGz = AGz[mo, np.argsort(Time2)]
        y_sgz = savgol_filter(orderAGz, window_size, poly_order)
        # pmz = ax1.plot(Time2_dt, AGz[mo, np.argsort(Time2)], color=colors[mo], linewidth=0.75)
        ax2.plot(Time2_dt, y_sgz, color=colors[mo], label=('Mode ' + str(mo)), linewidth=2)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=12)
    ax2.set_title(r'Velocity Mode Amplitude ($\alpha_{n}$)')
    ax2.set_ylabel('Mode Amplitude')
    ax2.set_xlabel('Date')
    ax2.set_ylim([-.12, .12])
    ax2.set_xlim([Time2_dt[0], Time2_dt[-1]])
    plot_pro(ax2)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --- MODE AMPLITUDE CORRELATIONS IN TIME AND SPACE
# --- only makes sense to use eta from individual profiles (doesn't make sense to map correlations from v because
# --- compute of v smears over 4 profiles)
prof_lon_i = np.nanmean(lon, axis=0)
prof_lat_i = np.nanmean(lat, axis=0)
plot_mode_corr = 0
if plot_mode_corr > 0:
    x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (prof_lon_i - ref_lon)
    y = 1852 * 60 * (prof_lat_i - ref_lat)
    x_tile = np.tile(x, (len(x), 1))
    y_tile = np.tile(y, (len(y), 1))
    time_tile = np.tile(time_rec_bin, (len(time_rec_bin), 1))
    dist = np.sqrt((x_tile - x_tile.T) ** 2 + (y_tile - y_tile.T) ** 2) / 1000
    time_lag = np.abs(time_tile - time_tile.T)

    # # this mode
    # mode_num = 1
    # AG_i = AG_all[mode_num, :]
    # # AGz_i = AGz[mode_num, :]
    #
    # # define each box as all points that fall within a time and space lag
    # dist_win = np.arange(0, 100, 10)
    # t_win = np.arange(0, 100, 10)
    # # try to compute lagged autocorrelation for all points within a given distance
    # # returns a tuple of coordinate pairs where distance criteria are met
    # corr_i = np.nan * np.zeros((len(t_win), len(dist_win)))
    # corr_z_i = np.nan * np.zeros((len(t_win), len(dist_win)))
    # for dd in range(len(dist_win) - 1):
    #     dist_small_i = np.where((dist > dist_win[dd]) & (dist < dist_win[dd + 1]))
    #     time_in = np.unique(time_lag[dist_small_i[0], dist_small_i[1]])
    #     AG_out = np.nan * np.zeros([len(dist_small_i[0]), 3])
    #     AGz_out = np.nan * np.zeros([len(dist_small_i[0]), 3])
    #     for i in range(len(dist_small_i[0])):
    #         AG_out[i, :] = [AG_i[dist_small_i[0][i]], AG_i[dist_small_i[1][i]],
    #                         time_lag[dist_small_i[0][i], dist_small_i[1][i]]]
    #         # AGz_out[i, :] = [AGz_i[dist_small_i[0][i]], AGz_i[dist_small_i[1][i]],
    #         #                  time_lag[dist_small_i[0][i], dist_small_i[1][i]]]
    #     no_doub, no_doub_i = np.unique(AG_out[:, 2], return_index=True)
    #     AG_out2 = AG_out[no_doub_i, :]
    #     # zno_doub, zno_doub_i = np.unique(AGz_out[:, 2], return_index=True)
    #     # AGz_out2 = AGz_out[zno_doub_i, :]
    #     for j in range(len(t_win) - 1):
    #         inn = AG_out2[((AG_out2[:, 2] > t_win[j]) & (AG_out2[:, 2] < t_win[j + 1])), 0:3]
    #         i_mean = np.nanmean(inn[:, 0:2])
    #         n = len(inn[:, 0:2])
    #         variance = np.nanvar(inn[:, 0:2])
    #         covi = np.nan * np.zeros(len(inn[:, 0]))
    #         for k in range(len(inn[:, 0])):
    #             covi[k] = (inn[k, 0] - i_mean) * (inn[k, 1] - i_mean)
    #         corr_i[j, dd] = (1 / (n * variance)) * np.nansum(covi)
    #
    #         # innz = AGz_out2[((AGz_out2[:, 2] > t_win[j]) & (AGz_out2[:, 2] < t_win[j + 1])), 0:3]
    #         # iz_mean = np.mean(innz[:, 0:2])
    #         # nz = len(innz[:, 0:2])
    #         # variancez = np.var(innz[:, 0:2])
    #         # covzi = np.nan * np.zeros(len(innz[:, 0]))
    #         # for k in range(len(innz[:, 0])):
    #         #     covzi[k] = (innz[k, 0] - iz_mean) * (innz[k, 1] - iz_mean)
    #         # corr_z_i[j, dd] = (1 / (nz * variancez)) * np.sum(covzi)
    # f, ax1 = plt.subplots()
    # cmap = plt.cm.get_cmap("viridis")
    # cmap.set_over('w')  # ('#E6E6E6')
    # pa = ax1.pcolor(dist_win, t_win, corr_i, vmin=-1, vmax=1, cmap='viridis')
    # # paz = ax2.pcolor(dist_win, t_win, corr_z_i, vmin=-1, vmax=1, cmap='viridis')
    # ax1.set_xlabel('Spatial Separation [km]')
    # ax1.set_ylabel('Time Lag [days]')
    # ax1.set_title('Displacement Mode 1 Amplitude')
    # plt.colorbar(pa, label='Correlation')
    # plot_pro(ax1)

    # -----
    # redo with only zonal separation
    x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (prof_lon_i - ref_lon) / 1000
    y = 1852 * 60 * (prof_lat_i - ref_lat) / 1000
    t = time_rec_bin.copy()
    AG_i = AG_all[1, :]
    AG_ii = AG_all[3, :]

    # distances apart
    # dist_x = np.nan * np.zeros((len(x), len(x)))
    # time_l = np.nan * np.zeros((len(x), len(x)))
    for i in range(len(x) - 1):
        if i < 1:
            dist_x = x[i+1:] - x[i]
            dist_y = y[i+1:] - y[i]
            dist_t = np.sqrt(dist_x**2 + dist_y**2)
            time_l = np.abs(time_rec_bin[i] - t[i+1:])
            AG_count = np.array((AG_i[i] * np.ones(len(x[i+1:])), AG_i[i+1:]))
            AG_count2 = np.array((AG_ii[i] * np.ones(len(x[i + 1:])), AG_ii[i + 1:]))
        else:
            dist_x = np.concatenate((dist_x, x[i+1:] - x[i]))
            dist_y = np.concatenate((dist_y, y[i + 1:] - y[i]))
            dist_t = np.concatenate((dist_t, np.sqrt((x[i+1:] - x[i])**2 + (y[i + 1:] - y[i])**2)))
            time_l = np.concatenate((time_l, np.abs(t[i] - t[i+1:])))
            AG_count = np.concatenate((AG_count, np.array((AG_i[i] * np.ones(len(x[i + 1:])), AG_i[i + 1:]))), axis=1)
            AG_count2 = np.concatenate((AG_count2,
                                        np.array((AG_ii[i] * np.ones(len(x[i + 1:])), AG_ii[i + 1:]))), axis=1)

    # define each box as all points that fall within a time and space lag
    dist_win = np.arange(-100, 105, 2)
    dist_t_win = np.arange(0, 105, 5)
    t_win = np.arange(0, 80, 2)
    t_t_win = np.arange(0, 80, 5)
    corr_i_z = np.nan * np.zeros((len(t_win), len(dist_win)))
    corr_i_z2 = np.nan * np.zeros((len(t_win), len(dist_win)))
    corr_i_all = np.nan * np.zeros((len(t_t_win), len(dist_t_win)))
    for dd in range(len(dist_win) - 1):
        for tt in range(len(t_win) - 1):
            in_box = np.where((dist_x > dist_win[dd]) & (dist_x < dist_win[dd + 1]) &
                              (time_l > t_win[tt]) & (time_l < t_win[tt + 1]))[0]
            if len(in_box) > 4:
                inski_with = AG_count[:, in_box]
                inski = np.unique(AG_count[:, in_box])
                i_mean = np.nanmean(inski)
                n = len(inski)
                variance = np.nanvar(inski)
                covi = np.nan * np.zeros(len(inski))
                for k in range(np.shape(inski_with)[1]):
                    covi[k] = (inski_with[0, k] - i_mean) * (inski_with[1, k] - i_mean)
                corr_i_z[tt, dd] = (1 / (n * variance)) * np.nansum(covi)

                inski_with = AG_count2[:, in_box]
                inski = np.unique(AG_count2[:, in_box])
                i_mean = np.nanmean(inski)
                n = len(inski)
                variance = np.nanvar(inski)
                covi = np.nan * np.zeros(np.shape(inski_with)[1])
                for k in range(np.shape(inski_with)[1]):
                    covi[k] = (inski_with[0, k] - i_mean) * (inski_with[1, k] - i_mean)
                corr_i_z2[tt, dd] = (1 / (n * variance)) * np.nansum(covi)

    for dd in range(len(dist_t_win) - 1):
        for tt in range(len(t_t_win) - 1):
            in_box = np.where((dist_t > dist_t_win[dd]) & (dist_t < dist_t_win[dd + 1]) &
                              (time_l > t_t_win[tt]) & (time_l < t_t_win[tt + 1]))[0]
            if len(in_box) > 5:
                inski_with = AG_count[:, in_box]
                inski = np.unique(AG_count[:, in_box])
                i_mean = np.nanmean(inski)
                n = len(inski)
                variance = np.nanvar(inski)
                covi = np.nan * np.zeros(np.shape(inski_with)[1])
                for k in range(np.shape(inski_with)[1]):
                    covi[k] = (inski_with[0, k] - i_mean) * (inski_with[1, k] - i_mean)
                corr_i_all[tt, dd] = (1 / (n * variance)) * np.nansum(covi)

    f, (ax1, ax2) = plt.subplots(1, 2)
    pa = ax1.pcolor(dist_win, t_win, corr_i_z, vmin=-1, vmax=.8, cmap='jet')
    pa2 = ax2.pcolor(dist_win, t_win, corr_i_z2, vmin=-1, vmax=.8, cmap='jet')
    ax1.set_xlabel('Zonal Separation [km]')
    ax1.set_ylabel('Time Lag [days]')
    ax2.set_xlabel('Zonal Separation [km]')
    ax1.set_title('Displacement Mode 1 Amplitude')
    ax2.set_title('Displacement Mode 3 Amplitude')
    f.colorbar(pa, ax=ax1, label='Correlation')
    f.colorbar(pa2, ax=ax2, label='Correlation')
    ax1.grid()
    plot_pro(ax2)

    f, ax1 = plt.subplots()
    pa = ax1.pcolor(dist_t_win, t_t_win, corr_i_all, vmin=-1, vmax=1, cmap='jet')
    ax1.set_xlabel('Spatial Separation [km]')
    ax1.set_ylabel('Time Lag [days]')
    ax1.set_title('Displacement Mode 1 Amplitude')
    plt.colorbar(pa, label='Correlation')
    plot_pro(ax1)

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
PE_ed = np.nanmean(PE_per_mass[:, ed_in[0]:ed_in[-1]], axis=1)
KE_ed = np.nanmean(HKE_per_mass[:, ed_in[0]:ed_in[-1]], axis=1)

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
mu = 1.88e-3 / (1 + 0.03222 * np.nanmean(theta_avg, axis=1) +
                0.002377 * np.nanmean(theta_avg, axis=1) * np.nanmean(theta_avg, axis=1))
nu = mu / gsw.rho(np.nanmean(salin_avg, axis=1), np.nanmean(cons_t_avg, axis=1), grid_p)
avg_nu = np.nanmean(nu)

# --- most and least energetic profiles
KE_i = HKE_per_mass[:, calmer]
PE_i = PE_per_mass[:, calmer]
KE_it = np.nan * np.ones(KE_i.shape[1])
PE_it = np.nan * np.ones(KE_i.shape[1])
for i in range(KE_i.shape[1]):
    KE_it[i] = np.trapz(KE_i[1:, i] + PE_i[1:, i], 1000 * f_ref / c[1:])
    PE_it[i] = np.trapz(PE_i[1:, i], 1000 * f_ref / c[1:])
KE_i_max = np.where(KE_it == np.nanmax(KE_it))[0]
KE_i_min = np.where(KE_it == np.nanmin(KE_it))[0]
PE_i_max = np.where(PE_it == np.nanmax(PE_it))[0]
PE_i_min = np.where(PE_it == np.nanmin(PE_it))[0]
k_h_max = 1e3 * (f_ref / c[1:]) * np.sqrt(np.squeeze(KE_i[1:, KE_i_max]) / np.squeeze(PE_i[1:, KE_i_max]))
k_h_min = 1e3 * (f_ref / c[1:]) * np.sqrt(np.squeeze(KE_i[1:, KE_i_min]) / np.squeeze(PE_i[1:, KE_i_min]))

# ----------------------------------------------------------------------------------------------------------------------
# --- CURVE FITTING TO FIND BREAK IN SLOPES, WHERE WE MIGHT SEE -5/3 AND THEN -3
xx = sc_x.copy()
yy = avg_PE[1:] / dk
yy2 = avg_KE[1:] / dk
ipoint = 5  # 8  # 11
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

    ## ax.plot(sc_x, this_TE, color='k')
    # ax.plot(10**x_grid, 10**fit, color='m')
# ax.set_yscale('log')
# ax.set_xscale('log')
# plot_pro(ax)

# -- Histogram of vorticity and transfer rates (for all profiles)
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
# ax1.hist(rms_vort_per[~np.isnan(rms_vort_per)], bins=np.arange(np.nanmin(rms_vort_per), 1*10**-9, 5*10**-11),
#          facecolor='blue', alpha=0.5)
# ax1.set_xlabel('RMS vorticity')
# ax2.hist(enst_xfer_per, bins=np.arange(np.nanmin(enst_xfer_per), 4*10**-16, 2.5*10**-17), facecolor='blue', alpha=0.5)
# ax2.set_xlabel('Enstrophy Transfer Rate')
# ax3.hist(1 / min_sp, 20, facecolor='blue', alpha=0.5)
# ax3.set_xlabel('Rossby Radius of Break [km]')
# ax1.set_ylim([0, 70])
# ax1.set_ylabel('Count (out of ' + str(num_profs) + ')')
# ax1.grid()
# ax2.grid()
# plot_pro(ax3)

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
pkl_file = open('/Users/jake/Desktop/bats/station_bats_pe_nov05.pkl', 'rb')
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
    test4 = np.nanmean(sta_bats_pe[3][i, :])
    sta_max[i - 1] = np.max([test1, test2, test3, test4])
    sta_min[i - 1] = np.min([test1, test2, test3, test4])
    dg_per_max[i - 1] = np.nanmax(PE_per_mass_all[i, :])
    dg_per_min[i - 1] = np.nanmin(PE_per_mass_all[i, :])

# -- load in HKE estimates from Obj. Map
pkl_file = open('/Users/jake/Documents/geostrophic_turbulence/BATS_OM_KE.pkl', 'rb')
bats_map = pickle.load(pkl_file)
pkl_file.close()
sx_c_om = bats_map['sc_x']
ke_om_u = bats_map['avg_ke_u']
ke_om_v = bats_map['avg_ke_v']
ke_om_tot = bats_map['avg_ke_u'] + bats_map['avg_ke_v']
dk_om = bats_map['dk']
# -- load in Station HOTS PE Comparison
SH = si.loadmat('/Users/jake/Desktop/bats/station_hots_pe.mat')
sta_hots_pe = SH['out']['PE'][0][0]
sta_hots_c = SH['out'][0][0][3]
sta_hots_f = SH['out'][0][0][2]
sta_hots_dk = SH['out']['dk'][0][0]
# -- LOAD ABACO
pkl_file = open('/Users/jake/Documents/geostrophic_turbulence/ABACO_2017_energy.pkl', 'rb')
abaco_energies = pickle.load(pkl_file)
pkl_file.close()
# ------------------------------------------------------------------------------------------------------------------
plot_eng = 1
if plot_eng > 0:
    fig0, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
    # Station (by season)
    ax0.fill_between(1000 * sta_bats_f / sta_bats_c[1:mmax+1], sta_min / sta_bats_dk, sta_max / sta_bats_dk,
                     label='APE$_{sta.}$', color='#D3D3D3')
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

    # -- Obj. Map
    # KE_om = ax0.plot(sx_c_om, ke_om_tot[1:]/dk_om, 'b', label='$KE_{map}$', linewidth=1.5)
    # KE_om_u = ax0.plot(sx_c_om,ke_om_u[1:]/dk_om, 'b', label='$KE_u$', linewidth=1.5)
    # ax0.scatter(sx_c_om, ke_om_u[1:]/dk_om, color='b', s=10)  # DG KE
    # KE_om_u = ax0.plot(sx_c_om,ke_om_v[1:]/dk_om, 'c', label='$KE_v$', linewidth=1.5)
    # ax0.scatter(sx_c_om, ke_om_v[1:]/dk_om, color='c', s=10)  # DG KE

    # -- Eddy energies
    # PE_e = ax0.plot(sc_x, PE_ed[1:] / dk, color='c', label='eddy PE', linewidth=2)
    # KE_e = ax0.plot(sc_x, KE_ed[1:] / dk, color='y', label='eddy KE', linewidth=2)
    # KE_e = ax0.plot([10**-2, 1000 * f_ref / c[1]], KE_ed[0:2] / dk, color='y', label='eddy KE', linewidth=2)

    # -- Slope fits
    ax0.plot(10 ** x_grid, 10 ** fit_total, color='#FF8C00', label=r'APE$_{fit}$')
    # PE
    ax0.plot(10 ** x_53, 10 ** y_g_53, color='b', linewidth=1.25)
    ax0.plot(10 ** x_3, 10 ** y_g_3, color='b', linewidth=1.25)
    ax0.text(10 ** x_53[0] - .012, 10 ** y_g_53[0], str(float("{0:.2f}".format(slope1[0]))), fontsize=10)
    ax0.text(10 ** x_3[0] - .11, 10 ** y_g_3[0], str(float("{0:.2f}".format(slope2[0]))), fontsize=10)
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
    ax0.plot(sc_x, 0.25 * PE_GM / dk, linestyle='--', color='k', linewidth=0.75)
    ax0.plot(sc_x, 0.25 * GMPE / dk, color='k', linewidth=0.75)
    ax0.text(sc_x[0] - .01, 0.5 * PE_GM[1] / dk, r'$1/4 PE_{GM}$', fontsize=10)
    # ax0.plot(np.array([10**-2, 10]), [PE_SD / dk, PE_SD / dk], linestyle='--', color='k', linewidth=0.75)
    ax1.plot(sc_x, 0.25 * GMKE / dk, color='k', linewidth=0.75)
    ax1.text(sc_x[0] - .01, 0.5 * GMKE[1] / dk, r'$1/4 KE_{GM}$', fontsize=10)

    # ax0.axis([10 ** -2, 10 ** 1, 10 ** (-4), 2 * 10 ** 3])
    ax0.set_xlim([10 ** -2, 2 * 10 ** 0])
    ax0.set_ylim([10 ** (-4), 2 * 10 ** 2])
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

# --- SAVE BATS ENERGIES DG TRANSECTS
# write python dict to a file
sa = 0
if sa > 0:
    my_dict = {'depth': grid, 'KE': avg_KE, 'PE': avg_PE, 'c': c, 'f': f_ref, 'N2': N2_all}
    output = open('/Users/jake/Documents/baroclinic_modes/DG/sg035_energy.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# -------------------------
# PE COMPARISON BETWEEN HOTS, BATS_SHIP, AND BATS_DG
plot_comp = 0
if plot_comp > 0:
    fig00, ax0 = plt.subplots()
    ax0.plot([3 * 10 ** -1, 3 * 10 ** 0], [1.5 * 10 ** 1, 1.5 * 10 ** -2], color='k', linewidth=0.75)
    ax0.plot([3 * 10 ** -2, 3 * 10 ** -1],
             [7 * 10 ** 2, ((5 / 3) * (np.log10(2 * 10 ** -1) - np.log10(2 * 10 ** -2)) + np.log10(7 * 10 ** 2))],
             color='k', linewidth=0.75)
    ax0.text(3.3 * 10 ** -1, 1.3 * 10 ** 1, '-3', fontsize=8)
    ax0.text(3.3 * 10 ** -2, 6 * 10 ** 2, '-5/3', fontsize=8)
    ax0.plot(sc_x, PE_GM / dk, linestyle='--', color='k')
    ax0.text(sc_x[0] - .009, PE_GM[1] / dk, r'$PE_{GM}$')
    PE_p0 = ax0.plot(sc_x, avg_PE[1:] / dk, color='#B22222', label=r'$PE_{BATS_{DG}}$')
    PE_sta_p0 = ax0.plot(1000 * sta_bats_f / sta_bats_c[1:], np.nanmean(sta_bats_pe[1:], axis=1) / sta_bats_dk,
                         color='#FF8C00', label=r'$PE_{BATS_{ship}}$')
    # PE_sta_hots = ax0.plot((1000)*sta_hots_f/sta_hots_c[1:],sta_hots_pe[1:]/sta_hots_dk,color='g',
    #   label=r'$PE_{HOTS_{ship}}$')
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.set_xlabel(r'Vertical Wavenumber = Inverse Rossby Radius = $\frac{f}{c}$ [$km^{-1}$]', fontsize=13)
    ax0.set_ylabel('Spectral Density (and Hor. Wavenumber)')
    ax0.set_title('Potential Energy Spectra (Site/Platform Comparison)')
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, fontsize=12)
    plot_pro(ax0)

    # PE (by mode number) COMPARISON BY SITE
    fig01, ax0 = plt.subplots()
    mode_num = np.arange(1, 61, 1)
    PE_sta_p1 = ax0.plot(mode_num, np.nanmean(sta_bats_pe[1:], axis=1) / sta_bats_dk, label=r'BATS$_{ship}$',
                         linewidth=2)
    # PE_ab = ax0.plot(mode_num, abaco_energies['avg_PE'][1:] / (abaco_energies['f_ref'] / abaco_energies['c'][1]),
    #                  label=r'$ABACO_{DG}$')
    PE_sta_hots = ax0.plot(mode_num, sta_hots_pe[1:] / sta_hots_dk, label=r'HOTS$_{ship}$', linewidth=2)
    PE_p1 = ax0.plot(mode_num, avg_PE[1:] / dk, label=r'BATS$_{DG}$', color='#708090', linewidth=1)
    ax0.set_xlabel('Mode Number', fontsize=16)
    ax0.set_ylabel('Spectral Density', fontsize=16)
    ax0.set_title('Potential Energy Spectra (Site Comparison)', fontsize=18)
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    # ax0.axis([8*10**-1, 10**2, 3*10**(-4), 2*10**(3)])
    ax0.axis([8 * 10 ** -1, 10 ** 2, 3 * 10 ** (-4), 10 ** 3])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, fontsize=12)
    plot_pro(ax0)

    # ABACO BATS PE/KE COMPARISONS
    fig02, ax0 = plt.subplots()
    mode_num = np.arange(1, 61, 1)
    PE_ab = ax0.plot(mode_num, abaco_energies['avg_PE'][1:] / (abaco_energies['f_ref'] / abaco_energies['c'][1]),
                     label=r'APE ABACO$_{DG}$', linewidth=2.5, color='r')
    KE_ab = ax0.plot(mode_num, abaco_energies['avg_KE'][1:] / (abaco_energies['f_ref'] / abaco_energies['c'][1]),
                     label=r'KE ABACO$_{DG}$', linewidth=2.5, color='g')
    # PE_sta_hots = ax0.plot(mode_num,sta_hots_pe[1:]/sta_hots_dk,label=r'HOTS$_{ship}$',linewidth=2)
    PE_p2 = ax0.plot(mode_num, avg_PE[1:] / dk, label=r'APE BATS$_{DG}$', color='#F08080', linewidth=1.5)
    KE_p2 = ax0.plot(mode_num, avg_KE[1:] / dk, 'g', label=r'KE BATS$_{DG}$', color='#90EE90', linewidth=1.5)
    ax0.plot([10 ** 1, 10 ** 2], [10 ** 1, 10 ** -2], color='k', linewidth=0.75)
    ax0.text(1.2 * 10 ** 1, 1.3 * 10 ** 1, '-3', fontsize=11)
    ax0.set_xlabel('Mode Number', fontsize=18)
    ax0.set_ylabel('Spectral Density', fontsize=18)
    ax0.set_title('ABACO, BATS Deepglider PE, KE', fontsize=20)
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    # ax0.axis([8*10**-1, 10**2, 3*10**(-4), 2*10**(3)])
    ax0.axis([8 * 10 ** -1, 10 ** 2, 3 * 10 ** (-4), 10 ** 3])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, fontsize=12)
    plot_pro(ax0)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --- FRACTIONS of ENERGY IN MODES AT EACH DEPTH
dg_mode_ke_z_frac = np.nan * np.zeros((num_profs, len(grid), 6))
dg_mode_pe_z_frac = np.nan * np.zeros((num_profs, len(grid), 7))
tke_tot_z = np.nan * np.zeros((len(grid), num_profs))
pe_tot_z = np.nan * np.zeros((len(grid), num_profs))
tke_m0_z = np.nan * np.zeros((len(grid), num_profs))
tke_m1_z = np.nan * np.zeros((len(grid), num_profs))
tke_m2_z = np.nan * np.zeros((len(grid), num_profs))
tke_m3_z = np.nan * np.zeros((len(grid), num_profs))
tke_m4_z = np.nan * np.zeros((len(grid), num_profs))
pe_m1_z = np.nan * np.zeros((len(grid), num_profs))
pe_m2_z = np.nan * np.zeros((len(grid), num_profs))
pe_m3_z = np.nan * np.zeros((len(grid), num_profs))
pe_m4_z = np.nan * np.zeros((len(grid), num_profs))
pe_m5_z = np.nan * np.zeros((len(grid), num_profs))
# - loop over each profile
ed_in_2 = ed_in_2 + 1
for pp in np.arange(5, 80):  # np.append(np.arange(5, 42), np.arange(140, 160)):
    # - loop over each depth
    for j in range(len(grid)):
        tke_tot_z[j, pp] = np.sum(0.5 * ((AGz[0:20, pp] ** 2) * (Gz[j, 0:20] ** 2)))  # ke sum over all modes at depths z
        pe_tot_z[j, pp] = np.sum(0.5 * ((AG[0:20, pp] ** 2) * N2_all[j] * (G[j, 0:20] ** 2)))  # pe sum over all modes at depths z
        tke_m0_z[j, pp] = 0.5 * ((AGz[0, pp] ** 2) * (Gz[j, 0] ** 2))  # ke mode 0 contribution to tke at depths z
        tke_m1_z[j, pp] = 0.5 * ((AGz[1, pp] ** 2) * (Gz[j, 1] ** 2))  # ke mode 1 contribution to tke at depths z
        tke_m2_z[j, pp] = 0.5 * ((AGz[2, pp] ** 2) * (Gz[j, 2] ** 2))  # ke mode 2 contribution to tke at depths z
        tke_m3_z[j, pp] = 0.5 * ((AGz[3, pp] ** 2) * (Gz[j, 3] ** 2))  # ke mode 3 contribution to tke at depths z
        tke_m4_z[j, pp] = 0.5 * ((AGz[4, pp] ** 2) * (Gz[j, 4] ** 2))  # ke mode 3 contribution to tke at depths z
        pe_m1_z[j, pp] = 0.5 * ((AG[1, pp] ** 2) * N2_all[j] * (G[j, 1] ** 2))  # pe mode 1 contribution to tke at depths z
        pe_m2_z[j, pp] = 0.5 * ((AG[2, pp] ** 2) * N2_all[j] * (G[j, 2] ** 2))  # pe mode 1 contribution to tke at depths z
        pe_m3_z[j, pp] = 0.5 * ((AG[3, pp] ** 2) * N2_all[j] * (G[j, 3] ** 2))  # pe mode 1 contribution to tke at depths z
        pe_m4_z[j, pp] = 0.5 * ((AG[4, pp] ** 2) * N2_all[j] * (G[j, 4] ** 2))  # pe mode 1 contribution to tke at depths z
        pe_m5_z[j, pp] = 0.5 * ((AG[5, pp] ** 2) * N2_all[j] * (G[j, 5] ** 2))  # pe mode 1 contribution to tke at depths z

        # # loop over first few modes
        # for mn in range(6):
        #     dg_mode_ke_z_frac[pp, j, mn] = 0.5 * (AGz[mn, pp] ** 2) * (Gz[j, mn] ** 2)
        # for mn in range(1, 7):
        #     dg_mode_pe_z_frac[pp, j, mn] = 0.5 * (AG[mn, pp] ** 2) * N2_all[j] * (G[j, mn] ** 2)

# f, ax = plt.subplots(5, 1, sharex=True)
# dps = [0, 40, 65, 115, 165]
# colo = ['r', 'g', 'b', 'k', 'c']
# ppe = ed_in_2 - 1
# count = 0
# for i in dps:
#     for pp in range(60):
#         ax[count].plot(np.array([0, 1, 2, 3, 4]),
#                        np.array([tke_m0_z[i, pp], tke_m1_z[i, pp], tke_m2_z[i, pp], tke_m3_z[i, pp], tke_m4_z[i, pp]]),
#                        color='b', linewidth=0.5)
#         if pp > 58:
#             ax[count].plot(np.array([0, 1, 2, 3, 4]),
#                            np.array([tke_m0_z[i, ppe], tke_m1_z[i, ppe],
#                                      tke_m2_z[i, ppe], tke_m3_z[i, ppe], tke_m4_z[i, ppe]]),
#                             color='r', linewidth=1.75)
#     if count < 2:
#         ax[count].set_ylim([0, 0.02])
#     else:
#         ax[count].set_ylim([0, 0.005])
#     ax[count].set_ylabel(r'KE [m$^2$/s$^2$]')
#     ax[count].grid()
#     ax[count].set_title('KE at ' + str(grid[i]) + 'm', fontsize='10')
#     count = count + 1
# ax[count - 1].set_xlabel('Mode Number')
# ax[count - 1].grid()
# plot_pro(ax[count - 1])

# -----------------------
# f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
# colors = ['#00BFFF', '#F4A460', '#00FF7F', '#FA8072', '#708090']
# # background ke
# ax0.fill_betweenx(grid, 0, np.nanmean(tke_m0_z / tke_tot_z, axis=1),
#                   label='Mode 0', color=colors[0])
# ax0.fill_betweenx(grid, np.nanmean(tke_m0_z / tke_tot_z, axis=1),
#                   np.nanmean((tke_m0_z + tke_m1_z) / tke_tot_z, axis=1),
#                   label='Mode 1', color=colors[1])
# ax0.fill_betweenx(grid, np.nanmean((tke_m0_z + tke_m1_z) / tke_tot_z, axis=1),
#                   np.nanmean((tke_m0_z + tke_m1_z + tke_m2_z) / tke_tot_z, axis=1),
#                   label='Mode 1', color=colors[2])
# ax0.fill_betweenx(grid, np.nanmean((tke_m0_z + tke_m1_z + tke_m2_z) / tke_tot_z, axis=1),
#                   np.nanmean((tke_m0_z + tke_m1_z + tke_m2_z + tke_m3_z) / tke_tot_z, axis=1),
#                   label='Mode 1', color=colors[3])
# ax0.fill_betweenx(grid, np.nanmean((tke_m0_z + tke_m1_z + tke_m2_z + tke_m3_z) / tke_tot_z, axis=1),
#                   np.nanmean((tke_m0_z + tke_m1_z + tke_m2_z + tke_m3_z + tke_m4_z) / tke_tot_z, axis=1),
#                   label='Mode 1', color=colors[4])
# # background pe
# # ax2.fill_betweenx(grid, 0, np.nanmean(pe_m1_z / pe_tot_z, axis=1),
# #                   label='Mode 0', color=colors[1])
# # ax2.fill_betweenx(grid, np.nanmean(pe_m1_z / pe_tot_z, axis=1),
# #                   np.nanmean((pe_m1_z + pe_m2_z) / pe_tot_z, axis=1),
# #                   label='Mode 1', color=colors[2])
# # ax2.fill_betweenx(grid, np.nanmean((pe_m1_z + pe_m2_z) / pe_tot_z, axis=1),
# #                   np.nanmean((pe_m1_z + pe_m2_z + pe_m3_z) / pe_tot_z, axis=1),
# #                   label='Mode 1', color=colors[3])
# # ax2.fill_betweenx(grid, np.nanmean((pe_m1_z + pe_m2_z + pe_m3_z) / pe_tot_z, axis=1),
# #                   np.nanmean((pe_m1_z + pe_m2_z + pe_m3_z + pe_m4_z) / pe_tot_z, axis=1),
# #                   label='Mode 1', color=colors[4])
# # eddy ke
# ax1.fill_betweenx(grid, 0, tke_m0_z[:, ed_in_2][:, 0] / tke_tot_z[:, ed_in_2][:, 0], label='Mode 0', color=colors[0])
# ax1.fill_betweenx(grid, tke_m0_z[:, ed_in_2][:, 0] / tke_tot_z[:, ed_in_2][:, 0],
#                   (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0]) / tke_tot_z[:, ed_in_2][:, 0],
#                   label='Mode 1', color=colors[1])
# ax1.fill_betweenx(grid, (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0]) / tke_tot_z[:, ed_in_2][:, 0],
#                   (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0] + tke_m2_z[:, ed_in_2][:, 0]) /
#                   tke_tot_z[:, ed_in_2][:, 0], label='Mode 2', color=colors[2])
# ax1.fill_betweenx(grid, (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0] + tke_m2_z[:, ed_in_2][:, 0]) /
#                   tke_tot_z[:, ed_in_2][:, 0],
#                   (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0] + tke_m2_z[:, ed_in_2][:, 0] +
#                    tke_m3_z[:, ed_in_2][:, 0]) / tke_tot_z[:, ed_in_2][:, 0], label='Mode 3', color=colors[3])
# ax1.fill_betweenx(grid, (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0] + tke_m2_z[:, ed_in_2][:, 0] +
#                   tke_m3_z[:, ed_in_2][:, 0]) / tke_tot_z[:, ed_in_2][:, 0],
#                   (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0] + tke_m2_z[:, ed_in_2][:, 0] +
#                   tke_m3_z[:, ed_in_2][:, 0] + tke_m4_z[:, ed_in_2][:, 0]) /
#                   tke_tot_z[:, ed_in_2][:, 0], label='Mode 4', color=colors[4])
# ax0.set_xlim([0, 1])
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(handles, labels, fontsize=12)
# ax0.set_title('Mean KE Partition')
# ax1.set_title('Eddy KE Partition')
# ax0.set_xlabel('Fraction')
# ax1.set_xlabel('Fraction')
# ax0.set_ylabel('Depth [m]')
# # ax2.set_title('Mean PE Partition')
# # ax3.set_title('Eddy PE Partition')
# ax0.invert_yaxis()
# ax0.grid()
# # ax1.grid()
# # ax2.grid()
# plot_pro(ax1)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# -- OLD PLOTS
# ---- PLAN VIEW REFERENCE
plot_map = 0
if plot_map > 0:
    # bathymetry
    bath = '/Users/jake/Desktop/bats/bats_bathymetry/GEBCO_2014_2D_-67.7_29.8_-59.9_34.8.nc'
    bath_fid = Dataset(bath, 'r')
    bath_lon = bath_fid.variables['lon'][:]
    bath_lat = bath_fid.variables['lat'][:]
    bath_z = bath_fid.variables['elevation'][:]
    levels = [-5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0]
    fig0, ax0 = plt.subplots()
    cmap = plt.cm.get_cmap("Blues_r")
    cmap.set_over('#808000')  # ('#E6E6E6')
    bc = ax0.contourf(bath_lon, bath_lat, bath_z, levels, cmap='Blues_r', extend='both', zorder=0)
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    bcl = ax0.contour(bath_lon, bath_lat, bath_z, [-4500, -4000], colors='k', zorder=0)
    ml = [(-65, 31.5), (-64.4, 32.435)]
    ax0.clabel(bcl, manual=ml, inline_spacing=-3, fmt='%1.0f', colors='k')
    ax0.plot(prof_lon_i, prof_lat_i, color='#DAA520', linewidth=2)
    ax0.plot(lon[:, -1], lat[:, -1], color='#DAA520',
            label='Dives (' + str(int(profile_tags[0])) + '-' + str(int(profile_tags[-2])) + ')', zorder=1)
    ax0.plot([-64.8, -63.59], [31.2, 31.2], color='w', zorder=2)
    ax0.text(-64.3, 31.1, '115km', color='w', fontsize=12, fontweight='bold')
    ax0.scatter(-(64 + (10 / 60)), 31 + (40 / 60), s=50, color='#E6E6FA', edgecolors='k', zorder=3)
    ax0.scatter(-(64 + (10 / 60)), 31 + (40 / 60), s=50, color='#E6E6FA', edgecolors='k', zorder=4)
    ax0.text(-(64 + (10 / 60)) + .05, 31 + (40 / 60) - .07, 'Sta. BATS', color='w', fontsize=12, fontweight='bold')
    w = 1 / np.cos(np.deg2rad(ref_lat))
    ax0.axis([-66, -63, 31, 33])
    ax0.set_aspect(w)
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig0.colorbar(bc, cax=cax, label='[m]')
    ax0.set_xlabel('Longitude', fontsize=14)
    ax0.set_ylabel('Latitude', fontsize=14)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, fontsize=10)
    ax0.set_title('Deepglider BATS Deployment: ' + np.str(t_s.month) + '/' + np.str(t_s.day) + '/' + np.str(
        t_s.year) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day) + '/' + np.str(t_e.year), fontsize=14)
    plt.tight_layout()
    ax0.grid()
    plot_pro(ax0)
# -------------------------------------------------------------------------------------------------------------------

# WORK ON CURVE FITTING TO FIND BREAK IN SLOPES 
# xx = sc_x
# yy = avg_PE[1:]/dk
# export to use findchangepts in matlab 
# np.savetxt('test_line_fit_x',xx)
# np.savetxt('test_line_fit_y',yy)
# --- index 11 is the point where the break in slope occurs
# ipoint = 6 # 11 
# x_53 = np.log10(xx[0:ipoint])
# y_53 = np.log10(yy[0:ipoint])
# slope1 = np.polyfit(x_53,y_53,1)
# x_3 = np.log10(xx[ipoint:])
# y_3 = np.log10(yy[ipoint:])
# slope2 = np.polyfit(x_3,y_3,1)
# y_g_53 = np.polyval(slope1,x_53)
# y_g_3 = np.polyval(slope2,x_3)

# vert_wave = sc_x/1000
# alpha = 10
# ak0 = xx[ipoint]/1000
# E0 = np.mean(yy[ipoint-3:ipoint+4])
# ak = vert_wave/ak0
# one = E0*( (ak**(5*alpha/3))*(1 + ak**(4*alpha/3) ) )**(-1/alpha)
# # enstrophy/energy transfers 
# mu = (1.88e-3)/(1 + 0.03222*theta_avg + 0.002377*theta_avg*theta_avg)
# nu = mu/sw.dens(salin_avg, theta_avg, grid_p)
# avg_nu = np.nanmean(nu)
# enst_xfer = (E0*ak0**3)**(3/2)
# ener_xfer = (E0*ak0**(5/3))**(3/2)
# enst_diss = np.sqrt(avg_nu)/((enst_xfer)**(1/6))

# figure
# fig,ax = plt.subplots()
# ax.plot(sc_x,avg_PE[1:]/dk,color='k')
# ax.plot(vert_wave*1000,one,color='b',linewidth=0.75)
# ax.plot(10**x_53, 10**y_g_53,color='r',linewidth=0.5,linestyle='--')
# ax.plot(10**x_3, 10**y_g_3,color='r',linewidth=0.5,linestyle='--')
# ax.text(10**x_53[0]-.01, 10**y_g_53[0],str(float("{0:.2f}".format(slope1[0]))),fontsize=8)
# ax.text(10**x_3[0]+.075, 10**y_g_3[0],str(float("{0:.2f}".format(slope2[0]))),fontsize=8)
# ax.plot([xx[ipoint], xx[ipoint]],[3*10**(-4), 2*10**(-3)],color='k',linewidth=2)
# ax.text(xx[ipoint+1], 2*10**(-3), str('=') + str(float("{0:.2f}".format(xx[ipoint])))+'km')
# ax.axis([10**-2, 10**1, 3*10**(-4), 2*10**(3)])
# ax.set_yscale('log')
# ax.set_xscale('log')
# plot_pro(ax)

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# -- inspect ability of G to describe individual eta profiles
# todo look at eof of eta profiles as a do the velocity profiles
# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
# ax1.plot(Eta2[:, 90], grid)
# ax1.plot(Eta_m[:, 90], grid, color='k', linestyle='--')
# ax1.axis([-600, 600, 0, 5000])
# ax1.invert_yaxis()
# ax2.plot(Eta2[:, 140], grid)
# ax2.plot(Eta_m[:, 140], grid, color='k', linestyle='--')
# ax2.axis([-600, 600, 0, 5000])
# ax2.invert_yaxis()
# ax3.plot(Eta2[:, 180], grid)
# ax3.plot(Eta_m[:, 180], grid, color='k', linestyle='--')
# ax3.axis([-600, 600, 0, 5000])
# ax3.invert_yaxis()
# ax4.plot(Eta2[:, 200], grid)
# ax4.plot(Eta_m[:, 200], grid, color='k', linestyle='--')
# ax4.axis([-600, 600, 0, 5000])
# ax4.invert_yaxis()
# plot_pro(ax4)
# todo fraction of variance unexplained should be between each profile alpha*F(z) and EOF1?

# --- BIN DEPTH (computed from grids, collect_dives, make_bin)
# limit bin depth to 4500 (to prevent fitting of velocity profiles past points at which we have data) (previous = 5000)
#  bin_depth = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4500, 20)])
# GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
# bin_depth = GD.variables['grid'][:]
# grid = bin_depth
# grid_p = gsw.p_from_z(-1 * grid, ref_lat)
# z = -1 * grid
# sz_g = grid.shape[0]
# df_lon = pd.DataFrame(GD['Longitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
# df_lat = pd.DataFrame(GD['Latitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
# df_lon[df_lon < -500] = np.nan
# df_lat[df_lat < -500] = np.nan
# ref_lon = np.nanmean(df_lon)
# ref_lat = np.nanmean(df_lat)

# --- LOAD gridded dives (gridded dives) --- NO OTHER PROCESSING HAS BEEN DONE
# df_den = pd.DataFrame(GD['Density'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
# df_theta = pd.DataFrame(GD['Theta'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
# df_ct = pd.DataFrame(GD['Conservative Temperature'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
# df_s = pd.DataFrame(GD['Absolute Salinity'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
# sig_var = np.nan * np.ones(df_den.shape)
# for i in range(df_den.shape[1]):
#     sig_var[:, i] = gsw.sigma2(df_s.iloc[:, i], df_ct.iloc[:, i])
# # redefine to test alternate potential density reference
# df_den = pd.DataFrame(sig_var[0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
# dac_u = GD.variables['DAC_u'][:]
# dac_v = GD.variables['DAC_v'][:]
# prof_lon_i = GD.variables['Longitude'][:][:]
# prof_lat_i = GD.variables['Latitude'][:][:]
# prof_lon_i[prof_lon_i < -360] = np.nan
# prof_lat_i[prof_lat_i < -90] = np.nan
# prof_lon_i = np.nanmean(prof_lon_i, axis=0)
# prof_lat_i = np.nanmean(prof_lat_i, axis=0)
# time_rec = GD.variables['time_start_stop'][:]
# time_rec_all = GD.variables['time_start_stop'][:]
# profile_list = np.float64(GD.variables['dive_list'][:]) - 35000
# df_den[df_den < 0] = np.nan
# df_theta[df_theta < 0] = np.nan
# df_ct[df_ct < 0] = np.nan
# df_s[df_s < 0] = np.nan
# dac_u[dac_u < -500] = np.nan
# dac_v[dac_v < -500] = np.nan
# t_s = datetime.date.fromordinal(np.int(np.min(time_rec_all)))
# t_e = datetime.date.fromordinal(np.int(np.max(time_rec_all)))

# # --- compute 4 seasonal averages (as with station bats)
# # -- construct four background profiles to represent seasons
# d_spring = np.where((time_rec_all > 735658) & (time_rec_all < 735750))[0]       # Mar 1 - June 1
# d_summer = np.where((time_rec_all > 735750) & (time_rec_all < 735842))[0]       # June 1 - Sept 1
# d_fall = np.where((time_rec_all > 735842) & (time_rec_all < 735903))[0]         # Sept 1 - Nov 1
# d_winter = np.where((time_rec_all > 735903) | (time_rec_all < 735658))[0]          # Nov 1 - Mar 1
# bckgrds = [d_spring, d_summer, d_fall, d_winter]
# bckgrds_wins = np.array([735658, 735750, 735842, 735903])
# salin_avg = np.nan * np.zeros((len(grid), 4))
# conservative_t_avg = np.nan * np.zeros((len(grid), 4))
# theta_avg = np.nan * np.zeros((len(grid), 4))
# sigma_theta_avg = np.nan * np.zeros((len(grid), 4))
# ddz_avg_sigma = np.nan * np.zeros((len(grid), 4))
# N2 = np.nan * np.zeros(sigma_theta_avg.shape)
# N = np.nan * np.zeros(sigma_theta_avg.shape)
# for i in range(4):
#     inn = bckgrds[i]
#     salin_avg[:, i] = np.nanmean(df_s.iloc[:, inn], axis=1)
#     conservative_t_avg[:, i] = np.nanmean(df_ct.iloc[:, inn], axis=1)
#     theta_avg[:, i] = np.nanmean(df_theta.iloc[:, inn], axis=1)
#     sigma_theta_avg[:, i] = np.nanmean(df_den.iloc[:, inn], axis=1)
#     ddz_avg_sigma[:, i] = np.gradient(sigma_theta_avg[:, i], z)
#     go = ~np.isnan(salin_avg[:, i])
#     N2[np.where(go)[0][0:-1], i] = gsw.Nsquared(salin_avg[go, i], conservative_t_avg[go, i], grid_p[go], lat=ref_lat)[0]
#     N2[N2[:, i] < 0] = np.nan
#     N2[:, i] = nanseg_interp(grid, N2[:, i])

# --- mission average background properties from DG (all profiles)
# sigma_theta_avg = df_den.mean(axis=1)
# ct_avg = df_ct.mean(axis=1)
# theta_avg = df_theta.mean(axis=1)
# salin_avg = df_s.mean(axis=1)
# ddz_avg_sigma = np.gradient(sigma_theta_avg, z)
# ddz_avg_theta = np.gradient(theta_avg, z)
# N2_old = np.nan * np.zeros(sigma_theta_avg.size)
# N2_old[1:] = np.squeeze(sw.bfrq(salin_avg, theta_avg, grid_p, lat=ref_lat)[0])

# N2_all = np.nan * np.zeros(len(grid))
# N2_all[0:-1] = gsw.Nsquared(np.nanmean(salin_avg, axis=1), np.nanmean(conservative_t_avg, axis=1),
#                             grid_p, lat=ref_lat)[0]
# N2_all[-2:] = N2_all[-3]
# N2_all[N2_all < 0] = np.nan
# N2_all = nanseg_interp(grid, N2_all)
# N_all = np.sqrt(N2_all)
# N2_all = savgol_filter(N2_all, 5, 3)

# ---- 2014 initial comparison
# why are 2014 DACs very! large --- need some reprocessing??
# GD = Dataset('BATs_2014_gridded.nc', 'r')
# df_lon2 = pd.DataFrame(GD['Longitude'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
# df_lat2 = pd.DataFrame(GD['Latitude'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
# profile_list2 = np.float64(GD.variables['dive_list'][:]) - 35000
# dac_u2 = GD.variables['DAC_u'][:]
# dac_v2 = GD.variables['DAC_v'][:]
# df_lon2[df_lon2 < -500] = np.nan
# df_lat2[df_lat2 < -500] = np.nan
# dac_u2[dac_u2 < -500] = np.nan
# dac_v2[dac_v2 < -500] = np.nan
# f, ax = plt.subplots()
# ax.quiver(np.nanmean(df_lon, axis=0), np.nanmean(df_lat, axis=0), dac_u, dac_v, color='r', scale=1)
# ax.quiver(np.nanmean(df_lon2, axis=0), np.nanmean(df_lat2, axis=0), dac_u2, dac_v2, color='k', scale=1)
# plot_pro(ax)
# BATS
# take velocity and displacement profiles and compute energy spectra / explore mode amplitude variability
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gsw
import seawater as sw
import pandas as pd
import scipy
import scipy.io as si
from scipy.optimize import fmin
from scipy.signal import savgol_filter
from netCDF4 import Dataset
import pickle
import datetime
# functions I've written 
from mode_decompositions import vertical_modes, PE_Tide_GM, vertical_modes_f
from toolkit import nanseg_interp, plot_pro


def functi(p, xe, xb):
    #  This is the target function that needs to be minimized
    fsq = (xe - p*xb)**2
    return fsq.sum()


# physical parameters
g = 9.81
rho0 = 1027
# limit bin depth to 4500 (to prevent fitting of velocity profiles past points at which we have data) (previous = 5000)
#  bin_depth = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4500, 20)])
GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
bin_depth = GD.variables['grid'][:]
df_lon = pd.DataFrame(GD['Longitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lat = pd.DataFrame(GD['Latitude'][:], index=GD['grid'][:], columns=GD['dive_list'][:])
df_lon[df_lon < -500] = np.nan
df_lat[df_lat < -500] = np.nan
ref_lon = np.nanmean(df_lon)
ref_lat = np.nanmean(df_lat)
grid = bin_depth
grid_p = gsw.p_from_z(-1 * grid, ref_lat)
z = -1 * grid
sz_g = grid.shape[0]
# MODE PARAMETERS
# frequency zeroed for geostrophic modes
omega = 0
# highest baroclinic mode to be calculated
mmax = 60
nmodes = mmax + 1
# maximum allowed deep shear [m/s/km]
deep_shr_max = 0.1
# minimum depth for which shear is limited [m]
deep_shr_max_dep = 3500

# --- LOAD gridded dives (gridded dives)
df_den = pd.DataFrame(GD['Density'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
df_theta = pd.DataFrame(GD['Theta'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
df_ct = pd.DataFrame(GD['Conservative Temperature'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
df_s = pd.DataFrame(GD['Absolute Salinity'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
dac_u = GD.variables['DAC_u'][:]
dac_v = GD.variables['DAC_v'][:]
time_rec = GD.variables['time_start_stop'][:]
time_rec_all = GD.variables['time_start_stop'][:]
profile_list = np.float64(GD.variables['dive_list'][:]) - 35000
df_den[df_den < 0] = np.nan
df_theta[df_theta < 0] = np.nan
df_ct[df_ct < 0] = np.nan
df_s[df_s < 0] = np.nan
dac_u[dac_u < -500] = np.nan
dac_v[dac_v < -500] = np.nan
t_s = datetime.date.fromordinal(np.int(np.min(time_rec_all)))
t_e = datetime.date.fromordinal(np.int(np.max(time_rec_all)))

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

# ---- LOAD IN TRANSECT TO PROFILE DATA COMPILED IN BATS_TRANSECTS.PY
# pkl_file = open('/Users/jake/Desktop/bats/transect_profiles_mar12_2.pkl', 'rb')
# pkl_file = open('/Users/jake/Desktop/bats/dep15_transect_profiles_mar23.pkl', 'rb')
pkl_file = open('/Users/jake/Desktop/bats/dep15_transect_profiles_may01.pkl', 'rb')
bats_trans = pickle.load(pkl_file)
pkl_file.close()
Time = bats_trans['Time']
Info = bats_trans['Info']
Sigma_Theta = bats_trans['Sigma_Theta'][0:sz_g, :]
Eta = bats_trans['Eta'][0:sz_g, :]
Eta_theta = bats_trans['Eta_theta'][0:sz_g, :]
V = bats_trans['V'][0:sz_g, :]
prof_lon = bats_trans['V_lon']
prof_lat = bats_trans['V_lat']

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
    ax0.plot(df_lon, df_lat, color='#DAA520', linewidth=2)
    ax0.plot(df_lon.iloc[:, -1], df_lat.iloc[:, -1], color='#DAA520',
            label='Dives (' + str(int(profile_list[0])) + '-' + str(int(profile_list[-2])) + ')', zorder=1)
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
# -----------------------------------------------------------------------------------

# --- AVERAGE background properties of profiles along these transects
sigma_theta_avg = df_den.mean(axis=1)
ct_avg = df_ct.mean(axis=1)
theta_avg = df_theta.mean(axis=1)
salin_avg = df_s.mean(axis=1)
ddz_avg_sigma = np.gradient(sigma_theta_avg, z)
ddz_avg_theta = np.gradient(theta_avg, z)
N2 = np.nan * np.zeros(sigma_theta_avg.size)
N2_old = np.nan * np.zeros(sigma_theta_avg.size)
N2_old[1:] = np.squeeze(sw.bfrq(salin_avg, theta_avg, grid_p, lat=ref_lat)[0])
N2[0:-1] = gsw.Nsquared(salin_avg, ct_avg, grid_p, lat=ref_lat)[0]
N2[-2:] = N2[-3]
N2[N2 < 0] = np.nan
N2 = nanseg_interp(grid, N2)
N = np.sqrt(N2)

window_size = 5
poly_order = 3
N2 = savgol_filter(N2, window_size, poly_order)

# --- compute vertical mode shapes
G, Gz, c = vertical_modes(N2, grid, omega, mmax)

# --- compute alternate vertical modes
bc_bot = 2  # 1 = flat, 2 = rough
grid2 = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4500, 10)])
n2_interp = np.interp(grid2, grid, N2)
F_int_g2, F_g2, c_ff, norm_constant = vertical_modes_f(n2_interp, grid2, omega, mmax, bc_bot)
F = np.nan * np.ones(G.shape)
F_int = np.nan * np.ones(G.shape)
for i in range(mmax + 1):
    F[:, i] = np.interp(grid, grid2, F_g2[:, i])
    F_int[:, i] = np.interp(grid, grid2, F_int_g2[:, i])

# bc_bot = 2  # 1 = flat, 2 = rough
# F_int_g3, F_g3, c_ff3, norm_constant3 = vertical_modes_f(n2_interp, grid2, omega, mmax, bc_bot)
# F_2 = np.nan * np.ones(G.shape)
# F_int_2 = np.nan * np.ones(G.shape)
# for i in range(mmax + 1):
#     F_2[:, i] = np.interp(grid, grid2, F_g3[:, i])
#     F_int_2[:, i] = np.interp(grid, grid2, F_int_g3[:, i])
# -----------------------------------------------------------------------------------

# ----- SOME VELOCITY PROFILES ARE TOO NOISY AND DEEMED UNTRUSTWORTHY --------------
# select only velocity profiles that seem reasonable
# criteria are slope of v (dont want kinks)
# criteria: limit surface velocity to greater that 40cm/s
good_v = np.zeros(np.size(Time))
v_dz = np.zeros(np.shape(V))
v_max = np.zeros(np.size(Time))
for i in range(np.size(Time)):
    v_max[i] = np.nanmax(np.abs(V[:, i]))
    v_dz[5:-20, i] = np.gradient(V[5:-20, i], z[5:-20])
    if np.nanmax(np.abs(v_dz[:, i])) < 0.0015:  # 0.075
        good_v[i] = 1
good_v[191] = 1
good_ex = np.where(v_max < 0.4)[0]
good_der = np.where(good_v > 0)[0]
good = np.intersect1d(good_der, good_ex)
V2 = V[:, good].copy()
Eta2 = Eta[:, good].copy()
Eta2_c = Eta[:, good].copy()
Eta_theta2 = Eta_theta[:, good].copy()
Time2 = Time[good].copy()
Info2 = Info[:, good].copy()
prof_lon2 = prof_lon[good].copy()
prof_lat2 = prof_lat[good].copy()
for i in range(len(Time2)):
    y_i = Eta2[:, i]
    if np.sum(np.isnan(y_i)) > 0:
        Eta2[:, i] = nanseg_interp(grid, y_i)
# -----------------------------------------------------------------------------------
# ---- PROJECT MODES ONTO EACH PROFILE -------
sz = np.shape(Eta2)
num_profs = sz[1]
eta_fit_depth_min = 100
eta_fit_depth_max = 3800  # 3900
eta_theta_fit_depth_max = 4200
AG = np.zeros([nmodes, num_profs])
AGz = np.zeros([nmodes, num_profs])
AG_theta = np.zeros([nmodes, num_profs])
Eta_m = np.nan * np.zeros([np.size(grid), num_profs])
V_m = np.nan * np.zeros([np.size(grid), num_profs])
Neta = np.nan * np.zeros([np.size(grid), num_profs])
NEta_m = np.nan * np.zeros([np.size(grid), num_profs])
Eta_theta_m = np.nan * np.zeros([np.size(grid), num_profs])
PE_per_mass = np.nan * np.zeros([nmodes, num_profs])
HKE_per_mass = np.nan * np.zeros([nmodes, num_profs])
PE_theta_per_mass = np.nan * np.zeros([nmodes, num_profs])
modest = np.arange(11, nmodes)
good_ke_prof = np.ones(num_profs)
HKE_noise_threshold = 1e-4  # 1e-5
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
    Neta[:, i] = N * this_eta
    this_eta_theta = Eta_theta2[:, i].copy()
    iw = np.where((grid >= eta_fit_depth_min) & (grid <= eta_fit_depth_max))
    iw_theta = np.where((grid >= eta_fit_depth_min) & (grid <= eta_theta_fit_depth_max))
    if len(iw[0]) > 1:
        eta_fs = Eta2[:, i].copy()  # ETA
        eta_theta_fs = Eta_theta2[:, i].copy()  # ETA THETA
        # -- taper fit as z approaches 0
        i_sh = np.where((grid < eta_fit_depth_min))
        eta_fs[i_sh[0]] = grid[i_sh] * this_eta[iw[0][0]] / grid[iw[0][0]]
        eta_theta_fs[i_sh[0]] = grid[i_sh] * this_eta_theta[iw[0][0]] / grid[iw[0][0]]
        # -- taper fit as z approaches -H
        i_dp = np.where((grid > eta_fit_depth_max))
        eta_fs[i_dp[0]] = (grid[i_dp] - grid[-1]) * this_eta[iw[0][-1]] / (grid[iw[0][-1]] - grid[-1])

        i_dp_theta = np.where((grid > eta_theta_fit_depth_max))
        eta_theta_fs[i_dp_theta[0]] = (grid[i_dp_theta] - grid[-1]) * this_eta_theta[iw_theta[0][-1]] / (
                grid[iw_theta[0][-1]] - grid[-1])
        # -- solve matrix problem
        AG[1:, i] = np.linalg.lstsq(G[:, 1:], eta_fs[:, np.newaxis])[0][:, 0]
        # AG[1:, i] = np.linalg.lstsq(F_int[:, 1:], eta_fs[:, np.newaxis])[0][:, 0]
        AG_theta[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], np.transpose(np.atleast_2d(eta_theta_fs)))[0])

        Eta_m[:, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG[:, i])))
        # Eta_m[:, i] = np.squeeze(np.matrix(F_int) * np.transpose(np.matrix(AG[:, i])))
        NEta_m[:, i] = N * np.array(np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG[:, i]))))
        Eta_theta_m[:, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG_theta[:, i])))
        PE_per_mass[:, i] = (1 / 2) * AG[:, i] * AG[:, i] * c * c
        PE_theta_per_mass[:, i] = (1 / 2) * AG_theta[:, i] * AG_theta[:, i] * c * c
    # output density structure for comparison

sa = 0
if sa > 0:
    mydict = {'bin_depth': grid, 'eta': Eta2, 'dg_v': V2}
    output = open('/Users/jake/Desktop/bats/den_v_profs.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()
# ------- END OF ITERATION ON EACH PROFILE TO COMPUTE MODE FITS
# --------------------------------------------------------------------------------------------------


# ---- COMPUTE EOF SHAPES AND COMPARE TO ASSUMED STRUCTURE

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
not_deep = np.isfinite(V2[-9, :])  # & (Time2 > 735750)
V3 = V2[:, not_deep]
check1 = 3      # upper index to include in eof computation
check2 = -8     # lower index to include in eof computation
grid_check = grid[check1:check2]
Uzq = V3[check1:check2, :].copy()
nq = np.size(V3[0, :])
avg_Uzq = np.nanmean(np.transpose(Uzq), axis=0)
Uzqa = Uzq - np.transpose(np.tile(avg_Uzq, [nq, 1]))
cov_Uzqa = (1 / nq) * np.matrix(Uzqa) * np.matrix(np.transpose(Uzqa))
D_Uzqa, V_Uzqa = np.linalg.eig(cov_Uzqa)

t1 = np.real(D_Uzqa[0:10])
PEV = t1 / np.sum(t1)
# ----------------------------------------------------------------------

# ------ VARIANCE EXPLAINED BY BAROCLINIC MODES ------------------------
eof1 = np.array(np.real(V_Uzqa[:, 0]))
eof1_sc = (1/2)*(eof1.max() - eof1.min()) + eof1.min()
bc1 = Gz[check1:check2, 1]  # flat bottom
bc2 = F[check1:check2, 0]   # sloping bottom

# -- minimize mode shapes onto eof shape
p = 0.8*eof1.min()/np.max(np.abs(F[:, 0]))
ins1 = np.transpose(np.concatenate([eof1, bc1[:, np.newaxis]], axis=1))
ins2 = np.transpose(np.concatenate([eof1, bc2[:, np.newaxis]], axis=1))
min_p1 = fmin(functi, p, args=(tuple(ins1)))
min_p2 = fmin(functi, p, args=(tuple(ins2)))

# -- plot inspect minimization of mode shapes
# f, ax = plt.subplots()
# ax.plot(eof1, grid_check, color='k')
# ax.plot(bc1*min_p1, grid_check, color='r')
# ax.plot(bc2*min_p2, grid_check, color='b')
# ax.invert_yaxis()
# plot_pro(ax)
fvu1 = np.sum((eof1[:, 0] - bc1*min_p1)**2)/np.sum((eof1 - np.mean(eof1))**2)
fvu2 = np.sum((eof1[:, 0] - bc2*min_p2)**2)/np.sum((eof1 - np.mean(eof1))**2)
# ---------------------------------------------------------------------

# --- PLOT V STRUCTURE
plot_v_struct = 0
if plot_v_struct > 0:
    f, (ax, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    for i in range(nq):
        ax.plot(V3[:, i], grid, color='#5F9EA0', linewidth=0.75)
    ax.plot(np.nanmean(np.abs(V3), axis=1), grid, color='k', label='Average |V|')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10)
    ax2.plot(np.zeros(10), np.arange(0, 5000, 500), color='k')
    ax3.plot(np.zeros(10), np.arange(0, 5000, 500), color='k')
    # ax4.plot(np.zeros(10), np.arange(0, 5000, 500), color='k')
    for i in range(4):
        ax2.plot(V_Uzqa[:, i], grid_check, label=r'PEV$_{' + str(i + 1) + '}$ = ' + str(100 * np.round(PEV[i], 3)),
                 linewidth=2)
        ax3.plot(F[:, i], grid, label='Mode' + str(i), linewidth=2)
        ax3.plot(Gz[:, i], grid, c='k', linestyle='--', linewidth=0.75)

        if i < 1:
            ax4.plot(F_int[:, i] + np.nanmax(np.abs(F_int[:, i])), grid)
        else:
            ax4.plot(F_int[:, i], grid)

        ax4.plot(G[:, i], grid, c='k', linestyle='--', linewidth=0.5)
        # ax4.plot(Gz[:, i], grid, c='k', linestyle='--', linewidth=0.5)
        # ax4.plot(F_2[:, i], grid, label='Mode' + str(i))
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=12)
    ax.axis([-.3, .3, 0, 4800])
    ax.set_title('Cross-Track Velocity [V]', fontsize=16)
    ax.set_xlabel('m/s', fontsize=16)
    ax.set_ylabel('Depth [m]', fontsize=16)
    ax2.axis([-.2, .2, 0, 4800])
    ax2.set_title('Principle EOFs of V', fontsize=16)
    ax2.set_xlabel('m/s', fontsize=16)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, fontsize=10)
    ax3.axis([-5, 5, 0, 4800])
    ax3.set_title(r'Vertical Structure of V $\phi$(z): BC' + str(bc_bot), fontsize=16)
    ax3.set_xlabel('Mode Amplitude', fontsize=16)
    ax4.set_title('Vertical Structure of Disp. G(z)')
    ax4.set_xlabel('Mode Amplitude')
    ax.invert_yaxis()
    ax.grid()
    ax2.grid()
    plot_pro(ax3)

# --- Isolate eddy dives
# 2015 - dives 62, 63 ,64
ed_prof_in = np.where(((profile_list) >= 62) & ((profile_list) <= 64))[0]
ed_in = np.where(((Info2[0, :] - 35000) >= 62) & ((Info2[0, :] - 35000) <= 64))[0]
ed_in_2 = np.where(((Info2[0, :] - 35000) > 61) & ((Info2[0, :] - 35000) < 63))[0]
ed_time_s = datetime.date.fromordinal(np.int(Time2[ed_in[0]]))
ed_time_e = datetime.date.fromordinal(np.int(Time2[ed_in[-1] + 1]))
# --- time series
mission_start = datetime.date.fromordinal(np.int(Time.min()))
mission_end = datetime.date.fromordinal(np.int(Time.max()))

# --- EDDY TEMPERATURE ANOMALY
# f, ax = plt.subplots()
# for i in range(len(ed_prof_in)):
#     ax.plot(df_ct.iloc[:, ed_prof_in[i]], grid)
# ax.plot(ct_avg, grid, color='k')
# ax.set_xlabel('T [degrees]')
# ax.set_ylabel('Depth [m]')
# ax.set_xlim([0, 20])
# ax.invert_yaxis()
# plot_pro(ax)


# --- PLOT ETA / EOF
noisy_profs = np.where(AGz[4, :] < -0.1)[0]
plot_eta = 0
if plot_eta > 0:
    f, (ax2, ax1, ax0) = plt.subplots(1, 3, sharey=True)
    for j in range(len(Time)):
        ax2.plot(Sigma_Theta[:, j] - sigma_theta_avg, grid, linewidth=0.75)
    ax2.set_xlim([-.5, .5])
    ax2.set_xlabel(r'$\sigma_{\theta} - \overline{\sigma_{\theta}}$', fontsize=12)
    ax2.set_title("DG35 BATS: " + str(mission_start) + ' - ' + str(mission_end))
    ax2.text(0.1, 4000, str(len(Time2)) + ' profiles', fontsize=10)
    for j in range(num_profs):
        ax1.plot(Eta2[:, j], grid, color='#4682B4', linewidth=1.25)
        ax1.plot(Eta_m[:, j], grid, color='k', linestyle='--', linewidth=.75)
        ax0.plot(V2[:, j], grid, color='#4682B4', linewidth=1.25)
        ax0.plot(V_m[:, j], grid, color='k', linestyle='--', linewidth=.75)
    for j in range(num_profs):
        if j in noisy_profs:
            ax0.plot(V2[:, j], grid, color='r', linewidth=2)
    for k in range(ed_in[0], ed_in[-1] + 2):
        ax1.plot(Eta2[:, k], grid, color='m', linewidth=2, label='eddy')
        ax0.plot(V2[:, k], grid, color='m', linewidth=2)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[0]], [labels[0]], fontsize=10)
    ax1.axis([-600, 600, 0, 4750])
    ax0.text(190, 800, str(num_profs) + ' Profiles')
    ax1.set_xlabel(r'Vertical Isopycnal Displacement, $\xi_{\sigma_{\theta}}$ [m]', fontsize=12)
    ax1.set_title(r'Isopycnal Displacement', fontsize=12)  # + '(' + str(Time[0]) + '-' )
    ax0.axis([-.4, .4, 0, 4750])
    ax0.set_title("Geostrophic Velocity", fontsize=12)  # (" + str(num_profs) + 'profiles)' )
    ax2.set_ylabel('Depth [m]', fontsize=12)
    ax0.set_xlabel('Cross-Track Velocity, U [m/s]', fontsize=12)
    ax0.invert_yaxis()
    ax2.grid()
    ax1.grid()
    plot_pro(ax0)
    # f.savefig('/Users/jake/Desktop/bats/dg035_15_Eta_a.png',dpi = 300)
    # plt.show()    

    max_plot = 3
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    n2p = ax1.plot((np.sqrt(N2) * (1800 / np.pi)), grid, color='k', label='N(z) [cph]')
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

# --- MODE AMPLITUDE IN TIME AND SPACE ----------
Time3 = Time2[np.argsort(Time2)]
Time2_dt = []
for i in range(len(Time3)):
    Time2_dt.append(datetime.date.fromordinal(np.int(Time3[i])))
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

plot_mode = 0
if plot_mode > 0:
    window_size, poly_order = 9, 2
    fm, ax = plt.subplots()
    colors = ['#8B0000', '#FF8C00', '#808000', '#5F9EA0', 'g', 'c']
    ax.plot(sb_dt, c[1] * sba_in[1, :], color='m', label='Hydrography Mode 1')
    for mo in range(1, 4):
        orderAG = AG[mo, np.argsort(Time2)]
        y_sg = savgol_filter(orderAG, window_size, poly_order)
        # pm = ax.plot(Time2_dt, AG[mo, np.argsort(Time2)], color=colors[mo - 1], linewidth=0.75)
        ax.plot(Time2_dt, c[mo] * y_sg, color=colors[mo], linewidth=2, label=('Mode ' + str(mo)))
    ax.set_xlim([Time2_dt[0], Time2_dt[-1]])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12)
    ax.set_title(r'Scaled Displacement Mode Amplitude (c$_{n}\beta_{n}$)')
    ax.set_xlabel('Date')
    plot_pro(ax)

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

    fm, (ax1, ax2) = plt.subplots(2, 1)
    for mo in range(3):
        orderAGz = AGz[mo, np.argsort(Time2)]
        y_sgz = savgol_filter(orderAGz, window_size, poly_order)
        pmz = ax1.plot(Time2_dt, AGz[mo, np.argsort(Time2)], color=colors[mo], linewidth=0.75)
        ax1.plot(Time2_dt, y_sgz, color=colors[mo], label=('Mode ' + str(mo)), linewidth=2)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=12)
    ax1.set_title(r'Velocity Mode Amplitude ($\alpha_{n}$)')
    ax1.set_ylabel('Mode Amplitude')
    ax1.set_xlabel('Date')
    ax1.set_ylim([-.12, .12])
    ax1.grid()

    orderAGz = AGz[:, np.argsort(Time2)]
    for mo_p in range(num_profs):
        y_sgz = savgol_filter(orderAGz, window_size, poly_order)
        if good_ke_prof[mo_p] > 0:
            pmz = ax2.plot(np.arange(0, 61), orderAGz[:, mo_p], linewidth=0.75)
    ax2.set_title('Velocity Mode Amplitude (by mode number)')
    ax2.set_ylabel('Mode Amplitude')
    ax2.set_xlabel('Mode Number')
    ax2.set_ylim([-.12, .12])
    plot_pro(ax2)
# ------------------------------------------------------------------------


# --- MODE AMPLITUDE CORRELATIONS IN TIME AND SPACE
plot_mode_corr = 0
if plot_mode_corr > 0:
    x = 1852 * 60 * np.cos(np.deg2rad(ref_lat)) * (prof_lon2 - ref_lon)
    y = 1852 * 60 * (prof_lat2 - ref_lat)
    x_tile = np.tile(x, (len(x), 1))
    y_tile = np.tile(y, (len(y), 1))
    time_tile = np.tile(Time2, (len(Time2), 1))
    dist = np.sqrt((x_tile - x_tile.T) ** 2 + (y_tile - y_tile.T) ** 2) / 1000
    time_lag = np.abs(time_tile - time_tile.T)

    # this mode
    mode_num = 1
    AG_i = AG[mode_num, :]
    AGz_i = AGz[mode_num, :]

    # define each box as all points that fall within a time and space lag
    dist_win = np.arange(0, 100, 10)
    t_win = np.arange(0, 100, 10)
    # try to compute lagged autocorrelation for all points within a given distance
    # returns a tuple of coordinate pairs where distance criteria are met
    corr_i = np.nan * np.zeros((len(t_win), len(dist_win)))
    corr_z_i = np.nan * np.zeros((len(t_win), len(dist_win)))
    for dd in range(len(dist_win) - 1):
        dist_small_i = np.where((dist > dist_win[dd]) & (dist < dist_win[dd + 1]))
        time_in = np.unique(time_lag[dist_small_i[0], dist_small_i[1]])
        AG_out = np.nan * np.zeros([len(dist_small_i[0]), 3])
        AGz_out = np.nan * np.zeros([len(dist_small_i[0]), 3])
        for i in range(len(dist_small_i[0])):
            AG_out[i, :] = [AG_i[dist_small_i[0][i]], AG_i[dist_small_i[1][i]],
                            time_lag[dist_small_i[0][i], dist_small_i[1][i]]]
            AGz_out[i, :] = [AGz_i[dist_small_i[0][i]], AGz_i[dist_small_i[1][i]],
                             time_lag[dist_small_i[0][i], dist_small_i[1][i]]]
        no_doub, no_doub_i = np.unique(AG_out[:, 2], return_index=True)
        AG_out2 = AG_out[no_doub_i, :]
        zno_doub, zno_doub_i = np.unique(AGz_out[:, 2], return_index=True)
        AGz_out2 = AGz_out[zno_doub_i, :]
        for j in range(len(t_win) - 1):
            inn = AG_out2[((AG_out2[:, 2] > t_win[j]) & (AG_out2[:, 2] < t_win[j + 1])), 0:3]
            i_mean = np.mean(inn[:, 0:2])
            n = len(inn[:, 0:2])
            variance = np.var(inn[:, 0:2])
            covi = np.nan * np.zeros(len(inn[:, 0]))
            for k in range(len(inn[:, 0])):
                covi[k] = (inn[k, 0] - i_mean) * (inn[k, 1] - i_mean)
            corr_i[j, dd] = (1 / (n * variance)) * np.sum(covi)

            innz = AGz_out2[((AGz_out2[:, 2] > t_win[j]) & (AGz_out2[:, 2] < t_win[j + 1])), 0:3]
            iz_mean = np.mean(innz[:, 0:2])
            nz = len(innz[:, 0:2])
            variancez = np.var(innz[:, 0:2])
            covzi = np.nan * np.zeros(len(innz[:, 0]))
            for k in range(len(innz[:, 0])):
                covzi[k] = (innz[k, 0] - iz_mean) * (innz[k, 1] - iz_mean)
            corr_z_i[j, dd] = (1 / (nz * variancez)) * np.sum(covzi)
    f, (ax1, ax2) = plt.subplots(1, 2)
    cmap = plt.cm.get_cmap("viridis")
    cmap.set_over('w')  # ('#E6E6E6')
    pa = ax1.pcolor(dist_win, t_win, corr_i, vmin=-1, vmax=1, cmap='viridis')
    paz = ax2.pcolor(dist_win, t_win, corr_z_i, vmin=-1, vmax=1, cmap='viridis')
    ax1.set_xlabel('Spatial Separation [km]')
    ax2.set_xlabel('Spatial Separation [km]')
    ax1.set_ylabel('Time Lag [days]')
    ax1.set_title('Displacement Mode Amplitude')
    ax2.set_title('Velocity Mode Amplitude')
    f.colorbar(pa)
    plot_pro(ax2)


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
for pp in np.append(np.arange(5, 42), np.arange(140, 160)):
    # - loop over each depth
    for j in range(len(grid)):
        tke_tot_z[j, pp] = np.sum(0.5 * ((AGz[0:20, pp] ** 2) * (Gz[j, 0:20] ** 2)))  # ke sum over all modes at depths z
        pe_tot_z[j, pp] = np.sum(0.5 * ((AG[0:20, pp] ** 2) * N2[j] * (G[j, 0:20] ** 2)))  # pe sum over all modes at depths z
        tke_m0_z[j, pp] = 0.5 * ((AGz[0, pp] ** 2) * (Gz[j, 0] ** 2))  # ke mode 0 contribution to tke at depths z
        tke_m1_z[j, pp] = 0.5 * ((AGz[1, pp] ** 2) * (Gz[j, 1] ** 2))  # ke mode 1 contribution to tke at depths z
        tke_m2_z[j, pp] = 0.5 * ((AGz[2, pp] ** 2) * (Gz[j, 2] ** 2))  # ke mode 2 contribution to tke at depths z
        tke_m3_z[j, pp] = 0.5 * ((AGz[3, pp] ** 2) * (Gz[j, 3] ** 2))  # ke mode 3 contribution to tke at depths z
        tke_m4_z[j, pp] = 0.5 * ((AGz[4, pp] ** 2) * (Gz[j, 4] ** 2))  # ke mode 3 contribution to tke at depths z
        pe_m1_z[j, pp] = 0.5 * ((AG[1, pp] ** 2) * N2[j] * (G[j, 1] ** 2))  # pe mode 1 contribution to tke at depths z
        pe_m2_z[j, pp] = 0.5 * ((AG[2, pp] ** 2) * N2[j] * (G[j, 2] ** 2))  # pe mode 1 contribution to tke at depths z
        pe_m3_z[j, pp] = 0.5 * ((AG[3, pp] ** 2) * N2[j] * (G[j, 3] ** 2))  # pe mode 1 contribution to tke at depths z
        pe_m4_z[j, pp] = 0.5 * ((AG[4, pp] ** 2) * N2[j] * (G[j, 4] ** 2))  # pe mode 1 contribution to tke at depths z
        pe_m5_z[j, pp] = 0.5 * ((AG[5, pp] ** 2) * N2[j] * (G[j, 5] ** 2))  # pe mode 1 contribution to tke at depths z

        # # loop over first few modes
        # for mn in range(6):
        #     dg_mode_ke_z_frac[pp, j, mn] = 0.5 * (AGz[mn, pp] ** 2) * (Gz[j, mn] ** 2)
        # for mn in range(1, 7):
        #     dg_mode_pe_z_frac[pp, j, mn] = 0.5 * (AG[mn, pp] ** 2) * N2[j] * (G[j, mn] ** 2)

f, ax = plt.subplots(5, 1, sharex=True)
dps = [0, 40, 65, 115, 165]
colo = ['r', 'g', 'b', 'k', 'c']
ppe = ed_in_2 - 1
count = 0
for i in dps:
    for pp in range(60):
        ax[count].plot(np.array([0, 1, 2, 3, 4]),
                       np.array([tke_m0_z[i, pp], tke_m1_z[i, pp], tke_m2_z[i, pp], tke_m3_z[i, pp], tke_m4_z[i, pp]]),
                       color='b', linewidth=0.5)
        if pp > 58:
            ax[count].plot(np.array([0, 1, 2, 3, 4]),
                           np.array([tke_m0_z[i, ppe], tke_m1_z[i, ppe],
                                     tke_m2_z[i, ppe], tke_m3_z[i, ppe], tke_m4_z[i, ppe]]),
                            color='r', linewidth=1.75)
    if count < 2:
        ax[count].set_ylim([0, 0.02])
    else:
        ax[count].set_ylim([0, 0.005])
    ax[count].set_ylabel(r'KE [m$^2$/s$^2$]')
    ax[count].grid()
    ax[count].set_title('KE at ' + str(grid[i]) + 'm', fontsize='10')
    count = count + 1
ax[count - 1].set_xlabel('Mode Number')
ax[count - 1].grid()
plot_pro(ax[count - 1])

f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
colors = ['#00BFFF', '#F4A460', '#00FF7F', '#FA8072', '#708090']
# background ke
ax0.fill_betweenx(grid, 0, np.nanmean(tke_m0_z / tke_tot_z, axis=1),
                  label='Mode 0', color=colors[0])
ax0.fill_betweenx(grid, np.nanmean(tke_m0_z / tke_tot_z, axis=1),
                  np.nanmean((tke_m0_z + tke_m1_z) / tke_tot_z, axis=1),
                  label='Mode 1', color=colors[1])
ax0.fill_betweenx(grid, np.nanmean((tke_m0_z + tke_m1_z) / tke_tot_z, axis=1),
                  np.nanmean((tke_m0_z + tke_m1_z + tke_m2_z) / tke_tot_z, axis=1),
                  label='Mode 1', color=colors[2])
ax0.fill_betweenx(grid, np.nanmean((tke_m0_z + tke_m1_z + tke_m2_z) / tke_tot_z, axis=1),
                  np.nanmean((tke_m0_z + tke_m1_z + tke_m2_z + tke_m3_z) / tke_tot_z, axis=1),
                  label='Mode 1', color=colors[3])
ax0.fill_betweenx(grid, np.nanmean((tke_m0_z + tke_m1_z + tke_m2_z + tke_m3_z) / tke_tot_z, axis=1),
                  np.nanmean((tke_m0_z + tke_m1_z + tke_m2_z + tke_m3_z + tke_m4_z) / tke_tot_z, axis=1),
                  label='Mode 1', color=colors[4])
# background pe
# ax2.fill_betweenx(grid, 0, np.nanmean(pe_m1_z / pe_tot_z, axis=1),
#                   label='Mode 0', color=colors[1])
# ax2.fill_betweenx(grid, np.nanmean(pe_m1_z / pe_tot_z, axis=1),
#                   np.nanmean((pe_m1_z + pe_m2_z) / pe_tot_z, axis=1),
#                   label='Mode 1', color=colors[2])
# ax2.fill_betweenx(grid, np.nanmean((pe_m1_z + pe_m2_z) / pe_tot_z, axis=1),
#                   np.nanmean((pe_m1_z + pe_m2_z + pe_m3_z) / pe_tot_z, axis=1),
#                   label='Mode 1', color=colors[3])
# ax2.fill_betweenx(grid, np.nanmean((pe_m1_z + pe_m2_z + pe_m3_z) / pe_tot_z, axis=1),
#                   np.nanmean((pe_m1_z + pe_m2_z + pe_m3_z + pe_m4_z) / pe_tot_z, axis=1),
#                   label='Mode 1', color=colors[4])
# eddy ke
ax1.fill_betweenx(grid, 0, tke_m0_z[:, ed_in_2][:, 0] / tke_tot_z[:, ed_in_2][:, 0], label='Mode 0', color=colors[0])
ax1.fill_betweenx(grid, tke_m0_z[:, ed_in_2][:, 0] / tke_tot_z[:, ed_in_2][:, 0],
                  (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0]) / tke_tot_z[:, ed_in_2][:, 0],
                  label='Mode 1', color=colors[1])
ax1.fill_betweenx(grid, (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0]) / tke_tot_z[:, ed_in_2][:, 0],
                  (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0] + tke_m2_z[:, ed_in_2][:, 0]) /
                  tke_tot_z[:, ed_in_2][:, 0], label='Mode 2', color=colors[2])
ax1.fill_betweenx(grid, (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0] + tke_m2_z[:, ed_in_2][:, 0]) /
                  tke_tot_z[:, ed_in_2][:, 0],
                  (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0] + tke_m2_z[:, ed_in_2][:, 0] +
                   tke_m3_z[:, ed_in_2][:, 0]) / tke_tot_z[:, ed_in_2][:, 0], label='Mode 3', color=colors[3])
ax1.fill_betweenx(grid, (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0] + tke_m2_z[:, ed_in_2][:, 0] +
                  tke_m3_z[:, ed_in_2][:, 0]) / tke_tot_z[:, ed_in_2][:, 0],
                  (tke_m0_z[:, ed_in_2][:, 0] + tke_m1_z[:, ed_in_2][:, 0] + tke_m2_z[:, ed_in_2][:, 0] +
                  tke_m3_z[:, ed_in_2][:, 0] + tke_m4_z[:, ed_in_2][:, 0]) /
                  tke_tot_z[:, ed_in_2][:, 0], label='Mode 4', color=colors[4])
ax0.set_xlim([0, 1])
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=12)
ax0.set_title('Mean KE Partition')
ax1.set_title('Eddy KE Partition')
ax0.set_xlabel('Fraction')
ax1.set_xlabel('Fraction')
ax0.set_ylabel('Depth [m]')
# ax2.set_title('Mean PE Partition')
# ax3.set_title('Eddy PE Partition')
ax0.invert_yaxis()
ax0.grid()
# ax1.grid()
# ax2.grid()
plot_pro(ax1)

# --- AVERAGE ENERGY
avg_PE = np.nanmean(PE_per_mass, 1)
good_prof_i = good_ke_prof  # np.where(good_prof > 0)
avg_KE = np.nanmean(HKE_per_mass[:, np.where(good_ke_prof > 0)[0]], 1)
# --- eddy kinetic and potential energy
PE_ed = np.nanmean(PE_per_mass[:, ed_in[0]:ed_in[-1]], axis=1)
KE_ed = np.nanmean(HKE_per_mass[:, ed_in[0]:ed_in[-1]], axis=1)

# --- ENERGY parameters
f_ref = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)
dk = f_ref / c[1]
sc_x = 1000 * f_ref / c[1:]
vert_wavenumber = f_ref / c[1:]
dk_ke = 1000 * f_ref / c[1]
k_h = 1e3 * (f_ref / c[1:]) * np.sqrt(avg_KE[1:] / avg_PE[1:])
PE_SD, PE_GM = PE_Tide_GM(rho0, grid, nmodes, np.transpose(np.atleast_2d(N2)), f_ref)

# --- CURVE FITTING TO FIND BREAK IN SLOPES
xx = sc_x
yy = avg_PE[1:] / dk
yy2 = avg_KE[1:] / dk
# export to use findchangepts in matlab 
# np.savetxt('test_line_fit_x',xx)
# np.savetxt('test_line_fit_y',yy)
# index 11 is the point where the break in slope occurs
ipoint = 8  # 8  # 11
# fit slopes to PE and KE spectra
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

# --- cascade rates
vert_wave = sc_x / 1000
alpha = 10
ak0 = xx[ipoint] / 1000
E0 = np.mean(yy[ipoint - 3:ipoint + 4])
ak = vert_wave / ak0
one = E0 * ((ak ** (5 * alpha / 3)) * (1 + ak ** (4 * alpha / 3))) ** (-1 / alpha)
# -  enstrophy/energy transfers
mu = 1.88e-3 / (1 + 0.03222 * theta_avg + 0.002377 * theta_avg * theta_avg)
nu = mu / gsw.rho(salin_avg, ct_avg, grid_p)
avg_nu = np.nanmean(nu)
enst_xfer = (E0 * ak0 ** 3) ** (3 / 2)
ener_xfer = (E0 * ak0 ** (5 / 3)) ** (3 / 2)
enst_diss = np.sqrt(avg_nu) / (enst_xfer ** (1 / 6))

# --- LOAD in other data
# load in Station BATs PE Comparison
pkl_file = open('/Users/jake/Desktop/bats/station_bats_pe_apr11.pkl', 'rb')
SB = pickle.load(pkl_file)
pkl_file.close()
sta_bats_pe = SB['PE']
sta_bats_c = SB['c']
sta_bats_f = np.pi * np.sin(np.deg2rad(31.6)) / (12 * 1800)
sta_bats_dk = sta_bats_f / sta_bats_c[1]
# load in HKE estimates from Obj. Map 
pkl_file = open('/Users/jake/Documents/geostrophic_turbulence/BATS_OM_KE.pkl', 'rb')
bats_map = pickle.load(pkl_file)
pkl_file.close()
sx_c_om = bats_map['sc_x']
ke_om_u = bats_map['avg_ke_u']
ke_om_v = bats_map['avg_ke_v']
dk_om = bats_map['dk']
# load in Station HOTS PE Comparison
SH = si.loadmat('/Users/jake/Desktop/bats/station_hots_pe.mat')
sta_hots_pe = SH['out']['PE'][0][0]
sta_hots_c = SH['out'][0][0][3]
sta_hots_f = SH['out'][0][0][2]
sta_hots_dk = SH['out']['dk'][0][0]
# LOAD ABACO
pkl_file = open('/Users/jake/Desktop/abaco/abaco_outputs_2.pkl', 'rb')
abaco_energies = pickle.load(pkl_file)
pkl_file.close()

plot_eng = 0
if plot_eng > 0:
    fig0, ax0 = plt.subplots()
    # PE_p = ax0.plot(sc_x, avg_PE[1:] / dk, color='#B22222', label='APE$_{DG}$', linewidth=3)
    # ax0.scatter(sc_x, avg_PE[1:] / dk, color='#B22222', s=20)  # DG PE
    # PE_sta_p = ax0.plot(1000 * sta_bats_f / sta_bats_c[1:], np.nanmean(sta_bats_pe[1:], axis=1) / sta_bats_dk,
    #                     color='#FF8C00',
    #                     label='APE$_{ship}$', linewidth=1.5)

    KE_p = ax0.plot(1000 * f_ref / c[1:], avg_KE[1:] / dk, 'g', label='KE$_{DG}$', linewidth=3)
    ax0.scatter(sc_x, avg_KE[1:] / dk, color='g', s=20)  # DG KE
    KE_p = ax0.plot([10**-2, 1000 * f_ref / c[1]], avg_KE[0:2] / dk, 'g', linewidth=3) # DG KE_0
    ax0.scatter(10**-2, avg_KE[0] / dk, color='g', s=25, facecolors='none')  # DG KE_0

    # -- Obj. Map
    # KE_om_u = ax0.plot(sx_c_om,ke_om_u[1:]/dk_om,'b',label='$KE_u$',linewidth=1.5)
    # ax0.scatter(sx_c_om,ke_om_u[1:]/dk_om,color='b',s=10) # DG KE
    # KE_om_u = ax0.plot(sx_c_om,ke_om_v[1:]/dk_om,'c',label='$KE_v$',linewidth=1.5)
    # ax0.scatter(sx_c_om,ke_om_v[1:]/dk_om,color='c',s=10) # DG KE

    # -- Eddy energies
    # PE_e = ax0.plot(sc_x, PE_ed[1:] / dk, color='c', label='eddy PE', linewidth=2)
    # KE_e = ax0.plot(sc_x, KE_ed[1:] / dk, color='y', label='eddy KE', linewidth=2)
    # KE_e = ax0.plot([10**-2, 1000 * f_ref / c[1]], KE_ed[0:2] / dk, color='y', label='eddy KE', linewidth=2)

    # -- Slope fits
    # ax0.plot(10 ** x_53, 10 ** y_g_53, color='k', linewidth=1, linestyle='--')
    # ax0.plot(10 ** x_3, 10 ** y_g_3, color='k', linewidth=1, linestyle='--')
    ax0.plot(10 ** x_3_2, 10 ** y_g_ke, color='k', linewidth=1.5, linestyle='--')
    # ax0.text(10 ** x_53[0] - .012, 10 ** y_g_53[0], str(float("{0:.2f}".format(slope1[0]))), fontsize=10)
    # ax0.text(10 ** x_3[0] + .085, 10 ** y_g_3[0], str(float("{0:.2f}".format(slope2[0]))), fontsize=10)
    ax0.text(10 ** x_3_2[3] + .05, 10 ** y_g_ke[3], str(float("{0:.2f}".format(slope_ke[0]))), fontsize=12)

    # ax0.scatter(vert_wave[ipoint] * 1000, one[ipoint], color='b', s=7)
    # ax0.plot([xx[ipoint], xx[ipoint]], [10 ** (-4), 4 * 10 ** (-4)], color='k', linewidth=2)
    # ax0.text(xx[ipoint + 1], 2 * 10 ** (-4),
    #          str('Break at ') + str(float("{0:.1f}".format(1 / xx[ipoint]))) + 'km')

    # -- Rossby Radii
    ax0.plot([sc_x[0], sc_x[0]], [10 ** (-4), 4 * 10 ** (-4)], color='k', linewidth=2)
    ax0.text(sc_x[0] - .6 * 10 ** -2, 7 * 10 ** (-4),
             str(r'$c_1/f$ = ') + str(float("{0:.1f}".format(1 / sc_x[0]))) + 'km', fontsize=12)
    ax0.plot([sc_x[4], sc_x[4]], [10 ** (-4), 4 * 10 ** (-4)], color='k', linewidth=2)
    ax0.text(sc_x[4] - .4 * 10 ** -2, 7 * 10 ** (-4),
             str(r'$c_5/f$ = ') + str(float("{0:.1f}".format(1 / sc_x[4]))) + 'km', fontsize=12)

    # GM
    # ax0.plot(sc_x, PE_GM / dk, linestyle='--', color='#B22222', linewidth=0.75)
    # ax0.text(sc_x[0] - .01, PE_GM[1] / dk, r'$PE_{GM}$', fontsize=12)
    # ax0.plot(np.array([10**-2, 10]), [PE_SD / dk, PE_SD / dk], linestyle='--', color='k', linewidth=0.75)

    # Limits/scales
    # ax0.plot( [3*10**-1, 3*10**0], [1.5*10**1, 1.5*10**-2],color='k',linewidth=0.75)
    # ax0.plot([3*10**-2, 3*10**-1],[7*10**2, ((5/3)*(np.log10(2*10**-1) - np.log10(2*10**-2) ) +
    #        np.log10(7*10**2) )] ,color='k',linewidth=0.75)
    # ax0.text(3.3*10**-1,1.3*10**1,'-3',fontsize=10)
    # ax0.text(3.3*10**-2,6*10**2,'-5/3',fontsize=10)
    # ax0.plot( [1000*f_ref/c[1], 1000*f_ref/c[-2]],[1000*f_ref/c[1], 1000*f_ref/c[-2]],
    #        linestyle='--',color='k',linewidth=0.8)
    # ax0.text( 1000*f_ref/c[-2]+.1, 1000*f_ref/c[-2], r'f/c$_m$',fontsize=10)
    # ax0.plot(sc_x,k_h,color='k',linewidth=.9,label=r'$k_h$')
    # ax0.text(sc_x[0]-.008,k_h[0]-.011,r'$k_{h}$ [km$^{-1}$]',fontsize=10)

    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.axis([10 ** -2, 10 ** 1, 10 ** (-4), 2 * 10 ** 3])
    ax0.axis([10 ** -2, 10 ** 1, 10 ** (-4), 10 ** 3])
    ax0.set_xlabel(r'Scaled Vertical Wavenumber = (Rossby Radius)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=18)
    ax0.set_ylabel('Spectral Density', fontsize=18)  # ' (and Hor. Wavenumber)')
    ax0.set_title('Energy Spectrum', fontsize=20)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, fontsize=12)
    # ax0.legend([handles[0], handles[1], handles[2], handles[3]], [labels[0], labels[1], labels[2], labels[3]])
    plt.tight_layout()
    plot_pro(ax0)
    # fig0.savefig('/Users/jake/Desktop/bats/dg035_15_PE_b.png',dpi = 300)
    # plt.close()
    # plt.show()

    # additional plot to highlight the ratio of KE to APE to predict the scale of motion
    fig0, ax0 = plt.subplots()
    # Limits/scales
    ax0.plot([1000 * f_ref / c[1], 1000 * f_ref / c[-2]], [1000 * f_ref / c[1], 1000 * f_ref / c[-2]], linestyle='--',
             color='k', linewidth=1.5, zorder=2, label=r'$L_n^{-1}$')
    ax0.text(1000 * f_ref / c[-2] + .1, 1000 * f_ref / c[-2], r'f/c$_n$', fontsize=14)
    ax0.plot(sc_x, k_h, color='k', label=r'$k_h$', linewidth=1.5)
    # ax0.text(sc_x[0]-.008,k_h[0]-.011,r'$k_{h}$ [km$^{-1}$]',fontsize=14)
    xx_fill = 1000 * f_ref / c[1:]
    yy_fill = 1000 * f_ref / c[1:]
    # ax0.fill_between(xx_fill, yy_fill, k_h, color='b',interpolate=True)
    ax0.fill_between(xx_fill, yy_fill, k_h, where=yy_fill >= k_h, facecolor='#FAEBD7', interpolate=True, alpha=0.75)
    ax0.fill_between(xx_fill, yy_fill, k_h, where=yy_fill <= k_h, facecolor='#6B8E23', interpolate=True, alpha=0.75)

    ax0.set_yscale('log')
    ax0.set_xscale('log')
    # ax0.axis([10**-2, 10**1, 3*10**(-4), 2*10**(3)])
    ax0.axis([10 ** -2, 10 ** 1, 10 ** (-3), 10 ** 3])
    ax0.set_title('Predicted Horizontal Length Scale', fontsize=18)
    ax0.set_xlabel(r'Inverse Deformation Radius ($L_n$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=18)
    ax0.set_ylabel(r'Horizontal Wavenumber [$km^{-1}$]', fontsize=18)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend([handles[0], handles[1]], [labels[0], labels[1]], fontsize=14)
    ax0.set_aspect('equal')
    plt.tight_layout()
    plot_pro(ax0)

# --- SAVE BATS ENERGIES DG TRANSECTS
# write python dict to a file
sa = 1
if sa > 0:
    my_dict = {'depth': grid, 'KE': avg_KE, 'PE': avg_PE, 'c': c, 'f': f_ref}
    output = open('/Users/jake/Documents/geostrophic_turbulence/BATs_DG_2015_energy.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()
# --------------------------------------------------------------------------------------------------


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
                     label=r'APE ABACO$_{DG}$', linewidth=2, color='r')
    KE_ab = ax0.plot(mode_num, abaco_energies['avg_KE'][1:] / (abaco_energies['f_ref'] / abaco_energies['c'][1]),
                     label=r'KE ABACO$_{DG}$', linewidth=2, color='g')
    # PE_sta_hots = ax0.plot(mode_num,sta_hots_pe[1:]/sta_hots_dk,label=r'HOTS$_{ship}$',linewidth=2)
    PE_p2 = ax0.plot(mode_num, 4 * avg_PE[1:] / dk, label=r'APE BATS$_{DG}$', color='#F08080', linewidth=1)
    KE_p2 = ax0.plot(mode_num, avg_KE[1:] / dk, 'g', label=r'KE BATS$_{DG}$', color='#90EE90', linewidth=1)
    ax0.set_xlabel('Mode Number', fontsize=16)
    ax0.set_ylabel('Spectral Density', fontsize=16)
    ax0.set_title('ABACO / BATS Comparison', fontsize=18)
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    # ax0.axis([8*10**-1, 10**2, 3*10**(-4), 2*10**(3)])
    ax0.axis([8 * 10 ** -1, 10 ** 2, 3 * 10 ** (-4), 10 ** 3])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, fontsize=12)
    plot_pro(ax0)

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
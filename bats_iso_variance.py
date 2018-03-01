# BATS
# take velocity and displacement profiles and compute energy spectra / explore mode amplitude variability
import numpy as np
import matplotlib.pyplot as plt
import seawater as sw
import pandas as pd
import scipy.io as si
from netCDF4 import Dataset
import pickle
from scipy.signal import savgol_filter
# functions I've written 
from mode_decompositions import vertical_modes, PE_Tide_GM
from toolkit import plot_pro

# physical parameters 
g = 9.81
rho0 = 1027
# limit bin depth to 4500 (to prevent fitting of velocity profiles past points at which we have data) (previous = 5000)
bin_depth = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4500, 20)])
ref_lat = 31.8
ref_lon = -64.2
lat_in = 31.7
lon_in = 64.2
grid = bin_depth[1:-1]
grid_p = sw.pres(grid, lat_in)
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
GD = Dataset('BATs_2015_gridded_3.nc', 'r')
df_den = pd.DataFrame(GD['Density'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
df_theta = pd.DataFrame(GD['Theta'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
df_s = pd.DataFrame(GD['Salinity'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
df_lon = pd.DataFrame(GD['Longitude'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
df_lat = pd.DataFrame(GD['Latitude'][0:sz_g, :], index=GD['grid'][0:sz_g], columns=GD['dive_list'][:])
dac_u = GD.variables['DAC_u'][:]
dac_v = GD.variables['DAC_v'][:]
time_rec = GD.variables['time_start_stop'][:]
time_rec_all = GD.variables['time_start_stop'][:]
profile_list = np.float64(GD.variables['dive_list'][:]) - 35000
df_den[df_den < 0] = np.nan
df_theta[df_theta < 0] = np.nan
df_s[df_s < 0] = np.nan
df_lon[df_lon < -500] = np.nan
df_lat[df_lat < -500] = np.nan
dac_u[dac_u < -500] = np.nan
dac_v[dac_v < -500] = np.nan

# ---- LOAD IN TRANSECT TO PROFILE DATA COMPILED IN BATS_TRANSECTS.PY
pkl_file = open('/Users/jake/Desktop/bats/transect_profiles_jan18.pkl', 'rb')
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
# Heading = bats_trans['Heading']

# average background properties of profiles along these transects 
sigma_theta_avg = df_den.mean(axis=1)
theta_avg = df_theta.mean(axis=1)
salin_avg = df_s.mean(axis=1)
ddz_avg_sigma = np.gradient(sigma_theta_avg, z)
ddz_avg_theta = np.gradient(theta_avg, z)
N2 = np.nan * np.zeros(sigma_theta_avg.size)
N2[1:] = np.squeeze(sw.bfrq(salin_avg, theta_avg, grid_p, lat=ref_lat)[0])
lz = np.where(N2 < 0)
lnan = np.isnan(N2)
N2[lz] = 0
N2[lnan] = 0
N = np.sqrt(N2)

# computer vertical mode shapes 
G, Gz, c = vertical_modes(N2, grid, omega, mmax)

# presort V 
good_v = np.zeros(np.size(Time))
for i in range(np.size(Time)):
    v_dz = np.gradient(V[:, i])
    if np.nanmax(np.abs(v_dz)) < 0.075:
        good_v[i] = 1
good0 = np.intersect1d(np.where((np.abs(V[-45, :]) < 0.2))[0], np.where((np.abs(V[10, :]) < 0.4))[0])
good = np.intersect1d(np.where(good_v > 0), good0)
V2 = V[:, good]
Eta2 = Eta[:, good]
Eta_theta2 = Eta_theta[:, good]
Time2 = Time[good]
prof_lon2 = prof_lon[good]
prof_lat2 = prof_lat[good]

sz = np.shape(Eta2)
num_profs = sz[1]
eta_fit_depth_min = 50
eta_fit_depth_max = 3800
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
good_prof = np.ones(num_profs)
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
        HKE_per_mass[:, i] = AGz[:, i] * AGz[:, i]
        ival = np.where(HKE_per_mass[modest, i] >= HKE_noise_threshold)
        if np.size(ival) > 0:
            good_prof[i] = 0  # flag profile as noisy
    else:
        good_prof[i] = 0  # flag empty profile as noisy as well

    # fit to eta profiles
    this_eta = Eta2[:, i].copy()
    # obtain matrix of NEta
    Neta[:, i] = N * this_eta
    this_eta_theta = Eta_theta2[:, i].copy()
    iw = np.where((grid >= eta_fit_depth_min) & (grid <= eta_fit_depth_max))
    iw_theta = np.where((grid >= eta_fit_depth_min) & (grid <= eta_theta_fit_depth_max))
    if iw[0].size > 1:
        eta_fs = Eta2[:, i].copy()  # ETA
        eta_theta_fs = Eta_theta2[:, i].copy()  # ETA THETA

        i_sh = np.where((grid < eta_fit_depth_min))
        eta_fs[i_sh[0]] = grid[i_sh] * this_eta[iw[0][0]] / grid[iw[0][0]]
        eta_theta_fs[i_sh[0]] = grid[i_sh] * this_eta_theta[iw[0][0]] / grid[iw[0][0]]

        i_dp = np.where((grid > eta_fit_depth_max))
        eta_fs[i_dp[0]] = (grid[i_dp] - grid[-1]) * this_eta[iw[0][-1]] / (grid[iw[0][-1]] - grid[-1])

        i_dp_theta = np.where((grid > eta_theta_fit_depth_max))
        eta_theta_fs[i_dp_theta[0]] = (grid[i_dp_theta] - grid[-1]) * this_eta_theta[iw_theta[0][-1]] / (
                grid[iw_theta[0][-1]] - grid[-1])

        AG[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], np.transpose(np.atleast_2d(eta_fs)))[0])
        AG_theta[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], np.transpose(np.atleast_2d(eta_theta_fs)))[0])
        Eta_m[:, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG[:, i])))
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

# ---- COMPUTE EOF SHAPES AND COMPARE TO ASSUMED STRUCTURE
# --- find EOFs of dynamic horizontal current (V mode amplitudes)
AGzq = AGz  # (:,quiet_prof)
nq = np.size(good_prof)  # good_prof and dg_good_prof
avg_AGzq = np.nanmean(np.transpose(AGzq), axis=0)
AGzqa = AGzq - np.transpose(np.tile(avg_AGzq, [nq, 1]))  # mode amplitude anomaly matrix
cov_AGzqa = (1 / nq) * np.matrix(AGzqa) * np.matrix(np.transpose(
    AGzqa))  # nmodes X nmodes covariance matrix (squared mode amplitude anomaly / number of profiles) = covariance
var_AGzqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa)))) * np.matrix(
    np.sqrt(np.diag(cov_AGzqa)))  # sqrt(cov)*sqrt(cov) = varaince
cor_AGzqa = cov_AGzqa / var_AGzqa  # nmodes X nmodes correlation matrix (cov/var) = correlation
# (look at how mode amplitude anomalies are correlated) =>
# to look at shape of eigenfunctions of the correlation matrix
# (project the shape of the eigenfunctions onto the vertical structure G, Gz )

D_AGzqa, V_AGzqa = np.linalg.eig(cov_AGzqa)  # columns of V_AGzqa are eigenvectors

EOFseries = np.transpose(V_AGzqa) * np.matrix(AGzqa)  # EOF "timeseries' [nmodes X nq]
EOFshape = np.matrix(Gz) * V_AGzqa  # depth shape of eigenfunctions [ndepths X nmodes]
EOFshape1_BTpBC1 = np.matrix(Gz[:, 0:2]) * V_AGzqa[0:2, 0]  # truncated 2 mode shape of EOF#1
EOFshape2_BTpBC1 = np.matrix(Gz[:, 0:2]) * V_AGzqa[0:2, 1]  # truncated 2 mode shape of EOF#2

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

# --- PLOT ETA / EOF
plot_eta = 0
if plot_eta > 0:
    f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
    for j in range(num_profs):
        ax1.plot(Eta2[:, j], grid, color='#4682B4', linewidth=1.25)
        ax1.plot(Eta_m[:, j], grid, color='k', linestyle='--', linewidth=.75)
        ax0.plot(V2[:, j], grid, color='#4682B4', linewidth=1.25)
        ax0.plot(V_m[:, j], grid, color='k', linestyle='--', linewidth=.75)
    ax1.axis([-400, 400, 0, 4750])
    ax0.text(190, 800, str(num_profs) + ' Profiles')
    ax1.set_xlabel(r'Vertical Isopycnal Displacement, $\xi_{\sigma_{\theta}}$ [m]', fontsize=14)
    ax1.set_title(r'Isopycnal Displacement', fontsize=18)  # + '(' + str(Time[0]) + '-' )
    ax0.axis([-.4, .4, 0, 4750])
    ax0.set_title("Geostrophic Velocity", fontsize=18)  # (" + str(num_profs) + 'profiles)' )
    ax0.set_ylabel('Depth [m]', fontsize=16)
    ax0.set_xlabel('Cross-Track Velocity, U [m/s]', fontsize=14)
    ax0.invert_yaxis()
    ax0.grid()
    plot_pro(ax1)
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
        p_eof_eta = ax2.plot(-EOFetashape[:, ii] / np.max(grid), grid, color=colors[ii, :],
                             label='EOF # = ' + str(ii + 1), linewidth=2.5)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=10)
    ax2.axis([-.7, .7, 0, 4750])
    ax2.set_title('EOF Displacement Mode Shapes', fontsize=18)
    ax2.set_xlabel('Normalized Mode Amp.', fontsize=14)
    ax2.invert_yaxis()
    plot_pro(ax2)

# --- MODE AMPLITUDE IN TIME AND SPACE
plot_mode = 1
if plot_mode > 0:

    window_size, poly_order = 11, 2
    fm, (ax, ax1) = plt.subplots(1, 2, sharey=True)
    colors = ['r', 'b', 'm', 'g', 'c']
    for mo in range(1, 6, 1):
        orderAG = AG[mo, np.argsort(Time2)]
        orderAGz = AGz[mo, np.argsort(Time2)]
        y_sg = savgol_filter(orderAG, window_size, poly_order)
        y_sgz = savgol_filter(orderAGz, window_size, poly_order)
        pm = ax.scatter(Time2, AG[mo, :], s=3, color=colors[mo-1], label=('Mode '+str(mo)))
        ax.plot(Time2[np.argsort(Time2)], y_sg, color=colors[mo-1])
        pmz = ax1.scatter(Time2, AGz[mo, :], s=3, color=colors[mo-1])
        ax1.plot(Time2[np.argsort(Time2)], y_sgz, color=colors[mo-1])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=14)
    ax.set_title('Displacement Mode Amplitude')
    ax1.set_title('Velocity Mode Amplitude')
    ax.grid()
    plot_pro(ax1)

plot_mode_corr = 1
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
    f, (ax1,ax2) = plt.subplots(1,2)
    pa = ax1.pcolor(dist_win, t_win, corr_i, vmin=-1, vmax=1)
    paz = ax2.pcolor(dist_win, t_win, corr_z_i, vmin=-1, vmax=1)
    ax1.set_xlabel('Spatial Separation [km]')
    ax2.set_xlabel('Spatial Separation [km]')
    ax1.set_ylabel('Time Lag [days]')
    ax1.set_title('Displacement Mode Amplitude')
    ax2.set_title('Velocity Mode Amplitude')
    f.colorbar(pa)
    plot_pro(ax2)

avg_PE = np.nanmean(PE_per_mass, 1)
good_prof_i = good_prof  # np.where(good_prof > 0)
avg_KE = np.nanmean(HKE_per_mass[:, np.where(good_prof > 0)[0]], 1)
# PLOT each KE profile
# fig0, ax0 = plt.subplots()
# for i in range(np.size(good_prof)):
#     if good_prof[i] > 0:
#         ax0.plot(np.arange(0,61,1),HKE_per_mass[:,i])
# ax0.set_xscale('log')      
# ax0.set_yscale('log')    
# plot_pro(ax0)    
# avg_PE_theta = np.nanmean(PE_theta_per_mass,1)

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
# export to use findchangepts in matlab 
# np.savetxt('test_line_fit_x',xx)
# np.savetxt('test_line_fit_y',yy)
# index 11 is the point where the break in slope occurs
ipoint = 8  # 11
x_53 = np.log10(xx[0:ipoint])
y_53 = np.log10(yy[0:ipoint])
slope1 = np.polyfit(x_53, y_53, 1)
x_3 = np.log10(xx[ipoint:])
y_3 = np.log10(yy[ipoint:])
slope2 = np.polyfit(x_3, y_3, 1)
y_g_53 = np.polyval(slope1, x_53)
y_g_3 = np.polyval(slope2, x_3)

vert_wave = sc_x / 1000
alpha = 10
ak0 = xx[ipoint] / 1000
E0 = np.mean(yy[ipoint - 3:ipoint + 4])
ak = vert_wave / ak0
one = E0 * ((ak ** (5 * alpha / 3)) * (1 + ak ** (4 * alpha / 3))) ** (-1 / alpha)
# -  enstrophy/energy transfers
mu = 1.88e-3 / (1 + 0.03222 * theta_avg + 0.002377 * theta_avg * theta_avg)
nu = mu / sw.dens(salin_avg, theta_avg, grid_p)
avg_nu = np.nanmean(nu)
enst_xfer = (E0 * ak0 ** 3) ** (3 / 2)
ener_xfer = (E0 * ak0 ** (5 / 3)) ** (3 / 2)
enst_diss = np.sqrt(avg_nu) / (enst_xfer ** (1 / 6))

# --- LOAD in other data
# load in Station BATs PE Comparison
SB = si.loadmat('/Users/jake/Desktop/bats/station_bats_pe.mat')
sta_bats_pe = SB['out'][0][0][0]
sta_bats_c = SB['out'][0][0][3]
sta_bats_f = SB['out'][0][0][2]
sta_bats_dk = SB['out'][0][0][1]
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
    PE_p = ax0.plot(sc_x, avg_PE[1:] / dk, color='#B22222', label='APE', linewidth=2)
    # PE_sta_p = ax0.plot((1000) * sta_bats_f / sta_bats_c[1:], sta_bats_pe[1:] / sta_bats_dk, color='#FF8C00',
    #                     label='$PE_{ship}$', linewidth=1.5)
    KE_p = ax0.plot(sc_x, avg_KE[1:] / dk, 'g', label='KE', linewidth=2)
    ax0.scatter(sc_x, avg_PE[1:] / dk, color='#B22222', s=12)  # DG PE
    # ax0.scatter((1000)*sta_bats_f/sta_bats_c[1:],sta_bats_pe[1:]/sta_bats_dk,color='#FF8C00',s=10) # BATS PE
    ax0.scatter(sc_x, avg_KE[1:] / dk, color='g', s=12)  # DG KE

    # Obj. Map
    # KE_om_u = ax0.plot(sx_c_om,ke_om_u[1:]/dk_om,'b',label='$KE_u$',linewidth=1.5)
    # ax0.scatter(sx_c_om,ke_om_u[1:]/dk_om,color='b',s=10) # DG KE
    # KE_om_u = ax0.plot(sx_c_om,ke_om_v[1:]/dk_om,'c',label='$KE_v$',linewidth=1.5)
    # ax0.scatter(sx_c_om,ke_om_v[1:]/dk_om,color='c',s=10) # DG KE

    # Slope fits
    ax0.plot(vert_wave * 1000, one, color='b', linewidth=1, label=r'APE$_{fit}$')
    ax0.plot(10 ** x_53, 10 ** y_g_53, color='k', linewidth=1, linestyle='--')
    ax0.plot(10 ** x_3, 10 ** y_g_3, color='k', linewidth=1, linestyle='--')
    ax0.text(10 ** x_53[0] - .01, 10 ** y_g_53[0], str(float("{0:.2f}".format(slope1[0]))), fontsize=10)
    ax0.text(10 ** x_3[0] + .065, 10 ** y_g_3[0], str(float("{0:.2f}".format(slope2[0]))), fontsize=10)
    ax0.scatter(vert_wave[ipoint] * 1000, one[ipoint], color='b', s=7)
    ax0.plot([xx[ipoint], xx[ipoint]], [10 ** (-4), 4 * 10 ** (-4)], color='k', linewidth=2)
    ax0.text(xx[ipoint + 1], 2 * 10 ** (-4),
             str('Slope break at ') + str(float("{0:.1f}".format(1 / xx[ipoint]))) + 'km')

    # Rossby Radii
    ax0.plot([sc_x[0], sc_x[0]], [10 ** (-4), 4 * 10 ** (-4)], color='k', linewidth=2)
    ax0.text(sc_x[0] + .2 * 10 ** -2, 4 * 10 ** (-4),
             str(r'$c_1/f$ = ') + str(float("{0:.1f}".format(1 / sc_x[0]))) + 'km')

    # GM
    ax0.plot(sc_x, 0.25 * PE_GM / dk, linestyle='--', color='#B22222', linewidth=0.75)
    ax0.text(sc_x[0] - .01, .25 * PE_GM[1] / dk, r'$\frac{1}{4}PE_{GM}$', fontsize=14)

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
    ax0.legend([handles[0], handles[1], handles[2]], [labels[0], labels[1], labels[2]], fontsize=14)
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
    PE_sta_p0 = ax0.plot(1000 * sta_bats_f / sta_bats_c[1:], sta_bats_pe[1:] / sta_bats_dk, color='#FF8C00',
                         label=r'$PE_{BATS_{ship}}$')
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
    PE_sta_p1 = ax0.plot(mode_num, sta_bats_pe[1:] / sta_bats_dk, label=r'BATS$_{ship}$', linewidth=2)
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

################################
# try 3
# from scipy.optimize import curve_fit
# xx1 = np.log10(sc_x)
# yy1 = np.log10(avg_PE[1:]/dk)
# xx1 = np.log10(np.array([2*10**-2, 7*10**-2, 10**-1, 1.5*10**-1, 2.5*10**-1, 4*10**-1]) )
# yy1 = np.log10(np.array([7*10**1, 3*10**1, 10**1, 10**0, 5*10**-2, 10**-3]) )

# def two_lines(x,a,b):
# one = (-.75)*x + b
# two = (-5)*x + d
# return np.maximum(one, two)
# pw0 = (10, .5) # a guess for slope, intercept, slope, intercept
# pw, cov = curve_fit(two_lines, xx1, yy1, pw0, sigma=0.01*np.ones(len(xx1)))
# crossover = 10**( (pw[1] - pw[0]) / ( (-5/3) - (-3)) )

# try 2 
# tck = interpolate.splrep(xx, yy, k=2, s=0)
# fig, axes = plt.subplots()
# axes.plot(xx, yy, 'x', label = 'data')
# axes.plot(xx, interpolate.splev(xx, tck, der=0), label = 'Fit')
# plot_pro(axes)

# try 1
# def piecewise_linear(x, x0, y0, k1, k2):
#     return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
# p , e = curve_fit(piecewise_linear, xx, yy)
# f, ax = plt.subplots()
# ax.plot(xx, yy, "o")
# ax.plot(xx, piecewise_linear(xx, *p))
# plot_pro(ax0)

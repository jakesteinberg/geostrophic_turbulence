# STATION BATS (PE profiles)

import numpy as np
import matplotlib.pyplot as plt
import gsw
import seawater as sw
import pickle
import scipy
from scipy.signal import savgol_filter
import glob
from netCDF4 import Dataset
# functions I've written
from grids import make_bin_gen
from mode_decompositions import vertical_modes, eta_fit, vertical_modes_f
from toolkit import plot_pro, nanseg_interp, find_nearest

# ------ physical parameters
g = 9.81
rho0 = 1027
GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
bin_depth = GD.variables['grid'][:]
# bin_depth = np.concatenate([np.arange(0, 150, 5), np.arange(150, 300, 5), np.arange(300, 4500, 10)])
ref_lat = 31.7
ref_lon = -64.2
grid = bin_depth[0:231]
grid_p = gsw.p_from_z(-1 * grid, ref_lat)
z = -1 * grid

# write this to process any set of shipboard hydrography (specify file type)

# file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Shipboard/BATS_station/CTD_data/b1*ctd.txt')
# file_list2 = file_list[0]
# # - file formatting from BATS site
# info_variables = ['cast ID', 'date deployed', 'date recovered', 'decimal date d', 'decimal date r', 'decimal day d',
#                   'time ctd d', 'time ctd r', 'ctd lat d', 'ctd lat r', 'ctd lon d', 'ctd lat d']
# cast_variables = ['cast ID', 'decimal year', 'lat', 'lon', 'pressure', 'depth', 'temp', 'conductivity', 'salinity',
#                   'DO', 'beam attenuation coef', 'fluoresence']
#
# #  --------- open txt files and place into array --------------
# output = []
# count0 = 0
# for f in file_list:
#     for l in open(f):
#         testi = l.strip().split("\t")
#         if len(testi) < 2:                                  # if tab spacing changes, re-split
#             testi = testi[0].split()
#         if count0 < 1:                                      # deal with first element of storage vs. all others
#             data = np.nan * np.zeros(len(testi))
#             count = 0
#             for i in testi:
#                 data[count] = np.float(i)                   # data = one row's worth of data
#                 count = count + 1
#         else:
#             intermed = np.nan * np.zeros(len(testi))
#             count = 0
#             for i in testi:
#                 intermed[count] = np.float(i)
#                 count = count + 1
#             if data.size > 13:
#                 data = np.concatenate((data, intermed[None, :]), axis=0)
#             else:
#                 data = np.concatenate((data[None, :], intermed[None, :]), axis=0) # data = n x 13 array
#         count0 = count0 + 1                                # loop over each line in each file
#     output.append(data)
#     count0 = 0                                             # reset count0 for next file
# # do stuff
#
# # ----------- take list (where each element is a txt file array) and grid/select deep dives -------------
# cast_t = np.nan * np.zeros((len(grid), 340))
# cast_s = np.nan * np.zeros((len(grid), 340))
# cast_lon = np.nan * np.zeros((1, 340))
# cast_lat = np.nan * np.zeros((1, 340))
# cast_date = np.nan * np.zeros((1, 340))
# cast_log = np.nan * np.zeros((1, 340))
# cast_max_z = np.nan * np.zeros((1, 340))
# count = 0
# for i in range(len(output)):
#     this_c = output[i]
#     if len(this_c) > 13:  # ensure more than one row of data
#         idc = this_c[:, 0]  # cruise type (and all info)
#         c_num = np.floor((idc - 10000000) / 1000)  # cruise number (should be one cruise number per file, just check)
#         for j in np.unique(c_num):
#             all_cast = np.where(c_num == j)[0]  # all casts in cruise number j
#             cast_no = idc[all_cast] - 10000000 - j * 1000  # each cast number for this cruise
#             data_per_cruise_j = this_c[all_cast, :]
#             for k in np.unique(cast_no):
#                 each_cast = np.where(cast_no == k)[0]
#                 cast_data = data_per_cruise_j[each_cast, :]  # data from each cast k
#                 cast_p = cast_data[:, 4]
#                 c_d = -1 * gsw.z_from_p(cast_p, ref_lat)  # sw.dpth(cast_p, ref_lat)
#
#                 lt = cast_data[:, 2]
#                 lo = cast_data[:, 3]
#                 cast_t_pre = cast_data[:, 6]
#                 cast_s_pre = cast_data[:, 8]
#                 cast_t_pre[cast_t_pre < 0] = np.nan
#                 cast_s_pre[cast_s_pre < 20] = np.nan
#                 lt[lt < -400] = np.nan
#                 lo[lo < -400] = np.nan
#                 lt[lt < -400] = np.nan
#                 # only select deep dives w/in lat/lon box
#                 if (c_d.max() > 3500) & (np.sum(np.isnan(cast_s_pre)) < len(cast_s_pre)/3):
#                     g_ol = np.where(grid <= c_d.max())[0]
#                     cast_t[g_ol, count], cast_s[g_ol, count] = make_bin_gen(grid[g_ol], c_d, cast_t_pre, cast_s_pre)
#                     # cast_t[g_overlap, count] = np.interp(grid[g_overlap], cast_d, cast_t_pre)
#                     # cast_s[g_overlap, count] = np.interp(grid[g_overlap], cast_d, cast_s_pre)
#                     cast_lat[0][count] = np.nanmean(cast_data[:, 2])
#                     cast_lon[0][count] = np.nanmean(cast_data[:, 3])
#                     cast_date[0][count] = np.nanmean(cast_data[:, 1])
#                     cast_max_z[0][count] = c_d.max()
#                     cast_log[0][count] = cast_data[0, 0]
#                     count = count + 1
# # --- end looping and cast selection
#
# savee = 0
# if savee > 0:
#     output_file = open('/Users/jake/Documents/baroclinic_modes/Shipboard/BATS_station/bats_ship_ctd_gridded_oct01.pkl',
#                        'wb')
#     my_dict = {'grid': grid, 'grid_p': grid_p, 'cast_log': cast_log, 'cast_date': cast_date, 'cast_lat': cast_lat,
#                'cast_lon': cast_lon, 'cast_temp': cast_t, 'cast_salin': cast_s, 'cast_max_z': cast_max_z}
#     pickle.dump(my_dict, output_file)
#     output_file.close()

# ------------------------------------------------------------------------------------------------------------
# ------------- LOAD GRIDDED CASTS -------------
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Shipboard/BATS_station/bats_ship_ctd_gridded_oct01.pkl', 'rb')
bats_ctd = pickle.load(pkl_file)
pkl_file.close()
c_lon = bats_ctd['cast_lon'][0]
c_lat = bats_ctd['cast_lat'][0]
span = np.where((c_lon > 64) & (c_lon < 64.3) & (c_lat > 31.5) & (c_lat < 31.9))[0]
c_lon = c_lon[span]
c_lat = c_lat[span]
c_log = bats_ctd['cast_log'][0, span]
c_date = bats_ctd['cast_date'][0, span]
c_t = bats_ctd['cast_temp'][:, span]
c_s = bats_ctd['cast_salin'][:, span]
c_max_z = bats_ctd['cast_max_z'][0, span]

c_t[c_t < 0.5] = np.nan
c_t[c_t > 30] = np.nan
c_s[c_s < 34] = np.nan
c_s[c_s > 40] = np.nan
for i in range(len(c_lon)):
    c_t[:, i] = nanseg_interp(grid, c_t[:, i])
    c_s[:, i] = nanseg_interp(grid, c_s[:, i])
    # c_t[np.isnan(c_t[:, i]), i] = c_t[np.where(c_t[~np.isnan(c_t[:, i]), i])[0][-1], i]
    # c_s[np.isnan(c_s[:, i]), i] = c_s[np.where(c_s[~np.isnan(c_s[:, i]), i])[0][-1], i]
    #
    # if c_t[(c_t[:, i] > 15) & (grid > 1000), i].size > 0:
    #     c_t[(c_t[:, i] > 15) & (grid > 1000), i] = np.nan

good_s = np.where(c_s[-10, :] < 35)[0]
good_t = np.where(c_t[-10, :] < 3)[0]
good = np.intersect1d(good_s, good_t)
c_s = c_s[:, good]
c_t = c_t[:, good]
c_date = c_date[good]
c_lon = c_lon[good]
c_lat = c_lat[good]
c_log = c_log[good]

# grid = GD.variables['grid'][:]
grid_p = gsw.p_from_z(-1 * grid, ref_lat)
z = -1 * grid
# c_s = np.concatenate([c_s, np.tile(c_s[-1, :], (19, 1))], axis=0)
# c_t = np.concatenate([c_t, np.tile(c_t[-1, :], (19, 1))], axis=0)


# ------------- initial processing
num_profs0 = np.sum(np.isfinite(c_log))
theta = np.nan * np.zeros((len(grid), num_profs0))
conservative_t = np.nan * np.zeros((len(grid), num_profs0))
abs_salin = np.nan * np.zeros((len(grid), num_profs0))
sigma0 = np.nan * np.zeros((len(grid), num_profs0))
sigma_theta = np.nan * np.zeros((len(grid), num_profs0))
for i in range(num_profs0):
    theta[:, i] = sw.ptmp(c_s[:, i], c_t[:, i], grid_p, 0)
    sigma_theta[:, i] = sw.dens(c_s[:, i], theta[:, i], 0) - 1000
    abs_salin[:, i] = gsw.SA_from_SP(c_s[:, i], grid_p, c_lon[i] * np.ones(len(grid_p)),
                                     c_lat[i] * np.ones(len(grid_p)))
    conservative_t[:, i] = gsw.CT_from_t(abs_salin[:, i], c_t[:, i], grid_p)
    sigma0[:, i] = gsw.sigma0(abs_salin[:, i], conservative_t[:, i])

# exclude profiles with density perturbations at depth that seem like ctd calibration offsets
cal_good = np.where(np.abs(sigma0[grid == 4100, :] - 27.89) < 0.0075)[1]
num_profs = len(cal_good)
c_log = c_log[cal_good]
c_date = c_date[cal_good]
c_lon = c_lon[cal_good]
c_lat = c_lat[cal_good]
theta = theta[:, cal_good]
salin = c_s[:, cal_good]
sigma_theta = sigma_theta[:, cal_good]
abs_salin = abs_salin[:, cal_good]
conservative_t = conservative_t[:, cal_good]
sigma0 = sigma0[:, cal_good]

# -- construct four background profiles to represent seasons
date_year = np.floor(c_date)
date_month = c_date - date_year
# Mar 1 - June 1
d_spring = np.where((date_month > 3 / 12) & (date_month < 6 / 12))[0]
# June 1 - Sept 1
d_summer = np.where((date_month > 6 / 12) & (date_month < 9 / 12))[0]
# Sept 1 - Nov 1
d_fall = np.where((date_month > 9 / 12) & (date_month < 11 / 12))[0]
# Nov 1 - Mar 1
d_winter = np.where((date_month > 11 / 12) | (date_month < 3 / 12))[0]

bckgrds = [d_spring, d_summer, d_fall, d_winter]
# -----------------------------------------------------------------------------

salin_avg = np.nan * np.zeros((len(grid), 4))
conservative_t_avg = np.nan * np.zeros((len(grid), 4))
theta_avg = np.nan * np.zeros((len(grid), 4))
sigma_theta_avg = np.nan * np.zeros((len(grid), 4))
ddz_avg_sigma = np.nan * np.zeros((len(grid), 4))
N2 = np.nan * np.zeros(sigma_theta_avg.shape)
N = np.nan * np.zeros(sigma_theta_avg.shape)
for i in range(4):
    inn = bckgrds[i]
    salin_avg[:, i] = np.nanmean(abs_salin[:, inn], axis=1)
    conservative_t_avg[:, i] = np.nanmean(conservative_t[:, inn], axis=1)
    theta_avg[:, i] = np.nanmean(theta[:, inn], axis=1)
    sigma_theta_avg[:, i] = np.nanmean(sigma0[:, inn], axis=1)

    # N2[1:] = np.squeeze(sw.bfrq(salin_avg, theta_avg, grid_p, lat=ref_lat)[0])
    go = ~np.isnan(salin_avg[:, i])
    N2[np.where(go)[0][0:-1], i] = gsw.Nsquared(salin_avg[go, i], conservative_t_avg[go, i], grid_p[go], lat=ref_lat)[0]
    N2[N2[:, i] < 0] = np.nan
    N2[:, i] = nanseg_interp(grid, N2[:, i])
    # last_good = np.where(~np.isnan(N2[:, i]))[0][-1]
    # if last_good.size > 0:
    #     N2[last_good + 1:, i] = N2[last_good, i]
    # # N[:, i] = np.sqrt(N2[:, i])
    # # N2[np.where(np.isnan(N2))[0]] = N2[np.where(np.isnan(N2))[0][0] - 1]

# correct last value of N2
for i in range(7, 0, -1):
    N2[-i, :] = N2[-i - 1, :] - 1*10**-9

# -- N2 using all profiles (not by season)
N2_all = np.nan * np.zeros(len(grid))
abs_s_avg = np.nanmean(abs_salin, axis=1)
cons_t_avg = np.nanmean(conservative_t_avg, axis=1)
go = ~np.isnan(abs_s_avg)
N2_all[np.where(go)[0][0:-1]] = gsw.Nsquared(abs_s_avg[go], cons_t_avg[go], grid_p[go], lat=ref_lat)[0]
N2_all[N2_all < 0] = np.nan
N2_all = nanseg_interp(grid, N2_all)
# last_good = np.where(~np.isnan(N2_all))[0][-1]r
# if last_good.size > 0:
#     N2_all[last_good + 1:] = N2_all[last_good]
# correct last value of N2
for i in range(7, 0, -1):
    N2_all[-i] = N2_all[-i - 1] - 1*10**-9

for i in range(4):
    ddz_avg_sigma[:, i] = (-1025/g) * N2[:, i]  # np.gradient(sigma_theta_avg[:, i], z)

# --- eta
eta = np.nan * np.zeros((len(grid), num_profs))
sigma_anom = np.nan * np.zeros((len(grid), num_profs))
conservative_t_anom = np.nan * np.zeros((len(grid), num_profs))
for i in range(num_profs):
    this_time = c_date[i] - np.floor(c_date[i])
    cor_b = np.zeros(4)
    for j in range(4):
        if j > 2:
            if (this_time > date_month[bckgrds[j]].min()) | (this_time < date_month[bckgrds[j]].max()):
                cor_b[j] = 1
        else:
            if (this_time > date_month[bckgrds[j]].min()) & (this_time < date_month[bckgrds[j]].max()):
                cor_b[j] = 1
    eta[:, i] = (sigma0[:, i] - sigma_theta_avg[:, cor_b > 0][:, 0]) / ddz_avg_sigma[:, cor_b > 0][:, 0]
    # eta[:, i] = (sigma0[:, i] - np.nanmean(sigma_theta_avg, axis=1))/np.nanmean(ddz_avg_sigma, axis=1)
    sigma_anom[:, i] = (sigma0[:, i] - np.nanmean(sigma_theta_avg, axis=1))
    conservative_t_anom[:, i] = (conservative_t[:, i] - conservative_t_avg[:, cor_b > 0][:, 0])

# -- look for long term warming/cooling trends
f, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, sharex=True)
m0 = np.polyfit(c_date, conservative_t_anom[0, :], 1)
cta0 = np.polyval(m0, c_date)
m1 = np.polyfit(c_date, conservative_t_anom[65, :], 1)
cta1 = np.polyval(m1, c_date)
m2 = np.polyfit(c_date, conservative_t_anom[115, :], 1)
cta2 = np.polyval(m2, c_date)
m3 = np.polyfit(c_date, conservative_t_anom[215, :], 1)
cta3 = np.polyval(m3, c_date)
ax0.plot(c_date, conservative_t_anom[0, :])
ax1.plot(c_date, conservative_t_anom[65, :])
ax2.plot(c_date, conservative_t_anom[115, :])
ax3.plot(c_date, conservative_t_anom[215, :])
ax0.plot(c_date, cta0, color='k')
ax1.plot(c_date, cta1, color='k')
ax2.plot(c_date, cta2, color='k')
ax3.plot(c_date, cta3, color='k')
ax0.set_title('T anom at 5m')
ax0.grid()
ax1.set_title('T anom at 1000m')
ax1.grid()
ax2.set_title('T anom at 2000m')
ax2.grid()
ax3.set_title('T anom at 4000m')
ax3.set_xlabel('Date')
plot_pro(ax3)

colors = plt.cm.Dark2(np.arange(0, 4, 1))
# --- T/S plot and lat/lon profile location
f, (ax, ax2, ax3, ax4) = plt.subplots(1, 4)
for i in range(num_profs):
    ax.scatter(abs_salin[:, i], conservative_t[:, i], s=1)
# colors = ['r', 'g', 'b', 'k']
for i in range(4):
    ax.plot(salin_avg[:, i], conservative_t_avg[:, i], linewidth=2, color=colors[i])
    ax2.plot(N2[:, i], grid, linewidth=2, color=colors[i])
    ax3.plot(ddz_avg_sigma[:, i], grid, linewidth=2, color=colors[i])
    ax4.plot(sigma_anom[:, bckgrds[i]], grid, color=colors[i], linewidth=0.5)
ax.grid()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax4.invert_yaxis()
ax2.grid()
ax3.grid()
plot_pro(ax4)

# MODE PARAMETERS
# frequency zeroed for geostrophic modes
omega = 0
# highest baroclinic mode to be calculated
mmax = 60
nmodes = mmax + 1
eta_fit_dep_min = 75
eta_fit_dep_max = 3750

# -- computer vertical mode shapes
G, Gz, c, epsilon = vertical_modes(N2_all, grid, omega, mmax)

# -- cycle through seasons
AG = []
PE_per_mass = []
eta_m = []
Neta_m = []
Eta_m = []
for i in range(4):
    AG_out, eta_m_out, Neta_m_out, PE_per_mass_out = eta_fit(len(bckgrds[i]), grid, nmodes,
                                                             N2_all, G, c, eta[:, bckgrds[i]],
                                                             eta_fit_dep_min,
                                                             eta_fit_dep_max)
    AG.append(AG_out)
    eta_m.append(eta_m_out)
    Neta_m.append(Neta_m_out)
    PE_per_mass.append(PE_per_mass_out)
    if i < 1:
        Eta_m = eta_m_out
    else:
        Eta_m = np.concatenate([Eta_m, eta_m_out], axis=1)

AG_all, eta_m_all, Neta_m_all, PE_per_mass_all = eta_fit(num_profs, grid, nmodes,
                                                         N2_all, G, c, eta, eta_fit_dep_min,
                                                         eta_fit_dep_max)

# --- find EOFs of dynamic vertical displacement (Eta mode amplitudes)
AG_avg = AG_all.copy()
good_prof_eof = np.where(~np.isnan(AG_avg[2, :]))
num_profs_2 = np.size(good_prof_eof)
AG2 = AG_avg[:, good_prof_eof[0]]
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

# # eof_test =
# grid_test = grid[50:750]
# eta_test = eta[50:750, 0:10]
# RR = np.matrix(eta_test) * np.transpose(np.matrix(eta_test))
# d_test, v_test = np.linalg.eig(RR)

# f, ax = plt.subplots()
# # for i in range(1):
# #     ax.plot(AGs[:, i], np.arange(0, 61, 1))
# ax.plot(eta[0:750, 1], grid[0:750])
# ax.invert_yaxis()
# plot_pro(ax)

# ----- PLOT DENSITY ANOM, ETA, AND MODE SHAPES
f, (ax, ax2, ax3) = plt.subplots(1, 3, sharey=True)
for i in range(num_profs):
    ax.plot(sigma_anom[:, i], grid, linewidth=0.75)
    ax2.plot(eta[:, i], grid, linewidth=.75, color='#808000')
    ax2.plot(Eta_m[:, i], grid, linewidth=.5, color='k', linestyle='--')
n2p = ax3.plot((np.sqrt(N2_all) * (1800 / np.pi)) / 4, grid, color='k', label='N(z) [cph]')
for j in range(4):
    ax3.plot(G[:, j] / grid.max(), grid, color='#2F4F4F', linestyle='--')
    p_eof = ax3.plot(-EOFetashape[:, j] / grid.max(), grid, color=colors[j, :], label='EOF # = ' + str(j + 1),
                     linewidth=2.5)
ax.text(0.2, 4000, str(num_profs) + ' profiles', fontsize=10)
ax.grid()
ax.set_title('BATS Hydrography ' + str(np.round(c_date.min())) + ' - ' + str(np.round(c_date.max())))
ax.set_xlabel(r'$\sigma_{\theta} - \overline{\sigma_{\theta}}$')
ax.set_ylabel('Depth [m]')
ax.set_xlim([-.5, .5])
ax2.grid()
ax2.set_title('Isopycnal Displacement [m]')
ax2.set_xlabel('[m]')
ax2.axis([-400, 400, 0, 4750])

handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels, fontsize=10)
ax3.set_title('EOFs of mode amplitudes (G(z))')
ax3.set_xlabel('Normalized Mode Amplitude')
ax3.set_xlim([-1, 1])
ax3.invert_yaxis()
plot_pro(ax3)

# --- PLOT PE PER SEASON
f_ref = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)
dk = f_ref / c[1]
sc_x = 1000 * f_ref / c[1:]
f, ax = plt.subplots()
ax.plot(sc_x, np.nanmean(PE_per_mass[0][1:], axis=1) / dk, color='r', linewidth=3)
ax.plot(sc_x, np.nanmean(PE_per_mass[1][1:], axis=1) / dk, color='m', linewidth=3)
ax.plot(sc_x, np.nanmean(PE_per_mass[2][1:], axis=1) / dk, color='c', linewidth=3)
ax.plot(sc_x, np.nanmean(PE_per_mass[3][1:], axis=1) / dk, color='b', linewidth=3)
ax.set_yscale('log')
ax.set_xscale('log')
ax.axis([10 ** -2, 10 ** 1, 10 ** (-4), 2 * 10 ** 3])
ax.axis([10 ** -2, 10 ** 1, 10 ** (-4), 10 ** 3])
plot_pro(ax)

# --- PLOT MODE AMPLITUDES IN TIME
window_size = 25
poly_order = 3
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# colors = ['r', 'g', 'b', 'm']
# for i in range(4):
#     AG1 = AG[i][1, :].copy()
#     AG1 = nanseg_interp(c_date[bckgrds[i]], AG1)
#     y_sg = savgol_filter(AG1, window_size, poly_order)
#     AG2 = AG[i][2, :].copy()
#     AG2 = nanseg_interp(c_date[bckgrds[i]], AG2)
#     y_sg2 = savgol_filter(AG2, window_size, poly_order)
#
#     # ax.plot(c_date[bckgrds[i]], AG[1, :], color='#FF8C00', linewidth=0.5)
#     ax1.plot(c_date[bckgrds[i]], y_sg, color=colors[i], linewidth=1, label='Mode 1')
#     # ax.plot(c_date[bckgrds[i]], AG[2, :], linewidth=0.5, color='#5F9EA0')
#     ax2.plot(c_date[bckgrds[i]], y_sg2, color=colors[i], linewidth=1, label='Mode 2')

AG1_all = AG_all[1, :].copy()
AG1_all = nanseg_interp(c_date, AG1_all)
y_sg_all = savgol_filter(AG1_all, window_size, poly_order)
ax1.plot(c_date, c[1] * AG1_all, color='k', linewidth=2, label='Mode 1')
AG2_all = AG_all[2, :].copy()
AG2_all = nanseg_interp(c_date, AG2_all)
y_sg2_all = savgol_filter(AG2_all, window_size, poly_order)
ax2.plot(c_date, c[2] * AG2_all, color='k', linewidth=2, label='Mode 1')
# ax1.legend(['Spring', 'Summer', 'Fall', 'Winter', 'All'], fontsize=10)
ax2.set_xlabel('Date')
ax1.set_ylabel('Mode Amplitude')
ax2.set_ylabel('Mode Amplitude')
ax1.set_title(r'Station BATS Scaled Mode 1 Amplitude (c$_{n}\beta_{n}$) in Time')
ax2.set_title(r'Station BATS Mode 2 Amplitude (c$_{n}\beta_{n}$) in Time')
ax1.set_ylim([-.16, 0.25])
ax2.set_ylim([-.16, 0.25])
ax1.grid()
plot_pro(ax2)

# -- attempt fft to find period of oscillation
window_size, poly_order = 9, 2
Time_grid = np.arange(np.round(c_date.min()), np.round(c_date.max()), 1/12)

order_0_AG = AG_all[1, :]
y_AG_0 = savgol_filter(order_0_AG, window_size, poly_order)
order_0_AG_grid = np.interp(Time_grid, c_date, y_AG_0)

order_1_AG = AG_all[2, :]
y_AGz_1 = savgol_filter(order_1_AG, window_size, poly_order)
order_1_AG_grid = np.interp(Time_grid, c_date, y_AGz_1)

N = len(order_0_AG_grid)
T = Time_grid[1] - Time_grid[0]
yf_0 = scipy.fftpack.fft(order_0_AG_grid)
yf_1 = scipy.fftpack.fft(order_1_AG_grid)
xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
f, ax = plt.subplots()
ax.plot(xf, 2.0 / N * np.abs(yf_0[:N // 2]), 'r')
ax.plot(xf, 2.0 / N * np.abs(yf_1[:N // 2]), 'b')
plot_pro(ax)

# --- SAVE
# write python dict to a file
sa = 1
if sa > 0:
    my_dict = {'depth': grid, 'Sigma0': sigma0, 'lon': c_lon, 'lat': c_lat, 'time': c_date,
               'N2_per_season': N2, 'background indices': bckgrds, 'background order': ['spr', 'sum', 'fall', 'wint'],
               'AG': AG_all, 'AG_per_season': AG, 'Eta': eta, 'Eta_m': eta_m_all, 'NEta_m': Neta_m_all,
               'PE': PE_per_mass_all, 'PE_by_season': PE_per_mass, 'c': c}
    output = open('/Users/jake/Desktop/bats/station_bats_pe_oct01.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()

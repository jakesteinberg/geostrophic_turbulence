# STATION BATS (PE profiles)

import numpy as np
import matplotlib.pyplot as plt
import gsw
import seawater as sw
import pickle
from scipy.signal import savgol_filter
# functions I've written
from grids import make_bin_gen
from mode_decompositions import vertical_modes, eta_fit, vertical_modes_f
from toolkit import plot_pro, nanseg_interp

# ------ physical parameters
g = 9.81
rho0 = 1027
bin_depth = np.concatenate([np.arange(0, 150, 5), np.arange(150, 300, 5), np.arange(300, 4600, 5)])
ref_lat = 31.7
ref_lon = -64.2
grid = bin_depth[0:-1]
grid_p = sw.pres(grid, ref_lat)
z = -1 * grid

# write this to process any set of shipboard hydrography (specify file type)

# file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/BATS_station/CTD_data/b1*ctd.txt')
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
#                 c_d = sw.dpth(cast_p, ref_lat)
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
#                     cast_t[g_ol[1:-1], count], cast_s[g_ol[1:-1], count] = make_bin_gen(grid[g_ol], c_d, cast_t_pre,
#                                                                                         cast_s_pre)
#                     # cast_t[g_overlap, count] = np.interp(grid[g_overlap], cast_d, cast_t_pre)
#                     # cast_s[g_overlap, count] = np.interp(grid[g_overlap], cast_d, cast_s_pre)
#                     cast_lat[0][count] = np.nanmean(cast_data[:, 2])
#                     cast_lon[0][count] = np.nanmean(cast_data[:, 3])
#                     cast_date[0][count] = np.nanmean(cast_data[:, 1])
#                     cast_log[0][count] = cast_data[0, 0]
#                     count = count + 1
# # --- end looping and cast selection
#
# savee = 1
# if savee > 0:
#     output_file = open('/Users/jake/Documents/baroclinic_modes/BATS_station/bats_ship_ctd_gridded.pkl', 'wb')
#     my_dict = {'grid': grid, 'grid_p': grid_p, 'cast_log': cast_log, 'cast_date': cast_date, 'cast_lat': cast_lat,
#                'cast_lon': cast_lon, 'cast_temp': cast_t, 'cast_salin': cast_s}
#     pickle.dump(my_dict, output_file)
#     output_file.close()

# ------------- LOAD GRIDDED CASTS -------------
pkl_file = open('/Users/jake/Documents/baroclinic_modes/BATS_station/bats_ship_ctd_gridded.pkl', 'rb')
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

salin_avg = np.nanmean(abs_salin, axis=1)
conservative_t_avg = np.nanmean(conservative_t, axis=1)
theta_avg = np.nanmean(theta, axis=1)
sigma_theta_avg = np.nanmean(sigma0, axis=1)
ddz_avg_sigma = np.gradient(sigma_theta_avg, z)
N2 = np.nan * np.zeros(sigma_theta_avg.size)
# N2[1:] = np.squeeze(sw.bfrq(salin_avg, theta_avg, grid_p, lat=ref_lat)[0])
N2[0:-1] = gsw.Nsquared(salin_avg, conservative_t_avg, grid_p, lat=ref_lat)[0]
N2[N2 < 0] = np.nan
N2 = nanseg_interp(grid, N2)
N = np.sqrt(N2)
N2[np.where(np.isnan(N2))[0]] = N2[np.where(np.isnan(N2))[0][0]-1]
# lz = np.where(N2 < 0)
# lnan = np.isnan(N2)
# N2[lz] = 0
# N2[lnan] = 0
# N = np.sqrt(N2)

# --- T/S plot and lat/lon profile location
# f, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.scatter(salin, theta, s=0.5)
# for i in range(num_profs):
#     ax2.plot(sigma_theta[:, i] - sigma_theta_avg, grid, linewidth=.5)
# ax2.invert_yaxis()
# ax2.axis([-1, 1, 0, 5000])
# ax3.scatter(c_lon, c_lat, s=3)
# plot_pro(ax3)

# --- eta
eta = np.nan * np.zeros((len(grid), num_profs))
for i in range(num_profs):
    eta[:, i] = (sigma0[:, i] - sigma_theta_avg) / ddz_avg_sigma

# MODE PARAMETERS
# frequency zeroed for geostrophic modes
omega = 0
# highest baroclinic mode to be calculated
mmax = 60
nmodes = mmax + 1
eta_fit_dep_min = 50
eta_fit_dep_max = 3600

# computer vertical mode shapes
G, Gz, c = vertical_modes(N2, grid, omega, mmax)

AG, eta_m, NEta_m, PE_per_mass = eta_fit(num_profs, grid, nmodes, N2, G, c, eta, eta_fit_dep_min, eta_fit_dep_max)

# --- find EOFs of dynamic vertical displacement (Eta mode amplitudes)
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

# eof_test =
grid_test = grid[50:750]
eta_test = eta[50:750, 0:10]
RR = np.matrix(eta_test) * np.transpose(np.matrix(eta_test))
d_test, v_test = np.linalg.eig(RR)

# f, ax = plt.subplots()
# # for i in range(1):
# #     ax.plot(AGs[:, i], np.arange(0, 61, 1))
# ax.plot(eta[0:750, 1], grid[0:750])
# ax.invert_yaxis()
# plot_pro(ax)

# ----- PLOT DENSITY ANOM, ETA, AND MODE SHAPES
colors = plt.cm.Dark2(np.arange(0, 4, 1))
f, (ax, ax2, ax3) = plt.subplots(1, 3, sharey=True)
for i in range(num_profs):
    ax.plot((sigma_theta[:, i] - sigma_theta_avg), grid, linewidth=0.75)
    ax2.plot(eta[:, i], grid, linewidth=.75, color='#808000')
    ax2.plot(eta_m[:, i], grid, linewidth=.5, color='k', linestyle='--')
n2p = ax3.plot((np.sqrt(N2) * (1800 / np.pi)) / 10, grid, color='k', label='N(z) [cph]')
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
ax2.axis([-600, 600, 0, 4750])

handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels, fontsize=10)
ax3.set_title('EOFs of mode amplitudes (G(z))')
ax3.set_xlabel('Normalized Mode Amplitude')
ax3.set_xlim([-1, 1])
ax3.invert_yaxis()
plot_pro(ax3)

# --- PLOT MODE AMPLITUDES IN TIME
window_size = 25
poly_order = 3
AG1 = AG[1, :].copy()
AG1 = nanseg_interp(c_date, AG1)
y_sg = savgol_filter(AG1, window_size, poly_order)
AG2 = AG[2, :].copy()
AG2 = nanseg_interp(c_date, AG2)
y_sg2 = savgol_filter(AG2, window_size, poly_order)
AG3 = AG[3, :].copy()
AG3 = nanseg_interp(c_date, AG3)
y_sg3 = savgol_filter(AG3, window_size, poly_order)

f, ax = plt.subplots()
ax.plot(c_date, AG[1, :], color='#FF8C00', linewidth=0.5)
ax.plot(c_date, y_sg, color='#FF8C00', linewidth=3, label='Mode 1')
ax.plot(c_date, AG[2, :], linewidth=0.5, color='#5F9EA0')
ax.plot(c_date, y_sg2, color='#5F9EA0', linewidth=2, label='Mode 2')
# ax.plot(c_date, AG[3, :], linewidth=0.5, color='#8B0000')
ax.plot(c_date, y_sg3, color='#8B0000', linewidth=1, label='Mode 3')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=10)
ax.set_xlabel('Date')
ax.set_ylabel('Mode Amplitude')
ax.set_title('Station BATS Mode Amplitude in Time')
plot_pro(ax)

# --- SAVE
# write python dict to a file
sa = 1
if sa > 0:
    my_dict = {'depth': grid, 'Sigma_Theta': sigma_theta, 'lon': c_lon, 'lat': c_lat, 'time': c_date, 'N2': N2,
               'AG': AG, 'Eta': eta, 'Eta_m': eta_m, 'NEta_m': NEta_m, 'PE': PE_per_mass, 'c': c}
    output = open('/Users/jake/Desktop/bats/station_bats_pe_apr09.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()


# todo how should mode amplitude decay as mode number increases?
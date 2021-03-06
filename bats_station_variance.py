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
from mode_decompositions import vertical_modes, eta_fit, PE_Tide_GM
from toolkit import plot_pro, nanseg_interp, find_nearest

# ------ physical parameters
g = 9.81
rho0 = 1027  # - 1027
GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
bin_depth = np.concatenate([np.arange(0, 150, 5), np.arange(150, 300, 5), np.arange(300, 4550, 10)])
ref_lat = 31.7
ref_lon = -64.2
grid = bin_depth  # [0:231]
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
#     output_file = open('/Users/jake/Documents/baroclinic_modes/Shipboard/BATS_station/bats_ship_ctd_gridded_oct01_prot3.pkl',
#                        'wb')
#     my_dict = {'grid': grid, 'grid_p': grid_p, 'cast_log': cast_log, 'cast_date': cast_date, 'cast_lat': cast_lat,
#                'cast_lon': cast_lon, 'cast_temp': cast_t, 'cast_salin': cast_s, 'cast_max_z': cast_max_z}
#     pickle.dump(my_dict, output_file)
#     output_file.close()


# ------------------------------------------------------------------------------------------------------------
# ------------- LOAD GRIDDED CASTS -------------
pkl_file = open(
    '/Users/jake/Documents/baroclinic_modes/Shipboard/BATS_station/bats_ship_ctd_gridded_oct01_prot3.pkl', 'rb')
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

ordered = np.argsort(c_date)
c_s = c_s[:, ordered]
c_t = c_t[:, ordered]
c_date = c_date[ordered]
c_lon = -1 * c_lon[ordered]
c_lat = c_lat[ordered]
c_log = c_log[ordered]

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

# good_s = np.where(c_s[-10, :] < 34)[0]
# good_t = np.where(c_t[-10, :] < 2)[0]
# good = np.intersect1d(good_s, good_t)
# c_s = c_s[:, good]
# c_t = c_t[:, good]
# c_date = c_date[good]
# c_lon = -1 * c_lon[good]
# c_lat = c_lat[good]
# c_log = c_log[good]
# grid_p[0] = 0

# --- convert to gamma in matlab (need to run glider environment) -----
# import matlab.engine
# import time as TT
# eng = matlab.engine.start_matlab()
# eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/')
# eng.addpath(r'/Users/jake/Documents/MATLAB/eos80_legacy_gamma_n/library/')
# gamma = np.nan * np.ones(np.shape(c_s))
# print('Opened Matlab')
# for i in range(len(c_date)):  # loop over each profile
#     tic = TT.clock()
#     gamma[:, i] = np.squeeze(np.array(eng.eos80_legacy_gamma_n(matlab.double(c_s[:, i].tolist()),
#                                                                matlab.double(c_t[:, i].tolist()),
#                                                                matlab.double(grid_p.tolist()),
#                                                                matlab.double([c_lon[i]]),
#                                                                matlab.double([c_lat[i]]))))
#     toc = TT.clock()
#     print('Time step = ' + str(i) + ' = '+ str(toc - tic) + 's')
# eng.quit()
# print('Closed Matlab')
# my_dict = {'z': -1 * grid, 'gamma': gamma}
# output = open('/Users/jake/Documents/baroclinic_modes/Shipboard/BATS_station/extracted_gridded_gamma.pkl', 'wb')
# pickle.dump(my_dict, output)
# output.close()
#
# from scipy.io import netcdf
# f = netcdf.netcdf_file('/Users/jake/Documents/baroclinic_modes/Shipboard/BATS_station/extracted_gridded_gamma_2.nc', 'w')
# f.createDimension('grid', np.size(grid))
# f.createDimension('profile_list', np.size(c_date))
# b_d = f.createVariable('grid', np.float64, ('grid',))
# b_d[:] = grid
# b_l = f.createVariable('dive_list', np.float64, ('profile_list',))
# b_l[:] = c_date
# b_t = f.createVariable('Gamma', np.float64, ('grid', 'profile_list'))
# b_t[:] = gamma
# f.close()
# -------------------------------------------------------------------------

# -- LOAD PROCESSED MODEL TIME STEPS WITH COMPUTED GAMMA
# pkl_file = open('/Users/jake/Documents/baroclinic_modes/Shipboard/BATS_station/extracted_gridded_gamma.pkl', 'rb')
# MOD = pickle.load(pkl_file)
# pkl_file.close()
# gamma = MOD['gamma'][:]

# --- load matlabbed gamma as nc
GD = Dataset('/Users/jake/Documents/baroclinic_modes/Shipboard/BATS_station/extracted_gridded_gamma_2.nc', 'r')
gamma = GD.variables['Gamma'][:]

# grid = GD.variables['grid'][:]
grid_p = gsw.p_from_z(-1 * grid, ref_lat)
z = -1 * grid
# c_s = np.concatenate([c_s, np.tile(c_s[-1, :], (19, 1))], axis=0)
# c_t = np.concatenate([c_t, np.tile(c_t[-1, :], (19, 1))], axis=0)

# --- initial processing
num_profs0 = np.sum(np.isfinite(c_log))
theta = np.nan * np.zeros((len(grid), num_profs0))
conservative_t = np.nan * np.zeros((len(grid), num_profs0))
abs_salin = np.nan * np.zeros((len(grid), num_profs0))
N2_per = np.nan * np.zeros((len(grid), num_profs0))
sigma0 = np.nan * np.zeros((len(grid), num_profs0))
sigma2 = np.nan * np.zeros((len(grid), num_profs0))
sigma4 = np.nan * np.zeros((len(grid), num_profs0))
sigma_theta = np.nan * np.zeros((len(grid), num_profs0))
# f, (ax, ax2) = plt.subplots(1, 2, sharey=True)
for i in range(num_profs0):
    abs_salin[:, i] = gsw.SA_from_SP(c_s[:, i], grid_p, c_lon[i] * np.ones(len(grid_p)),
                                     c_lat[i] * np.ones(len(grid_p)))
    conservative_t[:, i] = gsw.CT_from_t(abs_salin[:, i], c_t[:, i], grid_p)

    # old
    theta[:, i] = sw.ptmp(c_s[:, i], c_t[:, i], grid_p, 0)
    sigma_theta[:, i] = sw.dens(c_s[:, i], theta[:, i], 0) - 1000
    sigma0[:, i] = gsw.sigma0(abs_salin[:, i], conservative_t[:, i])
    sigma2[:, i] = gsw.rho(abs_salin[:, i], conservative_t[:, i], 2000) - 1000
    sigma4[:, i] = gsw.sigma4(abs_salin[:, i], conservative_t[:, i])

    go = ~np.isnan(abs_salin[:, i])
    N2_per[np.where(go)[0][0:-1], i] = gsw.Nsquared(abs_salin[go, i], conservative_t[go, i], grid_p[go], lat=ref_lat)[0]

    # ax.plot(abs_salin[:, i], grid_p, color='r', linewidth=0.5)
    # # ax2.plot(N2_per[:, i], grid_p, color='r', linewidth=0.5)
    # ax2.plot(sigma0[:, i], grid_p, color='r', linewidth=0.5)
    # ax2.plot(sigma2[:, i], grid_p, color='b', linewidth=0.5)
    # ax2.plot(sigma4[:, i] - 9, grid_p, color='g', linewidth=0.5)
    # # ax2.plot(sigma_theta[:, i], grid_p, color='g', linewidth=0.5)
# ax.set_title('Abs. Salin')
# # ax2.set_xlim([0, 0.0008])
# # ax2.set_title('N2')
# # ax3.set_xlim([24, 30.5])
# ax2.set_title('Reference Pressure Comp.')
# ax2.invert_yaxis()
# ax.grid()
# plot_pro(ax2)

# --- exclude profiles with density perturbations at depth that seem like ctd calibration offsets
# cal_good = np.where(np.abs(sigma0[grid == 4100, :] - 27.89) < 0.0075)[1]
cal_good = np.where(np.abs(sigma4[grid == 4100, :] - 45.9) < 0.0075)[1]
num_profs = len(cal_good)
c_log = c_log[cal_good]
c_date = c_date[cal_good]
c_lon = c_lon[cal_good]
c_lat = c_lat[cal_good]
theta = theta[:, cal_good]
salin = c_s[:, cal_good]
sigma_theta = sigma_theta[:, cal_good]
a_salin = abs_salin[:, cal_good]
c_temp = conservative_t[:, cal_good]
sigma0 = sigma0[:, cal_good]
sigma2 = sigma2[:, cal_good]
sigma4 = sigma4[:, cal_good]
gamma = gamma[:, cal_good]

# -- construct four background profiles to represent seasons
date_year = np.floor(c_date)
date_day = 365 * (c_date - date_year)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']  # (up through)
days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
year_days = np.cumsum(days_per_month)

# alternate to match DG at BATS (3 seasons
# Nov - May (11 - 5)
d_winter1 = np.where((date_day < year_days[3 - 1]) | (date_day > year_days[11 - 1]))[0]  # true winter
# June 1 - Sept 1
d_summer = np.where((date_day > year_days[5 - 1]) & (date_day < year_days[8 - 1]))[0]
# Sept 1 - Nov 1
d_fall = np.where((date_day > year_days[8 - 1]) & (date_day < year_days[11 - 1]))[0]
bckgrds = [d_winter1, d_summer, d_fall]
season_labs = ['Dec-Mar', 'May-Sept', 'Sept-Nov']  # mirror to match DG BATS

# cols = 'b', 'r', 'g', 'c', 'm'
cols = ['#2F4F4F', '#FF4500', '#DAA520', '#800080']
bckgrds_lims = np.nan * np.ones((len(bckgrds), 2))
for i in range(len(bckgrds)):
    bckgrds_lims[i, :] = np.array([date_day[bckgrds[i]].min(), date_day[bckgrds[i]].max()])

# -----------------------------------------------------------------------------
# DIFFERENT BACKGROUND PROFILES
slen = np.shape(bckgrds)[0]
salin_avg = np.nan * np.zeros((len(grid), slen))
conservative_t_avg = np.nan * np.zeros((len(grid), slen))
theta_avg = np.nan * np.zeros((len(grid), slen))
sigma_theta_avg = np.nan * np.zeros((len(grid), slen))
sigma_theta_avg2 = np.nan * np.zeros((len(grid), slen))
sigma_theta_avg4 = np.nan * np.zeros((len(grid), slen))
ddz_avg_sigma = np.nan * np.zeros((len(grid), slen))
ddz_avg_sigma2 = np.nan * np.zeros((len(grid), slen))
ddz_avg_sigma4 = np.nan * np.zeros((len(grid), slen))
N2_0 = np.nan * np.zeros(sigma_theta_avg.shape)
N2 = np.nan * np.zeros(sigma_theta_avg.shape)
N = np.nan * np.zeros(sigma_theta_avg.shape)
for i in range(slen):
    inn = bckgrds[i]
    salin_avg[:, i] = np.nanmean(a_salin[:, inn], axis=1)
    conservative_t_avg[:, i] = np.nanmean(c_temp[:, inn], axis=1)

    theta_avg[:, i] = np.nanmean(theta[:, inn], axis=1)

    sigma_theta_avg[:, i] = np.nanmean(gamma[:, inn], axis=1)

    sigma_theta_avg2[:, i] = np.nanmean(sigma2[:, inn], axis=1)
    sigma_theta_avg4[:, i] = np.nanmean(sigma4[:, inn], axis=1)

    # N2[1:] = np.squeeze(sw.bfrq(salin_avg, theta_avg, grid_p, lat=ref_lat)[0])
    go = ~np.isnan(salin_avg[:, i])
    N2_0[np.where(go)[0][0:-1], i] = gsw.Nsquared(salin_avg[go, i], conservative_t_avg[go, i], grid_p[go], lat=ref_lat)[0]
    N2_0[N2_0[:, i] < 0] = np.nan
    N2[:, i] = nanseg_interp(grid, N2_0[:, i])
    last_good = np.where(~np.isnan(N2[:, i]))[0][-1]
    if (last_good.size) > 0 & (last_good > 400):
        for k in range(len(grid) - (last_good + 1)):
            N2[last_good + 1 + k:, i] = N2[last_good + k, i] - 1 * 10**-9
    # # N[:, i] = np.sqrt(N2[:, i])
    # # N2[np.where(np.isnan(N2))[0]] = N2[np.where(np.isnan(N2))[0][0] - 1]

# correct last value of N2
# for i in range(7, 0, -1):
#     N2[-i, :] = N2[-i - 1, :] - 1*10**-9

# -- N2 using all profiles (not by season)
N2_all = np.nan * np.zeros(len(grid))
abs_s_avg = np.nanmean(abs_salin, axis=1)
cons_t_avg = np.nanmean(conservative_t_avg, axis=1)
go = ~np.isnan(abs_s_avg)
# N2_all[np.where(go)[0][0:-1]] = gsw.Nsquared(abs_s_avg[go], cons_t_avg[go], grid_p[go], lat=ref_lat)[0]
N2_all = np.nanmean(N2_per, axis=1)
N2_all[N2_all < 0] = np.nan
# N2_all = nanseg_interp(grid, N2_all)
# last_good = np.where(~np.isnan(N2_all))[0][-1]r
# if last_good.size > 0:
#     N2_all[last_good + 1:] = N2_all[last_good]
# correct last value of N2
for i in range(7, 0, -1):
    N2_all[-i] = N2_all[-i - 1] - 1*10**-9
N2_all = savgol_filter(N2_all, 7, 3)

for i in range(slen):
    # ddz_avg_sigma[:, i] = (-rho0/g) * N2[:, i]
    ddz_avg_sigma[:, i] = np.gradient(sigma_theta_avg[:, i], z)
    ddz_avg_sigma2[:, i] = np.gradient(sigma_theta_avg2[:, i], z)
    ddz_avg_sigma4[:, i] = np.gradient(sigma_theta_avg4[:, i], z)

# --- ETA COMPUTATION
# for each depth bin compute density relative to a local reference pressure, then compute a relevant background  also
# to the same reference pressure. Need local vertical density gradient. Use this to compute density anom, and thus Eta
eta = np.nan * np.zeros((len(grid), num_profs))
eta2 = np.nan * np.zeros((len(grid), num_profs))
eta4 = np.nan * np.zeros((len(grid), num_profs))
sigma_anom = np.nan * np.zeros((len(grid), num_profs))
sigma_anom2 = np.nan * np.zeros((len(grid), num_profs))
sigma_anom4 = np.nan * np.zeros((len(grid), num_profs))
conservative_t_anom = np.nan * np.zeros((len(grid), num_profs))
this_eta = np.nan * np.zeros((len(grid), num_profs))
eta_alt = np.nan * np.zeros((len(grid), num_profs))
eta_alt_3 = np.nan * np.zeros((len(grid), num_profs))
this_anom = np.nan * np.zeros((len(grid), num_profs))
this_gradient = np.nan * np.zeros((len(grid), num_profs))
for i in range(num_profs):
    # find relevant time indices
    this_time = c_date[i] - np.floor(c_date[i])

    correct = np.zeros(slen)
    for j in range(3):
        if (this_time > bckgrds_lims[j, 0]) & (this_time < bckgrds_lims[j, 1]):
            correct[j] = 1
    if np.sum(correct) < 1:
        correct[-1] = 1

    # cor_b = np.zeros(slen)
    # for j in range(slen):
    #     if j > 2:
    #         if (this_time > date_month[bckgrds[j]].min()) | (this_time < date_month[bckgrds[j]].max()):
    #             cor_b[j] = 1
    #     else:
    #         if (this_time > date_month[bckgrds[j]].min()) & (this_time < date_month[bckgrds[j]].max()):
    #             cor_b[j] = 1

    # average T/S for the relevant season
    avg_a_salin = salin_avg[:, correct > 0][:, 0]
    avg_c_temp = conservative_t_avg[:, correct > 0][:, 0]
    avg_sigma_theta = sigma_theta_avg[:, correct > 0][:, 0]
    avg_ddz_sigma_theta = ddz_avg_sigma[:, correct > 0][:, 0]

    # eta method = eta_alt_2 = local pot density reference
    for j in range(1, len(grid) - 1):
        # profile density at depth j with local
        this_sigma = gsw.rho(a_salin[j, i], c_temp[j, i], grid_p[j]) - 1000
        this_sigma_avg = gsw.rho(avg_a_salin[j-1:j+2], avg_c_temp[j-1:j+2], grid_p[j]) - 1000
        this_anom[j, i] = this_sigma - this_sigma_avg[1]
        # this_gradient = (this_sigma_avg[0] - this_sigma_avg[2]) / (z[j-1] - z[j+1])
        this_gradient[j, i] = np.nanmean(np.gradient(this_sigma_avg, z[j-1:j+2]))
        this_eta[j, i] = this_anom[j, i] / this_gradient[j, i]

    # eta method = eta_alt = divide gamma by local gradient
    this_sigma_theta_avg = sigma_theta_avg[:, correct > 0][:, 0]
    eta_alt[:, i] = (gamma[:, i] - avg_sigma_theta) / np.squeeze(avg_ddz_sigma_theta)

    # eta method = eta_alt_3
    for j in range(len(grid)):
        # find this profile density at j along avg profile
        idx, rho_idx = find_nearest(this_sigma_theta_avg, gamma[j, i])
        if idx <= 2:
            z_rho_1 = grid[0:idx + 3]
            eta_alt_3[j, i] = np.interp(gamma[j, i], this_sigma_theta_avg[0:idx + 3], z_rho_1) - grid[j]
        else:
            z_rho_1 = grid[idx - 2:idx + 3]
            eta_alt_3[j, i] = np.interp(gamma[j, i], this_sigma_theta_avg[idx - 2:idx + 3], z_rho_1) - grid[j]

    eta[:, i] = (sigma0[:, i] - sigma_theta_avg[:, correct > 0][:, 0]) / ddz_avg_sigma[:, correct > 0][:, 0]
    eta2[:, i] = (sigma2[:, i] - sigma_theta_avg2[:, correct > 0][:, 0]) / ddz_avg_sigma2[:, correct > 0][:, 0]
    eta4[:, i] = (sigma4[:, i] - sigma_theta_avg4[:, correct > 0][:, 0]) / ddz_avg_sigma4[:, correct > 0][:, 0]
    # eta[:, i] = (sigma0[:, i] - np.nanmean(sigma_theta_avg, axis=1))/np.nanmean(ddz_avg_sigma, axis=1)
    sigma_anom[:, i] = (gamma[:, i] - sigma_theta_avg[:, correct > 0][:, 0])
    sigma_anom2[:, i] = (sigma2[:, i] - np.nanmean(sigma_theta_avg2, axis=1))
    sigma_anom4[:, i] = (sigma4[:, i] - np.nanmean(sigma_theta_avg4, axis=1))
    conservative_t_anom[:, i] = (conservative_t[:, i] - conservative_t_avg[:, correct > 0][:, 0])

# store for later
this_eta_0 = this_eta.copy()

# --- T/S/rho_anom plot and lat/lon profile location
f, (ax, ax2) = plt.subplots(1, 2, sharey=True)
# for i in range(num_profs):
#     ax.scatter(abs_salin[:, i], conservative_t[:, i], s=1)
for i in range(slen):
    # ax.plot(salin_avg[:, i], conservative_t_avg[:, i], linewidth=2, color=cols[i])
    ax.plot(N2[:, i], grid, linewidth=2, color=cols[i], label=season_labs[i])
    # ax3.plot(-1 * ddz_avg_sigma[:, i], grid, linewidth=2, color=cols[i])
    ax2.plot(sigma_anom[:, bckgrds[i]], grid, color=cols[i], linewidth=0.5)
    # ax5.plot(sigma_anom2[:, bckgrds[i]], grid, color=cols[i], linewidth=0.5)
ax.grid()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=12)
ax.set_ylim([0, 1500])
ax.set_ylabel('Depth [m]')
ax.set_title(r'Seasonal $N^2$')
ax2.set_xlim([-.8, .8])
ax2.invert_yaxis()
ax2.set_title('Density Anomaly')
plot_pro(ax2)

# MODE PARAMETERS
# frequency zeroed for geostrophic modes
omega = 0
# highest baroclinic mode to be calculated
mmax = 45
nmodes = mmax + 1
eta_fit_dep_min = 250
eta_fit_dep_max = 3750

# -- computer vertical mode shapes
G, Gz, c, epsilon = vertical_modes(N2_all, grid, omega, mmax)
# -- per season
G_0, Gz_0, c_0, epsilon_0 = vertical_modes(N2[:, 0], grid, omega, mmax)
G_1, Gz_1, c_1, epsilon_1 = vertical_modes(N2[:, 1], grid, omega, mmax)
G_2, Gz_2, c_2, epsilon_2 = vertical_modes(N2[:, 2], grid, omega, mmax)
G_i = [G_0, G_1, G_2]
Gz_i = [Gz_0, Gz_1, Gz_2]
c_i = [c_0, c_1, c_2]

# -- choose eta to use
this_eta_used = eta_alt_3  # this_eta_0  # local sig_theta calc

# -- cycle through seasons
AG = []
PE_per_mass = []
eta_m = []
Neta_m = []
Eta_m = []
Eta_m = np.nan * np.ones(np.shape(this_eta_used))
for i in range(slen):
    AG_out, eta_m_out, Neta_m_out, PE_per_mass_out = eta_fit(len(bckgrds[i]), grid, nmodes,
                                                             N2[:, i], G_i[i], c_i[i], this_eta_used[:, bckgrds[i]],
                                                             eta_fit_dep_min,
                                                             eta_fit_dep_max)
    AG.append(AG_out)
    eta_m.append(eta_m_out)
    Neta_m.append(Neta_m_out)
    PE_per_mass.append(PE_per_mass_out)
    Eta_m[:, bckgrds[i]] = eta_m_out
    # if i < 1:
    #     Eta_m = eta_m_out
    # else:
    #     Eta_m = np.concatenate([Eta_m, eta_m_out], axis=1)

AG_all_2, eta_m_all_2, Neta_m_all_2, PE_per_mass_all_2 = eta_fit(num_profs, grid, nmodes,
                                                                 N2_all, G, c, eta2, eta_fit_dep_min, eta_fit_dep_max)
AG_all, eta_m_all, Neta_m_all, PE_per_mass_all = eta_fit(num_profs, grid, nmodes,
                                                         N2_all, G, c, eta4, eta_fit_dep_min, eta_fit_dep_max)

# --- PLOT differences in vertical structure
plot_vert = 1
if plot_vert > 0:
    # --- test season differences in modes
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    for i in range(slen):
        ax1.plot(G_i[i][:, 1], grid, color=cols[i])
        ax2.plot(G_i[i][:, 2], grid, color=cols[i])
        ax3.plot(G_i[i][:, 3], grid, color=cols[i])
        ax4.plot(G_i[i][:, 4], grid, color=cols[i], label=season_labs[i])
    ax1.set_title('Mode 1')
    ax1.set_ylabel('Depth')
    ax1.grid()
    ax2.set_title('Mode 2')
    ax2.grid()
    ax3.set_title('Mode 3')
    ax3.grid()
    ax4.set_title('Mode 4')
    ax4.invert_yaxis()
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels, fontsize=10)
    plot_pro(ax4)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    for i in range(slen):
        ax1.plot(Gz_i[i][:, 1], grid, color=cols[i])
        ax2.plot(Gz_i[i][:, 2], grid, color=cols[i])
        ax3.plot(Gz_i[i][:, 3], grid, color=cols[i])
        ax4.plot(Gz_i[i][:, 4], grid, color=cols[i], label=season_labs[i])
    ax1.set_title('Mode 1')
    ax1.set_ylabel('Depth')
    ax1.grid()
    ax2.set_title('Mode 2')
    ax2.grid()
    ax3.set_title('Mode 3')
    ax3.grid()
    ax4.set_title('Mode 4')
    ax4.invert_yaxis()
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels, fontsize=10)
    plot_pro(ax4)

    # --- plot mode shapes with swath to show variability with N2
    G_min = np.nan * np.ones((len(grid), 4))
    G_max = np.nan * np.ones((len(grid), 4))
    Gz_min = np.nan * np.ones((len(grid), 4))
    Gz_max = np.nan * np.ones((len(grid), 4))
    for i in range(1, 4):
        G_per_tot = np.concatenate((G_i[0][:, i][:, None], G_i[1][:, i][:, None], G_i[2][:, i][:, None]), axis=1)
        G_min[:, i] = np.nanmin(G_per_tot, axis=1)
        G_max[:, i] = np.nanmax(G_per_tot, axis=1)
        Gz_per_tot = np.concatenate((Gz_i[0][:, i][:, None], Gz_i[1][:, i][:, None], Gz_i[2][:, i][:, None]), axis=1)
        Gz_min[:, i] = np.nanmin(Gz_per_tot, axis=1)
        Gz_max[:, i] = np.nanmax(Gz_per_tot, axis=1)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.fill_betweenx(grid, G_min[:, 1], G_max[:, 1], color='#708090')
    ax1.fill_betweenx(grid, G_min[:, 2], G_max[:, 2], color='#DAA520')
    ax1.fill_betweenx(grid, G_min[:, 3], G_max[:, 3], color='#00FA9A')
    ax1.set_ylim([0, 4750])
    ax1.set_ylabel('Depth [m]')
    ax1.set_title('Displacement Modes G(z)')
    ax1.set_xlabel('Normalized Mode Amplitude')
    ax1.invert_yaxis()
    ax1.grid()
    ax2.fill_betweenx(grid, Gz_min[:, 1], Gz_max[:, 1], color='#708090', label='Mode 1')
    ax2.fill_betweenx(grid, Gz_min[:, 2], Gz_max[:, 2], color='#DAA520', label='Mode 2')
    ax2.fill_betweenx(grid, Gz_min[:, 3], Gz_max[:, 3], color='#00FA9A', label='Mode 3')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=10)
    ax2.set_title("Velocity Modes G'(z)")
    ax2.set_xlabel('Normalized Mode Amplitude')
    ax2.grid()
    # f.savefig("/Users/jake/Documents/Conferences/USClivar_19/mode_shapes.jpeg", dpi=450)
    plot_pro(ax2)

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
colors = plt.cm.Dark2(np.arange(0, 4, 1))
f, (ax, ax2, ax25, ax3, ax4) = plt.subplots(1, 5, sharey=True)
for i in range(num_profs):
    ax.plot(this_anom[:, i], grid, linewidth=0.75)
    # ax2.plot(eta[:, i], grid, linewidth=.75, color='#808000')
    # ax25.plot(eta2[:, i], grid, linewidth=.75, color='#808000')
    ax2.plot(eta_alt_3[:, i], grid, linewidth=.75, color='#808000')
    ax25.plot(eta_alt[:, i], grid, linewidth=.75, color='#808000')
    ax3.plot(eta_alt_3[:, i], grid, linewidth=.75, color='#808000')
    ax3.plot(Eta_m[:, i], grid, linewidth=.5, color='k', linestyle='--')
n2p = ax4.plot((np.sqrt(N2_all) * (1800 / np.pi)) / 4, grid, color='k', label='N(z) [cph]')
for j in range(4):
    ax4.plot(G[:, j] / grid.max(), grid, color='#2F4F4F', linestyle='--')
    p_eof = ax4.plot(-EOFetashape[:, j] / grid.max(), grid, color=colors[j, :], label='EOF # = ' + str(j + 1),
                     linewidth=2.5)
ax.text(0.2, 4000, str(num_profs) + ' profiles', fontsize=10)
ax.grid()
ax.set_title('BATS Hydrography ' + str(np.int(np.round(c_date.min()))) + ' - ' + str(np.int(np.round(c_date.max()))))
ax.set_xlabel(r'$\sigma_{\theta} - \overline{\sigma_{\theta}}$')
ax.set_ylabel('Depth [m]')
ax.set_xlim([-.4, .4])
ax2.grid()
ax2.set_title('Dir. Search Gamma')
ax2.set_xlabel('[m]')
ax2.axis([-350, 350, 0, 4750])
ax25.grid()
ax25.set_title('ddz Gamma')
ax25.set_xlabel('[m]')
ax25.axis([-350, 350, 0, 4750])
ax3.set_title('Local Sig_theta / ddz')
ax3.set_xlabel('[m]')
ax3.axis([-350, 350, 0, 4750])
ax3.grid()

handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles, labels, fontsize=10)
ax4.set_title('EOFs of mode amp. (G(z))')
ax4.set_xlabel('Normalized Mode Amplitude')
ax4.set_xlim([-1, 1])
ax4.invert_yaxis()
plot_pro(ax4)

# --- individual eta look
these_profiles = d_summer[0:5]
f, ax = plt.subplots(1, 5, sharey=True)
for i in range(5):
    ax[i].plot(eta_alt_3[:, these_profiles[i]], grid, color='r', linewidth=0.75, label=r'$\gamma$ Avg,Dir')  # avg direct search, gamma
    ax[i].plot(-1 * eta_alt[:, these_profiles[i]], grid, color='b', linewidth=0.75, label=r'$\gamma$ Avg,ddz')  # avg divide by ddz, gamma
    ax[i].plot(-1 * this_eta_0[:, these_profiles[i]], grid, color='g', linewidth=0.75, label=r'$\sigma_{\theta0}$ Avg, ddz')  # avg divide by ddz, pot den, local pref
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

f_ref = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)
sc_x = 1000 * f_ref / c_i[2][1:]
# -- max min per season
sta_max = np.nan * np.ones((3, len(sc_x)))
sta_min = np.nan * np.ones((3, len(sc_x)))
for i in range(1, mmax+1):
    sta_max[0, i - 1] = np.nanstd(PE_per_mass[0][i, :])
    sta_min[0, i - 1] = np.nanstd(PE_per_mass[0][i, :])
    sta_max[1, i - 1] = np.nanstd(PE_per_mass[1][i, :])
    sta_min[1, i - 1] = np.nanstd(PE_per_mass[1][i, :])
    sta_max[2, i - 1] = np.nanstd(PE_per_mass[2][i, :])
    sta_min[2, i - 1] = np.nanstd(PE_per_mass[2][i, :])

PE_SD, PE_GM, GMPE, GMKE = PE_Tide_GM(1025, grid, nmodes, np.transpose(np.atleast_2d(np.nanmean(N2, axis=1))), f_ref)

# --- PLOT PE PER SEASON
# attempt to filter for anomalous profiles
f, ax = plt.subplots()
for i in range(slen):
    dk = f_ref / c_i[i][1]
    sc_x = 1000 * f_ref / c_i[i][1:]
    # ax.fill_between(1000 * f_ref / c_i[i][1:mmax + 1],
    #                 (np.nanmean(PE_per_mass[i][1:], axis=1) / dk) - (sta_min[i, :] / dk),
    #                 (np.nanmean(PE_per_mass[i][1:], axis=1) / dk) + (sta_max[i, :] / dk), color=cols[i], alpha=0.5)
    PE_per_good = (PE_per_mass[i][1, :] / dk) < (1 * 10 ** 3)
    print(PE_per_good)
    ax.plot(sc_x, np.nanmean(PE_per_mass[i][1:, PE_per_good], axis=1) / dk, color=cols[i], linewidth=3, label=season_labs[i])
    ax.scatter(sc_x, np.nanmean(PE_per_mass[i][1:, PE_per_good], axis=1) / dk, s=15, color=cols[i])
dk = f_ref / c_i[1][1]
ax.plot(sc_x, PE_GM / dk, color='k', linewidth=0.75, linestyle='--')

ax.plot([10**-2, 10**1], [10**4, 10**-5], color='k', linewidth=0.5)
ax.text(1.1*10**0, 1.4*10**-3, '-3', fontsize=10)

ax.plot([10**-2, 10**1], [10**3, 10**-3], color='k', linewidth=0.5)
ax.text(2*10**0, 5*10**-3, '-2', fontsize=10)
ax.set_yscale('log')
ax.set_xscale('log')
# ax.axis([10 ** -2, 10 ** 1, 10 ** (-4), 2 * 10 ** 3])
ax.axis([10 ** -2, 3 * 10 ** 0, 10 ** (-3), 10 ** 3])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=10)
ax.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
ax.set_ylabel('Spectral Density', fontsize=12)  # ' (and Hor. Wavenumber)')
ax.set_title('BATS Hydro. Seasonal PE Spectrum', fontsize=12)
plot_pro(ax)

# --- TIME SERIES OF ISOSPYCNAL DEPTH AND MODE AMPLITUDES
time_series = 1
if time_series > 0:
    # ISOPYCNAL DEPTH IN TIME
    # isopycnals I care about
    # rho1 = 27.0
    # rho2 = 27.8
    # rho3 = 28.05
    rho1 = 36.2
    rho2 = 35.8
    rho3 = 35.05

    import datetime
    import matplotlib

    d_time_per_prof = c_date.copy()
    d_time_per_prof_date = []
    d_dep_rho1 = np.nan * np.ones((3, len(d_time_per_prof)))
    for i in range(len(d_time_per_prof)):
        d_time_per_prof_date.append(datetime.date.fromordinal(np.int(d_time_per_prof[i])))
        d_dep_rho1[0, i] = np.interp(rho1, np.flipud(salin[:, i]), np.flipud(grid))
        d_dep_rho1[1, i] = np.interp(rho2, np.flipud(salin[:, i]), np.flipud(grid))
        d_dep_rho1[2, i] = np.interp(rho3, np.flipud(salin[:, i]), np.flipud(grid))

    # ------
    matplotlib.rcParams['figure.figsize'] = (10, 7)
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.scatter(d_time_per_prof, d_dep_rho1[0, :], color='g', s=15, label=r'CTD$_{ind}$')
    ax2.scatter(d_time_per_prof, d_dep_rho1[1, :], color='g', s=15)
    ax3.scatter(d_time_per_prof, d_dep_rho1[2, :], color='g', s=15)

    # smooth
    date_grid = np.arange(1990, 2015, 0.5)
    dep_rho1_smooth = np.nan * np.ones((3, len(date_grid)))
    for i in range(len(date_grid)):
        inn = np.where( (c_date > (date_grid[i] - 0.25)) & (c_date < (date_grid[i] + 0.25)) )[0]
        if len(inn) > 2:
            dep_rho1_smooth[0, i] = np.nanmean(d_dep_rho1[0, inn])
            dep_rho1_smooth[1, i] = np.nanmean(d_dep_rho1[1, inn])
            dep_rho1_smooth[2, i] = np.nanmean(d_dep_rho1[2, inn])
    ax1.plot(date_grid, dep_rho1_smooth[0, :], color='b', linewidth=0.75)
    ax2.plot(date_grid, dep_rho1_smooth[1, :], color='b', linewidth=0.75)
    ax3.plot(date_grid, dep_rho1_smooth[2, :], color='b', linewidth=0.75)

    ax1.set_title('Depth of $\gamma^{n}$ = ' + str(rho1))
    ax2.set_title('Depth of $\gamma^{n}$ = ' + str(rho2))
    ax3.set_title('Depth of $\gamma^{n}$ = ' + str(rho3))
    ax1.set_ylabel('Depth [m]')
    ax2.set_ylabel('Depth [m]')
    ax3.set_ylabel('Depth [m]')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=10)
    # ax1.plot([datetime.date(2015, 5, 31), datetime.date(2015, 5, 31)], [400, 900], color='b', linewidth=1.2)
    # ax1.plot([datetime.date(2015, 6, 1), datetime.date(2015, 6, 1)], [400, 900], color='r', linewidth=1.2)
    # ax1.plot([datetime.date(2015, 9, 15), datetime.date(2015, 9, 15)], [400, 900], color='r', linewidth=1.2)
    # ax1.plot([datetime.date(2015, 9, 16), datetime.date(2015, 9, 16)], [400, 900], color='b', linewidth=1.2)
    # ax1.set_ylim([500, 850])
    # ax2.set_ylim([1050, 1400])
    # ax3.set_ylim([2600, 2950])
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax1.grid()
    ax2.grid()
    # ax3.set_xlim([datetime.date(2015, 1, 1), datetime.date(2015, 12, 1)])
    ax3.set_xlabel('Date')
    plot_pro(ax3)

# -- MODE AMPLITUDE IN TIME
f, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
ax.plot(c_date, AG_all[1, :], color='g')
ax.scatter(c_date[d_winter1], AG_all[1, d_winter1], s=15)
ax.set_title('Displacement Mode 1 Amp. in Time')
ax2.plot(c_date, AG_all[2, :], color='g')
ax2.scatter(c_date[d_winter1], AG_all[2, d_winter1], s=15)
ax2.set_title('Displacement Mode 2 Amp. in Time')
ax3.plot(c_date, AG_all[3, :], color='g')
ax3.scatter(c_date[d_winter1], AG_all[3, d_winter1], s=15)
ax3.set_title('Displacement Mode 3 Amp. in Time')
ax4.plot(c_date, AG_all[4, :], color='g')
ax4.scatter(c_date[d_winter1], AG_all[4, d_winter1], s=15)
ax4.set_title('Displacement Mode 4 Amp. in Time')
ax.grid()
ax2.grid()
ax3.grid()
plot_pro(ax4)

# -- density in time
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(c_date, gamma[90, :], color='g')
# ax2.plot(c_date, gamma[165, :], color='g')
# ax1.set_title('Potential Density at 600m')
# ax2.set_title('at 3400m')
# ax2.set_xlabel('Time [Yr]')
# ax2.set_ylabel(r'$\gamma$ P$_{ref}$ = 600 m')
# ax1.set_ylabel(r'$\gamma$ P$_{ref}$ = 3400 m')
# ax1.grid()
# plot_pro(ax2)

# -- attempt fft to find period of oscillation
# window_size, poly_order = 9, 2
# Time_grid = np.arange(np.round(c_date.min()), np.round(c_date.max()), 1/12)
#
# order_0_AG = AG_all[1, :]
# y_AG_0 = savgol_filter(order_0_AG, window_size, poly_order)
# order_0_AG_grid = np.interp(Time_grid, c_date, y_AG_0)
#
# order_1_AG = AG_all[2, :]
# y_AGz_1 = savgol_filter(order_1_AG, window_size, poly_order)
# order_1_AG_grid = np.interp(Time_grid, c_date, y_AGz_1)
#
# N = len(order_0_AG_grid)
# T = Time_grid[1] - Time_grid[0]
# yf_0 = scipy.fftpack.fft(order_0_AG_grid)
# yf_1 = scipy.fftpack.fft(order_1_AG_grid)
# xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
# f, ax = plt.subplots()
# ax.plot(xf, 2.0 / N * np.abs(yf_0[:N // 2]), 'r')
# ax.plot(xf, 2.0 / N * np.abs(yf_1[:N // 2]), 'b')
# plot_pro(ax)

# --- SAVE
# write python dict to a file
sa = 0
if sa > 0:
    my_dict = {'depth': grid, 'Sigma0': sigma0, 'Sigma2': sigma2, 'lon': c_lon, 'lat': c_lat, 'time': c_date,
               'N2_per_season': N2, 'background indices': bckgrds, 'background order': ['spr', 'sum', 'fall', 'wint'],
               'AG': AG_all, 'AG_per_season': AG, 'Eta': eta, 'Eta_m': eta_m_all, 'NEta_m': Neta_m_all,
               'Eta2': eta2, 'Eta_m_2': eta_m_all_2, 'PE': PE_per_mass_all, 'PE_by_season': PE_per_mass, 'c': c}
    output = open('/Users/jake/Desktop/bats/station_bats_pe_jun04_19.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()


# -- look for long term warming/cooling trends
# f, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, sharex=True)
# m0 = np.polyfit(c_date, conservative_t_anom[0, :], 1)
# cta0 = np.polyval(m0, c_date)
# m1 = np.polyfit(c_date, conservative_t_anom[65, :], 1)
# cta1 = np.polyval(m1, c_date)
# m2 = np.polyfit(c_date, conservative_t_anom[115, :], 1)
# cta2 = np.polyval(m2, c_date)
# m3 = np.polyfit(c_date, conservative_t_anom[215, :], 1)
# cta3 = np.polyval(m3, c_date)
# ax0.plot(c_date, conservative_t_anom[0, :])
# ax1.plot(c_date, conservative_t_anom[65, :])
# ax2.plot(c_date, conservative_t_anom[115, :])
# ax3.plot(c_date, conservative_t_anom[215, :])
# ax0.plot(c_date, cta0, color='k')
# ax1.plot(c_date, cta1, color='k')
# ax2.plot(c_date, cta2, color='k')
# ax3.plot(c_date, cta3, color='k')
# ax0.set_title('T anom at 5m')
# ax0.set_ylim([-5, 5])
# ax0.grid()
# ax1.set_title('T anom at 1000m')
# ax1.grid()
# ax2.set_title('T anom at 2000m')
# ax2.grid()
# ax3.set_title('T anom at 4000m')
# ax3.set_xlabel('Date')
# plot_pro(ax3)
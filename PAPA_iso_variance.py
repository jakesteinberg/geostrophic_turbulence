import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pickle
import glob
import gsw
from scipy.signal import savgol_filter
from grids import make_bin_gen
from toolkit import plot_pro, nanseg_interp
from mode_decompositions import vertical_modes, vertical_modes_f, eta_fit, PE_Tide_GM

file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Line_P/canada_DFO/*.txt')

GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
bin_depth = GD.variables['grid'][0:-9]  # -24
z = -1 * bin_depth
ref_lat = 49.98
ref_lon = -144.98
grid_p = gsw.p_from_z(-1 * bin_depth, ref_lat)

# Data = []
# for m in range(len(file_list)):
#     this_file = file_list[m]
#     count_r = 0
#     f = open(this_file, encoding="ISO-8859-1")
#     initial = f.readlines()
#     for line in initial:  # loops over each row
#         by_line = line.strip().split("\t")
#         by_item = by_line[0].split()
#         if len(by_item) > 1:
#             item_test0 = by_item[0]
#             item_test1 = by_item[1]
#             item_test2 = by_item[-1]
#             if item_test0[0].isdigit() & item_test1[0].isdigit() & item_test2[0].isdigit():
#                 count = 0  # count for each column value
#                 data = np.nan * np.zeros((1, len(by_item)))  # one row's worth of data
#                 for i in by_item:  # each element in the row
#                     data[0, count] = np.float(i)  # data = one row's worth of data
#                     count = count + 1
#                 if count_r < 1:  # deal with first element of storage vs. all others
#                     data_out = data
#                 data_out = np.concatenate((data_out, data), axis=0)
#                 count_r = count_r + 1
#     Data.append(data_out)
#
# num_profiles = len(Data)
# temp_g = np.nan*np.ones((bin_depth.size, num_profiles))
# salin_g = np.nan*np.ones((bin_depth.size, num_profiles))
# max_d = np.nan*np.ones(num_profiles)
# for i in range(num_profiles):
#     this = Data[i]
#     depth = gsw.p_from_z(-1 * this[:, 0], ref_lat)
#     max_d[i] = np.nanmax(depth)
#     temp = this[:, 1]
#     salin = this[:, 2]
#     salin[np.where(salin < 0)[0]] = np.nan
#     temp_g[:, i], salin_g[:, i], bin_cen, deepest_bin = make_bin_gen(bin_depth, depth, temp, salin)
#     check_t = temp_g[:, i]
#     check_t[np.where(check_t < 0)[0]] = np.nan
#     check_s = salin_g[:, i]
#     check_s[np.where(check_s < 0)[0]] = np.nan
#
# my_dict = {'temp': temp_g, 'salin': salin_g}
# output = open('/Users/jake/Documents/baroclinic_modes/Line_P/canada_DFO/text_file_processing.pkl', 'wb')
# pickle.dump(my_dict, output)
# output.close()

# --- LOAD CASTS (something like 49) for
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Line_P/canada_DFO/text_file_processing.pkl', 'rb')
papa = pickle.load(pkl_file)
salin = papa['salin']
temp = papa['temp']
pkl_file.close()

num_profiles = temp.shape[1]

SA = np.nan * np.ones(np.shape(salin))
CT = np.nan * np.ones(np.shape(salin))
sig0 = np.nan * np.ones(np.shape(salin))
for i in range(num_profiles):
    SA[:, i] = gsw.SA_from_SP(salin[:, i], bin_depth,
                              ref_lon * np.ones(len(salin[:, i])), ref_lat * np.ones(len(salin[:, i])))
    CT[:, i] = gsw.CT_from_t(SA[:, i], temp[:, i], grid_p)
    sig0[:, i] = gsw.sigma0(SA[:, i], CT[:, i])

SA_avg = np.nanmean(SA, axis=1)
CT_avg = np.nanmean(CT, axis=1)
sig0_avg = np.nanmean(sig0, axis=1)
sig0_anom = sig0 - np.tile(sig0_avg[:, np.newaxis], (1, num_profiles))

ddz_avg_sigma = np.gradient(sig0_avg, z)
N2 = gsw.Nsquared(SA_avg, CT_avg, grid_p, lat=ref_lat)[0]
window_size = 15
poly_order = 3
N2 = savgol_filter(N2, window_size, poly_order)
N2 = np.concatenate(([0], N2))
# N2 is one value less than grid_p
# N2[N2 < 0] = np.nan
# N2 = nanseg_interp(grid, N2)
# N2 = np.append(N2, (N2[-1] - N2[-1]/20))

eta = sig0_anom / np.tile(ddz_avg_sigma[:, np.newaxis], (1, num_profiles))

# MODE PARAMETERS
# frequency zeroed for geostrophic modes
omega = 0
# highest baroclinic mode to be calculated
mmax = 60
nmodes = mmax + 1
eta_fit_dep_min = 50
eta_fit_dep_max = 4000

# -- computer vertical mode shapes
G, Gz, c, epsilon = vertical_modes(N2, bin_depth, omega, mmax)

# --- compute alternate vertical modes
bc_bot = 1  # 1 = flat, 2 = rough
grid2 = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4500, 10)])
n2_interp = np.interp(grid2, bin_depth, N2)
F_int_g2, F_g2, c_ff, norm_constant, epsilon2 = vertical_modes_f(n2_interp, grid2, omega, mmax, bc_bot)
F = np.nan * np.ones((np.size(bin_depth), mmax + 1))
F_int = np.nan * np.ones((np.size(bin_depth), mmax + 1))
for i in range(mmax + 1):
    F[:, i] = np.interp(bin_depth, grid2, F_g2[:, i])
    F_int[:, i] = np.interp(bin_depth, grid2, F_int_g2[:, i])

# -- project modes onto eta profiles
AG, eta_m, Neta_m, PE_per_mass = eta_fit(num_profiles, bin_depth, nmodes, N2, G, c, eta,
                                         eta_fit_dep_min, eta_fit_dep_max)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
for i in range(num_profiles):
    ax1.plot(sig0_anom, bin_depth)
    ax2.plot(eta[:, i], bin_depth)
    ax2.plot(eta_m[:, i], bin_depth, 'k', linewidth=0.5)
ax2.invert_yaxis()
ax1.set_xlim([-.5, .5])
ax2.set_xlim([-600, 600])
ax1.grid()
plot_pro(ax2)

# --- ENERGY parameters
rho0 = 1027
f_ref = np.pi * np.sin(np.deg2rad(ref_lat)) / (12 * 1800)
dk = f_ref / c[1]
sc_x = 1000 * f_ref / c[1:]
vert_wavenumber = f_ref / c[1:]
dk_ke = 1000 * f_ref / c[1]
PE_SD, PE_GM = PE_Tide_GM(rho0, bin_depth, nmodes, np.transpose(np.atleast_2d(N2)), f_ref)
avg_PE = np.nanmean(PE_per_mass, 1)

# ----- energy spectra (comparative plots)
f, ax = plt.subplots()
mode_num = np.arange(1, 61, 1)
# PAPA
PE_p = ax.plot(mode_num, avg_PE[1:] / dk, color='#B22222', label='APE$_{PAPA}$', linewidth=2)  # sc_x

ax.set_yscale('log')
ax.set_xscale('log')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=14)
# ax.axis([10 ** -2, 10 ** 1, 10 ** (-4), 10 ** 3])
ax.axis([8 * 10 ** -1, 10 ** 2, 3 * 10 ** (-4), 10 ** 3])
ax.set_xlabel('Mode Number', fontsize=14)
ax.set_ylabel('Spectral Density', fontsize=18)  # ' (and Hor. Wavenumber)')
ax.set_title('PAPA Hydrography PE', fontsize=20)
plot_pro(ax)

# -- mode interactions
cmap = matplotlib.cm.get_cmap('Greys')
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.pcolor(epsilon2[0, :, :], cmap=cmap, vmin=0, vmax=3)
ax1.set_title('mode 0')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
ax2.pcolor(epsilon2[1, :, :], cmap=cmap, vmin=0, vmax=3)
ax2.set_title('mode 1')
plt.xticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))
plt.yticks(np.arange(0.5, 3.5, 1), ('0', '1', '2'))

c_map_ax = f.add_axes([0.92, 0.1, 0.02, 0.8])
norm = matplotlib.colors.Normalize(vmin=0, vmax=4)
cb1 = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=cmap, norm=norm, orientation='vertical')
cb1.set_label('Epsilon')
ax2.grid()
plot_pro(ax2)

f, ax = plt.subplots()
ax.plot(Gz[:, 0], bin_depth, linestyle='--', color='r')
ax.plot(Gz[:, 1], bin_depth, linestyle='--', color='b')
ax.plot(Gz[:, 2], bin_depth, linestyle='--', color='g')
ax.plot(Gz[:, 3], bin_depth, linestyle='--', color='m')
ax.plot(N2 * 30000, bin_depth, color='k')
ax.plot(F[:, 0], bin_depth, color='r')
ax.plot(F[:, 1], bin_depth, color='b')
ax.plot(F[:, 2], bin_depth, color='g')
ax.plot(F[:, 3], bin_depth, color='m')
ax.invert_yaxis()
plot_pro(ax)

# --- SAVE
# write python dict to a file
sa = 0
if sa > 0:
    my_dict = {'depth': bin_depth, 'Sigma0': sig0, 'N2': N2, 'AG_per_season': AG, 'Eta': eta, 'PE': PE_per_mass, 'c': c}
    output = open('/Users/jake/Documents/baroclinic_modes/Line_P/canada_DFO/papa_energy_spectra_jun13.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()

# with open(this_file, encoding="ISO-8859-1") as f:
#     for _ in range(112):
#         next(f)
#     a = f.strip().split("\t")
#     for line in f:
#         test0 = line.strip().split("\t")
#         test1 = test0[0].split()
#         count = 0
#         data = np.nan * np.zeros((1, len(test1)))
#         for i in test1:
#             data[0, count] = np.float(i)  # data = one row's worth of data
#             count = count + 1
#
#         if count0 < 1:  # deal with first element of storage vs. all others
#             data_out = data
#         data_out = np.concatenate((data_out, data), axis=0)
#         count0 = count0 + 1
# Data.append(data_out)

# for l in open(file_list[20], encoding="ISO-8859-1"):
#     test0 = l.strip().split("\t")
#     # tt = text_file.read().strip().split('\t')
#     test1 = test0[0].split()
#     if len(test1) > 6:
#         if count0 < 1:  # deal with first element of storage vs. all others
#             data = np.nan * np.zeros(len(test1))
#         count = 0
#         for i in test1:
#             data[count] = np.float(i)  # data = one row's worth of data
#             count = count + 1
#     count0 = count0 + 1



import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from glider_cross_section import Glider
from mode_decompositions import vertical_modes
from toolkit import plot_pro

# DIVE SELECTION
# -----------------------------------------------------------------------------------------
#        SG NUMBER,  DIVE NUMBERS,                DIVE FILE PATH
# -----------------------------------------------------------------------------------------
# ---- DG WA Coast
# x = Glider(30, np.arange(53, 65), '/Users/jake/Documents/Cuddy_tailored/DG_wa_coast/2006')
# x = Glider(30, np.arange(46, 50), '/Users/jake/Documents/Cuddy_tailored/DG_wa_coast/2006')
# ---- DG ABACO 2017
# x = Glider(38, np.arange(67, 78), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/sg038')
# ---- DG ABACO 2018
# x = Glider(37, np.concatenate((np.arange(69, 81), np.arange(82, 84))), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2018/sg037')
# x = Glider(39, np.arange(20, 86), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2018/sg039')  # 56-70
# x = Glider(39, np.concatenate((np.arange(70, 83), np.arange(84, 86))),
#            r'/Users/jake/Documents/baroclinic_modes/DG/ABACO_2018/sg039')
# ---- DG BATS 2014
# x = Glider(35, np.concatenate((np.arange(123, 126), np.array([127]))), '/Users/jake/Documents/baroclinic_modes/DG/BATS_2014/sg035')
# x = Glider(35, np.arange(123, 126), '/Users/jake/Documents/baroclinic_modes/DG/BATS_2014/sg035')
# ---- DG BATS 2015
x = Glider(35, np.arange(132, 136), '/Users/jake/Documents/baroclinic_modes/DG/BATS_2015/sg035')  # 114,117
# ---- DG BATS 2018
# x = Glider(41, np.arange(158, 162), '/Users/jake/Documents/baroclinic_modes/DG/BATS_2018/sg041')

# -- match max dive depth to bin_depth
# GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
# deepest_bin_i = find_nearest(GD.variables['grid'][:], x.avg_dep)[0]
# bin_depth = GD.variables['grid'][0:deepest_bin_i]
# bin_depth = np.concatenate((np.arange(0, 300, 5), np.arange(300, 1000, 10), np.arange(1000, 2700, 20)))  # deeper
# bin_depth = np.concatenate((np.arange(0, 300, 5), np.arange(300, 1000, 10), np.arange(1000, 1780, 20)))  # shallower
# bin_depth = np.concatenate((np.arange(0, 300, 5), np.arange(300, 1000, 10), np.arange(1000, 4710, 20)))  # shallower
bin_depth = np.concatenate((np.arange(0, 300, 5), np.arange(300, 1000, 10), np.arange(1000, 5000, 20)))  # 36N

# -----------------------------------------------------------------------------------------------
# vertical bin averaging separation of dive-climb cycle into two profiles (extractions from nc files)
Binned = x.make_bin(bin_depth)
d_time = Binned['time']
lon = Binned['lon']
lat = Binned['lat']
ref_lat = np.nanmean(lat)
t = Binned['temp']
s = Binned['sal']
dac_u = Binned['dac_u']
dac_v = Binned['dac_v']
profile_tags = Binned['profs']
if 'o2' in Binned.keys():
    o2 = Binned['o2']

# -----------------------------------------------------------------------------------------------
# try to make sg030 dive 60 better (interpolate across adjacent dives to fill in salinity values)
if np.int(x.ID[3:]) == 30:
    check = np.where(x.dives == 60)[0]
    if np.size(check) > 0:
        bad = np.where(np.isnan(s[:, 2 * check[0] + 1]))[0]
        for i in bad:
            this = 2 * check[0] + 1
            dx1 = 1.852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon[i, this] - lon[i, this - 1])  # zonal sfc disp [km]
            dy1 = 1.852 * 60 * (lat[i, this] - lat[i, this - 1])  # meridional sfc disp [km]
            dist1 = np.sqrt(dx1**2 + dy1**2)
            dx2 = 1.852 * 60 * np.cos(np.deg2rad(ref_lat)) * (lon[i, this + 1] - lon[i, this])  # zonal sfc disp [km]
            dy2 = 1.852 * 60 * (lat[i, this + 1] - lat[i, this])  # meridional sfc disp [km]
            dist2 = np.sqrt(dx2**2 + dy2**2)
            s[i, 2 * check[0] + 1] = np.interp(np.array([0]),
                                               np.array([dist1, dist2]), np.array([s[i, this - 1], s[i, this + 1]]))

# -----------------------------------------------------------------------------------------------
# computation of absolute salinity, conservative temperature, and potential density anomaly
sa, ct, theta, sig0, sig2, N2 = x.density(bin_depth, ref_lat, t, s, lon, lat)
# -----------------------------------------------------------------------------------------------
# compute M/W sections and compute velocity
sigth_levels = np.concatenate(
    [np.arange(23, 25.4, 0.4), np.arange(26.2, 27.2, 0.2),
     np.arange(27.2, 27.8, 0.2), np.arange(27.72, 27.8, 0.02), np.arange(27.8, 27.9, 0.01)])
# sigth_levels = np.concatenate(
#     [np.arange(23, 26.5, 0.5), np.arange(26.2, 27.2, 0.2),
#      np.arange(27.2, 27.7, 0.2), np.arange(27.7, 28, 0.02), np.arange(28, 28.15, 0.01)])
# --- for combined set of transects
# ds, dist, avg_ct_per_dep_0, avg_sa_per_dep_0, avg_sig0_per_dep_0, v_g, vbt, \
# isopycdep, isopycx, mwe_lon, mwe_lat, DACe_MW, DACn_MW, profile_tags_per = \
#     x.transect_cross_section_1(bin_depth, sig0, ct, sa, lon, lat, dac_u, dac_v, profile_tags, sigth_levels)
# use this one
# ds, dist, avg_ct_per_dep_0, avg_sa_per_dep_0, avg_sig0_per_dep_0, v_g, vbt, \
# isopycdep, isopycx, mwe_lon, mwe_lat, DACe_MW, DACn_MW, profile_tags_per, shear, v_g_east, v_g_north = \
#     x.transect_cross_section_1(bin_depth, sig0, ct, sa, lon, lat, dac_u, dac_v, profile_tags, sigth_levels)
# --- for single transects
ds, dist, v_g, vbt, isopycdep, isopycx, mwe_lon, mwe_lat, DACe_MW, DACn_MW, profile_tags_per = \
    x.transect_cross_section_0(bin_depth, sig0, lon, lat, dac_u, dac_v, profile_tags, sigth_levels)

# -----------------------------------------------------------------------------------------------
# PLOTTING cross section
u_levels = np.arange(-.4, .44, .04)
# choose which transect
# transect_no = 0
# x.plot_cross_section(bin_depth, ds[transect_no], v_g[transect_no], dist[transect_no],
#                      profile_tags_per[transect_no], isopycdep[transect_no], isopycx[transect_no],
#                      sigth_levels, d_time, u_levels)
fig0 = x.plot_cross_section(bin_depth, ds, v_g, dist, profile_tags_per, isopycdep, isopycx, sigth_levels, d_time, u_levels)

# fig0.savefig("/Users/jake/Documents/glider_flight_sim_paper/sample_cross.jpeg", dpi=300)
# -----------------------------------------------------------------------------------------------
# plot plan view
# load in bathymetry and lat/lon plotting bounds
# WA COAST
# bathy_path = '/Users/jake/Documents/Cuddy_tailored/DG_wa_coast/smith_sandwell_wa_coast.nc'
# plan_window = [-128.5, -123.75, 46.5, 48.5]
# ABACO
# bathy_path = '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/OceanWatch_smith_sandwell.nc'
# plan_window = [-77.5, -73.5, 25.5, 27]
# BATS
# bathy_path = '/Users/jake/Desktop/bats/bats_bathymetry/GEBCO_2014_2D_-67.7_29.8_-59.9_34.8.nc'
bathy_path = '/Users/jake/Desktop/bats/bats_bathymetry/bathymetry_b38e_27c7_f8c3_f3d6_790d_30c7.nc'
plan_window = [-66, -63, 31, 33]
# plan_window = [-66, -63, 32, 37]
# bath_fid = Dataset(bathy_path, 'r')

# from netCDF4 import Dataset
# bath_fid = Dataset(bathy_path, 'r')

# --- for combined set of transects ---
# x.plot_plan_view(lon, lat, mwe_lon[transect_no], mwe_lat[transect_no],
#                  DACe_MW[transect_no], DACn_MW[transect_no],
#                  ref_lat, profile_tags_per[transect_no], d_time, plan_window, bathy_path)
# --- for single transect ---
fig1 = x.plot_plan_view(lon, lat, mwe_lon, mwe_lat, DACe_MW, DACn_MW,
                 ref_lat, profile_tags_per, d_time, plan_window, bathy_path)

# plot t/s
# x.plot_t_s(ct, sa)

# fig1.savefig("/Users/jake/Documents/glider_flight_sim_paper/sample_plan.jpeg", dpi=300)
# -------------------
# vertical modes
# N2_avg = np.nanmean(N2, axis=1)
# N2_avg[N2_avg < 0] = 0
# N2_avg[0] = 1
# G, Gz, c, epsilon = vertical_modes(N2_avg, bin_depth, 0, 30)


# ____________________________________________________________________________________________________________________
# TESTING OF TRANSECT SEPARATION AND GROUPING
# def group_consecutives(vals, step=1):
#     """Return list of consecutive lists of numbers from vals (number list)."""
#     run = []
#     result = [run]
#     expect = None
#     for v in vals:
#         if (v == expect) or (expect is None):
#             run.append(v)
#         else:
#             run = [v]
#             result.append(run)
#         expect = v + step
#     return result
#
# # separate dives into unique transects
# target_test = 1000000 * x.target[:, 0] + np.round(x.target[:, 1], 3)
# unique_targets = np.unique(target_test)
# transects = []
# for m in range(len(unique_targets)):
#     indices = np.where(target_test == unique_targets[m])[0]
#     if len(indices) > 1:
#         transects.append(group_consecutives(indices, step=1))

# ds_out = []
# dist_out = []
# v_g_out = []
# vbt_out = []
# isopycdep_out = []
# isopycx_out = []
# mwe_lon_out = []
# mwe_lat_out = []
# DACe_MW_out = []
# DACn_MW_out = []
# profile_tags_out = []
# for n in range(len(transects)):  # loop over all transect segments
#     for o in range(len(transects[n])):  # loop over all times a glider executed that segment
#         this_transect = transects[n][o]
#         index_start = 2 * this_transect[0]
#         index_end = 2 * (this_transect[-1] + 1)
#         order_set = np.arange(0, 2 * len(this_transect), 2)
#         sig0_t = sig0[:, index_start:index_end]
#         lon_t = lon[:, index_start:index_end]
#         lat_t = lat[:, index_start:index_end]
#         dac_u_t = dac_u[index_start:index_end]
#         dac_v_t = dac_v[index_start:index_end]
#         profile_tags_t = profile_tags[index_start:index_end]

# _____________________________________________________________________________________________________________________
# COMPARISON OF O2
o2_comp = 0
if o2_comp > 0:
    file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Shipboard/ABACO/ship*.pkl')
    s_o2 = []
    s_sa = []
    s_ct = []
    s_dep = []
    f, ax = plt.subplots()
    for i in range(len(file_list)):
        pkl_file = open(file_list[i], 'rb')
        ship_adcp = pickle.load(pkl_file)
        pkl_file.close()
        s_dep.append(ship_adcp['bin_depth'])
        s_o2.append(ship_adcp['oxygen1'])
        s_ct.append(ship_adcp['CT'])
        s_sa.append(ship_adcp['SA'])
        for j in range(s_o2[i].shape[1]):
            if ship_adcp['den_dist'][i][j] < 600:  #  & (np.nanmax(s_o2[i][:, j]) > 240):
                ax.plot(s_o2[i][:, j], s_dep[i], color='k', label='Cast', linewidth=1.5)
            # if (ship_adcp['den_dist'][i][j] < 400) & (ship_adcp['den_dist'][i][j] > 50):
            #     ax.plot(s_sa[i][:, j], s_ct[i][:, j], color='k', label='Cast', linewidth=1.5)
    for k in range(len(profile_tags)):
        # ax.plot(o2[:, k], bin_depth, color='g', linewidth=0.6, label='Glider')
        ax.plot(sa[:, k], ct[:, k], color='g', linewidth=0.6, label='Glider')
    ax.set_xlim([120, 275])
    # ax.set_xlim([35, 35.5])
    # ax.set_ylim([1, 11])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0], handles[-1]], [labels[0], labels[-1]], fontsize=12)
    # ax.set_xlabel('Dissolved Oxygen [$\mu$mol / kg]')
    # ax.set_xlabel('SA')
    # ax.set_ylabel('CT')
    ax.set_title('T/S Comparison: ' + str(x.ID) + 'dives:' + str(x.dives[0]) + ':' + str(x.dives[-1]) + ', and ABACO Casts (2014-2017)')
    ax.invert_yaxis()
    ax.grid()
    plot_pro(ax)

# vehicle pitch: 'eng_pitchAng'
# DAC_u: 'depth_avg_curr_east'
# DAC_v: 'depth_avg_curr_north'

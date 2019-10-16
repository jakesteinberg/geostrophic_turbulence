# DG along track isopycnal variance 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
from netCDF4 import Dataset
import glob
import gsw
import pandas as pd
import scipy.io as si
import pickle
from scipy.optimize import fmin
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec
# functions I've written
from glider_cross_section import Glider
from mode_decompositions import vertical_modes, vertical_modes_f, PE_Tide_GM
from toolkit import spectrum_fit, plot_pro, find_nearest

# -------------------------------------------------------------------------------------------------------------------
# --- BATHYMETRY
bath = '/Users/jake/Desktop/abaco/abaco_bathymetry/GEBCO_2014_2D_-79.275_22.25_-67.975_29.1.nc'
bath_fid = Dataset(bath, 'r')
bath_lon = bath_fid['lon'][:]
bath_lat = bath_fid['lat'][:]
bath_z = bath_fid['elevation'][:]
# -------------------------------------------------------------------------------------------------
# USING GLIDER PACKAGE
# x = Glider(37, np.arange(45, 80), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/sg037')
# x = Glider(37, np.arange(50, 88), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2018/sg037')
x = Glider(39, np.concatenate((np.arange(18, 62), np.arange(63, 94))),
           r'/Users/jake/Documents/baroclinic_modes/DG/ABACO_2018/sg039')
# -------------------------------------------------------------------------------------------------------------------
# --- LOAD ABACO SHIPBOARD CTD DATA
ship_files = glob.glob('/Users/jake/Documents/baroclinic_modes/SHIPBOARD/ABACO/ship_ladcp*.pkl')
# all shipboard casts
for i in range(len(ship_files)):
    pkl_file = open(ship_files[i], 'rb')
    abaco_ship = pickle.load(pkl_file)
    pkl_file.close()
    if i < 1:
        time_key = np.int(ship_files[i][-14:-10]) * np.ones(abaco_ship['oxygen'].shape[1])
        ship_o2_1 = abaco_ship['oxygen']
        ship_SA = abaco_ship['SA']
        ship_CT = abaco_ship['CT']
        ship_sig0 = abaco_ship['den_grid']
        ship_dist = abaco_ship['den_dist']
        ship_lon = abaco_ship['cast_lon']
        ship_lat = abaco_ship['cast_lat']
        adcp_depth = abaco_ship['adcp_depth']
        adcp_dist = abaco_ship['adcp_dist']
        adcp_v = abaco_ship['adcp_v']
        adcp_time = abaco_ship['time_uv']
    else:
        time_key = np.concatenate((time_key, np.int(ship_files[i][-14:-10]) * np.ones(abaco_ship['oxygen'].shape[1])))
        ship_o2_1 = np.concatenate((ship_o2_1, abaco_ship['oxygen']), axis=1)
        ship_SA = np.concatenate((ship_SA, abaco_ship['SA']), axis=1)
        ship_CT = np.concatenate((ship_CT, abaco_ship['CT']), axis=1)
        ship_sig0 = np.concatenate((ship_sig0, abaco_ship['den_grid']), axis=1)
        ship_dist = np.concatenate((ship_dist, abaco_ship['den_dist']))
        ship_lon = np.concatenate((ship_lon, abaco_ship['cast_lon']))
        ship_lat = np.concatenate((ship_lat, abaco_ship['cast_lat']))
        adcp_dist = np.concatenate((adcp_dist, abaco_ship['adcp_dist']))
        adcp_v = np.concatenate((adcp_v, abaco_ship['adcp_v']), axis=1)
        adcp_time = np.concatenate((adcp_time, abaco_ship['time_uv']))

ship_depth_0 = abaco_ship['bin_depth']
ship_depth = np.repeat(np.transpose(np.array([abaco_ship['bin_depth']])), np.shape(ship_sig0)[1], axis=1)

pkl_file = open('/Users/jake/Documents/baroclinic_modes/SHIPBOARD/ABACO/no_ctd_ship_ladcp_2018-02-25.pkl', 'rb')
abaco_ship = pickle.load(pkl_file)
pkl_file.close()
adcp_dist = np.concatenate((adcp_dist, abaco_ship['adcp_dist']))
adcp_v = np.concatenate((adcp_v, abaco_ship['adcp_v']), axis=1)
adcp_time = np.concatenate((adcp_time, abaco_ship['time_uv']))

# --------
# single year analysis
pkl_file = open('/Users/jake/Documents/baroclinic_modes/SHIPBOARD/ABACO/ship_ladcp_2017-05-08.pkl', 'rb')
abaco_ship = pickle.load(pkl_file)
pkl_file.close()
this_ship_sig0 = abaco_ship['den_grid']
this_ship_CT = abaco_ship['CT']
this_ship_SA = abaco_ship['SA']
this_ship_dist = abaco_ship['den_dist']
this_ship_depth_0 = abaco_ship['bin_depth']
this_ship_depth = np.repeat(np.transpose(np.array([abaco_ship['bin_depth']])), np.shape(ship_sig0)[1], axis=1)
# pkl_file = open('/Users/jake/Documents/baroclinic_modes/SHIPBOARD/ABACO/no_ctd_ship_ladcp_2018-02-25.pkl', 'rb')
# abaco_ship = pickle.load(pkl_file)
# pkl_file.close()
this_adcp_dist = abaco_ship['adcp_dist']
this_adcp_depth = abaco_ship['adcp_depth']
this_adcp_v = abaco_ship['adcp_v']
# -------------------------------------------------------------------------------------------------------------------
# --- LOAD NEARBY ARGO CTD DATA
pkl_file = open('/Users/jake/Desktop/argo/deep_argo_nwa.pkl', 'rb')
abaco_argo = pickle.load(pkl_file)
pkl_file.close()
# -------------------------------------------------------------------------------------------------------------------
# --- GRID / SPATIAL PARAMETERS
ref_lat = 26.5
ref_lon = -76
lat_in = 26.5
lon_in = -77
GD = Dataset('/Users/jake/Documents/geostrophic_turbulence/BATs_2015_gridded_apr04.nc', 'r')
# grid = GD.variables['grid'][:]
grid = np.concatenate([GD.variables['grid'][:], np.arange(GD.variables['grid'][:][-1] + 20, 4700, 20)])
# bin_depth = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4750, 20)])
# grid = bin_depth[1:-1]
grid_p = gsw.p_from_z(-1 * grid, ref_lat)
den_grid = np.arange(24.5, 28, 0.02)
dist_grid_s = np.arange(2, 125, 0.005)
dist_grid = np.arange(20, 320, 10)

# output dataframe 
df_t = pd.DataFrame()
df_s = pd.DataFrame()
df_d = pd.DataFrame()
df_den = pd.DataFrame()
time_rec = []
time_rec_2 = np.zeros([len(x.files), 2])

# plot controls 
plot_plan = 0
plot_cross = 0
plot_eta = 0
plot_eng = 0
# -------------------------------------------------------------------------------------------------------------------
# prep plan view plot of dive locations
# if plot_plan > 0:
#     levels = [-5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0]
#     fig0, ax0 = plt.subplots()
#     cmap = plt.cm.get_cmap("Blues_r")
#     cmap.set_over('#808000')  # ('#E6E6E6')
#     bc = ax0.contourf(bath_lon, bath_lat, bath_z, levels, cmap='Blues_r', extend='both', zorder=0)
#     # ax0.contourf(bath_lon,bath_lat,bath_z,[0, 100, 200], cmap = 'YlGn_r')
#     matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
#     bcl = ax0.contour(bath_lon, bath_lat, bath_z, [-4500, -1000], colors='k', zorder=0)
#     ml = [(-76.75, 26.9), (-77.4, 26.8)]
#     ax0.clabel(bcl, manual=ml, inline_spacing=-3, fmt='%1.0f', colors='k')
#     w = 1 / np.cos(np.deg2rad(26.5))
#     ax0.axis([-77.5, -75, 25.75, 27.25])
#     ax0.set_aspect(w)
#     divider = make_axes_locatable(ax0)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     fig0.colorbar(bc, cax=cax, label='[m]')
#     ax0.set_xlabel('Longitude')
#     ax0.set_ylabel('Latitude')
# -------------------------------------------------------------------------------------------------
# Vertically Bin
Binned = x.make_bin(grid)
d_time = Binned['time']
lon = Binned['lon']
lat = Binned['lat']
t = Binned['temp']
s = Binned['sal']
dac_u = Binned['dac_u']
dac_v = Binned['dac_v']
profile_tags = Binned['profs']
if 'o2' in Binned.keys():
    o2 = Binned['o2']
ref_lat = np.nanmean(lat)
# -------------------------------------------------------------------------------------------------
# Compute density
sa, ct, theta, sig0, sig2, dg_N2 = x.density(grid, ref_lat, t, s, lon, lat)
# -----------------------------------------------------------------------------------------------
# compute M/W sections and compute velocity
# USING X.TRANSECT_CROSS_SECTION_1 (THIS WILL SEPARATE TRANSECTS BY TARGET OF EACH DIVE)
sigth_levels = np.concatenate(
    [np.arange(23, 26.5, 0.5), np.arange(26.2, 27.2, 0.2),
     np.arange(27.2, 27.8, 0.2), np.arange(27.7, 27.8, 0.02), np.arange(27.8, 27.9, 0.01)])
partial = 0
bbi = [0, 0, 0]
ds, dist, avg_ct_out, avg_sa_out, avg_sig0_per_dep_0, v_g, vbt, isopycdep, isopycx, mwe_lon, mwe_lat, DACe_MW, \
    DACn_MW, profile_tags_per, shear, box_side, v_g_east, v_g_north = x.transect_cross_section_1(grid, sig0, ct, sa,
                                                                                                 lon, lat,
                                                                                                 dac_u, dac_v,
                                                                                                 profile_tags,
                                                                                                 sigth_levels,
                                                                                                 partial, bbi)
# -----------------------------------------------------------------------------------------------
# PLOTTING cross section
# choose which transect
transect_no = 3
# x.plot_cross_section(grid, ds[transect_no], v_g[transect_no], dist[transect_no],
#                      profile_tags_per[transect_no], isopycdep[transect_no], isopycx[transect_no],
#                      sigth_levels, d_time)
# -----------------------------------------------------------------------------------------------
# PLOTTING PLAN VIEW
# ABACO
bathy_path = '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/OceanWatch_smith_sandwell.nc'
# x.plot_plan_view(lon, lat, mwe_lon[transect_no], mwe_lat[transect_no],
#                  DACe_MW[transect_no], DACn_MW[transect_no],
#                  ref_lat, profile_tags_per[transect_no], d_time, [-77.5, -73.5, 25.5, 27], bathy_path)

# -----------------------------------------------------------------------------------------------
# unpack velocity profiles from transect analysis
dg_v_0 = v_g[0][:, 0:-1].copy()
avg_sig0_per_dep = avg_sig0_per_dep_0[0].copy()
dg_v_lon = mwe_lon[0][0:-1].copy()
dg_v_lat = mwe_lat[0][0:-1].copy()
dg_v_dive_no = profile_tags_per[0][0:-1].copy()
for i in range(1, len(v_g)):
    dg_v_0 = np.concatenate((dg_v_0, v_g[i][:, 0:-1]), axis=1)
    avg_sig0_per_dep = np.concatenate((avg_sig0_per_dep, avg_sig0_per_dep_0[i]), axis=1)
    dg_v_lon = np.concatenate((dg_v_lon, mwe_lon[i][0:-1]))
    dg_v_lat = np.concatenate((dg_v_lat, mwe_lat[i][0:-1]))
    dg_v_dive_no = np.concatenate((dg_v_dive_no, profile_tags_per[i][0:-1]))

# ---------------------------------------------------------------------------------------------------------
sz = sig0.shape
time_rec = np.nanmean(d_time, axis=0)
num_profs = sz[1]
time_min = np.min(time_rec)
time_max = np.max(time_rec)
time_mid = 10.5 * (time_max - time_min) + time_min

# compute average density/temperature as a function of distance offshore
# need a reference point (seems to make sense to plot everything as a function of longitude)
# all profiles
df_d = (lon - lon_in) * (1852 * 60 * np.cos(np.deg2rad(26.5))) / 1000
# eta / v profiles after M/W technique applied
dg_v_dist = (dg_v_lon - lon_in) * (1852 * 60 * np.cos(np.deg2rad(26.5))) / 1000

# ---------------------------------------------------------------------------------------------------------
# USING GLIDER PROFILES COMPUTE AVERATE T/S/RHO AT INCREASING EASTWARD DISTANCE
count = 0
# uses glider profiles that have not been processed as M/W
mean_dist = np.nanmean(df_d, 0)
profs_per_avg = np.zeros(np.shape(dist_grid))
CT_avg_grid = np.zeros([np.size(grid), np.size(dist_grid)])
SA_avg_grid = np.zeros([np.size(grid), np.size(dist_grid)])
sigma_avg_grid = np.zeros([np.size(grid), np.size(dist_grid)])
for i in dist_grid:
    mask = (mean_dist > i - 10) & (mean_dist < i + 10)
    profs_per_avg[count] = np.sum(mask)
    CT_avg_grid[:, count] = np.nanmean(ct[:, mask], 1)  # np.nanmean(df_t[df_t.columns[mask]], 1)
    SA_avg_grid[:, count] = np.nanmean(sa[:, mask], 1)  # np.nanmean(df_s[df_s.columns[mask]], 1)
    sigma_avg_grid[:, count] = np.nanmean(sig0[:, mask], 1)  # np.nanmean(df_den[df_den.columns[mask]], 1)
    count = count + 1

# ---------------------------------------------------------------------------------------------------------
import_dg = si.loadmat('/Users/jake/Documents/geostrophic_turbulence/ABACO_dg_transect_9.mat')
dg_data = import_dg['out_t']
# ---------------------------------------------------------------------------------------------------------
# --- mean density profile as a function of distance offshore (avg profile changes (and represents the linear trend))
c2s = plt.cm.jet(np.linspace(0, 1, 33))
c2_ship = plt.cm.jet(np.linspace(0, 1, 11))
if plot_cross > 0:
    # plan view and comparison across platforms
    gs = gridspec.GridSpec(3, 4)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1:, 0:2])
    ax3 = plt.subplot(gs[1:, 2:])
    levels = [-5250, -5000, -4750, -4500, -4250, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0]
    cmap = plt.cm.get_cmap("Blues_r")
    cmap.set_over('#808000')  # ('#E6E6E6')
    bc = ax1.contourf(bath_lon, bath_lat, bath_z, levels, cmap='Blues_r', extend='both', zorder=0)
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    bcl = ax1.contour(bath_lon, bath_lat, bath_z, [-4500, -1000], colors='k', zorder=0)
    ml = [(-76.75, 26.9), (-77.4, 26.8)]
    ax1.clabel(bcl, manual=ml, inline_spacing=-3, fmt='%1.0f', colors='k')
    w = 1 / np.cos(np.deg2rad(26.5))
    ax1.axis([-77.25, -74, 26.25, 26.75])
    ax1.set_aspect(w)
    for i in x.files:
        nc_fid = Dataset(i, 'r')
        glid_num = nc_fid.glider
        dive_num = nc_fid.dive_number
        lat = nc_fid.variables['latitude'][:]
        lon = nc_fid.variables['longitude'][:]
        ax1.scatter(lon, lat, s=1, color='#A0522D')
    ax1.plot(dg_data['dac_lon'][0][0], dg_data['dac_lat'][0][0], color='y', linewidth=2)
    ax1.text(-75.3, 26.66, 'DG38: 67-77', fontsize=12, color='y')
    ax1.scatter(dg_data['dac_lon'][0][0], dg_data['dac_lat'][0][0], s=15, color='y')
    ax1.quiver(dg_data['dac_lon'][0][0], dg_data['dac_lat'][0][0], dg_data['dac_u'][0][0], dg_data['dac_v'][0][0],
               color='y', scale=2, headwidth=2, headlength=3, width=.005)
    ax1.scatter(abaco_ship['adcp_lon'], abaco_ship['adcp_lat'], s=25, color='#7CFC00')
    ax1.quiver(abaco_ship['adcp_lon'], abaco_ship['adcp_lat'], np.nanmean(abaco_ship['adcp_u'] / 100, axis=0),
               np.nanmean(abaco_ship['adcp_v'] / 100, axis=0), color='#7CFC00', scale=2,
               headwidth=2, headlength=3, width=.005)
    ax1.scatter(abaco_ship['cast_lon'], abaco_ship['cast_lat'], s=25, color='r')
    ax1.quiver(-74.5, 26.29, 0.1, 0, color='w', scale=2, headwidth=2, headlength=3, width=.005)
    ax1.text(-74.3, 26.29, '0.1 m/s', color='w', fontsize=8)
    ax1.text(-74.27, 26.66, 'ADCP', fontsize=12, color='#7CFC00')
    ax1.text(-74.27, 26.58, 'RV CTD', fontsize=12, color='r')
    ax1.scatter(-77, 26.5, s=15, color='m')
    ax1.text(-77, 26.44, 'Trans. St.', fontsize=6)
    ax1.set_title('RV Endeavour (5/8 - 5/15) CTD/ADCP and DG (5/9 - 5/25)', fontsize=18)

    lv = np.arange(-.6, .6, .05)
    lv2 = np.arange(-.6, .6, .1)
    ad = ax2.contourf(abaco_ship['adcp_dist'], abaco_ship['adcp_depth'], abaco_ship['adcp_v'] / 100, levels=lv)
    va = ax2.contour(abaco_ship['adcp_dist'], abaco_ship['adcp_depth'], abaco_ship['adcp_v'] / 100,
                     levels=lv2, colors='k')
    ax2.plot(35 * np.ones(10), np.linspace(0, 5000, 10), color='r', linewidth=.75, linestyle='--')
    ax2.plot(265 * np.ones(10), np.linspace(0, 5000, 10), color='r', linewidth=.75, linestyle='--')
    ax2.clabel(va, fontsize=6, inline=1, spacing=10, fmt='%1.2g')
    ax2.set_ylabel('Depth [m]', fontsize=16)
    ax2.set_xlabel('Distance Offshore [km]', fontsize=16)
    ax2.text(225, 4800, 'ADCP', fontsize=14)
    ax2.axis([0, 300, 0, 5000])
    ax2.invert_yaxis()

    dg_bin = dg_data['bin_depth'][0][0]
    dg_Ds = dg_data['dist'][0][0]
    dg_V = dg_data['V'][0][0]
    # 26.475, -76.648  ==== transect starting position
    # shift transect to match inshore distance point with adcp
    ax3.contourf(np.squeeze(dg_Ds) + 35, np.squeeze(dg_bin), dg_V, levels=lv)
    vc = ax3.contour(np.squeeze(dg_Ds) + 35, np.squeeze(dg_bin), dg_V, levels=lv2, colors='k')
    ax3.clabel(vc, fontsize=6, inline=1, spacing=10, fmt='%1.2g')
    ax3.set_xlabel('Distance Offshore [km]', fontsize=16)
    ax3.text(225, 4800, 'DG', fontsize=14)
    ax3.axis([0, 300, 0, 5000])
    ax3.invert_yaxis()
    ax3.axes.get_yaxis().set_visible(False)
    plot_pro(ax3)

rho_comp = 0
if rho_comp > 0:
    f = plt.subplots()
    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[0:2, :])
    ax2 = plt.subplot(gs[2:, 0:2])
    ax3 = plt.subplot(gs[2:, 2:])

    t_s = datetime.date.fromordinal(np.int(np.nanmin(d_time)))
    t_e = datetime.date.fromordinal(np.int(np.nanmax(d_time)))
    den_levels = np.append(np.arange(25, 27.78, .4), np.arange(27.78, 27.9, .02))

    # cmap = plt.cm.get_cmap("viridis")
    # cmap.set_over('#808000')  # ('#E6E6E6')
    # cross section density T/S profile comparisons
    dats = ~np.isnan(sigma_avg_grid[10, :])
    ax1.pcolor(dist_grid[dats], grid, sigma_avg_grid[:, dats], vmin=25, vmax=28)
    # ax1.contour(dist_grid,grid,t_avg_grid,colors='k',levels=[2,3,4,5,6,7,8,10,12,16,20])
    den_c = ax1.contour(dist_grid, grid, sigma_avg_grid, colors='k', levels= den_levels, linewidth=.35, label='DG')
    # ax1.clabel(den_c, fontsize=6, inline=1,fmt='%.4g',spacing=10)  
    den_ship = ax1.contour(ship_dist, ship_depth_0, ship_sig0, colors='r', levels=den_levels, linewidth=.75)
    # ax1.clabel(den_ship, fontsize=6, inline=1, fmt='%.4g', spacing=-50)
    max_ind = np.where((ship_dist > 275) & (ship_dist < 325))[0][0]
    for i in range(len(den_levels)):
        ax1.text(305, grid[np.where(ship_sig0[:, max_ind] >= den_levels[i])[0][0]], den_levels[i], fontsize=8)
    ax1.axis([0, 300, 0, 5000])
    # ax1.set_xlabel('Distance Offshore [km]')
    ax1.set_ylabel('Depth [m]')
    ax1.set_title(r'$\overline{\rho_{\theta}}$ Cross-Section from' + x.ID + '  ' + x.project + '  ' + np.str(
        t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day) + '(ctd in red)', fontsize=14)
    ax1.invert_yaxis()
    ax1.grid()

    # --- close up of T/S differences
    TT, SS = np.meshgrid(np.arange(0, 25, .1), np.arange(34, 38, .1))
    DD = gsw.sigma0(SS, TT) # sw.pden(SS, TT, np.zeros(np.shape(TT)), 0) - 1000
    DD_l = np.arange(25, 28.2, 0.2)
    DD_c = ax2.contour(SS, TT, DD, colors='k', linewidths=0.25, levels=DD_l)
    ax2.clabel(DD_c, fontsize=8, fmt='%.4g', inline_spacing=-1)  # , inline=1,fmt='%.4g',spacing=5)
    for i in range(np.size(dist_grid)):
        ax2.plot(SA_avg_grid[:, i], CT_avg_grid[:, i], color=c2s[i, :], linewidth=0.75)
    for i in range(10):
        ax2.plot(this_ship_SA[:, i], this_ship_CT[:, i], color=c2_ship[i, :], linestyle='--', linewidth=0.75)
        # ax2.plot(np.nanmean(abaco_argo['salin'][1:-1],axis=1),np.nanmean(abaco_argo['theta'][1:-1],axis=1),color='k')
    ax2.axis([35, 36.5, 1.5, 17])
    ax2.set_ylabel('Conservative Temperature', fontsize=12)
    ax2.set_xlabel('Absolute Salinity', fontsize=12)
    ax2.grid()
    ax2.grid()

    TT, SS = np.meshgrid(np.arange(0, 2.8, .001), np.arange(34.4, 35, .001))
    DD = gsw.sigma0(SS, TT)  # sw.pden(SS, TT, np.zeros(np.shape(TT)), 0) - 1000
    DD_l3 = np.arange(27.7, 28, 0.005)
    DD_c3 = ax3.contour(SS, TT, DD, colors='k', linewidths=0.25, levels=DD_l3)
    # ax3.clabel(DD_c3,inline=1,fontsize=8) #,inline_spacing=-5,fmt='%.4g') #, inline=1,fmt='%.4g',spacing=5)  
    ml = [(34.87, 2.1), (34.88, 2.13), (34.89, 2.15), (34.9, 1.9)]
    ax3.clabel(DD_c3, manual=ml, inline_spacing=-10, fmt='%.3f', colors='k')
    for i in range(np.size(dist_grid)):
        ts_D = ax3.plot(SA_avg_grid[:, i], CT_avg_grid[:, i], linewidth=0.75, color='b',
                        label='DG')  # c2s[i,:])
    for i in range(10):
        ts_s = ax3.plot(this_ship_SA[:, i], this_ship_CT[:, i], linestyle='--', linewidth=0.75, color='r',
                        label='Ship')  # c2_ship[i,:])
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend([handles[0], handles[-1]], [labels[0], labels[-1]], fontsize=12)
    ax3.set_xlabel('Absolute Salinity', fontsize=12)
    ax3.axis([35.025, 35.15, 1.6, 3.25])
    # ax3.axis([34.86, 34.93, 1.7, 2.52])
    plot_pro(ax3)

rho_z_comp = 0
if rho_z_comp > 0:
    # argo noodling
    argo_press = abaco_argo['bin_press']
    argo_dep = -1 * gsw.z_from_p(argo_press, 26)  # sw.dpth(argo_press, 26)
    argo_sigma_interp = np.interp(grid, argo_dep, np.nanmean(abaco_argo['sigma_theta'], axis=1))
    argo_theta_interp = np.interp(grid, argo_dep, np.nanmean(abaco_argo['theta'], axis=1))
    argo_salin_interp = np.interp(grid, argo_dep, np.nanmean(abaco_argo['salin'], axis=1))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    for i in range(np.size(dist_grid)):
        if np.sum(np.isnan(sigma_avg_grid[:, 1])) > 200:
            ix, iv = find_nearest(ship_dist, dist_grid[i])
            # ship noodling
            ship_den_interp = np.interp(grid, ship_depth_0, ship_sig0[:, ix])
            ship_salin_interp = np.interp(grid, ship_depth_0, this_ship_SA[:, ix])
            ship_theta_interp = np.interp(grid, ship_depth_0, this_ship_CT[:, ix])
            shp = ax1.plot((SA_avg_grid[:, i] - ship_salin_interp), grid, label='Shipboard', color='#48D1CC')
            shp = ax2.plot((CT_avg_grid[:, i] - ship_theta_interp), grid, label='Shipboard', color='#48D1CC')
            shp = ax3.plot((sigma_avg_grid[:, i] - ship_den_interp), grid, label='Shipboard', color='#48D1CC')
            # shp = ax1.plot( (salin_avg_grid[:,i]-argo_salin_interp ),grid,label='Argo',color='r') 
            # shp = ax2.plot( (theta_avg_grid[:,i]-argo_theta_interp ),grid,label='Argo',color='r') 
            # shp = ax3.plot( (sigma_avg_grid[:,i]-argo_sigma_interp ),grid,label='Argo',color='r')    
    ax1.axis([-0.02, 0.02, 0, 5000])
    ax1.grid()
    ax1.set_title('Salinity Offset')
    ax1.set_xlabel(r'$S_{DG} - S_{ship}$')
    ax2.axis([-.15, .15, 0, 5000])
    ax2.grid()
    ax2.set_title('Pot Temp. Offset')
    ax2.set_xlabel(r'$\theta_{DG} - \theta_{ship}$')
    ax3.axis([-.03, .03, 0, 5000])
    ax3.set_xlabel(r'$\sigma_{\theta_{DG}} - \sigma_{\theta_{ship}}$')
    ax3.set_title('Density Offset')
    ax3.invert_yaxis()
    plot_pro(ax3)

# ---------------------------------------------------------------------------------------------------------------------
# compute eta 
# instead of removing a mean, remove the linear trend 
# create average density profile that is a function of distance 
z = -1 * grid
ddz_avg_sigma = np.zeros([np.size(grid), np.size(dist_grid)])
ddz_avg_CT = np.zeros([np.size(grid), np.size(dist_grid)])
for i in range(np.size(dist_grid)):
    ddz_avg_sigma[:, i] = np.gradient(sigma_avg_grid[:, i], z)
    ddz_avg_CT[:, i] = np.gradient(CT_avg_grid[:, i], z)

# DG N2 profiles
dg_avg_N2_coarse = np.nanmean(dg_N2, axis=1)
dg_avg_N2_coarse[np.isnan(dg_avg_N2_coarse)] = dg_avg_N2_coarse[~np.isnan(dg_avg_N2_coarse)][0] - 1*10**(-5)
dg_avg_N2 = savgol_filter(dg_avg_N2_coarse, 15, 3)

# compute background N2 (as a function of distance eastward)
N2 = np.zeros(np.shape(sigma_avg_grid))
for i in range(np.size(dist_grid)):
    N2[1:, i] = gsw.Nsquared(SA_avg_grid[:, i], CT_avg_grid[:, i], grid_p, lat=ref_lat)[0]
    # np.squeeze(sw.bfrq(salin_avg_grid[:, i], theta_avg_grid[:, i], grid_p, lat=26.5)[0])
lz = np.where(N2 < 0)
lnan = np.isnan(N2)
N2[lz] = 0
N2[lnan] = 0
N = np.sqrt(N2)

# Shipboard CTD N2
ship_p = gsw.p_from_z(-1 * ship_depth_0, ref_lat)
ship_N2 = np.nan*np.ones(len(ship_p))
ship_N2[0:-1] = gsw.Nsquared(np.nanmean(this_ship_SA, axis=1), np.nanmean(this_ship_CT, axis=1), ship_p, lat=ref_lat)[0]
ship_N2[1] = ship_N2[2] - 1*10**(-5)
ship_N2[0] = ship_N2[1] - 1*10**(-5)
ship_N2[ship_N2 < 0] = np.nan
for i in np.where(np.isnan(ship_N2))[0]:
    ship_N2[i] = ship_N2[i - 1] - 1*10**(-8)
this_ship_N2 = savgol_filter(ship_N2, 15, 3)

# plot all N2
c2_n2 = plt.cm.plasma(np.linspace(0, 1, 33))
f, ax = plt.subplots()
for i in range(np.size(dist_grid)):
    ax.plot(N2[:, i], grid_p, color=c2_n2[i, :])
ax.plot(dg_avg_N2, grid, color='k')
ax.plot(ship_N2, ship_depth_0, color='k')
ax.grid()
ax.invert_yaxis()
plot_pro(ax)

# find closest average profile to subtract to find eta     
eta = np.zeros([np.size(grid), np.size(dist_grid)])
df_theta_anom = pd.DataFrame()
df_salin_anom = pd.DataFrame()
df_sigma_anom = pd.DataFrame()
df_eta = pd.DataFrame()
eta_theta = np.zeros([np.size(grid), np.size(dist_grid)])
df_eta_theta = pd.DataFrame()
closest_rec = np.nan * np.zeros([np.size(mean_dist)])

# ---------------------------------------------------------------------------------------------------------
# COMPUTE ANOMALIES AND ETA ... DEEP salinity offset
count = 0
for i in range(np.size(mean_dist)):
    dist_test = np.abs(mean_dist[i] - dist_grid)  # distance between this profile and every other on dist_grid
    closest_i = np.where(dist_test == dist_test.min())[0][0]  # find closest dist_grid station to this profile
    closest_rec[i] = np.int(closest_i)

    theta_anom = ct[:, i] - CT_avg_grid[:, closest_i]
    salin_anom = sa[:, i] - SA_avg_grid[:, closest_i]
    sigma_anom = sig0[:, i] - sigma_avg_grid[:, closest_i]
    eta = (sig0[:, i] - sigma_avg_grid[:, closest_i]) / np.squeeze(ddz_avg_sigma[:, closest_i])
    eta_theta = (ct[:, i] - ct[:, closest_i]) / np.squeeze(ddz_avg_CT[:, closest_i])

    if count < 1:
        df_theta_anom = pd.DataFrame(theta_anom, index=grid, columns=[profile_tags[i]])  # theta_anom = ct[:, i] - ct[:, closest_i]
        df_salin_anom = pd.DataFrame(salin_anom, index=grid, columns=[profile_tags[i]])
        df_sigma_anom = pd.DataFrame(sigma_anom, index=grid, columns=[profile_tags[i]])
        df_eta = pd.DataFrame(eta, index=grid, columns=[profile_tags[i]])
        df_eta_theta = pd.DataFrame(eta_theta, index=grid, columns=[profile_tags[i]])
    else:
        df_theta_anom2 = pd.DataFrame(theta_anom, index=grid, columns=[profile_tags[i]])
        df_salin_anom2 = pd.DataFrame(salin_anom, index=grid, columns=[profile_tags[i]])
        df_sigma_anom2 = pd.DataFrame(sigma_anom, index=grid, columns=[profile_tags[i]])
        eta2 = pd.DataFrame(eta, index=grid, columns=[profile_tags[i]])
        eta3 = pd.DataFrame(eta_theta, index=grid, columns=[profile_tags[i]])

        df_theta_anom = pd.concat([df_theta_anom, df_theta_anom2], axis=1)
        df_salin_anom = pd.concat([df_salin_anom, df_salin_anom2], axis=1)
        df_sigma_anom = pd.concat([df_sigma_anom, df_sigma_anom2], axis=1)
        df_eta = pd.concat([df_eta, eta2], axis=1)
        df_eta_theta = pd.concat([df_eta_theta, eta3], axis=1)
    count = count + 1

# Eta compute using M/W method
# need to pair mwe_lat/lon positions to closest position on dist_grid
# try selecting profiles that are farther offshore (excluding the DWBC)
dist_lim = 100
offshore = dg_v_dist > dist_lim
dg_v_dist = dg_v_dist[offshore]
avg_sig0_per_dep = avg_sig0_per_dep[:, offshore]
dg_v_0 = dg_v_0[:, offshore]
dg_v_dive_no = dg_v_dive_no[offshore]

eta_alt = np.nan * np.ones(np.shape(avg_sig0_per_dep))
for i in range(np.shape(avg_sig0_per_dep)[1]):
    dist_test = np.abs(dg_v_dist[i] - dist_grid)  # distance between this profile and every other on dist_grid
    closest_i = np.where(dist_test == dist_test.min())[0][0]  # find closest dist_grid station to this profile
    eta_alt[:, i] = (avg_sig0_per_dep[:, i] - sigma_avg_grid[:, closest_i]) / np.squeeze(ddz_avg_sigma[:, closest_i])

eta_dive_no = dg_v_dive_no
dg_eta_time = time_rec[np.in1d(profile_tags, eta_dive_no)]
num_eta_profs = np.shape(eta_alt)[1]

# ---------------------------------------------------------------------------------------------------------
# ------- Eta_fit / Mode Decomposition --------------
# define G grid 
omega = 0  # frequency zeroed for geostrophic modes
mmax = 60  # highest baroclinic mode to be calculated
nmodes = mmax + 1

G, Gz, c, epsilon = vertical_modes(dg_avg_N2, grid, omega, mmax)

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
# ---------------------------------------------------------------------------------------------------------
# first taper fit above and below min/max limits
# Project modes onto each eta (find fitted eta)
# Compute PE 
eta_fit_depth_min = 100
eta_fit_depth_max = 3750
eta_th_fit_depth_min = 50
eta_th_fit_depth_max = 4200
AG = np.zeros([nmodes, num_eta_profs])
AG_theta = np.zeros([nmodes, num_eta_profs])
Eta_m = np.nan * np.zeros([np.size(grid), num_eta_profs])
Neta = np.nan * np.zeros([np.size(grid), num_eta_profs])
NEta_m = np.nan * np.zeros([np.size(grid), num_eta_profs])
Eta_theta_m = np.nan * np.zeros([np.size(grid), num_eta_profs])
PE_per_mass = np.nan * np.zeros([nmodes, num_eta_profs])
PE_theta_per_mass = np.nan * np.zeros([nmodes, num_eta_profs])
for i in range(num_eta_profs):
    # this_eta = df_eta.iloc[:, i][:].copy()
    this_eta = eta_alt[:, i].copy()
    # obtain matrix of NEta
    Neta[:, i] = N[:, np.int(closest_rec[i])] * this_eta
    this_eta_theta = df_eta_theta.iloc[:, i][:].copy()
    iw = np.where((grid >= eta_fit_depth_min) & (grid <= eta_fit_depth_max))
    if iw[0].size > 1:
        # eta_fs = df_eta.iloc[:, i][:].copy()  # ETA
        eta_fs = eta_alt[:, i].copy()  # ETA
        eta_theta_fs = df_eta_theta.iloc[:, i][:].copy()

        i_sh = np.where((grid < eta_fit_depth_min))
        eta_fs[i_sh[0]] = grid[i_sh] * this_eta[iw[0][0]] / grid[iw[0][0]]
        i_sh = np.where((grid < eta_th_fit_depth_min))
        eta_theta_fs.iloc[i_sh[0]] = grid[i_sh] * this_eta_theta.iloc[iw[0][0]] / grid[iw[0][0]]

        i_dp = np.where((grid > eta_fit_depth_max))
        eta_fs[i_dp[0]] = (grid[i_dp] - grid[-1]) * this_eta[iw[0][-1]] / (grid[iw[0][-1]] - grid[-1])
        i_dp = np.where((grid > eta_th_fit_depth_max))
        eta_theta_fs.iloc[i_dp[0]] = (grid[i_dp] - grid[-1]) * this_eta_theta.iloc[iw[0][-1]] / (
                    grid[iw[0][-1]] - grid[-1])

        AG[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], np.transpose(np.atleast_2d(eta_fs)))[0])
        AG_theta[1:, i] = np.squeeze(np.linalg.lstsq(G[:, 1:], np.transpose(np.atleast_2d(eta_theta_fs)))[0])
        Eta_m[:, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG[:, i])))
        NEta_m[:, i] = N[:, np.int(closest_rec[i])] * np.array(
            np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG[:, i]))))
        Eta_theta_m[:, i] = np.squeeze(np.matrix(G) * np.transpose(np.matrix(AG_theta[:, i])))
        PE_per_mass[:, i] = (1 / 2) * AG[:, i] * AG[:, i] * c * c
        PE_theta_per_mass[:, i] = (1 / 2) * AG_theta[:, i] * AG_theta[:, i] * c * c

# ---------------------------------------------------------------------------------------------------------
# PLOTTING
# sample plots of G and Gz
sam_pl = 0
if sam_pl > 0:
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # colors = plt.cm.tab10(np.arange(0,5,1))
    colors = '#C24704', '#D9CC3C', '#A0E0BA', '#00ADA7'
    for i in range(4):
        gp = ax1.plot(G[:, i] / np.max(grid), grid, label='Mode ' + str(i), color=colors[i], linewidth=3)
        ax2.plot(Gz[:, i], grid, color=colors[i], linewidth=3)
    n2p = ax1.plot((np.sqrt(np.nanmean(N2, axis=1)) * (1800 / np.pi)) / 10, grid, color='k', label='N(z) [10 cph]')
    ax1.grid()
    ax1.axis([-1, 1, 0, 5000])
    ax1.set_ylabel('Depth [m]', fontsize=16)
    ax1.set_xlabel('Vert. Displacement Stucture', fontsize=14)
    ax1.set_title(r"$G_n$(z) Modes Shapes ($\sim \xi$)", fontsize=20)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[-1], handles[0], handles[1], handles[2], handles[3]],
               [labels[-1], labels[0], labels[1], labels[2], labels[3]], fontsize=12)
    ax2.axis([-4, 4, 0, 5000])
    ax2.set_xlabel('Vert. Structure of Hor. Velocity', fontsize=14)
    ax2.set_title(r"$G_n$'(z) Mode Shapes ($\sim u$)", fontsize=20)
    ax2.invert_yaxis()
    plot_pro(ax2)

# -------------------------------------------------------------------------------------------------------------------
# VELOCITY
# -------------------------------------------------------------------------------------------------------------------
def functi(p, xe, xb):
    #  This is the target function that needs to be minimized
    fsq = (xe - p*xb)**2
    return fsq.sum()
# -------------------------------------------------------------------------------------------------------------------
# DG Geostrophic Velocity Profiles
# good_v = np.zeros(np.shape(dg_v_0)[1], dtype=bool)
# for i in range(np.shape(dg_v_0)[1]):
#     dv_dz = np.gradient(dg_v_0[:, i], -1 * grid)
#     if (np.nanmax(np.abs(dv_dz)) < 0.002):
#         good_v[i] = True
good_v = np.ones(np.shape(dg_v_0)[1], dtype=bool)
dg_v = dg_v_0[:, good_v]
dg_v_dive_no_1 = dg_v_dive_no[good_v]
dg_np = np.shape(dg_v)[1]
dg_v_time = time_rec[np.in1d(profile_tags, dg_v_dive_no_1)]
# --------------------------------------------------------------------------------------------------------------------
# DG HKE est.
HKE_noise_threshold_dg = 1e-5
dg_AGz = np.zeros([nmodes, dg_np])
dg_v_m_2 = np.nan * np.zeros([np.size(grid), dg_np])
dg_HKE_per_mass = np.nan * np.zeros([nmodes, dg_np])
modest = np.arange(11, nmodes)
dg_good_prof = np.ones(dg_np)
for i in range(dg_np):
    # fit to velocity profiles
    this_dg_V = dg_v[:, i].copy()  # np.interp(grid, np.squeeze(dg_bin), dg_v[:, i])
    iv = np.where(~np.isnan(this_dg_V))
    if iv[0].size > 1:
        dg_AGz[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(Gz[iv, :]), np.transpose(np.atleast_2d(this_dg_V[iv])))[0])
        dg_v_m_2[:, i] = np.squeeze(np.matrix(Gz) * np.transpose(np.matrix(dg_AGz[:, i])))
        dg_HKE_per_mass[:, i] = dg_AGz[:, i] * dg_AGz[:, i]
        ival = np.where(dg_HKE_per_mass[modest, i] >= HKE_noise_threshold_dg)
        if np.size(ival) > 0:
            dg_good_prof[i] = 0  # flag profile as noisy
    else:
        dg_good_prof[i] = 0  # flag empty profile as noisy as well

# --- EOF of velocity profiles ---------------------------
not_deep = np.isfinite(dg_v[-9, :])  # & (Time2 > 735750)
V3 = dg_v[:, dg_good_prof > 0]
check1 = 3      # upper index to include in eof computation
check2 = -14     # lower index to include in eof computation
grid_check = grid[check1:check2]
Uzq = V3[check1:check2, :].copy()
nq = np.size(V3[0, :])
avg_Uzq = np.nanmean(np.transpose(Uzq), axis=0)
Uzqa = Uzq - np.transpose(np.tile(avg_Uzq, [nq, 1]))
cov_Uzqa = (1 / nq) * np.matrix(Uzqa) * np.matrix(np.transpose(Uzqa))
D_Uzqa, V_Uzqa = np.linalg.eig(cov_Uzqa)
t1 = np.real(D_Uzqa[0:10])
PEV = t1 / np.sum(t1)
# ------ VARIANCE EXPLAINED BY BAROCLINIC MODES -----------
eof1 = np.array(np.real(V_Uzqa[:, 1]))
eof1_sc = (1/2)*(eof1.max() - eof1.min()) + eof1.min()
bc1 = Gz[check1:check2, 1]  # flat bottom
bc2 = F[check1:check2, 0]   # sloping bottom
# -- minimize mode shapes onto eof shape
p = -0.8*eof1.min()/np.max(np.abs(Gz[:, 1]))  # initial guess for fmin
ins1 = np.transpose(np.concatenate([-1*eof1, bc1[:, np.newaxis]], axis=1))
ins2 = np.transpose(np.concatenate([eof1, bc2[:, np.newaxis]], axis=1))
# minimize the function functi, this computes the sum of square differences between eof1 and bc
# the parameter p changes where p is a scaling amplitude to multiply bc by
min_p1 = fmin(functi, p, args=(tuple(ins1)))
min_p2 = fmin(functi, p, args=(tuple(ins2)))

# fraction of unexplained variance
dg_fvu1 = np.sum((-1*eof1[:, 0] - bc1*min_p1)**2)/np.sum((-1*eof1 - np.mean(-1*eof1))**2)
dg_fvu2 = np.sum((eof1[:, 0] - bc2*min_p2)**2)/np.sum((eof1 - np.mean(eof1))**2)

# -------------------------------------------------------------------------------------------------------------------
# --- SHIP ADCP HKE est. (from the year of the DG dives)
# find adcp profiles that are deep enough and fit baroclinic modes to these 
HKE_noise_threshold_adcp = 1e-5
check = np.zeros(np.size(this_adcp_dist))
for i in range(np.size(this_adcp_dist)):
    check[i] = adcp_depth[np.where(~np.isnan(this_adcp_v[:, i]))[0][-1]]
adcp_in = np.where((check >= 4700) & (check <= 5300))
V = this_adcp_v[:, adcp_in[0]] / 100
V_dist = this_adcp_dist[adcp_in[0]]
adcp_np = np.size(V_dist)
ship_depth_1 = ship_depth_0[0:-39]
V_dist_this_year = V_dist.copy()

ship_G, ship_Gz, ship_c, ship_epsilon = vertical_modes(this_ship_N2[0:-39], ship_depth_1, omega, mmax)

# -- compute alternate ship vertical modes
bc_bot = 2  # 1 = flat, 2 = rough
grid2 = np.concatenate([np.arange(0, 150, 10), np.arange(150, 300, 10), np.arange(300, 4500, 10)])
n2_interp = np.interp(grid2, ship_depth_1, this_ship_N2[0:-39])
F_int_g2, F_g2, c_ff, norm_constant, epsilon2 = vertical_modes_f(n2_interp, grid2, omega, mmax, bc_bot)
ship_F = np.nan * np.ones((np.size(ship_depth_1), mmax + 1))
ship_F_int = np.nan * np.ones((np.size(ship_depth_1), mmax + 1))
for i in range(mmax + 1):
    ship_F[:, i] = np.interp(ship_depth_1, grid2, F_g2[:, i])
    ship_F_int[:, i] = np.interp(ship_depth_1, grid2, F_int_g2[:, i])

adcp_AGz = np.zeros([nmodes, adcp_np])
V_m = np.nan * np.zeros([np.size(ship_depth_1), adcp_np])
adcp_HKE_per_mass = np.nan * np.zeros([nmodes, adcp_np])
modest = np.arange(11, nmodes)
adcp_good_prof = np.zeros(adcp_np)
for i in range(adcp_np):
    # fit to velocity profiles
    this_V_0 = V[:, i].copy()
    this_V = np.interp(ship_depth_1, adcp_depth, this_V_0) # new using Gz profiles from shipboard ctd
    # this_V = np.interp(grid, adcp_depth, this_V_0)  # old using Gz profiles from DG
    iv = np.where(~np.isnan(this_V))
    if iv[0].size > 1:
        # Gz(iv,:)\V_g(iv,ip)
        # Gz*AGz[:,i];
        adcp_AGz[:, i] = np.squeeze(np.linalg.lstsq(np.squeeze(ship_Gz[iv, :]),
                                                    np.transpose(np.atleast_2d(this_V[iv])))[0])
        V_m[:, i] = np.squeeze(np.matrix(ship_Gz) * np.transpose(np.matrix(adcp_AGz[:, i])))
        adcp_HKE_per_mass[:, i] = adcp_AGz[:, i] * adcp_AGz[:, i]
        ival = np.where(adcp_HKE_per_mass[modest, i] >= HKE_noise_threshold_adcp)
        if np.size(ival) > 0:
            adcp_good_prof[i] = 1  # flag profile as noisy
    else:
        adcp_good_prof[i] = 1  # flag empty profile as noisy as well

# -------------------------------------------------------------------------------------------------------------------
# ----- EOF of velocity profiles ---------------------------
V3 = np.nan*np.ones((len(ship_depth_1), adcp_np))
for i in range(adcp_np):
    V3[:, i] = np.interp(ship_depth_1, adcp_depth, V[:, i]) # new using Gz profiles from shipboard ctd
check1 = 3      # upper index to include in eof computation
check2 = -1     # lower index to include in eof computation
ship_grid_check = ship_depth_1[check1:check2]
Uzq = V3[check1:check2, :].copy()
nq = np.size(V3[0, :])
avg_Uzq = np.nanmean(np.transpose(Uzq), axis=0)
Uzqa = Uzq - np.transpose(np.tile(avg_Uzq, [nq, 1]))
cov_Uzqa = (1 / nq) * np.matrix(Uzqa) * np.matrix(np.transpose(Uzqa))
D_Uzqa, V_Uzqa_adcp = np.linalg.eig(cov_Uzqa)
t1 = np.real(D_Uzqa[0:10])
PEV_adcp = t1 / np.sum(t1)

# ------ VARIANCE of LADCP EXPLAINED BY BAROCLINIC MODES -----------
eof1 = np.array(np.real(V_Uzqa_adcp[:, 1]))
eof1_sc = (1/2)*(eof1.max() - eof1.min()) + eof1.min()
bc1 = ship_Gz[check1:check2, 1]  # flat bottom
bc2 = ship_F[check1:check2, 0]   # sloping bottom

# -- minimize mode shapes onto eof shape
p1 = 0.8*eof1.min()/np.max(np.abs(ship_Gz[:, 1]))
p2 = 0.8*eof1.min()/np.max(np.abs(ship_F[:, 1]))
ins1 = np.transpose(np.concatenate([eof1, bc1[:, np.newaxis]], axis=1))
ins2 = np.transpose(np.concatenate([eof1, bc2[:, np.newaxis]], axis=1))
min_p1 = fmin(functi, p1, args=(tuple(ins1)))
min_p2 = fmin(functi, p2, args=(tuple(ins2)))

# -- plot inspect minimization of mode shapes
# f, ax = plt.subplots()
# ax.plot(eof1, ship_grid_check, color='k')
# ax.plot(bc1*min_p1, ship_grid_check, color='r')
# ax.plot(bc2*min_p2, ship_grid_check, color='b')
# ax.invert_yaxis()
# plot_pro(ax)

adcp_fvu1 = np.sum((eof1[:, 0] - bc1*min_p1)**2)/np.sum((eof1 - np.mean(eof1))**2)
adcp_fvu2 = np.sum((eof1[:, 0] - bc2*min_p2)**2)/np.sum((eof1 - np.mean(eof1))**2)

# ----------------------------- PLOT VELOCITY EOFS -------------------------------------------
fvu1 = np.sum((eof1[:, 0] - bc1*min_p1)**2)/np.sum((eof1 - np.mean(eof1))**2)
# fvu2 = np.sum((eof1[:, 0] - bc2*min_p2)**2)/np.sum((eof1 - np.mean(eof1))**2)
f, (ax2, ax1, ax3) = plt.subplots(1, 3, sharey=True)
ax1.plot(V_Uzqa[:, 0], grid_check, label=r'PEV$_{' + str(0 + 1) + '}$ = ' + str(100 * np.round(PEV[0], 3)),
        linewidth=2, color='r')
ax1.plot(V_Uzqa[:, 1], grid_check, label=r'PEV$_{' + str(1 + 1) + '}$ = ' + str(100 * np.round(PEV[1], 3)),
        linewidth=2, color='b')
ax1.plot(V_Uzqa[:, 2], grid_check, label=r'PEV$_{' + str(2 + 1) + '}$ = ' + str(100 * np.round(PEV[2], 3)),
        linewidth=2, color='g')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=10)
ax1.axis([-.2, .2, 0, 5000])
ax1.set_title('DG')
ax1.grid()

ax2.plot(V_Uzqa_adcp[:, 0], ship_grid_check,
         label=r'PEV$_{' + str(0 + 1) + '}$ = ' + str(100 * np.round(PEV_adcp[0], 3)), linewidth=2, color='r')
ax2.plot(V_Uzqa_adcp[:, 1], ship_grid_check,
         label=r'PEV$_{' + str(1 + 1) + '}$ = ' + str(100 * np.round(PEV_adcp[1], 3)), linewidth=2, color='b')
ax2.plot(V_Uzqa_adcp[:, 2], ship_grid_check,
         label=r'PEV$_{' + str(2 + 1) + '}$ = ' + str(100 * np.round(PEV_adcp[2], 3)), linewidth=2, color='g')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=10)
ax2.axis([-.2, .2, 0, 5000])
ax2.set_title('LADCP')
ax2.set_ylabel('Depth [m]')
ax2.grid()

ax3.plot(Gz[:, 1], grid, 'm', label='Flat Bot. Mode 1')
ax3.text(0, 3250, 'Frac. Var. Explained LADCP = ' + str(np.round(1 - adcp_fvu1, 2)), fontsize=10)
ax3.text(0, 3750, 'Frac. Var. Explained DG = ' + str(np.round(1 - dg_fvu1, 2)), fontsize=10)
ax3.set_title('Baroclinic Mode(s)')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels, fontsize=10)
# ax3.plot(Gz[:, 2], grid, 'm')
# ax3.plot(F[:, 0], grid, 'c')
# ax3.plot(F[:, 1], grid, 'c')
ax3.invert_yaxis()
plot_pro(ax3)

# ----------------------------- PLOT VELOCITY PROFILES AND COMPARE (also plot ETA) ------------------------------------
t_s = datetime.date.fromordinal(np.int(np.nanmin(dg_eta_time)))
t_e = datetime.date.fromordinal(np.int(np.nanmax(dg_eta_time)))
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
for i in range(adcp_np):
    if adcp_good_prof[i] < 2:
        ad1 = ax1.plot(V[:,i], adcp_depth, color='#CD853F')
        ax1.plot(V_m[:,i], ship_depth_1, color='k', linewidth=0.75)
    else:
        ax1.plot(V[:,i], adcp_depth, color='r')
for i in range(dg_np):
    if dg_good_prof[i] > 0:  # dg_quiet[0][i] < 1:
        ad1 = ax2.plot(dg_v[:, i], grid, color='#CD853F')
        ax2.plot(dg_v_m_2[:, i], grid, color='k', linestyle='--', linewidth=0.75)
    # else:
    #     ax2.plot(dg_v[:, i], grid, color='r')
ax1.axis([-.6, .6, 0, 4900])
ax1.set_xlabel('Meridional Velocity [m/s]', fontsize=12)
ax1.set_ylabel('Depth [m]', fontsize=12)
ax1.set_title(r'LADCP$_v$ (' + str(adcp_np) + ' profiles)', fontsize=12)
ax1.invert_yaxis()
ax1.grid()
ax2.axis([-.6, .6, 0, 4900])
ax2.set_xlabel('Cross-Track Velocity [m/s]', fontsize=12)
ax2.set_title(str(t_s) + ' - ' + str(t_e) + ' Velocity', fontsize=12)  # (' + str(dg_np) + ' profiles)')
ax2.invert_yaxis()
ax2.grid()

for i in range(num_eta_profs):
    p37_2 = ax3.plot(eta_alt[:, i], grid, color='#CD853F', linewidth=1, label='DG037')
    p37_f = ax3.plot(Eta_m[:, i], grid, 'k--', linewidth=.75)
ax3.plot([0, 0], [0, 5500], '--k')
ax3.set_title('Vertical Displacement', fontsize=12)  # (' + str(num_profs) + ' profiles)') # dg37
ax3.set_xlabel('Vertical Isopycal Displacement [m]', fontsize=12)
ax3.axis([-400, 400, 0, 4900])
ax3.invert_yaxis()
plot_pro(ax3)

# # - test smoothed eta against eta from individual profiles
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# for i in range(np.size(mean_dist)):  # range(np.size(subset[0])): #
#     ax1.plot(df_eta.iloc[:, i], grid, color='#CD853F', linewidth=1, label='DG038')
# ax1.axis([-600, 600, 0, 5100])
# for i in range(np.shape(eta_alt)[1]):
#     ax2.plot(eta_alt[:, i], grid)
# ax2.axis([-600, 600, 0, 5100])
# ax2.invert_yaxis()
# plot_pro(ax2)


# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
f_ref = np.pi * np.sin(np.deg2rad(26.5)) / (12 * 1800)
rho0 = 1025
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --- SHIP ADCP HKE est. (for all years)
adcp_datetime = np.nan*np.ones(len(adcp_time))
matlab_datenum = 731965.04835648148
for i in range(len(adcp_time)):
    intermed = datetime.date.fromordinal(int(np.min(adcp_time[i]))) + datetime.timedelta(
        days=matlab_datenum % 1) - datetime.timedelta(days=366)
    adcp_datetime[i] = intermed.year + (intermed.month / 12)
per_year = []
for j in range(len(np.unique(np.floor(adcp_datetime)))):
    per_year.append(np.where(np.floor(adcp_datetime) == np.unique(np.floor(adcp_datetime))[j])[0])

# Shipboard CTD N2
ship_p = gsw.p_from_z(-1 * ship_depth_0, ref_lat)
ship_N2 = np.nan*np.ones(len(ship_p))
ship_years = np.unique(time_key)
f, (ax0, ax1) = plt.subplots(1, 2)
c2_n2 = plt.cm.plasma(np.linspace(0, 1, len(ship_years) + 1))
for i in range(len(ship_years) + 1):
    # adcp time in (profiles for each year)
    adcp_v_iter = adcp_v[:, per_year[i]]

    # find adcp profiles that are deep enough and fit baroclinic modes to these
    HKE_noise_threshold_adcp = 1e-5
    check = np.zeros(np.size(adcp_dist[per_year[i]]))
    for k in range(np.size(adcp_dist[per_year[i]])):
        check[k] = adcp_depth[np.where(~np.isnan(adcp_v_iter[:, k]))[0][-1]]
    adcp_in = np.where((check >= 4700) & (check <= 5300))
    V = adcp_v_iter[:, adcp_in[0]] / 100
    V_dist = adcp_dist[per_year[i]][adcp_in[0]]
    V_time = adcp_time[per_year[i]][adcp_in[0]]
    adcp_np = np.size(V_dist)
    ship_depth_1 = ship_depth_0[0:-39]

    adcp_AGz = np.zeros([nmodes, adcp_np])
    V_m = np.nan * np.zeros([np.size(ship_depth_1), adcp_np])
    HKE_per_mass = np.nan * np.zeros([nmodes, adcp_np])
    modest = np.arange(11, nmodes)
    adcp_good_prof = np.zeros(adcp_np)
    dk_per = np.zeros(adcp_np)
    sc_x_per = np.zeros((adcp_np, nmodes-1))
    for ii in range(adcp_np):
        # find and compute appropriate N2
        # use N2 compue ship_Gz
        this_dist = V_dist[ii]
        # compute N2 that is an average of all density profiles within a set distance to velocity profile (different each year)
        # fill 2018 with density profiles from 2017
        if i < len(ship_years):
            ship_N2[0:-1] = gsw.Nsquared(np.nanmean(ship_SA[:, (time_key == ship_years[i]) & (np.abs(this_dist - ship_dist) < 100)], axis=1),
                                         np.nanmean(ship_CT[:, (time_key == ship_years[i]) & (np.abs(this_dist - ship_dist) < 100)], axis=1),
                                         ship_p, lat=ref_lat)[0]
        else:
            ship_N2[0:-1] = gsw.Nsquared(np.nanmean(ship_SA[:, (time_key == ship_years[i - 1]) & (np.abs(this_dist - ship_dist) < 100)], axis=1),
                                         np.nanmean(ship_CT[:, (time_key == ship_years[i - 1]) & (np.abs(this_dist - ship_dist) < 100)], axis=1),
                                         ship_p, lat=ref_lat)[0]
        # taper N2
        if i < 1:
            ship_N2[2] = ship_N2[3] - 5 * 10 ** (-5)
            ship_N2[1] = ship_N2[2] - 5 * 10 ** (-5)
            ship_N2[0] = ship_N2[1] - 5 * 10 ** (-5)
        else:
            ship_N2[1] = ship_N2[2] - 2 * 10 ** (-5)
            ship_N2[0] = ship_N2[1] - 2 * 10 ** (-5)
        ship_N2[ship_N2 < 0] = np.nan
        for j in np.where(np.isnan(ship_N2))[0]:
            ship_N2[j] = ship_N2[j - 1] - 1 * 10 ** (-8)

        # should compute a Gz and c as a function of longitude
        ship_G, ship_Gz, ship_c, ship_epsilon = vertical_modes(ship_N2[0:-39], ship_depth_1, omega, mmax)

        dk_per[ii] = f_ref / ship_c[1]
        sc_x_per[ii, :] = 1000 * f_ref / ship_c[1:]

        # fit to velocity profiles
        this_V_0 = V[:, ii].copy()
        this_V = np.interp(ship_depth_1, adcp_depth, this_V_0) # new using Gz profiles from shipboard ctd
        iv = np.where(~np.isnan(this_V))
        if iv[0].size > 1:
            # Gz(iv,:)\V_g(iv,ip)
            # Gz*AGz[:,i];
            adcp_AGz[:, ii] = np.squeeze(np.linalg.lstsq(np.squeeze(ship_Gz[iv, :]),
                                                        np.transpose(np.atleast_2d(this_V[iv])))[0])
            V_m[:, ii] = np.squeeze(np.matrix(ship_Gz) * np.transpose(np.matrix(adcp_AGz[:, ii])))
            HKE_per_mass[:, ii] = adcp_AGz[:, ii] * adcp_AGz[:, ii]
            ival = np.where(HKE_per_mass[modest, ii] >= HKE_noise_threshold_adcp)
            if np.size(ival) > 0:
                adcp_good_prof[ii] = 1  # flag profile as noisy
        else:
            adcp_good_prof[ii] = 1  # flag empty profile as noisy as well
    # end loop over all velocity profiles in each year

    # -- output
    if i < 1:
        adcp_v_fit_dist = V_dist
        adcp_v_fit_time = V_time
        adcp_v_fit = V_m
        adcp_hke_per = HKE_per_mass / np.tile(dk_per, (nmodes, 1))
    else:
        adcp_v_fit_dist = np.concatenate((adcp_v_fit_dist, V_dist))
        adcp_v_fit_time = np.concatenate((adcp_v_fit_time, V_time))
        adcp_v_fit = np.concatenate((adcp_v_fit, V_m), axis=1)
        adcp_hke_per = np.concatenate((adcp_hke_per, HKE_per_mass / np.tile(dk_per, (nmodes, 1))), axis=1)

    # -- plotting
    for jj in range(adcp_np):
        ax1.plot(V[:, jj], adcp_depth, linewidth=0.75, color=c2_n2[i, :])
        ax1.plot(V_m[:, jj], ship_depth_1, color='k', linewidth=0.5)

    avg_KE_adcp_i = np.nanmean(HKE_per_mass / np.tile(dk_per, (nmodes, 1)), 1)
    avg_sc_x = np.nanmean(sc_x_per, axis=0)
    if i < len(ship_years):
        KE_adcp = ax0.plot(avg_sc_x, avg_KE_adcp_i[1:], label=np.int(ship_years[i]), linewidth=1.75, color=c2_n2[i, :])
        ax0.scatter(avg_sc_x, avg_KE_adcp_i[1:], s=20, color=c2_n2[i, :])  # adcp
    else:
        KE_adcp = ax0.plot(avg_sc_x, avg_KE_adcp_i[1:], label='2018', linewidth=1.75, color=c2_n2[i, :])  # adcp
        ax0.scatter(avg_sc_x, avg_KE_adcp_i[1:], s=20, color=c2_n2[i, :])  # adcp
    # adcp KE_0
    KE_adcp0 = ax0.plot([10 ** -2, 1000 * f_ref / ship_c[1]], avg_KE_adcp_i[0:2], linewidth=1.75, color=c2_n2[i, :])
    ax0.scatter(10 ** -2, avg_KE_adcp_i[0], s=25, facecolors='none', color=c2_n2[i, :])  # adcp KE_0

ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.axis([10 ** -2, 10 ** 1, 10 ** (-3), 4 * 10 ** 3])
# ax0.axis([10 ** -2, 10 ** 1, 10 ** (-4), 10 ** 3])
ax0.set_xlabel(r'Scaled Vertical Wavenumber = (Rossby Radius)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=14)
ax0.set_ylabel('Spectral Density', fontsize=14)  # ' (and Hor. Wavenumber)')
ax0.set_title(r'LADCP$_v$ KE Spectrum', fontsize=14)
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles, labels, fontsize=12)
ax0.grid()
ax1.set_xlim([-0.75, 0.75])
ax1.invert_yaxis()
ax1.set_xlabel('Northward Velocity [m/s]')
plot_pro(ax1)
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# ------------ ENERGY SPECTRA
# BATS DG (2015)
pkl_file = open('/Users/jake/Documents/baroclinic_modes/DG/sg035_energy.pkl', 'rb')
bats_dg = pickle.load(pkl_file)
pkl_file.close()
dg_depth = bats_dg['depth']
dg_N2 = bats_dg['N2']
bats_ke = bats_dg['KE']
bats_pe = bats_dg['PE']
bats_c = bats_dg['c']
bats_f = bats_dg['f']
bats_dk = bats_f / c[1]

# ---- ABACO DG
# DG
avg_PE = np.nanmean(PE_per_mass, 1)
avg_PE_theta = np.nanmean(PE_theta_per_mass, 1)
avg_KE_dg = np.nanmean(np.squeeze(dg_HKE_per_mass[:, dg_good_prof > 0]), 1)
avg_KE_dg_all = np.nanmean(dg_HKE_per_mass, 1)

dk = f_ref / c[1]
sc_x = 1000 * f_ref / c[1:]
vert_wave = sc_x / 1000
PE_SD, PE_GM, GMPE, GMKE = PE_Tide_GM(rho0, grid, nmodes, N2, f_ref)
k_h = 1e3 * (f_ref / c[1:]) * np.sqrt(avg_KE_dg[1:] / avg_PE[1:])
alpha = 10
mu = 1.88e-3 / (1 + 0.03222 * np.nanmean(CT_avg_grid, axis=1) + 0.002377 * np.nanmean(CT_avg_grid, axis=1)**2)
nu = mu / gsw.rho(np.nanmean(SA_avg_grid, axis=1), np.nanmean(CT_avg_grid, axis=1), grid_p)
avg_nu = np.nanmean(nu)

# check dependence of ADCP HKE on longitude
f, ax = plt.subplots()
close = np.where(adcp_v_fit_dist < dist_lim)[0]
far = np.where((adcp_v_fit_dist > dist_lim) & (adcp_v_fit_dist < 450))[0]
ax.scatter(sc_x, np.nanmean(adcp_hke_per[1:, close], axis=1), s=10, color='r')
ax.plot(sc_x, np.nanmean(adcp_hke_per[1:, close], axis=1), color='r', label=r'KE$_{close}$', linewidth=1.75)  # adcp all
ax.plot([10 ** -2, 1000 * f_ref / c[1]], np.nanmean(adcp_hke_per[0:2, close], axis=1), 'r', linewidth=1.75)  # adcp KE_0 all
ax.scatter(10 ** -2, np.nanmean(adcp_hke_per[0, close]), color='r', s=25, facecolors='none')  # adcp KE_0 all
ax.plot(sc_x, np.nanmean(adcp_hke_per[1:, far], axis=1), color='b', label=r'KE$_{far}$', linewidth=1.75)  # adcp all
ax.plot([10 ** -2, 1000 * f_ref / c[1]], np.nanmean(adcp_hke_per[0:2, far], axis=1), 'b', linewidth=1.75)  # adcp KE_0 all
ax.scatter(10 ** -2, np.nanmean(adcp_hke_per[0, far]), color='b', s=25, facecolors='none')  # adcp KE_0 all
ax.axis([10 ** -2, 10 ** 1, 1 * 10 ** (-3), 4 * 10 ** (3)])
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title('Variability in LADCP KE by Longitude 2011-2018')
ax.set_xlabel(r'Scaled Vertical Wavenumber [km$^{-1}$]')
ax.set_ylabel('Spectral Density')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=10)
plot_pro(ax)

# LADCP
avg_KE_adcp = np.nanmean(adcp_hke_per[:, far], axis=1)  # np.nanmean(adcp_HKE_per_mass[:, V_dist_this_year > 100], 1)
dk_adcp = f_ref / ship_c[1]
k_h_adcp = 1e3 * (f_ref / ship_c[1:]) * np.sqrt(dk * avg_KE_adcp[1:] / avg_PE[1:])

# ----- xfers
# --- Use function and iterate over BATS and abaco
TE_spectrum = (avg_PE / dk) + (avg_KE_dg / dk)
PE_per_mass = PE_per_mass[1:, dg_good_prof > 0]
dg_HKE_per_mass = dg_HKE_per_mass[1:, dg_good_prof > 0]
TE_spectrum_per = (PE_per_mass / dk) + (dg_HKE_per_mass / dk)
# find break for every DG profile
start_g = sc_x[5]
min_sp = np.nan * np.ones(PE_per_mass.shape[1])
enst_xfer_per = np.nan * np.ones(PE_per_mass.shape[1])
ener_xfer_per = np.nan * np.ones(PE_per_mass.shape[1])
enst_diss_per = np.nan * np.ones(PE_per_mass.shape[1])
rms_vort_per = np.nan * np.ones(PE_per_mass.shape[1])
# f, ax = plt.subplots()
for i in range(PE_per_mass.shape[1]):
    in_sp = np.transpose(np.concatenate([sc_x[:, np.newaxis], TE_spectrum_per[:, i][:, np.newaxis]], axis=1))
    min_sp[i] = fmin(spectrum_fit, start_g, args=(tuple(in_sp)))

    # check to make sure break is not at too low a wavelength
    if min_sp[i] < 0.03:
        min_sp[i] = np.nan

    if ~np.isnan(min_sp[i]):
        this_TE = TE_spectrum_per[:, i]
        xx = np.log10(sc_x)
        pe = np.log10(this_TE)
        mid_p = np.log10(min_sp[i])
        l_b = np.nanmin(xx)
        r_b = np.nanmax(xx)
        x_grid = np.arange(l_b, r_b, 0.01)
        pe_grid = np.interp(x_grid, xx, pe)
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
        enst_xfer_per[i] = (E0 * ak0 ** 3) ** (3 / 2)
        ener_xfer_per[i] = (E0 * ak0 ** (5 / 3)) ** (3 / 2)
        enst_diss_per[i] = np.sqrt(avg_nu) / (enst_xfer_per[i] ** (1 / 6))
        rms_vort_per[i] = E0 * (ak0 ** 3) * (0.75 * (1 - (sc_x[0] / 1000) / ak0) ** (4/3) + np.log(enst_diss_per[i] / ak0))

#         ax.plot(10**x_grid, 10**fit, color='m')
# ax.set_yscale('log')
# ax.set_xscale('log')
# plot_pro(ax)

# find break for average profile (total)
in_sp = np.transpose(np.concatenate([sc_x[:, np.newaxis], TE_spectrum[1:][:, np.newaxis]], axis=1))
min_sp_avg = fmin(spectrum_fit, start_g, args=(tuple(in_sp)))
this_TE = TE_spectrum[1:]
xx = np.log10(sc_x)
pe = np.log10(this_TE)
mid_p = np.log10(min_sp_avg)
l_b = np.nanmin(xx)
r_b = np.nanmax(xx)
x_grid = np.arange(l_b, r_b, 0.01)
pe_grid = np.interp(x_grid, xx, pe)
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

# fit slope to TE
xx = sc_x.copy()
yy = TE_spectrum[1:]
ipoint = 50  # 8  # 11
x_53 = np.log10(xx[0:ipoint+1])
y_53 = np.log10(yy[0:ipoint+1])
slope1 = np.polyfit(x_53, y_53, 1)
y_g_53 = np.polyval(slope1, x_53)

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
rms_ener = E0 * ak0 * ( -3/2 + 3/2*( (ak0 ** (2/3))*((sc_x[0] / 1000) ** (-2/3))) -
                          0.5 * (ak0 ** 2) * (enst_diss ** -2) + 0.5 * ak0 ** 4)

# --- rhines scale
r_earth = 6371e3  # earth radius [m]
beta_ref = f_ref / (np.tan(np.deg2rad(ref_lat)) * r_earth)
# K_beta = 1 / np.sqrt(np.sqrt(np.sum(avg_KE)) / beta_ref)
K_beta = 1 / np.sqrt(np.sqrt(rms_ener) / beta_ref)
K_beta_2 = 1 / np.sqrt(np.sqrt(np.nanmean(dg_v[:, dg_good_prof > 0]**2)) / beta_ref)
non_linearity = np.sqrt(rms_ener) / (beta_ref * ((c[1] / f_ref) ** 2))

# TESTING ADCP DG/KE DIFFERENCES
plot_energy_limits = 0
if plot_energy_limits > 0:
    f1, ax = plt.subplots()
    PE_ref = ax.plot(sc_x, avg_PE[1:] / dk, color='k', label=r'PE$_{dg}$', linewidth=1)
    KE_adcp = ax.plot(sc_x, avg_KE_adcp[1:] / dk, color='g', label=r'KE$_{adcp}$', linewidth=2)

    KE_dg = ax.plot(sc_x, avg_KE_dg[1:] / dk, color='#FF8C00', label=r'KE$_{1e-4}$', linewidth=2)
    KE_dg_all = ax.plot(sc_x, avg_KE_dg_all[1:] / dk, color='#DAA520', label=r'KE$_{all}$', linewidth=2)
    ax.set_title('KE Noise Threshold Comparison')
    ax.set_xlabel('vertical wavenumber')
    ax.set_ylabel('variance per vert. wavenumber')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.axis([10 ** -2, 1.5 * 10 ** 1, 10 ** (-6), 10 ** (3)])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10)
    ax.axis([10 ** -2, 1.5 * 10 ** 1, 10 ** (-4), 10 ** (3)])
    plot_pro(ax)

plot_energy = 1
# sc_x = np.arange(1, 61)
if plot_energy > 0:
    fig0, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
    # ---- PE
    PE_p = ax0.plot(sc_x, avg_PE[1:] / dk, color='#B22222', label=r'PE$_{dg}$', linewidth=1.5)
    ax0.scatter(sc_x, avg_PE[1:] / dk, color='#B22222', s=15)

    # ---- limits/scales
    # ax0.plot([3 * 10 ** -1, 3 * 10 ** 0], [1.5 * 10 ** 1, 1.5 * 10 ** -2], color='k', linewidth=0.75)
    # ax0.plot([3 * 10 ** -2, 3 * 10 ** -1],
    #          [7 * 10 ** 2, ((5 / 3) * (np.log10(2 * 10 ** -1) - np.log10(2 * 10 ** -2)) + np.log10(7 * 10 ** 2))],
    #          color='k', linewidth=0.75)
    # ax0.text(3.3 * 10 ** -1, 1.3 * 10 ** 1, '-3', fontsize=12)
    # ax0.text(3.3 * 10 ** -2, 6 * 10 ** 2, '-5/3', fontsize=12)
    ax0.plot(sc_x, 0.25 * PE_GM / dk, linestyle='--', color='k', linewidth=1)
    # ax0.text(sc_x[0] - .009, PE_GM[0] / dk, r'$PE_{GM}$', fontsize=12)

    ax0.plot(sc_x, 0.25 * GMPE / dk, color='k', linewidth=0.75)
    ax0.text(sc_x[0] - .01, 0.5 * PE_GM[1] / dk, r'$1/4 PE_{GM}$', fontsize=10)
    ax1.plot(sc_x, 0.25 * GMKE / dk, color='k', linewidth=0.75)
    ax1.text(sc_x[0] - .01, 0.5 * GMKE[1] / dk, r'$1/4 KE_{GM}$', fontsize=10)

    # ---- KE
    # DG
    KE_dg = ax1.plot(sc_x, avg_KE_dg[1:] / dk, color='g', label=r'KE$_{dg}$', linewidth=1.75) # DG
    ax1.scatter(sc_x, avg_KE_dg[1:] / dk, color='g', s=15)
    KE_dg0 = ax1.plot([10**-2, sc_x[0]], avg_KE_dg[0:2] / dk, 'g', linewidth=1.75) # DG KE_0
    ax1.scatter(10**-2, avg_KE_dg[0] / dk, color='g', s=25, facecolors='none')  # DG KE_0
    # ADCP
    # KE_adcp = ax0.plot(sc_x, avg_KE_adcp[1:] / dk_adcp, color='m', label=r'FAR_KE$_{ladcp_v}$', linewidth=1.75) # adcp
    # KE_adcp0 = ax0.plot([10**-2, 1000 * f_ref / c[1]], avg_KE_adcp[0:2] / dk_adcp, 'm', linewidth=1.75) # adcp KE_0
    # ax0.scatter(10**-2, avg_KE_adcp[0] / dk_adcp, color='m', s=25, facecolors='none')  # adcp KE_0
    # total mean ADCP
    adcp_hke_all = np.nanmean(adcp_hke_per[:, far], axis=1)
    adcp_ke_all_cor = adcp_hke_all.copy()
    adcp_ke_all_cor[1:] = adcp_ke_all_cor[1:] - 0
    ax1.plot(sc_x, adcp_ke_all_cor[1:], color='m', label=r'KE$_{ladcp_v}$', linewidth=1.75) # adcp all
    ax1.plot([10**-2, 1000 * f_ref / c[1]], adcp_ke_all_cor[0:2], 'm', linewidth=1.75) # adcp KE_0 all
    ax1.scatter(10**-2, adcp_ke_all_cor[0], color='m', s=25, facecolors='none')  # adcp KE_0 all

    # ---- TE fit
    ax0.plot(sc_x, TE_spectrum[1:], color='c', label=r'TE$_{dg}$', linewidth=1.5)
    # ax0.plot(10 ** x_grid, 10 ** fit_total, color='#FF8C00', label=r'TE$_{fit}$')

    # ---- BATS
    # ax0.plot(sc_x, bats_ke[1:] / bats_dk, color='g', label=r'KE$_{bats}$', linewidth=1.75, linestyle='--')
    # ax0.scatter(sc_x, bats_ke[1:] / bats_dk, color='g', s=15)
    # ax0.plot([8*10**-1, sc_x[0]], bats_ke[0:2] / bats_dk, 'g', linewidth=1.75, linestyle='--')
    # ax0.scatter(8*10**-1, bats_ke[0] / bats_dk, color='g', s=25, facecolors='none')  # DG KE_0
    # ax0.plot(sc_x, bats_pe[1:] / bats_dk, color='#B22222', label=r'PE$_{bats}$', linewidth=1.5, linestyle='--')
    # ax0.scatter(sc_x, bats_pe[1:] / bats_dk, color='#B22222', s=15)

    # break
    # ax0.plot([sc_x[sc_x_break_i], sc_x[sc_x_break_i]], [10 ** (-3), 3 * 10 ** (-3)], color='k', linewidth=2)
    # ax0.text(sc_x[sc_x_break_i] - .4 * 10 ** -2, 5 * 10 ** (-3),
    #          'Break at Mode ' + str(sc_x_break_i) + ' = ' + str(float("{0:.1f}".format(1 / sc_x[sc_x_break_i]))) + 'km',
    #          fontsize=8)
    ax0.plot(10 ** x_53, 10 ** y_g_53, color='b', linewidth=1.25)
    ax0.text(10 ** x_53[0] - .008, 10 ** y_g_53[0], str(float("{0:.2f}".format(slope1[0]))), fontsize=10)

    # --- plot tailoring
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax1.set_xscale('log')
    # ax0.axis([8 * 10 ** -1, 10 ** 2, 1 * 10 ** (-3), 4 * 10 ** (3)])
    # ax0.set_xlabel('Mode Number (barotropic mode plotted along y-axis)', fontsize=13)
    ax0.axis([10 ** -2, 10 ** 1, 1 * 10 ** (-3), 4 * 10 ** (3)])
    ax1.set_xlim([10 ** -2, 10 ** 1])
    ax0.set_xlabel(r'Scaled Vertical Wavenumber = (Rossby Radius)$^{-1}$ = $\frac{f}{c}$ [$km^{-1}$]', fontsize=13)
    ax0.set_ylabel('Spectral Density', fontsize=13)  # '(and Hor. Wavenumber)',fontsize=13)
    ax0.set_title(x.project + ' - PE Spectra', fontsize=12)
    ax1.set_title(x.project + ' - KE Spectra', fontsize=12)
    ax1.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=14)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, fontsize=10)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=10)
    ax0.grid()
    plot_pro(ax1)

    # -------
    # --  dynamic eta and v mode amplitudes with time --
    # time_ord = np.argsort(dg_eta_time)
    # # time_label = datetime.date.fromordinal(np.int(time_rec))
    # myFmt = matplotlib.dates.DateFormatter('%m/%d')
    # mode_range = [1, 2, 3]
    # fig0, ((ax0), (ax1)) = plt.subplots(2, 1, sharex=True)
    # ax0.plot([time_min, time_max], [0, 0], 'k', linewidth=1.5)
    # for i in mode_range:
    #     ax_i = ax0.plot(dg_eta_time, AG[i, :], label='Baroclinic Mode ' + np.str(i))
    #     # ax0.scatter(time_rec, AG[i, :])
    # ax0.xaxis.set_major_formatter(myFmt)
    # handles, labels = ax0.get_legend_handles_labels()
    # ax0.set_title('Vertical Displacement Baroclinic Mode Amplitude Temporal Variability')
    # ax0.set_ylabel(r'Unscaled Dynamic Mode ($\beta_m$) Amplitude')
    # ax0.legend(handles, labels, fontsize=10)
    # ax0.set_ylim([-.1, .1])
    # ax0.grid()
    #
    # ax1.plot([time_min, time_max], [0, 0], 'k', linewidth=1.5)
    # mode_range = [0, 1, 2]
    # for i in mode_range:
    #     ax_i = ax1.plot(dg_v_time, dg_AGz[i, :], label='Baroclinic Mode ' + np.str(i))
    #     # ax0.scatter(time_rec, AG[i, :])
    # ax1.xaxis.set_major_formatter(myFmt)
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.set_title('Velocity Baroclinic Mode Amplitude Temporal Variability')
    # ax1.set_xlabel('Date')
    # ax1.set_ylabel(r'Unscaled Dynamic Mode ($\alpha_m$) Amplitude')
    # ax1.legend(handles, labels, fontsize=10)
    # ax0.set_ylim([-.2, .2])
    # plot_pro(ax1)

    # -------
    # estimate of horizontal wavenumber
    fig0, ax0 = plt.subplots()
    k_h_p = ax0.plot(sc_x, k_h, color='g', label=r'DG$_{k_h}$')
    k_h_p = ax0.plot(sc_x, k_h_adcp, color='m', label=r'LADCP$_{k_h}$')
    ax0.plot([10**-2, 10**1], 1e3 * np.array([K_beta_2, K_beta_2]), color='k', linestyle='-.')
    ax0.text(1.1, 0.02, r'k$_{Rhines}$', fontsize=12)
    ax0.plot(sc_x, sc_x, 'k', linestyle='--')
    ax0.set_xlabel([10 ** -2, 10 ** 1])
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.set_xlabel(r'Scaled Vertical Wavenumber = (Rossby Radius)$^{-1}$ = $\frac{f}{c}$ [km$^{-1}$]', fontsize=11)
    ax0.set_ylabel(r'Horizontal Wavenumber [km$^{-1}$]')
    ax0.set_title('Estimates of Hor. Scales from DG KE and LADCP KE')
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, fontsize=10)
    ax0.set_xlim([10 ** -2, 10 ** 1])
    ax0.set_ylim([1 * 10 ** (-2), 1 * 10 ** (1)])
    plot_pro(ax0)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------ EOF MODE SHAPESSSSS
# - find EOFs of dynamic horizontal current (v) mode amplitudes _____DG_____
AGzq = dg_AGz[:, dg_good_prof > 0]
nq = np.sum(dg_good_prof > 0)  # good_prof and dg_good_prof
avg_AGzq = np.nanmean(np.transpose(AGzq), axis=0)
AGzqa = AGzq - np.transpose(np.tile(avg_AGzq, [nq, 1]))  # mode amplitude anomaly matrix
cov_AGzqa = (1 / nq) * np.matrix(AGzqa) * np.matrix(np.transpose(AGzqa))  # nmodes X nmodes covariance matrix
var_AGzqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa)))) * np.matrix(np.sqrt(np.diag(cov_AGzqa)))
cor_AGzqa = cov_AGzqa / var_AGzqa  # nmodes X nmodes correlation matrix
D_AGzqa, V_AGzqa = np.linalg.eig(cov_AGzqa)  # columns of V_AGzqa are eigenvectors
EOFseries = np.transpose(V_AGzqa) * np.matrix(AGzqa)  # EOF "timeseries' [nmodes X nq]
EOFshape_dg = np.matrix(Gz) * V_AGzqa  # depth shape of eigenfunctions [ndepths X nmodes]
EOFshape1_BTpBC1 = np.matrix(Gz[:, 0:2]) * V_AGzqa[0:2, 0]  # truncated 2 mode shape of EOF#1
EOFshape2_BTpBC1 = np.matrix(Gz[:, 0:2]) * V_AGzqa[0:2, 1]  # truncated 2 mode shape of EOF#2

# -----------------------------------------------------------------------------------------
# -- find EOFs of dynamic horizontal current (v) mode amplitudes _____ADCP_____
AGzq = adcp_AGz  # (:,quiet_prof)
nq = np.size(adcp_good_prof)  # good_prof and dg_good_prof
avg_AGzq = np.nanmean(np.transpose(AGzq), axis=0)
AGzqa = AGzq - np.transpose(np.tile(avg_AGzq, [nq, 1]))  # mode amplitude anomaly matrix
cov_AGzqa = (1 / nq) * np.matrix(AGzqa) * np.matrix(np.transpose(AGzqa))  # nmodes X nmodes covariance matrix
var_AGzqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa)))) * np.matrix(np.sqrt(np.diag(cov_AGzqa)))
cor_AGzqa = cov_AGzqa / var_AGzqa  # nmodes X nmodes correlation matrix
D_AGzqa, V_AGzqa = np.linalg.eig(cov_AGzqa)  # columns of V_AGzqa are eigenvectors
EOFseries = np.transpose(V_AGzqa) * np.matrix(AGzqa)  # EOF "timeseries' [nmodes X nq]
EOFshape_adcp = np.matrix(ship_Gz) * V_AGzqa  # depth shape of eigenfunctions [ndepths X nmodes]
EOFshape1_BTpBC1 = np.matrix(ship_Gz[:, 0:2]) * V_AGzqa[0:2, 0]  # truncated 2 mode shape of EOF#1
EOFshape2_BTpBC1 = np.matrix(ship_Gz[:, 0:2]) * V_AGzqa[0:2, 1]  # truncated 2 mode shape of EOF#2

# -----------------------------------------------------------------------------------------
# - find EOFs of dynamic vertical displacement (eta) mode amplitudes
# extract noisy/bad profiles 
good_prof = np.where(~np.isnan(AG[2, :]))
num_profs_2 = np.size(good_prof)
AG2 = AG[:, good_prof[0]]
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

# --- plot mode shapes
plot_mode_shapes = 1
if plot_mode_shapes > 0:
    max_plot = 3
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    n2p = ax1.plot((np.sqrt(this_ship_N2) * (1800 / np.pi)) / 2, ship_depth_0, color='k', label='N(z) [cph]')
    colors = plt.cm.Dark2(np.arange(0, 4, 1))

    for ii in range(max_plot):
        ax1.plot(ship_Gz[:, ii], ship_depth_1, color='#2F4F4F', linestyle='--')
        p_eof = ax1.plot(-EOFshape_adcp[:, ii], ship_depth_1, color=colors[ii, :], label='EOF # = ' + str(ii + 1), linewidth=2)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=10)
    ax1.axis([-5, 5, 0, 5100])
    ax1.invert_yaxis()
    ax1.set_title('ABACO EOF Mode Shapes (ADCP)')
    ax1.set_ylabel('Depth [m]')
    ax1.set_xlabel('Hor. Vel. Mode Shapes (ADCP)')
    ax1.grid()

    for ii in range(max_plot):
        ax2.plot(Gz[:, ii], grid, color='#2F4F4F', linestyle='--')
        if ii < 1:
            p_eof = ax2.plot(-EOFshape_dg[:, ii], grid, color=colors[ii, :], label='EOF # = ' + str(ii + 1), linewidth=2)
        elif (ii > 0) & (ii < 2):
            p_eof = ax2.plot(EOFshape_dg[:, ii], grid, color=colors[ii, :], label='EOF # = ' + str(ii + 1), linewidth=2)
        else:
            p_eof = ax2.plot(-EOFshape_dg[:, ii], grid, color=colors[ii, :], label='EOF # = ' + str(ii + 1), linewidth=2)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=10)
    ax2.axis([-5, 5, 0, 5100])
    ax2.invert_yaxis()
    ax2.set_title('EOF Mode Shapes (DG)')
    ax2.set_xlabel('Hor. Vel. Mode Shapes (DG)')
    ax2.grid()

    for ii in range(max_plot):
        ax3.plot(G[:, ii + 1] / np.max(grid), grid, color='#2F4F4F', linestyle='--')
        p_eof_eta = ax3.plot(-EOFetashape[:, ii] / np.max(grid), grid, color=colors[ii, :],
                             label='EOF # = ' + str(ii + 1), linewidth=2)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, fontsize=10)
    ax3.axis([-.7, .7, 0, 5100])
    ax3.set_title('EOF Mode Shapes (DG)')
    ax3.set_xlabel('Vert. Disp. Mode Shapes')
    ax3.invert_yaxis()
    plot_pro(ax3)

# --- SAVE
# write python dict to a file
savee = 0
if savee > 0:
    mydict = {'bin_depth': grid, 'sigma_theta': df_den, 'salin': df_s, 'theta': df_t, 'eta': df_eta,
              'eta_m': Eta_m, 'avg_PE': avg_PE, 'avg_KE': avg_KE_dg, 'ADCP_KE': avg_KE_adcp,
              'k_h': k_h, 'f_ref': f_ref, 'c': c, 'G': G, 'Gz': Gz}
    output = open('/Users/jake/Documents/geostrophic_turbulence/ABACO_2017_energy.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# LADCP PER YEAR
# time, depth, distance, evolution of meridional flow

lv = np.arange(-65, 65, 5)
lv2 = np.arange(-60, 60, 10)
cmap = matplotlib.cm.get_cmap('viridis')
# f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharex='col', sharey='row')
# ax1.contourf(adcp_dist[per_year[0]], adcp_depth, adcp_v[:, per_year[0]], levels=lv, cmap=cmap)
# ax1.contour(adcp_dist[per_year[0]], adcp_depth, adcp_v[:, per_year[0]], levels=lv2, colors='k')
# ax1.axis([0, 400, 0, 5500])
# ax1.set_title(str(np.round(adcp_datetime[per_year[0][0]], 2)), fontsize=12)
# ax2.contourf(adcp_dist[per_year[1]], adcp_depth, adcp_v[:, per_year[1]], levels=lv, cmap=cmap)
# ax2.contour(adcp_dist[per_year[1]], adcp_depth, adcp_v[:, per_year[1]], levels=lv2, colors='k')
# ax2.set_title(str(np.round(adcp_datetime[per_year[1][0]], 2)), fontsize=12)
# ax2.axis([0, 400, 0, 5500])
# ax3.contourf(adcp_dist[per_year[2]], adcp_depth, adcp_v[:, per_year[2]], levels=lv, cmap=cmap)
# ax3.contour(adcp_dist[per_year[2]], adcp_depth, adcp_v[:, per_year[2]], levels=lv2, colors='k')
# ax3.set_title(str(np.round(adcp_datetime[per_year[2][0]], 2)), fontsize=12)
# ax3.axis([0, 400, 0, 5500])
# ax4.contourf(adcp_dist[per_year[3]], adcp_depth, adcp_v[:, per_year[3]], levels=lv, cmap=cmap)
# ax4.contour(adcp_dist[per_year[3]], adcp_depth, adcp_v[:, per_year[3]], levels=lv2, colors='k')
# ax4.set_title(str(np.round(adcp_datetime[per_year[3][0]], 2)), fontsize=12)
# ax4.axis([0, 400, 0, 5500])
# ax3.invert_yaxis()
#
# ax5.contourf(adcp_dist[per_year[4]], adcp_depth, adcp_v[:, per_year[4]], levels=lv, cmap=cmap)
# ax5.contour(adcp_dist[per_year[4]], adcp_depth, adcp_v[:, per_year[4]], levels=lv2, colors='k')
# ax5.set_title(str(np.round(adcp_datetime[per_year[4][0]], 2)), fontsize=12)
# ax5.axis([0, 400, 0, 5500])
# ax5.set_xlabel('Distance East [km]')
# ax6.contourf(adcp_dist[per_year[5]], adcp_depth, adcp_v[:, per_year[5]], levels=lv, cmap=cmap)
# ax6.contour(adcp_dist[per_year[5]], adcp_depth, adcp_v[:, per_year[5]], levels=lv2, colors='k')
# ax6.set_title(str(np.round(adcp_datetime[per_year[5][0]], 2)), fontsize=12)
# ax6.axis([0, 400, 0, 5500])
# ax6.set_xlabel('Distance East [km]')
# ax7.contourf(adcp_dist[per_year[6]], adcp_depth, adcp_v[:, per_year[6]], levels=lv, cmap=cmap)
# ax7.contour(adcp_dist[per_year[6]], adcp_depth, adcp_v[:, per_year[6]], levels=lv2, colors='k')
# ax7.set_title(str(np.round(adcp_datetime[per_year[6][0]], 2)), fontsize=12)
# ax7.axis([0, 400, 0, 5500])
# ax7.set_xlabel('Distance East [km]')
# ax8.contourf(adcp_dist[per_year[7]], adcp_depth, adcp_v[:, per_year[7]], levels=lv, cmap=cmap)
# ax8.contour(adcp_dist[per_year[7]], adcp_depth, adcp_v[:, per_year[7]], levels=lv2, colors='k')
# ax8.set_title(str(np.round(adcp_datetime[per_year[7][0]], 2)), fontsize=12)
# ax8.axis([0, 400, 0, 5500])
# ax8.set_xlabel('Distance East [km]')
# ax8.invert_yaxis()
#
# c_map_ax = f.add_axes([0.923, 0.1, 0.02, 0.8])
# norm = matplotlib.colors.Normalize(vmin=lv.min(), vmax=lv.max())
# cb1 = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=cmap, norm=norm, orientation='vertical')
# cb1.set_label('cm/s North')
# ax6.grid()
# plot_pro(ax6)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.contourf(adcp_dist[per_year[3]], adcp_depth, adcp_v[:, per_year[3]], levels=lv, cmap=cmap)
ax1.contour(adcp_dist[per_year[3]], adcp_depth, adcp_v[:, per_year[3]], levels=lv2, colors='k')
ax1.set_title(str(np.int(adcp_datetime[per_year[3][0]])), fontsize=12)
ax1.axis([30, 300, 0, 5000])
ax1.set_xlabel('Distance East [km]')
ax2.contourf(adcp_dist[per_year[6]], adcp_depth, adcp_v[:, per_year[6]], levels=lv, cmap=cmap)
ax2.contour(adcp_dist[per_year[6]], adcp_depth, adcp_v[:, per_year[6]], levels=lv2, colors='k')
ax2.set_title(str(np.int(adcp_datetime[per_year[6][0]])), fontsize=12)
ax2.axis([30, 300, 0, 5000])
ax2.set_xlabel('Distance East [km]')
ax3.contourf(adcp_dist[per_year[7]], adcp_depth, adcp_v[:, per_year[7]], levels=lv, cmap=cmap)
ax3.contour(adcp_dist[per_year[7]], adcp_depth, adcp_v[:, per_year[7]], levels=lv2, colors='k')
ax3.set_title(str(np.int(adcp_datetime[per_year[7][0]])), fontsize=12)
ax3.axis([30, 300, 0, 5000])
ax3.set_xlabel('Distance East [km]')
ax1.set_ylabel('Depth [m]')
ax3.invert_yaxis()
ax3.grid()
plot_pro(ax3)

# -----------------------------------------
# # --- SHIP ADCP HKE est.
#
# # Shipboard CTD N2
# ship_p = gsw.p_from_z(-1 * ship_depth_0, ref_lat)
# ship_N2 = np.nan*np.ones(len(ship_p))
#
# ship_years = np.unique(time_key)
# f, (ax0, ax1) = plt.subplots(1, 2)
# c2_n2 = plt.cm.plasma(np.linspace(0, 1, len(ship_years) + 1))
# adcp_v_fit = np.nan*np.ones((len(ship_depth_0), len(adcp_dist)))
# adcp_hke_per = np.nan*np.ones((len(nmodes), len(adcp_dist)))
# for i in range(len(ship_years) + 1):
#     ship_dd = ship_dist[time_key == ship_years[i]]
#
#     # fill 2018 with density profiles from 2017
#     if i < len(ship_years):
#         ship_N2[0:-1] = gsw.Nsquared(np.nanmean(ship_SA[:, time_key == ship_years[i]], axis=1),
#                                     np.nanmean(ship_CT[:, time_key == ship_years[i]], axis=1),
#                                     ship_p, lat=ref_lat)[0]
#     else:
#         ship_N2[0:-1] = gsw.Nsquared(np.nanmean(ship_SA[:, time_key == ship_years[i - 1]], axis=1),
#                                     np.nanmean(ship_CT[:, time_key == ship_years[i - 1]], axis=1),
#                                     ship_p, lat=ref_lat)[0]
#
#     # taper N2
#     if i < 1:
#         ship_N2[2] = ship_N2[3] - 5*10**(-5)
#         ship_N2[1] = ship_N2[2] - 5*10**(-5)
#         ship_N2[0] = ship_N2[1] - 5*10**(-5)
#     else:
#         ship_N2[1] = ship_N2[2] - 2*10**(-5)
#         ship_N2[0] = ship_N2[1] - 2*10**(-5)
#     ship_N2[ship_N2 < 0] = np.nan
#     for j in np.where(np.isnan(ship_N2))[0]:
#         ship_N2[j] = ship_N2[j - 1] - 1*10**(-8)
#     # ship_N2 = savgol_filter(ship_N2, 5, 3)
#
#     # adcp time in
#     adcp_v_iter = adcp_v[:, per_year[i]]
#
#     # find adcp profiles that are deep enough and fit baroclinic modes to these
#     HKE_noise_threshold_adcp = 1e-5
#     check = np.zeros(np.size(adcp_dist[per_year[i]]))
#     for k in range(np.size(adcp_dist[per_year[i]])):
#         check[k] = adcp_depth[np.where(~np.isnan(adcp_v_iter[:, k]))[0][-1]]
#     adcp_in = np.where((check >= 4700) & (check <= 5300))
#     V = adcp_v_iter[:, adcp_in[0]] / 100
#     V_dist = adcp_dist[per_year[i]][adcp_in[0]]
#     adcp_np = np.size(V_dist)
#     ship_depth_1 = ship_depth_0[0:-39]
#
#     ship_G, ship_Gz, ship_c, ship_epsilon = vertical_modes(ship_N2[0:-39], ship_depth_1, omega, mmax)
#
#     dk = f_ref / ship_c[1]
#     sc_x = 1000 * f_ref / ship_c[1:]
#
#     adcp_AGz = np.zeros([nmodes, adcp_np])
#     V_m = np.nan * np.zeros([np.size(ship_depth_1), adcp_np])
#     HKE_per_mass = np.nan * np.zeros([nmodes, adcp_np])
#     modest = np.arange(11, nmodes)
#     adcp_good_prof = np.zeros(adcp_np)
#     for ii in range(adcp_np):
#         # fit to velocity profiles
#         this_V_0 = V[:, ii].copy()
#         this_V = np.interp(ship_depth_1, adcp_depth, this_V_0) # new using Gz profiles from shipboard ctd
#         # this_V = np.interp(grid, adcp_depth, this_V_0)  # old using Gz profiles from DG
#         iv = np.where(~np.isnan(this_V))
#         if iv[0].size > 1:
#             # Gz(iv,:)\V_g(iv,ip)
#             # Gz*AGz[:,i];
#             adcp_AGz[:, ii] = np.squeeze(np.linalg.lstsq(np.squeeze(ship_Gz[iv, :]),
#                                                         np.transpose(np.atleast_2d(this_V[iv])))[0])
#             V_m[:, ii] = np.squeeze(np.matrix(ship_Gz) * np.transpose(np.matrix(adcp_AGz[:, ii])))
#             HKE_per_mass[:, ii] = adcp_AGz[:, ii] * adcp_AGz[:, ii]
#             ival = np.where(HKE_per_mass[modest, ii] >= HKE_noise_threshold_adcp)
#             if np.size(ival) > 0:
#                 adcp_good_prof[ii] = 1  # flag profile as noisy
#         else:
#             adcp_good_prof[ii] = 1  # flag empty profile as noisy as well
#
#     # -- output
#     if i < 1:
#         adcp_v_fit = V_m
#         adcp_hke_per = HKE_per_mass
#     else:
#         adcp_v_fit = np.concatenate((adcp_v_fit, V_m), axis=1)
#         adcp_hke_per = np.concatenate((adcp_hke_per, HKE_per_mass), axis=1)
#
#     # -- plotting
#     for jj in range(adcp_np):
#         ax1.plot(V[:, jj], adcp_depth, linewidth=0.75, color=c2_n2[i, :])
#         ax1.plot(V_m[:, jj], ship_depth_1, color='k', linewidth=0.5)
#
#     avg_KE_adcp_i = np.nanmean(HKE_per_mass, 1)
#     if i < len(ship_years):
#         KE_adcp = ax0.plot(sc_x, avg_KE_adcp_i[1:] / dk, label=np.int(ship_years[i]), linewidth=1.75, color=c2_n2[i, :])
#         ax0.scatter(sc_x, avg_KE_adcp_i[1:] / dk, s=20, color=c2_n2[i, :])  # adcp
#     else:
#         KE_adcp = ax0.plot(sc_x, avg_KE_adcp_i[1:] / dk, label='2018', linewidth=1.75, color=c2_n2[i, :])  # adcp
#         ax0.scatter(sc_x, avg_KE_adcp_i[1:] / dk, s=20, color=c2_n2[i, :])  # adcp
#     # adcp KE_0
#     KE_adcp0 = ax0.plot([10 ** -2, 1000 * f_ref / ship_c[1]], avg_KE_adcp_i[0:2] / dk, linewidth=1.75, color=c2_n2[i, :])
#     ax0.scatter(10 ** -2, avg_KE_adcp_i[0] / dk, s=25, facecolors='none', color=c2_n2[i, :])  # adcp KE_0
#
# ax0.set_yscale('log')
# ax0.set_xscale('log')
# ax0.axis([10 ** -2, 10 ** 1, 10 ** (-3), 4 * 10 ** 3])
# # ax0.axis([10 ** -2, 10 ** 1, 10 ** (-4), 10 ** 3])
# ax0.set_xlabel(r'Scaled Vertical Wavenumber = (Rossby Radius)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=18)
# ax0.set_ylabel('Spectral Density', fontsize=18)  # ' (and Hor. Wavenumber)')
# ax0.set_title('LADCP_v KE Spectrum', fontsize=18)
# handles, labels = ax0.get_legend_handles_labels()
# ax0.legend(handles, labels, fontsize=12)
# ax0.grid()
# ax1.set_xlim([-0.6, 0.6])
# ax1.invert_yaxis()
# plot_pro(ax1)
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# # --- comparisons of average density displacement profiles
# # load in Station HOTs eta Comparison
# SH = si.loadmat('/Users/jake/Desktop/bats/station_hots_pe.mat')
# sta_hots_depth = SH['out'][0][0][0]
# sta_hots_eta = SH['out'][0][0][1]
#
# # load bats eta
# pkl_file = open('/Users/jake/Desktop/bats/den_v_profs.pkl', 'rb')
# bats_eta_prof = pickle.load(pkl_file)
# pkl_file.close()
#
# window_size, poly_order = 31, 5
# eta_filt = np.zeros(np.shape(df_eta))
# for i in range(num_profs):
#     eta_filt[:, i] = savgol_filter(df_eta.iloc[:, i], window_size, poly_order)
# bats_eta_filt = np.zeros(np.shape(bats_eta_prof['eta']))
# for i in range(bats_eta_prof['eta'].shape[1]):
#     bats_eta_filt[:, i] = savgol_filter(bats_eta_prof['eta'][:, i], window_size, poly_order)
#
# abaco_avg_eta = np.nanmean(np.abs(eta_filt), axis=1)
# bats_avg_eta = np.nanmean(np.abs(bats_eta_filt), axis=1)
# hots_avg_eta = np.nanmean(np.abs(sta_hots_eta), axis=1)
#
# abaco_std_eta = np.nanstd(eta_filt, axis=1)
# bats_std_eta = np.nanstd(bats_eta_filt, axis=1)
# hots_std_eta = np.nanstd(sta_hots_eta, axis=1)
#
# a_a_e_2 = savgol_filter(abaco_avg_eta, window_size, poly_order)
# b_a_e_2 = savgol_filter(bats_avg_eta, window_size, poly_order)
# h_a_e_2 = savgol_filter(hots_avg_eta, window_size, poly_order)
# a_a_std_2 = savgol_filter(abaco_std_eta, window_size, poly_order)
# b_a_std_2 = savgol_filter(bats_std_eta, window_size, poly_order)
# h_a_std_2 = savgol_filter(hots_std_eta, window_size, poly_order)

# ---------------------------------------------------------------------------------------------------------------------
# REPLACED BY GLIDER PACKAGE TRANSECT ANALYSIS
# # loop over each dive
# count = 0
# for i in dg_list:
#     nc_fid = Dataset(i, 'r')
#     glid_num = nc_fid.glider
#     dive_num = nc_fid.dive_number
#
#     lat = nc_fid.variables['latitude'][:]
#     lon = nc_fid.variables['longitude'][:]
#     x = (lon - lon_in) * (1852 * 60 * np.cos(np.deg2rad(26.5)))
#     y = (lat - lat_in) * (1852 * 60)
#
#     press = nc_fid.variables['ctd_pressure'][:]
#     ctd_epoch_time = nc_fid.variables['ctd_time'][:]
#     temp = nc_fid.variables['temperature'][:]
#     salin = nc_fid.variables['salinity'][:]
#     pitch_ang = nc_fid.variables['eng_pitchAng'][:]
#     depth = -1 * gsw.z_from_p(press, ref_lat)
#     SA = gsw.SA_from_SP(salin, press, ref_lon * np.ones(len(salin)), ref_lat * np.ones(len(salin)))
#     CT = gsw.CT_from_t(SA, temp, press)
#
#     # time conversion
#     secs_per_day = 86400.0
#     dive_start_time = nc_fid.start_time
#     ctd_time = ctd_epoch_time - dive_start_time
#     datenum_start = 719163  # jan 1 1970
#
#     # put on distance/density grid (1 column per profile)
#     max_d = np.where(depth == depth.max())
#     max_d_ind = max_d[0]
#     pitch_ang_sub1 = pitch_ang[0:max_d_ind[0]]
#     pitch_ang_sub2 = pitch_ang[max_d_ind[0]:]
#     dive_mask_i = np.where(pitch_ang_sub1 < 0)
#     dive_mask = dive_mask_i[0][:]
#     climb_mask_i = np.where(pitch_ang_sub2 > 0)
#     climb_mask = climb_mask_i[0][:] + max_d_ind[0] - 1
#
#     # dive/climb time midpoints
#     time_dive = ctd_time[dive_mask]
#     time_climb = ctd_time[climb_mask]
#
#     serial_date_time_dive = datenum_start + dive_start_time / (60 * 60 * 24) + np.median(time_dive) / secs_per_day
#     serial_date_time_climb = datenum_start + dive_start_time / (60 * 60 * 24) + np.median(time_climb) / secs_per_day
#     time_rec_2[count, :] = np.array([serial_date_time_dive, serial_date_time_climb])
#     if np.sum(time_rec) < 1:
#         time_rec = np.concatenate([[serial_date_time_dive], [serial_date_time_climb]])
#     else:
#         time_rec = np.concatenate([time_rec, [serial_date_time_dive], [serial_date_time_climb]])
#
#     # interpolate (bin_average) to smooth and place T/S on vertical depth grid
#     CT_grid_dive, CT_grid_climb, SA_grid_dive, SA_grid_climb, x_grid_dive, x_grid_climb, \
#     y_grid_dive, y_grid_climb = make_bin(grid, depth[dive_mask], depth[climb_mask], CT[dive_mask], CT[climb_mask],
#                                          SA[dive_mask], SA[climb_mask], x[dive_mask], x[climb_mask],
#                                          y[dive_mask], y[climb_mask])
#
#     # theta_grid_dive = sw.ptmp(salin_grid_dive, temp_grid_dive, grid_p, pr=0)
#     # theta_grid_climb = sw.ptmp(salin_grid_climb, temp_grid_climb, grid_p, pr=0)
#
#     # compute distance to closest point on transect
#     dist_dive = np.zeros(np.size(x_grid_dive))
#     dist_climb = np.zeros(np.size(y_grid_dive))
#     for j in range(np.size(x_grid_dive)):
#         all_dist_dive = np.sqrt((x_grid_dive[j] - dist_grid_s) ** 2 + (y_grid_dive[j] - 0) ** 2)
#         all_dist_climb = np.sqrt((x_grid_climb[j] - dist_grid_s) ** 2 + (y_grid_climb[j] - 0) ** 2)
#         # dive
#         if np.isnan(x_grid_dive[j]):
#             dist_dive[j] = float('nan')
#         else:
#             closest_dist_dive_i = np.where(all_dist_dive == all_dist_dive.min())
#             dist_dive[j] = dist_grid_s[closest_dist_dive_i[0]]
#         # climb
#         if np.isnan(x_grid_climb[j]):
#             dist_climb[j] = float('nan')
#         else:
#             closest_dist_climb_i = np.where(all_dist_climb == all_dist_climb.min())
#             dist_climb[j] = dist_grid_s[closest_dist_climb_i[0]]
#
#     # create dataframes where each column is a profile
#     t_data_d = pd.DataFrame(CT_grid_dive, index=grid, columns=[glid_num * 1000 + dive_num])
#     t_data_c = pd.DataFrame(CT_grid_climb, index=grid, columns=[glid_num * 1000 + dive_num + .5])
#     s_data_d = pd.DataFrame(SA_grid_dive, index=grid, columns=[glid_num * 1000 + dive_num])
#     s_data_c = pd.DataFrame(SA_grid_climb, index=grid, columns=[glid_num * 1000 + dive_num + .5])
#     d_data_d = pd.DataFrame(dist_dive, index=grid, columns=[glid_num * 1000 + dive_num])
#     d_data_c = pd.DataFrame(np.flipud(dist_climb), index=grid, columns=[glid_num * 1000 + dive_num + .5])
#
#     if df_t.size < 1:
#         df_t = pd.concat([t_data_d, t_data_c], axis=1)
#         df_s = pd.concat([s_data_d, s_data_c], axis=1)
#         df_d = pd.concat([d_data_d, d_data_c], axis=1)
#     else:
#         df_t = pd.concat([df_t, t_data_d, t_data_c], axis=1)
#         df_s = pd.concat([df_s, s_data_d, s_data_c], axis=1)
#         df_d = pd.concat([df_d, d_data_d, d_data_c], axis=1)
#
#     # if interpolating on a depth grid, interpolate density
#     if grid[10] - grid[9] > 1:
#         # sw.pden(salin_grid_dive, temp_grid_dive, grid_p, pr=0) - 1000
#         den_grid_dive = gsw.sigma0(SA_grid_dive, CT_grid_dive)
#         den_grid_climb = gsw.sigma0(SA_grid_climb, CT_grid_climb)
#         den_data_d = pd.DataFrame(den_grid_dive, index=grid, columns=[glid_num * 1000 + dive_num])
#         den_data_c = pd.DataFrame(den_grid_climb, index=grid, columns=[glid_num * 1000 + dive_num + .5])
#         if df_t.size < 1:
#             df_den = pd.concat([den_data_d, den_data_c], axis=1)
#         else:
#             df_den = pd.concat([df_den, den_data_d, den_data_c], axis=1)
#
#     # plot plan view action if needed
#     if plot_plan > 0:
#         if glid_num > 37:
#             dg = ax0.scatter(1000 * x_grid_dive / (1852 * 60 * np.cos(np.deg2rad(26.5))) + lon_in,
#                               1000 * y_grid_dive / (1852 * 60) + lat_in, s=2, color='#FFD700', zorder=1, label='DG38')
#             ax0.scatter(1000 * x_grid_climb / (1852 * 60 * np.cos(np.deg2rad(26.5))) + lon_in,
#                         1000 * y_grid_climb / (1852 * 60) + lat_in, s=2, color='#FFD700', zorder=1)
#         else:
#             dg = ax0.scatter(1000 * x_grid_dive / (1852 * 60 * np.cos(np.deg2rad(26.5))) + lon_in,
#                               1000 * y_grid_dive / (1852 * 60) + lat_in, s=2, color='#B22222', zorder=1, label='DG37')
#             ax0.scatter(1000 * x_grid_climb / (1852 * 60 * np.cos(np.deg2rad(26.5))) + lon_in,
#                         1000 * y_grid_climb / (1852 * 60) + lat_in, s=2, color='#B22222', zorder=1)
#         ax0.scatter(1000 * dist_dive / (1852 * 60 * np.cos(np.deg2rad(26.5))) + lon_in,
#                     np.zeros(np.size(dist_dive)) + lat_in, s=0.75, color='k')
#         ax0.scatter(1000 * dist_climb / (1852 * 60 * np.cos(np.deg2rad(26.5))) + lon_in,
#                     np.zeros(np.size(dist_climb)) + lat_in, s=0.75, color='k')
#         ax0.text(1000 * (np.median(dist_grid) - 20) / (1852 * 60 * np.cos(np.deg2rad(26.5))) + lon_in,
#                  lat_in - .3, '~ 125km', fontsize=12, color='w')
#     count = count + 1
#
# # --------------- end of for loop running over each dive
# if plot_plan > 0:
#     sp = ax0.scatter(abaco_ship['cast_lon'], abaco_ship['cast_lat'], s=4, color='#7CFC00', label='Ship')
#     t_s = datetime.date.fromordinal(np.int(np.min(time_rec[0])))
#     t_e = datetime.date.fromordinal(np.int(np.max(time_rec[-1])))
#     ax0.set_title('Nine ABACO Transects (DG37,38 - 57 dive-cycles): ' +
#                   np.str(t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day))
#     handles, labels = ax0.get_legend_handles_labels()
#     ax0.legend(handles=[dg, sp])  # ,[np.unique(labels)],fontsize=10)
#     plt.tight_layout()
#     plot_pro(ax0)
#     # fig0.savefig('/Users/jake/Desktop/abaco/plan_2.png', dpi=200)
#     # plt.close()

# ------------ OLD DG Geostrophic Velocity Profiles
# import_ke = si.loadmat('/Users/jake/Documents/geostrophic_turbulence/ABACO_V_energy.mat')
# ke_data = import_ke['out']
# ke = ke_data['KE'][0][0]
# dg_bin = ke_data['bin_depth'][0][0]
# dg_v_0 = ke_data['V_g'][0][0]


# --------------------------------------------------------------------------------------------------------
# # deep oxygen selection
# dep_max = np.nan*np.ones(np.shape(ship_o2_1)[1])
# for i in range(1, np.shape(ship_o2_1)[1]):
#     deep_max = np.where(~np.isnan(ship_o2_1[10:, i]))[0]
#     dep_max[i] = abaco_ship['bin_depth'][deep_max[-1]]
#
# time_key_deep = time_key[(dep_max >= 4200) & (ship_lon > -76) & (ship_lon < -74)]
# ship_o2_1_deep = ship_o2_1[:, (dep_max >= 4200) & (ship_lon > -76) & (ship_lon < -74)]
# ship_o2_2_deep = ship_o2_2[:, (dep_max >= 4200) & (ship_lon > -76) & (ship_lon < -74)]
# ship_lat_deep = ship_lat[(dep_max >= 4200) & (ship_lon > -76) & (ship_lon < -74)]
# ship_lon_deep = ship_lon[(dep_max >= 4200) & (ship_lon > -76) & (ship_lon < -74)]
#
# from scipy.io import netcdf
# f = netcdf.netcdf_file('/Users/jake/Desktop/ABACO_shipboard_o2.nc', 'w')
# f.history = 'NOAA AOML Western Boundary Current CTD Casts'
# f.createDimension('depth', np.size(abaco_ship['bin_depth']))
# f.createDimension('year', np.size(time_key_deep))
# b_d = f.createVariable('depth', np.float64, ('depth',))
# b_d[:] = abaco_ship['bin_depth']
# b_d = f.createVariable('year', np.float64, ('year',))
# b_d[:] = time_key_deep
# b_d = f.createVariable('O2_1', np.float64, ('depth', 'year'))
# b_d[:] = ship_o2_1_deep
# b_d = f.createVariable('O2_2', np.float64, ('depth', 'year'))
# b_d[:] = ship_o2_2_deep
# b_d = f.createVariable('lon', np.float64, ('depth', 'year'))
# b_d[:] = ship_lon_deep
# b_d = f.createVariable('lat', np.float64, ('depth', 'year'))
# b_d[:] = ship_lat_deep
# f.close()
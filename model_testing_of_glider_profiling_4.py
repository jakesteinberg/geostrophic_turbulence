import numpy as np
import pickle
import glob
import datetime
from netCDF4 import Dataset
import gsw
import time as TT
from scipy.integrate import cumtrapz
from mode_decompositions import eta_fit, vertical_modes, PE_Tide_GM, vertical_modes_f
# -- plotting
import matplotlib
import matplotlib.pyplot as plt
from toolkit import plot_pro, nanseg_interp
from zrfun import get_basic_info, get_z


# file_list_0 = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/LiveOcean/simulated_dg_velocities/ve_ew_v*slp*_y10_11_2*.pkl')
file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/LiveOcean/simulated_dg_velocities/ve_ew_v*slp*_y*_*.pkl')
# file_list = np.concatenate([file_list_0, file_list_1])
save_metr = 0  # ratio
save_samp = 0  # sample velocity/eta
save_e = 0  # save energy spectra
save_rms = 0  # save v error plot

direct_anom = []
count = 0  # not all files have instantaneous model output
for i in range(len(file_list)):
    pkl_file = open(file_list[i], 'rb')
    MOD = pickle.load(pkl_file)
    pkl_file.close()

    glider_v = MOD['dg_v'][:]
    model_v = MOD['model_u_at_mwv']
    model_v_avg = MOD['model_u_at_mw_avg']
    z_grid = MOD['dg_z'][:]
    slope_error = MOD['shear_error'][:]
    igw = MOD['igw_var']
    eta_0 = MOD['eta_m_dg_avg'][:]
    eta_model_0 = MOD['eta_model'][:]
    ke_mod_0 = MOD['KE_mod'][:]
    pe_mod_0 = MOD['PE_model'][:]
    ke_dg_0 = MOD['KE_dg'][:]
    pe_dg_0 = MOD['PE_dg_avg'][:]
    pe_dg_ind_0 = MOD['PE_dg'][:]
    w_tag_0 = MOD['dg_w'][:]
    slope_tag_0 = MOD['glide_slope'][:]
    if 'PE_mod_ALL' in MOD.keys():
        ke_mod_tot_0 = MOD['KE_mod_ALL'][:]
        ke_mod_off_tot_0 = MOD['KE_mod_off_ALL'][:]
        pe_mod_tot_0 = MOD['PE_mod_ALL'][:]
        avg_N2_0 = MOD['avg_N2'][:]
        z_grid_n2_0 = MOD['z_grid_n2'][:]

    model_mean_per_mw = np.nan * np.ones(np.shape(glider_v))
    for j in range(len(model_v)):
        this_mod = model_v[j]
        g_rep = np.repeat(np.tile(glider_v[:, j][:, None], np.shape(this_mod)[1])[:, :, None],
                          np.shape(this_mod)[2], axis=2)
        direct_anom = g_rep - this_mod
        mod_space = np.nanmean(this_mod, axis=2)  # average across time
        mod_time = np.nanmean(this_mod, axis=1)  # average across space
        spatial_anom = np.nanmean(direct_anom, axis=2)  # average across time
        time_anom = np.nanmean(direct_anom, axis=1)  # average across space

        model_mean_per_mw[:, j] = np.nanmean(np.nanmean(model_v[j], axis=2), axis=1)

    if i < 1:
        v = glider_v.copy()
        mod_v = model_mean_per_mw.copy()
        mod_v_avg = model_v_avg.copy()
        mod_space_out = mod_space
        mod_time_out = mod_time
        anoms = glider_v - model_mean_per_mw
        anoms_space = spatial_anom
        anoms_time = time_anom
        slope_er = slope_error.copy()
        igw_var = igw.copy()
        eta = eta_0.copy()
        eta_model = eta_model_0.copy()
        ke_mod = ke_mod_0.copy()
        pe_mod = pe_mod_0.copy()
        ke_dg = ke_dg_0.copy()
        pe_dg = pe_dg_0.copy()
        pe_dg_ind = pe_dg_ind_0.copy()
        w_tag = 100 * w_tag_0.copy()
        slope_tag = slope_tag_0.copy()
    else:
        v = np.concatenate((v, glider_v.copy()), axis=1)
        mod_v = np.concatenate((mod_v, model_mean_per_mw), axis=1)
        mod_v_avg = np.concatenate((mod_v_avg, model_v_avg), axis=1)
        mod_space_out = np.concatenate((mod_space_out, mod_space), axis=1)
        mod_time_out = np.concatenate((mod_time_out, mod_time), axis=1)
        anoms = np.concatenate((anoms, glider_v - model_mean_per_mw), axis=1)
        anoms_space = np.concatenate((anoms_space, spatial_anom), axis=1)
        anoms_time = np.concatenate((anoms_time, time_anom), axis=1)
        slope_er = np.concatenate((slope_er, slope_error), axis=1)
        igw_var = np.concatenate((igw_var, igw), axis=1)
        eta = np.concatenate((eta, eta_0.copy()), axis=1)
        eta_model = np.concatenate((eta_model, eta_model_0.copy()), axis=1)
        ke_mod = np.concatenate((ke_mod, ke_mod_0), axis=1)
        pe_mod = np.concatenate((pe_mod, pe_mod_0), axis=1)
        ke_dg = np.concatenate((ke_dg, ke_dg_0), axis=1)
        pe_dg = np.concatenate((pe_dg, pe_dg_0), axis=1)
        pe_dg_ind = np.concatenate((pe_dg_ind, pe_dg_ind_0), axis=1)
        w_tag = np.concatenate((w_tag, 100 * w_tag_0), axis=0)
        slope_tag = np.concatenate((slope_tag, slope_tag_0), axis=0)
    if 'PE_mod_ALL' in MOD.keys():
        if count < 1:
            ke_mod_tot = ke_mod_tot_0.copy()
            ke_mod_off_tot = ke_mod_off_tot_0.copy()
            pe_mod_tot = pe_mod_tot_0.copy()
            count = count + 1
        else:
            ke_mod_tot = np.concatenate((ke_mod_tot, ke_mod_tot_0), axis=1)
            ke_mod_off_tot = np.concatenate((ke_mod_off_tot, ke_mod_off_tot_0), axis=1)
            pe_mod_tot = np.concatenate((pe_mod_tot, pe_mod_tot_0), axis=1)
            count = count + 1

slope_s = np.unique(slope_tag)

# vertical shear error as a function depth and igwsignal
matplotlib.rcParams['figure.figsize'] = (6.5, 8)
f, ax = plt.subplots()
low_er_mean = np.nan * np.ones(len(z_grid))
for i in range(len(z_grid)):
    low = np.where((igw_var[i, :] < 1) & (slope_tag > 2))[0]
    all_in = np.where(slope_tag > 2)[0]

    ax.scatter(slope_er[i, all_in], z_grid[i] * np.ones(len(slope_er[i, all_in])), s=2, color='#87CEEB')
    ax.scatter(slope_er[i, low], z_grid[i] * np.ones(len(slope_er[i, low])), s=4, color='#FA8072')
    # ax.scatter(np.nanmean(slope_er[i, low]), z_grid[i], s=20, color='r')
    low_er_mean[i] = np.nanmean(slope_er[i, low])
# ax.scatter(lo, z_grid, s=15, color='k', label=r'var$_{igw}$/var$_{gstr}$ < 1')
ax.plot(np.nanmedian(slope_er, axis=1), z_grid, color='#000080', linewidth=2.5, label='Error Median')
ax.plot(low_er_mean, z_grid, linewidth=2.5, color='#8B0000', label=r'Error Mean for var$_{gstr.}$/var$_{igw}$ > 1')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=12)
ax.set_xscale('log')
ax.set_xlabel('Percent Error')
ax.set_ylabel('z [m]')
ax.set_title('Percent Error between Model Shear and Glider Shear (1:' + str(np.int(slope_s[1])) + ')')
ax.set_xlim([1, 10**4])
ax.set_ylim([-3000, 0])
plot_pro(ax)
if save_metr > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/lo_mod_shear_error.png', dpi=300)

# RMS at different depths
anomy = v - mod_v_avg  # velocity anomaly
# estimate rms error by w
w_s = np.unique(w_tag)
w_cols = '#191970', 'g', '#FF8C00', '#B22222'
mm = np.nan * np.ones((len(slope_s), len(z_grid), len(w_s)))
avg_anom = np.nan * np.ones((len(slope_s), len(z_grid), len(w_s)))
for ss in range(len(np.unique(slope_tag))):
    for i in range(len(np.unique(w_tag))):
        inn = np.where((w_tag == w_s[i]) & (slope_tag == slope_s[ss]))[0]
        mm[ss, :, i] = np.nanmean(anomy[:, inn]**2, axis=1)  # rms error
        avg_anom[ss, :, i] = np.nanmean(anomy[:, inn], axis=1)
# std about error
min_a = np.nan * np.ones((len(slope_s), len(z_grid), 4))
max_a = np.nan * np.ones((len(slope_s), len(z_grid), 4))
for ss in range(len(np.unique(slope_tag))):
    for i in range(np.shape(anomy)[0]):
        for j in range(len(w_s)):
            inn = np.where((w_tag == w_s[j]) & (slope_tag == slope_s[ss]))[0]
            min_a[ss, i, j] = np.nanmean(anomy[i, inn]) - np.nanstd(anomy[i, inn])
            max_a[ss, i, j] = np.nanmean(anomy[i, inn]) + np.nanstd(anomy[i, inn])

matplotlib.rcParams['figure.figsize'] = (12, 6.5)
f, ax = plt.subplots(1, 4, sharey=True)
# w_cols_2 = '#48D1CC', '#32CD32', '#FFA500', '#CD5C5C'
w_cols_2 = '#40E0D0', '#2E8B57', '#FFA500', '#CD5C5C'
for i in range(len(w_s)):
    ax[i].fill_betweenx(z_grid, min_a[1, :, i], x2=max_a[1, :, i], color=w_cols_2[i], zorder=i, alpha=0.95)
    ax[i].plot(avg_anom[1, :, i], z_grid, color=w_cols[i], linewidth=3, zorder=4, label='dg w = ' + str(np.round(w_s[i]/100, decimals=3)) + ' m s$^{-1}$')
    ax[i].set_xlim([-.2, .2])
    ax[i].set_xlabel(r'm s$^{-1}$')
    ax[i].set_title(r'($u_{g}$ - $\overline{u_{model}}$) (|w|=$\mathbf{' + str(np.round(w_s[i]/100, decimals=2)) + '}$ m s$^{-1}$)', fontsize=10)
ax[0].set_ylabel('z [m]')
ax[0].set_ylim([-3000, 0])
ax[0].text(0.025, -2800, str(np.shape(anomy[:, slope_tag < 3])[1]) + ' profiles')
ax[0].grid()
ax[1].grid()
ax[2].grid()
plot_pro(ax[3])
if save_rms > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/lo_mod_dg_vel_e.png', dpi=200)

matplotlib.rcParams['figure.figsize'] = (6.5, 7)
f, ax2 = plt.subplots()
for i in range(np.shape(mm)[2]):
    ax2.plot(mm[0, :, i], z_grid, linewidth=1.5, color=w_cols[i], linestyle='--',
             label='w=' + str(np.round(w_s[i]/100, decimals=3)) + ' m s$^{-1}$) (gs=1:' + str(np.int(slope_s[0])) + ')')
    ax2.plot(mm[1, :, i], z_grid, linewidth=1.5, color=w_cols[i],
             label='w=' + str(np.round(w_s[i]/100, decimals=3)) + ' m s$^{-1}$) (gs=1:' + str(np.int(slope_s[1])) + ')')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=10, loc='lower right')
ax2.set_xlim([0, .005])
ax2.set_ylim([-3000, 0])
ax2.set_xlabel(r'm$^2$ s$^{-2}$')
ax2.set_title(r'LiveOcean: Glider/Model Velocity rms error ($u_{g}$ - $\overline{u_{model}}$)$^2$')
ax2.set_ylabel('z [m]')
plot_pro(ax2)
if save_rms > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/lo_mod_dg_vel_rms_e.png', dpi=200)

# ---------------------------------------------------------------------------------------------------------------------
# --- PLOT ENERGY SPECTRA
ff = np.pi * np.sin(np.deg2rad(44)) / (12 * 1800)  # Coriolis parameter [s^-1]
# --- Background density
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Model/LiveOcean/background_density.pkl', 'rb')
MODb = pickle.load(pkl_file)
pkl_file.close()
bck_sa = np.flipud(MODb['sa_back'][:][:, 40:120])
bck_ct = np.flipud(MODb['ct_back'][:][:, 40:120])
z_bm = [0, len(z_grid)-14]
p_grid = gsw.p_from_z(z_grid, 44)

# or if no big feature is present, avg current profiles for background
N2_bck_out = gsw.Nsquared(np.nanmean(bck_sa[0:z_bm[-1]+1, :], axis=1), np.nanmean(bck_ct[0:z_bm[-1]+1, :], axis=1),
                          p_grid[0:z_bm[-1]+1], lat=44)[0]
N2_bck_out[N2_bck_out < 0] = 1*10**-7

omega = 0
mmax = 25
mm = 25
G, Gz, c, epsilon = vertical_modes(N2_bck_out, -1.0 * z_grid[0:146], omega, mmax)  # N2

sc_x = 1000 * ff / c[1:mm]
l_lim = 3 * 10 ** -2
sc_x = np.arange(1, mm)
l_lim = 0.7
dk = ff / c[1]

matplotlib.rcParams['figure.figsize'] = (10, 10)
f, ax = plt.subplots(2, 2, sharey=True)
# slope 2
for i in range(len(w_s)):
    inn = np.where((w_tag == w_s[i]) & (slope_tag < 3))[0]
    # PE
    avg_PE = np.nanmean(pe_dg[:, inn], axis=1)
    ax[0,0].plot(sc_x, avg_PE[1:mm] / dk, color=w_cols[i], label=r'|w| = ' + str(np.round(w_s[i]/100, decimals=3)) + ' m s$^{-1}$', linewidth=1.25)
    ax[0,0].scatter(sc_x, avg_PE[1:mm] / dk, color=w_cols[i], s=10)
    # KE
    avg_KE = 2 * np.nanmean(ke_dg[:, inn], axis=1)
    ax[0,1].plot(sc_x, avg_KE[1:mm] / dk, color=w_cols[i], label=r'|w| = ' + str(np.round(w_s[i]/100, decimals=3)) + ' m s$^{-1}$', linewidth=1.25)
    ax[0,1].scatter(sc_x, avg_KE[1:mm] / dk, color=w_cols[i], s=10)  # DG KE
    ax[0,1].plot([l_lim, sc_x[0]], avg_KE[0:2] / dk, color=w_cols[i], linewidth=1.5)  # DG KE_0
    ax[0,1].scatter(l_lim, avg_KE[0] / dk, color=w_cols[i], s=10, facecolors='none')  # DG KE_0

# Model
good = np.where((ke_mod[1, :] < 1*10**0) & (slope_tag < 3))[0]
avg_PE_model = np.nanmean(pe_mod[:, good], axis=1)
avg_KE_model = 2 * np.nanmean(ke_mod[:, good], axis=1)
ax[0,0].plot(sc_x, avg_PE_model[1:mm] / dk, color='k', label='PE$_{Model}$', linewidth=2)
ax[0,0].scatter(sc_x, avg_PE_model[1:mm] / dk, color='k', s=10)
ax[0,1].plot(sc_x, avg_KE_model[1:mm] / dk, color='k', label='KE$_{Model}$', linewidth=2)
ax[0,1].scatter(sc_x, avg_KE_model[1:mm] / dk, color='k', s=10)
ax[0,1].plot([l_lim, sc_x[0]], avg_KE_model[0:2] / dk, color='k', linewidth=2)
ax[0,1].scatter(l_lim, avg_KE_model[0] / dk, color='k', s=10, facecolors='none')

# avg_KE_model_ind_all = 2 * np.nanmean(ke_mod_tot, axis=1)
# avg_PE_model_ind_all = np.nanmean(pe_mod_tot, axis=1)
# ax[0,1].plot(sc_x, avg_KE_model_ind_all[1:mm] / dk, color='k', label='KE$_{Model_{inst.}}$', linewidth=1, linestyle='--')
# ax[0,1].plot([l_lim, sc_x[0]], avg_KE_model_ind_all[0:2] / dk, color='k', linewidth=1, linestyle='--')
# ax[0,0].plot(sc_x, avg_PE_model_ind_all[1:mm] / dk, color='k', label='PE$_{Model_{inst.}}$', linewidth=1, linestyle='--')

ax[0,0].plot([10**0, 10**1], [10**-1, 10**-4], linewidth=0.75, color='k')
ax[0,0].plot([10**0, 10**1], [10**-1, 10**-3], linewidth=0.75, color='k')
ax[0,0].text(8*10**0, 2*10**-3, '-2', fontsize=8)
ax[0,0].text(2.5*10**0, 2*10**-3, '-3', fontsize=8)
ax[0,1].plot([10**0, 10**1], [10**-1, 10**-4], linewidth=0.75, color='k')
ax[0,1].plot([10**0, 10**1], [10**-1, 10**-3], linewidth=0.75, color='k')
ax[0,1].text(8*10**0, 2*10**-3, '-2', fontsize=8)
ax[0,1].text(2.5*10**0, 2*10**-3, '-3', fontsize=8)

limm = 5
ax[0,0].set_xlim([l_lim, 0.5 * 10 ** 2])
ax[0,1].set_xlim([l_lim, 0.5 * 10 ** 2])
ax[0,1].set_ylim([10 ** (-4), 3 * 10 ** 2])
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,1].set_xscale('log')
ax[0,0].set_ylabel('Variance per Vertical Wavenumber', fontsize=10)  # ' (and Hor. Wavenumber)')
# ax[0,0].set_xlabel('Mode Number', fontsize=12)
# ax[0,1].set_xlabel('Mode Number', fontsize=12)
ax[0,0].set_title('LiveOcean: Potential Energy (1:' + str(np.int(slope_s[0])) + ')', fontsize=12)
ax[0,1].set_title('LiveOcean: Kinetic Energy (1:' + str(np.int(slope_s[0])) + ')', fontsize=12)
handles, labels = ax[0,1].get_legend_handles_labels()
ax[0,1].legend(handles, labels, fontsize=10)
handles, labels = ax[0,0].get_legend_handles_labels()
ax[0,0].legend(handles, labels, fontsize=10)
ax[0,0].grid()
ax[0,1].grid()

# slope 3
for i in range(len(w_s)):
    inn = np.where((w_tag == w_s[i]) & (slope_tag > 2))[0]
    # PE
    avg_PE = np.nanmean(pe_dg[:, inn], axis=1)
    ax[1,0].plot(sc_x, avg_PE[1:mm] / dk, color=w_cols[i], label=r'|w| = ' + str(np.round(w_s[i]/100, decimals=3)) + ' m s$^{-1}$', linewidth=1.25)
    ax[1,0].scatter(sc_x, avg_PE[1:mm] / dk, color=w_cols[i], s=10)
    # KE
    avg_KE = 2 * np.nanmean(ke_dg[:, inn], axis=1)
    ax[1,1].plot(sc_x, avg_KE[1:mm] / dk, color=w_cols[i], label=r'|w| = ' + str(np.round(w_s[i]/100, decimals=3)) + ' m s$^{-1}$', linewidth=1.25)
    ax[1,1].scatter(sc_x, avg_KE[1:mm] / dk, color=w_cols[i], s=10)  # DG KE
    ax[1,1].plot([l_lim, sc_x[0]], avg_KE[0:2] / dk, color=w_cols[i], linewidth=1.5)  # DG KE_0
    ax[1,1].scatter(l_lim, avg_KE[0] / dk, color=w_cols[i], s=10, facecolors='none')  # DG KE_0

# Model
good = np.where((ke_mod[1, :] < 1*10**0) & (slope_tag > 2))[0]
avg_PE_model = np.nanmean(pe_mod[:, good], axis=1)
avg_KE_model = 2 * np.nanmean(ke_mod[:, good], axis=1)
ax[1,0].plot(sc_x, avg_PE_model[1:mm] / dk, color='k', label='PE$_{Model}$', linewidth=2)
ax[1,0].scatter(sc_x, avg_PE_model[1:mm] / dk, color='k', s=10)
ax[1,1].plot(sc_x, avg_KE_model[1:mm] / dk, color='k', label='KE$_{Model}$', linewidth=2)
ax[1,1].scatter(sc_x, avg_KE_model[1:mm] / dk, color='k', s=10)
ax[1,1].plot([l_lim, sc_x[0]], avg_KE_model[0:2] / dk, color='k', linewidth=2)
ax[1,1].scatter(l_lim, avg_KE_model[0] / dk, color='k', s=10, facecolors='none')

avg_KE_model_ind_all = 2 * np.nanmean(ke_mod_tot, axis=1)
avg_KE_model_off_ind_all = 2 * np.nanmean(ke_mod_off_tot, axis=1)
avg_PE_model_ind_all = np.nanmean(pe_mod_tot, axis=1)
ax[1,1].plot(sc_x, avg_KE_model_ind_all[1:mm] / dk, color='k', label='KE$_{Model_{inst.}}$', linewidth=1, linestyle='--')
ax[1,1].plot([l_lim, sc_x[0]], avg_KE_model_ind_all[0:2] / dk, color='k', linewidth=1, linestyle='--')
ax[1,1].plot(sc_x, avg_KE_model_off_ind_all[1:mm] / dk, color='k', label='KE$_{Model_{inst.}}$', linewidth=1, linestyle='-.')
ax[1,1].plot([l_lim, sc_x[0]], avg_KE_model_off_ind_all[0:2] / dk, color='k', linewidth=1, linestyle='-.')
ax[1,0].plot(sc_x, avg_PE_model_ind_all[1:mm] / dk, color='k', label='PE$_{Model_{inst.}}$', linewidth=1, linestyle='--')

ax[1,0].plot([10**0, 10**1], [10**-1, 10**-4], linewidth=0.75, color='k')
ax[1,0].plot([10**0, 10**1], [10**-1, 10**-3], linewidth=0.75, color='k')
ax[1,0].text(8*10**0, 2*10**-3, '-2', fontsize=8)
ax[1,0].text(2.5*10**0, 2*10**-3, '-3', fontsize=8)
ax[1,1].plot([10**0, 10**1], [10**-1, 10**-4], linewidth=0.75, color='k')
ax[1,1].plot([10**0, 10**1], [10**-1, 10**-3], linewidth=0.75, color='k')
ax[1,1].text(8*10**0, 2*10**-3, '-2', fontsize=8)
ax[1,1].text(2.5*10**0, 2*10**-3, '-3', fontsize=8)

limm = 5
ax[1,0].set_xlim([l_lim, 0.5 * 10 ** 2])
ax[1,1].set_xlim([l_lim, 0.5 * 10 ** 2])
ax[1,1].set_ylim([10 ** (-4), 3 * 10 ** 2])
ax[1,0].set_yscale('log')
ax[1,0].set_xscale('log')
ax[1,1].set_xscale('log')
ax[1,0].set_ylabel('Variance per Vertical Wavenumber', fontsize=10)  # ' (and Hor. Wavenumber)')
ax[1,0].set_xlabel('Mode Number', fontsize=10)
ax[1,1].set_xlabel('Mode Number', fontsize=10)
ax[1,0].set_title('LiveOcean: Potential Energy (1:' + str(np.int(slope_s[1])) + ')', fontsize=12)
ax[1,1].set_title('LiveOcean: Kinetic Energy (1:' + str(np.int(slope_s[1])) + ')', fontsize=12)
handles, labels = ax[1,1].get_legend_handles_labels()
ax[1,1].legend(handles, labels, fontsize=7)
handles, labels = ax[1,0].get_legend_handles_labels()
ax[1,0].legend(handles, labels, fontsize=10)
ax[1,0].grid()

plt.gcf().text(0.06, 0.9, 'a)', fontsize=12)
plt.gcf().text(0.5, 0.9, 'b)', fontsize=12)
plt.gcf().text(0.06, 0.48, 'c)', fontsize=12)
plt.gcf().text(0.5, 0.48, 'd)', fontsize=12)

plot_pro(ax[1,1])
if save_e > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/lo_mod_energy_eddy.png', dpi=200)

# ----------------
# horizontal scale
# ----------------
good = np.where((ke_mod[1, :] < 1*10**0) & (slope_tag > 2))[0]  # & (np.nanmax(np.abs(mod_v[10:, :]), 0) > 0.15)r
avg_PE_model = np.nanmean(pe_mod[:, good], axis=1)
avg_KE_model = 2 * np.nanmean(ke_mod[:, good], axis=1)

matplotlib.rcParams['figure.figsize'] = (6, 6)
f, ax = plt.subplots()
sc_x = 1000 * ff / c[1:mm]
k_h = 1e3 * (ff / c[1:mm]) * np.sqrt(avg_KE_model[1:mm] / avg_PE_model[1:mm])
k_h_tot = 1e3 * (ff / c[1:mm]) * np.sqrt(avg_KE_model_ind_all[1:mm] / avg_PE_model_ind_all[1:mm])
model_uv_ke_ind_all = np.nanmean(ke_mod_tot, axis=1) + np.nanmean(ke_mod_off_tot, axis=1)  # 1/2 aleady included
k_h_uv_tot = 1e3 * (ff / c[1:mm]) * np.sqrt(model_uv_ke_ind_all[1:mm] / avg_PE_model_ind_all[1:mm])

for i in range(len(w_s)):
    inn = np.where((w_tag == w_s[i]) & (slope_tag < 3))[0]
    avg_KE = 2 * np.nanmean(ke_dg[:, inn], axis=1)
    avg_PE = np.nanmean(pe_dg[:, inn], axis=1)
    k_h_dg = 1e3 * (ff / c[1:mm]) * np.sqrt(avg_KE[1:mm] / avg_PE[1:mm])
    ax.plot(sc_x, k_h_dg, color=w_cols[i], label=r'$k_h$', linewidth=1.5)

ax.plot(sc_x, k_h, color='k', label=r'$k_h$', linewidth=1.5)
ax.plot(sc_x, k_h_tot, color='r', label=r'$k_h$', linewidth=1, linestyle='-.')
ax.plot(sc_x, k_h_uv_tot, color='b', label=r'$k_h$', linewidth=1, linestyle='--')
ax.plot([10 ** -2, 10 ** 1], [10 ** (-2), 1 * 10 ** 1], linestyle='--', color='k')
ax.set_yscale('log')
ax.set_xscale('log')
ax.axis([10 ** -2, 10 ** 1, 10 ** (-2), 1 * 10 ** 1])
ax.set_aspect('equal')
plot_pro(ax)

# OLD ENERGY
# avg_PE = np.nanmean(pe_dg, axis=1)
# avg_PE_ind = np.nanmean(pe_dg_ind, axis=1)
# avg_KE = np.nanmean(ke_dg, axis=1)
# avg_PE_model = np.nanmean(pe_mod, axis=1)
# avg_KE_model = np.nanmean(ke_mod, axis=1)
#
# matplotlib.rcParams['figure.figsize'] = (10, 6)
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# # DG
# ax1.plot(sc_x, avg_PE[1:mm] / dk, 'r', label='PE$_{DG}$', linewidth=2)
# ax1.scatter(sc_x, avg_PE[1:mm] / dk, color='r', s=20)
# ax1.plot(sc_x, avg_PE_ind[1:mm] / dk, 'c', label='PE$_{DG_{ind}}$', linewidth=2)
# ax1.scatter(sc_x, avg_PE_ind[1:mm] / dk, color='c', s=20)
# ax2.plot(sc_x, avg_KE[1:mm] / dk, 'r', label='KE$_{DG}$', linewidth=3)
# ax2.scatter(sc_x, avg_KE[1:mm] / dk, color='r', s=20)  # DG KE
# ax2.plot([l_lim, sc_x[0]], avg_KE[0:2] / dk, 'r', linewidth=3)  # DG KE_0
# ax2.scatter(l_lim, avg_KE[0] / dk, color='r', s=25, facecolors='none')  # DG KE_0
# # Model
# ax1.plot(sc_x, avg_PE_model[1:mm] / dk, color='k', label='PE$_{Model}$', linewidth=2)
# ax1.scatter(sc_x, avg_PE_model[1:mm] / dk, color='k', s=20)
# ax2.plot(sc_x, avg_KE_model[1:mm] / dk, color='k', label='KE$_{Model}$', linewidth=2)
# ax2.scatter(sc_x, avg_KE_model[1:mm] / dk, color='k', s=20)
# ax2.plot([l_lim, sc_x[0]], avg_KE_model[0:2] / dk, color='k', linewidth=2)
# ax2.scatter(l_lim, avg_KE_model[0] / dk, color='k', s=25, facecolors='none')
#
# modeno = '1', '2', '3', '4', '5', '6', '7', '8'
# for j in range(len(modeno)):
#     ax2.text(sc_x[j], (avg_KE[j + 1] + (avg_KE[j + 1] / 2)) / dk, modeno[j], color='k', fontsize=10)
#
# limm = 5
# ax1.set_xlim([l_lim, 0.5 * 10 ** 2])
# ax2.set_xlim([l_lim, 0.5 * 10 ** 2])
# ax2.set_ylim([10 ** (-4), 1 * 10 ** 2])
# ax1.set_yscale('log')
# ax1.set_xscale('log')
# ax2.set_xscale('log')
#
# # ax1.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
# ax1.set_ylabel('Spectral Density', fontsize=12)  # ' (and Hor. Wavenumber)')
# # ax2.set_xlabel(r'Scaled Vertical Wavenumber = (L$_{d_{n}}$)$^{-1}$ = $\frac{f}{c_n}$ [$km^{-1}$]', fontsize=12)
# ax1.set_xlabel('Mode Number', fontsize=12)
# ax2.set_xlabel('Mode Number', fontsize=12)
# ax1.set_title('PE Spectrum (' + str(np.int(MOD['glide_slope'][0])) + ':1)', fontsize=12)
# ax2.set_title('KE Spectrum (' + str(np.int(MOD['glide_slope'][0])) + ':1)', fontsize=12)
# handles, labels = ax2.get_legend_handles_labels()
# ax2.legend(handles, labels, fontsize=12)
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(handles, labels, fontsize=12)
# ax1.grid()
# plot_pro(ax2)
# if save_e > 0:
#     f.savefig('/Users/jake/Documents/glider_flight_sim_paper/LO_model_dg_vel_energy_eddy.png', dpi=200)
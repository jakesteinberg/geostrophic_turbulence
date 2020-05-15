import numpy as np
import pickle
import glob
import gsw
import time as TT
from scipy.integrate import cumtrapz
from mode_decompositions import eta_fit, vertical_modes, PE_Tide_GM, vertical_modes_f
# -- plotting
import matplotlib
import matplotlib.pyplot as plt
from toolkit import plot_pro, nanseg_interp

file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/LiveOcean/simulated_dg_velocities/ve_ew_v*_slp*_y*_*.pkl')
save_metr = 0  # ratio
save_e = 0  # save energy spectra
save_rms = 0  # save v error plot
save_eof = 0

direct_anom = []
count = 0  # not all files have instantaneous model output
for i in range(len(file_list)):
    pkl_file = open(file_list[i], 'rb')
    MOD = pickle.load(pkl_file)
    pkl_file.close()

    glider_v = MOD['dg_v'][:]
    model_v = MOD['model_u_at_mwv']
    model_v_avg = MOD['model_u_at_mw_avg']
    model_v_off_avg = MOD['model_u_off_at_mw_avg']
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
    for j in range(len(model_v)):  # loop over each set of instantaneous profiles associated with each m/w profile
        this_mod = model_v[j]  # dimensions [depth, space, time]
        g_rep = np.repeat(np.tile(glider_v[:, j][:, None], np.shape(this_mod)[1])[:, :, None],
                          np.shape(this_mod)[2], axis=2)
        direct_anom = g_rep - this_mod
        mod_space = np.nanmean(this_mod, axis=2)  # average across time
        si = np.int((np.shape(this_mod)[1])/2.0)  # midpoint space index
        mod_time = np.squeeze(this_mod[:, si, :])  # midpoint profile, all times, old(average across space)
        spatial_anom = np.nanmean(direct_anom, axis=2)  # average across time
        time_anom = np.nanmean(direct_anom, axis=1)  # average across space

        model_mean_per_mw[:, j] = np.nanmean(np.nanmean(model_v[j], axis=2), axis=1)

    if i < 1:
        v = glider_v.copy()
        mod_v = model_mean_per_mw.copy()
        mod_v_avg = model_v_avg.copy()
        mod_v_off_avg = model_v_off_avg.copy()
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
        mod_v_off_avg = np.concatenate((mod_v_off_avg, model_v_off_avg), axis=1)
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
ax.plot(low_er_mean, z_grid, linewidth=2.5, color='#8B0000', label=r'Error Mean for g(z)/f(z) > 1')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=12)
ax.set_xscale('log')
ax.set_xlabel('Percent Error')
ax.set_ylabel('z [m]')
ax.set_title('Percent Error between Model Shear and Glider Shear (s = 1/' + str(np.int(slope_s[1])) + ')')
ax.set_xlim([1, 10**4])
ax.set_ylim([-3000, 0])
plot_pro(ax)
if save_metr > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/reviewer_comments_minor_revisions/revised_figures/lo_mod_shear_error.png', dpi=300)

# load in hycom and liveocean sampled as a mooring, to compute eof and compare (look at decay of eof with depth)
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Model/model_mooring_samplings.pkl', 'rb')
MM = pickle.load(pkl_file)
pkl_file.close()
hy_u_moor = MM['hy_mod_u'][:]  # [time, dep_levs, xy_pos]
hy_u_moor_filt = MM['hy_mod_u_xy1_lo_pass'][:]
hy_z = MM['hy_depth'][:]
hy_z_dep = MM['hy_z_samp_deps'][:]
lo_u_moor = MM['lo_mod_u'][:]  # [dep_levs, xy_pos, time]
lo_u_moor_filt = MM['lo_mod_u_xy1_lo_pass'][:]
lo_z = MM['lo_depth'][:]
lo_z_dep = MM['lo_z_samp_deps'][:]
# load in hycom and liveocean (2) sampled as a mooring, to compute eof and compare (look at decay of eof with depth)
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Model/model_mooring_samplings_3_1.pkl', 'rb')
MM2 = pickle.load(pkl_file)
pkl_file.close()
hy_u_moor2 = MM2['hy_mod_u'][:]  # [time, dep_levs, xy_pos]
hy_u_moor_filt2 = MM2['hy_mod_u_xy1_lo_pass'][:]
hy_z2 = MM2['hy_depth'][:]
hy_z_dep2 = MM2['hy_z_samp_deps'][:]
lo_u_moor2 = MM2['lo_mod_u'][:]  # [dep_levs, xy_pos, time]
lo_u_moor_filt2 = MM2['lo_mod_u_xy1_lo_pass'][:]
lo_z2 = MM2['lo_depth'][:]
lo_z_dep2 = MM2['lo_z_samp_deps'][:]
# load in hycom and liveocean (3) sampled as a mooring, to compute eof and compare (look at decay of eof with depth)
pkl_file = open('/Users/jake/Documents/baroclinic_modes/Model/model_mooring_samplings_3_2.pkl', 'rb')
MM3 = pickle.load(pkl_file)
pkl_file.close()
hy_u_moor3 = MM3['hy_mod_u'][:]  # [time, dep_levs, xy_pos]
hy_u_moor_filt3 = MM3['hy_mod_u_xy1_lo_pass'][:]
hy_z3 = MM3['hy_depth'][:]
hy_z_dep3 = MM3['hy_z_samp_deps'][:]
lo_u_moor3 = MM3['lo_mod_u'][:]  # [dep_levs, xy_pos, time]
lo_u_moor_filt3 = MM3['lo_mod_u_xy1_lo_pass'][:]
lo_z3 = MM3['lo_depth'][:]
lo_z_dep3 = MM3['lo_z_samp_deps'][:]
# ----------------------------------------------------
# EOF need to make sure that length of z is shorter than number of profiles
# ----------------------------------------------------
# --- EOFs of glider and model velocity profiles
# -- glider
check1 = 2     # upper index to include in eof computation
check2 = -25     # lower index to include in eof computation
not_shallow = np.isfinite(v[-25, :]) & (slope_tag < 3)
grid_check = z_grid[check1:check2]
Uzq = v[check1:check2, not_shallow].copy()
nq = np.size(Uzq[0, :])
avg_Uzq = np.nanmean(np.transpose(Uzq), axis=0)
Uzqa = Uzq - np.transpose(np.tile(avg_Uzq, [nq, 1]))
cov_Uzqa = (1.0 / nq) * np.matrix(Uzqa) * np.matrix(np.transpose(Uzqa))
D_Uzqa, V_Uzqa = np.linalg.eig(cov_Uzqa)
D_sort = np.flipud(np.argsort(D_Uzqa))
D_Uz = D_Uzqa[D_sort]
V_Uz = V_Uzqa[:, D_sort]
t1 = np.real(D_Uz[0:10])
PEV = t1 / np.sum(t1)
eof1 = np.array(np.real(V_Uz[:, 0]))
eof2 = np.array(np.real(V_Uz[:, 1]))
# ----------------------------------------------------
# -- model cross-transect
check1 = 2      # upper index to include in eof computation
check2 = -25     # lower index to include in eof computation
Uzq = mod_v_avg[check1:check2, not_shallow].copy()
nq = np.size(Uzq[0, :])
avg_Uzq = np.nanmean(np.transpose(Uzq), axis=0)
Uzqa = Uzq - np.transpose(np.tile(avg_Uzq, [nq, 1]))
cov_Uzqa = (1.0 / nq) * np.matrix(Uzqa) * np.matrix(np.transpose(Uzqa))
Dmod_Uzqa, Vmod_Uzqa = np.linalg.eig(cov_Uzqa)
D_sort = np.flipud(np.argsort(Dmod_Uzqa))
Dmod_Uz = Dmod_Uzqa[D_sort]
Vmod_Uz = Vmod_Uzqa[:, D_sort]
t1mod = np.real(Dmod_Uz[0:10])
PEV_model = t1mod / np.sum(t1mod)
mod_eof1 = np.array(np.real(Vmod_Uz[:, 0]))
mod_eof2 = np.array(np.real(Vmod_Uz[:, 1]))
# ----------------------------------------------------
# -- model along-transect
Uzq = mod_v_off_avg[check1:check2, not_shallow].copy()
nq = np.size(Uzq[0, :])
avg_Uzq = np.nanmean(np.transpose(Uzq), axis=0)
Uzqa = Uzq - np.transpose(np.tile(avg_Uzq, [nq, 1]))
cov_Uzqa = (1.0 / nq) * np.matrix(Uzqa) * np.matrix(np.transpose(Uzqa))
Dmodoff_Uzqa, Vmodoff_Uzqa = np.linalg.eig(cov_Uzqa)
D_sort = np.flipud(np.argsort(Dmodoff_Uzqa))
Dmodoff_Uz = Dmodoff_Uzqa[D_sort]
Vmodoff_Uz = Vmodoff_Uzqa[:, D_sort]
t1mod = np.real(Dmodoff_Uz[0:10])
PEV_model_off = t1mod / np.sum(t1mod)
mod_off_eof1 = np.array(np.real(Vmodoff_Uz[:, 0]))
mod_off_eof2 = np.array(np.real(Vmodoff_Uz[:, 1]))
# ----------------------------------------------------
# -- model cross-transect (no temporal averaging, select profile position at midpoints of m/w patterns)
check1 = 2      # upper index to include in eof computation
check2 = -25     # lower index to include in eof computation
Uzq = mod_time_out[check1:check2, :].copy()
nq = np.size(Uzq[0, :])
avg_Uzq = np.nanmean(np.transpose(Uzq), axis=0)
Uzqa = Uzq - np.transpose(np.tile(avg_Uzq, [nq, 1]))
cov_Uzqa = (1.0 / nq) * np.matrix(Uzqa) * np.matrix(np.transpose(Uzqa))
Dmods_Uzqa, Vmods_Uzqa = np.linalg.eig(cov_Uzqa)
D_sort = np.flipud(np.argsort(Dmods_Uzqa))
Dmods_Uz = Dmods_Uzqa[D_sort]
Vmods_Uz = Vmods_Uzqa[:, D_sort]
t1mod = np.real(Dmods_Uz[0:10])
PEV_models = t1mod / np.sum(t1mod)
mods_eof1 = np.array(np.real(Vmods_Uz[:, 0]))
mods_eof2 = np.array(np.real(Vmods_Uz[:, 1]))
# ----------------------------------------------------
# -- lo sampled as a mooring (5 points)
Uzq_lom = np.squeeze(lo_u_moor[:, 0, :].copy())
for i in range(1, np.shape(lo_u_moor)[1]):
    Uzq_lom = np.concatenate((Uzq_lom, np.squeeze(lo_u_moor[:, i, :])), axis=1)
nq = np.size(Uzq_lom[0, :])
avg_Uzq = np.nanmean(np.transpose(Uzq_lom), axis=0)
Uzqa = Uzq_lom - np.transpose(np.tile(avg_Uzq, [nq, 1]))
cov_Uzqa = (1.0 / nq) * np.matrix(Uzqa) * np.matrix(np.transpose(Uzqa))
Dmods_Uzqa, Vmods_Uzqa = np.linalg.eig(cov_Uzqa)
D_sort = np.flipud(np.argsort(Dmods_Uzqa))
Dmod_m_Uz = Dmods_Uzqa[D_sort]
Vmod_Uz = Vmods_Uzqa[:, D_sort]
t1mod = np.real(Dmod_m_Uz[0:10])
lo_moor_PEV_model = t1mod / np.sum(t1mod)
lo_moor_eof1 = np.array(np.real(Vmod_Uz[:, 0]))
lo_moor_eof2 = np.array(np.real(Vmod_Uz[:, 1]))
# ----------------------------------------------------
# -- lo sampled as a mooring (9 points)
Uzq_lom2 = np.squeeze(lo_u_moor2[:, 0, :].copy())
for i in range(1, np.shape(lo_u_moor2)[1]):
    Uzq_lom2 = np.concatenate((Uzq_lom2, np.squeeze(lo_u_moor2[:, i, :])), axis=1)
nq2 = np.size(Uzq_lom2[0, :])
avg_Uzq2 = np.nanmean(np.transpose(Uzq_lom2), axis=0)
Uzqa2 = Uzq_lom2 - np.transpose(np.tile(avg_Uzq2, [nq2, 1]))
cov_Uzqa2 = (1.0 / nq2) * np.matrix(Uzqa2) * np.matrix(np.transpose(Uzqa2))
Dmods_Uzqa2, Vmods_Uzqa2 = np.linalg.eig(cov_Uzqa2)
D_sort2 = np.flipud(np.argsort(Dmods_Uzqa2))
Dmod_m_Uz2 = Dmods_Uzqa2[D_sort2]
Vmod_Uz2 = Vmods_Uzqa2[:, D_sort2]
t1mod = np.real(Dmod_m_Uz2[0:10])
lo_moor2_PEV_model = t1mod / np.sum(t1mod)
lo_moor2_eof1 = np.array(np.real(Vmod_Uz2[:, 0]))
lo_moor2_eof2 = np.array(np.real(Vmod_Uz2[:, 1]))
# ----------------------------------------------------
# -- lo sampled as a mooring (16 points)
Uzq_lom3 = np.squeeze(lo_u_moor3[:, 0, :].copy())
for i in range(1, np.shape(lo_u_moor3)[1]):
    Uzq_lom3 = np.concatenate((Uzq_lom3, np.squeeze(lo_u_moor3[:, i, :])), axis=1)
nq3 = np.size(Uzq_lom3[0, :])
avg_Uzq3 = np.nanmean(np.transpose(Uzq_lom3), axis=0)
Uzqa3 = Uzq_lom3 - np.transpose(np.tile(avg_Uzq3, [nq2, 1]))
cov_Uzqa3 = (1.0 / nq3) * np.matrix(Uzqa3) * np.matrix(np.transpose(Uzqa3))
Dmods_Uzqa3, Vmods_Uzqa3 = np.linalg.eig(cov_Uzqa3)
D_sort3 = np.flipud(np.argsort(Dmods_Uzqa3))
Dmod_m_Uz3 = Dmods_Uzqa3[D_sort3]
Vmod_Uz3 = Vmods_Uzqa3[:, D_sort3]
t1mod = np.real(Dmod_m_Uz3[0:10])
lo_moor3_PEV_model = t1mod / np.sum(t1mod)
lo_moor3_eof1 = np.array(np.real(Vmod_Uz3[:, 0]))
lo_moor3_eof2 = np.array(np.real(Vmod_Uz3[:, 1]))
# ----------------------------------------------------
# -- lo sampled as a mooring (with low pass filtering)
# Uzq = np.transpose(lo_u_moor_filt[12:-12, :, 0].copy())
Uzq_lomf = np.transpose(np.squeeze(lo_u_moor_filt[12:-12, :, 0].copy()))
for i in range(1, np.shape(lo_u_moor_filt)[2]):
    Uzq_lomf = np.concatenate((Uzq_lom, np.transpose(np.squeeze(lo_u_moor_filt[12:-12, :, i]))), axis=1)
nqf = np.size(Uzq_lomf[0, :])
avg_Uzqf = np.nanmean(np.transpose(Uzq_lomf), axis=0)
Uzqaf = Uzq_lomf - np.transpose(np.tile(avg_Uzqf, [nqf, 1]))
cov_Uzqa_f = (1.0 / nqf) * np.matrix(Uzqaf) * np.matrix(np.transpose(Uzqaf))
Dmodf_Uzqa, Vmodf_Uzqa = np.linalg.eig(cov_Uzqa_f)
D_sort = np.flipud(np.argsort(Dmodf_Uzqa))
Dmodf_Uz = Dmodf_Uzqa[D_sort]
Vmodf_Uz = Vmodf_Uzqa[:, D_sort]
t1modf = np.real(Dmodf_Uz[0:10])
lo_moor_f_PEV_model = t1modf / np.sum(t1modf)
lo_moor_f_eof1 = np.array(np.real(Vmodf_Uz[:, 0]))
lo_moor_f_eof2 = np.array(np.real(Vmodf_Uz[:, 1]))

# ----------------------------------------------------
# ----------------------------------------------------
# matplotlib.rcParams['figure.figsize'] = (5.5, 7)
# f, ax = plt.subplots()
# for i in range(np.shape(v)[1]):
#     if slope_tag[i] < 3:
#         ax.plot(v[:, i], z_grid, color='#48D1CC', linewidth=0.5)
#     else:
#         ax.plot(v[:, i], z_grid, color='#B22222', linewidth=0.5)
# ax.set_title('Simulated DG Cross-Track Velocities (1:2 = blue, 1:3 = red)')
# ax.set_xlabel(r'm s$^{-1}$')
# ax.set_ylabel('z [m]')
# ax.set_xlim([-0.4, 0.4])
# plot_pro(ax)
# if save_eof > 0:
#     f.savefig('/Users/jake/Documents/glider_flight_sim_paper/lo_mod_all_v.png', dpi=300)


matplotlib.rcParams['figure.figsize'] = (10, 7)
f, (ax, ax2) = plt.subplots(1, 2)
ax.plot(-1.0*eof1*D_Uz[0], grid_check, color='b', label=r'dg$_1$ PEV=' + str(np.round(PEV[0]*100,1)))
ax.plot(mod_eof1*Dmod_Uz[0], grid_check, color='g', label=r'mod$_1$ PEV=' + str(np.round(PEV_model[0]*100,1)))
ax.plot(-1.0*mod_off_eof1*Dmodoff_Uz[0], grid_check, color='m', label=r'mod$_1$ off PEV=' + str(np.round(PEV_model_off[0]*100,1)))
ax.plot(-1.0*mods_eof1*Dmods_Uz[0], grid_check, color='c', label=r'mod$_1$ ind PEV=' + str(np.round(PEV_models[0]*100,1)))
# ax.plot(lo_moor_eof1, lo_z_dep, color='r', label='mooring PEV = ' + str(np.round(lo_moor_PEV_model[0]*100,1)))
ax.plot(lo_moor_eof1*Dmod_m_Uz[0], lo_z_dep, color='r', linestyle='-',label=r'mod$_1$ moor' + str(len(lo_z_dep)) + ' PEV = ' + str(np.round(lo_moor_PEV_model[0]*100,1)))
ax.plot(-1.0*lo_moor2_eof1*Dmod_m_Uz2[0], lo_z_dep2, color='r', linestyle='-.',label=r'mod$_1$ moor' + str(len(lo_z_dep2)) + ' PEV = ' + str(np.round(lo_moor2_PEV_model[0]*100,1)))
ax.plot(1.0*lo_moor3_eof1*Dmod_m_Uz3[0], lo_z_dep3, color='r', linestyle='--',label=r'mod$_1$ moor' + str(len(lo_z_dep3)) + ' PEV = ' + str(np.round(lo_moor3_PEV_model[0]*100,1)))

ax2.plot(eof2*D_Uz[0], grid_check, color='b', linestyle='--', label=r'dg$_2$ PEV=' + str(np.round(PEV[1]*100,1)))
ax2.plot(mod_eof2*Dmod_Uz[0], grid_check, color='g', linestyle='--', label=r'mod$_2$ PEV=' + str(np.round(PEV_model[1]*100,1)))
ax2.plot(-1.0*mod_off_eof2*Dmodoff_Uz[0], grid_check, color='m', linestyle='--', label=r'mod$_2$ off PEV=' + str(np.round(PEV_model_off[1]*100,1)))
ax2.plot(mods_eof2*Dmods_Uz[0], grid_check, color='c', linestyle='--', label=r'mod$_2$ ind PEV=' + str(np.round(PEV_models[1]*100,1)))
ax2.plot(lo_moor_eof2*Dmod_m_Uz[1], lo_z_dep, color='r', linestyle='-', label=r'mod$_2$ moor PEV = ' + str(np.round(lo_moor_PEV_model[1]*100,1)))
ax2.plot(lo_moor2_eof2*Dmod_m_Uz2[1], lo_z_dep2, color='r', linestyle='-.', label=r'mod$_2$ moor PEV = ' + str(np.round(lo_moor2_PEV_model[1]*100,1)))
ax2.plot(lo_moor3_eof2*Dmod_m_Uz3[1], lo_z_dep3, color='r', linestyle='--', label=r'mod$_2$ moor PEV = ' + str(np.round(lo_moor3_PEV_model[1]*100,1)))


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=9)
ax.set_title('Glider-Model Velocity EOF1 slope=1:2')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=9)
ax2.set_title('Glider-Model Velocity EOF2 slope=1:2')
ax.set_xlim([-.1, 0.025])
ax2.set_xlim([-.1, 0.1])
ax.grid()
plot_pro(ax2)
if save_eof > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/lo_mod_eofs_s3.png', dpi=300)

# --------------------------------------------------------------------------------------------
# RMS at different depths

# DAC error = 0.01
anomy = v - mod_v_avg  # velocity anomaly
# estimate rms error by w
w_s = np.unique(w_tag)
w_cols = '#191970', 'g', '#FF8C00', '#B22222'
mm = np.nan * np.ones((len(slope_s), len(z_grid), len(w_s)))
avg_anom = np.nan * np.ones((len(slope_s), len(z_grid), len(w_s)))
min_a = np.nan * np.ones((len(slope_s), len(z_grid), 4))
max_a = np.nan * np.ones((len(slope_s), len(z_grid), 4))
for ss in range(len(np.unique(slope_tag))):  # loop over slopes
    for i in range(len(np.unique(w_tag))):  # loop over w values
        inn = np.where((w_tag == w_s[i]) & (slope_tag == slope_s[ss]))[0]
        this_anom = anomy[:, inn]
        this_sq_anom = anomy[:, inn]**2
        good = np.where(~np.isnan(this_anom[100, :]))[0]
        mm[ss, :, i] = np.nanmean(this_sq_anom[:, good], axis=1)  # rms error
        avg_anom[ss, :, i] = np.nanmean(this_anom[:, good], axis=1)
        for j in range(np.shape(anomy)[0]):  # loop over depths
            min_a[ss, j, i] = np.nanmean(this_anom[j, good]) - np.nanstd(this_anom[j, good])
            max_a[ss, j, i] = np.nanmean(this_anom[j, good]) + np.nanstd(this_anom[j, good])

# DAC error = 0.02
file_list_2 = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/LiveOcean/simulated_dg_velocities/dg_vel_w_dac_err_2cms/ve_ew_v*_slp*_y*_*.pkl')
for i in range(len(file_list_2)):
    pkl_file = open(file_list_2[i], 'rb')
    MOD = pickle.load(pkl_file)
    pkl_file.close()
    glider_v_2 = MOD['dg_v'][:]
    model_v_avg_2 = MOD['model_u_at_mw_avg']
    if i < 1:
        v_2 = glider_v_2.copy()
        mod_v_avg_2 = model_v_avg_2.copy()
    else:
        v_2 = np.concatenate((v_2, glider_v_2.copy()), axis=1)
        mod_v_avg_2 = np.concatenate((mod_v_avg_2, model_v_avg_2), axis=1)
anomy_2 = v_2 - mod_v_avg_2  # velocity anomaly
mm_2 = np.nan * np.ones((len(slope_s), len(z_grid), len(w_s)))
avg_anom_2 = np.nan * np.ones((len(slope_s), len(z_grid), len(w_s)))
min_a_2 = np.nan * np.ones((len(slope_s), len(z_grid), 4))
max_a_2 = np.nan * np.ones((len(slope_s), len(z_grid), 4))
for ss in range(len(np.unique(slope_tag))):  # loop over slopes
    for i in range(len(np.unique(w_tag))):  # loop over w values
        inn = np.where((w_tag == w_s[i]) & (slope_tag == slope_s[ss]))[0]
        this_anom = anomy_2[:, inn]
        this_sq_anom = anomy_2[:, inn]**2
        good = np.where(~np.isnan(this_anom[100, :]))[0]
        mm_2[ss, :, i] = np.nanmean(this_sq_anom[:, good], axis=1)  # rms error
        avg_anom_2[ss, :, i] = np.nanmean(this_anom[:, good], axis=1)
        for j in range(np.shape(anomy_2)[0]):  # loop over depths
            min_a_2[ss, j, i] = np.nanmean(this_anom[j, good]) - np.nanstd(this_anom[j, good])
            max_a_2[ss, j, i] = np.nanmean(this_anom[j, good]) + np.nanstd(this_anom[j, good])

# PLOT v error and std
matplotlib.rcParams['figure.figsize'] = (12, 6.5)
f, ax = plt.subplots(1, 4, sharey=True)
# w_cols_2 = '#48D1CC', '#32CD32', '#FFA500', '#CD5C5C'
w_cols_2 = '#40E0D0', '#2E8B57', '#FFA500', '#CD5C5C'
for i in range(len(w_s)):
    ax[i].fill_betweenx(z_grid, min_a_2[1, :, i], x2=max_a_2[1, :, i], color='#C0C0C0', zorder=i, alpha=0.8)
    ax[i].fill_betweenx(z_grid, min_a[1, :, i], x2=max_a[1, :, i], color=w_cols_2[i], zorder=i, alpha=0.8)
    ax[i].plot(avg_anom[1, :, i], z_grid, color=w_cols[i], linewidth=3, zorder=4, label='dg w = ' + str(np.round(w_s[i]/100, decimals=3)) + ' m s$^{-1}$')
    ax[i].set_xlim([-.1, .1])
    ax[i].set_xlabel(r'[m s$^{-1}$]', fontsize=12)
    ax[i].set_title(r'($v$ - $\overline{v_{model}}$) (|w|=$\mathbf{' + str(np.round(w_s[i]/100, decimals=2)) + '}$ m s$^{-1}$)', fontsize=12)
ax[0].set_ylabel('z [m]', fontsize=12)
ax[0].set_ylim([-3000, 0])
good = np.where(~np.isnan(anomy[100, :]) & (slope_tag > 2))[0]
# ax[0].text(0.025, -2800, str(np.shape(anomy[:, slope_tag > 2])[1]) + ' profiles')
print(str(np.shape(anomy[:, slope_tag > 2])[1]) + ' profiles')
ax[0].grid()
ax[1].grid()
ax[2].grid()
plot_pro(ax[3])
if save_rms > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/reviewer_comments_minor_revisions/revised_figures/lo_mod_dg_vel_e.png', dpi=350)

matplotlib.rcParams['figure.figsize'] = (6.7, 7)
f, ax2 = plt.subplots()
for i in range(np.shape(mm)[2]):
    ax2.plot(mm[0, :, i], z_grid, linewidth=1.5, color=w_cols[i], linestyle='--',
             label='|w| = ' + str(np.round(w_s[i]/100, decimals=3)) + ' m s$^{-1}$ (s=1/' + str(np.int(slope_s[0])) + ')')
    ax2.plot(mm[1, :, i], z_grid, linewidth=1.5, color=w_cols[i],
             label='|w| = ' + str(np.round(w_s[i]/100, decimals=3)) + ' m s$^{-1}$ (s=1/' + str(np.int(slope_s[1])) + ')')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, fontsize=10, loc='lower right')
ax2.set_xlim([0, .005])
ax2.set_ylim([-3000, 0])
ax2.set_xlabel(r'[m$^2$ s$^{-2}$]', fontsize=12)
ax2.set_title(r'LiveOcean: Glider-Model Mean Square Error $\left< (v - \overline{v_{model}})^2 \right>$', fontsize=12)
ax2.set_ylabel('z [m]', fontsize=12)
plot_pro(ax2)
if save_rms > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/reviewer_comments_minor_revisions/revised_figures/lo_mod_dg_vel_rms_e.png', dpi=350)

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

avg_KE_model_ind_all = 2 * np.nanmean(ke_mod_tot, axis=1)
avg_KE_model_off_ind_all = 2 * np.nanmean(ke_mod_off_tot, axis=1)
avg_PE_model_ind_all = np.nanmean(pe_mod_tot, axis=1)
ax[0,1].plot(sc_x, avg_KE_model_ind_all[1:mm] / dk, color='k', label='KE$_{Model_{inst.,1}}$', linewidth=1, linestyle='--')
ax[0,1].plot([l_lim, sc_x[0]], avg_KE_model_ind_all[0:2] / dk, color='k', linewidth=1, linestyle='--')
ax[0,1].plot(sc_x, avg_KE_model_off_ind_all[1:mm] / dk, color='k', label='KE$_{Model_{inst.,2}}$', linewidth=1, linestyle='-.')
ax[0,1].plot([l_lim, sc_x[0]], avg_KE_model_off_ind_all[0:2] / dk, color='k', linewidth=1, linestyle='-.')
ax[0,0].plot(sc_x, avg_PE_model_ind_all[1:mm] / dk, color='k', label='PE$_{Model_{inst.}}$', linewidth=1, linestyle='--')

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
ax[0,0].set_ylabel('Variance per Vertical Wavenumber', fontsize=13)  # ' (and Hor. Wavenumber)')
# ax[0,0].set_xlabel('Mode Number', fontsize=12)
# ax[0,1].set_xlabel('Mode Number', fontsize=12)
ax[0,0].set_title('LiveOcean: Potential Energy (s = 1/' + str(np.int(slope_s[0])) + ')', fontsize=14)
ax[0,1].set_title('LiveOcean: Kinetic Energy (s = 1/' + str(np.int(slope_s[0])) + ')', fontsize=14)
handles, labels = ax[0,1].get_legend_handles_labels()
ax[0,1].legend(handles, labels, fontsize=10, loc=1)
handles, labels = ax[0,0].get_legend_handles_labels()
ax[0,0].legend(handles, labels, fontsize=10, loc=1)
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
ax[1,1].plot(sc_x, avg_KE_model_ind_all[1:mm] / dk, color='k', label='KE$_{Model_{inst.,1}}$', linewidth=1, linestyle='--')
ax[1,1].plot([l_lim, sc_x[0]], avg_KE_model_ind_all[0:2] / dk, color='k', linewidth=1, linestyle='--')
ax[1,1].plot(sc_x, avg_KE_model_off_ind_all[1:mm] / dk, color='k', label='KE$_{Model_{inst.,2}}$', linewidth=1, linestyle='-.')
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
ax[1,0].set_ylabel('Variance per Vertical Wavenumber', fontsize=13)  # ' (and Hor. Wavenumber)')
ax[1,0].set_xlabel('Mode Number', fontsize=13)
ax[1,1].set_xlabel('Mode Number', fontsize=13)
ax[1,0].set_title('LiveOcean: Potential Energy (s = 1/' + str(np.int(slope_s[1])) + ')', fontsize=14)
ax[1,1].set_title('LiveOcean: Kinetic Energy (s = 1/' + str(np.int(slope_s[1])) + ')', fontsize=14)
handles, labels = ax[1,1].get_legend_handles_labels()
ax[1,1].legend(handles, labels, fontsize=10, loc=1)
handles, labels = ax[1,0].get_legend_handles_labels()
ax[1,0].legend(handles, labels, fontsize=10, loc=1)
ax[1,0].grid()

plt.gcf().text(0.06, 0.9, 'a)', fontsize=12)
plt.gcf().text(0.5, 0.9, 'b)', fontsize=12)
plt.gcf().text(0.06, 0.48, 'c)', fontsize=12)
plt.gcf().text(0.5, 0.48, 'd)', fontsize=12)

plot_pro(ax[1,1])
if save_e > 0:
    f.savefig('/Users/jake/Documents/glider_flight_sim_paper/reviewer_comments_minor_revisions/revised_figures/lo_mod_energy_eddy.png', dpi=300)


# ----------------
# figure for defense
# ----------------
matplotlib.rcParams['figure.figsize'] = (7, 7)
f, ax = plt.subplots()
# glide (slope = 2)
i = 0
inn = np.where((w_tag == w_s[i]) & (slope_tag < 3))[0]
# KE
avg_KE = 2 * np.nanmean(ke_dg[:, inn], axis=1)
ax.plot(sc_x, avg_KE[1:mm] / dk, color=w_cols[3], label='simulated glider', linewidth=2.2)
ax.scatter(sc_x, avg_KE[1:mm] / dk, color=w_cols[3], s=10)  # DG KE
ax.plot([l_lim, sc_x[0]], avg_KE[0:2] / dk, color=w_cols[3], linewidth=2.2)  # DG KE_0
ax.scatter(l_lim, avg_KE[0] / dk, color=w_cols[3], s=10, facecolors='none')  # DG KE_0
# Model
good = np.where((ke_mod[1, :] < 1*10**0) & (slope_tag < 3))[0]
avg_KE_model = 2 * np.nanmean(ke_mod[:, good], axis=1)
ax.plot(sc_x, avg_KE_model[1:mm] / dk, color='k', label='model', linewidth=2.2)
ax.scatter(sc_x, avg_KE_model[1:mm] / dk, color='k', s=10)
ax.plot([l_lim, sc_x[0]], avg_KE_model[0:2] / dk, color='k', linewidth=2.2)
ax.scatter(l_lim, avg_KE_model[0] / dk, color='k', s=10, facecolors='none')
ax.set_title('LiveOcean: Kinetic Energy (s = 1/' + str(np.int(slope_s[0])) + ')', fontsize=16)

ax.plot([10**0, 10**1], [10**-1, 10**-4], linewidth=0.75, color='k')
ax.plot([10**0, 10**1], [10**-1, 10**-3], linewidth=0.75, color='k')
ax.text(8*10**0, 2*10**-3, '-2', fontsize=11)
ax.text(2.5*10**0, 2*10**-3, '-3', fontsize=11)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=12, loc=1)

limm = 5
ax.set_xlim([l_lim, 0.5 * 10 ** 2])
ax.set_ylim([10 ** (-4), 3 * 10 ** 2])
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('Variance per Vertical Wavenumber', fontsize=15)  # ' (and Hor. Wavenumber)')
ax.set_xlabel('Mode Number', fontsize=15)

plot_pro(ax)
f.savefig('/Users/jake/Desktop/defense_liveocean_ke_comp.png', dpi=300)

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
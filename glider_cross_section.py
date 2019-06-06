# I would like to create a project/tool/module
# Once imported, this tool will ask for glider (.nc) file location and dives to consider

# outputs will be a cross-section plot of the glider dive-climb profiles with density contours and DAC referenced
# absolute geostrophic velocity

# Class objects support two kinds of operations (attribute references and instantiation)
# Attribute references (Glider.name etc)
# Instantiation (x = Glider())
#   after instantiation you can define data attributes that are previously undeclared (x.test = 200)
#   after using an attribute you can use del x.test and it will disappear
#
#   after instantiation you can define a method (the special thing about methods is that the object is passed as the
#   first argument of the function) ... if x = Glider(35, np.array([30])) then x.f() is the same as Glider.f(x)
#
#   generally, instance variables are for unique data and class variables are for attributes and methods shared by all
#   instances of the class

import os
import numpy as np
import fnmatch
import datetime
from netCDF4 import Dataset
import gsw
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from toolkit import find_nearest, plot_pro, cart2pol, pol2cart, nanseg_interp


class Glider(object):

    def __init__(self, glider_no, d_num, path):
        self.ID = 'SG0' + str(glider_no)
        self.dives = d_num
        self.path = path
        result = []
        for this_dive in d_num:
            pattern = '*0' + str(this_dive) + '.nc'
            for root, dirs, files in os.walk(path):
                for name in files:
                    if fnmatch.fnmatch(name, pattern):
                        result.append(os.path.join(root, name))
        self.files = result
        self.project = Dataset(self.files[0], 'r').project
        self.num_profs = 2 * np.size(d_num)
        self.rho0 = 1025
        self.g = 9.812
        dep = np.nan * np.ones(len(self.files))
        target = np.nan * np.ones((len(self.files), 2))
        for m in range(len(self.files)):
            gd = Dataset(self.files[m], 'r')
            dep[m] = np.nanmax(gd['ctd_depth'][:])
            tgts = gd['log_TGT_LATLONG'][:]
            target[m, 0] = 10*np.int(tgts[0]) + 1*np.int(tgts[1]) + (1/60)*(10*np.int(tgts[2]) + 1*np.int(tgts[3])) + \
                (1/100)*(1/60)*(10*np.int(tgts[5]) + 1*np.int(tgts[6]))
            target[m, 1] = 10*np.int(tgts[10]) + 1*np.int(tgts[11]) + (1/60)*(10*np.int(tgts[12]) +
                                                                              1*np.int(tgts[13])) + \
                (1/100)*(1/60)*(10*np.int(tgts[15]) + 1*np.int(tgts[16]))
        self.avg_dep = np.nanmean(dep)
        self.target = target

    name = 'clementine'                         # this is an attribute of GLIDER

    @staticmethod
    def f():                                    # this is an attribute of GLIDER
        return 'I am buoyancy driven'

    def make_bin(self, bin_depth):
        t_out = []
        s_out = []
        o2_out = []
        lon_out = []
        lat_out = []
        time_out = []
        dac_u_out = []
        dac_v_out = []
        profile_tags = []
        for m in self.files:
            gd = Dataset(m, 'r')
            temp_d = []
            sal_d = []
            depth_d = []
            lon_d = []
            lat_d = []
            time_d = []
            press = gd['pressure'][:]
            pitch = gd['eng_pitchAng'][:]
            pitch_d = np.where((pitch < 1) & (np.abs(press - np.nanmax(press)) < 25))[0][-1]
            pitch_c = np.where((pitch > 1) & (np.abs(press - np.nanmax(press)) < 25))[0][0]
            temp_d.append(gd['temperature'][0:pitch_d])
            temp_d.append(gd['temperature'][pitch_c:])
            if 'aanderaa4330_instrument_dissolved_oxygen' in gd.variables:
                o2_d = []
                o2_inter = gd['aanderaa4330_instrument_dissolved_oxygen'][:]
                o2_d.append(o2_inter.data[0:pitch_d])
                o2_d.append(o2_inter.data[pitch_c:])
            # exceptions
            if (gd.glider == 30) & (gd.dive_number == 60):
                sal_d.append(gd['salinity_raw'][0:pitch_d])
                sal_d.append(gd['salinity'][pitch_c:])
            else:
                sal_d.append(gd['salinity'][0:pitch_d])
                sal_d.append(gd['salinity'][pitch_c:])
            depth_d.append(-1*gsw.z_from_p(press[0:pitch_d], np.nanmean(gd['latitude'][:])))
            depth_d.append(-1*gsw.z_from_p(press[pitch_c:], np.nanmean(gd['latitude'][:])))

            secs_per_day = 86400.0
            datenum_start = 719163  # jan 1 1970
            start_time = gd.start_time
            time_ctd = gd['ctd_time']
            time_from = time_ctd - start_time
            time_corr = datenum_start + start_time / (60 * 60 * 24) + time_from / secs_per_day
            time_d.append(time_corr[0:pitch_d])
            time_d.append(time_corr[pitch_c:])

            lon_d.append(gd['longitude'][0:pitch_d])
            lon_d.append(gd['longitude'][pitch_c:])
            lat_d.append(gd['latitude'][0:pitch_d])
            lat_d.append(gd['latitude'][pitch_c:])
            dac_u_out.append(gd['depth_avg_curr_east'][:])
            dac_u_out.append(gd['depth_avg_curr_east'][:])
            dac_v_out.append(gd['depth_avg_curr_north'][:])
            dac_v_out.append(gd['depth_avg_curr_north'][:])
            profile_tags.append(np.float(gd['log_DIVE'][:]))
            profile_tags.append(np.float(gd['log_DIVE'][:]) + np.float(0.5))

            # max depth attained
            d_test = -1*gsw.z_from_p(press, np.nanmean(gd['latitude'][:]))
            dep_max = np.round(d_test.max())
            deepest_bin = find_nearest(bin_depth, dep_max)[0]
            bin_up = bin_depth[0:(deepest_bin - 1)]
            bin_down = bin_depth[2:(deepest_bin + 1)]
            bin_cen = bin_depth[1:deepest_bin]

            temp_g = np.nan * np.ones((np.size(bin_depth), 2))
            sal_g = np.nan * np.ones((np.size(bin_depth), 2))
            lon_g = np.nan * np.ones((np.size(bin_depth), 2))
            lat_g = np.nan * np.ones((np.size(bin_depth), 2))
            time_g = np.nan * np.ones((np.size(bin_depth), 2))
            if 'aanderaa4330_instrument_dissolved_oxygen' in gd.variables:
                o2_g = np.nan * np.ones((np.size(bin_depth), 2))

            for k in range(2):
                temp = temp_d[k]
                sal = sal_d[k]
                depth = depth_d[k]
                lon = lon_d[k]
                lat = lat_d[k]
                time = time_d[k]
                if 'aanderaa4330_instrument_dissolved_oxygen' in gd.variables:
                    o2 = o2_d[k]

                # -- Case z = 0
                dp_in_d_1 = depth < bin_cen[0]
                if np.size(dp_in_d_1) >= 2:
                    temp_g[0, k] = np.nanmean(temp[dp_in_d_1])
                    sal_g[0, k] = np.nanmean(sal[dp_in_d_1])
                    lon_g[0, k] = np.nanmean(lon[dp_in_d_1])
                    lat_g[0, k] = np.nanmean(lat[dp_in_d_1])
                    time_g[0, k] = np.nanmean(time[dp_in_d_1])
                    if 'aanderaa4330_instrument_dissolved_oxygen' in gd.variables:
                        o2_g[0, k] = np.nanmean(o2[dp_in_d_1])
                # -- Case z > 0
                for j in range(np.size(bin_cen)):
                    i = j + 1
                    dp_in_d = (depth > bin_up[j]) & (depth < bin_down[j])
                    if np.size(dp_in_d) > 2:
                        temp_g[i, k] = np.nanmean(temp[dp_in_d])
                        sal_g[i, k] = np.nanmean(sal[dp_in_d])
                        lon_g[i, k] = np.nanmean(lon[dp_in_d])
                        lat_g[i, k] = np.nanmean(lat[dp_in_d])
                        time_g[i, k] = np.nanmean(time[dp_in_d])
                        if 'aanderaa4330_instrument_dissolved_oxygen' in gd.variables:
                            o2_g[i, k] = np.nanmean(o2[dp_in_d])
                # -- Case last_bin
                if deepest_bin == (len(bin_depth) - 1):
                    dp_in_d_e = (depth > bin_cen[-1]) & (depth < bin_cen[-1] + 75)
                    if np.size(dp_in_d_e) > 2:
                        temp_g[-1, k] = np.nanmean(temp[dp_in_d_e])
                        sal_g[-1, k] = np.nanmean(sal[dp_in_d_e])
                        lon_g[-1, k] = np.nanmean(lon[dp_in_d_e])
                        lat_g[-1, k] = np.nanmean(lat[dp_in_d_e])
                        time_g[-1, k] = np.nanmean(time[dp_in_d_e])
                        if 'aanderaa4330_instrument_dissolved_oxygen' in gd.variables:
                            o2_g[-1, k] = np.nanmean(o2[dp_in_d_e])
            for i in range(2):
                temp_g[:, i] = nanseg_interp(bin_depth, temp_g[:, i])
                sal_g[:, i] = nanseg_interp(bin_depth, sal_g[:, i])

            if np.size(s_out) < 1:
                s_out = sal_g
                t_out = temp_g
                lon_out = lon_g
                lat_out = lat_g
                time_out = time_g
                if 'aanderaa4330_instrument_dissolved_oxygen' in gd.variables:
                    o2_out = o2_g
            else:
                s_out = np.concatenate((s_out, sal_g), axis=1)
                t_out = np.concatenate((t_out, temp_g), axis=1)
                lon_out = np.concatenate((lon_out, lon_g), axis=1)
                lat_out = np.concatenate((lat_out, lat_g), axis=1)
                time_out = np.concatenate((time_out, time_g), axis=1)
                if 'aanderaa4330_instrument_dissolved_oxygen' in gd.variables:
                    o2_out = np.concatenate((o2_out, o2_g), axis=1)

        if 'aanderaa4330_instrument_dissolved_oxygen' in gd.variables:
            out = {'time': time_out, 'lon': lon_out, 'lat': lat_out,
                   'temp': t_out, 'sal': s_out, 'o2': o2_out,
                   'dac_u': dac_u_out, 'dac_v': dac_v_out, 'profs': profile_tags}
            return out
        else:
            out = {'time': time_out, 'lon': lon_out, 'lat': lat_out,
                   'temp': t_out, 'sal': s_out,
                   'dac_u': dac_u_out, 'dac_v': dac_v_out, 'profs': profile_tags}
            return out

    def density(self, bin_depth, ref_lat, temp, sal, lon, lat):
        press = gsw.p_from_z(-1*bin_depth, ref_lat)
        sa = np.nan * np.ones(np.shape(temp))
        ct = np.nan * np.ones(np.shape(temp))
        theta = np.nan * np.ones(np.shape(temp))
        sig0 = np.nan * np.ones(np.shape(temp))
        sig2 = np.nan * np.ones(np.shape(temp))
        N2 = np.nan * np.ones(np.shape(temp))
        for i in range(self.num_profs):
            sa[:, i] = gsw.SA_from_SP(sal[:, i], press, lon[:, i], lat[:, i])
            ct[:, i] = gsw.CT_from_t(sa[:, i], temp[:, i], press)
            theta[:, i] = gsw.pt_from_CT(sa[:, i], ct[:, i])
            sig0[:, i] = gsw.sigma0(sa[:, i], ct[:, i])
            sig2[:, i] = gsw.sigma2(sa[:, i], ct[:, i])
            N2[1:, i] = gsw.Nsquared(sa[:, i], ct[:, i], press, lat=ref_lat)[0]

        return sa, ct, theta, sig0, sig2, N2

    # this function will parse a set of dive-climb cycles into transects that are each bounded by glider targets. The
    # target associated with each dive-climb cycle is used to separate profiles into transects that are terminated when
    # glider target changes (when it reaches a targets and begins to head towards a new one). If transects are repeated,
    # each occupation of the line is treated as its own transect
    def transect_cross_section_1(self, bin_depth, sig0_0, ct_0, sa_0,
                                 lon_0, lat_0, dac_u_0, dac_v_0, profile_tags_0, sigth_levels):
        deep_shr_max = 0.1
        deep_shr_max_dep = 3500
        # -- for computation of eta (reference to average density profile and gradient of average density profile
        # - the average should span seasons rather than total mission
        sigma_theta_avg = np.nanmean(sig0_0, axis=1)
        ct_avg = np.nanmean(ct_0, axis=1)
        sa_avg = np.nanmean(sa_0, axis=1)
        ddz_avg_sigma = np.gradient(sigma_theta_avg, -1 * bin_depth)

        # ____________________________________________________________________________________
        def group_consecutives(vals, step=1):
            """Return list of consecutive lists of numbers from vals (number list)."""
            run = []
            result = [run]
            expect = None
            for v in vals:
                if (v == expect) or (expect is None):
                    run.append(v)
                else:
                    run = [v]
                    result.append(run)
                expect = v + step
            return result

        # separate dives into unique transects
        target_test = 1000000 * self.target[:, 0] + np.round(self.target[:, 1], 3)
        unique_targets = np.unique(target_test)
        transects = []
        for m in range(len(unique_targets)):
            indices = np.where(target_test == unique_targets[m])[0]
            if len(indices) > 1:
                # make sure each set has more than one dive
                sets = group_consecutives(indices, step=1)
                to_go = []
                for k in range(len(sets)):
                    if len(sets[k]) > 1:
                        to_go.append(sets[k])
                # transects.append(group_consecutives(indices, step=1))
                transects.append(to_go)

        ds_out = []
        dist_out = []
        avg_ct_out = []
        avg_sa_out = []
        avg_sig_out = []
        v_g_out = []
        v_g_east_out = []
        v_g_north_out = []
        vbt_out = []
        shear_out = []
        v_g_dir_out = []
        isopycdep_out = []
        isopycx_out = []
        mwe_lon_out = []
        mwe_lat_out = []
        DACe_MW_out = []
        DACn_MW_out = []
        profile_tags_out = []
        info_out = []
        for n in range(len(transects)):  # loop over all transect segments
            print('transects = ' + str(transects[n]))
            for o in range(len(transects[n])):  # loop over all times a glider executed that segment
                this_transect = transects[n][o]
                print(this_transect)
                index_start = 2 * this_transect[0]
                index_end = 2 * (this_transect[-1] + 1)
                print('index_start = ' + str(index_start))
                print('index_end = ' + str(index_end))
                order_set = np.arange(0, 2 * len(this_transect), 2)
                sig0 = sig0_0[:, index_start:index_end]
                ct = ct_0[:, index_start:index_end]
                sa = sa_0[:, index_start:index_end]
                lon = lon_0[:, index_start:index_end]
                lat = lat_0[:, index_start:index_end]
                dac_u = dac_u_0[index_start:index_end]
                dac_v = dac_v_0[index_start:index_end]
                profile_tags = profile_tags_0[index_start:index_end]
                print('dive cycles in this transect = ' + str(profile_tags))
                # ____________________________________________________________________________________
                # order_set = np.arange(0, self.num_profs, 2)

                info = np.nan * np.zeros((3, len(profile_tags) - 1))
                sigma_theta_out = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
                shear = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
                shear_east = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
                shear_north = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
                shear_dir = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
                # eta = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
                # eta_theta = np.nan * np.zeros((np.size(grid), np.size(this_set) - 1))
                avg_sig_pd = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
                avg_ct_pd = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
                avg_sa_pd = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
                isopycdep = np.nan * np.zeros((np.size(sigth_levels), len(profile_tags)))
                isopycx = np.nan * np.zeros((np.size(sigth_levels), len(profile_tags)))
                vbt = np.nan * np.zeros(len(profile_tags))
                DACe_MW = np.nan * np.zeros(len(profile_tags))
                DACn_MW = np.nan * np.zeros(len(profile_tags))
                ds = np.nan * np.zeros(len(profile_tags))
                dist = np.nan * np.zeros(np.shape(lon))
                dist_st = 0
                distance = 0
                for i in order_set:  # loop over dive cycles in each transect (each i corresponds to a v estimate)
                    # i = this dive
                    # M
                    lon_start = lon[0, i]  # start position of dive i along the transect
                    lat_start = lat[0, i]
                    lon_finish = lon[0, i + 1]  # end position of dive i + 1 along the transect
                    lat_finish = lat[0, i + 1]
                    lat_ref = 0.5 * (lat_start + lat_finish)
                    f_m = np.pi * np.sin(np.deg2rad(lat_ref)) / (12 * 1800)  # Coriolis parameter [s^-1]
                    dxs_m = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (lon_finish - lon_start)  # zonal sfc disp [km]
                    dys_m = 1.852 * 60 * (lat_finish - lat_start)  # meridional sfc disp [km]
                    # vector pointing along the center M dive direction
                    ds_a, ang_sfc_m = cart2pol(dxs_m, dys_m)

                    dx = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * \
                        (np.concatenate([lon[:, i], np.flipud(lon[:, i + 1])]) - lon[0, i])
                    dy = 1.852 * 60 * (np.concatenate([lat[:, i], np.flipud(np.array(lat[:, i + 1]))]) - lat[0, i])
                    ss, ang = cart2pol(dx, dy)
                    xx, yy = pol2cart(ss, ang - ang_sfc_m)
                    length1 = np.size(lon[:, i])
                    dist[:, i] = dist_st + xx[0:length1]
                    dist[:, i + 1] = dist_st + np.flipud(xx[length1:])
                    dist_st = dist_st + ds_a

                    distance = distance + np.nanmedian(xx)  # 0.5*ds # distance for each velocity estimate
                    ds[i] = distance
                    # DACe = dac_u[i]  # zonal depth averaged current [m/s]
                    # DACn = dac_v[i]  # meridional depth averaged current [m/s]
                    if i < 2:
                        # DACe = np.nanmean([[dac_u[i]], [dac_u[i + 2]]])
                        # DACn = np.nanmean([[dac_v[i]], [dac_v[i + 2]]])
                        DACe = dac_u[i]
                        DACn = dac_v[i]
                    elif (i >= 2) and (i < len(profile_tags) - 2):
                        # DACe = np.nanmean([[dac_u[i - 1]], [dac_u[i]], [dac_u[i + 2]]])
                        # DACn = np.nanmean([[dac_v[i - 1]], [dac_v[i]], [dac_v[i + 2]]])
                        DACe = dac_u[i]
                        DACn = dac_v[i]
                    elif i >= len(profile_tags) - 2:
                        # DACe = np.nanmean([[dac_u[i - 1]], [dac_u[i]]])
                        # DACn = np.nanmean([[dac_v[i - 1]], [dac_v[i]]])
                        DACe = dac_u[i]
                        DACn = dac_v[i]
                    mag_DAC, ang_DAC = cart2pol(DACe, DACn)
                    DACat, DACpot = pol2cart(mag_DAC, ang_DAC - ang_sfc_m)
                    vbt[i] = DACpot  # across-track barotropic current comp (>0 to left)
                    DACe_MW[i] = DACe.copy()
                    DACn_MW[i] = DACn.copy()

                    # W
                    if i >= order_set[-1]:
                        lon_finish = lon[0, -1]
                        lat_finish = lon[0, -1]
                        DACe = np.nanmean([[dac_u[i]], [dac_u[-1]]])  # zonal depth averaged current [m/s]
                        DACn = np.nanmean([[dac_v[i]], [dac_v[-1]]])  # meridional depth averaged current [m/s]
                    else:
                        lon_finish = lon[0, i + 3]
                        lat_finish = lat[0, i + 3]
                        DACe = np.nanmean([[dac_u[i]], [dac_u[i + 2]]])  # zonal depth averaged current [m/s]
                        DACn = np.nanmean([[dac_v[i]], [dac_v[i + 2]]])  # meridional depth averaged current [m/s]
                    DACe_MW[i + 1] = DACe.copy()
                    DACn_MW[i + 1] = DACn.copy()
                    lat_ref = 0.5 * (lat_start + lat_finish)
                    f_w = np.pi * np.sin(np.deg2rad(lat_ref)) / (12 * 1800)  # Coriolis parameter [s^-1]
                    dxs = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (lon_finish - lon_start)  # zonal sfc disp [km]
                    dys = 1.852 * 60 * (lat_finish - lat_start)  # meridional sfc disp [km]
                    # vector pointing along the center W dive direction
                    ds_w, ang_sfc_w = cart2pol(dxs, dys)
                    distance = distance + (ds_a - np.nanmedian(xx))  # distance for each velocity estimate
                    ds[i + 1] = distance
                    mag_DAC, ang_DAC = cart2pol(DACe, DACn)
                    DACat, DACpot = pol2cart(mag_DAC, ang_DAC - ang_sfc_w)
                    vbt[i + 1] = DACpot  # across-track barotropic current comp (>0 to left)

                    shearM = np.nan * np.zeros(np.size(bin_depth))
                    shear_x_M = np.nan * np.zeros(np.size(bin_depth))
                    shear_y_M = np.nan * np.zeros(np.size(bin_depth))
                    shearM_dir = 2.0 * np.ones(np.size(bin_depth))
                    shearW = np.nan * np.zeros(np.size(bin_depth))
                    shear_x_W = np.nan * np.zeros(np.size(bin_depth))
                    shear_y_W = np.nan * np.zeros(np.size(bin_depth))
                    shearW_dir = 2.0 * np.ones(np.size(bin_depth))
                    # etaM = np.nan * np.zeros(np.size(bin_depth))
                    # etaW = np.nan * np.zeros(np.size(bin_depth))
                    p_avg_sig_M = np.nan * np.zeros(np.size(bin_depth))
                    p_avg_sig_W = np.nan * np.zeros(np.size(bin_depth))
                    p_avg_ct_M = np.nan * np.zeros(np.size(bin_depth))
                    p_avg_ct_W = np.nan * np.zeros(np.size(bin_depth))
                    p_avg_sa_M = np.nan * np.zeros(np.size(bin_depth))
                    p_avg_sa_W = np.nan * np.zeros(np.size(bin_depth))
                    # eta_thetaM = np.nan * np.zeros(np.size(bin_depth))
                    # eta_thetaW = np.nan * np.zeros(np.size(bin_depth))
                    sigma_theta_pa_M = np.nan * np.zeros(np.size(bin_depth))
                    sigma_theta_pa_W = np.nan * np.zeros(np.size(bin_depth))
                    lon_pa_M = np.nan * np.zeros(np.size(bin_depth))
                    lon_pa_W = np.nan * np.zeros(np.size(bin_depth))
                    lat_pa_M = np.nan * np.zeros(np.size(bin_depth))
                    lat_pa_W = np.nan * np.zeros(np.size(bin_depth))
                    # LOOP OVER EACH BIN_DEPTH
                    for j in range(np.size(bin_depth)):
                        # find array of indices for M / W sampling
                        if i < 2:
                            # c_i_m = np.arange(i, i + 3)
                            c_i_m = []  # omit partial "M" estimate
                            c_i_w = np.arange(i, i + 4)
                        elif (i >= 2) and (i < len(profile_tags) - 2):
                            c_i_m = np.arange(i - 1, i + 3)
                            c_i_w = np.arange(i, i + 4)
                        elif i >= len(profile_tags) - 2:
                            # c_i_m = np.arange(i - 1, len(profile_tags))
                            c_i_m = []  # omit partial "M" estimated
                            c_i_w = []
                        nm = np.size(c_i_m)
                        nw = np.size(c_i_w)

                        # for M profile compute shear and eta
                        if nm > 2 and np.size(sig0[j, c_i_m]) > 2:
                            sigmathetaM = sig0[j, c_i_m]
                            sigma_theta_pa_M[j] = np.nanmean(sigmathetaM)  # average density across 4 profiles
                            lon_pa_M[j] = np.nanmean(lon[j, c_i_m])  # avg lat/lon across M/W profiles
                            lat_pa_M[j] = np.nanmean(lat[j, c_i_m])
                            # thetaM = theta[j, c_i_m]
                            # ctM = df_ct_set.iloc[j, c_i_m]
                            imv = ~np.isnan(np.array(sig0[j, c_i_m]))
                            c_i_m_in = c_i_m[imv]

                            if np.size(c_i_m_in) > 1:
                                xM = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (
                                        lon[j, c_i_m_in] - lon[j, c_i_m_in[0]])  # E loc [km]
                                yM = 1.852 * 60 * (
                                        lat[j, c_i_m_in] - lat[j, c_i_m_in[0]])  # N location [km]
                                XXM = np.concatenate(
                                    [np.ones((np.size(sigmathetaM[imv]), 1)), np.transpose(np.atleast_2d(np.array(xM))),
                                    np.transpose(np.atleast_2d(np.array(yM)))], axis=1)
                                d_anom0M = sigmathetaM[imv] - np.nanmean(sigmathetaM[imv])
                                ADM = np.squeeze(np.linalg.lstsq(XXM, np.transpose(np.atleast_2d(np.array(d_anom0M))))[0])
                                drhodxM = ADM[1]  # zonal gradient [kg/m^3/km]
                                drhodyM = ADM[2]  # meridional gradient [kg/m^3km]
                                drhodsM, ang_drhoM = cart2pol(drhodxM, drhodyM)  # angle and magnitude of gradient
                                # component along/acrosstrack
                                drhodatM, drhodpotM = pol2cart(drhodsM, ang_drhoM - ang_sfc_m)
                                shearM[j] = -self.g * drhodatM / (self.rho0 * f_m)  # shear to port of track [m/s/km]
                                if (np.abs(shearM[j]) > deep_shr_max) and bin_depth[j] >= deep_shr_max_dep:
                                    shearM[j] = np.sign(shearM[j]) * deep_shr_max
                                # project cross-track velocity vertical shear into E/N
                                # - ang_drhoM = angle of the density gradient! (3/4 contributing density measurements)
                                # - ang_sfc_m = angle of either the middle M dive or tail ends of the W dives
                                angle_of_shear = ang_sfc_m + np.pi/2  # angle in transect direction + 90deg
                                shear_x_M[j], shear_y_M[j] = pol2cart(shearM[j], angle_of_shear)  # project onto E/N

                                # flow along slope contours (transect heading in radians pi/2 - pi or 3pi/2 - 2pi)
                                if ((ang_sfc_m >= 5.0 * np.pi / 8.0) & (ang_sfc_m <= 7.0 * np.pi / 8.0)) | (
                                        (ang_sfc_m >= 13.0 * np.pi / 8.0) & (ang_sfc_m <= 15.0 * np.pi / 8.0)):
                                    shearM_dir[j] = 0
                                # flow across slope contours (transect heading in radians 0 - pi/2 or pi - 3pi/2)
                                if ((ang_sfc_m >= np.pi / 8.0) & (ang_sfc_m <= 3.0 * np.pi / 8.0)) | (
                                        (ang_sfc_m >= 9.0 * np.pi / 8.0) & (ang_sfc_m <= 11.0 * np.pi / 8.0)):
                                    shearM_dir[j] = 1

                                # --- Computation of Eta
                                # -- isopycnal position is average position of a few dives
                                # etaM[j] = (sigma_theta_avg[j] - np.nanmean(sigmathetaM[imv])) / ddz_avg_sigma[j]
                                # eta_thetaM[j] = (theta_avg[j] - np.nanmean(thetaM[imv])) / ddz_avg_theta[j]
                                # -- isopycnal position is position on this single profile
                                # etaM[j] = (sigma_theta_avg[j] - sig0[j, i]) / ddz_avg_sigma[j]  # j=dp, i=prof_ind
                                # eta_thetaM[j] = (ct_avg[j] - df_ct_set.iloc[j, i]) / ddz_avg_ct[j]

                                # testing and plotting of across-track angle computations
                                # if j == 180:
                                #     print('dive cycles in this m estimation = ' +
                                #           str(profile_tags[c_i_m[0]]) + ' - ' + str(profile_tags[c_i_m[-1]]))
                                #     f, ax = plt.subplots()
                                #     # position from start of contributing m/w profiles
                                #     x_ds = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (
                                #             lon[:, c_i_m_in] - lon[0, c_i_m_in[0]])
                                #     print(np.shape(x_ds))
                                #     y_ds = 1.852 * 60 * (lat[:, c_i_m_in] - lat[0, c_i_m_in[0]])  # N
                                #     xMj = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (
                                #             lon[j, c_i_m_in] - lon[0, c_i_m_in[0]])  # E
                                #     yMj = 1.852 * 60 * (lat[j, c_i_m_in] - lat[0, c_i_m_in[0]])  # N
                                #     ax.scatter(x_ds, y_ds, 4, color='k', label='glider path')
                                #     ax.scatter(x_ds[15, :], y_ds[15, :], color='k', s=50)
                                #     ax.scatter(xMj, yMj, color='r', s=120, label='loc. of density obs. at 2000m')
                                #     ax.plot(xMj, yMj, color='r')
                                #     # location of this velocity profile
                                #     if i < order_set[-1]:
                                #         me_lon = lon[np.where(np.isfinite(lon[:, i]))[0], i][-1]  # m
                                #         me_lat = lat[np.where(np.isfinite(lon[:, i]))[0], i][-1]
                                #         # mwe_lon[m + 1] = lon[0, m + 1]  # w
                                #         # mwe_lat[m + 1] = lat[0, m + 1]
                                #     else:
                                #         me_lon = lon[np.where(np.isfinite(lon[:, i]))[0], i][-1] # end m
                                #         me_lat = lat[np.where(np.isfinite(lat[:, i]))[0], i][-1]
                                #     # plot dive number associated with location of v profile
                                #     x_pr = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (me_lon - lon[0, c_i_m_in[0]])
                                #     y_pr = 1.852 * 60 * (me_lat - lat[0, c_i_m_in[0]])
                                #     ax.text(x_pr, y_pr + 1, profile_tags[i])
                                #     # vectors
                                #     testx, testy = pol2cart(1, ang_sfc_m)
                                #     ax.quiver(x_pr, y_pr, testx, testy, color='m', scale=4, label='along-track dir.')   # dives dir
                                #     ax.quiver(x_pr, y_pr, drhodxM, drhodyM, color='g', scale=.01, label='den. grad. dir.')  # den gradient dir
                                #     # shear to port
                                #     testshx, testshy = pol2cart(shearM[j], ang_sfc_m + np.pi/2)
                                #     ax.quiver(x_pr, y_pr, testshx, testshy, color='y', scale=.1, label='shear to port')  # shear to port dir
                                #     # cross-track shear projected into E
                                #     ax.quiver(x_pr, y_pr, shear_x_M[j], 0, color='b', scale=.1, label='shear E')
                                #     # cross-track shear projected into N
                                #     ax.quiver(x_pr, y_pr, 0, shear_y_M[j], color='b', scale=.1, label='shear N')
                                #     ax.set_title('M/W Schematic at 2000m, Transect Component = ' + str(i))
                                #     ax.axis([np.nanmin(x_ds)-3, np.nanmax(x_ds)+3, np.nanmin(y_ds)-3, np.nanmax(y_ds)+3])
                                #     ax.set_aspect('equal')
                                #     ax.set_xlabel('km East')
                                #     ax.set_ylabel('km North')
                                #     handles, labels = ax.get_legend_handles_labels()
                                #     ax.legend(handles, labels, fontsize=10)
                                #     plot_pro(ax)

                                # average isopycnal value at each depth level j, to be used to compute eta but
                                # using a background profile of the users desire. this average spans the same number
                                # of profiles used in computing vertical shear profiles (and then velocity profiles)
                                p_avg_sig_M[j] = np.nanmean(sig0[j, c_i_m])
                                p_avg_ct_M[j] = np.nanmean(ct[j, c_i_m])
                                p_avg_sa_M[j] = np.nanmean(sa[j, c_i_m])

                        # for W profile compute shear and eta
                        if nw > 2 and np.size(sig0[j, c_i_w]) > 2:
                            sigmathetaW = sig0[j, c_i_w]
                            sigma_theta_pa_W[j] = np.nanmean(sigmathetaW)  # average density across 4 profiles
                            lon_pa_W[j] = np.nanmean(lon[j, c_i_w])  # avg lat/lon across M/W profiles
                            lat_pa_W[j] = np.nanmean(lon[j, c_i_w])
                            # thetaW = df_theta_set.iloc[j, c_i_w]
                            # ctW = df_theta_set.iloc[j, c_i_w]
                            iwv = ~np.isnan(np.array(sig0[j, c_i_w]))
                            c_i_w_in = c_i_w[iwv]

                            if np.sum(c_i_w_in) > 1:
                                xW = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (
                                        lon[j, c_i_w_in] - lon[j, c_i_w_in[0]])  # E loc [km]
                                yW = 1.852 * 60 * (
                                        lat[j, c_i_w_in] - lat[j, c_i_w_in[0]])  # N location [km]
                                XXW = np.concatenate(
                                    [np.ones((np.size(sigmathetaW[iwv]), 1)), np.transpose(np.atleast_2d(np.array(xW))),
                                     np.transpose(np.atleast_2d(np.array(yW)))], axis=1)
                                d_anom0W = sigmathetaW[iwv] - np.nanmean(sigmathetaW[iwv])
                                ADW = np.squeeze(np.linalg.lstsq(XXW, np.transpose(np.atleast_2d(np.array(d_anom0W))))[0])
                                drhodxW = ADW[1]  # [zonal gradient [kg/m^3/km]
                                drhodyW = ADW[2]  # [meridional gradient [kg/m^3km]
                                drhodsW, ang_drhoW = cart2pol(drhodxW, drhodyW)
                                drhodatW, drhodpotW = pol2cart(drhodsW, ang_drhoW - ang_sfc_w)
                                shearW[j] = -self.g * drhodatW / (self.rho0 * f_w)  # shear to port of track [m/s/km]
                                if (np.abs(shearW[j]) > deep_shr_max) and bin_depth[j] >= deep_shr_max_dep:
                                    shearW[j] = np.sign(shearW[j]) * deep_shr_max
                                angle_of_shear = ang_sfc_w + np.pi/2
                                shear_x_W[j], shear_y_W[j] = pol2cart(shearW[j], angle_of_shear)

                                # flow along slope contours (transect heading in radians pi/2 - pi or 3pi/2 - 2pi)
                                if ((ang_sfc_w >= 5.0 * np.pi / 8.0) & (ang_sfc_w <= 7.0 * np.pi / 8.0)) | (
                                        (ang_sfc_w >= 13.0 * np.pi / 8.0) & (ang_sfc_w <= 15.0 * np.pi / 8.0)):
                                    shearW_dir[j] = 0
                                # flow across slope contours (transect heading in radians 0 - pi/2 or pi - 3pi/2)
                                if ((ang_sfc_w >= np.pi / 8.0) & (ang_sfc_w < 3.0 * np.pi / 8.0)) | (
                                        (ang_sfc_w >= 9.0 * np.pi / 8.0) & (ang_sfc_w < 11.0 * np.pi / 8.0)):
                                    shearW_dir[j] = 1

                                # --- Computation of Eta
                                # -- isopycnal position is average position of a few dives
                                # etaW[j] = (sigma_theta_avg[j] - np.nanmean(sigmathetaW[iwv])) / ddz_avg_sigma[j]
                                # eta_thetaW[j] = (theta_avg[j] - np.nanmean(thetaW[iwv])) / ddz_avg_theta[j]
                                # -- isopycnal position is position on this single profile
                                # etaW[j] = (sigma_theta_avg[j] - df_den_set.iloc[j, i + 1]) / ddz_avg_sigma[j]
                                # eta_thetaW[j] = (ct_avg[j] - df_ct_set.iloc[j, i + 1]) / ddz_avg_ct[j]


                                # average isopycnal value at each depth level j, to be used to compute eta but
                                # using a background profile of the users desire. this average spans the same number
                                # of profiles used in computing vertical shear profiles (and then velocity profiles)
                                p_avg_sig_W[j] = np.nanmean(sig0[j, c_i_w])
                                p_avg_ct_W[j] = np.nanmean(ct[j, c_i_w])
                                p_avg_sa_W[j] = np.nanmean(sa[j, c_i_w])

                        # END LOOP OVER EACH BIN_DEPTH

                    # OUTPUT FOR EACH TRANSECT (at least 2 DIVES)
                    # because this is M/W profiling, for a 3 dive transect, only 5 profiles of shear and eta are compiled
                    sigma_theta_out[:, i] = sigma_theta_pa_M
                    shear[:, i] = shearM
                    shear_east[:, i] = shear_x_M
                    shear_north[:, i] = shear_y_M
                    shear_dir[:, i] = shearM_dir
                    avg_sig_pd[:, i] = p_avg_sig_M
                    avg_ct_pd[:, i] = p_avg_ct_M
                    avg_sa_pd[:, i] = p_avg_sa_M
                    # eta[:, i] = etaM
                    # eta_theta[:, i] = eta_thetaM
                    info[0, i] = profile_tags[i]
                    info[1, i] = np.nanmean(lon_pa_M)
                    info[2, i] = np.nanmean(lat_pa_M)
                    if i < len(profile_tags) - 2:
                        sigma_theta_out[:, i + 1] = sigma_theta_pa_W
                        shear[:, i + 1] = shearW
                        shear_east[:, i + 1] = shear_x_W
                        shear_north[:, i + 1] = shear_y_W
                        shear_dir[:, i + 1] = shearW_dir
                        avg_sig_pd[:, i + 1] = p_avg_sig_W
                        avg_ct_pd[:, i + 1] = p_avg_ct_W
                        avg_sa_pd[:, i + 1] = p_avg_sa_W
                        # eta[:, i + 1] = etaW
                        # eta_theta[:, i + 1] = eta_thetaW
                        info[0, i + 1] = profile_tags[i + 1]
                        info[1, i + 1] = np.nanmean(lon_pa_W)
                        info[2, i + 1] = np.nanmean(lat_pa_W)

                    # ISOPYCNAL DEPTHS ON PROFILES ALONG EACH TRANSECT
                    sigthmin = np.nanmin(np.array(sig0[:, i]))
                    sigthmax = np.nanmax(np.array(sig0[:, i]))
                    isigth = np.where((sigth_levels > sigthmin) & (sigth_levels < sigthmax))
                    isopycdep[isigth, i] = np.interp(sigth_levels[isigth], sig0[:, i], bin_depth)
                    isopycx[isigth, i] = np.interp(sigth_levels[isigth], sig0[:, i], dist[:, i])

                    sigthmin = np.nanmin(np.array(sig0[:, i + 1]))
                    sigthmax = np.nanmax(np.array(sig0[:, i + 1]))
                    isigth = np.where((sigth_levels > sigthmin) & (sigth_levels < sigthmax))
                    isopycdep[isigth, i + 1] = np.interp(sigth_levels[isigth], sig0[:, i + 1], bin_depth)
                    isopycx[isigth, i + 1] = np.interp(sigth_levels[isigth], sig0[:, i + 1], dist[:, i + 1])

                    # ---- END LOOP OVER EACH DIVE IN TRANSECT

                # FOR EACH TRANSECT COMPUTE GEOSTROPHIC VELOCITY
                vbc_g = np.nan * np.zeros(np.shape(shear))
                vbc_g_e = np.nan * np.zeros(np.shape(shear))
                vbc_g_n = np.nan * np.zeros(np.shape(shear))
                v_g = np.nan * np.zeros((np.size(bin_depth), len(profile_tags)))
                v_g_dir = np.nan * np.zeros((np.size(bin_depth), len(profile_tags)))
                v_g_east = np.nan * np.zeros((np.size(bin_depth), len(profile_tags)))
                v_g_north = np.nan * np.zeros((np.size(bin_depth), len(profile_tags)))
                for m in range(len(profile_tags) - 1):
                    iq = np.where(~np.isnan(shear[:, m]))[0]
                    if np.size(iq) > 10:
                        z2 = -1.0 * bin_depth[iq]
                        vrel = cumtrapz(0.001 * shear[iq, m], x=z2, initial=0)
                        vrel_av = np.trapz(vrel / (z2[-1] - z2[0]), x=z2)
                        vbc = vrel - vrel_av
                        vbc_g[iq, m] = vbc
                        v_g[iq, m] = vbt[m] + vbc
                        v_g_dir[:, m] = shear_dir[:, m]

                        vrel_e = cumtrapz(0.001 * shear_east[iq, m], x=z2, initial=0)
                        vrel_av_e = np.trapz(vrel_e / (z2[-1] - z2[0]), x=z2)
                        vbc_e = vrel_e - vrel_av_e
                        vbc_g_e[iq, m] = vbc_e
                        v_g_east[iq, m] = DACe_MW[m] + vbc_e

                        vrel_n = cumtrapz(0.001 * shear_north[iq, m], x=z2, initial=0)
                        vrel_av_n = np.trapz(vrel_n / (z2[-1] - z2[0]), x=z2)
                        vbc_n = vrel_n - vrel_av_n
                        vbc_g_n[iq, m] = vbc_n
                        v_g_north[iq, m] = DACn_MW[m] + vbc_n
                    else:
                        vbc_g[iq, m] = np.nan
                        v_g[iq, m] = np.nan

                # M/W position in lat/lon (to locate a velocity and eta profile in space)
                mwe_lon = np.nan * np.zeros(len(vbt))
                mwe_lat = np.nan * np.zeros(len(vbt))
                for m in order_set:
                    if m < order_set[-1]:
                        mwe_lon[m] = lon[np.where(np.isfinite(lon[:, m]))[0], m][-1]
                        mwe_lat[m] = lat[np.where(np.isfinite(lon[:, m]))[0], m][-1]
                        mwe_lon[m + 1] = lon[0, m + 1]
                        mwe_lat[m + 1] = lat[0, m + 1]
                    else:
                        mwe_lon[m] = lon[np.where(np.isfinite(lon[:, m]))[0], m][-1]
                        mwe_lat[m] = lat[np.where(np.isfinite(lat[:, m]))[0], m][-1]

                print('profile # = ' + str(info[0,:]))

                ds_out.append(ds)                   # distance of each m/w profile to transect start position
                dist_out.append(dist)               # distance of every data point to transect start position
                v_g_out.append(v_g)                 # velocity profile at distance ds
                v_g_east_out.append(v_g_east)       # east velocity profile at distance ds
                v_g_north_out.append(v_g_north)     # north velocity profile at distance ds
                vbt_out.append(vbt)                 #
                shear_out.append(shear)
                v_g_dir_out.append(v_g_dir)
                isopycdep_out.append(isopycdep)     # isopycnal depth (along each profile)
                isopycx_out.append(isopycx)         # isopycnal distance (along-transect distance) of isopycnal depth
                mwe_lon_out.append(mwe_lon)         # lon of m/w profile
                mwe_lat_out.append(mwe_lat)         # lat of m/w profile
                avg_ct_out.append(avg_ct_pd)        # avg conservative t profile following m/w scheme, at distance ds
                avg_sa_out.append(avg_sa_pd)        # avg abs sal profile following m/w scheme, at distance ds
                avg_sig_out.append(avg_sig_pd)      # avg sig profile following m/w scheme, at distance ds
                DACe_MW_out.append(DACe_MW)         # DAC (east) at each m/w profile
                DACn_MW_out.append(DACn_MW)         # DAC (west) at each m/w profile
                profile_tags_out.append(profile_tags)   # profile numbers (linked to dist) (make-up of each transect)
                info_out.append(info)                   #

            # everywhere there is a len(profile_tags) there was a self.num_profs

        return ds_out, dist_out, avg_ct_out, avg_sa_out, avg_sig_out, v_g_out, vbt_out, isopycdep_out, isopycx_out, \
            mwe_lon_out, mwe_lat_out, DACe_MW_out, DACn_MW_out, profile_tags_out, shear_out, v_g_dir_out, \
               v_g_east_out, v_g_north_out,

    def plot_cross_section(self, bin_depth, ds, v_g, dist, profile_tags, isopycdep, isopycx, sigth_levels, time, levels):
            sns.set(context="notebook", style="whitegrid", rc={"axes.axisbelow": False})
            fig0, ax0 = plt.subplots()
            matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
            # levels = np.arange(np.float(np.nanmin(v_g)), np.float(np.nanmax(v_g)), .02)
            # levels = np.arange(-.64, .66, .04)
            vc = ax0.contourf(ds, bin_depth, v_g, levels=levels, cmap=plt.cm.PuOr)
            vcc = ax0.contour(ds, bin_depth, v_g, levels=levels, colors='k', linewidth=.75)
            ax0.contour(ds, bin_depth, v_g, levels=[0], colors='k', linewidth=1.25)
            for p in range(len(profile_tags)):  # self.num_profs
                ax0.scatter(dist[:, p], bin_depth, s=.75, color='k')
            dive_label = np.arange(0, len(profile_tags), 2)
            for pp in range(np.size(dive_label)):
                p = dive_label[pp]
                ax0.text(np.nanmax(dist[:, p]) - 1, np.max(bin_depth[~np.isnan(dist[:, p])]) + 200,
                         str(int(profile_tags[p])))
            sig_good = np.where(~np.isnan(isopycdep[:, 0]))
            for p in range(np.size(sig_good[0])):
                ax0.plot(isopycx[sig_good[0][p], :], isopycdep[sig_good[0][p], :], color='#708090', linewidth=.75)
                ax0.text(np.nanmax(isopycx[sig_good[0][p], :]) + 2, np.nanmean(isopycdep[sig_good[0][p], :]),
                         str(sigth_levels[sig_good[0][p]]), fontsize=7)
            ax0.clabel(vcc, inline=1, fontsize=8, fmt='%1.2f', color='k')
            ax0.axis([0, np.max(ds) + 4, 0, np.max(bin_depth) + 200])
            ax0.invert_yaxis()
            ax0.set_xlabel('Distance along transect [km]')
            ax0.set_ylabel('Depth [m]')
            t_s = datetime.date.fromordinal(np.int(time[0, 0]))
            t_e = datetime.date.fromordinal(np.int(time[0, -1]))
            ax0.set_title(self.ID + ' ' + self.project + '  ' + np.str(
                t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day), fontsize=14)
            plt.colorbar(vc, label='[m/s]')
            plt.tight_layout()
            plot_pro(ax0)
            return fig0

    def plot_t_s(self, ct, sa):
        sa_gr = np.arange(32, 38, 0.1)
        ct_gr = np.arange(1, 30, 0.5)
        X, Y = np.meshgrid(sa_gr, ct_gr)
        sig0_gr = gsw.sigma0(X, Y)
        levs = np.arange(np.round(np.min(sig0_gr), 1), np.max(sig0_gr), 0.2)
        f, ax = plt.subplots()
        dc = ax.contour(sa_gr, ct_gr, sig0_gr, levs, colors='k', linewidth=0.25)
        ax.clabel(dc, inline=1, fontsize=7, fmt='%1.2f', color='k')
        for i in range(self.num_profs):
            ax.plot(sa[:, i], ct[:, i])
        ax.axis([np.round(np.nanmin(sa), 1) - .2, np.round(np.nanmax(sa), 1) + .2,
                 np.round(np.nanmin(ct), 1) - .5, np.round(np.nanmax(ct), 1) + .5])
        ax.set_xlabel('Absolute Salinity')
        ax.set_ylabel('Conservative Temperature')
        ax.grid()
        plot_pro(ax)

    def plot_plan_view(self, lon, lat, mwe_lon, mwe_lat, dac_u, dac_v, ref_lat, profile_tags, time, limits, path):
        # bathymetry
        bath = path
        bath_fid = Dataset(bath, 'r')
        if 'longitude' in bath_fid.variables:
            if np.nanmax(bath_fid.variables['longitude'][:]) > 0:
                bath_lon = bath_fid.variables['longitude'][:] - 360
            else:
                bath_lon = bath_fid.variables['longitude'][:]
            bath_lat = bath_fid.variables['latitude'][:]
        elif 'lon' in bath_fid.variables:
            bath_lon = bath_fid.variables['lon'][:]
            bath_lat = bath_fid.variables['lat'][:]
        if 'ROSE' in bath_fid.variables:
            bath_z = bath_fid.variables['ROSE'][:]
        if 'z' in bath_fid.variables:
            bath_z = bath_fid.variables['z'][:]
        elif 'elevation' in bath_fid.variables:
            bath_z = bath_fid.variables['elevation'][:]
        levels = [-5000, -4500, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 5]
        cmap = plt.cm.get_cmap("Blues_r")
        cmap.set_over('#808000')  # ('#E6E6E6')

        fig, ax = plt.subplots()
        bc = ax.contourf(bath_lon, bath_lat, bath_z, levels, cmap='Blues_r', extend='both', zorder=0)
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        # ax.plot(mwe_lon, mwe_lat, color='k')
        ax.scatter(lon, lat, s=2, color='#FF8C00')
        ax.scatter(mwe_lon, mwe_lat, s=10, color='#8B0000')
        for i in range(0, len(mwe_lat)-1, 2):
            ax.text(mwe_lon[i] - 0.1, mwe_lat[i] + 0.02, str(np.int(profile_tags[i])), color='#00FA9A',
                    fontsize=10, fontweight='bold')
        ax.quiver(mwe_lon, mwe_lat, dac_u, dac_v, color='r', scale=1.5, headwidth=2, headlength=2, width=.0025)
        w = 1 / np.cos(np.deg2rad(ref_lat))
        ax.axis(limits)
        ax.set_aspect(w)
        t_s = datetime.date.fromordinal(np.int(time[0, 0]))
        t_e = datetime.date.fromordinal(np.int(time[0, -1]))
        ax.set_title(self.ID + '  ' + self.project + '  ' + np.str(
            t_s.month) + '/' + np.str(t_s.day) + ' - ' + np.str(t_e.month) + '/' + np.str(t_e.day), fontsize=14)
        ax.set_xlabel('Lon')
        ax.set_ylabel('Lat')
        ax.grid()
        plot_pro(ax)
        return fig

    def transect_cross_section_0(self, bin_depth, sig0, lon, lat, dac_u, dac_v, profile_tags, sigth_levels):
        deep_shr_max = 0.1
        deep_shr_max_dep = 3500

        order_set = np.arange(0, self.num_profs, 2)

        info = np.nan * np.zeros((3, len(profile_tags) - 1))
        sigma_theta_out = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
        shear = np.nan * np.zeros((np.size(bin_depth), len(profile_tags) - 1))
        # eta = np.nan * np.zeros((np.size(grid), np.size(this_set) - 1))
        # eta_theta = np.nan * np.zeros((np.size(grid), np.size(this_set) - 1))
        isopycdep = np.nan * np.zeros((np.size(sigth_levels), len(profile_tags)))
        isopycx = np.nan * np.zeros((np.size(sigth_levels), len(profile_tags)))
        vbt = np.nan * np.zeros(len(profile_tags))
        DACe_MW = np.nan * np.zeros(len(profile_tags))
        DACn_MW = np.nan * np.zeros(len(profile_tags))
        ds = np.nan * np.zeros(len(profile_tags))
        dist = np.nan * np.zeros(np.shape(lon))
        dist_st = 0
        distance = 0
        for i in order_set:
            # M
            lon_start = lon[0, i]  # start position of dive i along the transect
            lat_start = lat[0, i]
            lon_finish = lon[0, i + 1]  # end position of dive i along the transect
            lat_finish = lat[0, i + 1]
            lat_ref = 0.5 * (lat_start + lat_finish)
            f_m = np.pi * np.sin(np.deg2rad(lat_ref)) / (12 * 1800)  # Coriolis parameter [s^-1]
            dxs_m = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (lon_finish - lon_start)  # zonal sfc disp [km]
            dys_m = 1.852 * 60 * (lat_finish - lat_start)  # meridional sfc disp [km]
            ds_a, ang_sfc_m = cart2pol(dxs_m, dys_m)

            dx = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * \
                (np.concatenate([lon[:, i], np.flipud(lon[:, i + 1])]) - lon[0, i])
            dy = 1.852 * 60 * (np.concatenate([lat[:, i], np.flipud(np.array(lat[:, i + 1]))]) - lat[0, i])
            ss, ang = cart2pol(dx, dy)
            xx, yy = pol2cart(ss, ang - ang_sfc_m)
            length1 = np.size(lon[:, i])
            dist[:, i] = dist_st + xx[0:length1]
            dist[:, i + 1] = dist_st + np.flipud(xx[length1:])
            dist_st = dist_st + ds_a

            distance = distance + np.nanmedian(xx)  # 0.5*ds # distance for each velocity estimate
            ds[i] = distance
            # DACe = dac_u[i]  # zonal depth averaged current [m/s]
            # DACn = dac_v[i]  # meridional depth averaged current [m/s]
            if i < 2:
                DACe = np.nanmean([[dac_u[i]], [dac_u[i + 2]]])
                DACn = np.nanmean([[dac_v[i]], [dac_v[i + 2]]])
            elif (i >= 2) and (i < len(profile_tags) - 2):
                DACe = np.nanmean([[dac_u[i - 1]], [dac_u[i]], [dac_u[i + 2]]])
                DACn = np.nanmean([[dac_v[i - 1]], [dac_v[i]], [dac_v[i + 2]]])
            elif i >= len(profile_tags) - 2:
                DACe = np.nanmean([[dac_u[i - 1]], [dac_u[i]]])
                DACn = np.nanmean([[dac_v[i - 1]], [dac_v[i]]])
            mag_DAC, ang_DAC = cart2pol(DACe, DACn)
            DACat, DACpot = pol2cart(mag_DAC, ang_DAC - ang_sfc_m)
            vbt[i] = DACpot  # across-track barotropic current comp (>0 to left)
            DACe_MW[i] = DACe.copy()
            DACn_MW[i] = DACn.copy()

            # W
            if i >= order_set[-1]:
                lon_finish = lon[0, -1]
                lat_finish = lon[0, -1]
                DACe = np.nanmean([[dac_u[i]], [dac_u[-1]]])  # zonal depth averaged current [m/s]
                DACn = np.nanmean([[dac_v[i]], [dac_v[-1]]])  # meridional depth averaged current [m/s]
            else:
                lon_finish = lon[0, i + 3]
                lat_finish = lat[0, i + 3]
                DACe = np.nanmean([[dac_u[i]], [dac_u[i + 2]]])  # zonal depth averaged current [m/s]
                DACn = np.nanmean([[dac_v[i]], [dac_v[i + 2]]])  # meridional depth averaged current [m/s]
            DACe_MW[i + 1] = DACe.copy()
            DACn_MW[i + 1] = DACn.copy()
            lat_ref = 0.5 * (lat_start + lat_finish)
            f_w = np.pi * np.sin(np.deg2rad(lat_ref)) / (12 * 1800)  # Coriolis parameter [s^-1]
            dxs = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (lon_finish - lon_start)  # zonal sfc disp [km]
            dys = 1.852 * 60 * (lat_finish - lat_start)  # meridional sfc disp [km]
            ds_w, ang_sfc_w = cart2pol(dxs, dys)
            distance = distance + (ds_a - np.nanmedian(xx))  # distance for each velocity estimate
            ds[i + 1] = distance
            mag_DAC, ang_DAC = cart2pol(DACe, DACn)
            DACat, DACpot = pol2cart(mag_DAC, ang_DAC - ang_sfc_w)
            vbt[i + 1] = DACpot  # across-track barotropic current comp (>0 to left)

            shearM = np.nan * np.zeros(np.size(bin_depth))
            shear_x_M = np.nan * np.zeros(np.size(bin_depth))
            shear_y_M = np.nan * np.zeros(np.size(bin_depth))
            shearW = np.nan * np.zeros(np.size(bin_depth))
            etaM = np.nan * np.zeros(np.size(bin_depth))
            etaW = np.nan * np.zeros(np.size(bin_depth))
            eta_thetaM = np.nan * np.zeros(np.size(bin_depth))
            eta_thetaW = np.nan * np.zeros(np.size(bin_depth))
            sigma_theta_pa_M = np.nan * np.zeros(np.size(bin_depth))
            sigma_theta_pa_W = np.nan * np.zeros(np.size(bin_depth))
            lon_pa_M = np.nan * np.zeros(np.size(bin_depth))
            lon_pa_W = np.nan * np.zeros(np.size(bin_depth))
            lat_pa_M = np.nan * np.zeros(np.size(bin_depth))
            lat_pa_W = np.nan * np.zeros(np.size(bin_depth))
            # LOOP OVER EACH BIN_DEPTH
            for j in range(np.size(bin_depth)):
                # find array of indices for M / W sampling
                if i < 2:
                    c_i_m = np.arange(i, i + 3)
                    # c_i_m = []  # omit partial "M" estimate
                    c_i_w = np.arange(i, i + 4)
                elif (i >= 2) and (i < len(profile_tags) - 2):
                    c_i_m = np.arange(i - 1, i + 3)
                    c_i_w = np.arange(i, i + 4)
                elif i >= len(profile_tags) - 2:
                    c_i_m = np.arange(i - 1, len(profile_tags))
                    # c_i_m = []  # omit partial "M" estimated
                    c_i_w = []
                nm = np.size(c_i_m)
                nw = np.size(c_i_w)

                # for M profile compute shear and eta
                if nm > 2 and np.size(sig0[j, c_i_m]) > 2:
                    sigmathetaM = sig0[j, c_i_m]
                    sigma_theta_pa_M[j] = np.nanmean(sigmathetaM)  # average density across 4 profiles
                    lon_pa_M[j] = np.nanmean(lon[j, c_i_m])  # avg lat/lon across M/W profiles
                    lat_pa_M[j] = np.nanmean(lat[j, c_i_m])
                    # thetaM = theta[j, c_i_m]
                    # ctM = df_ct_set.iloc[j, c_i_m]
                    imv = ~np.isnan(np.array(sig0[j, c_i_m]))
                    c_i_m_in = c_i_m[imv]

                    if np.size(c_i_m_in) > 1:
                        xM = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (
                                lon[j, c_i_m_in] - lon[j, c_i_m_in[0]])  # E loc [km]
                        yM = 1.852 * 60 * (
                                lat[j, c_i_m_in] - lat[j, c_i_m_in[0]])  # N location [km]
                        XXM = np.concatenate(
                            [np.ones((np.size(sigmathetaM[imv]), 1)), np.transpose(np.atleast_2d(np.array(xM))),
                            np.transpose(np.atleast_2d(np.array(yM)))], axis=1)
                        d_anom0M = sigmathetaM[imv] - np.nanmean(sigmathetaM[imv])
                        ADM = np.squeeze(np.linalg.lstsq(XXM, np.transpose(np.atleast_2d(np.array(d_anom0M))))[0])
                        drhodxM = ADM[1]  # [zonal gradient [kg/m^3/km]
                        drhodyM = ADM[2]  # [meridional gradient [kg/m^3km]
                        drhodsM, ang_drhoM = cart2pol(drhodxM, drhodyM)
                        drhodatM, drhodpotM = pol2cart(drhodsM, ang_drhoM - ang_sfc_m)
                        shearM[j] = -self.g * drhodatM / (self.rho0 * f_m)  # shear to port of track [m/s/km]
                        if (np.abs(shearM[j]) > deep_shr_max) and bin_depth[j] >= deep_shr_max_dep:
                            shearM[j] = np.sign(shearM[j]) * deep_shr_max
                        angle_of_shear = ang_sfc_m + np.pi / 2  # angle in transect direction + 90deg
                        shear_x_M[j], shear_y_M[j] = pol2cart(shearM[j], angle_of_shear)  # project onto E/N

                        # --- Computation of Eta
                        # -- isopycnal position is average position of a few dives
                        # etaM[j] = (sigma_theta_avg[j] - np.nanmean(sigmathetaM[imv])) / ddz_avg_sigma[j]
                        # eta_thetaM[j] = (theta_avg[j] - np.nanmean(thetaM[imv])) / ddz_avg_theta[j]
                        # -- isopycnal position is position on this single profile
                        # etaM[j] = (sigma_theta_avg[j] - sig0[j, i]) / ddz_avg_sigma[j]  # j=dp, i=prof_ind
                        # eta_thetaM[j] = (ct_avg[j] - df_ct_set.iloc[j, i]) / ddz_avg_ct[j]

                        # testing of across-track angle computations
                        # if j == 180:
                        #     print('dive cycles in this m estimation = ' +
                        #           str(profile_tags[c_i_m[0]]) + ' - ' + str(profile_tags[c_i_m[-1]]))
                        #     print('x_grad=' + str(drhodxM))
                        #     print('y_grad=' + str(drhodyM))
                        #     print('along_grad=' + str(drhodatM))
                        #     f, ax = plt.subplots()
                        #     # position from start of contributing m/w profiles
                        #     x_ds = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (
                        #             lon[:, c_i_m_in] - lon[0, c_i_m_in[0]])
                        #     print(np.shape(x_ds))
                        #     y_ds = 1.852 * 60 * (lat[:, c_i_m_in] - lat[0, c_i_m_in[0]])  # N
                        #     xMj = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (
                        #             lon[j, c_i_m_in] - lon[0, c_i_m_in[0]])  # E
                        #     yMj = 1.852 * 60 * (lat[j, c_i_m_in] - lat[0, c_i_m_in[0]])  # N
                        #     ax.scatter(x_ds, y_ds, 4, color='k', label='glider path')
                        #     ax.scatter(x_ds[15, :], y_ds[15, :], color='k', s=50)
                        #     ax.scatter(xMj, yMj, color='r', s=120, label='loc. of density obs. at 2000m')
                        #     ax.plot(xMj, yMj, color='r')
                        #     # location of this velocity profile
                        #     if i < order_set[-1]:
                        #         me_lon = lon[np.where(np.isfinite(lon[:, i]))[0], i][-1]
                        #         me_lat = lat[np.where(np.isfinite(lon[:, i]))[0], i][-1]
                        #         # mwe_lon[m + 1] = lon[0, m + 1]
                        #         # mwe_lat[m + 1] = lat[0, m + 1]
                        #     else:
                        #         me_lon = lon[np.where(np.isfinite(lon[:, i]))[0], i][-1]
                        #         me_lat = lat[np.where(np.isfinite(lat[:, i]))[0], i][-1]
                        #     # plot dive number associated with location of v profile
                        #     x_pr = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (me_lon - lon[0, c_i_m_in[0]])
                        #     y_pr = 1.852 * 60 * (me_lat - lat[0, c_i_m_in[0]])
                        #     ax.text(x_pr, y_pr + 1, profile_tags[i])
                        #     # vectors
                        #     testx, testy = pol2cart(1, ang_sfc_m)
                        #     ax.quiver(x_pr, y_pr, testx, testy, color='m', scale=4,
                        #               label='along-track dir.')  # dives dir
                        #     ax.quiver(x_pr, y_pr, drhodxM, drhodyM, color='g', scale=.01,
                        #               label='den. grad. dir.')  # den gradient dir
                        #     # shear to port
                        #     testshx, testshy = pol2cart(shearM[j], ang_sfc_m + np.pi / 2)
                        #     ax.quiver(x_pr, y_pr, testshx, testshy, color='y', scale=.1,
                        #               label='shear to port')  # shear to port dir
                        #     # cross-track shear projected into E
                        #     ax.quiver(x_pr, y_pr, shear_x_M[j], 0, color='b', scale=.1, label='shear E')
                        #     # cross-track shear projected into N
                        #     ax.quiver(x_pr, y_pr, 0, shear_y_M[j], color='b', scale=.1, label='shear N')
                        #     ax.set_title('M/W Schematic at 2000m, Transect Component = ' + str(i))
                        #     ax.axis(
                        #         [np.nanmin(x_ds) - 3, np.nanmax(x_ds) + 3, np.nanmin(y_ds) - 3, np.nanmax(y_ds) + 3])
                        #     ax.set_aspect('equal')
                        #     ax.set_xlabel('km East')
                        #     ax.set_ylabel('km North')
                        #     handles, labels = ax.get_legend_handles_labels()
                        #     ax.legend(handles, labels, fontsize=10)
                        #     plot_pro(ax)

                # for W profile compute shear and eta
                if nw > 2 and np.size(sig0[j, c_i_w]) > 2:
                    sigmathetaW = sig0[j, c_i_w]
                    sigma_theta_pa_W[j] = np.nanmean(sigmathetaW)  # average density across 4 profiles
                    lon_pa_W[j] = np.nanmean(lon[j, c_i_w])  # avg lat/lon across M/W profiles
                    lat_pa_W[j] = np.nanmean(lon[j, c_i_w])
                    # thetaW = df_theta_set.iloc[j, c_i_w]
                    # ctW = df_theta_set.iloc[j, c_i_w]
                    iwv = ~np.isnan(np.array(sig0[j, c_i_w]))
                    c_i_w_in = c_i_w[iwv]

                    if np.sum(c_i_w_in) > 1:
                        xW = 1.852 * 60 * np.cos(np.deg2rad(lat_ref)) * (
                                lon[j, c_i_w_in] - lon[j, c_i_w_in[0]])  # E loc [km]
                        yW = 1.852 * 60 * (
                                lat[j, c_i_w_in] - lat[j, c_i_w_in[0]])  # N location [km]
                        XXW = np.concatenate(
                            [np.ones((np.size(sigmathetaW[iwv]), 1)), np.transpose(np.atleast_2d(np.array(xW))),
                                np.transpose(np.atleast_2d(np.array(yW)))], axis=1)
                        d_anom0W = sigmathetaW[iwv] - np.nanmean(sigmathetaW[iwv])
                        ADW = np.squeeze(np.linalg.lstsq(XXW, np.transpose(np.atleast_2d(np.array(d_anom0W))))[0])
                        drhodxW = ADW[1]  # [zonal gradient [kg/m^3/km]
                        drhodyW = ADW[2]  # [meridional gradient [kg/m^3km]
                        drhodsW, ang_drhoW = cart2pol(drhodxW, drhodyW)
                        drhodatW, drhodpotW = pol2cart(drhodsW, ang_drhoW - ang_sfc_w)
                        shearW[j] = -self.g * drhodatW / (self.rho0 * f_w)  # shear to port of track [m/s/km]
                        if (np.abs(shearW[j]) > deep_shr_max) and bin_depth[j] >= deep_shr_max_dep:
                            shearW[j] = np.sign(shearW[j]) * deep_shr_max

                        # --- Computation of Eta
                        # -- isopycnal position is average position of a few dives
                        # etaW[j] = (sigma_theta_avg[j] - np.nanmean(sigmathetaW[iwv])) / ddz_avg_sigma[j]
                        # eta_thetaW[j] = (theta_avg[j] - np.nanmean(thetaW[iwv])) / ddz_avg_theta[j]
                        # -- isopycnal position is position on this single profile
                        # etaW[j] = (sigma_theta_avg[j] - df_den_set.iloc[j, i + 1]) / ddz_avg_sigma[j]
                        # eta_thetaW[j] = (ct_avg[j] - df_ct_set.iloc[j, i + 1]) / ddz_avg_ct[j]
                # END LOOP OVER EACH BIN_DEPTH

            # OUTPUT FOR EACH TRANSECT (at least 2 DIVES)
            # because this is M/W profiling, for a 3 dive transect, only 5 profiles of shear and eta are compiled
            sigma_theta_out[:, i] = sigma_theta_pa_M
            shear[:, i] = shearM
            # eta[:, i] = etaM
            # eta_theta[:, i] = eta_thetaM
            info[0, i] = profile_tags[i]
            info[1, i] = np.nanmean(lon_pa_M)
            info[2, i] = np.nanmean(lat_pa_M)
            if i < len(profile_tags) - 2:
                sigma_theta_out[:, i + 1] = sigma_theta_pa_W
                shear[:, i + 1] = shearW
                # eta[:, i + 1] = etaW
                # eta_theta[:, i + 1] = eta_thetaW
                info[0, i + 1] = profile_tags[i]
                info[1, i + 1] = np.nanmean(lon_pa_W)
                info[2, i + 1] = np.nanmean(lat_pa_W)

            # ISOPYCNAL DEPTHS ON PROFILES ALONG EACH TRANSECT
            sigthmin = np.nanmin(np.array(sig0[:, i]))
            sigthmax = np.nanmax(np.array(sig0[:, i]))
            isigth = np.where((sigth_levels > sigthmin) & (sigth_levels < sigthmax))
            isopycdep[isigth, i] = np.interp(sigth_levels[isigth], sig0[:, i], bin_depth)
            isopycx[isigth, i] = np.interp(sigth_levels[isigth], sig0[:, i], dist[:, i])

            sigthmin = np.nanmin(np.array(sig0[:, i + 1]))
            sigthmax = np.nanmax(np.array(sig0[:, i + 1]))
            isigth = np.where((sigth_levels > sigthmin) & (sigth_levels < sigthmax))
            isopycdep[isigth, i + 1] = np.interp(sigth_levels[isigth], sig0[:, i + 1], bin_depth)
            isopycx[isigth, i + 1] = np.interp(sigth_levels[isigth], sig0[:, i + 1], dist[:, i + 1])

            # ---- END LOOP OVER EACH DIVE IN TRANSECT

        # FOR EACH TRANSECT COMPUTE GEOSTROPHIC VELOCITY
        vbc_g = np.nan * np.zeros(np.shape(shear))
        v_g = np.nan * np.zeros((np.size(bin_depth), len(profile_tags)))
        for m in range(len(profile_tags) - 1):
            iq = np.where(~np.isnan(shear[:, m]))
            if np.size(iq) > 10:
                z2 = -bin_depth[iq]
                vrel = cumtrapz(0.001 * shear[iq, m], x=z2, initial=0)
                vrel_av = np.trapz(vrel / (z2[-1] - z2[0]), x=z2)
                vbc = vrel - vrel_av
                vbc_g[iq, m] = vbc
                v_g[iq, m] = vbt[m] + vbc
            else:
                vbc_g[iq, m] = np.nan
                v_g[iq, m] = np.nan

        # M/W position in lat/lon (to locate a velocity and eta profile in space)
        mwe_lon = np.nan * np.zeros(len(vbt))
        mwe_lat = np.nan * np.zeros(len(vbt))
        for m in order_set:
            if m < order_set[-1]:
                mwe_lon[m] = lon[np.where(np.isfinite(lon[:, m]))[0], m][-1]
                mwe_lat[m] = lat[np.where(np.isfinite(lon[:, m]))[0], m][-1]
                mwe_lon[m + 1] = lon[0, m + 1]
                mwe_lat[m + 1] = lat[0, m + 1]
            else:
                mwe_lon[m] = lon[np.where(np.isfinite(lon[:, m]))[0], m][-1]
                mwe_lat[m] = lat[np.where(np.isfinite(lat[:, m]))[0], m][-1]

        return ds, dist, v_g, vbt, isopycdep, isopycx, mwe_lon, mwe_lat, DACe_MW, DACn_MW, profile_tags

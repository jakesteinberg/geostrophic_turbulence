import numpy as np
from netCDF4 import Dataset
import seawater as sw
from toolkit import find_nearest
import gsw


def make_bin_gen(bin_depth, depth_d, temp_d, salin_d):
    # max depth attained
    dep_max = np.round(depth_d.max())
    deepest_bin = find_nearest(bin_depth, dep_max)[0]

    if deepest_bin == (len(bin_depth) - 1):
        bin_up = bin_depth[0:(deepest_bin - 1)]
        bin_down = bin_depth[2:(deepest_bin + 1)]
        bin_cen = bin_depth[1:deepest_bin]

    temp_g = -999*np.ones(np.size(bin_depth))
    salin_g = -999*np.ones(np.size(bin_depth))
    # -- Case z = 0
    dp_in_d_1 = depth_d < bin_cen[0]
    if np.sum(dp_in_d_1) >= 2:
        temp_g[0] = np.nanmean(temp_d[dp_in_d_1])
        salin_g[0] = np.nanmean(salin_d[dp_in_d_1])
    # -- Case z > 0
    # bin_up = bin_depth[0:-2]
    # bin_down = bin_depth[2:]
    # bin_cen = bin_depth[1:-1]
    # temp_g_dive = np.empty(np.size(bin_cen)+1)
    # salin_g_dive = np.empty(np.size(bin_cen)+1)
    # temp_g_dive[0] = np.nanmean(temp_d[depth_d < bin_cen[0]])
    # salin_g_dive[0] = np.nanmean(salin_d[depth_d < bin_cen[0]])
    for j in range(np.size(bin_cen)):
        i = j + 1
        dp_in_d = (depth_d > bin_up[j]) & (depth_d < bin_down[j])
        if dp_in_d.size > 2:
            temp_g[i] = np.nanmean(temp_d[dp_in_d])
            salin_g[i] = np.nanmean(salin_d[dp_in_d])
    # -- Case last_bin
    dp_in_d_e = (depth_d > bin_cen[-1]) & (depth_d < bin_cen[-1] + 75)
    if dp_in_d.size > 2:
        temp_g[-1] = np.nanmean(temp_d[dp_in_d_e])
        salin_g[-1] = np.nanmean(salin_d[dp_in_d_e])

    return temp_g, salin_g


def make_bin(bin_depth, depth_d, depth_c, temp_d, temp_c, salin_d, salin_c, x_g_d, x_g_c, y_g_d, y_g_c):
    # max depth attained
    dep_max = np.round(np.max([depth_d.max(), depth_c.max()]))
    deepest_bin = find_nearest(bin_depth, dep_max)[0]

    bin_up = bin_depth[0:(deepest_bin - 1)]
    bin_down = bin_depth[2:(deepest_bin + 1)]
    bin_cen = bin_depth[1:deepest_bin]
    temp_g_dive = -999*np.ones(np.size(bin_depth))
    temp_g_climb = -999*np.ones(np.size(bin_depth))
    salin_g_dive = -999*np.ones(np.size(bin_depth))
    salin_g_climb = -999*np.ones(np.size(bin_depth))
    x_g_dive = -999*np.ones(np.size(bin_depth))
    x_g_climb = -999*np.ones(np.size(bin_depth))
    y_g_dive = -999*np.ones(np.size(bin_depth))
    y_g_climb = -999*np.ones(np.size(bin_depth))
    # -- Case z = 0
    dp_in_d_1 = depth_d < bin_cen[0]
    dp_in_c_1 = depth_c < bin_cen[0]
    if np.sum(dp_in_d_1) >= 2:
        temp_g_dive[0] = np.nanmean(temp_d[dp_in_d_1])
        salin_g_dive[0] = np.nanmean(salin_d[dp_in_d_1])
        x_g_dive[0] = np.nanmean(x_g_d[dp_in_d_1]) / 1000
        y_g_dive[0] = np.nanmean(y_g_d[dp_in_d_1]) / 1000
    if np.sum(dp_in_c_1) >= 2:
        temp_g_climb[0] = np.nanmean(temp_c[dp_in_c_1])
        salin_g_climb[0] = np.nanmean(salin_c[dp_in_c_1])
        x_g_climb[0] = np.nanmean(x_g_c[dp_in_c_1]) / 1000
        y_g_climb[0] = np.nanmean(y_g_c[dp_in_c_1]) / 1000
    # -- Case z > 0
    for j in range(np.size(bin_cen)):
        i = j + 1
        dp_in_d = (depth_d > bin_up[j]) & (depth_d < bin_down[j])
        dp_in_c = (depth_c > bin_up[j]) & (depth_c < bin_down[j])
        if np.sum(dp_in_d) >= 2:
            temp_g_dive[i] = np.nanmean(temp_d[dp_in_d])
            salin_g_dive[i] = np.nanmean(salin_d[dp_in_d])
            x_g_dive[i] = np.nanmean(x_g_d[dp_in_d]) / 1000
            y_g_dive[i] = np.nanmean(y_g_d[dp_in_d]) / 1000
        if np.sum(dp_in_c) >= 2:
            temp_g_climb[i] = np.nanmean(temp_c[dp_in_c])
            salin_g_climb[i] = np.nanmean(salin_c[dp_in_c])
            x_g_climb[i] = np.nanmean(x_g_c[dp_in_c]) / 1000
            y_g_climb[i] = np.nanmean(y_g_c[dp_in_c]) / 1000

    return temp_g_dive, temp_g_climb, salin_g_dive, salin_g_climb, x_g_dive, x_g_climb, y_g_dive, y_g_climb


def test_bin(a):
    return np.shape(a)


def collect_dives(dg_list, bin_depth, grid_p, ref_lat):
    # initial arrays
    heading_rec = []
    target_rec = []
    gps_rec = []
    time_rec = []
    time_rec_2 = []
    dac_u = []
    dac_v = []
    # time_rec_2 = np.zeros([np.size(dg_list), 2])

    secs_per_day = 86400.0
    datenum_start = 719163  # jan 1 1970

    # ----- loop over each dive
    count_st = 14
    count = count_st
    for i in dg_list[count_st:]:
        # dive_nc_file = netcdf.netcdf_file(i, 'r', mmap=False if sys.platform == 'darwin' else mmap, version=1)
        dive_nc_file = Dataset(i, 'r')
        # initial dive information 
        glid_num = dive_nc_file.glider
        dive_num = dive_nc_file.dive_number
        heading_ind = dive_nc_file.variables['log_MHEAD_RNG_PITCHd_Wd'][:]
        target_ind = dive_nc_file.variables['log_TGT_NAME'][:]
        GPS1_ind = dive_nc_file.variables['log_GPS1'][:]
        GPS2_ind = dive_nc_file.variables['log_GPS2'][:]

        # --------- extract target
        t_it = target_ind.shape[0]
        ii = []
        for j in range(t_it):
            ii = str(target_ind[j])
            if j < 1:
                targ = ii[2]
            else:
                targ = targ + ii[2]

        if targ[-1].isdigit():
            if count < count_st + 1:
                target_rec = np.concatenate([[targ[2:4]], [targ[2:4]]])
            else:
                target_rec = np.concatenate([target_rec, [targ[2:4]], [targ[2:4]]])
        else:
            if count < count_st + 1:
                target_rec = np.concatenate([[np.nan], [np.nan]])
            else:
                target_rec = np.concatenate([target_rec, [np.nan], [np.nan]])

        # --------- GPS 1,2
        GPS1 = '$$'
        for l in range(np.size(GPS1_ind)):
            GPS1 = GPS1 + (str(GPS1_ind[l])[2])
        GPS1_1 = GPS1[6:]
        count_g = 0
        m = 0
        GPS1_out = np.nan * np.zeros(13)
        for l in range(len(GPS1_1)):
            if (GPS1_1[l].isdigit()) | (GPS1_1[l] == '.') | (GPS1_1[l] == '-'):
                if m < 1:
                    numb = GPS1_1[l]
                else:
                    numb = numb + GPS1_1[l]
                m = m + 1
            else:
                m = 0
                GPS1_out[count_g] = numb
                count_g = count_g + 1
        GPS1_out[count_g] = numb

        GPS2 = '$$'
        for l in range(np.size(GPS2_ind)):
            GPS2 = GPS2 + (str(GPS2_ind[l])[2])
        GPS2_1 = GPS2[6:]
        count_g = 0
        m = 0
        GPS2_out = np.nan * np.zeros(13)
        for l in range(len(GPS2_1)):
            if (GPS2_1[l].isdigit()) | (GPS2_1[l] == '.') | (GPS2_1[l] == '-'):
                if m < 1:
                    numb = GPS2_1[l]
                else:
                    numb = numb + GPS2_1[l]
                m = m + 1
            else:
                m = 0
                GPS2_out[count_g] = numb
                count_g = count_g + 1
        GPS2_out[count_g] = numb

        if count < count_st + 1:
            gps_rec = np.array([[GPS1_out[3], GPS1_out[4], GPS2_out[3], GPS2_out[4]],
                                [GPS1_out[3], GPS1_out[4], GPS2_out[3], GPS2_out[4]]])
        else:
            gps_rec = np.concatenate([gps_rec, np.array([[GPS1_out[3], GPS1_out[4], GPS2_out[3], GPS2_out[4]],
                                                         [GPS1_out[3], GPS1_out[4], GPS2_out[3], GPS2_out[4]]])])

        # ---------- extract heading
        h1 = heading_ind[0]
        h_test_0 = heading_ind[1]
        h_test_1 = heading_ind[2]
        if h_test_0.isdigit() == True:
            if h_test_1.isdigit() == True:
                h2 = h_test_0
                h3 = h_test_1
                h = int(h1 + h2 + h3)
            else:
                h = int(h1 + h2)
        else:
            h = int(h1)

        if np.sum(heading_rec) < 1:
            heading_rec = np.concatenate([[h], [h]])
        else:
            heading_rec = np.concatenate([heading_rec, [h], [h]])

        # dive position
        lat = dive_nc_file.variables['latitude'][:]
        lon = dive_nc_file.variables['longitude'][:]

        # eng     
        ctd_epoch_time = dive_nc_file.variables['ctd_time'][:]
        pitch_ang = dive_nc_file.variables['eng_pitchAng'][:]

        # science 
        press = dive_nc_file.variables['ctd_pressure'][:]
        temp = dive_nc_file.variables['temperature'][:]
        salin = dive_nc_file.variables['salinity'][:]
        # theta = sw.ptmp(salin, temp, press, 0)
        # depth = sw.dpth(press, ref_lat)
        press[press < 0] = 0
        depth = -1*gsw.z_from_p(press, ref_lat)

        # time conversion 
        dive_start_time = dive_nc_file.start_time
        ctd_time = ctd_epoch_time - dive_start_time

        # put on vertical grid (1 column per profile)
        max_d = np.where(depth == depth.max())
        max_d_ind = max_d[0]
        pitch_ang_sub1 = pitch_ang[0:max_d_ind[0]]
        pitch_ang_sub2 = pitch_ang[max_d_ind[0]:]
        dive_mask_i = np.where(pitch_ang_sub1 < 0)
        dive_mask = dive_mask_i[0][:]
        climb_mask_i = np.where(pitch_ang_sub2 > 0)
        climb_mask = climb_mask_i[0][:] + max_d_ind[0] - 1

        # dive/climb time midpoints
        time_dive = ctd_time[dive_mask]
        time_climb = ctd_time[climb_mask]

        serial_date_time_dive = datenum_start + dive_start_time / (60 * 60 * 24) + np.median(time_dive) / secs_per_day
        serial_date_time_climb = datenum_start + dive_start_time / (60 * 60 * 24) + np.median(time_climb) / secs_per_day
        serial_date_time_dive_2 = datenum_start + dive_start_time / (60 * 60 * 24) + np.nanmin(time_dive) / secs_per_day
        serial_date_time_climb_2 = datenum_start + dive_start_time / (60 * 60 * 24) + np.nanmax(time_climb) / secs_per_day
        # time_rec_2[count,:] = np.array([ serial_date_time_dive, serial_date_time_climb ])
        # time_rec_2[count,:] = np.array([ serial_date_time_dive_2, serial_date_time_climb_2 ])
        if np.sum(time_rec) < 1:
            time_rec = np.concatenate([[serial_date_time_dive], [serial_date_time_climb]])
            time_rec_2 = np.concatenate([[serial_date_time_dive_2], [serial_date_time_climb_2]])
            dac_u = np.concatenate([[dive_nc_file.variables['depth_avg_curr_east'][:]],
                                    [dive_nc_file.variables['depth_avg_curr_east'][:]]])
            dac_v = np.concatenate([[dive_nc_file.variables['depth_avg_curr_north'][:]],
                                    [dive_nc_file.variables['depth_avg_curr_north'][:]]])
        else:
            time_rec = np.concatenate([time_rec, [serial_date_time_dive], [serial_date_time_climb]])
            time_rec_2 = np.concatenate([time_rec_2, [serial_date_time_dive_2], [serial_date_time_climb_2]])
            dac_u = np.concatenate([dac_u, [dive_nc_file.variables['depth_avg_curr_east'][:]],
                                    [dive_nc_file.variables['depth_avg_curr_east'][:]]])
            dac_v = np.concatenate([dac_v, [dive_nc_file.variables['depth_avg_curr_north'][:]],
                                    [dive_nc_file.variables['depth_avg_curr_north'][:]]])

        # interpolate (bin_average) to smooth and place T/S on vertical depth grid
        temp_grid_dive, temp_grid_climb, salin_grid_dive, salin_grid_climb, lon_grid_dive, lon_grid_climb, lat_grid_dive, lat_grid_climb = make_bin(
            bin_depth, depth[dive_mask], depth[climb_mask], temp[dive_mask], temp[climb_mask], salin[dive_mask],
            salin[climb_mask], lon[dive_mask] * 1000, lon[climb_mask] * 1000, lat[dive_mask] * 1000,
                               lat[climb_mask] * 1000)

        # compute absolute salinity, conservative temperature, and potential density anomaly (reference p = 0)
        SA_grid_dive = gsw.SA_from_SP(salin_grid_dive, grid_p, lon_grid_dive, lat_grid_dive)
        SA_grid_climb = gsw.SA_from_SP(salin_grid_climb, grid_p, lon_grid_climb, lat_grid_climb)
        CT_grid_dive = gsw.CT_from_t(SA_grid_dive, temp_grid_dive, grid_p)
        CT_grid_climb = gsw.CT_from_t(SA_grid_climb, temp_grid_climb, grid_p)
        sig0_grid_dive = gsw.sigma0(SA_grid_dive, CT_grid_dive)
        sig0_grid_climb = gsw.sigma0(SA_grid_climb, CT_grid_climb)

        # den_grid_dive = sw.pden(salin_grid_dive, temp_grid_dive, grid_p, pr=0) - 1000
        # den_grid_climb = sw.pden(salin_grid_climb, temp_grid_climb, grid_p, pr=0) - 1000
        theta_grid_dive = sw.ptmp(salin_grid_dive, temp_grid_dive, grid_p, pr=0)
        theta_grid_climb = sw.ptmp(salin_grid_climb, temp_grid_climb, grid_p, pr=0)

        # create dataframes where each column is a profile 
        # t_data_d = pd.DataFrame(theta_grid_dive, index=grid, columns=[glid_num * 1000 + dive_num])
        # t_data_c = pd.DataFrame(theta_grid_climb, index=grid, columns=[glid_num * 1000 + dive_num + .5])
        # s_data_d = pd.DataFrame(salin_grid_dive, index=grid, columns=[glid_num * 1000 + dive_num])
        # s_data_c = pd.DataFrame(salin_grid_climb, index=grid, columns=[glid_num * 1000 + dive_num + .5])
        # den_data_d = pd.DataFrame(den_grid_dive, index=grid, columns=[glid_num * 1000 + dive_num])
        # den_data_c = pd.DataFrame(den_grid_climb, index=grid, columns=[glid_num * 1000 + dive_num + .5])
        # lon_data_d = pd.DataFrame(lon_grid_dive, index=grid, columns=[glid_num * 1000 + dive_num])
        # lon_data_c = pd.DataFrame(lon_grid_climb, index=grid, columns=[glid_num * 1000 + dive_num + .5])
        # lat_data_d = pd.DataFrame(lat_grid_dive, index=grid, columns=[glid_num * 1000 + dive_num])
        # lat_data_c = pd.DataFrame(lat_grid_climb, index=grid, columns=[glid_num * 1000 + dive_num + .5])

        if count == count_st:
            # df_t = pd.concat([t_data_d, t_data_c], axis=1)
            theta_out = np.concatenate([theta_grid_dive[:, None], theta_grid_climb[:, None]], axis=1)
            SA_out = np.concatenate([SA_grid_dive[:, None], SA_grid_climb[:, None]], axis=1)
            CT_out = np.concatenate([CT_grid_dive[:, None], CT_grid_climb[:, None]], axis=1)
            lon_out = np.concatenate([lon_grid_dive[:, None], lon_grid_climb[:, None]], axis=1)
            lat_out = np.concatenate([lat_grid_dive[:, None], lat_grid_climb[:, None]], axis=1)
            sig0_out = np.concatenate([sig0_grid_dive[:, None], sig0_grid_climb[:, None]], axis=1)
            d_l = np.array([glid_num * 1000 + dive_num, glid_num * 1000 + dive_num + .5])
        else:
            # df_t = pd.concat([df_t, t_data_d, t_data_c], axis=1)
            theta_out = np.concatenate([theta_out, theta_grid_dive[:, None], theta_grid_climb[:, None]], axis=1)
            SA_out = np.concatenate([SA_out, SA_grid_dive[:, None], SA_grid_climb[:, None]], axis=1)
            CT_out = np.concatenate([CT_out, CT_grid_dive[:, None], CT_grid_climb[:, None]], axis=1)
            lon_out = np.concatenate([lon_out, lon_grid_dive[:, None], lon_grid_climb[:, None]], axis=1)
            lat_out = np.concatenate([lat_out, lat_grid_dive[:, None], lat_grid_climb[:, None]], axis=1)
            sig0_out = np.concatenate([sig0_out, sig0_grid_dive[:, None], sig0_grid_climb[:, None]], axis=1)
            d_l = np.append(d_l, np.array([glid_num * 1000 + dive_num, glid_num * 1000 + dive_num + .5]))

        count = count + 1

    return theta_out, CT_out, SA_out, sig0_out, lon_out, lat_out, dac_u, dac_v, time_rec, time_rec_2, target_rec, gps_rec, d_l

# procedure for take BATS dives and processing each to account for heading and vertical bin averaging
# (uses collect_dives and make_bin)
# LOAD EACH DIVE AND COLLECT IN MAIN DATAFRAME # ----- GRIDDING! -- ONLY DO ONCE
# df_t, df_s, df_den, df_lon, df_lat, dac_u, dac_v, time_rec, time_sta_sto, heading_rec, target_rec =
#   collect_dives(dg_list, bin_depth, grid, grid_p, ref_lat)
# dive_list = np.array(df_t.columns)
# f = netcdf.netcdf_file('BATs_2015_gridded_2.nc', 'w')    
# f.history = 'DG 2015 dives; have been gridded vertically and separated into dive and climb cycles'
# f.createDimension('grid',np.size(grid))
# f.createDimension('lat_lon',4)
# f.createDimension('dive_list',np.size(dive_list))
# b_d = f.createVariable('grid',  np.float64, ('grid',) )
# b_d[:] = grid
# b_l = f.createVariable('dive_list',  np.float64, ('dive_list',) )
# b_l[:] = dive_list
# b_t = f.createVariable('Temperature',  np.float64, ('grid','dive_list') )
# b_t[:] = df_t
# b_s = f.createVariable('Salinity',  np.float64, ('grid','dive_list') )
# b_s[:] = df_s
# b_den = f.createVariable('Density',  np.float64, ('grid','dive_list') )
# b_den[:] = df_den
# b_lon = f.createVariable('Longitude',  np.float64, ('grid','dive_list') )
# b_lon[:] = df_lon
# b_lat = f.createVariable('Latitude',  np.float64, ('grid','dive_list') )
# b_lat[:] = df_lat
# b_u = f.createVariable('DAC_u',  np.float64, ('dive_list',) )
# b_u[:] = dac_u
# b_v = f.createVariable('DAC_v',  np.float64, ('dive_list',) )
# b_v[:] = dac_v
# b_time = f.createVariable('time_rec',  np.float64, ('dive_list',) )
# b_time[:] = time_rec
# b_t_ss = f.createVariable('time_start_stop',  np.float64, ('dive_list',) )
# b_t_ss[:] = time_sta_sto
# b_h = f.createVariable('heading_record',  np.float64, ('dive_list',) )
# b_h[:] = heading_rec
# b_targ = f.createVariable('target_record',  np.float64, ('dive_list',) )
# b_targ[:] = target_rec
# b_gps = f.createVariable('gps_record',  np.float64, ('dive_list','lat_lon') )
# b_gps[:] = gps_rec
# f.close()

# -------------------- BATS CTD KEY
# ctd.txt
# b*_ctd.txt	where * is 5 digit cruise number
#
# File contains 2dbar downcast CTD data whereby all casts are in single cruise file.
#
# Data format as follows:
#
# col1: 8 digit cast_ID
#       $XXXX###  where,
#
#  	$= cruise type
#  	1=bats core
#  	2=bats bloom a
#  	3=bats bloom b
#   5=bats validation
#  	XXXX= cruise number
#  	### = cast number
#
# eg.	10155005  = bats core cruise, cruise 155, cast 5
#
# col2:   decimal year
# col3: 	Latitude (N)
# col4: 	Longitude (W)
# col5:	Pressure (dbar)
# col6:	Depth (m)
# col7:	Temperature (ITS-90, C)
# col8:	Conductivity (S/m)
# col9:	Salinity (PSS-78)
# col10:	Dissolved Oxygen (umol/kg)
# col11:	Beam Attenuation Coefficient (1/m)
# col12:	Fluorescence (relative fluorescence units)

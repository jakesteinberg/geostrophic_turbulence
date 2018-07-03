
import numpy as np
from glider_cross_section import Glider
from mode_decompositions import vertical_modes
from netCDF4 import Dataset
from toolkit import find_nearest

# DIVE SELECTION
# -----------------------------------------------------------------------------------------
#        SG NUMBER,  DIVE NUMBERS,                DIVE FILE PATH
# -----------------------------------------------------------------------------------------
# -- DG WA Coast
# x = Glider(30, np.arange(53, 65), '/Users/jake/Documents/Cuddy_tailored/DG_wa_coast/2006')
# x = Glider(30, np.arange(46, 50), '/Users/jake/Documents/Cuddy_tailored/DG_wa_coast/2006')
# -- DG ABACO 2017
# x = Glider(38, np.arange(66, 77), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/sg037')
# -- DG ABACO 2017
x = Glider(37, np.arange(20, 31), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2018/sg037')
# x = Glider(39, np.arange(12, 20), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2018/sg039')
# -- DG BATS 2015
# x = Glider(35, np.arange(58, 64), '/Users/jake/Documents/baroclinic_modes/DG/sg035_BATS_2015')

# -- match max dive depth to bin_depth
# GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
# deepest_bin_i = find_nearest(GD.variables['grid'][:], x.avg_dep)[0]
# bin_depth = GD.variables['grid'][0:deepest_bin_i]
# bin_depth = np.concatenate((np.arange(0, 300, 5), np.arange(300, 1000, 10), np.arange(1000, 2700, 20)))  # deeper
# bin_depth = np.concatenate((np.arange(0, 300, 5), np.arange(300, 1000, 10), np.arange(1000, 1780, 20)))  # shallower
bin_depth = np.concatenate((np.arange(0, 300, 5), np.arange(300, 1000, 10), np.arange(1000, 4710, 20)))  # shallower


# -------------------
# vertical bin averaging separation of dive-climb cycle into two profiles (extractions from nc files)
time, lon, lat, t, s, dac_u, dac_v, profile_tags = x.make_bin(bin_depth)
ref_lat = np.nanmean(lat)

# -------------------
# time conversion to datetime
d_time = x.time_conversion(time)

# -------------------
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

# -------------------
# computation of absolute salinity, conservative temperature, and potential density anomaly
sa, ct, sig0, N2 = x.density(bin_depth, ref_lat, t, s, lon, lat)

# -------------------
# compute M/W sections and compute velocity
sigth_levels = np.concatenate(
    [np.arange(23, 26.5, 0.5), np.arange(26.2, 27.2, 0.2),
     np.arange(27.2, 27.8, 0.2), np.arange(27.7, 27.8, 0.02), np.arange(27.8, 27.9, 0.01)])
ds, dist, v_g, vbt, isopycdep, isopycx, mwe_lon, mwe_lat = x.transect_cross_section(
    bin_depth, sig0, lon, lat, dac_u, dac_v, profile_tags, sigth_levels)

# -------------------
# vertical modes
N2_avg = np.nanmean(N2, axis=1)
N2_avg[N2_avg < 0] = 0
N2_avg[0] = 1
G, Gz, c = vertical_modes(N2_avg, bin_depth, 0, 30)

# -------------------
# PLOTTING
# plot cross section
x.plot_cross_section(bin_depth, ds, v_g, dist, profile_tags, isopycdep, isopycx, sigth_levels, d_time)

# plot plan view
# bathy_path = '/Users/jake/Documents/Cuddy_tailored/DG_wa_coast/smith_sandwell_wa_coast.nc'
# x.plot_plan_view(mwe_lon, mwe_lat, dac_u, dac_v, ref_lat, profile_tags, d_time,
#                  [-128.5, -123.75, 46.5, 48.5], bathy_path)
bathy_path = '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/OceanWatch_smith_sandwell.nc'
x.plot_plan_view(mwe_lon, mwe_lat, dac_u, dac_v, ref_lat, profile_tags, d_time, [-77.5, -74, 25.5, 27], bathy_path)

# plot t/s
x.plot_t_s(ct, sa)

# vehicle pitch: 'eng_pitchAng'
# DAC_u: 'depth_avg_curr_east'
# DAC_v: 'depth_avg_curr_north'

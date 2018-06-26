
from glider_cross_section import Glider
from netCDF4 import Dataset
import numpy as np
from toolkit import find_nearest

# DIVE SELECTION
# -----------------------------------------------------------------------------------------
#        SG NUMBER,  DIVE NUMBERS,                DIVE FILE PATH
x = Glider(30, np.arange(51, 65), '/Users/jake/Documents/Cuddy_tailored/DG_wa_coast/2006')
# x = Glider(37, np.arange(58, 64), '/Users/jake/Documents/baroclinic_modes/DG/ABACO_2017/sg037')
# x = Glider(35, np.arange(58, 64), '/Users/jake/Documents/baroclinic_modes/DG/sg035_BATS_2015')

# -- match max dive depth to bin_depth
# GD = Dataset('BATs_2015_gridded_apr04.nc', 'r')
# deepest_bin_i = find_nearest(GD.variables['grid'][:], x.avg_dep)[0]
# bin_depth = GD.variables['grid'][0:deepest_bin_i]
bin_depth = np.concatenate((np.arange(0, 300, 5), np.arange(300, 1000, 10), np.arange(1000, 2700, 20)))

# vertical bin averaging separation of dive-climb cycle into two profiles (extractions from nc files)
time, lon, lat, t, s, dac_u, dac_v, profile_tags = x.make_bin(bin_depth)
ref_lat = np.nanmean(lat)

# time conversion to datetime
d_time = x.time_conversion(time)

# computation of absolute salinity, conservative temperature, and potential density anomaly
sa, ct, sig0 = x.density(bin_depth, ref_lat, t, s, lon, lat)

# compute M/W sections and compute velocity
sigth_levels = np.concatenate(
    [np.arange(23, 26.5, 0.5), np.arange(26.2, 27.2, 0.2),
     np.arange(27.2, 27.8, 0.2), np.arange(27.7, 27.9, 0.02)])
ds, dist, v_g, vbt, isopycdep, isopycx, mwe_lon, mwe_lat = x.transect_cross_section(
    bin_depth, sig0, lon, lat, dac_u, dac_v, profile_tags, sigth_levels)

# plot cross section
x.plot_cross_section(bin_depth, ds, v_g, dist, profile_tags, isopycdep, isopycx, sigth_levels, d_time)
# plot plan view
bathy_path = '/Users/jake/Documents/Cuddy_tailored/DG_wa_coast/smith_sandwell_wa_coast.nc'
x.plot_plan_view(mwe_lon, mwe_lat, dac_u, dac_v, ref_lat, profile_tags, d_time, [-128.5, -123.75, 46.5, 48.5], bathy_path)

# vehicle pitch: 'eng_pitchAng'
# DAC_u: 'depth_avg_curr_east'
# DAC_v: 'depth_avg_curr_north'

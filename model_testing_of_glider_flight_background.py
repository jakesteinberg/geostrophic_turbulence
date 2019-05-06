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
import matplotlib.pyplot as plt
from toolkit import plot_pro, nanseg_interp
from zrfun import get_basic_info, get_z

folder_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Model/e_w*')

gamma_avg = np.nan * np.ones((160, 184, len(folder_list)))
ct_avg = np.nan * np.ones((160, 184, len(folder_list)))
sa_avg = np.nan * np.ones((160, 184, len(folder_list)))
for i in range(len(folder_list)):
    pkl_file = open(folder_list[i] + '/gamma_output/extracted_gridded_gamma.pkl', 'rb')
    MOD = pickle.load(pkl_file)
    pkl_file.close()
    ref_lat = 44
    sig0_out_s = MOD['gamma'][:]
    ct_out_s = MOD['CT'][:]  # temp
    sa_out_s = MOD['SA'][:]  # salin
    z_grid = MOD['z'][:]

    gamma_avg[:, :, i] = np.nanmean(sig0_out_s, axis=2)
    ct_avg[:, :, i] = np.nanmean(ct_out_s, axis=2)
    sa_avg[:, :, i] = np.nanmean(sa_out_s, axis=2)

save_anom = 1
if save_anom:
    my_dict = {'dg_z': z_grid,
               'gamma_back': np.nanmean(gamma_avg, axis=2),
               'ct_back': np.nanmean(ct_avg, axis=2),
               'sa_back': np.nanmean(sa_avg, axis=2),}
    output = open('/Users/jake/Documents/baroclinic_modes/Model/background_density.pkl', 'wb')
    pickle.dump(my_dict, output)
    output.close()
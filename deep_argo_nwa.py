# Deep Argo
# NW Atlantic 

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import datetime 
import glob
import seawater as sw 
import pandas as pd 
import scipy.io as si
from scipy.io import netcdf
from scipy.integrate import cumtrapz
import seaborn as sns
# functions I've written 
from grids import make_bin, collect_dives
from mode_decompositions import vertical_modes, PE_Tide_GM


DA = '/Users/jake/Documents/baroclinic_modes/Deep_Argo/da_nwa_4902322_prof.nc'
DA_fid = netcdf.netcdf_file(DA,'r',mmap=False if sys.platform == 'darwin' else mmap, version=1)
dive_num =  DA_fid.variables['CYCLE_NUMBER'][:]
number_profiles = np.size(dive_num)

lat = DA_fid.variables['LATITUDE'][:]
lon = DA_fid.variables['LONGITUDE'][:]
T = DA_fid.variables['TEMP_ADJUSTED'][:]
S = DA_fid.variables['PSAL_ADJUSTED'][:]
P = DA_fid.variables['PRES_ADJUSTED'][:]
depth = sw.dpth(P,np.nanmean(lat))
bin_depth = np.concatenate([np.arange(0,500,5), np.arange(500,300,10), np.arange(300,5800,25)])
bin_press = sw.pres(bin_depth,np.nanmean(lat))
ref_lat = np.nanmean(lat)

T[np.where(T>40)] = np.nan
S[np.where(T>40)] = np.nan
P[np.where(T>40)] = np.nan

bin_up = bin_depth[0:-2]
bin_down = bin_depth[2:]
bin_cen = bin_depth[1:-1] 
T_g = np.empty((np.size(bin_depth),number_profiles))
S_g = np.empty((np.size(bin_depth),number_profiles))
sigma_theta = np.empty((np.size(bin_depth),number_profiles))
theta = np.empty((np.size(bin_depth),number_profiles))
for j in range(number_profiles):
    temp_g = np.empty(np.size(bin_cen))
    salin_g = np.empty(np.size(bin_cen))
    for i in range(np.size(bin_cen)):
        dp_in = (depth[j,:] > bin_up[i]) & (depth[j,:] < bin_down[i])
        if dp_in.size > 1:
            temp_g[i] = np.nanmean(T[j,dp_in])
            salin_g[i] = np.nanmean(S[j,dp_in])
        
    T_g[:,j] = np.concatenate( ([np.nanmean(T[j,0:5])], temp_g,  [np.nanmean(T[j,-5:-0])] ) )
    S_g[:,j] = np.concatenate( ([np.nanmean(S[j,0:5])], salin_g, [np.nanmean(S[j,-5:-0])] ) )        
    sigma_theta[:,j] = sw.pden(S_g[:,j], T_g[:,j], bin_press, pr=0) - 1000
    theta[:,j] = sw.ptmp(S_g[:,j], T_g[:,j], bin_press, pr=0)

# average background properties of profiles along these transects 
sigma_theta_avg = np.array(np.nanmean(sigma_theta,1))
theta_avg = np.array(np.nanmean(theta,1))
salin_avg = np.array(np.nanmean(S_g,1))

# need to interpolate nan's in profilesss 

z = -1*bin_depth
ddz_avg_sigma = np.gradient(sigma_theta_avg,z)
ddz_avg_theta = np.gradient(theta_avg,z)
N2 = np.nan*np.zeros(np.size(sigma_theta_avg))
N2[1:] = np.squeeze(sw.bfrq(salin_avg, theta_avg, bin_press, lat=ref_lat)[0])  
lz = np.where(N2 < 0)   
lnan = np.isnan(N2)
N2[lz] = 0 
N2[lnan] = 0
N = np.sqrt(N2)   

# computer vertical mode shapes 
omega = 0  # frequency zeroed for geostrophic modes
mmax = 60  # highest baroclinic mode to be calculated
nmodes = mmax + 1  
G, Gz, c = vertical_modes(N2,grid,omega,mmax)       
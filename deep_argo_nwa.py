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
from scipy.signal import savgol_filter
import seaborn as sns
# functions I've written 
from grids import make_bin, collect_dives
from mode_decompositions import vertical_modes, PE_Tide_GM


DA = '/Users/jake/Documents/baroclinic_modes/Deep_Argo/da_nwa_4902322_prof.nc'
DA_fid = netcdf.netcdf_file(DA,'r',mmap=False if sys.platform == 'darwin' else mmap, version=1)
dive_num =  DA_fid.variables['CYCLE_NUMBER'][:]
number_profiles_0 = np.size(dive_num)

lat = DA_fid.variables['LATITUDE'][:]
lon = DA_fid.variables['LONGITUDE'][:]
T = DA_fid.variables['TEMP_ADJUSTED'][:]
S = DA_fid.variables['PSAL_ADJUSTED'][:]
P = DA_fid.variables['PRES_ADJUSTED'][:]
depth = sw.dpth(P,np.nanmean(lat))
bin_depth = np.concatenate([np.arange(0,500,5), np.arange(500,3000,15), np.arange(3000,5800,30)])
bin_press = sw.pres(bin_depth,np.nanmean(lat))
ref_lat = np.nanmean(lat)

T[np.where(T>40)] = np.nan
S[np.where(T>40)] = np.nan
P[np.where(T>40)] = np.nan

bin_up = bin_depth[0:-2]
bin_down = bin_depth[2:]
bin_cen = bin_depth[1:-1] 
T_g_0 = np.empty((np.size(bin_depth),number_profiles_0))
S_g_0 = np.empty((np.size(bin_depth),number_profiles_0))
sigma_theta_0 = np.empty((np.size(bin_depth),number_profiles_0))
theta_0 = np.empty((np.size(bin_depth),number_profiles_0))
for j in range(number_profiles_0):
    temp_g = np.empty(np.size(bin_cen))
    salin_g = np.empty(np.size(bin_cen))
    for i in range(np.size(bin_cen)):
        dp_in = (depth[j,:] > bin_up[i]) & (depth[j,:] < bin_down[i])
        if dp_in.size > 1:
            temp_g[i] = np.nanmean(T[j,dp_in])
            salin_g[i] = np.nanmean(S[j,dp_in])
        
    T_g_0[:,j] = np.concatenate( ([np.nanmean(T[j,0:5])], temp_g,  [np.nanmean(T[j,-5:-0])] ) )
    S_g_0[:,j] = np.concatenate( ([np.nanmean(S[j,0:5])], salin_g, [np.nanmean(S[j,-5:-0])] ) )        
    sigma_theta_0[:,j] = sw.pden(S_g_0[:,j], T_g_0[:,j], bin_press, pr=0) - 1000
    theta_0[:,j] = sw.ptmp(S_g_0[:,j], T_g_0[:,j], bin_press, pr=0)

# select only deep dives
S_g = S_g_0[:,5:]
theta = theta_0[:,5:]
sigma_theta = sigma_theta_0[:,5:]
sz = S_g.shape
number_profiles = sz[1]

# interpolate nans
high = 10
low = -20
for i in range(number_profiles):
     this_prof = sigma_theta[:,i]
     bad = np.where(np.isnan(sigma_theta[high:low,i]))
     this_dep = bin_depth[high:low]
     this_sig = sigma_theta[high:low,i]
     for j in range(np.size(bad)):
         this_sig[bad[0][j]] = np.interp( this_dep[bad[0][j]],   [this_dep[bad[0][j]-1], this_dep[bad[0][j]+1]], [this_sig[bad[0][j]-1], this_sig[bad[0][j]+1]]  )
     sigma_theta[high:low,i] = this_sig  
     # lower  
     bounds = np.where(np.isnan(this_prof)) 
     lower = np.where(bounds[0] > 100)   
     upper = np.where(bounds[0] < 100)
     if np.size(lower) > 0: 
         sigma_theta[bounds[0][lower],i] = sigma_theta[bounds[0][lower[0][0]]-1,i]+.0002
     if np.size(upper) > 0:
         sigma_theta[bounds[0][upper],i] = sigma_theta[bounds[0][upper[0][0]]+1,i]
    

# average background properties of profiles along these transects 
sigma_theta_avg = np.array(np.nanmean(sigma_theta,1))
theta_avg = np.array(np.nanmean(theta,1))
salin_avg = np.array(np.nanmean(S_g,1))

# need to interpolate nan's in profilesss 

z = -1*bin_depth
ddz_avg_sigma_0 = np.gradient(sigma_theta_avg,z)
ddz_avg_theta = np.gradient(theta_avg,z)

window_size, poly_order = 41, 3
ddz_avg_sigma = savgol_filter(ddz_avg_sigma_0, window_size, poly_order)

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
G, Gz, c = vertical_modes(N2,bin_depth,omega,mmax)     

# eta 
theta_anom = theta - np.transpose(np.tile(theta_avg,[number_profiles,1]))
salin_anom = S_g - np.transpose(np.tile(salin_avg,[number_profiles,1]))
sigma_anom = sigma_theta - np.transpose(np.tile(sigma_theta_avg,[number_profiles,1]))
eta = sigma_anom/np.transpose(np.tile(ddz_avg_sigma,[number_profiles,1]))
eta_theta = theta_anom/np.transpose(np.tile(ddz_avg_theta,[number_profiles,1]))

# first taper fit above and below min/max limits
# Project modes onto each eta (find fitted eta)
# Compute PE  
eta_fit_depth_min = 250
eta_fit_depth_max = 4000
AG = np.zeros([nmodes, number_profiles])
AG_theta = np.zeros([nmodes, number_profiles])
Eta_m = np.nan*np.zeros([np.size(bin_depth), number_profiles])
Neta = np.nan*np.zeros([np.size(bin_depth), number_profiles])
NEta_m = np.nan*np.zeros([np.size(bin_depth), number_profiles])
Eta_theta_m = np.nan*np.zeros([np.size(bin_depth), number_profiles])
PE_per_mass = np.nan*np.zeros([nmodes, number_profiles])
PE_theta_per_mass = np.nan*np.zeros([nmodes, number_profiles])
for i in range(number_profiles):
    this_eta = eta[:,i].copy() 
    # obtain matrix of NEta
    Neta[:,i] = N*this_eta
    this_eta_theta = eta_theta[:,i].copy()
    iw = np.where((bin_depth>=eta_fit_depth_min) & (bin_depth<=eta_fit_depth_max))
    if iw[0].size > 1:
        eta_fs = eta[:,i].copy() # ETA
        eta_theta_fs = eta_theta[:,i].copy()
    
        i_sh = np.where( (bin_depth < eta_fit_depth_min))
        eta_fs[i_sh[0]] = bin_depth[i_sh]*this_eta[iw[0][0]]/bin_depth[iw[0][0]]
        eta_theta_fs[i_sh[0]] = bin_depth[i_sh]*this_eta_theta[iw[0][0]]/bin_depth[iw[0][0]]
        
        i_dp = np.where( (bin_depth > eta_fit_depth_max) )
        eta_fs[i_dp[0]] = (bin_depth[i_dp] - bin_depth[-1])*this_eta[iw[0][-1]]/(bin_depth[iw[0][-1]]-bin_depth[-1])
        eta_theta_fs[i_dp[0]] = (bin_depth[i_dp] - bin_depth[-1])*this_eta_theta[iw[0][-1]]/(bin_depth[iw[0][-1]]-bin_depth[-1])
            
        AG[1:,i] = np.squeeze(np.linalg.lstsq(G[:,1:],np.transpose(np.atleast_2d(eta_fs)))[0])
        AG_theta[1:,i] = np.squeeze(np.linalg.lstsq(G[:,1:],np.transpose(np.atleast_2d(eta_theta_fs)))[0])
        Eta_m[:,i] = np.squeeze(np.matrix(G)*np.transpose(np.matrix(AG[:,i])))
        NEta_m[:,i] = N*np.array(np.squeeze(np.matrix(G)*np.transpose(np.matrix(AG[:,i]))))
        Eta_theta_m[:,i] = np.squeeze(np.matrix(G)*np.transpose(np.matrix(AG_theta[:,i])))
        PE_per_mass[:,i] = (1/2)*AG[:,i]*AG[:,i]*c*c
        PE_theta_per_mass[:,i] = (1/2)*AG_theta[:,i]*AG_theta[:,i]*c*c 


fig0, ax0 = plt.subplots()
for i in range(number_profiles):
     ax0.plot(theta_anom[:,i],z)
# ax0.axis([-600, 600, -5800, 0])  
plt.axis([-1, 1, -5800, 0])    
fig0.savefig('/Users/jake/Desktop/argo/nwa_theta_anom.png',dpi = 300)    

fig0, ax0 = plt.subplots()
for i in range(number_profiles):
     ax0.plot(eta[:,i],z)
     ax0.plot(Eta_m[:,i],z,color='k',linestyle='--')
ax0.axis([-600, 600, -5800, 0])  
# plt.axis([-1, 1, -5800, 0])    
fig0.savefig('/Users/jake/Desktop/argo/nwa_eta.png',dpi = 300)     


avg_PE = np.nanmean(PE_per_mass,1)
avg_PE_theta = np.nanmean(PE_theta_per_mass,1)
f_ref = np.pi*np.sin(np.deg2rad(26.5))/(12*1800)
rho0 = 1025
dk = f_ref/c[1]
sc_x = (1000)*f_ref/c[1:]

PE_SD, PE_GM = PE_Tide_GM(1025,bin_depth,nmodes,np.transpose(np.atleast_2d(N2)),f_ref)

fig0, ax0 = plt.subplots()
ax0.plot(sc_x,avg_PE[1:]/dk,color='r',linewidth=2)
ax0.scatter(sc_x,avg_PE[1:]/dk,color='r',s=10)
ax0.plot(sc_x,PE_GM/dk,linestyle='--',color='#DAA520')
ax0.axis([10**-2, 1.5*10**1, 10**(-4), 10**(3)])
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.grid()
fig0.savefig('/Users/jake/Desktop/argo/nwa_pe.png',dpi = 300)
 

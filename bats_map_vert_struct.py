# BATS (OBJECTIVE MAP OUTPUT)

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
import pickle
# functions I've written 
from grids import make_bin, collect_dives
from mode_decompositions import vertical_modes, PE_Tide_GM
from toolkit import cart2pol, pol2cart, plot_pro

# physical parameters 
g = 9.81
rho0 = 1027
bin_depth = np.concatenate([np.arange(0,150,10), np.arange(150,300,10), np.arange(300,5000,20)])
ref_lat = 31.8
lat_in = 31.7
lon_in = 64.2
grid = bin_depth[1:-1]
grid_p = sw.pres(grid,lat_in)
z = -1*grid
# mode parameters 
omega = 0  # frequency zeroed for geostrophic modes
mmax = 60  # highest baroclinic mode to be calculated
nmodes = mmax + 1  
deep_shr_max = 0.1 # maximum allowed deep shear [m/s/km]
deep_shr_max_dep = 3500 # minimum depth for which shear is limited [m]

# PLOTTING SWITCHES 
plot_eta = 0

#### LOAD IN TRANSECT TO PROFILE DATA COMPILED IN BATS_TRANSECTS.PY
pkl_file = open('/Users/jake/Documents/geostrophic_turbulence/BATS_obj_map.pkl', 'rb')
bats_trans = pickle.load(pkl_file)
pkl_file.close() 
Time = bats_trans['time'][0]
Sigma_Theta = np.transpose(bats_trans['Sigma_Theta'][0])
U = np.transpose(bats_trans['U_g'][0])
V = np.transpose(bats_trans['V_g'][0])

# average background properties of profiles along these transects 
sigma_theta_avg = np.nanmean(Sigma_Theta,axis=1)
ddz_avg_sigma = np.gradient(sigma_theta_avg,z)
# theta_avg = np.array(np.nanmean(df_t,1))
# salin_avg = np.array(np.nanmean(df_s,1))
# ddz_avg_theta = np.gradient(theta_avg,z)
# N2 = np.nan*np.zeros(np.size(sigma_theta_avg))
# N2[1:] = np.squeeze(sw.bfrq(salin_avg, theta_avg, grid_p, lat=ref_lat)[0])  
N2 = (-g/rho0)*ddz_avg_sigma
lz = np.where(N2 < 0)   
lnan = np.isnan(N2)
N2[lz] = 0 
N2[lnan] = 0
N = np.sqrt(N2)   

# f,ax = plt.subplots()
# ax.plot(N2,z)
# plot_pro(ax)

# computer vertical mode shapes 
G, Gz, c = vertical_modes(N2,grid,omega,mmax)

# first taper fit above and below min/max limits
# Project modes onto each eta (find fitted eta)
# Compute PE 

# presort V 
# good_v = np.zeros(np.size(Time))
# for i in range(np.size(Time)):
#     v_dz = np.gradient(V[10:,i])
#     if np.nanmax(np.abs(v_dz)) < 0.075:
#         good_v[i] = 1        
# good0 = np.intersect1d(np.where((np.abs(V[-45,:]) < 0.2))[0],np.where((np.abs(V[10,:]) < 0.4))[0])
# good = np.intersect1d(np.where(good_v > 0),good0)
# V2 = V[:,good]
# Eta2 = Eta[:,good]
# Eta_theta2 = Eta_theta[:,good]

sz = np.shape(V)
num_profs = sz[1]
eta_fit_depth_min = 50
eta_fit_depth_max = 3800
eta_theta_fit_depth_max = 4200
AG = np.zeros([nmodes, num_profs])
AGz_U = np.zeros([nmodes, num_profs])
AGz_V = np.zeros([nmodes, num_profs])
U_m = np.nan*np.zeros([np.size(grid), num_profs])
V_m = np.nan*np.zeros([np.size(grid), num_profs])
HKE_U_per_mass = np.nan*np.zeros([nmodes, num_profs])
HKE_V_per_mass = np.nan*np.zeros([nmodes, num_profs])
# PE_theta_per_mass = np.nan*np.zeros([nmodes, num_profs])
modest = np.arange(11,nmodes)
good_prof = np.ones(num_profs)
HKE_noise_threshold = 1e-4 # 1e-5
for i in range(num_profs):    
    # fit to velocity profiles
    this_U = U[:,i].copy()
    this_V = V[:,i].copy()
    iu = np.where( ~np.isnan(this_U) )
    iv = np.where( ~np.isnan(this_V) )
    
    if iu[0].size > 1:
        AGz_U[:,i] =  np.squeeze(np.linalg.lstsq( np.squeeze(Gz[iv,:]),np.transpose(np.atleast_2d(this_U[iu])))[0]) # Gz(iv,:)\V_g(iv,ip)                
        U_m[:,i] =  np.squeeze(np.matrix(Gz)*np.transpose(np.matrix(AGz[:,i])))  #Gz*AGz[:,i];
        HKE_U_per_mass[:,i] = AGz[:,i]*AGz[:,i]
        ival = np.where( HKE_U_per_mass[modest,i] >= HKE_noise_threshold )
        if np.size(ival) > 0:
            good_prof[i] = 0 # flag profile as noisy
    else:
        good_prof[i] = 0 # flag empty profile as noisy as well
    
    if iv[0].size > 1:
        AGz_V[:,i] =  np.squeeze(np.linalg.lstsq( np.squeeze(Gz[iv,:]),np.transpose(np.atleast_2d(this_V[iv])))[0]) # Gz(iv,:)\V_g(iv,ip)                
        V_m[:,i] =  np.squeeze(np.matrix(Gz)*np.transpose(np.matrix(AGz[:,i])))  #Gz*AGz[:,i];
        HKE_V_per_mass[:,i] = AGz[:,i]*AGz[:,i]
        ival = np.where( HKE_V_per_mass[modest,i] >= HKE_noise_threshold )
        if np.size(ival) > 0:
            good_prof[i] = 0 # flag profile as noisy
    else:
        good_prof[i] = 0 # flag empty profile as noisy as well
 
### COMPUTE EOF SHAPES AND COMPARE TO ASSUMED STRUCTURE 
### EOF SHAPES 
## find EOFs of dynamic horizontal current (U) mode amplitudes _____ADCP_____
AGzq = AGz_U #(:,quiet_prof)
nq = np.size(good_prof) # good_prof and dg_good_prof
avg_AGzq = np.nanmean(np.transpose(AGzq),axis=0)
AGzqa = AGzq - np.transpose(np.tile(avg_AGzq,[nq,1])) # mode amplitude anomaly matrix
cov_AGzqa = (1/nq)*np.matrix(AGzqa)*np.matrix(np.transpose(AGzqa)) # nmodes X nmodes covariance matrix
var_AGzqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa))))*np.matrix(np.sqrt(np.diag(cov_AGzqa)))
cor_AGzqa = cov_AGzqa/var_AGzqa # nmodes X nmodes correlation matrix

D_AGzqa,V_AGzqa = np.linalg.eig(cov_AGzqa) # columns of V_AGzqa are eigenvectors 
 
EOFseries_U = np.transpose(V_AGzqa)*np.matrix(AGzqa) # EOF "timeseries' [nmodes X nq]
EOFshape_U = np.matrix(Gz)*V_AGzqa # depth shape of eigenfunctions [ndepths X nmodes]
EOFshape1_BTpBC1 = np.matrix(Gz[:,0:2])*V_AGzqa[0:2,0] # truncated 2 mode shape of EOF#1
EOFshape2_BTpBC1 = np.matrix(Gz[:,0:2])*V_AGzqa[0:2,1] # truncated 2 mode shape of EOF#2  

## find EOFs of dynamic horizontal current (V) mode amplitudes _____ADCP_____
AGzq = AGz_V #(:,quiet_prof)
nq = np.size(good_prof) # good_prof and dg_good_prof
avg_AGzq = np.nanmean(np.transpose(AGzq),axis=0)
AGzqa = AGzq - np.transpose(np.tile(avg_AGzq,[nq,1])) # mode amplitude anomaly matrix
cov_AGzqa = (1/nq)*np.matrix(AGzqa)*np.matrix(np.transpose(AGzqa)) # nmodes X nmodes covariance matrix
var_AGzqa = np.transpose(np.matrix(np.sqrt(np.diag(cov_AGzqa))))*np.matrix(np.sqrt(np.diag(cov_AGzqa)))
cor_AGzqa = cov_AGzqa/var_AGzqa # nmodes X nmodes correlation matrix

D_AGzqa,V_AGzqa = np.linalg.eig(cov_AGzqa) # columns of V_AGzqa are eigenvectors 
 
EOFseries_V = np.transpose(V_AGzqa)*np.matrix(AGzqa) # EOF "timeseries' [nmodes X nq]
EOFshape_V = np.matrix(Gz)*V_AGzqa # depth shape of eigenfunctions [ndepths X nmodes]
EOFshape1_BTpBC1 = np.matrix(Gz[:,0:2])*V_AGzqa[0:2,0] # truncated 2 mode shape of EOF#1
EOFshape2_BTpBC1 = np.matrix(Gz[:,0:2])*V_AGzqa[0:2,1] # truncated 2 mode shape of EOF#2  
 

## PLOT ETA / EOF 
if plot_eta > 0: 
    f,(ax0,ax1) = plt.subplots(1,2,sharey=True)
    for j in range(num_profs):
        ax0.plot(U[:,j],grid,color='#CD853F',linewidth=1.25)  
        ax0.plot(U_m[:,j],grid,color='k',linestyle='--',linewidth=.75)
        ax1.plot(V[:,j],grid,color='#CD853F',linewidth=1.25)  
        ax1.plot(V_m[:,j],grid,color='k',linestyle='--',linewidth=.75)
    ax0.text(190,800,str(num_profs)+' Profiles')
    ax0.axis([-.4, .4, 0, 5000]) 
    ax1.axis([-.4, .4, 0, 5000]) 
    ax0.set_title("BATS '15 DG U")
    ax1.set_title("BATS '15 DG V")
    ax0.set_ylabel('Depth [m]',fontsize=14)
    ax0.set_xlabel('u [m/s]',fontsize=14)
    ax1.set_xlabel('v [m/s]',fontsize=14)
    ax0.invert_yaxis() 
    ax0.grid()    
    plot_pro(ax1)
    # f.savefig('/Users/jake/Desktop/bats/dg035_15_Eta_a.png',dpi = 300)
    # plt.show()    
    
    max_plot = 3 
    f, (ax1,ax2) = plt.subplots(1,2,sharey=True) 
    n2p = ax1.plot((np.sqrt(N2)*(1800/np.pi)),grid,color='k',label='N(z) [cph]') 
    colors = plt.cm.Dark2(np.arange(0,4,1))
    for ii in range(max_plot):
        ax1.plot(Gz[:,ii], grid,color='#2F4F4F',linestyle='--')
        p_eof_u=ax1.plot(-EOFshape_U[:,ii], grid,color=colors[ii,:],label='EOF # = ' + str(ii+1),linewidth=2)
        ax2.plot(Gz[:,ii], grid,color='#2F4F4F',linestyle='--')
        p_eof_v=ax2.plot(-EOFshape_V[:,ii], grid,color=colors[ii,:],label='EOF # = ' + str(ii+1),linewidth=2)
        
    handles, labels = ax1.get_legend_handles_labels()    
    ax1.legend(handles,labels,fontsize=10)    
    ax1.axis([-4,4,0,5000])    
    ax1.invert_yaxis()
    ax1.grid()
    ax1.set_ylabel('Depth [m]')
    ax1.set_title('BATS DG EOF Mode Shapes (U)')
    ax1.set_xlabel('U Mode Shapes (DG)')   
    ax2.set_title('BATS DG EOF Mode Shapes (V)')
    ax2.set_xlabel('V Mode Shapes (DG)')
    plot_pro(ax2)
    
    # for ii in range(max_plot):
    #     ax2.plot(G[:,ii+1]/np.max(grid),grid,color='#2F4F4F',linestyle='--')
    #     p_eof_eta=ax2.plot(-EOFetashape[:,ii]/np.max(grid), grid,color=colors[ii,:],label='EOF # = ' + str(ii+1),linewidth=2)
    # handles, labels = ax2.get_legend_handles_labels()    
    # ax2.legend(handles,labels,fontsize=10)    
    # ax2.axis([-.7,.7,0,5000])    
    # ax2.set_title('EOF Mode Shapes (DG)')
    # ax2.set_xlabel('Vert. Disp. Mode Shapes')
    # ax2.invert_yaxis()    
    # plot_pro(ax2)

# avg_PE = np.nanmean(PE_per_mass,1)
# good_prof_i = good_prof #np.where(good_prof > 0)
avg_KE_U = np.nanmean(HKE_U_per_mass[:,np.where(good_prof>0)[0]],1)
avg_KE_V = np.nanmean(HKE_V_per_mass[:,np.where(good_prof>0)[0]],1)

fig0, ax0 = plt.subplots()
for i in range(np.size(good_prof)):
    if good_prof[i] > 0:
        ax0.plot(np.arange(0,61,1),HKE_U_per_mass[:,i])
ax0.set_xscale('log')      
ax0.set_yscale('log')    
plot_pro(ax0)    
# avg_PE_theta = np.nanmean(PE_theta_per_mass,1)
f_ref = np.pi*np.sin(np.deg2rad(ref_lat))/(12*1800)
rho0 = 1025
dk = f_ref/c[1]
sc_x = (1000)*f_ref/c[1:]
vert_wavenumber = f_ref/c[1:]
    
PE_SD, PE_GM = PE_Tide_GM(rho0,grid,nmodes,np.transpose(np.atleast_2d(N2)),f_ref)

# KE parameters
dk_ke = 1000*f_ref/c[1]   
k_h = 1e3*(f_ref/c[1:])*np.sqrt( avg_KE[1:]/avg_PE[1:])

# load in Station BATs PE Comparison
SB = si.loadmat('/Users/jake/Desktop/bats/station_bats_pe.mat')
sta_bats_pe = SB['out'][0][0][0]
sta_bats_c = SB['out'][0][0][3]
sta_bats_f = SB['out'][0][0][2]
sta_bats_dk = SB['out'][0][0][1]

plot_eng = 0
plot_spec = 0
plot_comp = 0
if plot_eng > 0:    
    if plot_spec > 0:
        fig0, ax0 = plt.subplots()
        # PE_p = ax0.plot(sc_x,avg_PE[1:]/dk,color='#B22222',label='PE',linewidth=1.5)
        KE_p = ax0.plot(sc_x,avg_KE_U[1:]/dk,'g',label='KE_u',linewidth=1.5)        
        ax0.scatter(sc_x,avg_KE_U[1:]/dk,color='g',s=10) # DG KE
        KE_p = ax0.plot(sc_x,avg_KE_V[1:]/dk,'r',label='KE_v',linewidth=1.5)        
        ax0.scatter(sc_x,avg_KE_V[1:]/dk,color='r',s=10) # DG KE
        ax0.text( 1000*f_ref/c[-2]+.1, 1000*f_ref/c[-2], r'f/c$_m$',fontsize=10)  
        
        # limits/scales 
        ax0.plot( [3*10**-1, 3*10**0], [1.5*10**1, 1.5*10**-2],color='k',linewidth=0.75)
        ax0.plot([3*10**-2, 3*10**-1],[7*10**2, ((5/3)*(np.log10(2*10**-1) - np.log10(2*10**-2) ) +  np.log10(7*10**2) )] ,color='k',linewidth=0.75)
        ax0.text(3.3*10**-1,1.3*10**1,'-3',fontsize=10)
        ax0.text(3.3*10**-2,6*10**2,'-5/3',fontsize=10)
        ax0.plot( [1000*f_ref/c[1], 1000*f_ref/c[-2]],[1000*f_ref/c[1], 1000*f_ref/c[-2]],linestyle='--',color='k',linewidth=0.8)
         
        ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax0.axis([10**-2, 10**1, 3*10**(-4), 2*10**(3)])
        ax0.set_xlabel(r'Scaled Vertical Wavenumber = (Rossby Radius)$^{-1}$ = $\frac{f}{c_m}$ [$km^{-1}$]',fontsize=14)
        ax0.set_ylabel('Spectral Density, Hor. Wavenumber',fontsize=14) # ' (and Hor. Wavenumber)')
        ax0.set_title('DG 2015 BATS Deployment (Energy Spectra)',fontsize=14)
        # handles, labels = ax0.get_legend_handles_labels()
        # ax0.legend([handles[0],handles[1],handles[2]],[labels[0], labels[1], labels[2]],fontsize=12)
        plt.tight_layout()
        plot_pro(ax0)
        # fig0.savefig('/Users/jake/Desktop/bats/dg035_15_PE_b.png',dpi = 300)
        # plt.close()
        # plt.show()
        
### SAVE 
# write python dict to a file
sa = 1
if sa > 0:
    mydict = {'sc_x': sc_x,'avg_ke_u': avg_KE_U,'avg_ke_v': avg_KE_V, 'dk': dk}
    output = open('/Users/jake/Documents/geostrophic_turbulence/BATS_OM_KE.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close()        
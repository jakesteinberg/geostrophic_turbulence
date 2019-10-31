import numpy as np
import pickle
import matplotlib.pyplot as plt 
from toolkit import plot_pro

# BATS SHIP
pkl_file = open('/Users/jake/Desktop/bats/station_bats_pe_jan30.pkl', 'rb')  # jan 30 2018
SB = pickle.load(pkl_file)
pkl_file.close()
sta_bats_depth = SB['depth']
sta_bats_pe = SB['PE']
sta_bats_c = SB['c']
sta_bats_f = np.pi * np.sin(np.deg2rad(31.6)) / (12 * 1800)
sta_bats_dk = sta_bats_f / sta_bats_c[1]

# # winter1
sta_bats_n2_0 = SB['N2_per_season'][:, 0]
# # summer
sta_bats_n2_1 = SB['N2_per_season'][:, 1]

# GM79
modenum = np.arange(1, 40)
f_ref = sta_bats_f
Navg = np.sqrt(np.nanmean(sta_bats_n2_1))

bGM = 1300.0  # GM internal wave depth scale [m]
N0_GM = 0.0052  # GM N scale [s^-1];
jstar = 6.0  # GM vertical mode number scale
EGM = 0.000063  # GM energy level [no dimensions]
HHterm = 1.0 / (modenum * modenum + jstar * jstar)
HH = HHterm / np.sum(HHterm)
HH_check = np.nansum(HH)

# compute energy over a range of frequencies and then integrate
omega = np.arange(np.round(f_ref, 5), np.round(Navg, 5), 0.00001)
if omega[0] < f_ref:
    omega[0] = f_ref.copy() + 0.000001
BBterm = ((2.0 / 3.14159) * (f_ref / omega) * (((omega ** 2) - (f_ref ** 2))**(-0.5))) / np.trapz(((2.0 / 3.14159) * (f_ref / omega) * (((omega ** 2) - (f_ref ** 2))**(-0.5))), omega)
BB_check = np.trapz(BBterm, omega)

EE = np.tile(BBterm[:, None], (1, len(modenum))) * np.tile(HH[None, :], (len(omega), 1)) * EGM  # dimensions = [number of frequencies, number of modes]
omega_g = np.tile(omega[:, None], (1, len(modenum)))
FKE = 0.5 * bGM * bGM * N0_GM * Navg * ((omega_g ** 2) + (f_ref ** 2)) * (1.0 / (omega_g ** 2)) * EE

# FPE_int = np.nan * np.ones(np.shape(FPE[0, :]))
FKE_int = np.nan * np.ones(np.shape(FKE[0, :]))
for i in range(len(modenum[1:])):
    # FPE_int[i] = np.trapz(FPE[:, i], omega)
    FKE_int[i] = np.trapz(FKE[:, i], omega)


# e-folding scale of N
def n_exp(p, n_e, z):
    a = p[0]
    b = p[1]
    fsq = (n_e - a*np.exp((z/b)))**2
    return fsq.sum()

from scipy.optimize import fmin

ins = np.transpose(np.concatenate([sta_bats_n2_1[:, None], -1.0*sta_bats_depth[:, None]], axis=1))
p = [0.001, 500.0]
min_p1 = fmin(n_exp, p, args=(tuple(ins)), disp=0)

z_g = np.flipud(np.arange(-5000, 0, 10))
n_a = min_p1[0]*np.exp((z_g/min_p1[1]))

f, ax = plt.subplots()
ax.plot(sta_bats_n2_1,  -1.0*sta_bats_depth)
ax.plot(n_a, z_g)
plot_pro(ax)

# ------ GM75
# b = 1300.0
# n0 = 0.0052
# j_star = 6
# E = 0.000063

# n = np.sqrt(sta_bats_n2_1)/n0  # normalization of n
# w_i = sta_bats_f/n0 # normalization of f
# w = np.arange(w_i+0.001, np.nanmean(n), (np.nanmean(n) - w_i)/100)
# j = np.arange(1,40)  # first forty modes
# beta = j*np.pi*np.nanmean(n)  # vertical wavenumber (approximated for average n)
# beta_star = j_star*np.pi*np.nanmean(n)
#
# A = ((2.5 - 1)*(1 + beta/beta_star)**(-2.5))/(np.trapz((2.5 - 1)*(1 + beta/beta_star)**(-2.5), beta/beta_star))
# A_check = np.trapz(A, beta/beta_star)  # intergral of A = 1
#
# B = (2/np.pi)*w_i*(w**(-2))*(1 - (w_i/w)**2)**(-1/2)/(np.trapz((2/np.pi)*w_i*(w**(-2))*(1 - (w_i/w)**2)**(-1/2), w))
# B_check = np.trapz(B, w)
#
# X = np.nanmean(n*n0)*((n0*w)**(-4))*((n0*w)**2 + (n0*w_i)**2)
# F = np.nan * np.ones((len(beta), len(w)))
# for i in range(len(beta)):
#     F[i, :] = ((n0*w)**2)*np.sqrt(X)*(E/(beta_star*n0))*A[i]*B
#
# f, ax = plt.subplots()
# F_int = np.nan * np.ones(len(j))
# for m in range(len(j)):
#     F_int[m] = np.trapz(F[m, :], w*n0)
# ax.plot(j, F_int)
# ax.set_yscale('log')
# ax.set_xscale('log')
# plot_pro(ax)
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mode_decompositions import vertical_modes
from toolkit import cart2pol, pol2cart, plot_pro
import pandas as pd
import seawater as sw
import pickle
from scipy.io import netcdf

fig = plt.figure()
ax = fig.gca(projection='3d')
# X, Y, Z = axes3d.get_test_data(0.1)
y_pos = 40

# dives 
# x_s = np.arange(0,65,7.5)
x_s = np.arange(0,131,25/2)
y_s = y_pos*np.ones(np.size(x_s))
z_s = np.array([0,-5000,0,-5000,0,-5000,0,-5000,0])

# dacs
x_q = np.arange(x_s[1],101,25)
y_q = y_pos*np.ones(np.size(x_q))
u_q = np.array([0, -1, 0, 2])
v_q = np.array([0, -5, 12, 8])
# ax.scatter(x_q, y_q, np.zeros(np.size(x_q)),color='g',s=8,zorder=3)
# ax.quiver(x_q, y_q, np.zeros(np.size(x_q)), u_q, v_q, 0*np.ones(np.size(x_q)),color='g', length=10, normalize=True,zorder=3)
# ax.text(x_q[-1]+6,y_q[-1]+7,0,'DAC')

# alternate load in one dive transect from BATS
pkl_file = open('/Users/jake/Desktop/bats/schematic_dive.pkl', 'rb')
bats_trans = pickle.load(pkl_file)
pkl_file.close() 
grid = bats_trans['grid']
Ds = bats_trans['Ds']
V_g = bats_trans['V_g']
dist = bats_trans['dive_dist']
iso_x = bats_trans['isopyc_x']
iso_dep = bats_trans['isopyc_dep']
sig_good = bats_trans['sig_good']
dac_to = bats_trans['DAC_to_port']

# x_g = np.arange(7.5,55,0.5)
x_g = np.arange(25/2,101,0.5)
y_g = y_pos*np.ones(np.size(x_g))
dist_g = np.sqrt(x_g**2 + y_g**2)
z_g = np.linspace(0,-5000,150)
X,Z = np.meshgrid(x_g,z_g)
# in_x = np.where( (x_g > 25) & (x_g < 50))
# in_z = np.where( (z_g < -1250) & (z_g > -3000) )
in_x2 = np.where( (x_g > 50) & (x_g < 85))
in_z2 = np.where( (z_g < 0) & (z_g > -1000) )
in_x3 = np.where( (x_g > 50) & (x_g < 85))
in_z3 = np.where( (z_g < -1000) )
V = np.zeros(np.shape(X))
A = 2.5
B = 2.5
A2 = 1.5
B2 = .25e7
### V[in_z[0][0]:in_z[0][-1], in_x[0][0]:in_x[0][-1]] = A*np.sin((2*np.pi/50)*X[ in_z[0][0]:in_z[0][-1], in_x[0][0]:in_x[0][-1] ])*B*np.cos((2*np.pi/4000)*Z[ in_z[0][0]:in_z[0][-1], in_x[0][0]:in_x[0][-1] ])
V[in_z2[0][0]:in_z2[0][-1], in_x2[0][0]:in_x2[0][-1]] = -1*A*np.sin((2*np.pi/100)*X[ in_z2[0][0]:in_z2[0][-1], in_x2[0][0]:in_x2[0][-1] ])*B*np.cos((2*np.pi/4000)*Z[ in_z2[0][0]:in_z2[0][-1], in_x2[0][0]:in_x2[0][-1] ])
### V[in_z3[0][0]:in_z3[0][-1], in_x3[0][0]:in_x3[0][-1]] = A2*np.sin((2*np.pi/50)*X[ in_z3[0][0]:in_z3[0][-1], in_x3[0][0]:in_x3[0][-1] ])*B2*np.cos((2*np.pi/17000)*Z[ in_z3[0][0]:in_z3[0][-1], in_x3[0][0]:in_x3[0][-1] ])
V[in_z3[0][0]:in_z3[0][-1], in_x3[0][0]:in_x3[0][-1]] = A2*np.sin((2*np.pi/250)*X[ in_z3[0][0]:in_z3[0][-1], in_x3[0][0]:in_x3[0][-1] ])*(B2*((1/( (Z[ in_z3[0][0]:in_z3[0][-1], in_x3[0][0]:in_x3[0][-1] ])**2))))-4

# levels=np.concatenate((np.arange(-7,0,.25), np.arange(0,7,1)))
# cset = ax.contour(X[5:-5,5:-5], np.ones(np.shape(X[5:-5,5:-5]))*V[5:-5,5:-5], Z[5:-5,5:-5], zdir='y', offset=y_pos+2, cmap=cm.RdBu_r,levels=levels,zorder=0,linewidth=2)
lvs = np.arange(-.15,.3,.02)
cset = ax.contour(np.tile(Ds,(len(grid),1)), np.ones(np.shape(V_g))*V_g, np.transpose(np.tile(-1*grid,(8,1))), zdir='y', offset=y_pos+0, levels = lvs,cmap=cm.RdBu_r,zorder=0,linewidth=2)


# ax.plot3D(x_s, y_s, z_s,color='k',zorder=1,zdir='z',linewidth=0.75)  # old made up data
for p in range(8):
    ax.scatter3D( dist[:,p], y_pos*np.ones(len(grid)), -1*grid,s=.5,color='k',zorder=2,zdir='z',linewidth=0.75)  # bats dive transect 

ax.plot3D(x_s, y_s, np.zeros(np.size(x_s)),color='#696969',linestyle='--',zorder=1,zdir='z',linewidth=0.75) 

# G = 
ax.plot3D(-20*np.ones(5),y_pos*np.ones(5),np.linspace(-5000,0,5),color='k',zdir='z')
ax.plot3D(-20*np.ones(5),np.linspace(y_pos-10,y_pos+10,5),np.zeros(5),color='k',zdir='z')
sample_v = 120
# ax.plot3D(x_g[sample_v]*np.ones(5),y_pos*np.ones(5),np.linspace(-5000,0,5),color='r',zdir='z',linestyle='--',linewidth=1,zorder=3)
ax.plot3D(-20*np.ones(np.size(z_g[1:-1])),y_pos + V[1:-1,sample_v],np.flipud(np.linspace(-5000,0,np.size(z_g[1:-1]))),color='#800000',zdir='z')
ax.text(-15,y_pos+10,20,r'$u_{x=30} (z)$')

# G'
GD = netcdf.netcdf_file('BATs_2015_gridded.nc','r')
df_t = pd.DataFrame(np.float64(GD.variables['Temperature'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
df_s = pd.DataFrame(np.float64(GD.variables['Salinity'][:]),index=np.float64(GD.variables['grid'][:]),columns=np.float64(GD.variables['dive_list'][:]))
salin_avg = np.nanmean(df_s,axis=1)
theta_avg = np.nanmean(df_t,axis=1)
grid = np.float64(df_t.index)
grid_p = sw.pres(grid,26)
N2 = np.squeeze(sw.bfrq(salin_avg, theta_avg, grid_p, lat=26)[0])  
lz = np.where(N2 < 0)   
lnan = np.isnan(N2)
N2[lz] = 0 
N2[lnan] = 0
N = np.sqrt(N2)  
omega = 0  # frequency zeroed for geostrophic modes
mmax = 60  # highest baroclinic mode to be calculated
nmodes = mmax + 1  
G, Gz, c = vertical_modes(N2,-1*grid[1:],omega,mmax)

y_pos2 = 15
ax.plot3D(-20*np.ones(5),y_pos2*np.ones(5),np.linspace(-5000,0,5),color='k',zdir='z')
ax.plot3D(-20*np.ones(5),np.linspace(y_pos2-10,y_pos2+10,5),np.zeros(5),color='k',zdir='z')
ax.plot3D(-20*np.ones(np.size(grid[1:])),y_pos2 + 3*Gz[:,0],np.flipud(np.linspace(-5000,0,np.size(grid[1:]))),color='#6A5ACD',zdir='z',linewidth=0.75)
ax.plot3D(-20*np.ones(np.size(grid[1:])),y_pos2 + 3*Gz[:,1],np.flipud(np.linspace(-5000,0,np.size(grid[1:]))),color='#6A5ACD',zdir='z',linewidth=0.75)
ax.plot3D(-20*np.ones(np.size(grid[1:])),y_pos2 + 3*Gz[:,2],np.flipud(np.linspace(-5000,0,np.size(grid[1:]))),color='#6A5ACD',zdir='z',linewidth=0.75)
ax.plot3D(-20*np.ones(np.size(grid[1:])),y_pos2 + 3*Gz[:,3],np.flipud(np.linspace(-5000,0,np.size(grid[1:]))),color='b',zdir='z',linewidth=0.75)
ax.text(-20,y_pos2-20,0," G'(z) ")
# ax.text(-5,y_pos2-15,-5000," Decomposition ")
ax.text(-10,y_pos2-15,-5000,"Vertical Modes")

ax.set_xlim(-20, 100)
ax.set_ylim(0, 60)
ax.set_zlim(-5000, 100)

# ax.set_xlabel('X [km]')
# ax.set_ylabel('Y [km]')
ax.set_yticks([])
# ax.set_zlabel('Z [m]')
ax.set_title('Glider Transect, Cross-Track Velocity, Vertical Structure')

ax.view_init(30,-50)
# ax.grid('off')

# fig.savefig('/Users/jake/Documents/baroclinic_modes/generals/final_figures/schematic_v.png',dpi = 200)
# plt.close()
ax.grid()
plot_pro(ax)
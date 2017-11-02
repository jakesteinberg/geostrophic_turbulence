import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mode_decompositions import vertical_modes
import pandas as pd
import seawater as sw
from scipy.io import netcdf

fig = plt.figure()
ax = fig.gca(projection='3d')

y_pos = 40

# dives 
x_s = np.arange(0,100,25/21)
y_s = y_pos*np.ones(np.size(x_s))
z_s = np.array([0,-500,-1000,-1500,-2000,-2500,-3000,-3500,-4000,-4500,-5000,-4500,-4000,-3500,-3000,-2500,-2000,-1500,-1000,-500,0,
    0,-500,-1000,-1500,-2000,-2500,-3000,-3500,-4000,-4500,-5000,-4500,-4000,-3500,-3000,-2500,-2000,-1500,-1000,-500,0,
    0,-500,-1000,-1500,-2000,-2500,-3000,-3500,-4000,-4500,-5000,-4500,-4000,-3500,-3000,-2500,-2000,-1500,-1000,-500,0,
    0,-500,-1000,-1500,-2000,-2500,-3000,-3500,-4000,-4500,-5000,-4500,-4000,-3500,-3000,-2500,-2000,-1500,-1000,-500,0])

# dacs
x_q = np.arange(x_s[10],100,25)
y_q = y_pos*np.ones(np.size(x_q))
u_q = np.array([0, -1, 0, 2])
v_q = np.array([0, -5, 12, 8])
ax.scatter(x_q, y_q, np.zeros(np.size(x_q)),color='g',s=8)
ax.quiver(x_q, y_q, np.zeros(np.size(x_q)), u_q, v_q, 0*np.ones(np.size(x_q)),color='g', length=10, normalize=True)
ax.text(x_q[-1]+6,y_q[-1]+7,0,'DAC')

ax.plot3D(x_s, y_s, z_s,color='k',zorder=1,zdir='z',linewidth=0.75) 
ax.plot3D(x_s, y_s, np.zeros(np.size(x_s)),color='#696969',linestyle='--',zorder=1,zdir='z',linewidth=0.75) 

#### plot density contours
z_in = np.where( (z_s > -3100) & (z_s < -2900) ) 
ax.plot(x_s[z_in], y_s[z_in], z_s[z_in], color='r',linestyle='--',linewidth=0.75)
z_shift = [0, 500, 1000, 500, 0, 0, -500, 0]
x_shift = [x_s[z_in[0][0]], x_s[z_in[0][1]+1], x_s[z_in[0][2]-2], x_s[z_in[0][3]+1], x_s[z_in[0][4]], x_s[z_in[0][5]], x_s[z_in[0][6]+1], x_s[z_in[0][7]] ]  
z_d1 = z_s[z_in]+z_shift
ax.plot(x_shift, y_s[z_in], z_d1, color='r')

z_in2 = np.where( (z_s > -2100) & (z_s < -1900) ) 
ax.plot(x_s[z_in2], y_s[z_in2], z_s[z_in2], color='r',linestyle='--',linewidth=0.75)
z_shift = [0, 0, 500, 0, 0, 0, -500, 0]
x_shift = [x_s[z_in2[0][0]], x_s[z_in2[0][1]], x_s[z_in2[0][2]-1], x_s[z_in2[0][3]], x_s[z_in2[0][4]], x_s[z_in2[0][5]], x_s[z_in2[0][6]+1], x_s[z_in2[0][7]] ]  
z_d1 = z_s[z_in2]+z_shift
ax.plot(x_shift, y_s[z_in2], z_d1, color='r')

z_in2 = np.where( (z_s > -3600) & (z_s < -3400) ) 
ax.plot(x_s[z_in2], y_s[z_in2], z_s[z_in2], color='r',linestyle='--',linewidth=0.75)
z_shift = [0, 0, -500, 0, 0, 0, -500, 0]
x_shift = [x_s[z_in2[0][0]], x_s[z_in2[0][1]], x_s[z_in2[0][2]+1], x_s[z_in2[0][3]], x_s[z_in2[0][4]], x_s[z_in2[0][5]], x_s[z_in2[0][6]+1], x_s[z_in2[0][7]] ]  
z_d1 = z_s[z_in2]+z_shift
ax.plot(x_shift, y_s[z_in2], z_d1, color='r')

z_in3 = np.where( (z_s > -1100) & (z_s < -900) ) 
ax.plot(x_s[z_in3], y_s[z_in3], z_s[z_in3], color='r',linestyle='--',linewidth=0.75)
z_shift = [0, 0, 0, 0, 0, 0, -500, 0]
x_shift = [x_s[z_in3[0][0]], x_s[z_in3[0][1]], x_s[z_in3[0][2]], x_s[z_in3[0][3]], x_s[z_in3[0][4]], x_s[z_in3[0][5]], x_s[z_in3[0][6]+1], x_s[z_in3[0][7]] ]  
z_d1 = z_s[z_in3]+z_shift
ax.plot(x_shift, y_s[z_in3], z_d1, color='r')

# eta
z_g = np.flipud(np.linspace(-5000,0,200))
ax.plot3D(-10*np.ones(5),y_pos*np.ones(5),np.linspace(-5000,0,5),color='k',zdir='z')
ax.plot3D(-10*np.ones(5),np.linspace(y_pos-10,y_pos+10,5),np.zeros(5),color='k',zdir='z')
ax.plot3D(-10*np.ones(np.size(z_g)),y_pos + 5*np.sin( (2*np.pi/2000)*z_g ),np.flipud(np.linspace(-5000,0,np.size(z_g))),color='#800000',zdir='z')
ax.text(-10,y_pos+10,20,'v(z)')

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
ax.plot3D(-10*np.ones(5),y_pos2*np.ones(5),np.linspace(-5000,0,5),color='k',zdir='z')
ax.plot3D(-10*np.ones(5),np.linspace(y_pos2-10,y_pos2+10,5),np.zeros(5),color='k',zdir='z')
ax.plot3D(-10*np.ones(np.size(grid[1:])),y_pos2 + .005*G[:,1],np.flipud(np.linspace(-5000,0,np.size(grid[1:]))),color='#6A5ACD',zdir='z',linewidth=0.75)
ax.plot3D(-10*np.ones(np.size(grid[1:])),y_pos2 + .005*G[:,2],np.flipud(np.linspace(-5000,0,np.size(grid[1:]))),color='#6A5ACD',zdir='z',linewidth=0.75)
ax.plot3D(-10*np.ones(np.size(grid[1:])),y_pos2 + .005*G[:,3],np.flipud(np.linspace(-5000,0,np.size(grid[1:]))),color='#6A5ACD',zdir='z',linewidth=0.75)
# ax.plot3D(-10*np.ones(np.size(grid[1:])),y_pos2 + 3*Gz[:,3],np.flipud(np.linspace(-5000,0,np.size(grid[1:]))),color='b',zdir='z',linewidth=0.75)
ax.text(-5,y_pos2,-5000," G(z) ")
ax.text(-5,y_pos2-15,-5000," Decomposition ")
ax.text(10,y_pos2-15,-5000,"into Vertical Modes")

ax.text(-10,y_pos+10,0,'\eta (z)')

ax.set_xlim(-10, 90)
ax.set_ylim(0, 60)
ax.set_zlim(-5000, 100)

ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [m]')
ax.set_title('Data Collection Method and Vertical Structure')

ax.view_init(30,-50)
# ax.grid('off')

plt.show()
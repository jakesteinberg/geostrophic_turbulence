# Map of locations to be used in research 

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.set_extent([-160, -50, 0, 50])
ax.gridlines(draw_labels=True, linewidth=.5, color='#808080', alpha=.75, linestyle='-', zorder=2)
# ABACO
ax.plot([-77, -75], [26.5, 26.5], color='g', linestyle='-', transform=ccrs.PlateCarree())
ax.text(-74,25.5,'ABACO',color='k',fontsize=10, transform=ccrs.PlateCarree())
# ARGO
ax.scatter([-74], [24], color='r',s=7, transform=ccrs.PlateCarree(),zorder=3)
ax.text(-71,23,'Deep Argo',color='k',fontsize=10, transform=ccrs.PlateCarree())
# ARGO
ax.scatter([-59], [26], color='r',s=7, transform=ccrs.PlateCarree(),zorder=3)
# BATS 
ax.scatter([-64.2], [31.6], color='k',s=7, transform=ccrs.PlateCarree(),zorder=3)
ax.text(-63.2,30.6,'BATS',color='k',fontsize=10, transform=ccrs.PlateCarree())
# PAPA 
ax.scatter([-145], [50], color='k',s=7, transform=ccrs.PlateCarree(),zorder=3)
ax.text(-144,49,'PAPA',color='k',fontsize=10, transform=ccrs.PlateCarree())
# HOTS 
ax.scatter([-158], [22.75], color='k',s=7, transform=ccrs.PlateCarree(),zorder=3)
ax.text(-157,21.75,'ALOHA',color='k',fontsize=10, transform=ccrs.PlateCarree())

ax.text(-0.07, 0.55, 'latitude', va='bottom', ha='center',
         rotation='vertical', rotation_mode='anchor',
         transform=ax.transAxes)
ax.text(0.5, -0.2, 'longitude', va='bottom', ha='center',
         rotation='horizontal', rotation_mode='anchor',
         transform=ax.transAxes)


fig.savefig('/Users/jake/Desktop/abaco/map_2.png',dpi = 300)

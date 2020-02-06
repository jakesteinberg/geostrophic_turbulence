% Live Ocean schematic 

addpath /Users/jake/Documents/seaglider/glider_flight_regressions_jun19/
a = ncload('/Users/jake/Documents/baroclinic_modes/Model/mooring/v_slice.nc');
%% glider track 
dg_z = 0:-5:-2800;
dg_gs = 1/3;
% first dive 
dg_x = nan * ones(length(dg_z), 1);
dg_x(1) = 0;
for i = 2:length(dg_z)
    dg_x(i) = dg_x(i-1) + -1*(dg_z(i) - dg_z(i-1))/dg_gs; 
end
% first climb
dg_x_c = nan * ones(length(dg_z), 1);
dg_x_c(end) = dg_x(end);
for i = (length(dg_z)-1):-1:1
    dg_x_c(i) = dg_x_c(i+1) + 1*(dg_z(i) - dg_z(i+1))/dg_gs; 
end
dce = dg_x_c(1);
dg_pos = [dg_x, dg_x_c, dg_x+dce, dg_x_c+dce, dg_x+2*dce, dg_x_c+2*dce]; % dive climb
dg_pos_lon = dg_pos/(1852*60*cosd(43)) + -128;
%%
bot_wm = a.bottom_val;
bot_wm(isnan(bot_wm)) = 10;
% slice_lon = a.x_grid/(1852*60*cosd(43.9)) + nanmin(nanmin(a.bottom_lon));
colomap = [gray(100); [143, 188, 143]/255];
addpath /Users/jake/Documents/MATLAB/cmocean_v2.0/cmocean/
ccmap = cmocean('balance');

figure('units','inches','position',[1 1 15.5 9]);
ax1 = axes;
s = surf(ax1, a.bottom_lon, a.bottom_lat, bot_wm);
s.EdgeColor = 'none';
colormap(colomap)
caxis([-4000, 0])
ylim([43, 50])
xlim([-122-( (50-43) /cosd(32)) -122])
zlim([-4500, 500])
xlabel('Longitude [^{\circ} E]', 'fontsize', 14)
ylabel('Latitude [^{\circ} N]', 'fontsize', 13)
zlabel('z [m]')
title('LiveOcean subdomain and extracted transect', 'fontsize', 15)
view([-40, 30])

ax2 = axes; 
v_scale = (a.v_vel')/100 + 43.9; 
C = (v_scale - nanmin(nanmin(v_scale))) / (nanmax(nanmax(v_scale)) - nanmin(nanmin(v_scale))); 
s2 = surf(ax2, repmat(a.lon_grid', length(a.depth(1, :)), 1), (a.v_vel')/1000 + 44, a.depth', C);
hold on
col_d = [255,20,147]/255;
scatter3(dg_pos_lon(:, 1), 43.8*ones(size(dg_pos_lon(:,1))), dg_z' + 60, 2, col_d) 
scatter3(dg_pos_lon(:, 2), 43.8*ones(size(dg_pos_lon(:,2))), dg_z' + 60, 2, col_d)
scatter3(dg_pos_lon(:, 3), 43.8*ones(size(dg_pos_lon(:,1))), dg_z' + 60, 2, col_d) 
scatter3(dg_pos_lon(:, 4), 43.8*ones(size(dg_pos_lon(:,2))), dg_z' + 60, 2, col_d)
scatter3(dg_pos_lon(:, 5), 43.8*ones(size(dg_pos_lon(:,1))), dg_z' + 60, 2, col_d) 
scatter3(dg_pos_lon(:, 6), 43.8*ones(size(dg_pos_lon(:,2))), dg_z' + 60, 2, col_d)
shading interp
colorbar
ylim([43, 50])
xlim([-122-( (50-43) /cosd(32)) -122])
zlim([-4500, 500])
view([-40, 30])
%%Link them together
% linkaxes([ax1,ax2])
%%Hide the top axes
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];
%%Give each one its own colormap
colormap(ax1, colomap)
colormap(ax2, ccmap)
%%Then add colorbars and get everything lined up
set([ax1,ax2],'Position',[.17 .11 .685 .815]);
cb1 = colorbar(ax1,'Position',[.05 .11 .03 .815]);
cb1.Label.String = 'z [m]';
cb1.FontWeight = 'bold';
cb1.FontSize = 12;
cb1.Color = 'k';
cb2 = colorbar(ax2,'Position',[.88 .11 .03 .815]);
cb2.Label.String = 'Northward Velocity, v [m s^{-1}]';
cb2.TickLabels = -.35:.07:.35;
cb2.FontWeight = 'bold';
cb2.FontSize = 12;
cb2.Color = 'k';
print -djpeg -r450 ~/Documents/glider_flight_sim_paper/reviewer_comments_minor_revisions/LiveOcean_domain_kh.jpeg
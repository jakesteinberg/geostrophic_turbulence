import sys
import numpy as np
import pandas as pd 
from scipy.io import netcdf
import seawater as sw 

def make_bin(bin_depth,depth_d,depth_c,temp_d,temp_c,salin_d,salin_c, x_g_d, x_g_c, y_g_d, y_g_c):
    bin_up = bin_depth[0:-2]
    bin_down = bin_depth[2:]
    bin_cen = bin_depth[1:-1] 
    temp_g_dive = np.empty(np.size(bin_cen))
    temp_g_climb = np.empty(np.size(bin_cen))
    salin_g_dive = np.empty(np.size(bin_cen))
    salin_g_climb = np.empty(np.size(bin_cen))
    x_g_dive = np.empty(np.size(bin_cen))
    x_g_climb = np.empty(np.size(bin_cen))
    y_g_dive = np.empty(np.size(bin_cen))
    y_g_climb = np.empty(np.size(bin_cen))
    for i in range(np.size(bin_cen)):
        dp_in_d = (depth_d > bin_up[i]) & (depth_d < bin_down[i])
        dp_in_c = (depth_c > bin_up[i]) & (depth_c < bin_down[i])
        
        if dp_in_d.size > 2:
            temp_g_dive[i] = np.nanmean(temp_d[dp_in_d])
            salin_g_dive[i] = np.nanmean(salin_d[dp_in_d])
            x_g_dive[i] = np.nanmean(x_g_d[dp_in_d])/1000
            y_g_dive[i] = np.nanmean(y_g_d[dp_in_d])/1000
        if dp_in_c.size > 2:    
            temp_g_climb[i] = np.nanmean(temp_c[dp_in_c])       
            salin_g_climb[i] = np.nanmean(salin_c[dp_in_c])
            x_g_climb[i] = np.nanmean(x_g_c[dp_in_c])/1000
            y_g_climb[i] = np.nanmean(y_g_c[dp_in_c])/1000
    return(temp_g_dive, temp_g_climb, salin_g_dive, salin_g_climb, x_g_dive, x_g_climb, y_g_dive, y_g_climb)
    
    
def test_bin(A):
    return(np.shape(A))    
    
    
def collect_dives(dg_list, bin_depth, grid, grid_p, ref_lat):
    # initial arrays and dataframes 
    df_t = pd.DataFrame()
    df_s = pd.DataFrame()
    df_den = pd.DataFrame()
    df_lon = pd.DataFrame()
    df_lat = pd.DataFrame()
    heading_rec = []
    time_rec = []
    time_rec_2 = []
    dac_u = []
    dac_v = []
    # time_rec_2 = np.zeros([np.size(dg_list), 2])
    
    secs_per_day = 86400.0
    datenum_start = 719163 # jan 1 1970
    
    ## loop over each dive 
    count_st = 14
    count = count_st
    for i in dg_list[count_st:]:
        dive_nc_file = netcdf.netcdf_file(i,'r',mmap=False if sys.platform == 'darwin' else mmap, version=1)
        # initial dive information 
        glid_num = dive_nc_file.glider 
        dive_num = dive_nc_file.dive_number 
        heading_ind = dive_nc_file.variables['log_MHEAD_RNG_PITCHd_Wd']   
    
        # extract heading 
        h1 = heading_ind.data[0]
        h_test_0 = heading_ind.data[1]
        h_test_1 = heading_ind.data[2]
        if h_test_0.isdigit() == True:
            if h_test_1.isdigit() == True:
                h2 = h_test_0
                h3 = h_test_1
                h = int(h1+h2+h3)
            else:
                h = int(h1+h2) 
        else:
            h = int(h1)   
        
        if np.sum(heading_rec) < 1:
            heading_rec = np.concatenate([ [h], [h] ])
        else:
            heading_rec = np.concatenate([ heading_rec, [h], [h] ])                  
    
        # dive position                                       
        lat = dive_nc_file.variables['latitude'][:]
        lon = dive_nc_file.variables['longitude'][:] 
    
        # eng     
        ctd_epoch_time = dive_nc_file.variables['ctd_time'][:]
        pitch_ang = dive_nc_file.variables['eng_pitchAng'][:]
    
        # science 
        press = dive_nc_file.variables['ctd_pressure'][:]
        temp = dive_nc_file.variables['temperature'][:]
        salin = dive_nc_file.variables['salinity'][:]    
        theta = sw.ptmp(salin, temp, press, 0)
        depth = sw.dpth(press,ref_lat)
    
        # time conversion 
        dive_start_time = dive_nc_file.start_time
        ctd_time = ctd_epoch_time - dive_start_time    
    
        # put on vertical grid (1 column per profile)
        max_d = np.where(depth == depth.max())
        max_d_ind = max_d[0]
        pitch_ang_sub1 = pitch_ang[0:max_d_ind[0]]
        pitch_ang_sub2 = pitch_ang[max_d_ind[0]:]
        dive_mask_i= np.where(pitch_ang_sub1 < 0)
        dive_mask = dive_mask_i[0][:]
        climb_mask_i = np.where(pitch_ang_sub2 > 0)
        climb_mask = climb_mask_i[0][:] + max_d_ind[0] - 1      
    
        # dive/climb time midpoints
        time_dive = ctd_time[dive_mask]
        time_climb = ctd_time[climb_mask]
    
        serial_date_time_dive = datenum_start + dive_start_time/(60*60*24) + np.median(time_dive)/secs_per_day
        serial_date_time_climb = datenum_start + dive_start_time/(60*60*24) + np.median(time_climb)/secs_per_day
        serial_date_time_dive_2 = datenum_start + dive_start_time/(60*60*24) + np.nanmin(time_dive)/secs_per_day
        serial_date_time_climb_2 = datenum_start + dive_start_time/(60*60*24) + np.nanmax(time_climb)/secs_per_day
        # time_rec_2[count,:] = np.array([ serial_date_time_dive, serial_date_time_climb ])
        # time_rec_2[count,:] = np.array([ serial_date_time_dive_2, serial_date_time_climb_2 ])
        if np.sum(time_rec) < 1:
            time_rec = np.concatenate([ [serial_date_time_dive], [serial_date_time_climb] ])
            time_rec_2 = np.concatenate([ [serial_date_time_dive_2], [serial_date_time_climb_2] ])
            dac_u = np.concatenate([ [dive_nc_file.variables['depth_avg_curr_east'].data], [dive_nc_file.variables['depth_avg_curr_east'].data] ])
            dac_v = np.concatenate([ [dive_nc_file.variables['depth_avg_curr_north'].data], [dive_nc_file.variables['depth_avg_curr_north'].data] ])
        else:
            time_rec = np.concatenate([ time_rec, [serial_date_time_dive], [serial_date_time_climb] ])
            time_rec_2 = np.concatenate([ time_rec_2, [serial_date_time_dive_2], [serial_date_time_climb_2] ])
            dac_u = np.concatenate([ dac_u,[dive_nc_file.variables['depth_avg_curr_east'].data],[dive_nc_file.variables['depth_avg_curr_east'].data] ])
            dac_v = np.concatenate([ dac_v,[dive_nc_file.variables['depth_avg_curr_north'].data],[dive_nc_file.variables['depth_avg_curr_north'].data] ])
                
        # interpolate (bin_average) to smooth and place T/S on vertical depth grid 
        temp_grid_dive, temp_grid_climb, salin_grid_dive, salin_grid_climb, lon_grid_dive, lon_grid_climb, lat_grid_dive, lat_grid_climb = make_bin(bin_depth,
            depth[dive_mask],depth[climb_mask],temp[dive_mask],temp[climb_mask],salin[dive_mask],salin[climb_mask], lon[dive_mask]*1000,lon[climb_mask]*1000,
            lat[dive_mask]*1000,lat[climb_mask]*1000)
    
        den_grid_dive = sw.pden(salin_grid_dive, temp_grid_dive, grid_p, pr=0) - 1000
        den_grid_climb = sw.pden(salin_grid_climb, temp_grid_climb, grid_p, pr=0) - 1000
        theta_grid_dive = sw.ptmp(salin_grid_dive, temp_grid_dive, grid_p, pr=0)
        theta_grid_climb = sw.ptmp(salin_grid_climb, temp_grid_climb, grid_p, pr=0)
    
        # create dataframes where each column is a profile 
        t_data_d = pd.DataFrame(theta_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
        t_data_c = pd.DataFrame(theta_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
        s_data_d = pd.DataFrame(salin_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
        s_data_c = pd.DataFrame(salin_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
        den_data_d = pd.DataFrame(den_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
        den_data_c = pd.DataFrame(den_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
        lon_data_d = pd.DataFrame(lon_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
        lon_data_c = pd.DataFrame(lon_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
        lat_data_d = pd.DataFrame(lat_grid_dive,index=grid,columns=[glid_num*1000 + dive_num])
        lat_data_c = pd.DataFrame(lat_grid_climb,index=grid,columns=[glid_num*1000 + dive_num+.5])
    
        if df_t.size < 1:
            df_t = pd.concat([t_data_d,t_data_c],axis=1)
            df_s = pd.concat([s_data_d,s_data_c],axis=1)
            df_den = pd.concat([den_data_d,den_data_c],axis=1)
            df_lon = pd.concat([lon_data_d,lon_data_c],axis=1)
            df_lat = pd.concat([lat_data_d,lat_data_c],axis=1)
        else:
            df_t = pd.concat([df_t,t_data_d,t_data_c],axis=1)
            df_s = pd.concat([df_s,s_data_d,s_data_c],axis=1)
            df_den = pd.concat([df_den,den_data_d,den_data_c],axis=1)
            df_lon = pd.concat([df_lon,lon_data_d,lon_data_c],axis=1)
            df_lat = pd.concat([df_lat,lat_data_d,lat_data_c],axis=1)
 
        count = count+1
    
    
    
    
    return(df_t, df_s, df_den, df_lon, df_lat, dac_u, dac_v, time_rec, time_rec_2, heading_rec)        
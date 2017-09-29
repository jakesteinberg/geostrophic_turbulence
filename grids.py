import numpy as np

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
    
    
        
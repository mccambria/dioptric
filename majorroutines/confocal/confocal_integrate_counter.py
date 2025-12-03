# Author: Eric Gediman
# Simple program that integrates total counts for a nv with provided center coordinates

import utils.data_manager as dm
import numpy as np

def calculate_totalcount(data):
    radius = 2 #radius in pixel of nv center

    nv_centers = np.array([[-0.013, 0.003]]) # Add your NV center coordinates here as [[x1,y1],[x2,y2],...]
    nv_centers_counts = np.zeros(len(nv_centers)) # Will hold total counts for each nv center
    x0 = np.zeros(len(nv_centers))
    y0 = np.zeros(len(nv_centers))
    for k in range(len(nv_centers)):
        x0[k] = len(data['x_coords_1d']) -1 - int((np.abs(data['x_coords_1d'] - nv_centers[k][0])).argmin())
        y0[k] = len(data['y_coords_1d']) -1 - int((np.abs(data['y_coords_1d'] - nv_centers[k][1])).argmin())    
    print(x0,y0)
    print(int(data['img_array'][int(y0[0])][int(x0[0])]))
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            for k in range(len(nv_centers)):
                nv_centers_counts[k] += int(data['img_array'][int(y0[k])+i][int(x0[k])+j])
    return nv_centers_counts
               
if __name__ == "__main__":
    file_name = "2025_12_02-03_06_44-(Rubin)"
    data = dm.get_raw_data(file_name)
    print("Total counts for each NV center:", calculate_totalcount(data))
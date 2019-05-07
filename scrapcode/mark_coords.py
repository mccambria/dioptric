# -*- coding: utf-8 -*-
"""
Mark a list of coordinates on an image 

Created on Tue May  7 09:46:42 2019

@author: mccambria
"""

import utils.tool_belt as tool_belt
import json
import numpy
import cv2

def voltages_to_coords(x_scan_voltages, y_scan_voltages,
                       x_voltage, y_voltage):
    x_voltage = numpy.asarray(x_voltage)
    y_voltage = numpy.asarray(y_voltage)
    x_coord = (numpy.abs(x_scan_voltages - x_voltage)).argmin()
    y_coord = (numpy.abs(y_scan_voltages - y_voltage)).argmin()
    return x_coord, y_coord

nv0 = [0.005, 0.017, 49.6]
nv1 = [0.000, 0.100, 49.8]
nv2 = [-0.021, 0.019, 49.7]
nv3 = [-0.027, -0.041, 49.8]
nv4 = [-0.070, -0.035, 49.9]
nv5 = [-0.101, -0.032, 49.7]
nv6 = [-0.057, 0.084, 49.7]
nv7 = [-0.067, 0.062, 49.7]
nv8 = [-0.062, 0.128, 49.6]
nv9 = [-0.162, 0.082, 49.7]
nv10 = [-0.053, 0.111, 49.7]
nv11 = [-0.044, 0.102, 49.7]
nv12 = [-0.183, 0.131, 49.7]
nv13 = [-0.166, 0.135, 49.7]
nv14 = [0.075, 0.188, 49.5]
nv15 = [0.092, 0.190, 49.4]
nv_list = [nv0, nv1, nv2, nv3, nv4, nv5, nv6, nv7, nv8,
           nv9, nv10, nv11, nv12, nv13, nv14, nv15]

directory = 'E:\\Team Drives\\Kolkowitz Lab Group\\nvdata\\image_sample\\'
file_name = '2019-05-06_17-12-50_ayrton12.txt'
file_path = directory + file_name

with open(file_path, 'r') as file:
    data = json.load(file)
    
img_array = numpy.array(data['img_array'])
x_voltages = data['x_voltages']
y_voltages = data['y_voltages']

# Convert to kcps
#readout_sec = data['readout'] / 10**9
#img_array = (img_array / 1000) / readout_sec

# Convert to 8 bit
img = numpy.copy(img_array)
img = img.astype(numpy.float64)
img -= numpy.nanmin(img)  # Set the lowest value to 0
img *= (255/numpy.nanmax(img))
img = img.astype(numpy.uint8)

# Convert to rgb
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
# Set up axes
x_low = x_voltages[0]
x_high = x_voltages[-1]

y_low = y_voltages[0]
y_high = y_voltages[-1]

pixel_size = (x_low - x_high) / len(x_voltages)

half_pixel_size = pixel_size / 2
img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
              y_low - half_pixel_size, y_high + half_pixel_size]

# Show the image
fig = tool_belt.create_image_figure(img_array, img_extent)

for nv in nv_list:
    input()
    x_coord, y_coord = voltages_to_coords(x_voltages, y_voltages, nv[0], nv[1])
    img[-y_coord, -x_coord] = (255, 0, 0)
    img_array[-y_coord-1, -x_coord-1] = 10000
    tool_belt.update_image_figure(fig, img_array)
    
    
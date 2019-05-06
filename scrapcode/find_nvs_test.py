# -*- coding: utf-8 -*-
"""
Here is how it works.
1) Load image raw data from file and convert to 8 bits (0-255).
2) Run a Gaussian blur to remove high frequency noise. Set the kernel size
    based on the expected 1/e2 diameter of a diffraction limited spot.
3) Run Canny edge detection. The definite edge value is set empirically.
    The hysteresis value is also set empirically, although it should be high
    (> 0.5) since the edge should have just about the same gradient all
    the way around. Lowering this value could allow for better detection of
    overlapping NVs. ApertureSize is set to 7 empirically - I do not totally
    know what it does. Setting L2gradient to True gives a more accurate
    gradient.
4) Turn connected edges into discrete objects called contours in opencv
    parlance.
5) Filter out contours that are open or have holes. This leaves 'blobs'.
6) Filter our blobs that do not have valid areas. I am not sure why some blobs
    do not have valid areas.
7) Fit a minimum enclosing circle to each blob.
8) Filter out blobs that are not adequately round. Roundness is measured by
    the fraction of actual area over fit circle area.
9) Filter out blobs whose fit circle radii are out of the min/max allowed
    radii range. This range is defined based on scaling up/down the
    expected 1/e2 radius of a diffraction limited spot.
10) Get the centers of the fit circles.

Created on Thu May  2 19:51:46 2019

@author: mccambria
"""

import cv2
import numpy
from matplotlib import pyplot as plt
import json

####################### Files #######################

#directory = 'G:\\Team Drives\\Kolkowitz Lab Group\\nvdata\\image_sample\\'
directory = 'E:\\Team Drives\\Kolkowitz Lab Group\\nvdata\\image_sample\\'

file_name = '2019-04-29_16-37-06_ayrton12.txt'
#file_name = '2019-04-29_16-37-56_ayrton12.txt'

#file_name = '2019-04-29_16-19-11_ayrton12.txt'
#file_name = '2019-04-30_14-45-29_ayrton12.txt'

#file_name = '2019-04-29_15-33-39_ayrton12.txt'
#file_name = '2019-05-01_15-55-20_ayrton12.txt'
file_path = directory + file_name

####################### Parameters #######################

diff_lim_spot_diam = 0.015  # volts
gaussian_kernel_frac = 1/2
# edge_rate = 2000e3  # counts per volts maybe??
edge_rate = 1000e3  # counts per volts maybe??
# edge_rate = 200e3  # counts per volts maybe??
# edge_rate = 150e3  # counts per volts maybe??
# edge_rate = 130e3  # counts per volts maybe??
# edge_rate = 100e3  # counts per volts maybe??
# canny_low_scaling = 0.5  # A number between 0 and 1
canny_low_scaling = 0.00001  # A number between 0 and 1
roundness = 0.55
valid_radius_range = [0, 2/3]

#######################

with open(file_path, 'r') as file:
    data = json.load(file)

img_array = numpy.array(data['img_array'])
# img_array = img_array[0:75, 75:150]

# Convert to 8 bit
# img = (img_array / 1000) / (readout / 10**9)
img = numpy.copy(img_array)
img = img.astype(numpy.float64)
img -= numpy.nanmin(img)  # Set the lowest value to 0
img *= (255/numpy.nanmax(img))
img = img.astype(numpy.uint8)
contour_img = numpy.copy(img)
contour_img[:] = 0

# Determine the pixel size
# x_voltages = data['x_voltages']
# y_voltages = data['y_voltages']
x_range = data['x_range']
y_range = data['y_range']
min_range = min(x_range, y_range)
num_steps = data['num_steps']
volts_per_pixel = min_range / num_steps

# Convert to pixels
diff_lim_spot_diam /= volts_per_pixel
low = valid_radius_range[0]
high = valid_radius_range[1]
valid_radius_range = [low * diff_lim_spot_diam, high * diff_lim_spot_diam]

# Calculate the Gaussian kernel size
# The Gaussian kernel should just about fit in the dimmest NV
# we want to find
gaussian_kernel_size = gaussian_kernel_frac * diff_lim_spot_diam
print(gaussian_kernel_size)
if gaussian_kernel_size < 3:
    print('The resolution is too low for accurate analysis.\n'
          'volts per pixel: {}'.format(volts_per_pixel))
    gaussian_kernel_size = 3
else:
    floor_gaussian_kernel_size = int(gaussian_kernel_size)
    # Round to the nearest odd integer since the kernel size must be odd
    if floor_gaussian_kernel_size % 2 == 0:
        gaussian_kernel_size = floor_gaussian_kernel_size + 1
    else:
        gaussian_kernel_size = floor_gaussian_kernel_size

gaussian_kernel = (gaussian_kernel_size, gaussian_kernel_size)
img = cv2.GaussianBlur(img, gaussian_kernel, 0)

# Calculate the hystersis values for the edge detection
# Again, this is empirical and based on the pixel size
# The min value for the hystersis should be very near the max since a
# the shapes we are looking for (~gaussian discs) should have radial symmetry
canny_high = int(edge_rate * volts_per_pixel)
canny_low = int(canny_high * canny_low_scaling)

edges = cv2.Canny(img, canny_high, canny_low,
                  apertureSize=7, L2gradient=True)

# Turn connected edges into discrete objects - contours in opencv parlance
ret_vals = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = ret_vals
hierarchy = hierarchy[0]  # Not sure why the actual hierarchy list is embedded

img_array = numpy.copy(img)

# Convert to rgb so we can plot centers in color
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Clean up the contour list - only keep circles below a certain size
# Keep a contour if it has no children (ie it is innermost) and
# has a parent (ie it is contained within another contour). findContours
# finds internal and external lines around closed loops in the image
# so this criteria selects only the internal lines. It can be fooled, but it
# should work for our edge-processed images.
for ind in range(len(contours)):
    hier = hierarchy[ind]
    cnt = contours[ind]
    child_cnt_id = hier[2]
    parent_cnt_id = hier[3]
    if (child_cnt_id == -1) and (parent_cnt_id != -1):
        # Filter out non-circular shapes
        area = cv2.contourArea(cnt)
        circle_center, circle_radius = cv2.minEnclosingCircle(cnt)
        circle_area = numpy.pi * circle_radius**2
        if (area == 0) or (circle_area == 0):
            continue
        if area / circle_area <= roundness:
            continue
        if circle_radius < valid_radius_range[0]:
            continue
        if (circle_radius > valid_radius_range[1]):
            continue
        circle_x = int(circle_center[0])
        circle_y = int(circle_center[1])
        # Plot the centroids of the contours
        img[circle_y, circle_x] = (255, 0, 0)

####################### Plotting #######################

plot_mode = 1
if plot_mode == 1:
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img)
elif plot_mode == 2:
    fig, axes_pack = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes_pack[0]
    ax.imshow(img)
    ax = axes_pack[1]
    ax.imshow(edges, cmap='gray')
elif plot_mode == 3:
    fig, axes_pack = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes_pack[0]
    ax.imshow(img_array, cmap='gray')
    ax = axes_pack[1]
    ax.imshow(img, cmap='gray')
    ax = axes_pack[2]
    ax.imshow(edges, cmap='gray')

fig.show()
fig.tight_layout()
# Maximize the window
fig_manager = plt.get_current_fig_manager()
fig_manager.window.showMaximized()

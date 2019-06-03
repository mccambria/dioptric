# -*- coding: utf-8 -*-
"""
Gaussian template matching.

Created on Mon May  6 15:31:59 2019

@author: mccambria
"""

####################### Imports #######################

import cv2
import numpy
from matplotlib import pyplot as plt
import json
import sys
import utils.image_processing as image_processing

####################### Files #######################

directory = 'G:\\Team Drives\\Kolkowitz Lab Group\\nvdata\\image_sample\\'
# directory = 'E:\\Team Drives\\Kolkowitz Lab Group\\nvdata\\image_sample\\'

file_name = '2019-04-29_16-37-06_ayrton12.txt'
#file_name = '2019-04-29_16-37-56_ayrton12.txt'

#file_name = '2019-04-29_16-19-11_ayrton12.txt'
#file_name = '2019-04-30_14-45-29_ayrton12.txt'

#file_name = '2019-04-29_15-33-39_ayrton12.txt'
#file_name = '2019-05-01_15-55-20_ayrton12.txt'
file_path = directory + file_name

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

####################### Parameters #######################

diff_lim_spot_diam = 0.015  # volts

####################### Read the file #######################

with open(file_path, 'r') as file:
    data = json.load(file)

x_range = data['x_range']
y_range = data['y_range']
min_range = min(x_range, y_range)
num_steps = data['num_steps']
volts_per_pixel = min_range / num_steps

img_array = numpy.array(data['img_array'])
img_array = image_processing.convert_to_8bit(img_array)

img_array = cv2.GaussianBlur(img_array, (5, 5), 0)

# fig, ax = plt.subplots(figsize=(5,5))
# ax.imshow(img_array)
# sys.exit()

method = eval(methods[3])

#######################
print(volts_per_pixel)
diff_lim_spot_diam = int(diff_lim_spot_diam / volts_per_pixel)
print(diff_lim_spot_diam)
template = cv2.getGaussianKernel(diff_lim_spot_diam, diff_lim_spot_diam)
print(template)
sys.exit()
res = cv2.matchTemplate(img_array, template, method)

####################### Gaussian #######################

gaussian_size = 10
single_axis = numpy.linspace(-1, 1, gaussian_size)
x, y = numpy.meshgrid(single_axis, single_axis)

pixel_distances = numpy.sqrt(x**2 + y**2)



gaussian = numpy.zeros([gaussian_size, gaussian_size])


####################### Plotting #######################

plot_mode = 1
if plot_mode == 1:
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img_array)
elif plot_mode == 2:
    fig, axes_pack = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes_pack[0]
    ax.imshow(img_array)
    ax = axes_pack[1]
    ax.imshow(res)
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

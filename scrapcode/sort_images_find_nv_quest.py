# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:07:15 2019

@author: mccambria
"""

import os
import shutil
import json
import utils.tool_belt as tool_belt
import numpy

mapping = {
        '2019-05-29_15-08-53_ayrton12': '04-29_center',
        '2019-05-29_15-25-13_ayrton12': '04-29_1right',
        '2019-05-29_15-40-05_ayrton12': '04-29_1left',
        '2019-05-29_16-04-13_ayrton12': '05-10_1left',
        '2019-05-29_16-17-55_ayrton12': '05-10_center',
        '2019-05-29_16-31-13_ayrton12': '05-10_1right',
        '2019-05-29_16-43-26_ayrton12': '05-10_2right',
        '2019-05-29_16-59-55_ayrton12': '05-10_3right',
        '2019-05-29_16-47-50_ayrton12': '04-29_2right',
        '2019-05-29_17-01-21_ayrton12': '04-29_3right',
        }

for key in mapping:

    # Move and rename raw data
    if False:
        source_dir = 'G:\\Team Drives\\Kolkowitz Lab Group\\nvdata\\image_sample\\'
        dest_dir = 'C:\\Users\\Matt\\Desktop\\lost_nv_quest'
        source_file_path = '{}\\{}.txt'.format(source_dir, key)
        shutil.copy(source_file_path, dest_dir)
        dest_file_path = '{}\\{}.txt'.format(dest_dir, key)
        new_dest_file_path = '{}\\{}.txt'.format(dest_dir, mapping[key])
        os.rename(dest_file_path, new_dest_file_path)

    # Create svgs scaled to 100
    if True:
        file_dir = 'C:\\Users\\Matt\\Desktop\\lost_nv_quest'
        file_path = '{}\\{}.txt'.format(file_dir, mapping[key])

        with open(file_path) as file:
            data = json.load(file)

        x_voltages = data['x_voltages']
        y_voltages = data['y_voltages']
        img_array = data['img_array']
        readout = data['readout']

        x_low = x_voltages[0]
        x_high = x_voltages[-1]
        y_low = y_voltages[0]
        y_high = y_voltages[-1]

        half_pixel_size = (x_voltages[1] - x_voltages[0]) / 2
        img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                      y_low - half_pixel_size, y_high + half_pixel_size]

        img_array = numpy.array(img_array)
        img_array_kcps = (img_array / 1000) / (readout / 10**9)
        fig = tool_belt.create_image_figure(img_array_kcps, img_extent)

        # Clip past 100
        axes = fig.get_axes()
        ax = axes[0]
        images = ax.get_images()
        img = images[0]
        img.set_clim(None, 100)

        # fig.set_size_inches(10, 10)

        img_file_path = '{}\\{}'.format(file_dir, mapping[key])
        tool_belt.save_figure(fig, img_file_path)

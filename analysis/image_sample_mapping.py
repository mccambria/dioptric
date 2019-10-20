# -*- coding: utf-8 -*-
"""Saves the data necessary to relocate specific NVs. Also can probably (?)
illustrate a mapping from an NV list to an image_sample.

Created on Mon Jun 10 13:54:07 2019

@author: mccambria
"""


# %% Imports


import json
import majorroutines.image_sample as image_sample
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
import os
import labrad
from pathlib import Path


# %% Functions


def illustrate_mapping(file_name, nv_indices=None):

    data = tool_belt.get_raw_data(__file__, file_name)
    image_sample_file_name = data['image_sample_file_name']
    nv_sig_list = data['nv_sig_list']

    fig = image_sample.create_figure(image_sample_file_name)
    axes = fig.get_axes()
    ax = axes[0]
    images = ax.get_images()
    im = images[0]
    im.set_clim(0, 100)
    fig.set_size_inches(8.5, 8.5)

    # Get the expected radius of an NV
    try:
        with labrad.connect() as cxn:
            shared_params = tool_belt.get_shared_parameters_dict(cxn)
        airy_radius_nm = shared_params['airy_radius']
        galvo_nm_per_volt = shared_params['galvo_nm_per_volt']
        airy_radius_volts = airy_radius_nm / galvo_nm_per_volt
    except Exception:
        airy_radius_volts = 0.004

    if nv_indices is None:
        nv_indices = range(len(nv_sig_list))

    for ind in nv_indices:
        nv_sig = nv_sig_list[ind]
        if type(nv_sig) is dict:
            coords = nv_sig['coords']
        else:
            coords = nv_sig  # backwards compatibility
        circle = plt.Circle(tuple(coords[0:2]), 2*airy_radius_volts,
                            ec='g', fill=False, lw=2.0)
        ax.add_patch(circle)

    return fig

def generate_mapping_files(sample_name, micrometer_coords,
                           image_sample_file_name, nv_sig_list):

    raw_data = {
            'sample_name': sample_name,
            'micrometer_coords': micrometer_coords,
            'micrometer_coords-units': 'mm',
            'image_sample_file_name': image_sample_file_name,
            'nv_sig_list': nv_sig_list,
            'nv_sig_list-units': tool_belt.get_nv_sig_units(),
            }

    file_name = '{}-mapping'.format(image_sample_file_name)
    file_path = tool_belt.get_file_path(__file__, name=file_name)

    tool_belt.save_raw_data(raw_data, file_path)
    fig = illustrate_mapping(file_name)

    tool_belt.save_figure(fig, file_path)


# %% Run the file


if __name__ == '__main__':

#    image_sample_file_name = '2019-07-25_18-37-46_ayrton12_search'

    # Ignore this...
    if True:
        # Circle NVs from an existing mapping
        file_name = '2019-06-10_15-26-39_ayrton12_mapping'
        illustrate_mapping(file_name, [13])
    else:

        coords_list = [   [0.225, 0.142, 5.03],
                          [0.180, 0.190, 5.02],
                          [0.016, 0.242, 5.03],
                          [-0.038, 0.231, 5.01],
                          [0.003, 0.216, 5.02], # take g(2) again
                          [0.061, 0.164, 5.03],  #  great! nv5_2019_07_25
                          [0.006, 0.187, 5.03],  # take g(2) again
                          [0.003, 0.170, 5.03],
                          [-0.010, 0.145, 5.01],
                          [-0.080, 0.162, 5.01],
                          [-0.169, 0.161, 5.03], # great! nv10_2019_07_25
                          [-0.148, 0.111, 5.03],
                          [-0.221, 0.154, 5.03],
                          [-0.235, 0.140, 5.03],
                          [-0.229, 0.116, 5.02],
                          [-0.128, 0.049, 5.02], # possibly nv15_2019_07_25
                          [-0.191, 0.041, 5.04], # great! nv16_2019_07_25
                          [-0.101, 0.048, 5.02],
                          [0.032, 0.006, 5.03],  # great! low counts nv18_2019_07_25
                          [-0.075, 0.042, 5.02],
                          [-0.085, -0.006, 5.04],
                          [-0.012, -0.032, 5.03],
                          [0.045, -0.042, 5.01],
                          [0.026, -0.068, 5.01], # take g(2) again
                          [0.036, -0.188, 5.03],
                          [0.122, -0.219, 5.02], # great! nv25_2019_07_25
                          [-0.101, -0.082, 5.00],
                          [-0.229, -0.052, 5.03], # great! nv27_2019_07_25
                          [-0.209, -0.105, 5.05],
                          [-0.222, -0.121, 5.03], # possibly nv29_2019_07_25
                          [-0.056, -0.015, 5.02],
                          [-0.137, -0.046, 5.03],
                          [0.242, -0.018, 5.03],
                          [0.229, -0.024, 5.07]] # take g(2) again

        sample_name = 'ayrton12'
        micrometer_coords = [3.154, 2.193, 11.118, 120.21]
        image_sample_file_name = '2019-07-25_18-37-46_ayrton12_search'

        nv_sig_list = []
        for ind in range(len(coords_list)):
            coords = coords_list[ind]
            name = '{}-nv{}_2019_07_25'.format(sample_name, ind)
            nd_filter = 'nd_1.5'
            nv_sig = {'coords': coords, 'name': name, nd_filter: nd_filter}
            nv_sig_list.append(nv_sig)

        generate_mapping_files(sample_name, micrometer_coords,
                              image_sample_file_name, nv_sig_list)

# -*- coding: utf-8 -*-
"""Saves the data necessary to relocate specific NVs. Also can probably (?)
illustrate a mapping from an NV list to an image_sample.


11/24/2020 needs work... I added specific fixes to plot a few things, but needs to be generalized. 

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
import numpy
from pathlib import Path


# %% Functions


def illustrate_mapping(file_name, mapping_sub_folder, image_sub_folder, nv_indices=None):
    data = tool_belt.get_raw_data(mapping_sub_folder, file_name + '-untitled')
    image_sample_file_name = data['image_sample_file_name']
    nv_sig_list = data['nv_sig_list']

    mapping_full_file = image_sub_folder + '/' + image_sample_file_name
    fig = image_sample.create_figure(mapping_full_file )
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
                           image_sample_file_name, branch, month_folder, nv_sig_list):

    raw_data = {
            'sample_name': sample_name,
            'micrometer_coords': micrometer_coords,
            'micrometer_coords-units': 'mm',
            'image_sample_file_name': image_sample_file_name,
            'nv_sig_list': nv_sig_list,
            'nv_sig_list-units': tool_belt.get_nv_sig_units(),
            }

    file_name = '{}-mapping'.format(image_sample_file_name)
    file_path = tool_belt.get_file_path(__file__, file_name)
    print(file_path)
    
    mapping_sub_folder = 'pc_rabi/{}/image_sample_mapping/{}'.format(branch,month_folder )
    image_sub_folder = 'pc_rabi/{}/image_sample/{}'.format(branch,month_folder )
    tool_belt.save_raw_data(raw_data, file_path)
    fig = illustrate_mapping(file_name,  mapping_sub_folder, image_sub_folder,)

    tool_belt.save_figure(fig, file_path)


# %% Run the file


if __name__ == '__main__':

#    image_sample_file_name = '2019-07-25_18-37-46_ayrton12_search'

    # Ignore this...
#    if True:
        # Circle NVs from an existing mapping
#        file_name = '2019-06-10_15-26-39_ayrton12_mapping'
#    illustrate_mapping(file_name, [13])
#    else:
    drift_x = 0.005
    coords_list = [    
    [0.317, 0.338, 5.09],
[0.190, 0.345, 5.00],
[-0.031, 0.299, 4.94],
[-0.042, 0.265, 4.98],
[-0.084, 0.266, 4.96],
[0.333, 0.277, 4.93],
[0.316, 0.223, 5.01],
[-0.060, 0.179, 4.93],
[0.187, 0.127, 4.97],
[0.172, 0.140, 4.91],
[0.033, 0.085, 4.93],
[0.125, 0.049, 4.95],
[-0.010, 0.052, 4.92],
[0.057, -0.106, 4.93],
[0.385, -0.174, 4.99],
[0.134, -0.192, 4.93],
[0.400, -0.299, 4.97],
[0.374, -0.296, 4.96],
[-0.194, -0.326, 4.97],
[0.260, -0.382, 4.98],
]
#    coords_list = numpy.array(coords_list) + numpy.array([-0.007710279036012624, -0.003293695710258837, -0.012155496756263595])

    sample_name = 'goeppert-mayer'
    micrometer_coords = [3.154, 2.193, 11.118, 120.21]
    image_sample_file_name = '2021_01_26-10_41_58-goeppert-mayer-search'
    branch = 'branch_Spin_to_charge'
    month_folder = '2021_01'

    nv_sig_list = []
    for ind in range(len(coords_list)):
        coords = coords_list[ind]
        name = '{}-nv{}_2020_11_18'.format(sample_name, ind)
        nd_filter = 'nd_1.0'
        nv_sig = {'coords': coords, 'name': name, nd_filter: nd_filter}
        nv_sig_list.append(nv_sig)
    generate_mapping_files(sample_name, micrometer_coords,
                          image_sample_file_name, branch, month_folder,  nv_sig_list)

    # Would be great to include the ability to adjust due to drift...
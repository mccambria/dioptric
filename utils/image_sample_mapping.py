# -*- coding: utf-8 -*-
"""
Saves the data necessary to relocate specific NVs.

Also can probably (?) illustrate a mapping from an NV list to an image_sample.

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


# %% Functions


def illustrate_mapping(file_name):
    
    folder_dir = tool_belt.get_folder_dir(__file__)
    file_path = os.path.abspath(os.path.join(folder_dir, file_name))
    with open(file_path) as file:
        data = json.load(file)
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
        airy_radius_nm = shared_params['airy_radius_nm']
        galvo_nm_per_volt = shared_params['galvo_nm_per_volt']
        airy_radius_volts = airy_radius_nm / galvo_nm_per_volt
    except Exception:
        airy_radius_volts = 0.004

    for ind in range(len(nv_sig_list)):
        nv_sig = nv_sig_list[ind]
        circle = plt.Circle(tuple(nv_sig[0:2]), 2*airy_radius_volts,
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
            'nd_filter': nd_filter,
            'nv_sig_list': nv_sig_list,
            'nv_sig_list-units': tool_belt.get_nv_sig_units(),
            'nv_sig_list-format': tool_belt.get_nv_sig_format(),
            }
    
    folder_dir = tool_belt.get_folder_dir(__file__)
    file_name = '{}_mapping'.format(image_sample_file_name.split('.')[0])
    file_path = os.path.abspath(os.path.join(folder_dir, file_name))
    
    tool_belt.save_raw_data(raw_data, file_path)
    
    file_name_ext = '{}.txt'.format(file_name)
    fig = illustrate_mapping(file_name_ext)
    tool_belt.save_figure(fig, file_path)


# %% Run the file


if __name__ == '__main__':
    
    # Ignore this...
#    if False:
#        # Circle NVs from an existing mapping
#        file_name = '2019-06-10_15-26-39_ayrton12_mapping.txt'
#        illustrate_mapping(file_name)
#    else:
    
    # Generate the mapping files
    sample_name = 'ayrton12'
    micrometer_coords = [208.3, 277.3, 146.8]
    image_sample_file_name = ''
    nd_filter = 1.5
    z_voltage = 50.0
    background_count_rate = None
    nv_sig_list = [
               [-0.142, 0.501, z_voltage, 53, background_count_rate],
               [-0.133, 0.420, z_voltage, 45, background_count_rate],
               [-0.141, 0.269, z_voltage, 92, background_count_rate],
               [-0.224, 0.070, z_voltage, 49, background_count_rate],
               [-0.234, 0.123, z_voltage, 83, background_count_rate],
               [-0.236, 0.163, z_voltage, 78, background_count_rate],
               [-0.269, 0.184, z_voltage, 40, background_count_rate],
               [-0.306, 0.160, z_voltage, 64, background_count_rate],
               [-0.269, 0.184, z_voltage, 40, background_count_rate],
               [-0.287, 0.260, z_voltage, 66, background_count_rate],
               [-0.308, 0.270, z_voltage, 30, background_count_rate],
               [-0.335, 0.280, z_voltage, 74, background_count_rate],
               [-0.324, 0.325, z_voltage, 90, background_count_rate],
               [-0.379, 0.280, z_voltage, 43, background_count_rate],
               [-0.388, 0.294, z_voltage, 31, background_count_rate],
               [-0.389, 0.264, z_voltage, 85, background_count_rate],
               [-0.375, 0.183, z_voltage, 45, background_count_rate],
               [-0.416, 0.398, z_voltage, 35, background_count_rate],
               [-0.397, 0.383, z_voltage, 100, background_count_rate],
               [-0.397, 0.337, z_voltage, 85, background_count_rate],
               [-0.456, 0.152, z_voltage, 63, background_count_rate],
               [-0.415, 0.398, z_voltage, 33, background_count_rate],
               [-0.393, 0.484, z_voltage, 60, background_count_rate]]
    
#    nv_sig_list = [[*nv, None, None] for nv in nv_list]
    
    generate_mapping_files(sample_name, micrometer_coords,
                           image_sample_file_name, nv_sig_list)

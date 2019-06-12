# -*- coding: utf-8 -*-
"""
Illustrates a mapping from an NV list to an image_sample.

Created on Mon Jun 10 13:54:07 2019

@author: mccambria
"""

import json
import majorroutines.image_sample as image_sample
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt

diff_lim_spot_radius = 0.010

def create_mapping(file_name,
                   folder_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata/image_sample_mapping/'):
    
    with open('{}{}'.format(folder_dir, file_name)) as file:
        data = json.load(file)
        image_sample_file_name = data['image_sample_file_name']
        nv_list = data['nv_list']
        
    fig = image_sample.create_figure(image_sample_file_name)
    axes = fig.get_axes()
    ax = axes[0]
    
    for ind in range(len(nv_list)):
        nv = nv_list[ind]
        circle = plt.Circle(tuple(nv[0:2]), diff_lim_spot_radius,
                            ec='g', fill=False, lw=2.0)
        ax.add_patch(circle)
    
def save_mapping_file(image_sample_file_name, nv_list,
                      image_sample_mapping_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata/image_sample_mapping/'):
    
    raw_data = {
            'image_sample_file_name': file_name,
            'nv_list': nv_list
            }
    
    file_path = '{}{}'.format(image_sample_mapping_dir,
                 image_sample_file_name.split('.')[0])
    
    tool_belt.save_raw_data(raw_data, file_path)

if __name__ == '__main__':
    
#    file_name = '2019-06-03_16-25-40_ayrton12.txt'
    file_name = '2019-06-04_09-18-30_ayrton12.txt'

    create_mapping(file_name)
    
#    nv_list = [[0.251, 0.235, 54.0],
#               [0.005, 0.226, 53.9],
#               [0.140, 0.052, 54.1],
#               [0.032, -0.126, 54.1],
#               [-0.176, -0.077, 54.1],
#               [-0.188, -0.112, 53.9],
#               [0.208, 0.195, 54.0],
#               [0.131, 0.102, 54.0]]
##    
#    save_mapping_file(file_name, nv_list)

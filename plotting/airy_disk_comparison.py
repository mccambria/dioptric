# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:35:01 2021

Work in progress...

@author: agard
"""


import matplotlib.pyplot as plt 
import numpy
import utils.tool_belt as tool_belt
import math

# %%
def extract_data(file, folder):
    data = tool_belt.get_raw_data('', folder + '/' + file)
    try:
        img_array = data['img_array']
    except Exception:
        img_array = data['readout_image_array']
        
    try:
        y_voltages = data['y_voltages']
    except Exception:
        y_voltages = data['y_voltages_1d']
        
    
    return img_array, y_voltages

def do_plot(file_conf, folder_conf, file_space, folder_space):
    conf_c = 20
    space_c = 23
    
    conf_img_array, conf_y_volts = extract_data(file_conf, folder_conf)
    space_img_array, space_y_volts = extract_data(file_space, folder_space)
    
    # find midpoint:
    voltage_list_len = len(conf_y_volts)
    ind_mid_low = math.floor(voltage_list_len/2)
    ind_mid_high = math.ceil(voltage_list_len/2)
    
    print(conf_y_volts[ind_mid_high])
    mid_value = ((conf_y_volts[ind_mid_high] - conf_y_volts[ind_mid_low]) / 2) + conf_y_volts[ind_mid_high]
    
    
    print(len(conf_y_volts))
    print(len(space_y_volts))
    
    f_size = 10
    tick_f_size = f_size
    fig_w = 3.8
    fig_l = fig_w*0.75
    
    # mu = u"\u03BC" 
    
    fig_tick_l = 3
    fig_tick_w = 0.75

    
    
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    ax.plot((numpy.array(conf_y_volts)-mid_value)*35/10**3, numpy.array(conf_img_array[conf_c])/100, label = 'Confocal slice')
    ax.plot((numpy.array(space_y_volts)-mid_value)*35/10**3, space_img_array[conf_c], label = 'SPaCE slice')
    ax.set_xlabel('Position (V)')#' (' + mu + 's)', fontsize = f_size)
    ax.set_ylabel("Counts (arb.)",  fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)
    ax.legend()
    fig.tight_layout()
    
# %%

folder_conf = 'pc_rabi/branch_Spin_to_charge/image_sample/2020_12'
file_conf = '2020_12_09-15_36_03-goeppert-mayer-nv1_2020_12_02'

folder_space = 'pc_rabi/branch_Spin_to_charge/isolate_nv_charge_dynamics_moving_target/2020_12'
file_space = '2020_12_08-18_04_02-goeppert-mayer-nv1_2020_12_02-img'

do_plot(file_conf, folder_conf, file_space, folder_space)
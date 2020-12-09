# -*- coding: utf-8 -*-
"""

Created on Fri Dec 4 11:26:49 2020

@author: agardill
"""

import os
import json
import numpy
import csv
import utils.tool_belt as tool_belt

def convert(folder_name):
    
    folder_items = os.listdir(folder_name)

    for json_file_name in folder_items:
    
        # Only process txt files, which we assume to be json files
        if not json_file_name.endswith('.txt'):
            continue
        
        # check if it is the image file or the measurement file
        if json_file_name[-7:] == 'img.txt':
            
            with open('{}/{}'.format(folder_name, json_file_name)) as json_file:
                # print(json_file_name)
                try:
                    data = json.load(json_file)
                    init_color = data['init_color']
                    pulse_color = data['pulse_color']
                    center_coords = data['start_coords']
                    ind_list = data['ind_list']
                    num_steps = data['num_steps']
                    num_runs = data['num_runs']
                    coords_voltages = data['coords_voltages']
                    timestamp = data['timestamp']
                    nv_sig = data['nv_sig']
                    pulsed_ionization_dur=nv_sig['pulsed_ionization_dur']
                    pulsed_reionization_dur=nv_sig['pulsed_reionization_dur']
                    green_optical_power_mW = data['green_optical_power_mW']
                    red_optical_power_mW = data['red_optical_power_mW']
                    yellow_optical_power_mW = data['yellow_optical_power_mW']
                    # color_filter = nv_sig['color_filter']
                except Exception:
                    # Skip txt files that are evidently not data files
                    print('skipped {} when looking for img file'.format(json_file_name))
                    continue
            
        elif json_file_name[-7:] != 'img.txt':
                
            with open('{}/{}'.format(folder_name, json_file_name)) as json_file:        
                try:
                    data = json.load(json_file)
                    readout_counts_array_sh = data['readout_counts_array']
                    rad_dist_sh = data['rad_dist']
                except Exception:
                    # Skip txt files that are evidently not data files
                    print('skipped {}'.format(json_file_name))
                    continue
            
    # create unshuffled arrays to fill with the data
    readout_counts_array_unsh = numpy.empty([num_steps*num_steps, num_runs])
    rad_dist_unsh = numpy.empty(len(rad_dist_sh))
    
    # transpose the data so that the elements correspond to the coordinate, not the run
    readout_counts_array_sh = numpy.transpose(readout_counts_array_sh)
    
    # unshuffle the data
    list_ind = 0
    for f in ind_list:
        readout_counts_array_unsh[f] = readout_counts_array_sh[list_ind]
        rad_dist_unsh[f] = rad_dist_sh[list_ind]
        list_ind += 1
        
    # Populate the data to save
    csv_data = []
    # if pulse_color == 532:
    #     csv_data.append(pulsed_reionization_dur)
    # elif pulse_color == 638:
    #     csv_data.append(pulsed_ionization_dur)
    # csv_data.append('pulse: {} nm'.format(pulse_color))
    # csv_data.append('center coords: {} V'.format(center_coords))
    # csv_data.append('coords (V), radial distance to center (V), raw counts')
    
    for ind in range(num_steps*num_steps):
        row = []
        row.append(coords_voltages[ind][0])
        row.append(coords_voltages[ind][1])
        row.append(coords_voltages[ind][2])
        row.append(rad_dist_unsh[ind])
        for i in range(len(readout_counts_array_unsh[ind])):
            row.append(readout_counts_array_unsh[ind][i])
        csv_data.append(row)
    if pulse_color == 532:
        csv_file_name = '2020_12_04-{}_to_{}-{:.1f}_mW'.format(init_color, pulse_color, green_optical_power_mW)
    elif pulse_color == 638:
        csv_file_name = '2020_12_04-{}_to_{}-{:.1f}_mW'.format(init_color, pulse_color, red_optical_power_mW)

    with open('{}/{}.csv'.format(folder_name, csv_file_name),
              'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONE)
        csv_writer.writerows(csv_data)
        
    raw_data = {'timestamp': timestamp,
                'init_color': init_color,
                'pulse_color': pulse_color,
            'center_coords': center_coords,
            'num_steps': num_steps,
            'nv_sig': nv_sig,
            'green_optical_power_mW': green_optical_power_mW,
            'green_optical_power_mW-units': 'mW',
            'red_optical_power_mW': red_optical_power_mW,
            'red_optical_power_mW-units': 'mW',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
            'num_runs':num_runs,
            'coords_voltages': coords_voltages,
            'coords_voltages-units': '[V, V]',
            'ind_list': ind_list,
            'rad_dist_unsh': rad_dist_unsh.tolist(),
            'rad_dist_unsh-units': 'V',
            'readout_counts_array_unsh': readout_counts_array_unsh.tolist(),
            'readout_counts_array_unsh-units': 'counts',
            }
        
    # file_path = tool_belt.get_file_path(csv_file_name)
    tool_belt.save_raw_data(raw_data, folder_name + '-unshuffled')
        
def convert_charge_counts(folder_name):
    
    folder_items = os.listdir(folder_name)

    for json_file_name in folder_items:
    
        # Only process txt files, which we assume to be json files
        if not json_file_name.endswith('.txt'):
            continue

                
        with open('{}/{}'.format(folder_name, json_file_name)) as json_file:        
            try:
                data = json.load(json_file)
                nv0_list = data['nv0_list']
                nvm_list = data['nvm_list']
                parameters_sig = data['parameters_sig']
                name = parameters_sig['name']
            except Exception:
                # Skip txt files that are evidently not data files
                print('skipped {}'.format(json_file_name))
                continue

        
    # Populate the data to save
    csv_data = []
    
    for ind in range(len(nv0_list[0])):
        row = []
        row.append(nv0_list[0][ind])
        row.append(nvm_list[0][ind])
        csv_data.append(row)

    csv_file_name = '{}'.format(name)

    with open('{}/{}.csv'.format(folder_name, csv_file_name),
              'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONE)
        csv_writer.writerows(csv_data)
        
                        
if __name__ == '__main__':
    
    top_folder_name = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/isolate_nv_charge_dynamics_moving_target/branch_Spin_to_charge/2020_12/{}'
        
    sub_folder_names = ['2020_12_08-goeppert-mayer_10us'
                        ]
#    
    for el in sub_folder_names:
        
        convert(top_folder_name.format(el))
        
    # convert(top_folder_name.format('relaxation rate paper data'))
    # top_folder_name2 = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/collect_charge_counts/branch_Spin_to_charge/2020_12'
    # convert_charge_counts(top_folder_name2)
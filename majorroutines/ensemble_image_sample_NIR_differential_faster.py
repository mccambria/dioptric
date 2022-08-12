# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:03:37 2022

@author: kolkowitz
"""

from majorroutines import ensemble_image_sample_NIR_differential
from majorroutines import image_sample
import labrad
import time
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import numpy as np

def NIR_offon(cxn,ind,nir_laser_voltage):
    cxn_power_supply = cxn.power_supply_mp710087
    if ind == 1:
        cxn_power_supply.output_on()
        cxn_power_supply.set_voltage(nir_laser_voltage)
    elif ind == 0:
        cxn_power_supply.output_off()

def plot_diff_counts(diff_counts, image_extent,imgtitle,cbarlabel,cbarmin=None,cbarmax=None):

    fig = tool_belt.create_image_figure(
        diff_counts,
        image_extent,
        title=imgtitle,
        color_bar_label=cbarlabel,
        cmin=cbarmin,
        cmax=cbarmax,
    )
    return fig

def main(nv_sig, scan_range, num_steps, apd_indices, nir_laser_voltage, sleeptime,nv_minus_initialization=False,
         cbarmin_percdiff=None,cbarmax_percdiff=None,cbarmin_counts=None,cbarmax_counts=None,cbarmin_diffcounts=None,cbarmax_diffcounts=None):
    
    with labrad.connect() as cxn:
        cxn_power_supply = cxn.power_supply_mp710087
        images = []
        readout_sec = nv_sig['imaging_readout_dur']/10**9
        for ind in range(2):
            NIR_offon(cxn,ind,nir_laser_voltage)
            time.sleep(sleeptime)
            img_array, x_voltages, y_voltages = image_sample.main(nv_sig,scan_range, scan_range, 
                                                                  num_steps,apd_indices,nv_minus_initialization=nv_minus_initialization,plot_data=False)
            img_array_kcps = img_array / 1000 / readout_sec
            
            images.append( np.fliplr(img_array_kcps) )
        
        NIR_img = images[1]
        noNIR_img = images[0]
        diff_counts = NIR_img - noNIR_img
        perc_diff_counts = (NIR_img - noNIR_img)/noNIR_img
        cxn_power_supply.output_off()  
        
        
        drift = tool_belt.get_drift()
        coords = nv_sig["coords"]
        image_center_coords = (np.array(coords) + np.array(drift)).tolist()
        x_center, y_center, z_center = image_center_coords
        image_extent = tool_belt.calc_image_extent(x_center, y_center, scan_range, num_steps)
        
        # x_num_steps = len(x_voltages)
        # x_low = x_voltages[0]
        # x_high = x_voltages[x_num_steps-1]
        # y_num_steps = len(y_voltages)
        # y_low = y_voltages[0]
        # y_high = y_voltages[y_num_steps-1]

        # pixel_size = x_voltages[1] - x_voltages[0]
        # half_pixel_size = pixel_size / 2
        # image_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
        #               y_low - half_pixel_size, y_high + half_pixel_size]
        
        title = r'{}, {} ms readout'.format(nv_sig['imaging_laser'], readout_sec*1000)
                
        fig1 = plot_diff_counts(perc_diff_counts, image_extent,title,'(NIR-noNIR)/noNIR Counts',cbarmin_percdiff,cbarmax_percdiff)
        fig2 = plot_diff_counts(diff_counts, image_extent,title,'NIR-noNIR Counts (kcps)',cbarmin_diffcounts,cbarmax_diffcounts)
        fig3 = plot_diff_counts(noNIR_img, image_extent,title,'noNIR Counts (kcps)',cbarmin_counts,cbarmax_counts)
        fig4 = plot_diff_counts(NIR_img, image_extent,title,'NIR Counts (kcps)',cbarmin_counts,cbarmax_counts)
        timestamp = tool_belt.get_time_stamp()
        # print(nv_sig['coords'])
        print(timestamp)
        rawData = {'timestamp': timestamp,
                   'nv_sig': nv_sig,
                   'nv_sig-units': tool_belt.get_nv_sig_units(),
                   'drift': drift,
                   'x_range': scan_range,
                   'x_range-units': 'V',
                   'y_range': scan_range,
                   'y_range-units': 'V',
                   'num_steps': num_steps,
                   'img_extent': image_extent,
                   'readout_laser': nv_sig['imaging_laser'],
                   'readout': readout_sec*10**9,
                   'readout-units': 'ns',
                   'x_voltages': x_voltages.tolist(),
                   'x_voltages-units': 'V',
                   'y_voltages': y_voltages.tolist(),
                   'y_voltages-units': 'V',
                   'NIR_img': NIR_img.astype(int).tolist(),
                   'noNIR_img': noNIR_img.astype(int).tolist(),
                   'diff_counts': diff_counts.astype(int).tolist(),
                   'perc_diff_counts': perc_diff_counts.tolist(),
                   'img_array-units': 'counts'}

    filePath1 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_percentdiff")
    filePath2 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_diff")
    filePath3 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_noNIR")
    filePath4 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_NIR")
    tool_belt.save_raw_data(rawData, filePath1)
    tool_belt.save_figure(fig1, filePath1)
    tool_belt.save_figure(fig2, filePath2)
    tool_belt.save_figure(fig3, filePath3)
    tool_belt.save_figure(fig4, filePath4)
    

if __name__ == '__main__':
    filename = '2022_08_12-09_40_21-hopper-search_percentdiff'
    data = tool_belt.get_raw_data(filename,path_from_nvdata='pc_hahn/branch_master/ensemble_image_sample_NIR_differential_faster/2022_08')
    NIR_img = data['NIR_img']
    noNIR_img = data['noNIR_img']
    diff_img = data['diff_counts']
    percdiff_img = data['perc_diff_counts']
    cbarmin_percdiff, cbarmax_percdiff = -0.00,.125
    cbarmin_diffcounts,cbarmax_diffcounts = 0,220.0
    cbarmin_counts,cbarmax_counts = None,None
    title = r'{}, {} ms readout'.format(data['readout_laser'], data['readout']/10**6)
    image_extent = data['img_extent']
    fig1 = plot_diff_counts(percdiff_img, image_extent,title,'(NIR-noNIR)/noNIR Counts',cbarmin_percdiff,cbarmax_percdiff)
    # fig2 = plot_diff_counts(diff_img, image_extent,title,'NIR-noNIR Counts (kcps)',cbarmin_diffcounts,cbarmax_diffcounts)
    # fig3 = plot_diff_counts(noNIR_img, image_extent,title,'noNIR Counts (kcps)',cbarmin_counts,cbarmax_counts)
    # fig4 = plot_diff_counts(NIR_img, image_extent,title,'NIR Counts (kcps)',cbarmin_counts,cbarmax_counts)
    
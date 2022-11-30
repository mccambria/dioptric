# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:29:08 2022

@author: kolkowitz
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:36:14 2022

@author: carterfox

This file will be for finding the necessary number of averages for scc.
What it does is vary the bin width and find the point where the separation is larger than some threshold.
The threshold I set is sigma_level*(std_0 + std_1), which should mean there is only overlap at the sigma level inputted. 
"""


import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats
import json
from pathlib import Path
import utils.tool_belt as tool_belt
from matplotlib.ticker import FormatStrFormatter
import os
# sigma_level = 3

def get_raw_data(file_name):
    with open(file_name) as f:
        res = json.load(f)
        return res


def get_binnings(files,photon_thresholds=[],photon_thresholds_readout_times=[],sigma_level=2):

    best_binnings, readout_times_us, averaging_times_ms, readout_powers = [],[],[],[]
    for file in files:
        data = tool_belt.get_raw_data(file)
        
        nv_sig = data['nv_sig']
        
        readout_time = nv_sig['charge_readout_dur']
        readout_power = nv_sig['charge_readout_laser_power']
        init_power = nv_sig['nv-_reionization_dur']
        readout_times_us.append(readout_time/1000)
        readout_powers.append(readout_power)
        # print(readout_time/1000)
        bin_width = 25
        threshold_reached = False
        norm_seps = []
        bin_widths = []
        
        
            
        sig_counts = np.array(data['sig_counts_eachshot_array'])[0]
        ref_counts = np.array(data['ref_counts_eachshot_array'])[0]
        
        if len(photon_thresholds) != 0:
            photon_thresholds = np.array(photon_thresholds)
            photon_thresholds_readout_times = np.array(photon_thresholds_readout_times)
            photon_thresh = photon_thresholds[np.where(photon_thresholds_readout_times==readout_time)][0]
            # if photon_thresh==0:
            #     photon_thresh=1
            # print(photon_thresh)
            sig_counts[np.where(sig_counts<photon_thresh)]=0
            sig_counts[np.where(sig_counts>=photon_thresh)]=1
            
            ref_counts[np.where(ref_counts<photon_thresh)]=0
            ref_counts[np.where(ref_counts>=photon_thresh)]=1
            print('using photon threshold')
        
        while threshold_reached == False:    
        
            binned_sig_counts = sig_counts[:(sig_counts.size // bin_width) * bin_width].reshape(-1,bin_width).sum(axis=1)
            binned_ref_counts = ref_counts[:(ref_counts.size // bin_width) * bin_width].reshape(-1,bin_width).sum(axis=1)
            
            mean_0 = np.average(binned_sig_counts)
            mean_1 = np.average(binned_ref_counts)
            std_0 = np.std(binned_sig_counts)
            std_1 = np.std(binned_ref_counts)
            separation = mean_0 - mean_1
            norm_sep = separation/( sigma_level * (std_0 + std_1))
            norm_seps.append(norm_sep)
            bin_widths.append(bin_width)
            #need norm_sep to be greater than 1 so they don't overlap at two sigma
            
            # print(separation,norm_sep)
            if norm_sep >=1:
                threshold_reached = True
                best_binning = bin_width
                best_binning_sig_counts = binned_sig_counts
                best_binning_ref_counts = binned_ref_counts
                best_binnings.append(best_binning)
                averaging_time_ms = best_binning*readout_time/1000000
                averaging_times_ms.append(averaging_time_ms)
                # print('threshold reached')
            
            else:
                if norm_sep <= .4:
                    bin_width = bin_width*2
                elif norm_sep <= .6:
                    bin_width = int(bin_width*1.15)
                else:
                    bin_width = int(bin_width*1.05)
        
        
        if False:
            plt.figure()
            plt.hist(best_binning_sig_counts,bins=10)
            plt.hist(best_binning_ref_counts,bins=10)
            plt.xlabel('Summed Counts (bin width = {})'.format(best_binning))
            plt.ylabel('Occurences')
            plt.show()
            
            plt.figure()
            plt.plot(bin_widths,norm_seps)
            plt.xlabel('Bin Width')
            plt.ylabel(r'2$\sigma$ Normalized Separation')
            plt.show()
        
    print('binnings necessary for {} sigma'.format(sigma_level))
    print('readout times (us): ',readout_times_us)
    print('averaging times (ms): ',averaging_times_ms)
    print('binnings: ',best_binnings)
    print('')
    
    
    raw_data = {'readout_powers':readout_powers,
                'readout_times_us': readout_times_us,
                'averaging_times_ms':averaging_times_ms,
                'best_binnings':best_binnings,
                'sigma_level':sigma_level}
    
    direc = 'C:/Users/kolkowitz/Documents/GitHub/kolkowitz-nv-experiment-v1.0/analysis/determine_scc_averaging/'
    timestamp = tool_belt.get_time_stamp()
    savename = timestamp+'_'+str(readout_power*1000)+'mV_'+str(sigma_level)+'sigma_data.txt'
    
    pathtosave = Path(direc+savename)
    tool_belt.save_raw_data(raw_data, pathtosave)
    
    return np.array(readout_times_us),np.array(best_binnings),np.array(averaging_times_ms), readout_power
    

if __name__ == "__main__":

    power_sweep_folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_Carr/branch_opx-setup/determine_scc_pulse_params/2022_11/11_18_power_sweep'
    files_300mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/300mV')]
    files_350mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/350mV')]
    files_400mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/400mV')]
    files_450mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/450mV')]
    files_500mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/500mV')]
    files_550mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/550mV')]
    files_600mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/600mV')]
    files_650mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/650mV')]
    files_700mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/700mV')]
    files_750mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/750mV')]
    files_800mv = [ f[0:-4] for f in os.listdir(power_sweep_folder+'/800mV')]
    
    files_test = ['2022_11_16-15_31_11-johnson-search-ion_pulse_dur']
    if True:
        fileslist= [files_550mv]
        photon_thresholds,photon_thresholds_readout_times = [], []
        photon_thresholds = [3,2,1,1,1,1]
        photon_thresholds_readout_times = [4e6,1e6,400e3,100e3,50e3,10e3]
        
        for files in fileslist:
            # readout_times_us_3,best_binnings_3,averaging_times_ms_3,readout_power = get_binnings(files,photon_thresholds,photon_thresholds_readout_times,3)
            readout_times_us_2,best_binnings_2,averaging_times_ms_2,readout_power = get_binnings(files,photon_thresholds,photon_thresholds_readout_times,2)
            # readout_times_us_1,best_binnings_1,averaging_times_ms_1,readout_power = get_binnings(files,photon_thresholds,photon_thresholds_readout_times,1)
        
            timestamp = tool_belt.get_time_stamp()
            direc = 'C:/Users/kolkowitz/Documents/GitHub/kolkowitz-nv-experiment-v1.0/analysis/determine_scc_averaging/'
            figsavename = timestamp+'_avg_time_vs_readout_time'
            figsave = Path(direc+figsavename)
        
            fig = plt.figure()
            # plt.scatter(readout_times_us_3,averaging_times_ms_3,c='r',label='3$\sigma$')
            plt.plot(readout_times_us_2,averaging_times_ms_2,'o-',c='b',label='2$\sigma$')
            # plt.scatter(readout_times_us_1,averaging_times_ms_1,c='g',label='1$\sigma$')
            plt.xlabel('Readout Time ($\mu$s)')
            plt.ylabel('Averaging Time (ms)')
            plt.title('Readout Power: {}V'.format(readout_power))
            plt.legend()
            plt.show()
            
            tool_belt.save_figure(fig,figsave)
        
    
    if False:
        pathnv = Path('C:/Users/kolkowitz/Documents/GitHub/kolkowitz-nv-experiment-v1.0/analysis/')
        pathfrom = 'determine_scc_averaging'
        
        # fname1_300 = "2022_11_21-08_56_43_1sigma_data"
        # fname3_300 = "2022_11_21-08_47_14_3sigma_data"
        # fname2_300 = "2022_11_21-08_49_06_2sigma_data"
        # fname1_350 = "2022_11_21-08_51_05_1sigma_data"
        # fname3_350 = "2022_11_21-08_53_14_3sigma_data"
        # fname2_350 = "2022_11_21-08_54_53_2sigma_data"
        # fname3_400 = "2022_11_21-09_02_48_400.0mV_3sigma_data"
        # fname2_400 = "2022_11_21-09_04_42_400.0mV_2sigma_data"
        # fname1_400 = "2022_11_21-09_06_22_400.0mV_1sigma_data"
        # fname3_450 = "2022_11_21-09_08_22_450.0mV_3sigma_data"
        # fname2_450 = "2022_11_21-09_10_05_450.0mV_2sigma_data"
        # fname1_450 = "2022_11_21-09_11_42_450.0mV_1sigma_data"
        # fname3_500 = "2022_11_21-09_13_29_500.0mV_3sigma_data"
        # fname2_500 = "2022_11_21-09_15_11_500.0mV_2sigma_data"
        # fname1_500 = "2022_11_21-09_16_50_500.0mV_1sigma_data"
        # fname3_550 = "2022_11_21-09_18_42_550.0mV_3sigma_data"
        # fname2_550 = "2022_11_21-09_20_23_550.0mV_2sigma_data"
        # fname1_550 = "2022_11_21-09_21_55_550.0mV_1sigma_data"
        # fname3_600 = "2022_11_21-09_23_55_600.0mV_3sigma_data"
        # fname2_600 = "2022_11_21-09_25_40_600.0mV_2sigma_data"
        # fname1_600 = "2022_11_21-09_27_17_600.0mV_1sigma_data"
        # fname3_650 = "2022_11_21-09_29_13_650.0mV_3sigma_data"
        # fname2_650 = "2022_11_21-09_30_55_650.0mV_2sigma_data"
        # fname1_650 = "2022_11_21-09_32_34_650.0mV_1sigma_data"
        # fname3_700 = "2022_11_21-09_34_25_700.0mV_3sigma_data"
        # fname2_700 = "2022_11_21-09_36_11_700.0mV_2sigma_data"
        # fname1_700 = "2022_11_21-09_37_50_700.0mV_1sigma_data"
        # fname3_750 = "2022_11_21-09_39_39_750.0mV_3sigma_data"
        # fname2_750 = "2022_11_21-09_41_22_750.0mV_2sigma_data"
        # fname1_750 = "2022_11_21-09_43_01_750.0mV_1sigma_data"
        # fname3_800 = "2022_11_21-09_44_48_800.0mV_3sigma_data"
        # fname2_800 = "2022_11_21-09_46_33_800.0mV_2sigma_data"
        # fname1_800 = "2022_11_21-09_48_14_800.0mV_1sigma_data"
        
        # data_3_300 = tool_belt.get_raw_data(fname3_300,pathfrom,pathnv)
        # data_2_300 = tool_belt.get_raw_data(fname2_300,pathfrom,pathnv)
        # data_1_300 = tool_belt.get_raw_data(fname1_300,pathfrom,pathnv)
        
        # data_3_350 = tool_belt.get_raw_data(fname3_350,pathfrom,pathnv)
        # data_2_350 = tool_belt.get_raw_data(fname2_350,pathfrom,pathnv)
        # data_1_350 = tool_belt.get_raw_data(fname1_350,pathfrom,pathnv)
        
        # data_3_400 = tool_belt.get_raw_data(fname3_400,pathfrom,pathnv)
        # data_2_400 = tool_belt.get_raw_data(fname2_400,pathfrom,pathnv)
        # data_1_400 = tool_belt.get_raw_data(fname1_400,pathfrom,pathnv)
        
        # data_3_450 = tool_belt.get_raw_data(fname3_450,pathfrom,pathnv)
        # data_2_450 = tool_belt.get_raw_data(fname2_450,pathfrom,pathnv)
        # data_1_450 = tool_belt.get_raw_data(fname1_450,pathfrom,pathnv)
        
        # data_3_500 = tool_belt.get_raw_data(fname3_500,pathfrom,pathnv)
        # data_2_500 = tool_belt.get_raw_data(fname2_500,pathfrom,pathnv)
        # data_1_500 = tool_belt.get_raw_data(fname1_500,pathfrom,pathnv)
        
        # data_3_550 = tool_belt.get_raw_data(fname3_550,pathfrom,pathnv)
        # data_2_550 = tool_belt.get_raw_data(fname2_550,pathfrom,pathnv)
        # data_1_550 = tool_belt.get_raw_data(fname1_550,pathfrom,pathnv)
        
        # data_3_600 = tool_belt.get_raw_data(fname3_600,pathfrom,pathnv)
        # data_2_600 = tool_belt.get_raw_data(fname2_600,pathfrom,pathnv)
        # data_1_600 = tool_belt.get_raw_data(fname1_600,pathfrom,pathnv)
        
        # data_3_650 = tool_belt.get_raw_data(fname3_650,pathfrom,pathnv)
        # data_2_650 = tool_belt.get_raw_data(fname2_650,pathfrom,pathnv)
        # data_1_650 = tool_belt.get_raw_data(fname1_650,pathfrom,pathnv)
        
        # data_3_700 = tool_belt.get_raw_data(fname3_700,pathfrom,pathnv)
        # data_2_700 = tool_belt.get_raw_data(fname2_700,pathfrom,pathnv)
        # data_1_700 = tool_belt.get_raw_data(fname1_700,pathfrom,pathnv)
        
        # data_3_750 = tool_belt.get_raw_data(fname3_750,pathfrom,pathnv)
        # data_2_750 = tool_belt.get_raw_data(fname2_750,pathfrom,pathnv)
        # data_1_750 = tool_belt.get_raw_data(fname1_750,pathfrom,pathnv)
        
        # data_3_800 = tool_belt.get_raw_data(fname3_800,pathfrom,pathnv)
        # data_2_800 = tool_belt.get_raw_data(fname2_800,pathfrom,pathnv)
        # data_1_800 = tool_belt.get_raw_data(fname1_800,pathfrom,pathnv)
        
        fname_2_550_thresh = "2022_11_22-09_44_14_550.0mV_2sigma_data"
        fname_2_550_nothresh = "2022_11_22-09_43_50_550.0mV_2sigma_data"
        data_2_550_thresh = tool_belt.get_raw_data(fname_2_550_thresh,pathfrom,pathnv)
        data_2_550_nothresh = tool_belt.get_raw_data(fname_2_550_nothresh,pathfrom,pathnv)
        
        def plot_data(data_1,data_2,data_3):
            readout_power = data_3['readout_powers'][0]
            readout_times_us_3 = np.array(data_3['readout_times_us'])
            sort_inds = np.argsort(readout_times_us_3)
            readout_times_us_3 = readout_times_us_3[sort_inds]
            
            averaging_times_ms_3 = np.array(data_3['averaging_times_ms'])[sort_inds]
            best_binnings_3 = np.array(data_3['best_binnings'])[sort_inds]
            readout_times_us_2 = np.array(data_2['readout_times_us'])[sort_inds]
            averaging_times_ms_2 = np.array(data_2['averaging_times_ms'])[sort_inds]
            best_binnings_2 = np.array(data_2['best_binnings'])[sort_inds]
            readout_times_us_1 = np.array(data_1['readout_times_us'])[sort_inds]
            averaging_times_ms_1 = np.array(data_1['averaging_times_ms'])[sort_inds]
            best_binnings_1 = np.array(data_1['best_binnings'])[sort_inds]

            # print(best_binnings_1)# best_binnings = data['best_binnings']
            print(best_binnings_2)
            # print(best_binnings_3)
            # sigma_level = data['sigma_level']
            
            fig = plt.figure()
            # plt.plot(readout_times_us_3,averaging_times_ms_3,marker='.',c='r',label='3$\sigma$')
            plt.plot(readout_times_us_2,averaging_times_ms_2,marker='.',c='b',label='2$\sigma$-no threshold')
            plt.plot(readout_times_us_1,averaging_times_ms_1,marker='.',c='g',label='2$\sigma$-threshold')
            plt.xlabel('Readout Time ($\mu$s)')
            plt.ylabel('Averaging Time (ms)')
            plt.title('Readout Power: {}V'.format(readout_power))
            plt.legend()
            plt.yscale('log')
            plt.xscale('log')
            tty = [30,100,1000,10000]
            plt.yticks(ticks = tty,labels=tty)
            ttx = [10,100,1000,10000]
            plt.xticks(ticks = ttx,labels=ttx)
            plt.show()
        
        plot_data(data_2_550_thresh, data_2_550_nothresh, data_2_550_thresh)
        # plot_data(data_1_300, data_2_300, data_3_300)
        # plot_data(data_1_350, data_2_350, data_3_350)
        # plot_data(data_1_400, data_2_400, data_3_400)
        # plot_data(data_1_450, data_2_450, data_3_450)
        # plot_data(data_1_500, data_2_500, data_3_500)
        # plot_data(data_1_550, data_2_550, data_3_550)
        # plot_data(data_1_600, data_2_600, data_3_600)
        # plot_data(data_1_650, data_2_650, data_3_650)
        # plot_data(data_1_700, data_2_700, data_3_700)
        # plot_data(data_1_750, data_2_750, data_3_750)
        # plot_data(data_1_800, data_2_800, data_3_800)
        
























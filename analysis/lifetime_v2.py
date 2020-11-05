# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:13:01 2019

@author: Aedan
"""
import numpy
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

    
# %%

def exp_decay(t, a, d):
    return a*numpy.exp(-t/d)

def exp_decay_double(t, a, d1, d2):
    return a*numpy.exp(-t/d1) + a*numpy.exp(-t/d2)

# %%

def plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list= None ):
    start_num = 0
    
    fig, ax= plt.subplots(1, 1, figsize=(10, 8))
#    fmt_data_list = ['b.', 'y.', 'g.']
#    fmt_fit_list = ['b-', 'y-', 'g-']
    
#    text_eq = r'$A_0 (e^{-t/d1})$'
#
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    ax.text(0.55, 0.8, text_eq, transform=ax.transAxes, fontsize=12,
#        verticalalignment='top', bbox=props)
    
#    d_list = []
#    voltage_list_1 = [-0.5,0, 0.5, 1, 1.5, 2, 2.5, 3]
#    voltage_list_2  = [0.1, 1, 2, 3] 
#    voltage_list_3 = [0.1, 2.2, 4, -3]
#    t_ind = 0
    
    for f in range(len(file_list)):
        file = file_list[f] 
        with open(file_dir + '/'+ file + '.txt') as json_file:
        
            # Load the data from the file
            data = json.load(json_file)
            counts = numpy.array(data["binned_samples"])
            bin_centers = numpy.array(data["bin_centers"])/10**3
        if background_file_list:
            bkgd_file = background_file_list[f]
            with open(file_dir + '/'+ bkgd_file + '.txt') as json_file:
                bkgd_data = json.load(json_file)
                bkgd_counts = numpy.array(bkgd_data["binned_samples"])
                
            

        bin_centers_norm = numpy.array(bin_centers)-bin_centers[start_num]

        counts = counts - bkgd_counts
        
        first_point = counts[start_num]
        last_point = counts[-1]
        norm_counts = (counts - last_point) / (first_point - last_point)

        #fit the data to single exponential
#        init_guess = [1, 100]
#        popt, pcov = curve_fit(exp_decay, bin_centers_norm[start_num:-66],
#                                         norm_counts[start_num:-66], p0=init_guess)
#        
#        text_popt = r'{}: $A_0 = {}, d1 = {} us$'.format(label_list[f],'%.3f'%popt[0],'%.1f'%popt[1])
#        ax.text(0.55, 0.75 - t_ind, text_popt, transform=ax.transAxes, fontsize=12,
#            verticalalignment='top', bbox=props)
#        time_linspace = numpy.linspace(bin_centers_norm[start_num], bin_centers_norm[-66], 1000)
#        ax.plot(time_linspace, exp_decay(time_linspace, *popt),'-')        
#        d_list.append(popt[1])
        #fit the data to double exponential
#        init_guess = [1, 100, 500]
#        popt, pcov = curve_fit(exp_decay_double, bin_centers_norm[start_num+1:-50],
#                                         norm_counts[start_num+1:-50], p0=init_guess)
#        
#        text_popt = r'{}: $A_0 = {}, d1 = {} us , d2 = {} us$'.format(label_list[f],'%.3f'%popt[0],'%.1f'%popt[1],
#                      '%.1f'%popt[2])
#        ax.text(0.55, 0.75 - t_ind, text_popt, transform=ax.transAxes, fontsize=12,
#            verticalalignment='top', bbox=props)
#        time_linspace = numpy.linspace(bin_centers_norm[start_num+1], bin_centers_norm[-50], 1000)
#        ax.plot(time_linspace, exp_decay_double(time_linspace, *popt),fmt_fit_list[f])

        
        ax.plot(bin_centers_norm[start_num:], counts[start_num:],'.' ,label=label_list[f])

        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Counts (arb.)')
        ax.set_title(title)
        ax.legend()
        ax.set_yscale("log", nonposy='clip')
#        t_ind = t_ind + 0.05

    
#    fig_decay, ax= plt.subplots(1, 1, figsize=(10, 8))
#    ax.plot(voltage_list_1, d_list, 'o')
#    ax.set_xlabel('gate voltage (V)')
#    ax.set_ylabel('decay constant (us)')
#    ax.set_title(title + ', single exponential decay constant')
    
    
def main():
#    directory_sept = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_09'
    
    directory = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_11'
    
#    file_sample_removed_aug = '2020_08_03-15_38_07-5nmEr-search'
#    with open(directory_aug + '/'+ file_sample_removed_aug + '.txt') as json_file:
#        data = json.load(json_file)
#        background_counts = numpy.array(data["binned_samples"])
        
    # No filter
#    file_take_1 = '2020_11_03-14_28_48-5nmEr-nrg'
#    file_take_2 = '2020_11_03-17_37_12-5nmEr-nrg'
#    bkgd_file_list = ['2020_11_03-14_29_01-5nmEr-nrg',
#                      '2020_11_03-17_37_29-5nmEr-nrg',]
    
    # 550 Shortpass
#    file_take_1 = '2020_11_03-14_29_23-5nmEr-nrg'
#    file_take_2 = '2020_11_03-17_37_55-5nmEr-nrg'
#    bkgd_file_list = ['2020_11_03-14_29_37-5nmEr-nrg',
#                      '2020_11_03-17_38_08-5nmEr-nrg',]
    
    # 630 longpass
    file_take_1 = '2020_11_03-14_29_59-5nmEr-nrg'
    file_take_2 = '2020_11_03-17_38_34-5nmEr-nrg'
    bkgd_file_list = ['2020_11_03-14_30_13-5nmEr-nrg',
                      '2020_11_03-17_38_48-5nmEr-nrg',]
    
    file_list = [file_take_1, file_take_2]

    # Make list for the data
    
    start_num = [0, 0, 0]
    counts_list = []
    bin_center_list =[]
    data_fmt_list = ['b.','k.', 'g.']
#    fit_fmt_list=['b--','k--']
    directory_list = [directory, directory, directory]
    label_list = ['-0.2 V', '-1.0 V']
    
    fig_fit, ax= plt.subplots(1, 1, figsize=(10, 8))
    
    # Open the specified file
    for i in range(len(file_list)):
        file = file_list[i]
        directory = directory_list[i]
        with open(directory + '/'+ file + '.txt') as json_file:
        
            # Load the data from the file
            data = json.load(json_file)
            counts = numpy.array(data["binned_samples"])
            bin_centers = numpy.array(data["bin_centers"])/10**3
            
        bkgd_file = bkgd_file_list[i]
        with open(directory + '/'+ bkgd_file + '.txt') as json_file:
            bkgd_data = json.load(json_file)
            bkgd_counts = numpy.array(bkgd_data["binned_samples"])
                
        counts_list.append(counts)
        bin_center_list.append(bin_centers)
        
        bin_centers = bin_center_list[i]
        bin_centers_norm = numpy.array(bin_centers)-bin_centers[start_num[i]]
        
        counts = numpy.array(counts_list[i])
        ###################################
#        counts = counts - bkgd_counts
#        
#        first_point = counts[start_num[i]]
#        last_point = numpy.average(counts[-10:]) #counts[-1]
#        norm_counts = (counts - last_point) / (first_point - last_point)
        ax.plot(bin_centers_norm[start_num[i]:], counts[start_num[i]:], data_fmt_list[i],label=label_list[i])
    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Counts (arb.)')
    ax.set_title('5 nm Er nanoribbons (670 bandpass filter)')
    ax.legend()
#    ax.set_xlim([0,500])
    ax.set_yscale("log", nonposy='clip')

def Er_graphene_sheet_nr():
    file_dir = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_11'
    label_list = [
                   # '-0.1 V',
                  # '-0.2 V (CNP)', 
                  # '-0.3 V', 
                  # '-0.4 V', 
                  # '-0.5 V',
                  # '-0.6 V', 
                  # '-0.7 V',
                  # '-0.8 V', 
                  # '-0.9 V', 
                  # '-1.0 V',  
                  # '-1.1 V', 
                  # '-1.2 V', 
                  # '-1.3 V',
                  # '-1.4 V', 
                  # '-1.5 V', 
                  # '-1.6 V', 
                  # '-1.7 V', 
                  # '-1.8 V', 
                   
                  ]
    
    # no filter
    file_list = [
        # '2020_11_03-14_07_39-5nmEr-nrg',
                 '2020_11_03-14_28_48-5nmEr-nrg',
                 # '2020_11_03-14_49_57-5nmEr-nrg',
                 # '2020_11_03-15_21_07-5nmEr-nrg',
                 '2020_11_03-15_42_25-5nmEr-nrg',
                 # '2020_11_03-15_59_00-5nmEr-nrg',
                 # '2020_11_03-16_17_37-5nmEr-nrg',
                 # '2020_11_03-16_37_59-5nmEr-nrg',
                 # '2020_11_03-16_57_00-5nmEr-nrg',
                 '2020_11_03-17_37_12-5nmEr-nrg',
                 # '2020_11_03-18_01_13-5nmEr-nrg',
                 # '2020_11_03-18_35_17-5nmEr-nrg',
                 # '2020_11_03-18_54_11-5nmEr-nrg',
                 # '2020_11_03-19_10_18-5nmEr-nrg',
                 '2020_11_03-19_24_12-5nmEr-nrg',
                 # '2020_11_03-19_39_26-5nmEr-nrg',
                 # '2020_11_03-19_54_28-5nmEr-nrg',
                 '2020_11_03-20_09_51-5nmEr-nrg',
                 
                 ]
    background_file_list = [
        # '2020_11_03-14_07_51-5nmEr-nrg',
                            '2020_11_03-14_29_01-5nmEr-nrg',
                            # '2020_11_03-14_50_10-5nmEr-nrg',
                            # '2020_11_03-15_21_20-5nmEr-nrg',
                            '2020_11_03-15_42_38-5nmEr-nrg',
                            # '2020_11_03-15_59_13-5nmEr-nrg',
                            # '2020_11_03-16_17_51-5nmEr-nrg',
                            # '2020_11_03-16_38_12-5nmEr-nrg',
                            # '2020_11_03-16_57_13-5nmEr-nrg',
                            '2020_11_03-17_37_29-5nmEr-nrg',
                            # '2020_11_03-18_01_27-5nmEr-nrg',
                            # '2020_11_03-18_35_32-5nmEr-nrg',
                            # '2020_11_03-18_54_25-5nmEr-nrg',
                            # '2020_11_03-19_10_32-5nmEr-nrg',
                            '2020_11_03-19_24_27-5nmEr-nrg',
                            # '2020_11_03-19_39_41-5nmEr-nrg',
                            # '2020_11_03-19_54_42-5nmEr-nrg',
                            '2020_11_03-20_10_04-5nmEr-nrg',
                            
                            ]
    title = '5 nm Er graphene nanoribbons, no filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #560 bandpass
    file_list = [
        # '2020_11_03-14_08_13-5nmEr-nrg',
                 '2020_11_03-14_29_23-5nmEr-nrg',
                 # '2020_11_03-14_50_45-5nmEr-nrg',
                 # '2020_11_03-15_21_43-5nmEr-nrg',
                 '2020_11_03-15_43_00-5nmEr-nrg',
                 # '2020_11_03-15_59_36-5nmEr-nrg',
                 # '2020_11_03-16_18_13-5nmEr-nrg',
                 # '2020_11_03-16_38_35-5nmEr-nrg',
                 # '2020_11_03-16_57_36-5nmEr-nrg',
                 '2020_11_03-17_37_55-5nmEr-nrg',
                 # '2020_11_03-18_01_49-5nmEr-nrg',
                 # '2020_11_03-18_35_57-5nmEr-nrg',
                 # '2020_11_03-18_54_48-5nmEr-nrg',
                 # '2020_11_03-19_10_54-5nmEr-nrg',
                 '2020_11_03-19_24_50-5nmEr-nrg',
                 # '2020_11_03-19_40_04-5nmEr-nrg',
                 # '2020_11_03-19_55_05-5nmEr-nrg',
                 '2020_11_03-20_10_26-5nmEr-nrg',
                 
                 ]
    background_file_list = [
        # '2020_11_03-14_08_26-5nmEr-nrg',
                            '2020_11_03-14_29_37-5nmEr-nrg',
                            # '2020_11_03-14_50_32-5nmEr-nrg',
                            # '2020_11_03-15_21_57-5nmEr-nrg',
                            '2020_11_03-15_43_14-5nmEr-nrg',
                            # '2020_11_03-15_59_49-5nmEr-nrg',
                            # '2020_11_03-16_18_27-5nmEr-nrg',
                            # '2020_11_03-16_38_50-5nmEr-nrg',
                            # '2020_11_03-16_57_49-5nmEr-nrg',
                            '2020_11_03-17_38_08-5nmEr-nrg',
                            # '2020_11_03-18_02_03-5nmEr-nrg',
                            # '2020_11_03-18_36_12-5nmEr-nrg',
                            # '2020_11_03-18_55_02-5nmEr-nrg',
                            # '2020_11_03-19_11_08-5nmEr-nrg',
                            '2020_11_03-19_25_04-5nmEr-nrg',
                            # '2020_11_03-19_40_18-5nmEr-nrg',
                            # '2020_11_03-19_55_19-5nmEr-nrg',
                            '2020_11_03-20_10_39-5nmEr-nrg',
                            
                            ]
    title = '5 nm Er graphene nanoribbons, 560 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #670 bandpass 
    file_list = [
        # '2020_11_03-14_08_47-5nmEr-nrg',
                 '2020_11_03-14_29_59-5nmEr-nrg',
                 # '2020_11_03-14_51_06-5nmEr-nrg',
                 # '2020_11_03-15_21_57-5nmEr-nrg',
                 '2020_11_03-15_43_35-5nmEr-nrg',
                 # '2020_11_03-16_00_11-5nmEr-nrg',
                 # '2020_11_03-16_18_49-5nmEr-nrg',
                 # '2020_11_03-16_39_12-5nmEr-nrg',
                 # '2020_11_03-16_58_11-5nmEr-nrg',
                 '2020_11_03-17_38_34-5nmEr-nrg',
                 # '2020_11_03-18_02_25-5nmEr-nrg',
                 # '2020_11_03-18_36_37-5nmEr-nrg',
                 # '2020_11_03-18_55_26-5nmEr-nrg',
                 # '2020_11_03-19_11_32-5nmEr-nrg',
                 '2020_11_03-19_25_29-5nmEr-nrg',
                 # '2020_11_03-19_40_43-5nmEr-nrg',
                 # '2020_11_03-19_55_44-5nmEr-nrg',
                 '2020_11_03-20_11_03-5nmEr-nrg',
                 
                 ]
    background_file_list =[
        # '2020_11_03-14_09_00-5nmEr-nrg',
                           '2020_11_03-14_30_13-5nmEr-nrg',
                           # '2020_11_03-14_51_19-5nmEr-nrg',
                           # '2020_11_03-15_22_19-5nmEr-nrg',
                           '2020_11_03-15_43_49-5nmEr-nrg',
                           # '2020_11_03-16_00_25-5nmEr-nrg',
                           # '2020_11_03-16_19_03-5nmEr-nrg',
                           # '2020_11_03-16_39_27-5nmEr-nrg',
                           # '2020_11_03-16_58_25-5nmEr-nrg',
                           '2020_11_03-17_38_48-5nmEr-nrg',
                           # '2020_11_03-18_02_40-5nmEr-nrg',
                           # '2020_11_03-18_36_53-5nmEr-nrg',
                           # '2020_11_03-18_55_41-5nmEr-nrg',
                           # '2020_11_03-19_11_47-5nmEr-nrg',
                           '2020_11_03-19_25_44-5nmEr-nrg',
                           # '2020_11_03-19_40_59-5nmEr-nrg',
                           # '2020_11_03-19_55_58-5nmEr-nrg',
                           '2020_11_03-20_11_16-5nmEr-nrg',
                           
                           ]
    title = '5 nm Er graphene nanoribbons, 670 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
def Er_graphene_sheet_nr_2():
    file_dir = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_11'
    label_list = [
                   '-1.8 V', 
                   '-1.5 V', 
                   '-1.0 V', 
                   '-0.5 V',
                   '0.0 V (CNP)'
                   
                  ]
    
    # no filter
    file_list = ['2020_11_03-20_09_51-5nmEr-nrg',
                   '2020_11_03-20_30_06-5nmEr-nrg',
                   '2020_11_03-20_50_00-5nmEr-nrg',
                   '2020_11_03-21_08_50-5nmEr-nrg',
                   '2020_11_03-21_24_56-5nmEr-nrg'
                 ]
    background_file_list = [
                            
                 '2020_11_03-20_10_04-5nmEr-nrg',
                            '2020_11_03-20_30_20-5nmEr-nrg',
                            '2020_11_03-20_50_14-5nmEr-nrg',
                            '2020_11_03-21_09_03-5nmEr-nrg',
                            '2020_11_03-21_25_09-5nmEr-nrg'
                            ]
    title = '5 nm Er graphene nanoribbons, no filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #560 bandpass
    file_list = [
                 
                            '2020_11_03-20_10_26-5nmEr-nrg',
                            '2020_11_03-20_30_43-5nmEr-nrg',
                            '2020_11_03-20_50_38-5nmEr-nrg',
                            '2020_11_03-21_09_25-5nmEr-nrg',
                            '2020_11_03-21_25_30-5nmEr-nrg'
                 ]
    background_file_list = [
                            
                 '2020_11_03-20_10_39-5nmEr-nrg',
                 '2020_11_03-20_30_57-5nmEr-nrg',
                 '2020_11_03-20_50_52-5nmEr-nrg',
                 '2020_11_03-21_09_38-5nmEr-nrg',
                 '2020_11_03-21_25_43-5nmEr-nrg'
                            ]
    title = '5 nm Er graphene nanoribbons, 560 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #670 bandpass 
    file_list = [
                 '2020_11_03-20_11_03-5nmEr-nrg',
                 '2020_11_03-20_31_21-5nmEr-nrg',
                 '2020_11_03-20_51_16-5nmEr-nrg',
                 '2020_11_03-21_10_01-5nmEr-nrg',
                 '2020_11_03-21_26_05-5nmEr-nrg'
                 ]
    background_file_list =['2020_11_03-20_11_16-5nmEr-nrg',
                           '2020_11_03-20_31_35-5nmEr-nrg',
                           '2020_11_03-20_51_30-5nmEr-nrg',
                           '2020_11_03-21_10_15-5nmEr-nrg',
                           '2020_11_03-21_26_19-5nmEr-nrg'
                           ]
    title = '5 nm Er graphene nanoribbons, 670 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
def Er_graphene_sheet_nr_3():
    file_dir = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_11'
    label_list = [
                   '0.0 V (CNP)',
                   '-1.0 V',
                   '-1.8 V'
                   
                  ]
    
    # no filter
    file_list = ['2020_11_03-21_24_56-5nmEr-nrg',
                 '2020_11_03-21_43_40-5nmEr-nrg',
                 '2020_11_03-22_07_04-5nmEr-nrg',
                   
                 ]
    background_file_list = ['2020_11_03-21_25_09-5nmEr-nrg',
                            '2020_11_03-21_43_52-5nmEr-nrg',
                            '2020_11_03-22_07_18-5nmEr-nrg',
                            ]
    title = '5 nm Er graphene nanoribbons, no filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #560 bandpass
    file_list = ['2020_11_03-21_25_30-5nmEr-nrg',
                 '2020_11_03-21_44_14-5nmEr-nrg',
                 '2020_11_03-22_07_41-5nmEr-nrg',
                 ]
    background_file_list = ['2020_11_03-21_25_43-5nmEr-nrg',
                            '2020_11_03-21_44_28-5nmEr-nrg',
                            '2020_11_03-22_07_54-5nmEr-nrg',
                            ]
    title = '5 nm Er graphene nanoribbons, 560 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #670 bandpass 
    file_list = ['2020_11_03-21_26_05-5nmEr-nrg',
                 '2020_11_03-21_44_50-5nmEr-nrg',
                 '2020_11_03-22_08_17-5nmEr-nrg',
                 ]
    background_file_list =['2020_11_03-21_26_19-5nmEr-nrg',
                           '2020_11_03-21_45_04-5nmEr-nrg',
                           '2020_11_03-22_08_32-5nmEr-nrg'
                           ]
    title = '5 nm Er graphene nanoribbons, 670 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
#%%
    
if __name__ == '__main__':
    Er_graphene_sheet_nr_2()
    Er_graphene_sheet_nr_3()
#    main()


    
    
    
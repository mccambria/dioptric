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
    start_num = 5
    
    fig, ax= plt.subplots(1, 1, figsize=(10, 8))
#    fmt_data_list = ['b.', 'y.', 'g.']
#    fmt_fit_list = ['b-', 'y-', 'g-']
    
    text_eq = r'$A_0 (e^{-t/d1})$'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.55, 0.8, text_eq, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
    
    d_list = []
    voltage_list_1 = [-0.5,0, 0.5, 1, 1.5, 2, 2.5, 3]
#    voltage_list_2  = [0.1, 1, 2, 3] 
#    voltage_list_3 = [0.1, 2.2, 4, -3]
    t_ind = 0
    
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
        init_guess = [1, 100]
        popt, pcov = curve_fit(exp_decay, bin_centers_norm[start_num:-66],
                                         norm_counts[start_num:-66], p0=init_guess)
        
        text_popt = r'{}: $A_0 = {}, d1 = {} us$'.format(label_list[f],'%.3f'%popt[0],'%.1f'%popt[1])
        ax.text(0.55, 0.75 - t_ind, text_popt, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
        time_linspace = numpy.linspace(bin_centers_norm[start_num], bin_centers_norm[-66], 1000)
        ax.plot(time_linspace, exp_decay(time_linspace, *popt),'-')        
        d_list.append(popt[1])
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

        
        ax.plot(bin_centers_norm[start_num:], norm_counts[start_num:],'.' ,label=label_list[f])

        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Counts (arb.)')
        ax.set_title(title)
        ax.legend()
        ax.set_yscale("log", nonposy='clip')
        t_ind = t_ind + 0.05
    print(title)
    print(d_list)
    
    fig_decay, ax= plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(voltage_list_1, d_list, 'o')
    ax.set_xlabel('gate voltage (V)')
    ax.set_ylabel('decay constant (us)')
    ax.set_title(title + ', single exponential decay constant')
    
    
def main():
    directory_sept = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_09'
    
    directory = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_08'
    
#    file_sample_removed_aug = '2020_08_03-15_38_07-5nmEr-search'
#    with open(directory_aug + '/'+ file_sample_removed_aug + '.txt') as json_file:
#        data = json.load(json_file)
#        background_counts = numpy.array(data["binned_samples"])
        
    # No filter
#    file_take_1 = '2020_09_22-12_18_14-5nmEr-graphene_sheet'
#    file_take_2 = '2020_09_22-15_58_27-5nmEr-graphene_sheet'
#    file_take_3 = '2020_09_22-18_12_25-5nmEr-graphene_sheet'
#    bkgd_file_list = ['2020_09_22-12_18_30-5nmEr-graphene_sheet',
#                      '2020_09_22-15_58_42-5nmEr-graphene_sheet',
#                      '2020_09_22-18_12_41-5nmEr-graphene_sheet',]
    
    # 550 Shortpass
#    file_take_1 = '2020_09_22-12_18_57-5nmEr-graphene_sheet'
#    file_take_2 = '2020_09_22-15_59_08-5nmEr-graphene_sheet'
#    file_take_3 = '2020_09_22-18_05_17-5nmEr-graphene_sheet'
#    bkgd_file_list = ['2020_09_22-12_19_13-5nmEr-graphene_sheet',
#                      '2020_09_22-15_59_24-5nmEr-graphene_sheet',
#                      '2020_09_22-18_05_33-5nmEr-graphene_sheet',]
    
    # 630 longpass
    file_take_1 = '2020_09_22-12_19_38-5nmEr-graphene_sheet'
    file_take_2 = '2020_09_22-15_59_47-5nmEr-graphene_sheet'
    file_take_3 = '2020_09_22-18_05_58-5nmEr-graphene_sheet'
    bkgd_file_list = ['2020_09_22-15_05_55-5nmEr-graphene_sheet',
                      '2020_09_22-16_00_04-5nmEr-graphene_sheet',
                      '2020_09_22-18_06_15-5nmEr-graphene_sheet',]
    
    file_list = [file_take_1, file_take_2, file_take_3]

    # Make list for the data
    
    start_num = [4,4, 4]
    counts_list = []
    bin_center_list =[]
    data_fmt_list = ['b.','k.', 'g.']
#    fit_fmt_list=['b--','k--']
    directory_list = [directory_sept, directory_sept, directory_sept]
    label_list = ['Take 1 CNP (-0.4 V)', 'Take 2 CNP (0.0 V)', 'Take 3 CNP (+0.4 V)']
    
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
        counts = counts - bkgd_counts
        
        first_point = counts[start_num[i]]
        last_point = numpy.average(counts[-10:]) #counts[-1]
        norm_counts = (counts - last_point) / (first_point - last_point)
        ax.plot(bin_centers_norm[start_num[i]:], norm_counts[start_num[i]:], data_fmt_list[i],label=label_list[i])
    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Counts (arb.)')
    ax.set_title('5 nm Er CNP comparison (670 nm bandpass filter)')
    ax.legend()
#    ax.set_xlim([0,500])
    ax.set_yscale("log", nonposy='clip')

def Er_graphene_sheet_1():
    file_dir = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_09'
    label_list = ['-0.5 V (CNP)',
                  '0.0 V', '+0.5 V', 
                    '+1.0 V', '+1.5 V', 
                    '+2.0 V', 
                    '+2.5 V', 
                  '+3.0 V'
                  ]
    
    # no filter
    file_list = ['2020_09_22-12_18_14-5nmEr-graphene_sheet', 
                 '2020_09_22-12_41_19-5nmEr-graphene_sheet',
                 '2020_09_22-12_59_34-5nmEr-graphene_sheet', 
                    '2020_09_22-13_18_54-5nmEr-graphene_sheet',
                 '2020_09_22-13_42_33-5nmEr-graphene_sheet',
                 '2020_09_22-14_08_07-5nmEr-graphene_sheet',
                 '2020_09_22-14_29_18-5nmEr-graphene_sheet', 
                 '2020_09_22-15_04_13-5nmEr-graphene_sheet',
                 ]
    background_file_list = ['2020_09_22-12_18_30-5nmEr-graphene_sheet', 
                            '2020_09_22-12_41_34-5nmEr-graphene_sheet',
                            '2020_09_22-12_59_50-5nmEr-graphene_sheet', 
                            '2020_09_22-13_19_11-5nmEr-graphene_sheet',
                            '2020_09_22-13_42_49-5nmEr-graphene_sheet', 
                            '2020_09_22-14_08_24-5nmEr-graphene_sheet',
                            '2020_09_22-14_29_34-5nmEr-graphene_sheet', 
                            '2020_09_22-15_04_30-5nmEr-graphene_sheet',
                            ]
    title = '5 nm Er graphene sheet, no filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #560 bandpass
    file_list = ['2020_09_22-12_18_57-5nmEr-graphene_sheet', 
                 '2020_09_22-12_42_01-5nmEr-graphene_sheet',
                 '2020_09_22-13_00_17-5nmEr-graphene_sheet', 
                    '2020_09_22-13_19_38-5nmEr-graphene_sheet',
                 '2020_09_22-13_43_14-5nmEr-graphene_sheet', 
                           '2020_09_22-14_05_14-5nmEr-graphene_sheet',
                 '2020_09_22-14_30_02-5nmEr-graphene_sheet', 
                 '2020_09_22-15_04_57-5nmEr-graphene_sheet',
                 ]
    background_file_list = ['2020_09_22-12_19_13-5nmEr-graphene_sheet', 
                            '2020_09_22-12_42_17-5nmEr-graphene_sheet',
                            '2020_09_22-13_00_33-5nmEr-graphene_sheet', 
                            '2020_09_22-13_19_55-5nmEr-graphene_sheet',
                            '2020_09_22-13_43_30-5nmEr-graphene_sheet', 
                           '2020_09_22-14_05_31-5nmEr-graphene_sheet',
                            '2020_09_22-14_30_18-5nmEr-graphene_sheet', 
                            '2020_09_22-15_05_13-5nmEr-graphene_sheet',
                            ]
    title = '5 nm Er graphene sheet, 560 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #670 bandpass 
    file_list = ['2020_09_22-12_19_38-5nmEr-graphene_sheet', 
                 '2020_09_22-12_42_40-5nmEr-graphene_sheet',
                 '2020_09_22-13_00_57-5nmEr-graphene_sheet', 
                '2020_09_22-13_20_19-5nmEr-graphene_sheet',
                 '2020_09_22-13_43_55-5nmEr-graphene_sheet', 
                           '2020_09_22-14_05_56-5nmEr-graphene_sheet',
                 '2020_09_22-14_30_43-5nmEr-graphene_sheet', 
                 '2020_09_22-15_05_38-5nmEr-graphene_sheet',
                 ]
    background_file_list =['2020_09_22-12_19_55-5nmEr-graphene_sheet', 
                           '2020_09_22-12_42_57-5nmEr-graphene_sheet',
                           '2020_09_22-13_01_14-5nmEr-graphene_sheet', 
                            '2020_09_22-13_20_36-5nmEr-graphene_sheet',
                           '2020_09_22-13_44_12-5nmEr-graphene_sheet', 
                            '2020_09_22-14_06_13-5nmEr-graphene_sheet',
                           '2020_09_22-14_30_59-5nmEr-graphene_sheet', 
                           '2020_09_22-15_05_55-5nmEr-graphene_sheet',
                           ]
    title = '5 nm Er graphene sheet, 670 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
def Er_graphene_sheet_2():
    file_dir = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_09'
    label_list = ['+0.1 V (CNP)', '+1.0 V', '+2.0 V', '+3.0 V']
    
    # no filter
    file_list = ['2020_09_22-15_58_27-5nmEr-graphene_sheet' , '2020_09_22-16_33_30-5nmEr-graphene_sheet',
                 '2020_09_22-16_56_56-5nmEr-graphene_sheet', '2020_09_22-17_21_44-5nmEr-graphene_sheet',
                 ]
    background_file_list = ['2020_09_22-15_58_42-5nmEr-graphene_sheet', '2020_09_22-16_33_46-5nmEr-graphene_sheet',
                            '2020_09_22-16_57_12-5nmEr-graphene_sheet', '2020_09_22-17_22_00-5nmEr-graphene_sheet',
                            ]
    title = '5 nm Er graphene sheet, no filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #560 bandpass
    file_list = ['2020_09_22-15_59_08-5nmEr-graphene_sheet' , '2020_09_22-16_34_12-5nmEr-graphene_sheet',
                 '2020_09_22-16_57_39-5nmEr-graphene_sheet', '2020_09_22-17_22_27-5nmEr-graphene_sheet',
                 ]
    background_file_list = ['2020_09_22-15_59_24-5nmEr-graphene_sheet', '2020_09_22-16_34_28-5nmEr-graphene_sheet',
                            '2020_09_22-16_57_55-5nmEr-graphene_sheet', '2020_09_22-17_22_44-5nmEr-graphene_sheet',
                            ]
    title = '5 nm Er graphene sheet, 560 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #670 bandpass 
    file_list = ['2020_09_22-15_59_47-5nmEr-graphene_sheet', '2020_09_22-16_34_52-5nmEr-graphene_sheet',
                 '2020_09_22-16_58_19-5nmEr-graphene_sheet', '2020_09_22-17_23_08-5nmEr-graphene_sheet',
                 ]
    background_file_list =['2020_09_22-16_00_04-5nmEr-graphene_sheet', '2020_09_22-16_35_09-5nmEr-graphene_sheet',
                           '2020_09_22-16_58_35-5nmEr-graphene_sheet', '2020_09_22-17_23_25-5nmEr-graphene_sheet',
                           ]
    title = '5 nm Er graphene sheet, 670 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
def Er_graphene_sheet_3():
    file_dir = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_09'
    label_list = ['+0.1 V (CNP)', '+2.2 V', '+4.0 V', '-3.0 V']
    
    # no filter
    file_list = ['2020_09_22-18_12_25-5nmEr-graphene_sheet', '2020_09_22-18_34_53-5nmEr-graphene_sheet',
                 '2020_09_22-19_04_51-5nmEr-graphene_sheet', '2020_09_22-19_34_29-5nmEr-graphene_sheet']
    background_file_list = ['2020_09_22-18_12_41-5nmEr-graphene_sheet','2020_09_22-18_35_09-5nmEr-graphene_sheet',
                            '2020_09_22-19_05_07-5nmEr-graphene_sheet', '2020_09_22-19_34_46-5nmEr-graphene_sheet',]
    title = '5 nm Er graphene sheet, no filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #560 bandpass
    file_list = ['2020_09_22-18_13_07-5nmEr-graphene_sheet' , '2020_09_22-18_35_37-5nmEr-graphene_sheet',
                 '2020_09_22-19_05_35-5nmEr-graphene_sheet', '2020_09_22-19_35_14-5nmEr-graphene_sheet']
    background_file_list = ['2020_09_22-18_13_23-5nmEr-graphene_sheet', '2020_09_22-18_35_54-5nmEr-graphene_sheet',
                            '2020_09_22-19_05_52-5nmEr-graphene_sheet', '2020_09_22-19_35_31-5nmEr-graphene_sheet']
    title = '5 nm Er graphene sheet, 560 nm bandpass filter'  
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #670 bandpass 
    file_list = ['2020_09_22-18_13_47-5nmEr-graphene_sheet', '2020_09_22-18_36_18-5nmEr-graphene_sheet',
                 '2020_09_22-19_06_16-5nmEr-graphene_sheet', '2020_09_22-19_35_55-5nmEr-graphene_sheet']
    background_file_list =['2020_09_22-18_14_03-5nmEr-graphene_sheet', '2020_09_22-18_36_35-5nmEr-graphene_sheet',
                           '2020_09_22-19_06_33-5nmEr-graphene_sheet', '2020_09_22-19_36_11-5nmEr-graphene_sheet']
    title = '5 nm Er graphene sheet, 670 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
def Er_preparation():
    file_dir = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_09'
    label_list = ['post-anneal', 'post-graphene', 'post-ionic gel']
    
    # no filter
    file_list = ['2020_09_16-15_09_21-5nmEr-annealed', '2020_09_17-13_39_17-5nmEr-graphene',
                 '2020_09_22-09_46_06-5nmEr-ionic_gel']
    background_file_list = ['2020_09_16-15_09_36-5nmEr-annealed','2020_09_17-13_39_33-5nmEr-graphene',
                            '2020_09_22-09_46_22-5nmEr-ionic_gel']
    title = '5 nm Er graphene sheet, no filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #560 bandpass
    file_list = ['2020_09_16-15_09_57-5nmEr-annealed', '2020_09_17-13_39_56-5nmEr-graphene',
                 '2020_09_22-09_46_44-5nmEr-ionic_gel']
    background_file_list = ['2020_09_16-15_10_13-5nmEr-annealed','2020_09_17-13_40_13-5nmEr-graphene',
                            '2020_09_22-09_47_00-5nmEr-ionic_gel']
    title = '5 nm Er graphene sheet, 560 nm bandpass filter'  
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #670 bandpass 
    file_list = ['2020_09_16-15_10_33-5nmEr-annealed', '2020_09_17-13_40_32-5nmEr-graphene',
                 '2020_09_22-09_47_19-5nmEr-ionic_gel']
    background_file_list = ['2020_09_16-15_10_49-5nmEr-annealed','2020_09_17-13_40_49-5nmEr-graphene',
                            '2020_09_22-09_47_36-5nmEr-ionic_gel']
    title = '5 nm Er graphene sheet, 670 nm bandpass filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
#%%
    
if __name__ == '__main__':
    Er_graphene_sheet_1()
#    main()


    
    
    
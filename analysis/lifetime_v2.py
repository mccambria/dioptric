# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:13:01 2019

@author: Aedan
"""
import numpy
import json
import matplotlib.pyplot as plt

    
# %%

def plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list= None ):
    start_num = 4
    
    fig, ax= plt.subplots(1, 1, figsize=(10, 8))
    
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

#        counts = counts - bkgd_counts
        
        first_point = counts[start_num]
        last_point = counts[-1]
        norm_counts = (counts - last_point) / (first_point - last_point)
        ax.plot(bin_centers_norm[start_num:], norm_counts[start_num:],'o',label=label_list[f])
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Counts (arb.)')
        ax.set_title(title)
        ax.legend()
        ax.set_yscale("log", nonposy='clip')
    
def main():
    directory_mar = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_03'
    
    directory = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_08'
    
#    file_sample_removed_aug = '2020_08_03-15_38_07-5nmEr-search'
#    with open(directory_aug + '/'+ file_sample_removed_aug + '.txt') as json_file:
#        data = json.load(json_file)
#        background_counts = numpy.array(data["binned_samples"])
        
    # No filter
#    file_preanneal = '2020_03_09-15_46_02-Y2O3_no_graphene_no_IG_2'
#    file_postanneal = '2020_08_14-09_52_49-5nmEr-noncapped'
    
    # 550 Shortpass
#    file_MM = '2020_08_25-09_29_04-5nmEr-noncapped'
#    file_SM = '2020_08_21-14_15_44-5nmEr-noncapped'
#    bkgd_file_list = ['2020_08_25-09_32_44-5nmEr-noncapped',
#                      '2020_08_21-14_16_01-5nmEr-noncapped']
    # 630 longpass
    file_MM = '2020_08_25-09_45_48-5nmEr-noncapped'
    file_SM = '2020_08_21-14_16_20-5nmEr-noncapped'
    bkgd_file_list = ['2020_08_25-09_58_35-5nmEr-noncapped',
                      '2020_08_21-14_16_37-5nmEr-noncapped']
    
    file_list = [file_MM, file_SM]

    # Make list for the data
    
    start_num = [4,4]
    counts_list = []
    bin_center_list =[]
    data_fmt_list = ['bo','ko']
#    fit_fmt_list=['b--','k--']
    directory_list = [directory, directory]
    label_list = ['Multimode fiber', 'Singlemode fiber']
    
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
    ax.set_title('5 nm Er fiber comparison w/ subtraction, longpass (670 nm) filter')
    ax.legend()
#    ax.set_xlim([0,500])
    ax.set_yscale("log", nonposy='clip')

def august_cap_noncap_plots():
    file_dir = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2020_08'
    label_list = ['8/10/2020', '8/11/2020', '8/12/2020', '8/13/2020', '8/14/2020', '8/21/2020']
    
    # capped_file_list_nf
    file_list = ['2020_08_10-17_28_02-5nmEr-capped', '2020_08_11-09_59_01-5nmEr-capped',
                 '2020_08_12-10_32_04-5nmEr-capped', '2020_08_13-08_50_54-5nmEr-capped',
                 '2020_08_14-12_18_35-5nmEr-capped', '2020_08_21-10_38_07-5nmEr-capped']
    background_file_list = ['2020_08_10-17_28_18-5nmEr-capped', '2020_08_11-09_59_17-5nmEr-capped',
                            '2020_08_12-10_32_19-5nmEr-capped', '2020_08_13-08_51_09-5nmEr-capped',
                            '2020_08_14-12_18_50-5nmEr-capped', '2020_08_21-10_38_22-5nmEr-capped']
    title = '5 nm Er, capped, no filter'    
#    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    # noncapped_file_list_nf 
    file_list = ['2020_08_10-16_57_03-5nmEr-noncapped', '2020_08_11-09_24_27-5nmEr-noncapped',
                 '2020_08_12-12_11_08-5nmEr-noncapped-center', '2020_08_13-08_18_55-5nmEr-noncapped',
                 '2020_08_14-09_52_49-5nmEr-noncapped', '2020_08_21-14_15_07-5nmEr-noncapped']
    background_file_list = ['2020_08_10-16_57_18-5nmEr-noncapped', '2020_08_11-09_24_43-5nmEr-noncapped',
                            '2020_08_12-12_09_30-5nmEr-noncapped-center', '2020_08_13-08_19_10-5nmEr-noncapped',
                            '2020_08_14-09_53_03-5nmEr-noncapped', '2020_08_21-14_15_23-5nmEr-noncapped', ]
    title = '5 nm Er, noncapped, no filter'    
    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #capped_file_list_sp 
    file_list = ['2020_08_10-17_29_15-5nmEr-capped', '2020_08_11-10_00_45-5nmEr-capped',
                 '2020_08_12-10_32_39-5nmEr-capped', '2020_08_13-08_51_31-5nmEr-capped',
                 '2020_08_14-12_19_20-5nmEr-capped', '2020_08_21-10_38_50-5nmEr-capped']
    background_file_list = ['2020_08_10-17_29_32-5nmEr-capped', '2020_08_11-10_01_03-5nmEr-capped',
                            '2020_08_12-10_32_55-5nmEr-capped', '2020_08_13-08_51_49-5nmEr-capped',
                            '2020_08_14-12_19_37-5nmEr-capped', '2020_08_21-10_39_06-5nmEr-capped']
    title = '5 nm Er, capped, shortpass filter'    
#    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    #noncapped_file_list_sp 
    file_list = ['2020_08_10-16_57_40-5nmEr-noncapped', '2020_08_11-09_25_46-5nmEr-noncapped',
                 '2020_08_12-12_09_49-5nmEr-noncapped-center', '2020_08_13-08_19_29-5nmEr-noncapped',
                 '2020_08_14-09_53_29-5nmEr-noncapped', '2020_08_21-14_15_44-5nmEr-noncapped']
    background_file_list = ['2020_08_10-16_57_56-5nmEr-noncapped', '2020_08_11-09_26_04-5nmEr-noncapped',
                            '2020_08_12-12_10_05-5nmEr-noncapped-center', '2020_08_13-08_19_46-5nmEr-noncapped',
                            '2020_08_14-09_53_44-5nmEr-noncapped', '2020_08_21-14_16_01-5nmEr-noncapped' ]
    title = '5 nm Er, noncapped, shortpass filter'    
#    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
    #capped_file_list_lp 
    file_list = ['2020_08_10-17_29_49-5nmEr-capped', '2020_08_11-10_01_20-5nmEr-capped',
                 '2020_08_12-10_33_12-5nmEr-capped', '2020_08_13-08_52_07-5nmEr-capped',
                 '2020_08_14-12_19_53-5nmEr-capped', '2020_08_21-10_39_24-5nmEr-capped']
    background_file_list =['2020_08_10-17_30_05-5nmEr-capped', '2020_08_11-10_01_36-5nmEr-capped',
                           '2020_08_12-10_33_28-5nmEr-capped', '2020_08_13-08_52_24-5nmEr-capped',
                           '2020_08_14-12_20_09-5nmEr-capped', '2020_08_21-10_39_41-5nmEr-capped']
    title = '5 nm Er, capped, longpass filter'    
#    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    #noncapped_file_list_lp 
    file_list = ['2020_08_10-16_58_13-5nmEr-noncapped', '2020_08_11-09_26_21-5nmEr-noncapped',
                 '2020_08_12-12_10_22-5nmEr-noncapped-center', '2020_08_13-08_20_03-5nmEr-noncapped',
                 '2020_08_14-09_54_01-5nmEr-noncapped', '2020_08_21-14_16_20-5nmEr-noncapped']
    background_file_list = ['2020_08_10-16_58_29-5nmEr-noncapped', '2020_08_11-09_26_37-5nmEr-noncapped',
                            '2020_08_12-12_10_38-5nmEr-noncapped-center', '2020_08_13-08_20_19-5nmEr-noncapped',
                            '2020_08_14-09_54_16-5nmEr-noncapped', '2020_08_21-14_16_37-5nmEr-noncapped' ]
    title = '5 nm Er, noncapped, longpass filter'    
#    plot_lifetime_list(file_list, file_dir, title,label_list, background_file_list )
    
#%%
    
if __name__ == '__main__':
#    august_cap_noncap_plots()
    main()


    
    
    
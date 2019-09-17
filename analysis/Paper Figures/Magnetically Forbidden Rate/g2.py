# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:48:13 2019

Making figures of g2 measuremments

@author: Aedan
"""
# %%

import numpy
import matplotlib.pyplot as plt
import json

# %%
def calculate_relative_g2_zero(hist):
    
    # We take the values on the wings to be representatives for g2(inf)
    # We take the wings to be the first and last 1/6 of collected data
    num_bins = len(hist)
    wing_length = num_bins // 12
    neg_wing = hist[0: wing_length]
    pos_wing = hist[num_bins - wing_length: ]
    inf_delay_differences = numpy.average([neg_wing, pos_wing])
    
    # Use the parity of num_bins to determine differences at 0 ns
    if num_bins % 2 == 0:
        # As an example, say there are 6 bins. Then we want the differences
        # from bins 2 and 3 (indexing starts from 0).
        midpoint_high = num_bins // 2
        zero_delay_differences = numpy.average(hist[midpoint_high - 1,
                                                    midpoint_high])
    else:
        # Now say there are 7 bins. We'd like bin 3. 
        midpoint = int(numpy.floor(num_bins / 2))
        zero_delay_differences = hist[midpoint]
        
    return zero_delay_differences / inf_delay_differences, inf_delay_differences

# %%
    
folder_name = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/g2_measurement'
file_name_list = [
                    '2019-05-10_12-32-26_ayrton12.txt', # NV1
                    '2019-04-30_12-43-18_ayrton12.txt', # NV2
                    '2019-07-25_21-36-31_ayrton12_nv16_2019_07_25.txt',  # NV16
                    'branch_time-tagger-counter/2019-06-04_18-36-53_ayrton12.txt', # NV0
                    '2019-06-10_22-52-19_ayrton12.txt' # NV13
                  ]

fig , axes = plt.subplots(3, 2, figsize=(16, 16))
r_ind = 0
c_ind = 0

for i in range(len(file_name_list)):
#    print(r_ind, c_ind)

    file_name = file_name_list[i]
    with open('{}/{}'.format(folder_name, file_name)) as file:
        data = json.load(file)
        differences = data['differences']
#        print(len(differences))
        
        try:
            num_bins = data['num_bins']
        except Exception:
            num_bins = 301
#        print(num_bins)
    hist, bin_edges = numpy.histogram(differences, num_bins)
    bin_center_offset = (bin_edges[1] - bin_edges[0]) / 2
    bin_centers = bin_edges[0: num_bins] + bin_center_offset
    
    g2_zero, inf_delay_diff = calculate_relative_g2_zero(hist)
    print(g2_zero)
    ax = axes[r_ind, c_ind]
    
    ax.plot(bin_centers / 1000, hist/inf_delay_diff, label = 'NV{}'.format(i+1))
    ax.tick_params(which = 'both', length=10, width=2, colors='k',
                    grid_alpha=0.7, labelsize = 20)
    plt.xlabel('Delay time (ns)', fontsize=20)
    plt.ylabel(r'$g^{2}(\tau)$', fontsize=20)
    
    #ax.tick_params(which = 'major', length=10, width=1)
    ax.set_ylim(bottom=0, top = 1.5)
    ax.legend(fontsize=20)
#    except Exception:
#        continue
    
    if c_ind == 0:
        c_ind = 1
    elif c_ind == 1:
        r_ind = r_ind +  1
        c_ind = 0
        
fig.delaxes(axes[2][1])

fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/g2_figure.pdf", bbox_inches='tight')
    


# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:48:36 2020

analysis file for lifetime data taken on Er doped Y2O3 sample, no graphene added
Compares the 5 nm and 10 nm samples.

This data was taken from 3/9/2020 to 3/10/2020

@author: agardill
"""
# %%
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import numpy
from scipy.optimize import curve_fit

# %%
data_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata'

folder = 'lifetime_v2/2020_03'

file_05_lp = '2020_03_09-16_05_16-Y2O3_no_graphene_no_IG_2'
file_05_sp = '2020_03_09-15_56_14-Y2O3_no_graphene_no_IG_2'
file_05_nf = '2020_03_09-15_46_02-Y2O3_no_graphene_no_IG_2'

file_05_IG_lp = '2020_03_09-17_35_20-Y2O3_no_graphene_yes_IG_2'
file_05_IG_sp = '2020_03_09-17_24_57-Y2O3_no_graphene_yes_IG_2'
file_05_IG_nf = '2020_03_09-17_14_21-Y2O3_no_graphene_yes_IG_2'

file_10_lp = '2020_03_10-12_21_38-Y2O3_no_graphene_no_IG_10nm'
file_10_sp = '2020_03_10-12_12_32-Y2O3_no_graphene_no_IG_10nm'
file_10_nf = '2020_03_10-12_03_13-Y2O3_no_graphene_no_IG_10nm'

file_10_IG_lp = '2020_03_10-15_33_14-Y2O3_no_graphene_yes_IG_10nm'
file_10_IG_sp = '2020_03_10-15_24_21-Y2O3_no_graphene_yes_IG_10nm'
file_10_IG_nf = '2020_03_10-15_12_30-Y2O3_no_graphene_yes_IG_10nm'


files = [file_05_lp,file_05_sp,file_05_nf,
         file_05_IG_lp, file_05_IG_sp, file_05_IG_nf,
        file_10_lp, file_10_sp, file_10_nf,
        file_10_IG_lp, file_10_IG_sp, file_10_IG_nf]

files = [file_05_lp,file_05_sp,file_05_nf, # no ionic gel
        file_10_lp, file_10_sp, file_10_nf]

files = [file_05_IG_lp, file_05_IG_sp, file_05_IG_nf,
        file_10_IG_lp, file_10_IG_sp, file_10_IG_nf]


count_list = []
bin_centers_list = []
text_list = []
#data_fmt_list = ['b.','k.','r.','g.', 'y.', 'm.', 'c.']
data_fmt_list = ['bo','ko','ro','go', 'yo', 'mo', 'co']
fit_fmt_list = ['b-','k-','r-','g-', 'y-', 'm-', 'c-']
#label_list = ['5 nm w/out ionic gel', '5 nm w/ ionic gel','10 nm w/out ionic gel', '10 nm w/ ionic gel']
#label_list = ['5 nm w/out ionic gel', '10 nm w/out ionic gel']
label_list = ['5 nm w/ ionic gel', '10 nm w/ ionic gel']
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

init_params_list_1 = [10,10]
init_params_list_2 = [10, 1, 1, 100]
init_params_list_3 = [30, 1e-1, 1, 10,  10**-2,80]

# %%

def decayExp(t, amplitude, decay):
    return amplitude * numpy.exp(- t / decay)

def double_decay(t, a1, d1, a2, d2):
    return decayExp(t, a1, d1) + decayExp(t, a2, d2)

def triple_decay(t, a1, d1, a2, d2, a3, d3):
    return decayExp(t, a1, d1) + decayExp(t, a2, d2) + decayExp(t, a3, d3)

def tetra_decay(t, a, d1, d2, d3, d4):
    return decayExp(t, a, d1) + decayExp(t, a, d2) + decayExp(t, a, d3) \
                + decayExp(t, a, d4)
                
# %%
def pol_counts(num_data, file_list, file_ind, title):
    fig, ax= plt.subplots(1, 1, figsize=(10, 8))
    first_point_list = []
    for i in range(num_data):
        f = i*3 + file_ind
        file = file_list[f]
        data = tool_belt.get_raw_data(folder, file)
        
        counts = numpy.array(data["binned_samples"])/10**5
#        bin_centers = numpy.array(data["bin_centers"])/10**3
        
        first_point_list.append(counts[0])
        
    ax.plot(label_list, first_point_list, 'bo')
    ax.set_xlabel('Gate Voltage')
    ax.set_ylabel('Counts (x 10^5)')
    ax.set_title(title)
        
    
        
    
def fig_lifetime(num_data, file_list, file_ind, title, fit_eq, fit_params):

    fig, ax= plt.subplots(1, 1, figsize=(10, 8))
    for i in range(num_data):
        f = i*3 + file_ind 
        file = file_list[f]
        data = tool_belt.get_raw_data(folder, file)
    
        counts = numpy.array(data["binned_samples"])
        bin_centers = numpy.array(data["bin_centers"])/10**3 - 200
        
        
        background = numpy.average(counts[75:101])
        norm_points = counts[9]
#        print(norm_points)
        
        counts_bkgd = counts-background
        norm_counts = counts_bkgd/norm_points
        
        popt,pcov = curve_fit(fit_eq,bin_centers[11:], norm_counts[11:],
                                      p0=fit_params)
#        print(popt)
        lin_centers = numpy.linspace(0,bin_centers[-1], 1000)
    
        ax.semilogy(bin_centers[11:], norm_counts[11:], data_fmt_list[i], label=label_list[i])
        ax.semilogy(lin_centers, fit_eq(lin_centers,*popt), fit_fmt_list[i])
        ax.set_xlabel('Time after illumination (us)')
        ax.set_ylabel('Counts')
        ax.set_title(title)
#        ax.set_ylim([0.9*10**2, 1.1*10**3])
        ax.legend()
        
        if fit_eq == double_decay:
            text = "\n".join((label_list[i],
                              r'$A_1 = $' + "%.1f"%(popt[0]) + ', '
                              r'$d_1 = $' + "%.1f"%(popt[1]) + " us",
                              r'$A_2 = $' + "%.1f"%(popt[2]) + ', '
                              r'$d_2 = $' + "%.1f"%(popt[3]) + " us"))
            ax.text(0.55, 0.95 - (1.1*i)/10, text, transform=ax.transAxes, fontsize=12,
                                    verticalalignment="top", bbox=props)
        
        elif fit_eq == decayExp:
            text = "\n".join((label_list[i],
                              r'$A_1 = $' + "%.2f"%(popt[0]) + ', '
                              r'$d_1 = $' + "%.1f"%(popt[1]) + " us"
                              ))
            
        
        
            ax.text(0.70, 0.85 - (1*i)/10, text, transform=ax.transAxes, fontsize=12,
                                    verticalalignment="top", bbox=props)
    
    if fit_eq == double_decay:
        text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2}$' 
    elif fit_eq == triple_decay:
        text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2} +  A_3 e^{-t / d_3}$'
    elif fit_eq == decayExp:
        text_eq = r'$A_1 e^{-t / d_1}$'
    ax.text(0.3, 0.9, text_eq, transform=ax.transAxes, fontsize=12,
                                verticalalignment="top", bbox=props)
    return
    
# %%
        
if __name__ == '__main__':
    
    num_files = 2
    
    #file_ind: 0 for long pass, 1 for short pass, 2 for no filter 

    fig_lifetime(num_files, files, 1, 'Lifetime, shortpass filter', decayExp, init_params_list_1) # shortpass
    fig_lifetime(num_files, files, 0, 'Lifetime, longpass filter', decayExp, init_params_list_1) # longpass


#    pol_counts(num_files, files, 1, 'Fluorescence while Er is polarized, Shortpass filter')
#    pol_counts(num_files, files, 0, 'Fluorescence while Er is polarized, Longpass filter')
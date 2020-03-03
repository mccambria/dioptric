# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:48:36 2020

analysis file for lifetime data taken on Er doped Y2O3 sample, 5 nm deep.

This data was taken the afternoon of 3/2/2020

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

file_p02_lp = '2020_03_02-17_42_24-Y2O3_graphene_Er_5nm'
file_p02_sp = '2020_03_02-17_52_33-Y2O3_graphene_Er_5nm'
file_p02_nf = '2020_03_02-18_01_45-Y2O3_graphene_Er_5nm'

file_m05_lp = '2020_03_02-18_12_48-Y2O3_graphene_Er_5nm'
file_m05_sp = '2020_03_02-18_22_45-Y2O3_graphene_Er_5nm'
file_m05_nf = '2020_03_02-18_32_34-Y2O3_graphene_Er_5nm'

file_m10_lp = '2020_03_02-18_44_47-Y2O3_graphene_Er_5nm'
file_m10_sp = '2020_03_02-18_54_00-Y2O3_graphene_Er_5nm'
file_m10_nf = '2020_03_02-19_03_04-Y2O3_graphene_Er_5nm'

file_m15_lp = '2020_03_02-19_13_46-Y2O3_graphene_Er_5nm'
file_m15_sp = '2020_03_02-19_22_37-Y2O3_graphene_Er_5nm'
file_m15_nf = '2020_03_02-19_32_03-Y2O3_graphene_Er_5nm'

file_m20_lp = '2020_03_02-19_42_39-Y2O3_graphene_Er_5nm'
file_m20_sp = '2020_03_02-19_51_42-Y2O3_graphene_Er_5nm'
file_m20_nf = '2020_03_02-20_00_41-Y2O3_graphene_Er_5nm'

file_m10_lp_2 = '2020_03_02-20_12_45-Y2O3_graphene_Er_5nm'
file_m10_sp_2 = '2020_03_02-20_21_48-Y2O3_graphene_Er_5nm'
file_m10_nf_2 = '2020_03_02-20_30_37-Y2O3_graphene_Er_5nm'

file_m25_lp = '2020_03_02-20_44_24-Y2O3_graphene_Er_5nm'
file_m25_sp = '2020_03_02-20_53_10-Y2O3_graphene_Er_5nm'



files = [file_p02_lp,file_p02_sp,file_p02_nf,
           file_m05_lp,file_m05_sp,file_m05_nf, 
         file_m10_lp,file_m10_sp,file_m10_nf,
         file_m15_lp, file_m15_sp, file_m15_nf,
         file_m20_lp, file_m20_sp, file_m20_nf,
         file_m10_lp_2, file_m10_sp_2, file_m10_nf_2,
         file_m25_lp, file_m25_sp]

count_list = []
bin_centers_list = []
text_list = []
data_fmt_list = ['b.','k.','r.','g.', 'y.', 'm.', 'c.']
fit_fmt_list = ['b-','k-','r-','g-', 'y-', 'm-', 'c-']
label_list = ['+0.3 V', '-0.5', '-1.0V', '-1.5 V', '-2.0V', '-1.0V', '-2.5V']
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

init_params_list_1 = [10,1]
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

def fig(num_data, file_list, file_ind, title, fit_eq, fit_params):

    fig, ax= plt.subplots(1, 1, figsize=(10, 8))
    for i in range(num_data):
        f = i*3 + file_ind
        file = file_list[f]
        data = tool_belt.get_raw_data(folder, file)
    
        counts = numpy.array(data["binned_samples"])
        bin_centers = numpy.array(data["bin_centers"])/10**3
        
        
        background = numpy.average(counts[75:101])
        first_two_points = numpy.average(counts[0:2])
        
        counts_bkgd = counts-background
        norm_counts = counts_bkgd/first_two_points
        
        popt,pcov = curve_fit(fit_eq,bin_centers[1:], norm_counts[1:],
                                      p0=fit_params)
        print(popt)
        lin_centers = numpy.linspace(bin_centers[0],bin_centers[-1], 1000)
    
        ax.semilogy(bin_centers, counts, data_fmt_list[i], label=label_list[i])
#        ax.semilogy(lin_centers, fit_eq(lin_centers,*popt), fit_fmt_list[i])
        ax.set_xlabel('Time after illumination (us)')
        ax.set_ylabel('Counts')
        ax.set_title(title)
        ax.legend()
        
        if fit_eq == double_decay:
            text = "\n".join((label_list[i],
                              r'$A_1 = $' + "%.1f"%(popt[0]) + ', '
                              r'$d_1 = $' + "%.1f"%(popt[1]) + " us",
                              r'$A_2 = $' + "%.1f"%(popt[2]) + ', '
                              r'$d_2 = $' + "%.1f"%(popt[3]) + " us"))
#            ax.text(0.55, 0.95 - (1.1*i)/10, text, transform=ax.transAxes, fontsize=12,
#                                    verticalalignment="top", bbox=props)
        
        elif fit_eq == triple_decay:
            text = "\n".join((label_list[i],
                              r'$A_1 = $' + "%.1f"%(popt[0]) + ', '
                              r'$d_1 = $' + "%.1f"%(popt[1]) + " us",
                              r'$A_2 = $' + "%.1f"%(popt[2]) + ', '
                              r'$d_2 = $' + "%.1f"%(popt[3]) + " us",
                              r'$A_3 = $' + "%.1f"%(popt[4]) + ', '
                              r'$d_3 = $' + "%.1f"%(popt[5]) + " us"
                              ))
            
        
        
            ax.text(0.55, 0.95 - (1.5*i)/10, text, transform=ax.transAxes, fontsize=12,
                                    verticalalignment="top", bbox=props)
    
    if fit_eq == double_decay:
        text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2}$' 
    elif fit_eq == triple_decay:
        text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2} +  A_3 e^{-t / d_3}$'
#    ax.text(0.2, 0.8, text_eq, transform=ax.transAxes, fontsize=12,
#                                verticalalignment="top", bbox=props)
    return
    
# %%
        
if __name__ == '__main__':
    
    num_files = 7
    
    #file_ind: 0 for long pass, 1 for short pass, 2 for no filter 

    fig(num_files, files, 1, 'Lifetime, shortpass filter', double_decay, init_params_list_2) # shortpass
    fig(num_files, files, 0, 'Lifetime, longpass filter', double_decay, init_params_list_2) # longpass


#file_path = str(data_path + '/' + folder + '/plotted_data/' + file + '-loglog')
#tool_belt.save_figure(fig, file_path)
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:48:36 2020

@author: kolkowitz
"""
# %%
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import numpy
from scipy.optimize import curve_fit

# %%
data_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata'

folder = 'lifetime_v2/2020_02'

file_p1_nf = '2020_02_19-13_58_06-graphene_Y2O3'
file_p1_sp = '2020_02_19-13_47_08-graphene_Y2O3'
file_p1_lp = '2020_02_19-13_53_39-graphene_Y2O3'

file_0_nf = '2020_02_19-14_47_34-graphene_Y2O3'
file_0_sp = '2020_02_19-14_36_09-graphene_Y2O3'
file_0_lp = '2020_02_19-14_43_08-graphene_Y2O3'

file_m1_nf = '2020_02_19-14_16_57-graphene_Y2O3'
file_m1_sp = '2020_02_19-14_21_05-graphene_Y2O3'
file_m1_lp = '2020_02_19-14_12_15-graphene_Y2O3'

file_m15_nf = '2020_02_19-15_03_07-graphene_Y2O3'
file_m15_sp = '2020_02_19-15_10_08-graphene_Y2O3'
file_m15_lp = '2020_02_19-20_18_22-graphene_Y2O3'

file_m2_nf = '2020_02_19-19_12_14-graphene_Y2O3'
file_m2_sp = '2020_02_19-19_07_02-graphene_Y2O3'
file_m2_lp = '2020_02_19-19_00_16-graphene_Y2O3'

file_m26_nf = '2020_02_19-15_47_52-graphene_Y2O3'
file_m26_sp = '2020_02_19-15_15_55-graphene_Y2O3'
file_m26_lp = '2020_02_19-15_20_20-graphene_Y2O3'

file_m3_nf = '2020_02_19-20_04_26-graphene_Y2O3'
file_m3_sp = '2020_02_19-20_09_33-graphene_Y2O3'
file_m3_lp = '2020_02_19-20_13_37-graphene_Y2O3'

files = [file_p1_nf,file_p1_sp,file_p1_lp, 
         file_0_nf,file_0_sp,file_0_lp,
         file_m1_nf,file_m1_sp,file_m1_lp,
         file_m15_nf, file_m15_sp, file_m15_lp,
         file_m2_nf, file_m2_sp, file_m2_lp,
         file_m26_nf,file_m26_sp,file_m26_lp,
         file_m3_nf, file_m3_sp, file_m3_lp]

count_list = []
bin_centers_list = []
text_list = []
data_fmt_list = ['bo','ko','ro','go', 'yo', 'mo', 'co']
fit_fmt_list = ['b-','k-','r-','g-', 'y-', 'm-', 'c-']
label_list = ['+1 V (CNP)', '0 V', '-1 V', '-1.6V', '-2.0V','-2.6 V', '-3.0V']
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

init_params_list_2 = [10**5, 10, 10**5, 100]
init_params_list_3 = [10**5, 10, 10**5,70,  10**5,300]

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

#fig1, ax= plt.subplots(1, 1, figsize=(10, 8))
#for i in range(7):
#    f = i*3
#    file = files[f]
#    data = tool_belt.get_raw_data(folder, file)
#
#    counts = data["binned_samples"]
#    bin_centers = numpy.array(data["bin_centers"])/10**3
#    
#    
#    background = numpy.average(counts[75:150])
#    
#    short_norm_counts = numpy.array(counts[0:30])-background
#    short_centers = bin_centers[0:30]
#    
#    popt,pcov = curve_fit(double_decay,short_centers, short_norm_counts,
#                                  p0=init_params_list_2)
#    lin_centers = numpy.linspace(0,short_centers[-1], 1000)
#
#    ax.semilogy(short_centers, short_norm_counts, data_fmt_list[i], label=label_list[i])
#    ax.semilogy(lin_centers, double_decay(lin_centers,*popt), fit_fmt_list[i])
#    ax.set_xlabel('Time after illumination (us)')
#    ax.set_ylabel('Counts')
#    ax.set_title('Lifetime, no filters')
#    ax.legend()
    
    
#    text = "\n".join((label_list[i],
#                      r'$A_1 = $' + "%.1f"%(popt[0]) + ', '
#                      r'$d_1 = $' + "%.1f"%(popt[1]) + " us",
#                      r'$A_2 = $' + "%.1f"%(popt[2]) + ', '
#                      r'$d_2 = $' + "%.1f"%(popt[3]) + " us"))
#    
#
#
#    ax.text(0.05, 0.8 - (1.1*i)/10, text, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
    
#text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2}$'    
#ax.text(0.5, 0.8, text_eq, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)

fig2, ax= plt.subplots(1, 1, figsize=(10, 8))
for i in range(7):
    f = i*3+1
    file = files[f]
    data = tool_belt.get_raw_data(folder, file)

    counts = numpy.array(data["binned_samples"])
    bin_centers = numpy.array(data["bin_centers"])/10**3
    
    
    background = numpy.average(counts[75:150])
    
    short_norm_counts = counts[0:30]-background
    short_centers = bin_centers[0:30]
    
    popt,pcov = curve_fit(double_decay,short_centers, short_norm_counts,
                                  p0=init_params_list_2)
    lin_centers = numpy.linspace(0,short_centers[-1], 1000)

    ax.plot(bin_centers, counts-background, data_fmt_list[i], label=label_list[i])
    ax.plot(lin_centers, double_decay(lin_centers,*popt), fit_fmt_list[i])
    ax.set_xlabel('Time after illumination (us)')
    ax.set_ylabel('Counts')
    ax.set_title('Lifetime, shortpass filter')
    ax.legend()
    
    
#    text = "\n".join((label_list[i],
#                      r'$A_1 = $' + "%.1f"%(popt[0]) + ', '
#                      r'$d_1 = $' + "%.1f"%(popt[1]) + " us",
#                      r'$A_2 = $' + "%.1f"%(popt[2]) + ', '
#                      r'$d_2 = $' + "%.1f"%(popt[3]) + " us"))
#    
#
#
#    ax.text(0.05, 0.8 - (1.1*i)/10, text, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
    
text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2}$'    
ax.text(0.5, 0.8, text_eq, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)
    
fig3, ax= plt.subplots(1, 1, figsize=(10, 8))
for i in range(7):
    f = i*3+2
    file = files[f]
    data = tool_belt.get_raw_data(folder, file)

    counts = numpy.array(data["binned_samples"])
    bin_centers = numpy.array(data["bin_centers"])/10**3
    
    
    background = numpy.average(counts[75:150])
    
    short_norm_counts = counts[0:30]-background
    short_centers = bin_centers[0:30]
    
    popt,pcov = curve_fit(double_decay,short_centers, short_norm_counts,
                                  p0=init_params_list_2)
    lin_centers = numpy.linspace(0,short_centers[-1], 1000)

    ax.plot(bin_centers, counts - background, data_fmt_list[i], label=label_list[i])
    ax.plot(lin_centers, double_decay(lin_centers,*popt), fit_fmt_list[i])
    ax.set_xlabel('Time after illumination (us)')
    ax.set_ylabel('Counts')
    ax.set_title('Lifetime, longpass filter')
    ax.legend()
    
    
#    text = "\n".join((label_list[i],
#                      r'$A_1 = $' + "%.1f"%(popt[0]) + ', '
#                      r'$d_1 = $' + "%.1f"%(popt[1]) + " us",
#                      r'$A_2 = $' + "%.1f"%(popt[2]) + ', '
#                      r'$d_2 = $' + "%.1f"%(popt[3]) + " us"))
#    
#
#
#    ax.text(0.05, 0.8 - (1.1*i)/10, text, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    
#text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2}$'    
#ax.text(0.5, 0.8, text_eq, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
    
        
    


#file_path = str(data_path + '/' + folder + '/plotted_data/' + file + '-loglog')
#tool_belt.save_figure(fig, file_path)
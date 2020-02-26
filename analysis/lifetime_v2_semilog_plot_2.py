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

file_0_nf_s = '2020_02_20-16_47_27-graphene_Y2O3'
file_0_sp_s = '2020_02_20-16_41_29-graphene_Y2O3'
file_0_lp_s = '2020_02_20-16_31_51-graphene_Y2O3'

file_0_nf_l = '2020_02_20-16_50_41-graphene_Y2O3'
file_0_sp_l = '2020_02_20-16_44_48-graphene_Y2O3'
file_0_lp_l = '2020_02_20-16_38_08-graphene_Y2O3'

file_m1_nf_s = '2020_02_20-16_58_19-graphene_Y2O3'
file_m1_sp_s = '2020_02_20-17_03_53-graphene_Y2O3'
file_m1_lp_s = '2020_02_20-17_13_34-graphene_Y2O3'

file_m1_nf_l = '2020_02_20-17_01_23-graphene_Y2O3'
file_m1_sp_l = '2020_02_20-17_08_51-graphene_Y2O3'
file_m1_lp_l = '2020_02_20-17_11_02-graphene_Y2O3'

file_m26_nf_s = '2020_02_20-17_27_10-graphene_Y2O3'
file_m26_sp_s = '2020_02_20-17_35_29-graphene_Y2O3'
file_m26_lp_s = '2020_02_20-17_21_24-graphene_Y2O3'

file_m26_nf_l = '2020_02_20-17_30_39-graphene_Y2O3'
file_m26_sp_l = '2020_02_20-17_38_52-graphene_Y2O3'
file_m26_lp_l = '2020_02_20-17_24_36-graphene_Y2O3'

files_s = [file_0_nf_s,file_0_sp_s,file_0_lp_s, 
         file_m1_nf_s,file_m1_sp_s,file_m1_lp_s,
         file_m26_nf_s,file_m26_sp_s,file_m26_lp_s]

files_l = [file_0_nf_l,file_0_sp_l,file_0_lp_l, 
         file_m1_nf_l,file_m1_sp_l,file_m1_lp_l,
         file_m26_nf_l,file_m26_sp_l,file_m26_lp_l]

count_list = []
bin_centers_list = []
text_list = []
data_fmt_list = ['bo','ko','ro','go', 'yo', 'mo', 'co']
fit_fmt_list = ['b-','k-','r-','g-', 'y-', 'm-', 'c-']
label_list = ['0 V', '-1 V', '-2.6 V']
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

init_params_list_2 = [10**3, 1, 10**3, 100]
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

#fig2, ax= plt.subplots(1, 1, figsize=(10, 8))
#for i in range(3):
#    f = i*3+1
#    file = files_l[f]
#    data = tool_belt.get_raw_data(folder, file)
#
#    counts = numpy.array(data["binned_samples"])
#    bin_centers = numpy.array(data["bin_centers"])/10**3
    
    
#    background = numpy.average(counts[75:150])
#    
#    short_norm_counts = counts[0:30]-background
#    short_centers = bin_centers[0:30]
    
#    popt,pcov = curve_fit(double_decay,bin_centers, counts,
#                                  p0=init_params_list_2)
#    lin_centers = numpy.linspace(0,50, 1000)

#    ax.plot(bin_centers, counts, data_fmt_list[i], label=label_list[i])
#    ax.semilogy(lin_centers, double_decay(lin_centers,*popt), fit_fmt_list[i])
#    ax.set_xlabel('Time after illumination (us)')
#    ax.set_ylabel('Counts')
#    ax.set_title('Lifetime, shortpass filter')
#    ax.legend()
#    
#    
#    text = "\n".join((label_list[i],
#                      r'$A_1 = $' + "%.1f"%(popt[0]) + ', '
#                      r'$d_1 = $' + "%.1f"%(popt[1]) + " us",
#                      r'$A_2 = $' + "%.1f"%(popt[2]) + ', '
#                      r'$d_2 = $' + "%.1f"%(popt[3]) + " us"))
    


#    ax.text(0.05, 0.8 - (1.1*i)/10, text, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
    
text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2}$'    
#ax.text(0.5, 0.8, text_eq, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
    
fig3, ax= plt.subplots(1, 1, figsize=(10, 8))
for i in range(3):
    f = i*3+2
    file = files_l[f]
    data = tool_belt.get_raw_data(folder, file)

    counts = numpy.array(data["binned_samples"])
    bin_centers = numpy.array(data["bin_centers"])/10**3
    
    
    background = numpy.average(counts[50:100])
    
    print(background)
    
#    short_norm_counts = counts[0:30]-background
#    short_centers = bin_centers[0:30]
    
    popt,pcov = curve_fit(double_decay,counts, bin_centers,
                                  p0=init_params_list_2)
    lin_centers = numpy.linspace(0,50, 1000)

    ax.plot(bin_centers, counts, data_fmt_list[i], label=label_list[i])
#    ax.semilogy(lin_centers, double_decay(lin_centers,*popt), fit_fmt_list[i])
    ax.set_xlabel('Time after illumination (us)')
    ax.set_ylabel('Counts')
    ax.set_title('Lifetime, longpass filter')
    ax.legend()
    
    
    text = "\n".join((label_list[i],
                      r'$A_1 = $' + "%.1f"%(popt[0]) + ', '
                      r'$d_1 = $' + "%.1f"%(popt[1]) + " us",
                      r'$A_2 = $' + "%.1f"%(popt[2]) + ', '
                      r'$d_2 = $' + "%.1f"%(popt[3]) + " us"))
    


#    ax.text(0.05, 0.8 - (1.1*i)/10, text, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    
#text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2}$'    
#ax.text(0.5, 0.8, text_eq, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
    
        
    


#file_path = str(data_path + '/' + folder + '/plotted_data/' + file + '-loglog')
#tool_belt.save_figure(fig, file_path)
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:13:01 2019

subtraction of no filter and 630 long pass

only fit and show first 500 us 

@author: Aedan
"""
import numpy
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#init_params_list_2 = [[10**5, 10, 10**5, 100],
#                        [10**5, 10, 10**5,  100],
#                        [10**5, 10, 10**5,  100],
#                        [10**5, 10, 10**5,  100]
#                        ]


init_params_list_3 = [[10**5, 10, 10**5,70,  10**5,300],
                    [10**5, 10, 10**5,70,  10**5,300],
                    [10**5, 10, 10**5,70,  10**5,300],
                    [10**5, 10, 10**5,70,  10**5,300],
                    ]

#init_params_list_4 = [[10**4, 10, 30,  100,  1000],
#                    [10**4, 10, 30,  100,  1000],
#                    [10**4, 10, 30,  100,  1000],
#                    [10**4, 10, 30,  100,  1000],
#                    ]
    
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

def main():
    directory = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2019_11/'
    
    # No filter
#    file_HIGH = '2019_11_27-09_23_06-graphene_Y2O3-no_filter'
#    file_ZERO = '2019_11_27-09_31_48-graphene_Y2O3-no_filter'
#    file_LOW = '2019_11_27-09_52_23-graphene_Y2O3-no_filter'
#    file_NG = '2019_11_27-10_31_03-Y2O3-no_filter'
    
    # 550 Bandpass
    file_HIGH = '2019_11_27-09_10_09-graphene_Y2O3-550_bandpass'
    file_ZERO = '2019_11_27-09_37_59-graphene_Y2O3-550_bandpass'
    file_LOW = '2019_11_27-10_06_12-graphene_Y2O3-550_bandpass'
    file_NG = '2019_11_27-10_44_01-Y2O3-550_bandpass'
    #
    # 630 longpass
#    file_HIGH = '2019_11_27-09_16_22-graphene_Y2O3-630_longpass'
#    file_ZERO = '2019_11_27-09_44_07-graphene_Y2O3-630_longpass'
#    file_LOW = '2019_11_27-09_59_07-graphene_Y2O3-630_longpass'
#    file_NG = '2019_11_27-10_36_39-Y2O3-630_longpass'
    
    file_list = [file_HIGH, file_ZERO, file_LOW, file_NG]
    # Make list for the data
    
    counts_list = []
    bin_center_list =[]
    popt_list = []
    data_fmt_list = ['bo','ko','ro','go']
    fit_fmt_list=['b--','k--','r--','g--']
    label_list = ['w/ graphene, 4.2 V', 'w/ graphene, 0 V', 'w/ graphene, -3 V', 
                  'w/out graphene']
    
    fig_fit, ax= plt.subplots(1, 1, figsize=(10, 8))
    
    # Open the specified file
    for file in file_list:
        with open(directory + file + '.txt') as json_file:
        
            # Load the data from the file
            data = json.load(json_file)
            counts = numpy.array(data["binned_samples"])
            readout_time = data["readout_time"]/10**3
            bin_centers = numpy.array(data["bin_centers"])/10**3
            
        counts_list.append(counts)
        bin_center_list.append(bin_centers)
        
    linspaceTime = numpy.linspace(0, 500, num=1000)

    
    for i in range(len(file_list)):
    
        popt,pcov = curve_fit(triple_decay, bin_center_list[i][0:34], counts_list[i][0:34],
                                  p0=init_params_list_3[i])

        popt_list.append(popt)
        
        ax.plot(bin_center_list[i], counts_list[i], data_fmt_list[i],label=label_list[i])
#        ax.plot(linspaceTime, triple_decay(linspaceTime,*popt),fit_fmt_list[i])
        
    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Counts (arb.)')
    ax.set_title('Er implanted Y2O3 lifetime, 550 bandpass filter')
    ax.legend()
#    ax.set_xlim([0,500])
    ax.set_yscale("log", nonposy='clip')
    print(popt_list)
    
    text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2} +  A_3 e^{-t / d_3} $'
    text_4V = "\n".join((
                      'w/ graphene 4.2 V',
                      r'$A_1 = $' + "%.1f"%(popt_list[0][0]) + ', '
                      r'$d_1 = $' + "%.1f"%(popt_list[0][1]) + " us",
                      r'$A_2 = $' + "%.1f"%(popt_list[0][2]) + ', '
                      r'$d_2 = $' + "%.1f"%(popt_list[0][3]) + " us",
                      r'$A_3 = $' + "%.1f"%(popt_list[0][3]) + ', '
                      r'$d_3 = $' + "%.1f"%(popt_list[0][5]) + " us"))
    text_0V = "\n".join((
                      'w/ graphene 0 V',
                      r'$A_1 = $' + "%.1f"%(popt_list[1][0]) + ', '
                      r'$d_1 = $' + "%.1f"%(popt_list[1][1]) + " us",
                      r'$A_2 = $' + "%.1f"%(popt_list[1][2]) + ', '
                      r'$d_2 = $' + "%.1f"%(popt_list[1][3]) + " us",
                      r'$A_3 = $' + "%.1f"%(popt_list[1][3]) + ', '
                      r'$d_3 = $' + "%.1f"%(popt_list[1][5]) + " us"))
    text_3V = "\n".join((
                      'w/ graphene -3 V',
                      r'$A_1 = $' + "%.1f"%(popt_list[2][0]) + ', '
                      r'$d_1 = $' + "%.1f"%(popt_list[2][1]) + " us",
                      r'$A_2 = $' + "%.1f"%(popt_list[2][2]) + ', '
                      r'$d_2 = $' + "%.1f"%(popt_list[2][3]) + " us",
                      r'$A_3 = $' + "%.1f"%(popt_list[2][3]) + ', '
                      r'$d_3 = $' + "%.1f"%(popt_list[2][5]) + " us"))
    text_NOF = "\n".join((
                      'w/out graphene',
                      r'$A_1 = $' + "%.1f"%(popt_list[3][0]) + ', '
                      r'$d_1 = $' + "%.1f"%(popt_list[3][1]) + " us",
                      r'$A_2 = $' + "%.1f"%(popt_list[3][2]) + ', '
                      r'$d_2 = $' + "%.1f"%(popt_list[3][3]) + " us",
                      r'$A_3 = $' + "%.1f"%(popt_list[3][4]) + ', '
                      r'$d_3 = $' + "%.1f"%(popt_list[3][5]) + " us",
                      ))
    
    
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
#    ax.text(0.02, 0.48, text_eq, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    ax.text(0.02, 0.38, text_4V, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    ax.text(0.34, 0.38, text_0V, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    ax.text(0.02, 0.18, text_3V, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    ax.text(0.34, 0.18, text_NOF, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
    
#    ax.text(0.35, 0.95, text_eq, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    ax.text(0.35, 0.85, text_4V, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    ax.text(0.67, 0.8, text_0V, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    ax.text(0.35, 0.65, text_3V, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    ax.text(0.67, 0.6, text_NOF, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#    
#    ax.set_yscale("log", nonposy='clip')
    
    fig_fit.canvas.draw()
    fig_fit.canvas.flush_events()
    
    #file_path = directory + open_file_name
    #tool_belt.save_figure(fig_fit, file_path+'-triple_fit_semilog')
    #    fig_fit.savefig(open_file_name + '-replot.svg')
  
# %%
def subtract():
    
    directory = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime_v2/2019_11/'
    
    # No filter
    file_HIGH = '2019_11_27-09_23_06-graphene_Y2O3-no_filter'
    file_ZERO = '2019_11_27-09_31_48-graphene_Y2O3-no_filter'
    file_LOW = '2019_11_27-09_52_23-graphene_Y2O3-no_filter'
    file_NG = '2019_11_27-10_31_03-Y2O3-no_filter'
    
    file_list_NOF = [file_HIGH, file_ZERO, file_LOW, file_NG]
    # 550 Bandpass
    file_HIGH = '2019_11_27-09_10_09-graphene_Y2O3-550_bandpass'
    file_ZERO = '2019_11_27-09_37_59-graphene_Y2O3-550_bandpass'
    file_LOW = '2019_11_27-10_06_12-graphene_Y2O3-550_bandpass'
    file_NG = '2019_11_27-10_44_01-Y2O3-550_bandpass'
    
    file_list_550 = [file_HIGH, file_ZERO, file_LOW, file_NG]
    
    # 630 longpass
    file_HIGH = '2019_11_27-09_16_22-graphene_Y2O3-630_longpass'
    file_ZERO = '2019_11_27-09_44_07-graphene_Y2O3-630_longpass'
    file_LOW = '2019_11_27-09_59_07-graphene_Y2O3-630_longpass'
    file_NG = '2019_11_27-10_36_39-Y2O3-630_longpass'
    
    file_list_670 = [file_HIGH, file_ZERO, file_LOW, file_NG]
    
    # Make list for the data
    
    counts_subt_list = []
    counts_list = []
    bin_center_list =[]
    popt_list = []
    data_fmt_list = ['b.','k.','r.','g.']
    subt_fmt_list=['bo','ko','ro','go']
    fit_fmt_list=['b--','k--','r--','g--']
    label_list = ['w/ graphene, 4.2 V', 'w/ graphene, 0 V', 'w/ graphene, -3 V', 
                  'w/out graphene']
    label_subt_list = ['w/ graphene, 4.2 V (subtracted)', 
                       'w/ graphene, 0 V (subtracted)', 
                       'w/ graphene, -3 V (subtracted)', 
                  'w/out graphene (subtracted)']
    
    fig, ax= plt.subplots(1, 1, figsize=(10, 8))
    
    linspaceTime = numpy.linspace(0, 500, num=1000)
    
    # Open the specified file
    for i in range(len(file_list_NOF)):
        with open(directory + file_list_NOF[i] + '.txt') as json_file:
        
            # Load the data from the file
            data = json.load(json_file)
            counts_NOF = numpy.array(data["binned_samples"])
            readout_time = data["readout_time"]/10**3
            bin_centers = numpy.array(data["bin_centers"])/10**3
            
        with open(directory + file_list_670[i] + '.txt') as json_file:
        
            # Load the data from the file
            data = json.load(json_file)
            counts_670 = numpy.array(data["binned_samples"])
            readout_time = data["readout_time"]/10**3
            bin_centers = numpy.array(data["bin_centers"])/10**3
            
        with open(directory + file_list_550[i] + '.txt') as json_file:
        
            # Load the data from the file
            data = json.load(json_file)
            counts = numpy.array(data["binned_samples"])
            readout_time = data["readout_time"]/10**3
            bin_centers = numpy.array(data["bin_centers"])/10**3
            
        counts_subt_list.append(counts_NOF - counts_670)
        bin_center_list.append(bin_centers)
        counts_list.append(counts)
        
            
        popt,pcov = curve_fit(triple_decay, bin_center_list[i][0:34], counts_subt_list[i][0:34],
                                  p0=init_params_list_3[i])

        popt_list.append(popt)
        print(popt)
        
#        ax.plot(bin_center_list[i], counts_list[i], data_fmt_list[i],
#                label=label_list[i])
        ax.plot(bin_center_list[i], counts_subt_list[i],subt_fmt_list[i],
                label = label_subt_list[i])
        
        ax.plot(linspaceTime, triple_decay(linspaceTime,*popt),fit_fmt_list[i])
    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Counts (arb.)')
    ax.set_title('Er implanted Y2O3 lifetime, 550 signal (obtained via subtraction)')
    ax.legend()
    ax.set_xlim([0,500])
    
    text_eq = r'$A_1 e^{-t / d_1} +  A_2 e^{-t / d_2} +  A_3 e^{-t / d_3} $'
    text_4V = "\n".join((
                      'w/ graphene 4.2 V',
                      r'$A_1 = $' + "%.1f"%(popt_list[0][0]) + ', '
                      r'$d_1 = $' + "%.1f"%(popt_list[0][1]) + " us",
                      r'$A_2 = $' + "%.1f"%(popt_list[0][2]) + ', '
                      r'$d_2 = $' + "%.1f"%(popt_list[0][3]) + " us",
                      r'$A_3 = $' + "%.1f"%(popt_list[0][3]) + ', '
                      r'$d_3 = $' + "%.1f"%(popt_list[0][5]) + " us"))
    text_0V = "\n".join((
                      'w/ graphene 0 V',
                      r'$A_1 = $' + "%.1f"%(popt_list[1][0]) + ', '
                      r'$d_1 = $' + "%.1f"%(popt_list[1][1]) + " us",
                      r'$A_2 = $' + "%.1f"%(popt_list[1][2]) + ', '
                      r'$d_2 = $' + "%.1f"%(popt_list[1][3]) + " us",
                      r'$A_3 = $' + "%.1f"%(popt_list[1][3]) + ', '
                      r'$d_3 = $' + "%.1f"%(popt_list[1][5]) + " us"))
    text_3V = "\n".join((
                      'w/ graphene -3 V',
                      r'$A_1 = $' + "%.1f"%(popt_list[2][0]) + ', '
                      r'$d_1 = $' + "%.1f"%(popt_list[2][1]) + " us",
                      r'$A_2 = $' + "%.1f"%(popt_list[2][2]) + ', '
                      r'$d_2 = $' + "%.1f"%(popt_list[2][3]) + " us",
                      r'$A_3 = $' + "%.1f"%(popt_list[2][3]) + ', '
                      r'$d_3 = $' + "%.1f"%(popt_list[2][5]) + " us"))
    text_NOF = "\n".join((
                      'w/out graphene',
                      r'$A_1 = $' + "%.1f"%(popt_list[3][0]) + ', '
                      r'$d_1 = $' + "%.1f"%(popt_list[3][1]) + " us",
                      r'$A_2 = $' + "%.1f"%(popt_list[3][2]) + ', '
                      r'$d_2 = $' + "%.1f"%(popt_list[3][3]) + " us",
                      r'$A_3 = $' + "%.1f"%(popt_list[3][4]) + ', '
                      r'$d_3 = $' + "%.1f"%(popt_list[3][5]) + " us",
                      ))
    
    
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.05, 0.6, text_eq, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)
    ax.text(0.05, 0.5, text_4V, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)
    ax.text(0.4, 0.5, text_0V, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)
    ax.text(0.05, 0.3, text_3V, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)
    ax.text(0.4, 0.3, text_NOF, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)
    
    
    ax.set_yscale("log", nonposy='clip')
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    #file_path = directory + open_file_name
    #tool_belt.save_figure(fig_fit, file_path+'-triple_fit_semilog')
    #    fig_fit.savefig(open_file_name + '-replot.svg')
#%%
    
if __name__ == '__main__':
#    subtract()
    main()
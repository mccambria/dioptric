# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:09:33 2021

@author: agard
"""



# %% Imports


import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

two_pi = 2*numpy.pi
mu = u"\u03BC" 
blue = '#00aeef'
light_blue = '#abd0ee'
red = '#ee3542'
light_red = '#f69481'
# %%
def tC_A(tau, d, Offset, A):
    exp_part = numpy.exp(-((tau / d) ** 4))
    # sin_part = A - C * numpy.sin(two_pi *a1 * tau/2)**2 * numpy.sin(two_pi*a2 * tau/2)**2
    return  Offset - A*exp_part

def tC_B(tau,  a1, a2,  d, Offset, C, A):
    exp_part = numpy.exp(-((tau / d) ** 4))
    sin_part = A - C * numpy.sin(two_pi *a1 * tau/2)**2 * numpy.sin(two_pi*a2 * tau/2)**2
    return  Offset - exp_part *sin_part 


def t2(tau,  t2, Offset, A,  B, C):
    t2_part =numpy.exp(-((tau / t2) ** 3))
    sin_part = numpy.sin(two_pi *C * tau/2)**4 
    return  (Offset - A* numpy.exp(-B*sin_part) *t2_part)

def t2_decay(tau,  t2, Offset, A,  B, C,overal_decay ):
    t2_part =numpy.exp(-((tau / t2) ** 3))
    sin_part = numpy.sin(two_pi *C * tau/2)**4 
    return  (Offset - A* numpy.exp(-B*sin_part) *t2_part)*numpy.exp(-tau/overal_decay)


def exp_decay(tau,  D,   B):
    return  B*numpy.exp(-tau/D)

# %%
def do_fit_plot_C(file, folder):
    max_ind = 351
    fig_w = 2.7
    fig_l = 1.5
    
    marker_size = 2#3
    data_linewidth = 0.5
    fit_linewidth = 0.5
    
    f_size = 7
    
    fig_tick_l = 3
    fig_tick_w = 0.75
    
    
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
        
    data = tool_belt.get_raw_data(file, folder)
    norm_avg_sig = numpy.array(data['norm_avg_sig'])
    uwave_pi_pulse = data['uwave_pi_pulse']
    
    taus = data['taus']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
        
    # Fit data
    lin_taus = numpy.linspace(plot_taus[0], plot_taus[max_ind], 1000)
    
    fit_func = tC_A
    #       w0, w1, Tc, Offset, oscillation amplitude, decay amplitude     
    # init_params = [20, 0.08,  4, 1.08, 0.02, 0.06] # A   
    init_params = [5, 1.08, 0.05] # B
    
    popt, pcov = curve_fit(
        fit_func,
        plot_taus,
        norm_avg_sig ,
        p0=init_params,
    )
    
    print(popt)
    ax.plot(plot_taus[0:max_ind],norm_avg_sig[0:max_ind] , 'b^-', color=blue, 
            linewidth = data_linewidth, markersize = marker_size, mfc = light_blue)
    ax.plot(lin_taus, fit_func(lin_taus, *popt), 'k-', linewidth = fit_linewidth)
    
    # text_eq = r"A + C * e$^{-D sin^4(w t)}$ * e$^{-(\tau / T_2)^3}$"
    
    # text_popt = "\n".join(
    #     ( #A, B, C, D, T2
    #         r"A=%.3f" % (popt[0]),
    #         r"C=%.3f" % (popt[2]),
    #         r"D=%.3f" % (popt[3]),
    #         r"w=%.5f MHz" % (popt[1]),
    #         r"T$_2$=%.3f (us)" % (popt[4]),
    #     )
    # )
    
    # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.05,
    #     0.15,
    #     text_popt,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment="top",
    #     bbox=props,
    # )
    # ax.text(
    #     0.6,
    #     0.95,
    #     text_eq,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment="top",
    #     bbox=props,
    # )
    # ax.set_xlim([-0.181, 3.819])
    ax.set_ylim([0.9566, 1.1249])
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax.set_xlabel(r'Free precession time, $t$ (' + mu + 's)',  fontsize = f_size)
    ax.set_ylabel(r'Normalized contrast, $\mathcal{C}$',  fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    # ax.legend(loc='lower right',  fontsize = f_size)
    
# %%
def do_fit_plot_D(file, folder):
    max_ind = 351
    fig_w = 2.7
    fig_l = 1.5
    
    marker_size = 2#2.5
    data_linewidth = 0.5
    fit_linewidth = 0.5
    
    f_size = 7
    
    fig_tick_l = 3
    fig_tick_w = 0.75
    
    
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
        
        
    data = tool_belt.get_raw_data(file, folder)
    norm_avg_sig = numpy.array(data['norm_avg_sig'])
    uwave_pi_pulse = data['uwave_pi_pulse']
    
    taus = data['taus']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
        
    # Fit data
    lin_taus = numpy.linspace(plot_taus[0], plot_taus[max_ind], 1000)
    
    # using omega found from fft
    # fit_func = lambda tau, a1,  d, Offset, C, A: tC_B(tau,  a1, 0.1,  d, Offset, C, A)
    #       w0, w1, Tc, Offset, oscillation amplitude, decay amplitude   
    # init_params = [8,  3, 1.08, 0.1, 0.05] # B
    
    fit_func =tC_B
          # w0, w1, Tc, Offset, oscillation amplitude, decay amplitude   
    init_params = [9, 0.167,  3, 1.08, 0.1, 0.05] # B
    
    popt, pcov = curve_fit(
        fit_func,
        plot_taus,
        norm_avg_sig ,
        p0=init_params,
    )
    
    print(popt[0])
    print(numpy.sqrt(pcov[0][0]))
    print(popt[1])
    print(numpy.sqrt(pcov[1][1]))
    ax.plot(plot_taus[0:max_ind],norm_avg_sig[0:max_ind]  , 'bs-', color=red, 
            linewidth = data_linewidth, markersize = marker_size, mfc = light_red)
    ax.plot(lin_taus, fit_func(lin_taus, *popt), 'k-', linewidth = fit_linewidth)
    
    # text_eq = r"A + C * e$^{-D sin^4(w t)}$ * e$^{-(\tau / T_2)^3}$"
    
    # text_popt = "\n".join(
    #     ( #A, B, C, D, T2
    #         r"A=%.3f" % (popt[0]),
    #         r"C=%.3f" % (popt[2]),
    #         r"D=%.3f" % (popt[3]),
    #         r"w=%.5f MHz" % (popt[1]),
    #         r"T$_2$=%.3f (us)" % (popt[4]),
    #     )
    # )
    
    # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.05,
    #     0.15,
    #     text_popt,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment="top",
    #     bbox=props,
    # )
    # ax.text(
    #     0.6,
    #     0.95,
    #     text_eq,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment="top",
    #     bbox=props,
    # )
    
    # ax.set_xlim([-0.181, 3.819])
    # ax.set_ylim([0.949, 1.1584])
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax.set_xlabel(r'Free precession time, $t$ (' + mu + 's)',  fontsize = f_size)
    ax.set_ylabel(r'Normalized contrast, $\mathcal{C}$',  fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    
# %%
def do_fit_plot_C_zoom(file, folder):
    fig_w = 0.7
    fig_l = 0.5
    
    marker_size = 2
    data_linewidth = 0.5
    fit_linewidth = 0.5
    
    f_size = 8
    
    fig_tick_l = 3
    fig_tick_w = 0.75
    
    
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
        
    data = tool_belt.get_raw_data(file, folder)
    norm_avg_sig = numpy.array(data['norm_avg_sig'])
    uwave_pi_pulse = data['uwave_pi_pulse']
    
    taus = data['taus']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
        
    # Fit data
    lin_taus = numpy.linspace(plot_taus[0], plot_taus[-1], 1000)
    
    fit_func = tC_A
    #       w0, w1, Tc, Offset, oscillation amplitude, decay amplitude     
    # init_params = [20, 0.08,  4, 1.08, 0.02, 0.06] # A   
    init_params = [5, 1.08, 0.05] # B
    
    popt, pcov = curve_fit(
        fit_func,
        plot_taus,
        norm_avg_sig ,
        p0=init_params,
    )
    
    print(popt)
    ax.plot(plot_taus,norm_avg_sig , 'b^-', color=blue, 
            linewidth = data_linewidth, markersize = marker_size, mfc = light_blue)
    ax.plot(lin_taus, fit_func(lin_taus, *popt), 'k-', linewidth = fit_linewidth)
    
    ax.set_xlim([1.95, 3.05])
    ax.set_xticks([2,2.5,3])
    # ax.set_xlabel(r'$\tau$ ($\mu$s)',  fontsize = f_size)
    # ax.set_ylabel(r'$\mathcal{C}$',  fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    # ax.legend(loc='lower right',  fontsize = f_size)
    
# %%
def do_fit_plot_D_zoom(file, folder):
    fig_w = 0.7
    fig_l = 0.5
    
    marker_size = 2
    data_linewidth = 0.5
    fit_linewidth = 0.5
    
    f_size = 8
    
    fig_tick_l = 3
    fig_tick_w = 0.75
    
    
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
        
        
    data = tool_belt.get_raw_data(file, folder)
    norm_avg_sig = numpy.array(data['norm_avg_sig'])
    uwave_pi_pulse = data['uwave_pi_pulse']
    
    taus = data['taus']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
        
    # Fit data
    lin_taus = numpy.linspace(plot_taus[0], plot_taus[-1], 1000)
    
    fit_func = tC_B
    #       w0, w1, Tc, Offset, oscillation amplitude, decay amplitude   
    init_params = [9, 0.167,  3, 1.08, 0.1, 0.05] # B
    
    popt, pcov = curve_fit(
        fit_func,
        plot_taus,
        norm_avg_sig ,
        p0=init_params,
    )
    
    print(popt)
    ax.plot(plot_taus,norm_avg_sig , 'bs-', color=red, 
            linewidth = data_linewidth, markersize = marker_size, mfc = light_red)
    ax.plot(lin_taus, fit_func(lin_taus, *popt), 'k-', linewidth = fit_linewidth)
    
    
    ax.set_ylim([0.9791, 1.17])
    ax.set_xlim([1.95, 3.05])
    ax.set_xticks([2,2.5,3])
    # ax.set_xlabel(r'$\tau$ ($\mu$s)',  fontsize = f_size)
    # ax.set_ylabel(r'$\mathcal{C}$',  fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
# %%
def do_fit_plot_t2(file, folder, plt_color = blue, overal_decay = 865):
    fig_w = 0.7
    fig_l = 0.5
    
    marker_size = 0.75
    data_linewidth = 0.5
    fit_linewidth = 0.5
    
    f_size = 8
    
    fig_tick_l = 3
    fig_tick_w = 0.75
    
    
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
        
    data = tool_belt.get_raw_data(file, folder)
    norm_avg_sig = numpy.array(data['norm_avg_sig']) - 1
    uwave_pi_pulse = data['uwave_pi_pulse']
    
    taus = data['taus']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
        
    # Fit data
    lin_taus = numpy.linspace(plot_taus[0], plot_taus[-1], 1000)
    
    # fit_func = t2_decay
    #       T2, Offset, revival amp, oscillation amplitude, revival rate      
    init_params = [200, 0.125, 0.06, 100, 0.0323] 
    
    fit_func = lambda tau,  t2, Offset, A,  B, C :t2_decay(tau,  t2, Offset, A,  B, C,overal_decay )
    # fit_func = t2
    popt, pcov = curve_fit(
        fit_func,
        plot_taus,
        norm_avg_sig ,
        p0=init_params,
    )
    
    # popt[0] = 1e6
    print(popt)
    print(numpy.sqrt(numpy.diag(pcov)))
    ax.plot(plot_taus,norm_avg_sig+1 , 'b.-', color=plt_color, 
            markersize=marker_size,linewidth = data_linewidth)
    ax.plot(lin_taus, fit_func(lin_taus, *popt)+1, 'k-', linewidth = fit_linewidth)
    
    # text_eq =  "\n".join(
    #     (r"(Offset - A * e$^{-B sin^4(2\pi C \tau/2)}$ * e$^{-(\tau / T_2)^3}$) X",
    #       r"e$^{-(\tau / 1000 us)}$"))
    
    text_eq =  r"(Offset - A * e$^{-B sin^4(2\pi C \tau/2)}$ * e$^{-(\tau / T_2)^3}$)"
    
    text_popt = "\n".join(
        ( #A, B, C, D, T2
            r"$T_2$=%.3f" % (popt[0]) + r"$\pm$ %.3f us" % (numpy.sqrt(pcov[0][0])),
            r"Offset=%.3f" % (popt[1]) + r"$\pm$ %.3f" % (numpy.sqrt(pcov[1][1])),
            r"A=%.3f" % (popt[2]) + r"$\pm$ %.3f" % (numpy.sqrt(pcov[2][2])),
            r"B=%.3f" % (popt[3]) + r"$\pm$ %.3f" % (numpy.sqrt(pcov[3][3])),
            r"C=%.3f" % (popt[4]*1e3) + r"$\pm$ %.3f kHz" % (numpy.sqrt(pcov[4][4])*1e3)
        )
    )
    
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # ax.text(
    #     0.55,
    #     0.25,
    #     text_popt,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment="top",
    #     bbox=props,
    # )
    # ax.text(
    #     0.5,
    #     0.95,
    #     text_eq,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment="top",
    #     bbox=props,
    # )
    
    ax.set_xticks([0,100,200, 300])
    ax.set_xlabel(r'$\tau$ ($\mu$s)',  fontsize = f_size)
    ax.set_ylabel(r'$\mathcal{C}$ (arb.)',  fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    
     
# %%
def do_fit_t2_inbetween_revivials(file, folder):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
        
    data = tool_belt.get_raw_data(file, folder)
    norm_avg_sig = data['norm_avg_sig']
    uwave_pi_pulse = data['uwave_pi_pulse']
    
    taus = data['taus']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    plot_taus = plot_taus.tolist()
    
    # get just the data between the revivials:
    non_revival_sig = []
    non_revivial_taus = []
    for i in range(10):
        non_revival_sig += norm_avg_sig[9 + 31*i: 20 + 31*i]
        non_revivial_taus += plot_taus[9 + 31*i: 20 + 31*i]
        
    non_revival_sig = numpy.array(non_revival_sig) - 1
    non_revivial_taus = numpy.array(non_revivial_taus)
        
    # Fit data
    lin_taus = numpy.linspace(non_revivial_taus[0], non_revivial_taus[-1], 1000)
    
    fit_func = exp_decay
    #       Decay, Offset, amp     
    init_params = [336, 0.13] 
    
    popt, pcov = curve_fit(
        fit_func,
        non_revivial_taus,
        non_revival_sig ,
        p0=init_params,
        # bounds = ([0, 0.0, 0], [500, numpy.infty, 0.5]
            # )
    )
    
    print(popt)
    print(numpy.sqrt(numpy.diag(pcov)))
    ax.plot(lin_taus, fit_func(lin_taus, *popt), 'r-', linewidth = 2)
    
    text_eq = r"B *  e$^{-(\tau / t_D)}$"
    
    text_popt = "\n".join(
        ( #A, B, C, D, T2
            r"$t_D$=%.3f" % (popt[0]) + r"$\pm$ %.3f us" % (numpy.sqrt(pcov[0][0])),
            r"B=%.3f" % (popt[1]) + r"$\pm$ %.3f" % (numpy.sqrt(pcov[1][1])),
        )
    )
    
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.55,
        0.15,
        text_popt,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )
    ax.text(
        0.5,
        0.95,
        text_eq,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )
    
    ax.plot(non_revivial_taus,non_revival_sig , 'ro', linewidth = 0.5)
    ax.plot(plot_taus,numpy.array(norm_avg_sig)-1 , 'b.', linewidth = 0.5)
    ax.set_xlabel('Taus (us)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend(loc='lower right')
    
    
# %%

def do_fft(file_C,file_D, folder):
    data = tool_belt.get_raw_data(file_D, folder)
    taus = data['taus']
    norm_avg_sig_D = numpy.array(data['norm_avg_sig'])
    uwave_pi_pulse = 0#data['uwave_pi_pulse']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
    ft = numpy.fft.rfft(norm_avg_sig_D)
    freqs = numpy.fft.rfftfreq(len(plot_taus), d = (plot_taus[1] - plot_taus[0]))
    ft_mag_D = numpy.absolute(ft)
    
    fit_func = tool_belt.gaussian
    #       amp, position, width, offset  
    init_params = [2, 9, 1, 0] 
    
    popt, pcov = curve_fit(
        fit_func,
        freqs[3:],
        ft_mag_D[3:] ,
        p0=init_params,
        # bounds = ([0, 0.0, 0], [500, numpy.infty, 0.5]
            # )
    )
    print('{} +/- {} MHz'.format(popt[1],numpy.sqrt(pcov[1][1]) ))
    
    lin_f = numpy.linspace(freqs[0], freqs[-1], 1000)
    
    
    fig_w = 8.2
    fig_l = 2.5
    marker_size = 4
    linewidth = 2
    f_size = 8
    fig_tick_l = 3
    fig_tick_w = 0.75
    fig, axes = plt.subplots(1,2)
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    ax = axes[0]
    ax.plot(lin_f, fit_func(lin_f, *popt), 'b', color = red, linewidth = linewidth)

    ax.plot(freqs[1:], ft_mag_D[1:], 'rs', color=red,  linewidth = linewidth, 
            markersize = marker_size, mfc=light_red)
    
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    ax.set_ylabel('Magnitude', fontsize = f_size)
    ax.set_xlabel('Frequency (MHz)', fontsize = f_size)
    
    ax = axes[1]
    data = tool_belt.get_raw_data(file_C, folder)
    taus = data['taus']
    norm_avg_sig_C = numpy.array(data['norm_avg_sig'])
    uwave_pi_pulse = data['uwave_pi_pulse']
    plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
    
    ft = numpy.fft.rfft(norm_avg_sig_C)
    freqs = numpy.fft.rfftfreq(len(plot_taus), d = (plot_taus[1] - plot_taus[0]))
    ft_mag_C = numpy.absolute(ft)
    
    ax.plot(freqs[1:], ft_mag_C[1:], 'r^', color=blue,  linewidth = linewidth, 
            markersize = marker_size, mfc=light_blue)
    
    
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    
    
    ax.set_ylabel('Magnitude', fontsize = f_size)
    ax.set_xlabel('Frequency (MHz)', fontsize = f_size)
    
    fig2, ax =  plt.subplots()
    fig2.set_figwidth(1.5)
    fig2.set_figheight(1.1)
    
    ax.plot(lin_f, fit_func(lin_f, *popt), 'b', color = red, linewidth = linewidth)

    ax.plot(freqs[1:], ft_mag_D[1:], 'rs', color=red,  linewidth = linewidth, 
            markersize = 3, mfc=light_red)
    
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    ax.set_ylabel('Magnitude', fontsize = f_size)
    ax.set_xlabel('Frequency (MHz)', fontsize = f_size)
    ax.set_xlim([5.8, 12.2])
    ax.set_xticks([6,8,10,12])
    
    
    fig3, ax =  plt.subplots()
    fig3.set_figwidth(1.5)
    fig3.set_figheight(1.1)
    
    ax.plot(freqs[1:], ft_mag_C[1:], 'r^', color=blue,  linewidth = linewidth, 
            markersize = marker_size, mfc=light_blue)

    
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    ax.set_ylabel('Magnitude', fontsize = f_size)
    ax.set_xlabel('Frequency (MHz)', fontsize = f_size)
    ax.set_xlim([5.8, 12.2])
    ax.set_xticks([6,8,10,12])

# %% Plot data on same plot

def do_plot_comp(file_list, folder, fmt_list, label_list):
    
   
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    
    for f in range(len(file_list)):
        file = file_list[f]
        data = tool_belt.get_raw_data(file, folder)
        taus = data['taus']
        norm_avg_sig = numpy.array(data['norm_avg_sig'])
        uwave_pi_pulse = data['uwave_pi_pulse']
        plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
        
        ax.plot(plot_taus, norm_avg_sig, fmt_list[f], label = label_list[f])
        
    ax.set_ylabel('Contrast (arb. units)')
    ax.set_xlabel('Taus (us)')
    ax.legend(loc='lower right')
    
def do_plot_comp_AB(file_A, file_B, folder_sr, file_C, folder_scc):
    
    data = tool_belt.get_raw_data(file_A, folder_sr)
    uwave_pi_pulse = data['uwave_pi_pulse']
    taus_A = data['taus']
    plot_taus_A = (numpy.array(taus_A) + uwave_pi_pulse) / 1000
    norm_avg_sig_A = numpy.array(data['norm_avg_sig'])
    
    data = tool_belt.get_raw_data(file_B, folder_sr)
    uwave_pi_pulse = data['uwave_pi_pulse']
    taus_B = data['taus']
    plot_taus_B = (numpy.array(taus_B) + uwave_pi_pulse) / 1000
    norm_avg_sig_B = numpy.array(data['norm_avg_sig'])
    
    norm_avg_sig_AB = (norm_avg_sig_A + norm_avg_sig_B)/2
    
    #for first revival
    # s_AB = numpy.average(norm_avg_sig_AB[:20])
    # f_AB = numpy.average(norm_avg_sig_AB[-20:])
    #for t2
    s_AB = numpy.average(norm_avg_sig_AB[0])
    f_AB = numpy.average(norm_avg_sig_AB[9:20])
    
    
    
    data = tool_belt.get_raw_data(file_C, folder_scc)
    uwave_pi_pulse = data['uwave_pi_pulse']
    taus_C = data['taus']
    plot_taus_C = (numpy.array(taus_C) + uwave_pi_pulse) / 1000
    norm_avg_sig_C = numpy.array(data['norm_avg_sig'])
    
    #for first revival
    # s_C = numpy.average(norm_avg_sig_C[:20])
    # f_C = numpy.average(norm_avg_sig_C[-20:])
    #for t2
    s_C = numpy.average(norm_avg_sig_C[0])
    f_C = numpy.average(norm_avg_sig_C[9:20])
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    
    ax.plot(plot_taus_A, (norm_avg_sig_AB - s_AB)/(f_AB - s_AB),  'b-', label = 'A and B combined')
    ax.plot(plot_taus_C, (norm_avg_sig_C - s_C)/(f_C - s_C), 'g-', label = 'Non-resolved (scc)')
        
    # ax.plot(plot_taus_A, norm_avg_sig_AB,  'b-', label = 'A and B combined')
    # ax.plot(plot_taus_C, norm_avg_sig_C, 'g-', label = 'Non-resolved (scc)')
    
    ax.set_ylabel('Contrast (arb. units)')
    ax.set_xlabel('Taus (us)')
    ax.legend(loc='lower right')
    
    
# %%
folder_sr = 'pc_rabi/branch_master/super_resolution_spin_echo/2021_10'
folder_scc = 'pc_rabi/branch_master/scc_spin_echo/2021_10'

# first decay
file_C = '2021_10_14-10_39_50-johnson-dnv5_2021_09_23'
file_D = '2021_10_15-09_38_47-johnson-dnv5_2021_09_23'
file_CD = '2021_10_15-09_15_43-johnson-dnv5_2021_09_23'

#total t2
file_C_tot = '2021_10_18-10_10_20-johnson-dnv5_2021_09_23'
file_D_tot = '2021_10_18-10_10_33-johnson-dnv5_2021_09_23'
file_CD_tot = '2021_10_16-04_28_59-johnson-dnv5_2021_09_23'




### fit the data
# do_fit_plot_C(file_C, folder_sr)
# do_fit_plot_D(file_D, folder_sr)
# do_fit_plot_C_zoom(file_C, folder_sr)
# do_fit_plot_D_zoom(file_D, folder_sr)







# do_fit_plot_t2(file_C_tot, folder_sr, plt_color=green, overal_decay = 730)
# do_fit_plot_t2(file_D_tot, folder_sr, plt_color=orange, overal_decay = 1000)


# do_fit_plot_t2(file_CD_tot, folder_scc)
# do_fit_t2_inbetween_revivials(file_C_tot, folder_sr)
# do_fit_t2_inbetween_revivials(file_D_tot, folder_sr)

### plot fourier transform
do_fft(file_C, file_D, folder_sr)

### plot data together
# file_list = [file_C, file_D]
# fmt_list = ['b-', 'r-']
# label_list = ['C', 'D']
# do_plot_comp(file_list, folder_sr, fmt_list, label_list)


# do_plot_comp_CD(file_C, file_D, folder_sr, file_CD, folder_scc)


# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:19:47 2022

@author: kolkowitz
"""

import numpy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.special import eval_chebyt
from scipy.special import eval_chebyu
from numpy import pi

# uN = 0.762e-3 #MHz/G
# uN_13C = 0.7*uN
uN_13C = 2*pi*1.07e-3 #MH/G

Bz= 15#G
f_La = 0.098 #MHz

dd_model_coeff_dict = tool_belt.get_dd_model_coeff_dict()

def S_bath(t, fL, lambd, sigma, T2, a_list  ):
    
    sum_expr = a_list[0]
    for i in range(len(a_list)-1):
        n=i+1
        sum_expr += a_list[n]*numpy.exp(-n**2 * (t)**2 * sigma**2 / 2) * numpy.cos(n*t*(fL*2*pi))

    X =4* lambd**2 * sum_expr
    return numpy.exp(-X) * numpy.exp(-(2*t/T2)**3)


def S_13C(t, N, fL, fh, theta):
    '''
    T taminiau equations
    '''
    wh = fh * 2*pi
    wL = fL * 2*pi
    A = wh*numpy.cos(theta)
    B = wh*numpy.sin(theta)
    
    w_tild = numpy.sqrt((A+wL)**2 + B**2)
    mz = (A + wL) / w_tild
    mx = B/w_tild
    
    alpha = w_tild*t
    beta = wL*t
    
    term_c = numpy.cos(alpha) * numpy.cos(beta)
    term_s = numpy.sin(alpha) * numpy.sin(beta)
    cosphi=  term_c - mz * term_s

    # print(alpha)
    num = (1-numpy.cos(alpha))*(1-numpy.cos(beta))
    den = (1 + numpy.cos(alpha)*numpy.cos(beta) - mz*numpy.sin(alpha)*numpy.sin(beta))
    onemdot = mx**2 * num / den

    k = int((N/2)-1)
    return 1 - onemdot * eval_chebyu(k, cosphi)**2 * (1 - cosphi**2)

# def S_13C(t, N, Ax, Az):
#     t = 2*t
#     B_vec = numpy.array([0,0,Bz])
#     A_vec = numpy.array([Ax, 0, Az])
    
#     w0 = numpy.linalg.norm(uN_13C*B_vec)
#     w1 = numpy.linalg.norm(uN_13C*B_vec + A_vec)
#     n0 = uN_13C*B_vec/w0
#     n1 = (uN_13C*B_vec + A_vec)/w1
    
#     p0 = w0 * t / 2
#     p1 = w1 * t / 2
    
    
#     term_c = numpy.cos(w0*t*2*pi/2) * numpy.cos(w1*t*2*pi/2)
#     term_s = numpy.sin(w0*t*2*pi/2) * numpy.sin(w1*t*2*pi/2)
#     cosphi=  term_c - numpy.dot(n0, n1) * term_s


#     dot_n = numpy.dot(n0, n1)
#     # print(dot_n)
#     # print(n1)
#     # term_ss = numpy.sin(w0*t*2*pi/2)**2 * numpy.sin(w1*t*2*pi/2)**2
#     # term_phi = numpy.sin(N*phi/2)**2 / numpy.cos(phi / 2)**2
#     k = int((N/2))
#     return 1 - 4*(1 - dot_n**2) * numpy.sin(p0/2)**2 * numpy.sin(p1/2)**2 *(1-cosphi) * eval_chebyu(k-1, cosphi)**2
#     # return 1 - 4*(1 - dot_n**2) * numpy.sin(p0)**2/2 * numpy.sin(p1)**2/2 *(1-cosphi) * eval_chebyu(k-1, cosphi)**2

    
def pop_S(t, N, fh, theta, fL, lambd, sigma,T2, a_list):
    return (S_13C(t, N, fL, fh, theta)*S_bath(t, fL, lambd, sigma,T2, a_list) + 1)/2

    
# file = '2022_08_26-10_11_36-rubin-nv1_2022_08_10' # XY4-1
# folder = 'pc_rabi/branch_master/dynamical_decoupling_xy4/2022_08'
# file='2022_09_08-08_19_09-rubin-nv4_2022_08_10' #XY4-1
# file = '2022_09_08-09_20_15-rubin-nv1_2022_08_10' # XY4-1
# file = '2022_09_11-00_27_23-rubin-nv8_2022_08_10' # XY4-1
# file = '2022_09_02-07_01_43-rubin-nv1_2022_08_10' # XY4-2f

# file = '2022_09_14-01_56_22-rubin-nv4_2022_08_10' #xy4-2
# file = '2022_09_14-09_53_52-rubin-nv4_2022_08_10' #XY4-2

# file = '2022_09_07-09_06_55-rubin-nv8_2022_08_10' # XY4-2
file = '2022_09_11-15_02_12-rubin-nv8_2022_08_10'
folder = 'pc_rabi/branch_master/dynamical_decoupling_xy4/2022_09'


data = tool_belt.get_raw_data(file, folder)
sig_counts = numpy.array(data['sig_counts'])
ref_counts = numpy.array(data['ref_counts'])
try:
    num_xy_reps = data['num_xy4_reps']
    xy_mode = 4
    title = 'XY4-{}'.format(num_xy_reps)
except Exception:
    num_xy_reps = data['num_xy8_reps']
    xy_mode = 8
    title = 'XY8-{}'.format(num_xy_reps)
num_pi_pulses = num_xy_reps*xy_mode

precession_time_range = data['precession_time_range']
num_steps = data['num_steps']
try:
    taus = numpy.array(data['taus'])
except Exception:    
    taus = numpy.linspace(
        precession_time_range[0],
        precession_time_range[-1],
        num=num_steps,
    )
            
plot_taus = taus/1e3 
    
avg_sig_counts = numpy.average(sig_counts, axis=0)
avg_ref_counts = numpy.average(ref_counts, axis=0)

max_ref =  numpy.average(avg_ref_counts)
min_ref = numpy.average(avg_sig_counts[:3])
contrast = min_ref - max_ref

norm_avg_sig = (avg_sig_counts - max_ref) / contrast 
    
taus_lin = numpy.linspace(plot_taus[0], plot_taus[-1],600)


c13_coupling = True


if c13_coupling:
    ###____ include single 13C coupling ____###
    fit_func = lambda  t, fh, theta, lambd, sigma: pop_S(t,num_pi_pulses, fh, theta, 0.096, lambd, sigma,  40, 
                                                      dd_model_coeff_dict['{}'.format(num_pi_pulses)] ) 

    for amp in [10]:#1.4
        for theta in [60*pi/180]:
            fig, ax = plt.subplots()
            init_params = [amp, theta, 0.3, 0.01]
            # init_params = [0.32661016, 0.98844318, 0.5221769,  0.04097683]
            popt, pcov = curve_fit(
                fit_func,
                numpy.array(plot_taus),
                norm_avg_sig,
                # sigma=norm_avg_sig_ste,
                # absolute_sigma=True,
                p0=init_params,
                bounds=(0, [numpy.inf,pi,numpy.inf,numpy.inf]),
            )
            print(popt)
            # popt = init_params
            ax.plot(
                    taus_lin,
                    fit_func(taus_lin, *popt),
                    "-",
                    color="red",
                    label="XY4-2",
                ) 
            
            ax.plot(
                    plot_taus,
                    norm_avg_sig,
                    ".-",
                    color="blue",
                    label="data",
                )    
            ax.set_xlabel(r"Inter-pulse time, $\tau$ (us)")
            ax.set_ylabel("Normalized signal Counts")
            ax.set_title(title)
            ax.legend()
else:

    fig, ax = plt.subplots()
    ###____ exclude single 13C coupling ____###
    fit_func = lambda  t,f_L, lambd, sigma:( S_bath(t,f_L, lambd, sigma, 100, 
                                                    dd_model_coeff_dict['{}'.format(num_pi_pulses)]) +1)/2
    init_params = [ 0.1, 0.2, 0.01]
    popt, pcov = curve_fit(
        fit_func,
        numpy.array(plot_taus),
        norm_avg_sig,
        # sigma=norm_avg_sig_ste,
        # absolute_sigma=True,
        p0=init_params,
        # bounds=(0, numpy.inf),
    )
    print(popt)
    ax.plot(
            taus_lin,
            fit_func(taus_lin, *popt),
            "-",
            color="red",
            label="Spin bath",
        ) 
    
    
    # text_popt = '\n'.join((
    #                     r'$f_L = $' + '%.4f'%(popt[-3]) + ' MHz',
    #                   r'$\lambda = $' + '%.2f'%(popt[-2]),
    #                   r'$\sigma = $' + '%.4f'%(popt[-1])))

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.4, 0.9, text_popt, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)
    
    
    ax.plot(
            plot_taus,
            norm_avg_sig,
            ".-",
            color="blue",
            label="data",
        )    
    ax.set_xlabel(r"Inter-pulse time, $\tau$ (us)")
    ax.set_ylabel("Normalized signal Counts")
    ax.set_title(title)
    ax.legend()


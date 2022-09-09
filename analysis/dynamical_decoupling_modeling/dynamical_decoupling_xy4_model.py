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
from numpy import pi

uN = 0.762e-3 #MHz/G
uN_13C = 0.7*uN

Bz= 94.6 #G
# Ax = 100 #MHz
# Ay = 0 #MHz
# Az = -75 #MHz

dd_model_coeff_dict = tool_belt.get_dd_model_coeff_dict()

# def S_decay(t, N, Ax, Az, T):
#     B_vec = numpy.array([0,0,Bz])
#     A_vec = numpy.array([Ax, 0, Az])
    
#     w0 = numpy.linalg.norm(uN_13C*B_vec)
#     w1 = numpy.linalg.norm(uN_13C*B_vec + A_vec)
#     n0 = uN_13C*B_vec/w0
#     n1 = (uN_13C*B_vec + A_vec)/w1
    
    
#     term_c = numpy.cos(w0*t*2*pi/2)* numpy.cos(w1*t*2*pi/2)
#     term_s = numpy.sin(w0*t*2*pi/2)* numpy.sin(w1*t*2*pi/2)
#     phi=  numpy.arccos(term_c - numpy.dot(n0, n1) * term_s)


#     cross_n = numpy.cross(n0, n1)
#     term_ss = numpy.sin(w0*t*2*pi/2)**2 * numpy.sin(w1*t*2*pi/2)**2
#     term_phi = numpy.sin(N*phi/2)**2 / numpy.cos(phi / 2)**2
#     return (1 - numpy.linalg.norm(cross_n)**2 * term_ss * term_phi)*numpy.exp((-t/T)**3)
    
# def pop_S_decay(t, N, Ax, Az, T):
#     return (S_decay(t, N, Ax, Az, T) + 1)/2

def S_bath(t, fL, lambd, sigma,T2, a_list  ):
    sum_expr = a_list[0]
    for i in range(len(a_list)-1):
        n=i+1
        sum_expr += a_list[n]*numpy.exp(-n**2 * (t)**2 * sigma**2 / 2) * numpy.cos(n*t*(fL*2*pi))

    X =4* lambd**2 * sum_expr
    return numpy.exp(-X) * numpy.exp(-(2*t/T2)**3)

def S_bath_test(t, fL, lambd, sigma,T2, a_list ):
    # for XY4
    # a_list = [9, -4, -12, 4, 8, -4, -4, 4, -1]
    
    sum_expr = a_list[0]
    # lambd = 0.25
    # sigma = 0.1 * fL #/ (2*pi)
    for i in range(len(a_list)-1):
        n=i+1
        # print(n)
        sum_expr += a_list[n]*numpy.exp(-n**2 * (t*2*pi)**2 * (sigma)**2 / 2) * numpy.cos(n*t*2*pi)#*fL*2*pi)

    X = 4*lambd**2 * sum_expr
    return numpy.exp(-X)#* numpy.exp(-(t/T2)**3)

# def S_bath_8(t, fL, lambd, sigma,T2 ):
#     # for XY8
#     a_list = [17, -4, -28, 4, 24, 
#               -4, -20, 4, 16, -4,
#               -12, 4, 8, -4,
#               -4, 4, -1]
    
#     sum_expr = a_list[0]
#     # lambd = 0.25
#     # sigma = 0.1 * fL #/ (2*pi)
#     for i in range(len(a_list)-1):
#         n=i+1
#         # print(n)
#         sum_expr += a_list[n]*numpy.exp(-n**2 * t**2 * sigma**2 / 8) * numpy.cos(n*t*pi)#*fL*2*pi)

#     X = lambd**2 * sum_expr
#     return numpy.exp(-X)#* numpy.exp(-(t/T2)**3)

# def S_bath_SE(t, fL, lambd, sigma, T2):
#     t_se = 2*t
#     # for spin echo
#     a_list = [3, -4, 1]
    
#     sum_expr = a_list[0]
#     # lambd = 0.25
#     # sigma = 0.1 * fL #/ (2*pi)
#     for i in range(len(a_list)-1):
#         n=i+1
#         # print(n)
#         sum_expr += a_list[n]*numpy.exp(-n**2 * t_se**2 * sigma**2 / 8) * numpy.cos(n*t_se*pi)#*fL*2*pi)

#     X = lambd**2 * sum_expr
#     return numpy.exp(-X)# * numpy.exp(-(t/T2)**3)
    
# def F(w, t):
#     # return 2*numpy.sin(N*w*2*pi*t/2)**2 * (1 - 1/numpy.cos(w*2*pi*t/2))**2
#     return 8*numpy.sin(w*t/2)**4

# def SS(w ,t):
#     sigma = 0.1 * w
#     lamd = 0.25
#     wL = 0.10 * 2*pi
#     return numpy.sqrt(2*pi)/sigma * lamd**2 * (w*2*pi)**2 * numpy.exp(-(w*2*pi - wL)**2 / (2*sigma**2))

# def S_bath_SE(t, fL):
#     # F = 8*numpy.sin(w*t/2)**4
#     # S = numpy.sqrt(2*pi)/sigma * lamd**2 * (w*2*pi)**2 * numpy.exp(-(w*2*pi - wL)**2 / (2*sigma**2))
    
#     integrand = lambda w:-F(w, t)*SS(w, t)/(2*pi*(w*2*pi)**2)
#     # arg = lambda w:-F(w, t)*SS(w, t)/(2*pi*(w*2*pi)**2)
    
#     res, _ = quad(lambda w:integrand, 0,numpy.infty)

#     return numpy.exp(-res)

def S(t, N, Ax, Az):
    B_vec = numpy.array([0,0,Bz])
    A_vec = numpy.array([Ax, 0, Az])
    
    w0 = numpy.linalg.norm(uN_13C*B_vec)
    w1 = numpy.linalg.norm(uN_13C*B_vec + A_vec)
    n0 = uN_13C*B_vec/w0
    n1 = (uN_13C*B_vec + A_vec)/w1
    
    
    term_c = numpy.cos(w0*t*2*pi/2) * numpy.cos(w1*t*2*pi/2)
    term_s = numpy.sin(w0*t*2*pi/2) * numpy.sin(w1*t*2*pi/2)
    phi=  numpy.arccos(term_c - numpy.dot(n0, n1) * term_s)


    cross_n = numpy.cross(n0, n1)
    term_ss = numpy.sin(w0*t*2*pi/2)**2 * numpy.sin(w1*t*2*pi/2)**2
    term_phi = numpy.sin(N*phi/2)**2 / numpy.cos(phi / 2)**2
    return (1 - numpy.linalg.norm(cross_n)**2 * term_ss * term_phi)
    
def pop_S(t, N, Ax, Az):
    return (S(t, N, Ax, Az) + 1)/2

    
# file0 = "2022_08_31-15_13_26-rubin-nv4_2022_08_10"
# file0 = '2022_08_26-10_11_36-rubin-nv1_2022_08_10' # XY4
file0='2022_09_08-08_19_09-rubin-nv4_2022_08_10' #XY4-1
folder = 'pc_rabi/branch_master/dynamical_decoupling_xy4/2022_09'

file_list = [file0]
master_plot_taus = []
master_norm_sig = []
i = 0
for file in file_list:
    data = tool_belt.get_raw_data(file, folder)
    sig_counts = numpy.array(data['sig_counts'])
    ref_counts = numpy.array(data['ref_counts'])
    precession_time_range = data['precession_time_range']
    num_steps = data['num_steps']
    # taus = numpy.linspace(
    #     precession_time_range[0],
    #     precession_time_range[-1],
    #     num=num_steps,
    # )
    taus = numpy.array(data['taus'])
    plot_taus = taus/1e3 
    
    # plot_taus = numpy.array(data['plot_taus'])
    
    avg_sig_counts = numpy.average(sig_counts, axis=0)
    avg_ref_counts = numpy.average(ref_counts, axis=0)
    
    if i == 0:
        max_ref =  numpy.average(avg_ref_counts)
        min_ref = numpy.average(avg_sig_counts[:3])
        contrast = min_ref - max_ref
    
    norm_avg_sig = (avg_sig_counts - max_ref) / contrast 
    
    master_plot_taus = master_plot_taus + plot_taus.tolist()
    master_norm_sig = master_norm_sig + norm_avg_sig.tolist()
    i += 1
    
# fit_func = lambda t, Ax, Az: pop_S(t, 4, Ax, Az)
# init_params = [100e-3, -75e-3]
# # fit_func = lambda t, Ax, Az, T: pop_S_decay(t, 4, Ax, Az, T)
# # init_params = [0.2, -0.5, 1000]
# popt, pcov = curve_fit(
#     fit_func,
#     numpy.array(master_plot_taus),
#     master_norm_sig,
#     # sigma=norm_avg_sig_ste,
#     # absolute_sigma=True,
#     p0=init_params,
#     # bounds=(min_bounds, max_bounds),
# )
# print(popt)


fig, ax = plt.subplots()
taus_lin = numpy.linspace(master_plot_taus[0], master_plot_taus[-1],600)
# taus_lin = numpy.linspace(0, 2.5,600)

# fit_func = lambda t, Ax, Az: pop_S(t, 4, Ax, Az)
# init_params = [100e-3, -75e-3]
# ax.plot(
#         taus_lin,
#         fit_func(taus_lin, *init_params),
#         "-",
#         color="orange",
#         label="XY4-1",
#     ) 
# fit_func = lambda t, Ax, Az: pop_S(t, 8, Ax, Az)
# init_params = [100e-3, -75e-3]
# ax.plot(
#         taus_lin,
#         fit_func(taus_lin, *init_params),
#         "-",
#         color="red",
#         label="XY4-2",
#     ) 

# taus_lin = numpy.linspace(0, 4/(Bz*uN_13C),600)
# print(S_bath(0, 0.1, 0.25, 0.01, 0))

#######________________________#############
# fit_func = lambda  t, fL, lambd, sigma, T2: (S_bath(t,fL/(2*4), lambd, sigma, T2, a_list_4 )   +1)/2
# init_params = [0.1, 2, 0.001, 500]
# popt, pcov = curve_fit(
#     fit_func,
#     numpy.array(master_plot_taus),
#     master_norm_sig,
#     # sigma=norm_avg_sig_ste,
#     # absolute_sigma=True,
#     p0=init_params,
#     # bounds=([0,0,0,100], [numpy.inf,numpy.inf, numpy.inf, numpy.inf]),
# )
# print(popt)
# ax.plot(
#         taus_lin,
#         fit_func(taus_lin, *popt),
#         "-",
#         color="red",
#         label="Spin bath, XY4",
#     ) 
#######________________________#############

# fit_func = lambda  t, fL, lambd, sigma: (S_bath_test(t,fL, lambd, sigma, 0, a_list_2 ) )#+1)/2
# # init_params = [0.25, 0.1]
# ax.plot(
#         taus_lin,
#         fit_func(taus_lin, *init_params),
#         "-",
#         color="black",
#         label="Spin bath, XY2",
#     ) 

#######________________________#############
fit_func = lambda  t, fL, lambd, sigma, T2: (S_bath(t,fL, lambd, sigma, T2, a_list_4 ) +1)/2
init_params = [0.1, 0.4, 0.003, 400]
popt, pcov = curve_fit(
    fit_func,
    numpy.array(master_plot_taus),
    master_norm_sig,
    # sigma=norm_avg_sig_ste,
    # absolute_sigma=True,
    p0=init_params,
    bounds=(0, numpy.inf),
)
print(popt)
ax.plot(
        taus_lin,
        fit_func(taus_lin, *popt),
        "-",
        color="red",
        label="Spin bath, XY4-2",
    ) 
    #######________________________#############
# fit_func = lambda  t, fL, lambd, sigma: (S_bath_test(t,fL,lambd, sigma, 0, a_list_se ) )#+1)/2
# # # init_params = [0.75, 0.1]
# ax.plot(
#         taus_lin,
#         fit_func(taus_lin, *init_params),
#         "-",
#         color="green",
#         label="Spin bath, Spin Echo",
#     ) 

# ax.set_xlabel("Evolution time (1/w)")
# ax.set_ylabel("Coherence")


ax.plot(
        master_plot_taus,
        master_norm_sig,
        "o",
        color="blue",
        label="data",
    )    
ax.set_xlabel("Inter-pulse time, tau (us)")
ax.set_ylabel("Normalized signal Counts")
ax.set_title('XY4-1')
ax.legend()


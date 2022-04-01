# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 08:15:27 2021

@author: agard
"""
from scipy.special import j1
import numpy
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
from scipy.optimize import curve_fit
# %%

fig_tick_l = 3
fig_tick_w = 0.75

color_red = '#ec2626'
color_yellow = '#ffd436'
color_flour = '#9a2765'#'#f89522'
fig_w = 4
fig_l = 1.52

f_size = 8

mu = u"\u03BC"

#gaussian (sigma) to Abbe diffraction limit (d)
# d = sigma/c
c = 0.21
abbe_515 = 198 #515 nm light, 1.3 NA
fwhm =2.355
conf_fwhm = 145*fwhm




wavelength = 638
NA = 1.3  
pi = numpy.pi
# v = 50 / (30 * NA**2 * pi /wavelength**2)**2 # I should measure the actual value here
v = 1.6e10
P = 20
t = 100
offset = 300

# %%
def power_law(x, a, b):
    return a*x**b
    
    
def intensity_scaling(P):
    I = P*NA**2*pi/wavelength**2
    return I


def radial_scaling(r):
    x = 2*pi*NA*r/wavelength
    return x

def intensity_airy_func(r, P):
    I = intensity_scaling(P)
    x = radial_scaling(r)
    
    return I * (2*j1(x) / x)**2 

def intensity_airy_func_offset(r, P, offset):
    I = intensity_scaling(P)
    x = radial_scaling(r-offset)
    
    return I * (2*j1(x) / (x))**2 #+0.001*I


def eta_offset(r,v, P, t, offset):
    return numpy.exp(-v*t*intensity_airy_func_offset(r, P, offset)**2)


# %%
def do_plot(file, folder, scale):
    data = tool_belt.get_raw_data(file, folder)
    counts = data['readout_counts_avg']
    coords_voltages = data['coords_voltages']
    x_voltages = numpy.array([el[0] for el in coords_voltages])
    y_voltages = numpy.array([el[1] for el in coords_voltages])
    try:
        rad_dist=numpy.array(data['rad_dist'])*scale
    except Exception:
        rad_dist = numpy.sqrt((x_voltages - x_voltages[0])**2 +( y_voltages - y_voltages[0])**2)*scale
    
    
    ### Exclude the left side data:
    # counts = counts[9:]
    # rad_dist = rad_dist[9:]
    
    #manually offset the real data
    # Data is about 0.55X the size it shoudl be...
    # rad_dist = numpy.array(rad_dist) -1860
    rad_dist = numpy.array(rad_dist) + 35
    
    eta_n = lambda r, t, offset: eta_offset(r,v, P, t, offset)
    
    # init_fit = [350, -50, 1.5, 3, 0.63] 
    # init_fit = [350, 4, 1.5, 3, 0.7] 
    # opti_params, cov_arr = curve_fit(eta_n,
    #       rad_dist,
    #       counts,
    #       p0=init_fit
    #       )
    # print(opti_params)
    opti_params = [326.13302951,   4.22767238]
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)

    
    s = wavelength/(2*NA)
    x_tix = numpy.linspace(-2, 5, 8)*s
    x_tix = numpy.linspace(-1/4, 5/4, 7)*wavelength
    ax.set_xticks(x_tix)
    # ax.set_yticks([])
    ax.set_ylim([-5e-7,1e-05])

    # x_lin = numpy.linspace(-1500, 1500, 601)
    x_lin = numpy.linspace(-300, 900, 601)
    y_vals = intensity_airy_func_offset(x_lin, P, opti_params[1]) 
    ax.plot(x_lin, y_vals, '-', color= color_red,  linewidth = 1)
    
    ax2=ax.twinx()
    right_side = ax2.spines["right"]
    right_side.set_visible(False)
    left_side = ax2.spines["left"]
    left_side.set_visible(False)
    top_side = ax2.spines["top"]
    top_side.set_visible(False)
    bot_side = ax2.spines["bottom"]
    bot_side.set_visible(False)
    # ax2.set_yticks([])
    # ax2.set_xlim(-350, 1350)
    ax2.set_xlim(-270, 790)
    
    y_vals = eta_n(x_lin, *opti_params)
    
    ax2.plot(x_lin, y_vals, '-', color= color_flour,  linewidth = 1)
    # ax2.plot(rad_dist, counts, 'o', color= color_flour,  markersize = 1)




    ######Plot a mini red airy ring plot too
    fig, ax = plt.subplots()
    fig.set_figwidth(1)
    fig.set_figheight(2/3)
    
    x_lin = numpy.linspace(-700, 700, 1401)
    y_vals = intensity_airy_func_offset(x_lin, P, 0)
    ax.plot(x_lin, y_vals, '-', color= color_red,  linewidth = 1)
    # ax.set_ylim([-3e-05,0.000273])
    
    x_tix = numpy.linspace(-5, 5, 11)*s
    # ax.set_xticks(x_tix)
    
    # ax2=ax.twinx()
    # y_vals = eta_n(x_lin, *opti_params)
    # ax2.plot(x_lin, y_vals, '-', color= color_flour,  linewidth = 1)
    # ax2.set_ylim([-2, 18])
# %%

# plot_red_airy()
# plot_yellow_airy()
# plot_NV_response()
# plot_all_three()

# file = '2021_11_18-03_37_50-johnson-nv1_2021_11_17' #[1,0, 3, 12, 0.5]
# file = '2021_11_18-11_50_46-johnson-nv1_2021_11_17' #  [200, 0, 3, 3, 0.56] 
# folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_11'
# scale = 1e3

file = '2021_12_12-13_19_14-johnson-nv0_2021_12_10'
folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_12'
scale = 1e3

# file = '2021_07_27-09_03_48-johnson-nv1_2021_07_21'
# file = '2021_07_28-00_46_22-johnson-nv1_2021_07_27' # [2, -100, 8, 23.5, 0.75]
# folder = 'pc_rabi/branch_master/SPaCE/2021_07'
# scale = 35e3
do_plot(file, folder, scale)

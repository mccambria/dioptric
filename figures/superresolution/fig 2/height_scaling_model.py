# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:19:51 2022

@author: agard
"""
import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %%
NA = 1.3
wavelength = 638
fwhm =1.825 # 2* (ln(2))^1/4
scale = 0.99e3

mu = u"\u03BC"
superscript_minus = u"\u207B"
fig_tick_l = 3
fig_tick_w = 0.75
f_size = 8

# %%
y2 = numpy.array([0.10725208267422458, 0.061904059004131616, 0.22799717741944828,
                  0.2746149845700901, 0.3914981041785149, 0.4173360144810147, 
                  0.527191672104991, 0.5145939416015899, 0.5775893827543029,
                  0.5921215196025877, 0.5875214039394381, 0.5903589737905577,
                  0.5499300634709423]) #heights list, in norm. NV population

y2_err = numpy.array([0.005984916244208165, 0.005170204769169157,
                      0.010894691236281381, 0.011984575860211743,
                      0.012392123510894438, 0.011174913578460573,
                      0.013134481921080056, 0.010777221436760243,
                      0.011500316844723583, 0.014574535007293887,
                      0.014299228018942033, 0.010661145902811675,
                      0.013614070527270644])

t = numpy.array([10.0, 11.0, 7.5, 5.0, 2.5, 1.0, 0.75, 0.5,
              0.25, 0.1, 0.075, 0.05, 0.01]) # in ms

lin_x_vals = numpy.linspace((0.01), (11), 100)
# %%

def exp_decay(t, A, alpha, e):
    return A * numpy.exp(-t* (numpy.log(2)/alpha) * e**2)

def gaussian_quad(x,  *params):
    """
    Calculates the value of a gaussian with a x^4 in the exponent
    for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the Gaussian
            0: coefficient that defines the peak height
            1: mean, defines the center of the Gaussian
            2: standard deviation-like parameter, defines the width of the Gaussian
            3: constant y value to account for background
    """

    coeff, mean, stdev, offset = params
    var = stdev ** 4  # variance squared
    centDist = x - mean  # distance from the center
    return offset + coeff ** 2 * numpy.exp(-(centDist ** 4) / (var))

def plot_inset(file_name, folder, threshold):

    fig_w =0.9
    fig_l = fig_w * 1
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)

    ax.set_xlabel(r'$\Delta$x (nm)', fontsize = f_size)
    ax.set_ylabel(r'$\eta$', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                    direction='in',grid_alpha=0.7, labelsize = f_size)


    data = tool_belt.get_raw_data( file_name, folder)

    # convert single shot measurements to NV- population
    raw_counts = numpy.array(data['readout_counts_array'])
    for r in range(len(raw_counts)):
        row = raw_counts[r]
        for c in range(len(row)):
            current_val = raw_counts[r][c]
            if current_val < threshold:
                set_val = 0
            elif current_val >= threshold:
                set_val = 1
            raw_counts[r][c] = set_val
    counts = numpy.average(raw_counts, axis = 1)

    nv_sig = data['nv_sig']
    coords = nv_sig['coords']
    # rad_dist = numpy.array(data['rad_dist'])*scale
    num_steps = data['num_steps_a']
    offset_2D = data['offset_2D']
    coords_voltages = data['coords_voltages']
    x_voltages = numpy.array([el[0] for el in coords_voltages]) -offset_2D[0] - coords[0]
    rad_dist = -x_voltages*scale

    ax.plot(numpy.flip(rad_dist), numpy.flip(counts),
            'o',  color= 'orange',  markersize = 2,
            markeredgewidth=0.0,
            )


    opti_params = []
    fit_func = gaussian_quad


    init_fit = [2, rad_dist[int(num_steps/2)], 15, 7]
    try:
        opti_params, cov_arr = curve_fit(fit_func,
              rad_dist,
              counts,
              p0=init_fit
              )
        lin_radii = numpy.linspace(rad_dist[0],
                        rad_dist[-1], 100)
        ax.plot(lin_radii,
               fit_func(lin_radii, *opti_params), color= 'black',#color_list[f],
               linestyle = 'dashed' , linewidth = 1)

        print(opti_params)
    except Exception:
        text = 'Peak could not be fit'
        print(text)



def plot_height_vs_duration():
    fit_func = exp_decay
    e =  4.39e-4
    A = 0.579
    alpha = 7.7e-7


    params = [A, alpha, e]

    fig_w =3.3
    fig_l = fig_w * 0.9
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    ax.plot(lin_x_vals, fit_func(lin_x_vals, *params),
                color = 'purple',  linestyle = (0,(6,3))   , linewidth = 1,)


    ax.errorbar(t, numpy.array(y2),
                yerr = y2_err,
                    fmt='o', color = 'black',
                    linewidth = 1, markersize = 5, mfc='#d6d6d6')

    ax.set_xlabel(r'Depletion pulse duration, $\tau$ (ms)', fontsize = f_size)
    ax.set_ylabel('Normalized NV pop. height', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    ax.set_yscale('log')

# %%

plot_height_vs_duration()

file_name = '2021_11_25-04_39_56-johnson-nv1_2021_11_17' #0.5 ms (yellow)
# file_name = '2021_11_25-16_38_19-johnson-nv1_2021_11_17' #0.05 ms (green)
folder = 'paper_materials/super_resolution/Fig 2/1D data'
threshold = 5
plot_inset(file_name, folder, threshold)

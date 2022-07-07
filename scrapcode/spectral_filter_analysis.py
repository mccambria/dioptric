# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:43:33 2022

@author: kolkowitz
"""

import numpy
import utils.tool_belt as tool_belt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
 #%%

folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/horiba_spectrometer/2022_05'
filter_file = 'thorlabs_740_bp'
spectrum_file = '2022_05_16-spot1-nofilter'

f_data = tool_belt.get_raw_data(filter_file, folder)
s_data = tool_belt.get_raw_data(spectrum_file, folder)

f_wavlength = f_data['wavelengths']
s_wavlength = s_data['wavelengths']

f_vals = numpy.array(f_data['counts'])/100
s_vals = numpy.array(s_data['counts']) - 600

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

#%% fit

# spectrum
fit_func_s = tool_belt.lorentzian
init_params = [740, 2500, 25, 0]

popt_s, _ = curve_fit(fit_func_s, s_wavlength, s_vals,
                        p0=init_params)
# Thorlabs FB740-10 filter
# fit_func_f = tool_belt.gaussian
# init_params = [1, 740, 10, 0]

# popt_f, _ = curve_fit(fit_func_f, f_wavlength, f_vals,
#                         p0=init_params)

# Edmund optics 67-841 filter
fit_func_f = gaussian_quad
init_params = [numpy.sqrt(0.67), 742, 5, 0]

popt_f = init_params
# popt_f, _ = curve_fit(fit_func_f, f_wavlength, f_vals,
#                         p0=init_params)

#%% plot

lin_w = numpy.linspace(600, 900, 500)

fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.plot(s_wavlength, s_vals,'b.',label='spectra data')
ax.plot(lin_w, fit_func_s(lin_w, *popt_s), 'r-', label='fit')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Counts')



# multiplied
filtered_spectrum = fit_func_s(lin_w, *popt_s)*fit_func_f(lin_w, *popt_f)

ax.plot(lin_w, filtered_spectrum, 'g-', label='expected filtered spectrum')
ax.legend()

ax.set_title('Edmund optics 740 nm bandpass filter (67-841)')

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.2, 0.6, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

# ax2.plot(f_wavlength[1700: 2000], f_vals[1700: 2000],'k.',label='data')
ax2.plot(lin_w, fit_func_f(lin_w, *popt_f), 'r-', label='fit')
ax2.set_xlim([720, 760])
ax2.set_ylim([-0.033, 1.03])
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Transmission %')
ax2.set_title('Filter transmission')

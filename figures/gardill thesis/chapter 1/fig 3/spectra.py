# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:40:03 2023

@author: kolkowitz
"""


import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
from utils.kplotlib import Size
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
       """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
   3     The Savitzky-Golay filter removes high frequency noise from data.
   4     It has the advantage of preserving the original shape and
   5     features of the signal better than other types of filtering
   6     approaches, such as moving averages techniques.
   7     Parameters
   8     ----------
   9     y : array_like, shape (N,)
  10         the values of the time history of the signal.
  11     window_size : int
  12         the length of the window. Must be an odd integer number.
  13     order : int
  14         the order of the polynomial used in the filtering.
  15         Must be less then `window_size` - 1.
  16     deriv: int
  17         the order of the derivative to compute (default = 0 means only smoothing)
  18     Returns
  19     -------
  20     ys : ndarray, shape (N)
  21         the smoothed signal (or it's n-th derivative).
  22     Notes
  23     -----
  24     The Savitzky-Golay is a type of low-pass filter, particularly
  25     suited for smoothing noisy data. The main idea behind this
  26     approach is to make for each point a least-square fit with a
  27     polynomial of high order over a odd-sized window centered at
  28     the point.
  29     Examples
  30     --------
  31     t = np.linspace(-4, 4, 500)
  32     y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
  33     ysg = savitzky_golay(y, window_size=31, order=4)
  34     import matplotlib.pyplot as plt
  35     plt.plot(t, y, label='Noisy signal')
  36     plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
  37     plt.plot(t, ysg, 'r', label='Filtered signal')
  38     plt.legend()
  39     plt.show()
  40     References
  41     ----------
  42     .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
          Data by Simplified Least Squares Procedures. Analytical
          Chemistry, 1964, 36 (8), pp 1627-1639.
       .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
          W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
          Cambridge University Press ISBN-13: 9780521880688
       """
       import numpy as np
       from math import factorial
       
       try:
           window_size = np.abs(np.int(window_size))
           order = np.abs(np.int(order))
       except ValueError:
           raise ValueError("window_size and order have to be of type int")
       if window_size % 2 != 1 or window_size < 1:
           raise TypeError("window_size size must be a positive odd number")
       if window_size < order + 2:
           raise TypeError("window_size is too small for the polynomials order")
       order_range = range(order+1)
       half_window = (window_size -1) // 2
       # precompute coefficients
       b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
       m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
       # pad the signal at the extremes with
       # values taken from the signal itself
       firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
       lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
       y = np.concatenate((firstvals, y, lastvals))
       return np.convolve( m[::-1], y, mode='valid')
  
def plot(file):
    kpl.init_kplotlib()
    
    file_name = file + '.txt'
    with open(file_name) as f:
        data = json.load(f)
        
    counts = numpy.array(data['counts'])
    wavelengths = numpy.array(data['wavelengths'])
    
    min_counts = min(counts)
    sub_counts = counts - min_counts
    
    yhat = savitzky_golay(sub_counts, 41, 1)
    
    # Plot setup
    fig, ax = plt.subplots(1, 1,  figsize=(7, 4))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel("Counts")
    # ax.set_title(title)

    # Plotting
    # marker_size = 
    ax.plot(wavelengths-10, sub_counts, 'r.', color=KplColors.RED)
    # kpl.plot_points(ax,  wavelengths, sub_counts, color=KplColors.RED, size=Size.TINY)
    # kpl.plot_line(ax,  wavelengths, sub_counts, color=KplColors.RED)
    # kpl.plot_line(ax,  wavelengths, yhat, color=KplColors.RED)
    ax.set_xlim([609, 900])
    
    
    # ax.legend()
    

# file = '2020_07_16-hopper_green_illumination'
file ='2023_05_01-18_13_03-2023_04_30_nv_spectra_hoppercsv'
plot(file)
# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import numpy
from numpy import pi
import matplotlib.pyplot as plt


# %% Constants


boltzmann = 1.380649e-23  # J / K
# boltzmann = 2.0836612e10  # Hz / K
planck = 6.626e-34  # J s
debye_freq = 38.76e12  # Hz
unit_cell_vol = (0.357e-9)**3  # m^3
speed_of_sound = 12e3  # m / s


# %% Functions


def density_of_states(freq):
    d_norm = unit_cell_vol * (freq**2) / (2 * (pi**2) * (speed_of_sound**3))
    if type(freq) is numpy.ndarray:
        freq_copy = numpy.copy(freq)
        freq_copy *= (freq < debye_freq)  # Mask freqs > debye_freq to 0
    else:
        if freq > debye_freq:
            freq_copy = 0
    return d_norm * ((freq_copy / debye_freq)**2)


def bose_einstein_dist(freq, temp=300):
    return 1 / (numpy.exp((planck * freq) / (boltzmann * temp)) - 1)



# %% Main


def main(freq_range):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    freqs = numpy.linspace(freq_range[0], freq_range[1], 1000)

    # densities_of_states = density_of_states(freqs)
    # fig, ax = plt.subplots(figsize=(8.5, 8.5))
    # ax.plot(freqs, densities_of_states)

    dist = bose_einstein_dist(freqs)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    # ax.plot(freqs, (dist + 1)*dist)
    ax.plot(freqs, dist)


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here
    freq_range = [1, 3e9]

    # Run the script
    main(freq_range)

# -*- coding: utf-8 -*-
"""
Template for major routines.

Created on Sun Jun 16 11:38:17 2019

@author: mccambria
"""


# %% Imports


# %% Functions


# %% Figure functions


"""For a major routine we'll typically have a count rate as a function of
some variable (maybe relaxation time or microwave frequency). We'll also
typically record two count rates for each data point: a signal (when the
microwaves are on, for example) and a reference (when the microwaves are off,
following the same example). To present this data, we should be consistent and
use the following pattern. For an example, see the figures generated for rabi.

Figure 1: Raw data line plots

    Axis 1: The average signal and reference count rates on the same axis

    Axis 2: The normalized (average signal / average reference) signal

Figure 2 (1 axis): Fit with the normalized signal as a scatter plot and the
fit as a smooth line through the data - text box displaying the fit function
and optimized fit parameters
"""


def create_raw_figure():
    """This figu"""


# %% Main


def main():
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    pass


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here


    # Run the script
    main()


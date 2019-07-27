# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:59:05 2019

@author: matth
"""

import numpy
import matplotlib.pyplot as plt

def gaussian(freq, constrast, sigma, center):
    return constrast * numpy.exp(-((freq-center)**2) / (2 * (sigma**2)))

if __name__ == '__main__':

    freqs = numpy.linspace(2.84, 2.91, 1000)

    fig, ax = plt.subplots()

    deviation = 0.0005

    gaussian_low = gaussian(freqs, 0.2, 0.005, 2.87-deviation)
    gaussian_high = gaussian(freqs, 0.2, 0.005, 2.87+deviation)
    gaussian_sum = gaussian_low + gaussian_high
    print(numpy.std(gaussian_low))
    print(numpy.std(gaussian_high))
    print(numpy.std(gaussian_sum))
    ax.plot(gaussian_low)
    ax.plot(gaussian_high)
    ax.plot(gaussian_sum)

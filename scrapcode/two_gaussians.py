# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:11:40 2019

@author: matth
"""


import matplotlib.pyplot as plt
import numpy


def gaussian(freq, constrast, sigma, center):
    return 1 - (constrast * numpy.exp(-((freq-center)**2) / (2 * (sigma**2))))


if __name__ == '__main__':
    smooth_freqs = numpy.linspace(2.810, 2.900, 100)
    plt.plot(smooth_freqs, gaussian(smooth_freqs, 0.162, 0.0107, 2.8519))
    plt.plot(smooth_freqs, gaussian(smooth_freqs, 0.147, 0.0082, 2.8690))

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:59:05 2019

@author: matth
"""

import numpy
import matplotlib.pyplot as plt
import scipy.signal as signal

def gaussian(freq, constrast, sigma, center):
    return constrast * numpy.exp(-((freq-center)**2) / (2 * (sigma**2)))

if __name__ == '__main__':

    freqs = numpy.linspace(2.85, 2.89, 1000)

    max_dev = 0.0010
    rel_stds = []
    for max_dev in numpy.linspace(0, 0.01, 101):
        gaussians = []
        for dev in numpy.linspace(-max_dev, +max_dev, 51):
            gaussians.append(gaussian(freqs, 0.2, 0.005, 2.87+dev))
        gaussian_sum = numpy.sum(gaussians, axis=0)
        norm_gaussian_sum = gaussian_sum / numpy.sum(gaussian_sum)
        norm_gaussian = gaussians[25] / numpy.sum(gaussians[25])
        # funcs = [gaussians[0], gaussian_sum]
        # for func in funcs:
        #     peaks = signal.find_peaks(func)[0]
        #     print(signal.peak_widths(func, peaks, rel_height=0.75)[0:2])
        dist_1 = []
        for ind in range(10000):
            dist_1.append(numpy.random.choice(len(norm_gaussian), p=norm_gaussian))
        dist_samp = []
        samp = 100
        for ind in range(10000//samp):
            ind_samp = samp*ind
            dist_samp.append(numpy.sum(dist_1[ind_samp:ind_samp+samp]))
        dist_samp = numpy.array(dist_samp)
        # fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
        # axes[0].plot(dist_1)
        # axes[1].plot(dist_samp/samp)
        rel_std_1 = numpy.std(dist_1/numpy.average(dist_1))
        rel_std_samp = numpy.std(dist_samp/numpy.average(dist_samp))
        # print(rel_std_1)
        # print(rel_std_samp)
        # print(rel_std_samp/rel_std_1)
        rel_stds.append(rel_std_samp/rel_std_1)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(rel_stds)

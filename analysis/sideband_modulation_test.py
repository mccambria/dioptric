# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 12:30:47 2022

@author: gardill
"""

import os
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit
from scipy.stats import sem

import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt

pi = numpy.pi


def cos(t, f, amp):
    return amp * numpy.cos(t * 2 * pi * f)


num_steps = 20000
taus, tau_step = numpy.linspace(0, 2, num_steps, retstep=True)
# print(tau_step)

conv = lambda t, f1, f2, a1, a2, offset: cos(t, f1, a1) * (offset + cos(t, f2, a2))

params = [50, 0.5, 1, 1, 0.5]

fig, axes = plt.subplots(1, 2)
ax = axes[0]
ax.plot(taus, conv(taus, *params))


transform = numpy.fft.rfft(conv(taus, *params))
freqs = numpy.fft.rfftfreq(num_steps, d=tau_step)
# print(freqs[0], freqs[1])
transform_mag = numpy.absolute(transform)
ax = axes[1]
ax.plot(freqs, transform_mag)

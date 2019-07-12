# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:48:33 2019

@author: mccambria
"""

import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt

data = tool_belt.get_raw_data('ramsey', '2019-07-11_13-51-17_johnson1', 'branch_ramsey2')

norm_avg_sig = data['norm_avg_sig']
num_steps = data['num_steps']
precession_time_range = data['precession_time_range']

# Fake 
min_tau = precession_time_range[0]
max_tau = precession_time_range[1]
max_tau *= 2
num_steps = 101

#min_tau = precession_time_range[0]
#max_tau = precession_time_range[1]
time_step = (max_tau - min_tau) / (num_steps - 1)
time_step /= 1000  # to us

transform = numpy.fft.rfft(norm_avg_sig)
window = max_tau - min_tau
freqs = numpy.fft.rfftfreq(num_steps, d=time_step)
print(freqs)

transform_mag = numpy.absolute(transform)

fig, ax= plt.subplots(1, 1, figsize=(10, 8))
fig.set_tight_layout(True)
ax.plot(freqs[1:], transform_mag[1:])  # [1:] excludes frequency 0 (DC component)
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('FFT magnitude')
ax.set_title('Ramsey FFT')

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:00:05 2021

Test what the photodiode reads for different settings on the red AO settings.

@author: agardil
"""

import labrad
import numpy
import matplotlib.pyplot as plt
import time
import utils.tool_belt as tool_belt

power_list = numpy.linspace(0.8, 0.9, 301)
cxn = labrad.connect()

measured_power = []
measured_powers_ste = []
num_reps = 20

for p in power_list:
    cxn.pulse_streamer.constant([],p,0);
    optical_power_list = [];
    for i in range(num_reps):
            optical_power_list.append(cxn.photodiode.read_optical_power())
            time.sleep(0.001)
    optical_power = numpy.average(optical_power_list)
    measured_power.append(optical_power)
    optical_power_ste = numpy.std(optical_power_list) / numpy.sqrt(num_reps)
    measured_powers_ste.append(optical_power_ste)
    
converted_power = 6.7*numpy.array(measured_power)+0.78
converted_power_ste = 6.7*numpy.array(measured_powers_ste)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.errorbar(power_list, converted_power,
                            yerr = converted_power_ste)
ax.set_xlabel('AO power (V)')
ax.set_ylabel('measured optical power (mW)')

timestamp = tool_belt.get_time_stamp()

rawData = {'timestamp': timestamp,
           'power_list': power_list.tolist(),
           'power_list-units': 'V',
           'measured_power': measured_power,
           'measured_power-units': 'V',
           'measured_powers_ste': measured_powers_ste,
           'measured_powers_ste-units': measured_powers_ste,
           'converted_power': converted_power.tolist(),
           'converted_power-units': 'mW',
           'converted_power_ste': converted_power_ste.tolist(),
           'converted_power_ste-units': 'mW',           
           'note': 'measured on 638 nm AO',
           
           }


filePath = tool_belt.get_file_path(__file__, timestamp)
tool_belt.save_raw_data(rawData, filePath)

tool_belt.save_figure(fig, filePath)

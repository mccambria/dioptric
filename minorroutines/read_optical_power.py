# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:05:38 2020

NOT FINISHED 3/23/2020

Choose a laser, set the laser power, and read off the voltage from the 
photodiode

Then the file will convert the voltage to an approximate optical power at the 
the objective.

@author: agardill
"""

import labrad
import numpy
import utils.tool_belt as tool_belt
import time
import matplotlib.pyplot as plt

# %%

def set_laser_power(color_ind, set_515_power = None, set_589_AI = None,
                    set_638_AI = None):
    
    with labrad.connect() as cxn:
        
        if color_ind == 532:
            cxn.pulse_streamer.constant([3], 0.0, 0.0)
            
            return set_515_power
        
        elif color_ind == 589:
            tool_belt.aom_ao_589_pwr_err(set_589_AI)
            cxn.pulse_streamer.constant([], 0.0, set_589_AI)
            
            return set_589_AI
        
        elif color_ind == 638:
            tool_belt.ao_638_pwr_err(set_638_AI)
            cxn.pulse_streamer.constant([], set_638_AI, 0.0)
        
            return set_638_AI
        
# %%
            
def main(color_ind, totel_measure_time, set_515_power = None, set_589_AI = None, 
                    set_589_ND = None, set_638_AI = None, plot= False):
           
    time_step = 0.5 # s
    num_steps = int(totel_measure_time / time_step)
    
    laser_power_indicator = set_laser_power(color_ind, set_515_power, set_589_AI, 
                    set_638_AI)
    time.sleep(1)
    optical_power_list = []
    
    with labrad.connect() as cxn:
        
        if set_589_ND:
            cxn.filter_slider_ell9k.set_filter(set_589_ND)
            
        for t in range(num_steps):
            optical_power = cxn.photodiode.read_optical_power()
            optical_power_list.append(optical_power)
            time.sleep(time_step)
        
    time_linspace = numpy.linspace(0,totel_measure_time, num=num_steps)
    
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
        ax.plot(time_linspace, numpy.array(optical_power_list)*1000)
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Voltage (mV)')
    
    avg_voltage = numpy.average(optical_power_list)
    
    return avg_voltage # V
        
    #Save the information...
# %% Run the file


if __name__ == '__main__':
    
    avg_voltage = main('515a', 5, set_515_power =0.615, plot=False)

    print('Avg voltage: {} mV'.format(avg_voltage*1000))


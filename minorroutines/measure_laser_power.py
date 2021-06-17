# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:01:05 2021

@author: kolkowitz
"""

import labrad
import numpy
import time

def measure_pd_power(color_ind, laser_power, nd_filter):
    with labrad.connect() as cxn:
        if color_ind == 589:
            cxn.filter_slider_ell9k.set_filter(nd_filter)
            cxn.pulse_streamer.constant([],0,laser_power)
        elif color_ind == 638:
            cxn.pulse_streamer.constant([7],0,0)
        elif color_ind == 532:
            cxn.pulse_streamer.constant([3],0,0)
        elif color_ind == '515a':
            cxn.pulse_streamer.constant([],laser_power,0)
        
        pd_list = []
        for i in range(10):
            pd_list.append(cxn.photodiode.read_optical_power())
            time.sleep(0.001)
            
        pd=numpy.average(pd_list)
        
    return pd

# %%
if __name__ == '__main__':
    pd = measure_pd_power(589, 0.6, 'nd_1.5') #0.05 - 0.6, every 0.05
    # nd filters:
    # 'nd_0'
    # 'nd_0.5'
    # 'nd_1.0'
    # 'nd_1.5'
    print(str(pd) + ' V')
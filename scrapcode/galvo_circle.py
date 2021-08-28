# -*- coding: utf-8 -*-
"""
Created on August 24th, 2021

@author: mccambria
"""


import labrad
import utils.tool_belt as tool_belt


def galvo_circle(radius, num_steps, period, laser_name):
    
    with labrad.connect() as cxn:
        
        
        cxn.galvo.load_circle_scan_xy(radius, num_steps, period)
        
        seq_args = [0, period, 0, laser_name, None]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        _ = cxn.pulse_streamer.stream_immediate("simple_readout.py", 
                                                -1, seq_args_string)
        
        input("Press enter to stop...")
        
        cxn.galvo.write_xy(0.0, 0.0)
        cxn.pulse_streamer.reset()


if __name__ == "__main__":
    
    radius = 1.0
    num_steps = 200
    period = int(0.025E9)
    laser_name = "cobolt_638"
    
    galvo_circle(radius, num_steps, period, laser_name)

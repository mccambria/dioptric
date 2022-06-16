# -*- coding: utf-8 -*-
"""
The power for the 1064 laser is controlled by a BNC input from the 
Multicomp Pro 710087 power supply. Here's a few utility functions

Created on June 15th, 2022

@author: mccambria
"""


# %% Imports


import labrad


# %% Functions


    


# %% Run the file


if __name__ == '__main__':

    with labrad.connect() as cxn:
        
        power_supply = cxn.power_supply_mp710087
        
        power_supply.output_off()
        
        # power_supply.output_on()
        # power_supply.set_voltage(0.3)  # 5.6 mW nominal, 5.0 before objective
        
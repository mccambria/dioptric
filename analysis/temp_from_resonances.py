# -*- coding: utf-8 -*-
"""
Get the NV temp based on the ZFS, using numbers from: 'Temperature dependent 
energy level shifts of nitrogen-vacancy centers in diamond'

Created on Fri Mar  5 12:42:32 2021

@author: matth
"""


# %% Imports


import numpy
from numpy.linalg import eigvals
from numpy import pi
from scipy.optimize import root_scalar
from numpy import exp
import matplotlib.pyplot as plt


# %% Functions


def zfs_from_temp(temp):
    """
    This is the 5th order polynomial used as a fit in the paper 'Temperature
    dependent energy level shifts of nitrogen-vacancy centers in diamond'
    """
    coeffs = [2.87771, -4.625E-6, 1.067E-7, -9.325E-10, 1.739E-12, -1.838E-15]
    ret_val = 0
    for ind in range(6):
        ret_val += coeffs[ind] * (temp**ind)
    return ret_val





# %% Main


def main(resonances):
    
    zfs = (resonances[1] + resonances[0]) / 2
    
    zfs_diff = lambda temp: zfs_from_temp(temp) - zfs
    
    results = root_scalar(zfs_diff, x0=50, x1=300)
    
    print(results)


# %% Run the file


if __name__ == '__main__':
    
    # Resonances in GHz
    resonances = [2.8568, 2.8901]  # 250 K
    resonances = [2.8568, 2.8901]  # 250 K
    
    main(low_res, high_res)
    # print(zfs_from_temp(200))
    


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


# def main(resonances):
    
#     zfs = (resonances[1] + resonances[0]) / 2

def main(zfs, zfs_err):
    
    zfs_diff = lambda temp: zfs_from_temp(temp) - zfs
    results = root_scalar(zfs_diff, x0=50, x1=300)
    temp_mid = results.root
    
    zfs_low = zfs - zfs_err
    zfs_diff = lambda temp: zfs_from_temp(temp) - zfs_low
    results = root_scalar(zfs_diff, x0=50, x1=300)
    temp_low = results.root
    
    zfs_high = zfs + zfs_err
    zfs_diff = lambda temp: zfs_from_temp(temp) - zfs_high
    results = root_scalar(zfs_diff, x0=50, x1=300)
    temp_high = results.root
    
    print('T: [{}, {}, {}]'.format(temp_low, temp_mid, temp_high))


# %% Run the file


if __name__ == '__main__':
    
    # Resonances in GHz
    # resonances = [2.8568, 2.8901]  # 250 K
    # resonances = [2.8587, 2.8926]  # 200 K
    resonances = [2.8576, 2.8914]  # 225 K
    
    # 225 K
    # zfs = 2.8745645361129957
    # zfs_err = 0.00020017187345691895
    
    # 300 K
    # zfs = 2.871011389583322
    # zfs_err = 0.0008097208662379307
    
    # 175 K
    # zfs = 2.8764
    # zfs_err = 6.368357262294881e-05  
    
    # 50 K fake
    zfs = 2.8776
    zfs_err = 0.0001
    
    

    # main(resonances)
    main(zfs, zfs_err)
    # print(zfs_from_temp(175))
    # x_vals = numpy.linspace(0, 300, 300)
    # y_vals = zfs_from_temp(x_vals)
    # plt.plot(x_vals, y_vals)
    


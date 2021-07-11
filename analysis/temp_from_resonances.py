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
from majorroutines.pulsed_resonance import return_res_with_error
import utils.tool_belt as tool_belt


# %% Functions


def process_res_list():
    
    paths = [
# 275
# 200
# 262.5
# 225
# 212.5
# 237.5
# 287.5
# 300
# 250
# 150
# 85
# 175
# 125
# 295
# 350
# 400
        ]
    resonances = [
# 275
# 200
# 262.5
# 225
# 212.5
# 237.5
# 287.5
# 300
# 250
# 150
# 85
# 175
# 125
# 295
# 350
# 400
]
    
    for ind in range(len(paths)):
        path_pair = paths[ind]
        res_pair = resonances[ind]
        main_files(path_pair, res_pair)


def sub_room_zfs_from_temp(temp):
    coeffs = [2.87771, -4.625E-6, 1.067E-7, -9.325E-10, 1.739E-12, -1.838E-15]
    ret_val = 0
    for ind in range(6):
        ret_val += coeffs[ind] * (temp**ind)
    return ret_val


def super_room_zfs_from_temp(temp):
    coeffs = [2.8697, 9.7E-5, -3.7E-7, 1.7E-10]
    coeff_errs = [0.0009, 0.6E-5, 0.1E-7, 0.1E-10]
    ret_val = 0
    for ind in range(4):
        ret_val += coeffs[ind] * (temp**ind)
    return ret_val


def zfs_from_temp(temp):
    """
    This is a combination of 2 results. For temp < 300 K, we pull the 
    5th order polynomial from 'Temperature dependent energy level shifts 
    of nitrogen-vacancy centers in diamond.' Then we stitch that to
    'Measurement and Control of Single Nitrogen-Vacancy Center Spins above 
    600 K' above 300 K
    """
    # Branch depending on if temp is single- or multi-valued
    if type(temp) in [list, numpy.ndarray]:
        ret_vals = []
        for val in temp:
            if val < 300:
                zfs = sub_room_zfs_from_temp(val)
            else:
                zfs = super_room_zfs_from_temp(val)
            ret_vals.append(zfs)
        ret_vals = numpy.array(ret_vals)
        return ret_vals
    else:
        if temp < 300:
            return sub_room_zfs_from_temp(temp)
        else:
            return super_room_zfs_from_temp(temp)



# %% Main


def main_files(paths, files):
    
    resonances = []
    res_errs = []
    
    for ind in range(2):
        path = paths[ind]
        file = files[ind]
        data = tool_belt.get_raw_data(path, file)
        res, res_err = return_res_with_error(data)
        resonances.append(res)
        res_errs.append(res_err)
    
    main_res(resonances, res_errs)


def main_res(resonances, res_errs):
    
    zfs = (resonances[0] + resonances[1]) / 2
    zfs_err = numpy.sqrt(res_errs[0]**2 + res_errs[1]**2) / 2
    
    main(zfs, zfs_err)


def main(zfs, zfs_err):
    
    zfs_diff = lambda temp: zfs_from_temp(temp) - zfs
    results = root_scalar(zfs_diff, x0=50, x1=500)
    temp_mid = results.root
    
    zfs_lower = zfs - zfs_err
    zfs_diff = lambda temp: zfs_from_temp(temp) - zfs_lower
    results = root_scalar(zfs_diff, x0=50, x1=500)
    temp_higher = results.root
    
    zfs_higher = zfs + zfs_err
    zfs_diff = lambda temp: zfs_from_temp(temp) - zfs_higher
    results = root_scalar(zfs_diff, x0=50, x1=500)
    temp_lower = results.root
    
    print('T: [{}, {}, {}]'.format(temp_lower, temp_mid, temp_higher))


# %% Run the file


if __name__ == '__main__':
    
    path = 'pc_rabi/branch_laser-consolidation/pulsed_resonance/2021_07'
    file_low = '2021_07_09-18_04_22-hopper-nv1_2021_03_16'
    file_high = '2021_07_09-18_07_34-hopper-nv1_2021_03_16'
    paths = [path, path]
    files = [file_low, file_high]

    main_files(paths, files)
#    
#    print(zfs_from_temp(280))
    
    # temps = numpy.linspace(5,500,1000)
    # # plt.plot(temps, zfs_from_temp(temps))
    # fig, ax = plt.subplots()
    # ax.plot(temps, sub_room_zfs_from_temp(temps), label='sub')
    # ax.plot(temps, super_room_zfs_from_temp(temps), label='super')
    # ax.legend()
    


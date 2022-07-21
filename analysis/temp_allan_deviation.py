# -*- coding: utf-8 -*-
"""
Analysis of four-point ESR measurements

Created on July 21st, 2022

@author: mccambria
"""

# region Imports

import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import utils.kplotlib as kpl
import majorroutines.four_point_esr as four_point_esr
import analysis.temp_from_resonances as temp_from_resonances

# endregion

# region Constants

num_circle_samples = 1000

phi_linspace = np.linspace(0, 2 * pi, num_circle_samples, endpoint=False)
cos_phi_linspace = np.cos(phi_linspace)
sin_phi_linspace = np.sin(phi_linspace)

# endregion

# region Functions


def get_temps_from_files(files):
    
    temps = []
    temp_errs = []
    
    for f_pair in files:
        
        low_file, high_file = f_pair
        low_res, low_error = four_point_esr.calc_resonance_from_file(low_file)
        high_res, high_error = four_point_esr.calc_resonance_from_file(high_file)
        
        zfs = (low_res + high_res) / 2
        zfs_err = np.sqrt(low_error ** 2 + high_error ** 2) / 2
        temp, temp_err = temp_from_resonances.main(zfs, zfs_err)
        
        temps.append(temp)
        temp_errs.append(temp_err)
        
    return temps, temp_errs
        

# endregion


# region Main functions


def temp_vs_time(sig_files, ref_files):
    
    fig, ax = plt.subplots()
    
    x_vals = range(len(sig_files))
    sig_vals, sig_errs = get_temps_from_files(sig_files)
    ax.errorbar(x_vals, sig_vals, yerrs=sig_errs)
    ref_vals, ref_errs = get_temps_from_files(ref_files)
    ax.errorbar(x_vals, ref_vals, yerrs=ref_errs)


# endregion

# region Run the file

if __name__ == "__main__":
    
    kpl.init_kplotlib()
    
    file = "2022_07_19-18_14_50-hopper-search"
    data = tool_belt.get_raw_data(file)
    sig_files = data["sig_files"]
    ref_files = data["ref_files"]
    
    temp_vs_time(file)

# endregion
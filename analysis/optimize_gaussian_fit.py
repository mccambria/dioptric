# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:46:22 2019

Fit gaussian to the optimize data. Helpful to find the standard deviation of
the stop size

The bounds of the curve_fit are currently commented out

@author: Aedan
"""
import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

# %%

def do_plot_data(fig, ax, title, voltages, k_counts_per_sec, 
                 optimizationFailed, optiParams):
    ax.plot(voltages, k_counts_per_sec)
    ax.set_title(title)
    ax.set_xlabel('Volts (V)')
    ax.set_ylabel('Count rate (kcps)')

    # Plot the fit
    if not optimizationFailed:
        first = voltages[0]
        last = voltages[len(voltages)-1]
        linspaceVoltages = numpy.linspace(first, last, num=1000)
        gaussianFit = tool_belt.gaussian(linspaceVoltages, *optiParams)
        ax.plot(linspaceVoltages, gaussianFit)

        # Add info to the axes
        # a: coefficient that defines the peak height
        # mu: mean, defines the center of the Gaussian
        # sigma: standard deviation, defines the width of the Gaussian
        # offset: constant y value to account for background
        text = 'a={:.3f}\n $\mu$={:.3f}\n ' \
            '$\sigma$={:.4f}'.format(*optiParams)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

    fig.canvas.draw()
    fig.canvas.flush_events()

# %%
    
def main(file_name):
    
    folder_name = 'G:/Team Drives/Kolkowitz Lab Group/nvdata/optimize'
    
    # collect the neccessary data from the json file 
    with open('{}/{}'.format(folder_name, file_name)) as file:
        data = json.load(file)
        x_voltages = data['x_voltages']
        y_voltages = data['y_voltages']
        z_voltages = data['z_voltages']
        
        x_counts = numpy.array(data['x_counts'])
        y_counts = numpy.array(data['y_counts'])
        z_counts = numpy.array(data['z_counts'])
        
        x_center, y_center, z_center = data['coords']
        xy_range = data['xy_range']
        z_range = data['z_range']
        
        readout = data['readout']
        
    # convert the readout to seconds   
    readout_sec = readout / 10**9  
    
    # List to store the optimized centers
    opti_centers = [None, None, None]
    
    # create the plot to be used
    fig, axes_pack = plt.subplots(1, 3, figsize=(17, 8.5))
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # X Fit
    k_counts_per_sec = (x_counts / 1000) / readout_sec
    init_fit = ((100. / readout) * 10**6, x_center, (xy_range / 3))
#    fit_param_bounds = ([(0 / readout_sec) / 10**3, -5., 0.], 
#                        [(1000 / readout_sec) / 10**3, 5., 0.1])
    fit_param_bounds = ([(0 / readout_sec) * 10**3, -2, -0.01], 
                        [(1000 / readout_sec) * 10**3, 2, 0.01])
    try:
        optiParams, cov_arr = curve_fit(tool_belt.gaussian, x_voltages,
                                        k_counts_per_sec, p0=init_fit, 
                                        bounds = fit_param_bounds
                                        )      
        optimizationFailed = False
    except Exception:
        optimizationFailed = True

    if not optimizationFailed:
        opti_centers[0] = optiParams[1]
        
    # X Plot
    do_plot_data(fig, axes_pack[0], 'X Axis', x_voltages, k_counts_per_sec, 
                 optimizationFailed, optiParams)
    
    # Y Fit
    k_counts_per_sec = (y_counts / 1000) / readout_sec
    init_fit = ((100. / readout) * 10**6, y_center, (xy_range / 3))
    fit_param_bounds = ([(0 / readout_sec) * 10**3, -2, -0.01], 
                        [(1000 / readout_sec) * 10**3, 2, 0.01])
    
    try:
        optiParams, cov_arr = curve_fit(tool_belt.gaussian, y_voltages,
                                        k_counts_per_sec, p0=init_fit,
                                        bounds = fit_param_bounds
                                        )
                
        optimizationFailed = False
    except Exception:
        optimizationFailed = True

    if not optimizationFailed:
        opti_centers[1] = optiParams[1]
    
    # Y Plot
    do_plot_data(fig, axes_pack[1], 'Y Axis', y_voltages, k_counts_per_sec, 
                     optimizationFailed, optiParams)
    

    # Z Fit
    k_counts_per_sec = (z_counts / 1000) / readout_sec
    init_fit = ((100. / readout) * 10**6, z_center, (z_range / 2))
    fit_param_bounds = ([(0 / readout_sec) * 10**3, 40, -10], 
                        [(1000 / readout_sec) * 10**3, 60, 10])
    try:
        optiParams, cov_arr = curve_fit(tool_belt.gaussian, z_voltages,
                                        k_counts_per_sec, p0=init_fit,
                                        bounds = fit_param_bounds
                                        )
        

        
        optimizationFailed = False
    except Exception:
        optimizationFailed = True

    if not optimizationFailed:
        opti_centers[2] = optiParams[1]
    
    # Z Plot
    do_plot_data(fig, axes_pack[2], 'Z Axis', z_voltages, k_counts_per_sec, 
                 optimizationFailed, optiParams)
    
    
    # %%
    
if __name__ == '__main__':
    main('2019-06-04_13-28-18_ayrton12.txt')
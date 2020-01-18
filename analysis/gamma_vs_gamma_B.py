# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 13:47:21 2020

@author: matth
"""


import analysis.extract_hamiltonian as eh
import analysis.rotation_dq_sq_ratio_v2 as rot
import numpy
from numpy import pi
import matplotlib.pyplot as plt
from scipy import integrate


def empirical_fit(splitting, coeff, gamma_inf):
    return (coeff * splitting**-2) + gamma_inf


def main(name, popt, splittings, gammas, gamma_errors, empirical_fit_params):
    
    smooth_mag_Bs = numpy.linspace(0, 1.0, 100)
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.set_tight_layout(True)
    ax.set_title('{} Fit Comparison'.format(name))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Splitting (GHz)')
    ax.set_ylabel(r'$\gamma$ (kHz)')
    
    dq_factors = []
    
    for mag_B in smooth_mag_Bs:
    
        low_to_high_integral, low_to_high_err = integrate.dblquad(
                                            rot.calc_B_factor_surface,
                                            0, 2*pi, lambda x: 0, lambda x: pi,
                                            args=(mag_B, popt, 2))
        dq_factors.append(low_to_high_integral)
        
    # Scale from GHz to MHz
    smooth_splittings = 1000 * eh.calc_splitting(smooth_mag_Bs, *popt)
    
    ax.errorbar(splittings, gammas, yerr=2*gamma_errors,
            label='Data', fmt='o')
    empirical_fit_vals = empirical_fit(smooth_splittings, *empirical_fit_params)
    ax.plot(smooth_splittings, empirical_fit_vals, 
            label='Empirical fit')
    dq_factors = numpy.array(dq_factors)
    norm = (empirical_fit_vals[0] / dq_factors[0]) 
    ax.plot(smooth_splittings, dq_factors * norm,
            label='Squared matrix elements')
    
    ax.set_xlim(min(splittings) * 0.8, max(splittings) * 1.5)
    
    ax.legend()
        
    
if __name__ == '__main__':
    
    ##############################

    # name = 'NV1'  # NV1 nv1_2019_05_10
    # popt = [0.6474219686681678, -0.005159086817872651, 0.009754609612326834, 0.0, 0.0]
    # empirical_fit_params = [33.9e3, 0.74]
    # meas_splittings = numpy.array([19.5, 19.8, 27.7, 28.9, 32.7, 51.8,
    #                     97.8, 116, 268, 350, 561.7, 1016.8])
    # meas_gammas = numpy.array([58.3, 117, 64.5, 56.4, 42.6, 13.1, 3.91,
    #                                 4.67, 1.98, 1.57, 0.70, 0.41])
    # error_gammas = numpy.array([1.4, 4, 1.4, 1.3, 0.9, 0.2, 0.1,
    #                             0.11, 0.1, 0.12, 0.05, 0.05])

    ##############################

    # name = 'NV2'  # NV2 nv2_2019_04_30
    # popt = [0.5589906727480959, -0.0042160071239980765, 0.007650711647574739, 0.0, 0.0]
    # empirical_fit_params = [15.7e3, 0.20]
    # meas_splittings = numpy.array([15.3, 29.1, 44.8, 56.2, 56.9, 101.6])
    # meas_gammas = numpy.array([124, 20.9, 6.4, 3.64, 3.77, 1.33])
    # error_gammas = numpy.array([3, 0.3, 0.12, 0.08, 0.09, 0.05])

    ##############################

    # name = 'NV2-take2'  # NV2 nv2_2019_04_30
    # popt = [1.1159475749589527, -0.0030477768450143414, 0.007614582138594034, 0.0, 0.0]  # take2
    # empirical_fit_params = [15.7e3, 0.20]
    # meas_splittings = numpy.array([15.3, 29.2, 45.5, 85.2, 280.4, 697.4])
    # meas_gammas = numpy.array([124, 31.1, 8.47, 2.62, 0.443, 0.81])
    # error_gammas = numpy.array([3, 0.4, 0.11, 0.05, 0.014, 0.06])
    
    ##############################

    name = 'NV3'  # NV3 NV16_2019_07_25
    popt = [0.8932610776776148, -0.0069876377243371516, 0.001394689887147546, 0.0, 0.0]
    empirical_fit_params = [59e3, 4.7]
    meas_splittings = numpy.array([17.1, 28.6, 53.0, 81.2,
                                    128.0, 283.1, 495.8, 746])
    meas_gammas = numpy.array([108, 90, 26.2, 17.5, 11.3, 5.6, 3.7, 2.8])
    error_gammas = numpy.array([10, 5, 0.9, 0.6, 0.4, 0.3, 0.4, 0.3])

    ##############################

    # name = 'NV4'  # NV4 NV0_2019_06_06
    # popt = [0.1630358986041586, -0.00335511843080785, 0.011701742088160093, 0.0, 0.0]
    # empirical_fit_params = [29.4e3, 0.56]
    # meas_splittings = numpy.array([23.4, 26.2, 36.2, 48.1, 60.5, 92.3, 150.8,
    #                                 329.6, 884.9, 1080.5, 1148.4])
    # meas_gammas = numpy.array([34.5, 29.0, 20.4, 15.8, 9.1, 6.4, 4.08,
    #                             1.23, 0.45, 0.69, 0.35])
    # error_gammas = numpy.array([1.3, 1.1, 0.5, 0.3, 0.3, 0.1,
    #                             0.15, 0.07, 0.03, 0.12, 0.03])

    ##############################

    # name = 'NV5'  # NV5 nv13_2019_06_10
    # popt = [1.2335027581388316, -0.03024777463308297, 0.004077396780253218, 0.0, 0.0]
    # empirical_fit_params = [18.7e3, 3.7]
    # meas_splittings = numpy.array([10.9, 23.1, 29.8, 51.9,
    #                                 72.4, 112.9, 164.1, 256.2])
    # meas_gammas = numpy.array([240, 62, 19.3, 17.7, 16.2, 12.1, 5.6, 2.1])
    # error_gammas = numpy.array([25, 8,  1.1, 1.4, 1.1, 0.9, 0.5, 0.3])
    
    ##############################
    
    main(name, popt, meas_splittings, meas_gammas, error_gammas, empirical_fit_params)

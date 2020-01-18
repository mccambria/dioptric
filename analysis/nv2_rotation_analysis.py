# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 22:27:52 2020

@author: matth
"""


import analysis.extract_hamiltonian as eh
import analysis.rotation_dq_sq_ratio_v2 as rot
from scipy.optimize import minimize_scalar
import numpy
from numpy import pi
import matplotlib.pyplot as plt
from scipy import integrate

names = []
popts = []
splittings = []
gammas = []
# names.append('30deg')
# popts.append([0.6398153129728315, -0.0044880947609542005, 0.0070490732314452695, 0.0, 0.0])
# splittings.append([15.3, 29.1, 44.8, 56.2, 56.9, 101.6])
# gammas.append([124, 20.9, 6.4, 3.64, 3.77, 1.33])
names.append('60deg')
popts.append([1.1162003323335492, -0.0031494625116033634, 0.007006402029975579, 0.0, 0.0])
# popts.append([0.50, -0.0031494625116033634, 0.007006402029975579, 0.0, 0.0])
splittings.append([15.3, 29.2, 45.5, 85.2, 280.4, 697.5])
gammas.append([124, 31.1, 8.5, 2.62, 0.44, 0.81])

smooth_mag_Bs = numpy.linspace(0, 0.5, 100)

fig, ax = plt.subplots(figsize=(8.5, 8.5))
fig.set_tight_layout(True)
# ax.set_title('Generating fit vector: {}'.format(name))
ax.set_xscale('log')
ax.set_yscale('log')

for ind in range(len(names)):
    
    name = names[ind]
    popt = popts[ind]
    
    # if name in ['30deg']:
    #     continue
    
    high_vecs = []
    dq_factors = []

    for mag_B in smooth_mag_Bs:
            
        # vecs = eh.calc_eigenvectors(mag_B, *popt)
        # high_vecs.append(list(vecs[1]))
    
        low_to_high_integral, low_to_high_err = integrate.dblquad(
                                            rot.calc_B_factor_surface,
                                            0, 2*pi, lambda x: 0, lambda x: pi,
                                            args=(mag_B, popt, 2))
        # low_to_high_integral = rot.calc_B_factor_surface(1.0, 0, mag_B, popt, 2)
        dq_factors.append(low_to_high_integral)
        
    # Scale from GHz to MHz
    smooth_splittings = 1000 * eh.calc_splitting(smooth_mag_Bs, *popt)
    
    dq_factors = numpy.array(dq_factors)
    ax.plot(smooth_splittings, dq_factors * (gammas[0][0] / dq_factors[0]), label=names[ind])
    ax.scatter(splittings[ind], gammas[ind], label=names[ind])
        
    # high_vecs = numpy.array(high_vecs)
    # ax.plot(smooth_mag_Bs, numpy.abs(high_vecs[:, 0])**2, label='+1')
    # ax.plot(smooth_mag_Bs, numpy.abs(high_vecs[:, 1])**2, label='0')
    # ax.plot(smooth_mag_Bs, numpy.abs(high_vecs[:, 2])**2, label='-1')
    # ax.legend()
    
    
    
ax.legend()

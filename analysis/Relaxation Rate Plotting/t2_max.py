# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:12:42 2019

This file calculates the T2,max of the various NV rates we've collected.

From putting the NV into a coherent superposition between 0 and +/-1, the
maximum coherence time is set by the relaxation rates:
    
T2,max = 2 / (3*omega + gamma)

The error on this associated maximum T2 is then:
    
delta(T2,max) = (3*omega + gamma)**-2 * Sqrt( (6*del(omega))**2 + (2*del(gamma))**2 )

@author: Aedan
"""
import numpy
import matplotlib.pyplot as plt

nv1_splitting_list = [ 27.7, 28.9, 30.5, 32.7, 51.8, 97.8, 116, 268, 563.6, 1016.8]
nv1_omega_avg_list = numpy.array([ 1.30,  1.000, 1.2, 1.42, 1.85, 1.41, 1.18, 1.04, 1.19, 0.58])
nv1_omega_error_list = numpy.array([ 0.06, 0.016, 0.06, 0.05, 0.08, 0.05, 0.06, 0.04, 0.06, 0.03])*3
nv1_gamma_avg_list = numpy.array([64.5, 56.4, 30.5, 42.6, 13.1, 3.91, 4.67, 1.98, 0.70, 0.41])
nv1_gamma_error_list = numpy.array([1.4, 1.3, 1.6, 0.9, 0.2, 0.1, 0.11, 0.1, 0.05, 0.05])*3

# The data
nv2_splitting_list = [29.1, 44.8, 56.2, 56.9,  101.6, 29.2, 45.5, 85.2, 280.4,697.4]
nv2_omega_avg_list = numpy.array([0.412, 0.356, 0.326, 0.42,  0.312, 0.328, 0.266, 0.285, 0.276, 0.29])
nv2_omega_error_list = numpy.array([0.011, 0.012, 0.008, 0.05,  0.009, 0.013, 0.01, 0.011, 0.011, 0.02])*3
nv2_gamma_avg_list = numpy.array([18.7, 6.43, 3.64, 3.77,  1.33, 31.1, 8.47, 2.62, 0.443, 0.81])
nv2_gamma_error_list = numpy.array([0.3, 0.12, 0.08, 0.09,  0.05, 0.4, 0.11, 0.05, 0.014, 0.06])*3

nv16_splitting_list = [28.6, 53.0, 81.2, 128.0, 283.7, 495.8, 746]
nv16_omega_avg_list = numpy.array([0.53, 0.87, 1.7, 0.60, 0.70, 1.4, 1.03])
nv16_omega_error_list = numpy.array([0.05, 0.09, 0.2, 0.05, 0.07, 0.4, 0.17])*3
nv16_gamma_avg_list = numpy.array([90, 26.2, 17.5, 11.3, 5.6, 3.7, 2.8])
nv16_gamma_error_list = numpy.array([5, 0.9, 0.6, 0.4, 0.3, 0.4, 0.3])*3

# %% NV1

nv1_color = '#87479b'

T2_max_1 = 2 / (3 * nv1_omega_avg_list + nv1_gamma_avg_list) 
T2_max_error_1 = (3*nv1_omega_avg_list + nv1_gamma_avg_list)**-2 * numpy.sqrt( (6*nv1_omega_error_list)**2 + (2*nv1_gamma_error_list)**2 )

T2_max_traditional_1 = 2 / (3 * nv1_omega_avg_list)
T2_max_traditional_error_1 = T2_max_traditional_1 * nv1_omega_error_list / nv1_omega_avg_list

average_traditional_t2_max_1= numpy.empty([1000]) 
average_traditional_t2_max_1[:] = numpy.average(T2_max_traditional_1)

average_traditional_t2_error_1= numpy.empty([1000]) 
average_traditional_t2_error_1[:]= numpy.sqrt(sum(T2_max_traditional_error_1**2)) / len(T2_max_traditional_error_1)

#print(average_traditional_t2_max_1)
#print(average_traditional_t2_error_1)

# %% NV2

nv2_color = '#2e008a' 

T2_max_2 = 2 / (3 * nv2_omega_avg_list + nv2_gamma_avg_list) 
T2_max_error_2 = (3*nv2_omega_avg_list + nv2_gamma_avg_list)**-2 * numpy.sqrt( (6*nv2_omega_error_list)**2 + (2*nv2_gamma_error_list)**2 )

T2_max_traditional_2 = 2 / (3 * nv2_omega_avg_list)
T2_max_traditional_error_2 = T2_max_traditional_2 * nv2_omega_error_list / nv2_omega_avg_list

average_traditional_t2_max_2= numpy.empty([1000]) 
average_traditional_t2_max_2[:] = numpy.average(T2_max_traditional_2)

average_traditional_t2_error_2= numpy.empty([1000]) 
average_traditional_t2_error_2[:]= numpy.sqrt(sum(T2_max_traditional_error_2**2)) / len(T2_max_traditional_error_2)

# %% NV16

nv16_color = '#e14097' 

T2_max_16 = 2 / (3 * nv16_omega_avg_list + nv16_gamma_avg_list) 
T2_max_error_16 = (3*nv16_omega_avg_list + nv16_gamma_avg_list)**-2 * numpy.sqrt( (6*nv16_omega_error_list)**2 + (2*nv16_gamma_error_list)**2 )

T2_max_traditional_16 = 2 / (3 * nv16_omega_avg_list)
T2_max_traditional_error_16 = T2_max_traditional_16 * nv16_omega_error_list / nv16_omega_avg_list

average_traditional_t2_max_16= numpy.empty([1000]) 
average_traditional_t2_max_16[:] = numpy.average(T2_max_traditional_16)

average_traditional_t2_error_16= numpy.empty([1000]) 
average_traditional_t2_error_16[:]= numpy.sqrt(sum(T2_max_traditional_error_16**2)) / len(T2_max_traditional_error_16)



# %%

linspace = numpy.linspace(0, 2000, 1000)


fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.errorbar(nv1_splitting_list, T2_max_1, yerr = T2_max_error_1, 
            label = 'NV1',  color= nv1_color, fmt='s',markersize = 10)
#ax.hlines(average_traditional_t2_max_1, 0, 1000, linewidth=5, colors = 'red')
#ax.fill_between(linspace, average_traditional_t2_max_1 + average_traditional_t2_error_1,
#                        average_traditional_t2_max_1 - average_traditional_t2_error_1,
#                        color = 'red', alpha=0.2)
ax.plot(linspace, average_traditional_t2_max_1,  
            label = 'NV1', linestyle=':', color = nv1_color, linewidth=3)

ax.errorbar(nv2_splitting_list, T2_max_2, yerr = T2_max_error_2, 
            label = 'NV2',  color= nv2_color, fmt='*',markersize = 15)
ax.plot(linspace, average_traditional_t2_max_2,  
            label = 'NV2', linestyle='-.', color = nv2_color, linewidth=3)


ax.errorbar(nv16_splitting_list, T2_max_16, yerr = T2_max_error_16, 
            label = 'NV16',  color= nv16_color, fmt='D',markersize = 10)
ax.plot(linspace, average_traditional_t2_max_16,  
            label = 'NV16', linestyle='--', color = nv16_color, linewidth=3)


ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()

ax.set_xlim([-20,1100])
ax.set_ylim([-0.1,2.3])


plt.xlabel('Splitting (MHz)', fontsize=18)
plt.ylabel(r'$T_{2,max}$ (ms)', fontsize=18)

#ax.legend(fontsize=18)

fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_3d.pdf", bbox_inches='tight')

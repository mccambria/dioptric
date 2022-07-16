# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:15:49 2022

https://stats.stackexchange.com/questions/348765/simultaneously-curve-fitting-for-2-models-with-shared-parameters-in-r

fit the fwhm and height together, sharing parameters e and alpha.

@author: agard
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import j1
from scipy.special import jv

# %% Constants
NA = 1.3
wavelength = 638
fwhm =1.825 # 2* (ln(2))^1/4
scale = 0.99e3
x0=7.0156 #Position of this Airy disk (n2), in dimensionless units
R_guess_nm = 10
R_guess = 2*np.pi*NA* R_guess_nm /wavelength

def bessel_scnd_der(x):
    term_1 = 24*j1(x)**2
    term_2 = 16*j1(x)*(jv(0,x) - jv(2,x))
    term_3 = 4* (0.5* (jv(0,x) - jv(2,x))**2 + j1(x)* (0.5*(jv(3,x) - j1(x)) - j1(x)))
    
    return term_1/x**4 - term_2/x**3 + term_3/x**2


# %% Define the data (obtained from super_resolution_1D_scan_fits.py)
y1 = np.array([0.21638739, 0.23514312, 0.270726 ,  0.27750125, 0.39541004, 0.59534277,
 0.58171584, 0.69582389, 0.86908403, 1.11268992, 1.22341671, 1.36458656,
 2.36018764]) #fwhm list, in dimensionless units

y1_err = np.array([0.01053939, 0.01709594, 0.01140871, 0.0106756,
                   0.01095163, 
                   0.01374391, 0.01250901 ,0.01245438, 0.01452769, 
                   0.02224011 ,0.0237046 , 0.02058593, 0.04687215])
y2 = np.array([0.10725226150792853, 0.06190012343462058, 0.22799568710400495, 
               0.27458371231064505, 0.3914980260623759, 0.41733672866339744, 
               0.5271959432866865, 0.5145959956581315, 0.5775909397584066, 
               0.5921214987564455, 0.5875214037295355, 0.5903589782122276, 
               0.5499300166778723]) #heights list, in norm. NV population
y2_err = np.array([8.34948890712762e-05, 0.00010793648980823788, 
                   0.00013014837012369804, 0.00013073034217673103,
                   9.806252983981712e-05, 7.480836398390166e-05, 
                   8.181131441195924e-05, 5.642847964186914e-05,
                   5.724635703157688e-05, 8.968477049723815e-05, 
                   8.700462838501345e-05, 4.813162265693258e-05, 
                   8.425765285908609e-05])
comboY = np.append(y1, y2)
comboY_err = np.append(y1_err, y2_err)

t = np.array([10.0, 11.0, 7.5, 5.0, 2.5, 1.0, 0.75, 0.5,
              0.25, 0.1, 0.075, 0.05, 0.01]) # in ms
comboX = np.append(t ,t)

# %% The fitting functions
def mod1(data, e, alpha,  A, R): # fitting to width scaling with duration
        # R = 2*np.pi*NA*R_guess/wavelength
        C = bessel_scnd_der(x0) #Calculate a constant to use in fit
        return np.sqrt(4/C * (-e + np.sqrt(e**2+ alpha/data)) + R**2) 

def mod2(data, e, alpha, A, R): # fitting to amplitude scaling with duration
        return  A *  np.exp(-data*e**2 * (np.log(2) / alpha))

# %% Combine the functions
def comboFunc(comboData,  e, alpha,  A, R):
    # single data set passed in, extract separate data
    extract1 = comboData[:len(y1)] # first data
    extract2 = comboData[len(y2):] # second data
    
    result1 = mod1(extract1,  e, alpha, A, R)
    result2 = mod2(extract2,  e, alpha, A, R)

    return np.append(result1, result2)

# %% Fit and plot

# some initial parameter values
initialParameters = np.array([0.002, 0.0000035, 3.64804464, R_guess  ])

# curve fit the combined data to the combined function
fittedParameters, pcov = curve_fit(comboFunc, comboX, comboY, initialParameters,
                                    sigma = comboY_err, #this "weights" the data, based on the uncertainty of each point
                                    absolute_sigma=True,
                                   bounds = ([0,0,0,
                                              0],
                                             # 2*np.pi*NA* 5.0 /wavelength], 
                                              [0.1, np.infty,
                                               np.infty,
                                                # 2*np.pi*NA* 20 /wavelength])
                                               np.infty])
                                   )

# values for display of fitted function
e, alpha, A, R = fittedParameters

y_fit_1 = mod1(t,  e, alpha, A, R) # first data set, first equation
y_fit_2 = mod2(t,  e, alpha,  A, R) # second data set, second equation

# FWHM scaling
fig1, ax = plt.subplots()
y1_nm = np.array(y1) * wavelength/ (2*np.pi*NA)
y1_err_nm = np.array(y1_err) * wavelength/ (2*np.pi*NA)
ax.errorbar(t, y1_nm, yerr = y1_err_nm, fmt = 'Dr') # plot the raw data
y_fit_1_nm = np.array(y_fit_1) * wavelength/ (2*np.pi*NA)
ax.plot(t, y_fit_1_nm, '-r') # plot the equation using the fitted parameters

ax.set_xlabel(r'Depletion pulse duration, $\tau$ (ms)')
ax.set_ylabel('FWHM (nm)')
ax.set_xscale('log')
ax.set_yscale('log')

# amplitude scaling
fig2, ax = plt.subplots()
ax.plot(t, y2, 'Db') # plot the raw data
ax.plot(t, y_fit_2, '-b') # plot the equation using the fitted parameters

ax.set_xlabel(r'Depletion pulse duration, $\tau$ (ms)')
ax.set_ylabel('Normalized peak height')
ax.set_yscale('log')

#print the fitted params
print(fittedParameters)

print('e = {:.7f} +/- {:.7f}'.format(fittedParameters[0], np.sqrt(pcov[0][0])))
print('alpha = {:.7f} +/- {:.7f}'.format(fittedParameters[1], np.sqrt(pcov[1][1])))
print('A = {:.7f} +/- {:.7f}'.format(fittedParameters[2], np.sqrt(pcov[2][2])))
R_val_conv = (fittedParameters[3])*wavelength/(2*np.pi*NA)
R_err_conv = np.sqrt(pcov[3][3])*wavelength/(2*np.pi*NA)
print('R = {:.5f} +/- {:.5f}'.format(R_val_conv, R_err_conv))

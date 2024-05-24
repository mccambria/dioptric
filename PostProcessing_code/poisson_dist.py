# -*- coding: utf-8 -*-
"""
Fitting Poisson Ditribution Post Processing Code
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import factorial
from scipy import stats

def poisson(k, lamb):
    """poisson pdf, parameter lamb is the fit parameter"""
    return (lamb**k/factorial(k)) * np.exp(-lamb)


def negative_log_likelihood(params, data):
    """
    The negative log-Likelihood-Function
    """

    lnl = - np.sum(np.log(poisson(data, params[0])))
    return lnl

def negative_log_likelihood(params, data):
    ''' better alternative using scipy '''
    return -stats.poisson.logpmf(data, params[0]).sum()

# get poisson deviated random numbers
data = []

# minimize the negative log-Likelihood

result = minimize(negative_log_likelihood,  # function to minimize
                  x0=np.ones(1),            # start value
                  args=(data,),             # additional arguments for function
                  method='Powell',          # minimization method, see docs
                  )
# result is a scipy optimize result object, the fit parameters 
# are stored in result.x
print(result)

# plot poisson-distribution with fitted parameter
x_plot = np.arange(0, 15)

plt.plot(
    x_plot,
    stats.poisson.pmf(x_plot, result.x),
    marker='o', linestyle='',
    label='Fit result',
)
plt.legend()
plt.show()
# Example usage
file_path = "/Users/schand/Library/CloudStorage/GoogleDrive-schand@berkeley.edu/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_master/image_sample_diff/2023_11/2023_11_13-19_04_14-johnson-nv0_2023_11_09.npz"  # Change this to the actual file path
fit_poisson(file_path)

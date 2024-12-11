# -*- coding: utf-8 -*-
"""
Created on Fall 2024
@author: saroj chand
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Data
power = np.array(
    [
        110.27047475,
        124.2237543,
        139.66267129,
        156.71662825,
        175.52387778,
        196.23196779,
        218.99820133,
        243.99011125,
        271.38595003,
        301.37519478,
        334.15906787,
        369.95107334,
        408.97754925,
        451.47823631,
        497.70686299,
        547.93174729,
        602.43641545,
        661.52023786,
        725.49908232,
        794.70598492,
        869.4918388,
        950.226101,
        1037.29751762,
        1131.11486747,
        1232.1077246,
        1340.72723978,
        1457.44694119,
        1582.76355465,
        1717.1978435,
        1861.2954684,
    ]
)

readout_fidelity = np.array(
    [
        0.76451044,
        0.77859885,
        0.78702115,
        0.79943652,
        0.81509084,
        0.82235814,
        0.83620542,
        0.84194636,
        0.85407271,
        0.86075761,
        0.87159519,
        0.88218824,
        0.88656868,
        0.89609638,
        0.90140651,
        0.90983053,
        0.91319595,
        0.91668867,
        0.92060538,
        0.92689868,
        0.92801948,
        0.9302162,
        0.93122104,
        0.93136359,
        0.93246886,
        0.93208983,
        0.93213398,
        0.93232209,
        0.93140419,
        0.92989206,
    ]
)

prep_fidelity = np.array(
    [
        0.60713766,
        0.62041635,
        0.65043388,
        0.64347289,
        0.67294158,
        0.6746419,
        0.66978895,
        0.67282926,
        0.70392233,
        0.6729217,
        0.68044498,
        0.68690872,
        0.67836426,
        0.66258777,
        0.65973769,
        0.66068842,
        0.64262605,
        0.62183731,
        0.62296502,
        0.59714595,
        0.5926367,
        0.58422066,
        0.57116595,
        0.56346773,
        0.54781795,
        0.53493157,
        0.53398275,
        0.52878229,
        0.51476742,
        0.50875939,
    ]
)


from scipy.optimize import curve_fit, differential_evolution
import numpy as np
import matplotlib.pyplot as plt


# Logistic function for readout fidelity
def logistic_model(P, F_min, F_max, P_mid, n):
    P = np.maximum(P, 1e-10)
    return F_min + (F_max - F_min) / (1 + (P_mid / P) ** n)


# Modified logistic function for asymmetric behavior
def asymmetric_logistic_model(P, F_min, F_max, P_mid, n1, n2, c):
    P = np.maximum(P, 1e-10)
    return F_min + (F_max - F_min) / (1 + (P_mid / P) ** n1 + c * (P / P_mid) ** n2)


# Adjusted Hill-like function to avoid numerical instability
def hill_decay_model(P, A, K, n, B):
    P = np.maximum(P, 1e-10)  # Prevent zero or negative power
    return (A * P**n) / (K**n + P**n) - B * P


# Gaussian-Hill Hybrid Model
def gaussian_hill_model(P, A, P_peak, sigma, B, K, n):
    P = np.maximum(P, 1e-10)  # Prevent zero or negative power
    gaussian_term = A * np.exp(-(((P - P_peak) / sigma) ** 2))
    hill_term = (B * P**n) / (K**n + P**n)
    return gaussian_term + hill_term


# Polynomial-Hill Model
def polynomial_hill_model(P, a, b, c, K, n):
    P = np.maximum(P, 1e-10)  # Prevent invalid inputs
    polynomial_term = a * P**2 + b * P + c
    hill_term = 1 / (1 + (P / K) ** n)
    return polynomial_term * hill_term


# Weighted fitting
def weighted_loss(params, func, x, y, weights):
    return np.sum(weights * (y - func(x, *params)) ** 2)


# Scale power
power_scaled = power / np.max(power)

# Initial guesses
logistic_p0 = [0.7, 0.95, 0.5, 2]
p0_asymmetric = [0.7, 0.95, 0.5, 2, 2, 0.1]

# Weights (optional)
weights_readout = 1 / readout_fidelity
weights_prep = 1 / prep_fidelity

# Fit logistic model for readout fidelity
params_readout, _ = curve_fit(
    asymmetric_logistic_model,
    power_scaled,
    readout_fidelity,
    p0=p0_asymmetric,
    maxfev=100000,
)


# Plot results
plt.figure(figsize=(6, 5))
# Readout fidelity plot
plt.scatter(power_scaled, readout_fidelity, label="Data", color="blue")
plt.plot(
    power_scaled,
    asymmetric_logistic_model(power_scaled, *params_readout),
    label="Fit",
    color="red",
)
plt.title("Readout Fidelity (Asymmetric Logistic Model)")
plt.xlabel("Scaled Power")
plt.ylabel("Fidelity")
plt.legend()
plt.show(block=False)

# Initial guesses
hill_p0 = [0.7, 0.5, 2, 0.0001]
p0_gaussian_hill = [0.7, 0.5, 0.1, 0.3, 0.5, 2]
# Fit Hill model for preparation fidelity
params_prep, _ = curve_fit(
    gaussian_hill_model,
    power_scaled,
    prep_fidelity,
    p0=p0_gaussian_hill,
    maxfev=100000,
)
# Preparation fidelity plot
plt.figure(figsize=(6, 5))
plt.scatter(power_scaled, prep_fidelity, label="Data", color="blue")
plt.plot(
    power_scaled,
    gaussian_hill_model(power_scaled, *params_prep),
    label="Fit",
    color="red",
)
plt.title("Preparation Fidelity (Gaussian-Hill Model)")
plt.xlabel("Scaled Power")
plt.ylabel("Fidelity")
plt.legend()
plt.show(block=False)


# Initial guesses
p0_polynomial_hill = [0.1, 0.1, 0.7, 0.5, 2]

# Fit Polynomial-Hill model to preparation fidelity
params_polynomial_hill, _ = curve_fit(
    polynomial_hill_model,
    power_scaled,
    prep_fidelity,
    p0=p0_polynomial_hill,
    maxfev=10000,
)

# Plot results
plt.figure(figsize=(6, 5))
plt.scatter(power_scaled, prep_fidelity, label="Data", color="blue")
plt.plot(
    power_scaled,
    polynomial_hill_model(power_scaled, *params_polynomial_hill),
    label="Polynomial-Hill Fit",
    color="red",
)
plt.title("Preparation Fidelity (Polynomial-Hill Model)")
plt.xlabel("Scaled Power")
plt.ylabel("Fidelity")
plt.legend()
plt.show(block=False)

# Fitted parameters
print("Optimized Readout Fidelity Parameters:", params_readout)
print("Optimized Preparation Fidelity Parameters:", params_prep)
print("Fitted Parameters (Polynomial-Hill):", params_polynomial_hill)

plt.show(block=True)

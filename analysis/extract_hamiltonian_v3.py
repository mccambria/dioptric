# -*- coding: utf-8 -*-
"""The NV hamiltonian is a function of 6 parameters: the components of the 
B field and the effective E field. We can express this as 
    [mag_B, theta_B, phi_B, par_E, perp_E, phi_E]
where writing B in spherical coordinates and E in cylindrical coordinates will
simplify our analysis. The parameter ranges are
    0 < mag_B
    0 < theta_B < pi
    0 < phi_B < 2 * pi
    par_E is unbounded
    0 < perp_E
    0 < phi_E < 2 * pi
We'd like to determine these parameters as best we can
from the resonances we measure as we vary mag_B keeping the other parameters
fixed. We don't actually know mag_B as we vary it, so unfortunately this is
not a straightforward curve_fit problem. We do, however, have two pieces of
information (two resonances) at each value of mag_B. In the abstract this
problem is then to find the vector
    fit_vec = [theta_B, phi_B, par_E, perp_E, phi_E]
that best reproduces
    a(mag_B; fit_vec) = low_resonance
    b(mag_B; fit_vec) = high_resonance
for unknown but varying values of mag_B. We can also express this as
    f(mag_B; fit_vec) = (low_resonance + high_resonance) / 2 = center_freq
    g(mag_B; fit_vec) = low_resonance - high_resonance = splitting
so that each function takes into consideration both resonances equally. If we
knew mag_B and just had the value of one of these functions, then we
could use curve_fit to find fit_vec. Because we have a second piece of
information we actually can turn our problem into that simpler curve_fit
problem. Mathematically, there exists some mag_B such that 
    g(mag_B; fit_vec) = splitting
for a given fit_vec. Now it's not necessarily the case that this value is
unique, but let's assume it is. We can find this value numerically and then
use it as the input for f, in which case we ended up just where we wanted to,
at the simple curve_fit problem. 

As an aid to curve_fit, we can determine par_E and perp_E by the resonances
at zero B field, as described below. Then we can guess theta_B, phi_B, and
phi_E as the centers of their possible ranges. There's another simplifying
factor. The characteristic polynomial of the NV Hamiltonian in the general
case only includes phi_E and phi_B in the form (2 * phi_B) + phi_E. Let's just
call this term phi. We can only find out anything out about phi; we can't
resolve differences between phi_B and phi_E. As a result, we can just set
phi_B = 0 and phi_E = phi and go about our business with one less fit
parameter. Additionally, by symmetry, phi_E can only range from 0 to 
2 * pi / 3. This can be seen more explicitly by looking at what happens to
(2 * phi_B) + phi_E if we rotate by 120 deg. We pick up an extra 360 deg and
cycle back to the same angle we had before the rotation. So ultimately
our fit vector and parameter bounds are
    fit_vec = [theta_B, par_E, perp_E, phi]
    0 < mag_B
    0 < theta_B < pi
    par_E is unbounded
    0 < perp_E
    0 < phi < 2 * pi / 3

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import numpy
from numpy.linalg import eigvals
from numpy import pi
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from numpy import inf
from numpy import exp
import matplotlib.pyplot as plt


# %% Constants


# GHz
d_gs = 2.87

# numbers
inv_sqrt_2 = 1/numpy.sqrt(2)


# %% Functions


def calc_single_hamiltonian(mag_B, par_E, perp_E, theta_B, phi):
    par_B = mag_B * numpy.cos(theta_B)
    perp_B = mag_B * numpy.sin(theta_B)
    phi_B = 0
    phi_E = phi
    hamiltonian = numpy.array([[d_gs + par_E + par_B,
                                inv_sqrt_2 * perp_B * exp(-1j * phi_B),
                                -perp_E * exp(1j * phi_E)],
                               [inv_sqrt_2 * perp_B * exp(1j * phi_B),
                                0,
                                inv_sqrt_2 * perp_B * exp(-1j * phi_B)],
                               [-perp_E * exp(-1j * phi_E),
                                inv_sqrt_2 * perp_B * exp(-1j * phi_B),
                                d_gs + par_E - par_B]])
    return hamiltonian


def calc_hamiltonian(mag_B, par_E, perp_E, theta_B, phi):
    if (type(mag_B) is list) or (type(mag_B) is numpy.ndarray):
        fit_vec = [par_E, perp_E, theta_B, phi]
        hamiltonian_list = [calc_single_hamiltonian(val, *fit_vec)
                            for val in mag_B]
        return hamiltonian_list
    else:
        return calc_single_hamiltonian(mag_B, par_E, perp_E, theta_B, phi)


def calc_res_pair(mag_B, par_E, perp_E, theta_B, phi):
    hamiltonian = calc_hamiltonian(mag_B, par_E, perp_E, theta_B, phi)
    if (type(mag_B) is list) or (type(mag_B) is numpy.ndarray):
        vals = numpy.sort(eigvals(hamiltonian), axis=1)
        resonance_low = numpy.real(vals[:,1] - vals[:,0])
        resonance_high = numpy.real(vals[:,2] - vals[:,0])
    else:
        vals = numpy.sort(eigvals(hamiltonian))
        resonance_low = numpy.real(vals[1] - vals[0])
        resonance_high = numpy.real(vals[2] - vals[0])
    return resonance_low, resonance_high


def find_mag_B(res_pair, par_E, perp_E, theta_B, phi):
    # Find the mag_B that minimizes the distance between the measured
    # resonances and the calculated resonances for a given fit_vec
    args = (res_pair, par_E, perp_E, theta_B, phi)
    result = minimize_scalar(find_mag_B_objective, bounds=(0, 1.0), args=args,
                             method='bounded')
    if result.success:
        mag_B = result.x
        if mag_B < 0:
            print('huh')
    else:
        # If we didn't find an optimal value, return something that will blow
        # up chisq and push us away from this fit_vec
        mag_B = 1.0
    return mag_B


def find_mag_B_objective(x, res_pair, par_E, perp_E, theta_B, phi):
    calculated_res_pair = calc_res_pair(x, par_E, perp_E, theta_B, phi)
    differences = calculated_res_pair - res_pair
    sum_squared_differences = numpy.sum(differences**2)
    return sum_squared_differences


def plot_resonances(mag_B_range, par_E, perp_E, theta_B, phi,
                    name='untitled'):

    smooth_mag_B = numpy.linspace(mag_B_range[0], mag_B_range[1], 1000)
    res_pairs = calc_res_pair(smooth_mag_B, par_E, perp_E, theta_B, phi)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.set_tight_layout(True)
    ax.set_title('Extracted resonance curves: {}'.format(name))
    ax.plot(smooth_mag_B, res_pairs[0])
    ax.plot(smooth_mag_B, res_pairs[1])
    ax.set_xlabel('B magnitude (GHz)')
    ax.set_ylabel('Resonance (GHz)')

    textstr = '\n'.join((
        r'$\theta_{B}=%.3f \ rad$' % (theta_B, ),
        r'$E_{\parallel}=%.3f \ GHz$' % (par_E, ),
        r'$E_{\perp}=%.3f \ GHz$' % (perp_E, ),
        r'$\phi=%.3f \ rad$' % (phi, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, fontsize=14, transform=ax.transAxes,
            verticalalignment='top', bbox=props)

    return fig, ax


def chisq_func(fit_vec, resonances, par_E, perp_E):

    num_resonance_pairs = len(resonances)
    mag_Bs = [find_mag_B(val, par_E, perp_E, *fit_vec) for val in resonances]

    # Guess the variance - this is very loosely based on the width of
    # our resonances
    estimated_var = 0.0001

    # find_mag_B_objective returns the sum of squared residuals for a single
    # pair of resonances. We want to sum this over all pairs.
    squared_residuals = [find_mag_B_objective(mag_Bs[ind], resonances[ind],
                         par_E, perp_E, *fit_vec) for ind
                         in range(num_resonance_pairs)]
    sum_squared_residuals = numpy.sum(squared_residuals)

    chisq = sum_squared_residuals / estimated_var

    return chisq



# %% Main


def main(name, zero_field_resonances, non_zero_field_resonances):

    ############ Setup ############

    if zero_field_resonances is not None:
        zero_field_resonances.sort()
    non_zero_field_resonances = numpy.array(non_zero_field_resonances)
    non_zero_field_resonances.sort(axis=1)

    ############ Zero field case ############

    if zero_field_resonances is not None:
        # Get the splitting and center_freq from the resonances
        zero_field_low = min(zero_field_resonances)
        zero_field_high = max(zero_field_resonances)
    
        # At B = 0 the Hamiltonian has the form
        # [     d_gs + par_E,           0,     - perp_E * exp(i phi_E)   ]
        # [            0,               0,                0              ]
        # [-perp_E * exp(-i phi_E),     0,           d_gs + par_E        ]
    
        # The eigenvalues are simple in this case
        # [0, d_gs + par_E - perp_E, d_gs + par_E + perp_E]
        # The resonances are thus
        # [d_gs + par_E - perp_E, d_gs + par_E + perp_E]]
        # and so
        # zero_field_center = (d_gs + par_E - perp_E + d_gs + par_E + perp_E) / 2
        # zero_field_center = d_gs + par_E
        zero_field_center = (zero_field_high + zero_field_low) / 2
        par_E = zero_field_center - d_gs
    
        # Similarly
        # zero_field_splitting = (d_gs + par_E + perp_E) - (d_gs + par_E - perp_E)
        # zero_field_splitting = 2 * perp_E
        zero_field_splitting = zero_field_high - zero_field_low
        perp_E = zero_field_splitting / 2
    
        # We won't consider these definite values; we'll just use them to inform
        # curve_fit, which will take into account all the available information.

    # If we didn't get anything to work with, just guess 0
    else:
        par_E = 0
        perp_E = 0

    ############ General case ############

    # fit_vec = [par_E, perp_E, theta_B, phi]
    guess_params = (pi/6, 0)
    param_bounds = ((0, pi), (0, 2*pi/3))
    args = (non_zero_field_resonances, par_E, perp_E)
    res = minimize(chisq_func, guess_params, args=args,
                   bounds=param_bounds, method='SLSQP')
    success = res.success
    if not success:
        print(res.message)
        return

    popt = res.x
    print(*popt)

    chisq = res.fun
    print(chisq)
    degrees_of_freedom = len(non_zero_field_resonances) - len(guess_params)
    reduced_chisq = res.fun / degrees_of_freedom
    print(reduced_chisq)

    ############ Plot the result ############

    # Get the mag_B for each pair of resonances with this fit_vec
    mag_Bs = [find_mag_B(val, par_E, perp_E, *popt) for val
              in non_zero_field_resonances]

    # Plot the calculated resonances up to the max mag_B
    fig, ax = plot_resonances([0, max(mag_Bs)], par_E, perp_E, *popt, name)

    # Plot the zero field resonances
    if zero_field_resonances is not None:
        ax.plot(0, zero_field_resonances[0])
        ax.plot(0, zero_field_resonances[1])

    # Plot the non-zero field resonances
    ax.scatter(mag_Bs, non_zero_field_resonances[:,0])
    ax.scatter(mag_Bs, non_zero_field_resonances[:,1])


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Each pair of resonances should be a 2-list. So zero_field_resonances
    # is a single 2-list and non_zero_field_resonances is a list of 2-lists.

    # name = 'nv2_2019_04_30'
    # zero_field_resonances = None
    # non_zero_field_resonances = [[2.8507, 2.8798],
    #                               [2.8434, 2.8882],
    #                               [2.8380, 2.8942],
    #                               [2.8379, 2.8948],
    #                               [2.8308, 2.9006],
    #                               [2.8228, 2.9079],
    #                               [2.8155, 2.9171]]

    name = 'nv2_2019_04_30_take2'
    zero_field_resonances = None
    non_zero_field_resonances = [[2.8512, 2.8804],
                                  [2.8435, 2.8990],
                                  [2.8265, 2.9117],
                                  [2.7726, 3.0530],
                                  [2.7738, 3.4712]]

    # name = 'nv1_2019_05_10'
    # zero_field_resonances = None
    # non_zero_field_resonances = [[2.8554, 2.8752],
    #                               [2.8512, 2.8790],
    #                               [2.8520, 2.8800],
    #                               [2.8536, 2.8841],
    #                               [2.8496, 2.8823],
    #                               [2.8396, 2.8917],
    #                               [2.8166, 2.9144],
    #                               [2.8080, 2.9240],
    #                               [2.7357, 3.0037],
    #                               [2.6061, 3.1678],
    #                               [2.6055, 3.1691],
    #                               [2.4371, 3.4539]]


    # name = test
    # zero_field_resonances = [2.88, 2.88]
    # non_zero_field_resonances = [[2.86, 2.90],
    #                               [2.87, 2.89],
    #                               [2.78, 2.98],
    #                               [2.83, 2.93],
    #                               [2.84, 2.92]]

    # Run the script
    main(name, zero_field_resonances, non_zero_field_resonances)

    # Test plot
    # args: mag_B_range, par_E, perp_E, theta_B, phi
    # plot_resonances([0, 0.5], pi/2, 0, 0, 0)

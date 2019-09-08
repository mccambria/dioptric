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


def calc_single_hamiltonian(mag_B, theta_B, par_E, perp_E, phi):
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


def calc_hamiltonian(mag_B, theta_B, par_E, perp_E, phi):
    if (type(mag_B) is list) or (type(mag_B) is numpy.ndarray):
        fit_vec = [theta_B, par_E, perp_E, phi]
        hamiltonian_list = [calc_single_hamiltonian(val, *fit_vec)
                            for val in mag_B]
        return hamiltonian_list
    else:
        return calc_single_hamiltonian(mag_B, theta_B, par_E, perp_E, phi)


def calc_res_pair(mag_B, theta_B, par_E, perp_E, phi):
    hamiltonian = calc_hamiltonian(mag_B, theta_B, par_E, perp_E, phi)
    if (type(mag_B) is list) or (type(mag_B) is numpy.ndarray):
        vals = numpy.sort(eigvals(hamiltonian), axis=1)
        resonance_low = numpy.real(vals[:,1] - vals[:,0])
        resonance_high = numpy.real(vals[:,2] - vals[:,0])
    else:
        vals = numpy.sort(eigvals(hamiltonian))
        resonance_low = numpy.real(vals[1] - vals[0])
        resonance_high = numpy.real(vals[2] - vals[0])
    return resonance_low, resonance_high


def find_mag_B(res_desc, theta_B, par_E, perp_E, phi):
    # Just return the given mag_B if it's known
    if res_desc[0] is not None:
        return res_desc[0]
    # Otherwise we'll determine the most likely mag_B for this fit_vec by
    # finding the mag_B that minimizes the distance between the measured
    # resonances and the calculated resonances for a given fit_vec
    args = (res_desc, theta_B, par_E, perp_E, phi)
    result = minimize_scalar(find_mag_B_objective, bounds=(0, 1000), args=args,
                             method='bounded')
    if result.success:
        mag_B = result.x
    else:
        # If we didn't find an optimal value, return something that will blow
        # up chisq and push us away from this fit_vec
        mag_B = 0.0
    return mag_B


def find_mag_B_objective(x, res_desc, theta_B, par_E, perp_E, phi):
    calculated_res_pair = calc_res_pair(x, theta_B, par_E, perp_E, phi)
    differences = calculated_res_pair - res_desc[1:3]
    sum_squared_differences = numpy.sum(differences**2)
    return sum_squared_differences


def plot_resonances(mag_B_range, theta_B, par_E, perp_E, phi,
                    name='untitled'):

    smooth_mag_B = numpy.linspace(mag_B_range[0], mag_B_range[1], 1000)
    res_pairs = calc_res_pair(smooth_mag_B, theta_B, par_E, perp_E, phi)

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


def chisq_func(fit_vec, res_descs):

    num_resonance_descs = len(res_descs)
    mag_Bs = [find_mag_B(desc, *fit_vec) for desc in res_descs]

    # Guess the variance - this is very loosely based on the width of
    # our resonances
    estimated_var = 1.0

    # find_mag_B_objective returns the sum of squared residuals for a single
    # pair of resonances. We want to sum this over all pairs.
    squared_residuals = [find_mag_B_objective(mag_Bs[ind], res_descs[ind],
                         *fit_vec) for ind in range(num_resonance_descs)]
    sum_squared_residuals = numpy.sum(squared_residuals)

    chisq = sum_squared_residuals / estimated_var

    return chisq


# %% Main


def main(name, res_descs):

    ############ Setup ############

    res_descs = numpy.array(res_descs)
    for desc in res_descs:
        # Set degenerate resonances to the same value
        if desc[2] is None:
            desc[2] = desc[1]
        # Make sure resonances are sorted
        desc[1:3] = numpy.sort(desc[1:3])

    ############ Zero field case ############

    # See if we have zero-field resonances
    zero_field_res_desc = None
    for desc in res_descs:
        if desc[0] == 0.0:
            zero_field_res_desc = desc
            break

    if zero_field_res_desc is not None:
        # Get the splitting and center_freq from the resonances
        zero_field_low = zero_field_res_desc[1]
        zero_field_high = zero_field_res_desc[2]
    
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

    # fit_vec = [theta_B, par_E, perp_E, phi]
    guess_params = (pi/6, par_E, perp_E, 0)
    param_bounds = ((0, pi/2), (-100, 100), (0, 100), (0, 2*pi/3))
    args = (res_descs)
    res = minimize(chisq_func, guess_params, args=args,
                   bounds=param_bounds, method='SLSQP')
    if not res.success:
        print(res.message)
        return

    popt = res.x

    chisq = res.fun
    print('Chi squared: {:.4g}'.format(chisq))
    degrees_of_freedom = len(res_descs) - len(guess_params)
    reduced_chisq = res.fun / degrees_of_freedom
    print('Reduced chi squared: {:.4g}'.format(reduced_chisq))

    # jac_e5 = res.jac * 10**5
    # print(jac_e5)
    # print(numpy.outer(jac_e5, jac_e5))
    # pcov = reduced_chisq * numpy.linalg.inv(numpy.outer(jac_e5, jac_e5))
    # test = numpy.outer(jac_e5, jac_e5)
    # print(numpy.matmul(test, numpy.linalg.inv(test)))
    # # return
    # print(pcov)
    # st_errors = [numpy.sqrt(pcov[ind, ind]) for ind in range(len(jac))]
    # print(st_errors)

    ############ Plot the result ############

    # Get the mag_B for each pair of resonances with this fit_vec
    mag_Bs = [find_mag_B(desc, *popt) for desc in res_descs]

    # Plot the calculated resonances up to the max mag_B
    fig, ax = plot_resonances([0, max(mag_Bs)], *popt, name)

    # Plot the resonances
    ax.scatter(mag_Bs, res_descs[:,1])
    ax.scatter(mag_Bs, res_descs[:,2])


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Each pair of resonances should be a 2-list. So zero_field_resonances
    # is a single 2-list and non_zero_field_resonances is a list of 2-lists.

    # name = 'nv2_2019_04_30'
    # res_descs = [[0.0, 2.8658, None],
    #               [None, 2.8507, 2.8798],
    #               [None, 2.8434, 2.8882],
    #               [None, 2.8380, 2.8942],
    #               [None, 2.8379, 2.8948],
    #               [None, 2.8308, 2.9006],
    #               [None, 2.8228, 2.9079],
    #               [None, 2.8155, 2.9171]]

    name = 'nv2_2019_04_30_take2'
    # res_descs = [[0.0, 2.8572, None],
    #               [None, 2.8512, 2.8804],
    #               [None, 2.8435, 2.8990],
    #               [None, 2.8265, 2.9117],
    #               [None, 2.7726, 3.0530],
    #               [None, 2.7738, 3.4712]]
    res_descs = [[0.0, 2857.2, None],
                  [None, 2851.2, 2880.4],
                  [None, 2843.5, 2899.0],
                  [None, 2826.5, 2911.7],
                  [None, 2772.6, 3053.0],
                  [None, 2773.8, 3471.2]]
    # res_descs = [[None, 2.8512, 2.8804],
    #               [None, 2.8435, 2.8990],
    #               [None, 2.8265, 2.9117],
    #               [None, 2.7726, 3.0530],
    #               [None, 2.7738, 3.4712]]

    # name = 'nv1_2019_05_10'
    # res_descs = [[None, 2.8554, 2.8752],
    #               [None, 2.8512, 2.8790],
    #               [None, 2.8520, 2.8800],
    #               [None, 2.8536, 2.8841],
    #               [None, 2.8496, 2.8823],
    #               [None, 2.8396, 2.8917],
    #               [None, 2.8166, 2.9144],
    #               [None, 2.8080, 2.9240],
    #               [None, 2.7357, 3.0037],
    #               [None, 2.6061, 3.1678],
    #               [None, 2.6055, 3.1691],
    #               [None, 2.4371, 3.4539]]

    # name = 'NV16_2019_07_25'
    # res_descs = [[0.0, 2.8655, None],
    #               [None, 2.8519, 2.8690],
    #               [None, 2.8460, 2.8746],
    #               [None, 2.8337, 2.8867],
    #               [None, 2.8202, 2.9014],
    #               [None, 2.8012, 2.9292],
    #               [None, 2.7393, 3.0224],
    #               [None, 2.6995, 3.1953],
    #               [None, 2.5830, 3.3290]]


    # name = 'test'
    # res_descs = [[0.0, 2.88, None],
    #               [None, 2.90, 2.86],
    #               [None, 2.87, 2.89],
    #               [None, 2.78, 2.98],
    #               [None, 2.83, 2.93],
    #               [None, 2.84, 2.92]]

    # Run the script
    main(name, res_descs)

    # Test plot
    # args: mag_B_range, theta_B, par_E, perp_E, phi
    # plot_resonances([0, 0.5], pi/2, 0, 0, 0)

# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import numpy


# %% Constants


sqrt_2 = numpy.sqrt(2)


# %% Functions


# %% Main


def main(gammas, gamma_errors, omegas, omega_errors):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    # Convert to arrays
    gammas = numpy.array(gammas)
    gamma_errors = numpy.array(gamma_errors)
    omegas = numpy.array(omegas)
    omega_errors = numpy.array(omega_errors)

    # Gamma omega ratios
    gamma_omega_ratios = gammas / omegas
    rel_squared_errors = (gamma_errors/gammas)**2 + (omega_errors/omegas)**2
    gamma_omega_ratio_errors = gamma_omega_ratios * numpy.sqrt(rel_squared_errors)

    # Weighted mean gamma_omega_ratio
    weights = gamma_omega_ratio_errors**-2  # Inverse squared error
    mean_gamma_omega_ratio = numpy.sum(gamma_omega_ratios*weights)
    mean_gamma_omega_ratio /= numpy.sum(weights)

    # gamma_omega_ratio standard error
    mean_gamma_omega_ratio_ste_direct = numpy.std(gamma_omega_ratios, ddof=1)
    mean_gamma_omega_ratio_ste_direct /=  numpy.sqrt(len(gamma_omega_ratios))
    mean_gamma_omega_ratio_ste_prop = numpy.sqrt((numpy.sum(weights))**-1)

    print('gamma_omega_ratios: {}'.format(gamma_omega_ratios))
    print('gamma_omega_ratio_errors: {}'.format(gamma_omega_ratio_errors))

    # d_perp_prime calculation
    ratio = mean_gamma_omega_ratio
    ratio_unc = mean_gamma_omega_ratio_ste_prop
    print('mean_gamma_omega_ratio: {}'.format(mean_gamma_omega_ratio))
    print('mean_gamma_omega_ratio_ste_prop: {}'.format(mean_gamma_omega_ratio_ste_prop))
    d_parallel = 0.35
    d_parallel_unc = 0.02
    d_perp = 17
    d_perp_unc = 3
    d_perp_prime = d_parallel**2 + sqrt_2*d_parallel*d_perp + d_perp**2
    d_perp_prime = numpy.sqrt((mean_gamma_omega_ratio / 2) * d_perp_prime)
    print('d_perp_prime: {}'.format(d_perp_prime))
    d_perp_prime_unc = ((1/2)*(d_parallel**2 + sqrt_2*d_parallel*d_perp + d_perp**2))**2 * ratio_unc**2
    d_perp_prime_unc += (ratio*d_parallel + sqrt_2*ratio*d_perp/2)**2 * d_parallel_unc**2
    d_perp_prime_unc += (sqrt_2*ratio*d_parallel/2 + ratio*d_perp)**2 * d_perp_unc**2
    d_perp_prime_unc *= (1/(2*d_perp_prime))**2
    d_perp_prime_unc = numpy.sqrt(d_perp_prime_unc)
    print('d_perp_prime_unc: {}'.format(d_perp_prime_unc))


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here
    omegas = [0.063, 0.053, 0.059, 0.061, 0.060]
    omega_errors = [0.009, 0.003, 0.006, 0.006, 0.004]
    gammas = [0.127, 0.111, 0.144, 0.132, 0.114]
    gamma_errors = [0.023, 0.009, 0.025, 0.017, 0.012]

    # Run the script
    main(gammas, gamma_errors, omegas, omega_errors)

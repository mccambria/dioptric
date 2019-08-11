# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import numpy


# %% Constants


# GHz
d_gs = 2.87

# numbers
inv_sqrt_2 = 1/numpy.sqrt(2)


# %% Functions


def non_zero_b_splitting(center_freq, b_par, b_perp, phi_b):

    scaled_b_perp = inv_sqrt_2 * b_mag
    zero_field_center = d_gs + pi_par

    hamiltonian = numpy.array([[zero_field_center + b_par, scaled_b_perp * numpy.exp(-j phi_b), -pi_perp],
                                [scaled_b_perp * numpy.exp(j phi_b), 0, scaled_b_perp * numpy.exp(-j phi_b)],
                                [-pi_perp, scaled_b_perp * numpy.exp(j phi_b), zero_field_center - b_par]])

    vals = numpy.linalg.eigvals(hamiltonian)
    vals = numpy.sort(vals)


def extract_b_field(res_low, res_high, pi_par, pi_perp):

    # We need res_low = eigvals(H)

    scaled_b_perp = inv_sqrt_2 * b_mag
    zero_field_center = d_gs + pi_par

    hamiltonian = numpy.array([[zero_field_center + b_par, scaled_b_perp, -pi_perp],
                                [scaled_b_perp, 0, scaled_b_perp],
                                [-pi_perp, scaled_b_perp, zero_field_center - b_par]])

    vals = numpy.linalg.eigvals(hamiltonian)
    vals = numpy.sort(vals)




# %% Main


def main(zero_field_splitting, non_zero_field_splittings):

    # lambda_0, lambda_1, and lambda_2 are the eigenvalues in increasing order

    # At B = 0 and pi_perp along the x axis, the Hamiltonian has the form
    # [d_gs + pi_par,      0,       -pi_perp   ]
    # [        0,          0,           0      ]
    # [    -pi_perp,       0,     d_gs + pi_par]

    # pi_par adds a constant offset to the center frequency so
    zero_field_center = (zero_field_splitting[0] + zero_field_splitting[1]) / 2
    pi_par = zero_field_center - d_gs

    # pi_perp splits the levels about the center frequency
    zero_field_diff = zero_field_splitting[1] - zero_field_splitting[0]
    pi_perp = zero_field_diff / 2

    # For non-zero B, the Hamiltonian has the form
    # [d_gs + pi_par + b_par,               inv_sqrt_2 b_perp exp(-j phi_b),                 -pi_perp           ]
    # [inv_sqrt_2 b_perp exp(j phi_b),                  0,                       inv_sqrt_2 b_perp exp(-j phi_b)]
    # [          -pi_perp,                  inv_sqrt_2 b_perp exp(j phi_b),             d_gs + pi_par - b_par   ]
    # where phi_b  is the azimuthal angle arctan(b_y/b_x)

    # We can't extract the values we want so simply anymore since there
    # are three unknowns and each set of resonances only contains two pieces
    # of information. We'll have to fit the eigenvalues numerically. The
    # independent variable will be the lower resonance and the dependent
    # variable will be the higher resonance

    # The angle phi_b has no effect on the resonant frequencies (only the
    # contrasts, so let's ignore them)
    # Each point has two pieces of information and we have two values to
    # extract so we'll calculate the values exactly for each point and
    # then take the average/standard deviation







# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here
    zero_field_splitting = []
    non_zero_field_splittings = [[]]

    # Run the script
    main()

# -*- coding: utf-8 -*-
"""
Maximum likelihood ellipse fitting API. Entry point is main. Determines the 
phase phi that gives the probability distribution that is most likely
to have yielded the passed points distributions. For now you must pass
the contrast and number of atoms, though there is no reason these could
not also be determined by the algorithm.

Created November 2nd, 2022

@author: mccambria
"""

import numpy as np
from scipy.optimize import minimize

num_ellipse_samples = 1000
theta_linspace = np.linspace(0, 2 * np.pi, num_ellipse_samples, endpoint=False)


def biv_normal(data_point, ellipse_sample, num_atoms):
    """Returns the probability density of a bivariate normal centered at
    ellipse_sample and evaluated at data_point. The distribution approximates
    the bivariate binomial distribution produced by quantum projection noise.
    Accordingly, num_atoms is used to calculate the x/y variances
    """
    data_point_x, data_point_y = data_point
    ellipse_sample_x, ellipse_sample_y = ellipse_sample
    varx = ellipse_sample_x * (1 - ellipse_sample_x) / num_atoms
    vary = ellipse_sample_y * (1 - ellipse_sample_y) / num_atoms
    sdx = np.sqrt(varx)
    sdy = np.sqrt(vary)
    z = (((data_point_x - ellipse_sample_x) / sdx) ** 2) + (
        ((data_point_y - ellipse_sample_y) / sdy) ** 2
    )
    return (1 / (2 * np.pi * sdx * sdy)) * np.exp(-z / 2)


def cost(phi, points, contrast, num_atoms):
    """Cost function - returns the log likelihood of the probability distribution
    produced by a given phi
    """
    amp = contrast / 2
    ellipse_samples = ellipse_point(theta_linspace, phi, amp)
    all_probs = [
        biv_normal(point, ellipse_samples, num_atoms) for point in points
    ]
    point_probs = [np.sum(el) / num_ellipse_samples for el in all_probs]
    point_probs = [-10 if (el < 1e-10) else np.log10(el) for el in point_probs]
    log_likelihood = np.sum(point_probs)
    cost = -log_likelihood  # Best should be minimum
    return cost


def ellipse_point(theta, phi, amp):
    """Returns a tuple describing the coordinates of a point on the ellipse
    for the passed angles and amplitude (half contrast). Assumes the ellipse
    is centered at (0.5, 0.5)
    """
    return (0.5 + amp * np.cos(theta + phi), 0.5 + amp * np.cos(theta - phi))


def main(points, contrast, num_atoms):
    """API entry point

    Parameters
    ----------
    points : list
        List of coordinates of experimental data points
    contrast : float
        Ellipse contrast, between 0 and 1
    num_atoms : int
        Number of atoms in the experiments

    Returns
    -------
    float
        Maximum likelihood estimator for phi
    """

    points_x = np.array([point[0] for point in points])
    points_y = np.array([point[1] for point in points])

    # Use the correlation in the data points to give us a good guess phi
    corr = np.corrcoef(points_x, points_y)[0, 1]
    corr_phi = np.arccos(corr) / 2

    res = minimize(
        cost,
        (corr_phi,),
        args=(points, contrast, num_atoms),
        bounds=((0, np.pi / 2),),
    )
    opti_phi = res.x[0]

    # Remove degeneracies
    opti_phi = opti_phi % np.pi
    if opti_phi > np.pi / 2:
        opti_phi = np.pi - opti_phi

    return opti_phi

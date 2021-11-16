# -*- coding: utf-8 -*-
"""
Various calculations from the note "Confocal Microscope Optimizations"

Created on November 15th, 2021

@author: mccambria
"""


import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from numpy import pi
import numpy as np
import scipy.integrate as integrate
from scipy.special import jv as bessel_func

wavelength = 700e-9
wavenumber = 2 * pi / wavelength
sample_focal_length = 2.87e-3
# sample_aperture_radius = 2.35e-3
sample_aperture_radius = 100e-6


def psf_field_integrand(r_prime, r, z, f, k):
    lorentzian = r_prime / np.sqrt(r_prime ** 2 + f ** 2)
    # lorentzian = r_prime / f
    phase = np.exp(1j * k * (r ** 2 + r_prime ** 2) / (2 * z))
    bessel = bessel_func(0, -k * r * r_prime / z)
    return lorentzian * phase * bessel


def psf_field(r, z, f, k, norm, phi):
    """Calculate the field associated with the light emitted from an isotropic
    point emitter located at the focus of a lens

    Parameters
    ----------
    r : float
        Radius in the input plane
    z : float
        Distance from the input plane
    f : float
        Focal length of the lens
    k : float
        Wavenumber of the light
    norm : float
        Normalization coefficient
    phi : float
        Radius of the lens aperture

    Returns
    -------
    np.array(float)
        1D array of the psf field values along the radial sweep
    """
    coeff = norm / z
    if type(r) in [list, np.ndarray]:
        integral = []
        for val in r:
            psf_field_integrand_lambda = lambda r_prime: psf_field_integrand(
                r_prime, val, z, f, k
            )
            ret_vals = integrate.quad(psf_field_integrand_lambda, 0, phi)
            integral.append(ret_vals[0])
        integral = np.array(integral)
    else:
        psf_field_integrand_lambda = lambda r_prime: psf_field_integrand(
            r_prime, r, z, f, k
        )
        integral = integrate.quad(psf_field_integrand_lambda, 0, phi)
    return coeff * integral


def plot_psf():

    norm = 100
    num_points = 100

    z = 1
    r_range = 5e-3
    r_linspace = np.linspace(-r_range, +r_range, num_points)

    r = 0
    z_range = 10
    z_linspace = np.linspace(-r_range, +r_range, num_points)

    field = psf_field(
        r,  # r_linspace,
        z_linspace,  # z,
        sample_focal_length,
        wavenumber,
        norm,
        sample_aperture_radius,
    )
    intensity = np.abs(field) ** 2
    fig, ax = plt.subplots()
    # ax.plot(r_linspace, intensity)
    ax.plot(z_linspace, intensity)


if __name__ == "__main__":

    tool_belt.init_matplotlib()

    plot_psf()

    plt.show(block=True)

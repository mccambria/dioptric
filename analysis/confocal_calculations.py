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
sample_aperture_radius = 2.35e-3
# sample_aperture_radius = 10e-6
# sample_aperture_radius = np.infty
fiber_mfr = 2e-6


def psf_field_integrand(r_prime, r, z, f, k):
    input_profile = np.exp(-(r_prime ** 2) / (2 * 1e-3 ** 2))
    # input_profile = 1 / np.sqrt(r_prime ** 2 + f ** 2)
    # input_profile = 1
    phase = np.exp(1j * k * (r ** 2 + r_prime ** 2) / (2 * z))
    bessel = bessel_func(0, -k * r * r_prime / z)
    return r_prime * input_profile * phase * bessel


def psf_field_single(r, z, f, k, norm, phi):
    psf_field_integrand_lambda = lambda r_prime: psf_field_integrand(
        r_prime, r, z, f, k
    )
    integrand_real = lambda r_prime: np.real(
        psf_field_integrand_lambda(r_prime)
    )
    integrand_imag = lambda r_prime: np.imag(
        psf_field_integrand_lambda(r_prime)
    )
    real = integrate.quad(integrand_real, 0, phi)[0]
    imag = integrate.quad(integrand_imag, 0, phi)[0]
    return norm * (real + 1j * imag)


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
    if type(r) in [list, np.ndarray]:
        result = []
        for val in r:
            result.append(psf_field_single(val, z, f, k, norm, phi))
        result = np.array(result)
    else:
        result = psf_field_single(r, z, f, k, norm, phi)
    return result


def plot_psf():

    num_points = 1000

    z = 100
    # r_range = z / 1000
    # r_range = z / 500
    r_range = 0.05
    r_linspace = np.linspace(-r_range, +r_range, num_points)

    # r = 0
    # z_range = 10
    # z_linspace = np.linspace(-r_range, +r_range, num_points)

    norm = 1 / psf_field_single(
        0, z, sample_focal_length, wavenumber, 1, sample_aperture_radius
    )

    field = psf_field(
        r_linspace,
        z,
        # r,  # r_linspace,
        # z_linspace,  # z,
        sample_focal_length,
        wavenumber,
        norm,
        sample_aperture_radius,
    )
    intensity = np.abs(field) ** 2
    fig, ax = plt.subplots()
    ax.plot(r_linspace, intensity)
    # ax.plot(z_linspace, intensity)


def calc_overlap():

    num_points = 100

    z = 50
    r_range = 0.05
    r_linspace = np.linspace(0, +r_range, num_points)
    r_step = r_linspace[1] - r_linspace[0]

    norm = 1 / psf_field_single(
        0, z, sample_focal_length, wavenumber, 1, sample_aperture_radius
    )
    NV_field = psf_field(
        r_linspace,
        z,
        # r,  # r_linspace,
        # z_linspace,  # z,
        sample_focal_length,
        wavenumber,
        norm,
        sample_aperture_radius,
    )

    overlaps = []
    collection_focal_lengths = np.linspace(100e-6, 20e-3, 100)

    for f in collection_focal_lengths:

        omega_col = 8 * f / (wavenumber * fiber_mfr)
        fiber_field = (1 / (np.pi * omega_col ** 2)) * np.exp(
            -((r_linspace / omega_col) ** 2)
        )
        overlaps.append(
            np.abs(
                np.sum(
                    (
                        NV_field
                        * fiber_field
                        * (2 * np.pi * r_linspace * r_step)
                    )
                )
            )
            ** 2
        )

    overlaps = np.array(overlaps)

    fig, ax = plt.subplots()
    ax.plot(collection_focal_lengths, overlaps)


if __name__ == "__main__":

    tool_belt.init_matplotlib()

    # plot_psf()

    calc_overlap()

    plt.show(block=True)

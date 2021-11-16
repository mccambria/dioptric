# -*- coding: utf-8 -*-
"""
Various calculations from the note "Confocal Microscope Optimizations". 
Everything assumes cylindrical symmetry about the optical axis.

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
k = 2 * pi / wavelength
sample_focal_length = 2.87e-3
sample_aperture_radius = 2.35e-3
# sample_aperture_radius = 1.76e-3
# sample_aperture_radius = 50e-6
# sample_aperture_radius = np.infty
fiber_mfr = 2e-6


def riemann_sum(integrand, delta):
    """Calculate an 1D integral using a midpoint Riemann sum with a uniform
    discretization delta. The domain of the integral is the entirety of the
    integrand

    Parameters
    ----------
    integrand : numpy.ndarray(float)
        May be complex
    delta : float
        Discretization size
    """

    return np.sum(integrand) * delta


def intensity(field):
    return np.abs(field) ** 2


def normalize_field(field, r_linspace):
    """Normalizes the power of a field by integrating the squared magnitude
    of the field over theta=[0, 2pi] and r=[0,max(r_linspace)].
    """

    # return field
    field_intensity = intensity(field)
    delta = get_linspace_delta(r_linspace)
    integrand = field_intensity * (2 * np.pi * r_linspace)
    inv_norm = riemann_sum(integrand, delta)
    field /= np.sqrt(inv_norm)
    return field


def lens_phase_mask(field, r, f):
    """Apply a phase mask for a parabolic lens to the input field

    Parameters
    ----------
    field : [type]
        [description]
    """

    phase_mask = np.exp(-1j * k * r ** 2 / (2 * f))
    return field * phase_mask


def get_linspace_delta(linspace):
    return (linspace[-1] - linspace[0]) / (len(linspace) - 1)


def aperture_propagate(input_field, input_r, output_r, z, aperture_rad):

    delta = get_linspace_delta(input_r)
    output_field = []

    # Apply the aperture
    trunc_inds = np.where(input_r <= aperture_rad)[0]
    input_r_trunc = input_r[trunc_inds]
    input_field_trunc = input_field[trunc_inds]

    for val in output_r:
        coeff = k / z
        phase = np.exp(1j * k * (val ** 2 + input_r_trunc ** 2) / (2 * z))
        bessel = bessel_func(0, -k * val * input_r_trunc / z)
        integrand = coeff * input_field_trunc * input_r_trunc * phase * bessel
        output_field.append(riemann_sum(integrand, delta))
    return np.array(output_field)


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

    overlaps = []
    collection_focal_lengths = np.linspace(100e-6, 20e-3, 50)
    fiber_r_range = fiber_mfr * 4
    num_points = 1000
    r_linspace = np.linspace(0, fiber_r_range, num_points)
    delta = get_linspace_delta(r_linspace)

    fiber_field = np.exp(-2 * ((r_linspace / fiber_mfr) ** 2))
    fiber_field = normalize_field(fiber_field, r_linspace)

    for f in collection_focal_lengths:

        nv_field = calc_nv_field_at_fiber(fiber_r_range, num_points, f)
        integrand = nv_field * fiber_field * (2 * np.pi * r_linspace)
        overlaps.append(np.abs(riemann_sum(integrand, delta)) ** 2)

    overlaps = np.array(overlaps)

    fig, ax = plt.subplots()
    ax.plot(collection_focal_lengths, overlaps)


def calc_nv_field_at_fiber(
    fiber_r_range=fiber_mfr * 3,
    num_points=1000,
    collection_focal_length=18e-3,
    do_plot=False,
):

    r_range = 0.0075
    # r_range = 150e-6
    input_r_linspace = np.linspace(0, +r_range, num_points)
    free_space_distance = 0.1

    ###### Calculate fields coming out of objective here ######

    # Point source

    # phase = np.exp(
    #     1j * k * np.sqrt(input_r_linspace ** 2 + sample_focal_length ** 2)
    # )
    # mag = 1 / np.sqrt(input_r_linspace ** 2 + sample_focal_length ** 2)
    # nv_field = phase * mag
    # nv_field = lens_phase_mask(nv_field, input_r_linspace, sample_focal_length)

    # Collimated Gaussian beam coming out of objective (standard calculation,
    # focus is placed at objective aperture)

    nv_field = np.exp(-((input_r_linspace / sample_aperture_radius) ** 2))

    # Plane wave

    # nv_field = np.array([1] * num_points)

    ######

    # Make sure we have a normalized field before we propagate (or at least
    # before we pass through the collection objective). Otherwise we could
    # have inaccuracies resulting from inconsistent normalization.
    nv_field = normalize_field(nv_field, input_r_linspace)

    # # Sample objective aperture
    # nv_field = aperture_propagate(
    #     nv_field,
    #     input_r_linspace,
    #     input_r_linspace,
    #     0.4,
    #     sample_aperture_radius,
    # )

    # # First sample telescope lens
    # nv_field = lens_phase_mask(nv_field, input_r_linspace, 0.4)
    # output_r_linspace = input_r_linspace / 4
    # nv_field = aperture_propagate(
    #     nv_field,
    #     input_r_linspace,
    #     output_r_linspace,
    #     0.5,
    #     0.025,
    # )

    # # Second sample telescope lens
    # input_r_linspace = output_r_linspace
    # output_r_linspace = input_r_linspace
    # nv_field = lens_phase_mask(nv_field, input_r_linspace, 0.1)
    # nv_field = aperture_propagate(
    #     nv_field,
    #     input_r_linspace,
    #     output_r_linspace,
    #     free_space_distance,
    #     0.025,
    # )

    # # First collection telescope lens
    # input_r_linspace = output_r_linspace
    # output_r_linspace = input_r_linspace * 3
    # nv_field = lens_phase_mask(nv_field, input_r_linspace, 0.05)
    # nv_field = aperture_propagate(
    #     nv_field,
    #     input_r_linspace,
    #     output_r_linspace,
    #     0.2,
    #     0.0125,
    # )

    # # Second collection telescope lens
    # input_r_linspace = output_r_linspace
    # output_r_linspace = input_r_linspace
    # nv_field = lens_phase_mask(nv_field, input_r_linspace, 0.15)
    # nv_field = aperture_propagate(
    #     nv_field,
    #     input_r_linspace,
    #     output_r_linspace,
    #     free_space_distance,
    #     0.0125,
    # )

    # Collection objective
    # input_r_linspace = output_r_linspace
    output_r_linspace = np.linspace(0, fiber_r_range, num_points)
    nv_field = lens_phase_mask(
        nv_field, input_r_linspace, collection_focal_length
    )
    nv_field = aperture_propagate(
        nv_field,
        input_r_linspace,
        output_r_linspace,
        collection_focal_length,
        0.0125,
    )
    # print(nv_field)

    if do_plot:
        fig, ax = plt.subplots()
        intensity_nv_field = intensity(nv_field)
        ax.plot(output_r_linspace, intensity_nv_field, label="NV field")
        fiber_field = np.exp(-2 * ((output_r_linspace / fiber_mfr) ** 2))
        fiber_field = normalize_field(fiber_field, output_r_linspace)
        ax.plot(output_r_linspace, intensity(fiber_field), label="Fiber mode")
        ax.legend()

    return nv_field


if __name__ == "__main__":

    tool_belt.init_matplotlib()

    # plot_psf()
    calc_overlap()
    # calc_nv_field_at_fiber(collection_focal_length=7e-3, do_plot=True)

    plt.show(block=True)

    # x_vals = np.linspace(0, 3, 1000)
    # delta = (x_vals[-1] - x_vals[0]) / (len(x_vals) - 1)
    # integrand = np.exp(1j * x_vals)
    # print(riemann_sum(integrand, delta))

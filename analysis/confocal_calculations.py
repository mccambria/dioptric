# -*- coding: utf-8 -*-
"""
Various calculations from the note "Confocal Microscope Optimizations". 
Everything assumes cylindrical symmetry about the optical axis.

Created on November 15th, 2021

@author: mccambria
"""


from numpy.lib.type_check import imag
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from numpy import pi
import numpy as np
import scipy.integrate as integrate
from scipy.special import jv as bessel_func

wavelength = 700e-9
k = 2 * pi / wavelength
sample_focal_length = 2.87e-3
# sample_aperture_radius = 2.35e-3
sample_na = 0.82
sample_aperture_radius = sample_focal_length * sample_na
sample_divergence_angle = np.arcsin(sample_na)
# sample_aperture_radius = 1e-3
# sample_aperture_radius = 10e-6
# sample_aperture_radius = np.infty
fiber_mfr = 2e-6
inch = 25.4e-3


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


def get_intensity_norm(field, r_linspace, r_max=None):
    field_intensity = intensity(field)
    delta = get_linspace_delta(r_linspace)
    if r_max is None:
        r_linspace_trunc = r_linspace
        field_intensity_trunc = field_intensity
    else:
        trunc_inds = np.where(r_linspace <= r_max)[0]
        r_linspace_trunc = r_linspace[trunc_inds]
        field_intensity_trunc = field_intensity[trunc_inds]
    integrand = field_intensity_trunc * (2 * np.pi * r_linspace_trunc)
    norm = riemann_sum(integrand, delta)
    return norm


def normalize_field(field, r_linspace, r_max=None):
    """Normalizes the power of a field by integrating the squared magnitude
    of the field over theta=[0, 2pi] and r=[0, r_max]
    """

    norm = get_intensity_norm(field, r_linspace, r_max)
    field /= np.sqrt(norm)
    return field


def get_linspace_delta(linspace):
    return (linspace[-1] - linspace[0]) / (len(linspace) - 1)


def lens_phase_mask(field, r, f):
    """Apply a phase mask for a spherical lens to the input field

    Parameters
    ----------
    field : [type]
        [description]
    """

    # phase_mask = np.exp(-1j * k * np.sqrt(r ** 2 + f ** 2))
    phase_mask = np.exp(-1j * k * r ** 2 / (2 * f))
    return field * phase_mask


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


def gu_psf_integrand(r, z, theta):
    phase = np.exp(-1j * k * z * np.cos(theta))
    bessel = bessel_func(0, k * r * np.sin(theta))
    integrand = k * phase * bessel * np.sin(theta)
    return integrand


def gu_psf(r_linspace, z, alpha):
    # Charrière optics letter 2007
    psf = []
    theta_linspace = np.linspace(0, alpha, 1000)
    delta = get_linspace_delta(theta_linspace)
    for r in r_linspace:

        # real_integrand = lambda theta: np.real(gu_psf_integrand(r, z, theta))
        # real_part = integrate.quad(real_integrand, 0, sample_divergence_angle)[
        #     0
        # ]
        # imag_integrand = lambda theta: np.imag(gu_psf_integrand(r, z, theta))
        # imag_part = integrate.quad(imag_integrand, 0, sample_divergence_angle)[
        #     0
        # ]
        # integral = real_part + 1j * imag_part
        # integrand = [gu_psf_integrand(r, z, theta) for theta in theta_linspace]
        integrand = gu_psf_integrand(r, z, theta_linspace)
        psf.append(riemann_sum(integrand, delta))

    return np.array(psf)


def calc_overlap(field_1, field_2, r_linspace):

    delta = get_linspace_delta(r_linspace)
    integrand = field_1 * field_2 * (2 * np.pi * r_linspace)
    overlap = np.abs(riemann_sum(integrand, delta)) ** 2
    return overlap


def calc_overlap_sweep():

    overlaps = []
    fiber_r_range = 10e-6
    num_points = 10000
    r_linspace = np.linspace(0, fiber_r_range, num_points)

    # Collection focal lengths
    # sweep_linspace = np.linspace(100e-6, 2e-3, 50)
    # fiber_field = np.exp(-2 * ((r_linspace / fiber_mfr) ** 2))
    # fiber_field = normalize_field(fiber_field, r_linspace)
    # for f in collection_focal_lengths:
    #     nv_field = calc_nv_field_at_fiber(fiber_r_range, num_points, f)
    #     integrand = nv_field * fiber_field * (2 * np.pi * r_linspace)
    #     overlaps.append(np.abs(riemann_sum(integrand, delta)) ** 2)

    # Fiber mode field radii
    nv_field = calc_nv_field_at_fiber(fiber_r_range, num_points)
    sweep_linspace = np.linspace(0.1e-6, 4.1e-6, 21)
    for mfr in sweep_linspace:
        fiber_field = np.exp(-((r_linspace / mfr) ** 2))
        fiber_field = normalize_field(fiber_field, r_linspace)
        overlap = calc_overlap(fiber_field, nv_field, r_linspace)
        overlaps.append(overlap)

    overlaps = np.array(overlaps)
    fig, ax = plt.subplots()
    ax.plot(sweep_linspace, overlaps)


def calc_nv_field_at_fiber(
    fiber_r_range=fiber_mfr * 3,
    num_points=1000,
    collection_focal_length=18e-3,
    do_plot=False,
):

    # We discard more than the aperture radius so start with that
    input_r_linspace = np.linspace(0, sample_aperture_radius, num_points)
    output_r_linspace = np.linspace(0, 5 * sample_aperture_radius, num_points)
    first_aperture_radius = sample_aperture_radius
    skip_sample_propagation = False
    norm_r_max = sample_aperture_radius
    free_space_distance = 0.1
    sample_telescope_1_f = 0.4

    if do_plot:
        fig, ax = plt.subplots()

    ###### Calculate fields leaving objective here (before aperture) ######

    # Point source
    # phase = np.exp(
    #     1j
    #     * k
    #     * (
    #         sample_focal_length
    #         + (input_r_linspace ** 2 / (2 * sample_focal_length))
    #     )
    # )
    # phase = 1
    # phase = np.exp(
    #     1j * k * np.sqrt(input_r_linspace ** 2 + sample_focal_length ** 2)
    # )
    # mag = 1 / np.sqrt(input_r_linspace ** 2 + sample_focal_length ** 2)
    # nv_field = phase * mag
    # nv_field = lens_phase_mask(nv_field, input_r_linspace, sample_focal_length)
    # nv_field = mag

    gaussian = False
    # gaussian = True

    # Collimated Gaussian beam coming out of objective (standard calculation,
    # focus is placed at objective aperture)
    if gaussian:
        first_aperture_radius = inch
        norm_r_max = None
        input_r_linspace = output_r_linspace
        nv_field = np.exp(-((input_r_linspace / sample_aperture_radius) ** 2))

    # # Plane wave
    # else:
    #     nv_field = np.array([1.0] * num_points)

    # Gu psf
    else:
        skip_sample_propagation = True
        spot_size = (wavelength / np.pi) * (
            sample_telescope_1_f / sample_aperture_radius
        )
        output_r_linspace = np.linspace(0, 5 * spot_size, num_points)
        alpha = np.arctan(
            sample_na * sample_focal_length / sample_telescope_1_f
        )
        nv_field = gu_psf(output_r_linspace, 0, alpha)

    ######

    # Sample objective aperture
    if not skip_sample_propagation:
        nv_field = normalize_field(
            nv_field, input_r_linspace, r_max=norm_r_max
        )
        nv_field = aperture_propagate(
            nv_field,
            input_r_linspace,
            output_r_linspace,
            sample_telescope_1_f,
            first_aperture_radius,
        )
    else:
        nv_field = normalize_field(
            nv_field, output_r_linspace, r_max=norm_r_max
        )

    # print(get_intensity_norm(nv_field, output_r_linspace))
    # # ax.plot(output_r_linspace, np.angle(nv_field))
    # # ax.plot(output_r_linspace, output_r_linspace * intensity(nv_field))
    # ax.plot(output_r_linspace, intensity(nv_field))
    # # test_field = np.exp(-((output_r_linspace / sample_aperture_radius) ** 2))
    # # test_field = normalize_field(
    # #     test_field, output_r_linspace, r_max=norm_r_max
    # # )
    # # ax.plot(output_r_linspace, output_r_linspace * intensity(test_field))
    # # ax.plot(output_r_linspace, intensity(test_field))
    # return

    # First sample telescope lens
    if not skip_sample_propagation:
        nv_intensity = intensity(nv_field)
        nv_field = lens_phase_mask(
            nv_field, input_r_linspace, sample_telescope_1_f
        )
        input_r_linspace = output_r_linspace
        output_r_linspace = input_r_linspace / 4
        nv_field = aperture_propagate(
            nv_field,
            input_r_linspace,
            output_r_linspace,
            0.5,
            inch,
        )
    else:
        input_r_linspace = output_r_linspace
        output_r_linspace = np.linspace(
            0, 5 * sample_aperture_radius / 4, num_points
        )
        nv_field = aperture_propagate(
            nv_field,
            input_r_linspace,
            output_r_linspace,
            0.1,
            inch,
        )

    # Second sample telescope lens
    input_r_linspace = output_r_linspace
    output_r_linspace = input_r_linspace
    nv_field = lens_phase_mask(nv_field, input_r_linspace, 0.1)
    nv_field = aperture_propagate(
        nv_field,
        input_r_linspace,
        output_r_linspace,
        free_space_distance,
        inch,
    )

    # print(get_intensity_norm(nv_field, output_r_linspace))
    # ax.plot(output_r_linspace, intensity(nv_field))
    # return

    # First collection telescope lens
    input_r_linspace = output_r_linspace
    output_r_linspace = input_r_linspace * 2
    nv_field = lens_phase_mask(nv_field, input_r_linspace, 0.050)
    nv_field = aperture_propagate(
        nv_field,
        input_r_linspace,
        output_r_linspace,
        0.2,
        inch / 2,
    )

    # Second collection telescope lens
    input_r_linspace = output_r_linspace
    output_r_linspace = input_r_linspace
    nv_field = lens_phase_mask(nv_field, input_r_linspace, 0.150)
    nv_field = aperture_propagate(
        nv_field,
        input_r_linspace,
        output_r_linspace,
        free_space_distance,
        inch / 2,
    )

    # ax.plot(output_r_linspace, np.angle(nv_field))
    # print(get_intensity_norm(nv_field, output_r_linspace))
    # # ax.plot(output_r_linspace, output_r_linspace * intensity(nv_field))
    # # width = sample_aperture_radius * 3 / 4
    # # test_field = np.exp(-((output_r_linspace / width) ** 2))
    # # test_field = normalize_field(
    # #     test_field, output_r_linspace, r_max=norm_r_max
    # # )
    # # ax.plot(output_r_linspace, output_r_linspace * intensity(test_field))
    # return

    # Collection objective
    input_r_linspace = output_r_linspace
    output_r_linspace = np.linspace(0, fiber_r_range, num_points)
    nv_field = lens_phase_mask(
        nv_field, input_r_linspace, collection_focal_length
    )
    nv_field = aperture_propagate(
        nv_field,
        input_r_linspace,
        output_r_linspace,
        collection_focal_length,
        inch / 2,
    )

    # print(get_intensity_norm(nv_field, output_r_linspace))
    # ax.plot(output_r_linspace, output_r_linspace * intensity(nv_field))
    # return

    if do_plot:
        nv_intensity = intensity(nv_field)
        ax.plot(
            output_r_linspace,
            output_r_linspace * nv_intensity,
            label="NV field",
        )
        fiber_field = np.exp(-((output_r_linspace / fiber_mfr) ** 2))
        fiber_field = normalize_field(fiber_field, output_r_linspace)
        ax.plot(
            output_r_linspace,
            output_r_linspace * intensity(fiber_field),
            label="Fiber mode",
        )
        ax.legend()
        overlap = calc_overlap(nv_field, fiber_field, output_r_linspace)
        print("Overlap: {}".format(overlap))

    return nv_field


if __name__ == "__main__":

    tool_belt.init_matplotlib()

    # plot_psf()
    calc_overlap_sweep()
    # calc_nv_field_at_fiber(
    #     fiber_r_range=2 * 3.5e-6,
    #     collection_focal_length=18e-3,
    #     do_plot=True,
    #     num_points=1000,
    # )

    plt.show(block=True)

    # x_vals = np.linspace(0, 3, 1000)
    # delta = (x_vals[-1] - x_vals[0]) / (len(x_vals) - 1)
    # integrand = np.exp(1j * x_vals)
    # print(riemann_sum(integrand, delta))

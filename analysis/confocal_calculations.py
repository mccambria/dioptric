# -*- coding: utf-8 -*-
"""
Various calculations from the note "Confocal Microscope Optimizations". 
Everything assumes cylindrical symmetry about the optical axis.

Created on November 15th, 2021

@author: mccambria
"""


from numpy.lib.type_check import imag
from utils import tool_belt as tb
from utils import kplotlib as kpl
import matplotlib.pyplot as plt
from numpy import pi
import numpy as np
import scipy.integrate as integrate
from scipy.special import jv as bessel_func
from multiprocessing import Pool

wavelength = 700e-9
k = 2 * pi / wavelength
# sample_focal_length = 2.87e-3
# sample_aperture_radius = 2.35e-3
# sample_na = 0.82
# sample_aperture_radius = sample_focal_length * sample_na
# sample_divergence_angle = np.arcsin(sample_na)
# # sample_aperture_radius = 1e-3
# # sample_aperture_radius = 10e-6
# # sample_aperture_radius = np.infty
# fiber_mfr = 2e-6
# inch = 25.4e-3
# air_index = 1
# diamond_index = 2.4


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
    phase_mask = np.exp(-1j * k * r**2 / (2 * f))
    return field * phase_mask


def aperture_propagate(input_field, input_r, output_r, z, aperture_rad):
    delta = get_linspace_delta(input_r)
    output_field = []

    # Apply the aperture
    trunc_inds = np.where(input_r <= aperture_rad)[0]
    input_r_trunc = input_r[trunc_inds]
    input_field_trunc = input_field[trunc_inds]

    coeff = k / z

    global fn

    def fn(val):
        phase = np.exp(1j * k * (val**2 + input_r_trunc**2) / (2 * z))
        bessel = bessel_func(0, -k * val * input_r_trunc / z)
        integrand = coeff * input_field_trunc * input_r_trunc * phase * bessel
        return riemann_sum(integrand, delta)

    with Pool() as p:
        output_field = p.map(fn, output_r)

    # for val in output_r:
    #     phase = np.exp(1j * k * (val**2 + input_r_trunc**2) / (2 * z))
    #     bessel = bessel_func(0, -k * val * input_r_trunc / z)
    #     integrand = coeff * input_field_trunc * input_r_trunc * phase * bessel
    #     output_field.append(riemann_sum(integrand, delta))
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


def widefield_oof():
    radius = 2e-3
    num_points = int(1e4)
    focal_length = 2e-3
    # wavelength = 700e-9
    # wavenumber = 2 * np.pi / wavelength
    aperture = 2.0e-3

    # Initial field at objective from NV
    r_linspace = np.linspace(0, radius, num_points)
    distances = np.sqrt(r_linspace**2 + (1.0 * focal_length) ** 2)
    field = np.exp(1j * k * distances) / distances
    # field = np.where(r_linspace < aperture, field, 0)

    # field = np.array([1] * num_points)
    # field = aperture_propagate(field, r_linspace, r_linspace, 1000e-3, 1e-3)

    # Past objective
    # field = lens_phase_mask(field, r_linspace, focal_length)
    # # field = aperture_propagate(field, r_linspace, r_linspace, focal_length, aperture)
    # field = aperture_propagate(field, r_linspace, r_linspace, 200e-3, aperture)

    field = gu_psf(r_linspace, 0, np.arctan(aperture / 200e-3))
    # field += gu_psf(r_linspace, 100 * 100e-6, np.arctan(aperture / 200e-3))

    field = np.where(r_linspace < 100 * 200e-9, field, 0)

    field = aperture_propagate(field, r_linspace, r_linspace, 50e-3, 12e-3)
    field = lens_phase_mask(field, r_linspace, 50e-3)
    field = aperture_propagate(field, r_linspace, r_linspace, 50e-3, 12e-3)

    # Fourier plane
    # field = np.where(r_linspace > 0.0004, field, 0)

    field = aperture_propagate(field, r_linspace, r_linspace, 150e-3, 12e-3)
    field = lens_phase_mask(field, r_linspace, 150e-3)
    field = aperture_propagate(field, r_linspace, r_linspace, 150e-3, 12e-3)

    fig, ax = plt.subplots()
    kpl.plot_line(ax, r_linspace, intensity(field))
    # kpl.plot_line(ax, r_linspace, np.real(field))
    # kpl.plot_line(ax, r_linspace, distances)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # plot_psf()
    # calc_overlap_sweep()
    # calc_nv_field_at_fiber(
    #     collection_telescope_1_f=60e-3,  # 13.86e-3,
    #     collection_telescope_2_f=250e-3,
    #     # collection_focal_length=4.5e-3,
    #     # collection_to_fiber_distance=4.5e-3,
    #     do_plot=True,
    # )

    widefield_oof()

    plt.show(block=True)

    # x_vals = np.linspace(0, 3, 1000)
    # delta = (x_vals[-1] - x_vals[0]) / (len(x_vals) - 1)
    # integrand = np.exp(1j * x_vals)
    # print(riemann_sum(integrand, delta))

    # nv_to_objective_efficiency()

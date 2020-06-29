# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:04:19 2019

@author: matth
"""


# %% Imports


import csv
import numpy
from numpy import matmul
from numpy import exp
from numpy import conj
import matplotlib
import matplotlib.pyplot as plt
import analysis.extract_hamiltonian as eh
from analysis.extract_hamiltonian import conj_trans
from analysis.extract_hamiltonian import calc_eig_static_cartesian_B
from analysis.extract_hamiltonian import calc_splitting
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
import matplotlib.gridspec as gridspec
from scipy.integrate import quad


# %% Constants


# Plotting
# ms = 7
# lw = 1.75
ms = 5.25
lw = 5.25/4

from numpy import pi
from scipy.constants import Boltzmann
from scipy.constants import hbar
from scipy.constants import Planck
from analysis.extract_hamiltonian import d_gs  # in GHz

static_B_z= 0.01  # GHz

im = 0+1j
inv_sqrt_2 = 1/numpy.sqrt(2)
gmuB = 2.8e-3  # gyromagnetic ratio in GHz / G

temp = 295#*(2**(1/3))
kT = Boltzmann*temp  # measurement thermal energy

Omega = (3.567e-10)**3  # unit cell volume in diamond
# Omega = (10e-10)**3  # unit cell volume in diamond
v_s = 1.2e4  # speed of sound in diamond
omega_D = 2*pi*38.76e12  # Debye angular frequency in diamond
# v_s = 1.8e4  # speed of sound in diamond
# omega_D = 39.39e13  # Debye angular frequency in diamond
x_D = (hbar*omega_D) / kT  # dimensionless phonon energy limit

x_0 = (Planck * 10**9 * d_gs) / kT  # dimensionless zfs energy

rate_coeff = (Omega**2 * kT**5) / (2 * pi**3 * v_s**6 * omega_D**2 * hbar**9)
# This rate coefficient absorbs (2*pi*hbar)**4 from the matrix elements
# rate_coeff = (8 * pi * Omega**2 * kT**5) / (v_s**6 * hbar**5 * omega_D**2)  # /h**4

# Eigenvectors and values are sorted in increasing order of eigenvalue.
# Assume B is small enough such that there are no crossings and theta <= pi/2.
state_mapping = {-1: 1, 0: 0, +1: 2}


# %% Phonon fitting


def phonon_fit(nv_data, p0=None):
    """
    Fits gamma and omega as functions of perp_B to a model of relaxation
    limited by two-phonon processes.
    """

    # %% Get the rates and the values of perp_B
    
    all_omega = []
    all_omega_err = []
    all_gamma = []
    all_gamma_err = []
    all_mag_B = []
    all_theta_B = []
    all_perp_B = []

    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        name = nv['name']
        # if name in ['NVA1', 'NVA2']:
        #     continue
        # if name != 'test':
        #     continue
        
        omega = numpy.array(nv['omega'])
        omega_err = numpy.array(nv['omega_err'])
        gamma = numpy.array(nv['gamma'])
        gamma_err = numpy.array(nv['gamma_err'])
        mag_B = numpy.array(nv['mag_B'])
        theta_B = numpy.array(nv['theta_B'])
        perp_B = numpy.array(nv['perp_B'])
        
        # Only consider points with known B field components
        # mask = mag_B != None
        
        # Only consider points with known perp_B < 80
        mask = []
        for el in perp_B:
            if el is not None:
                if el < 80:
                    mask.append(True)
                    continue
            mask.append(False)
        
        
        # Calculate based on all measurements with components, including
        # those off axis
        all_omega.extend(omega[mask])
        all_omega_err.extend(omega_err[mask])
        all_gamma.extend(gamma[mask])
        all_gamma_err.extend(gamma_err[mask])
        all_mag_B.extend(mag_B[mask])
        all_theta_B.extend(theta_B[mask])
        all_perp_B.extend(perp_B[mask])
    
    # Cast to arrays
    all_omega = numpy.array(all_omega)
    all_omega_err = numpy.array(all_omega_err)
    all_gamma = numpy.array(all_gamma)
    all_gamma_err = numpy.array(all_gamma_err)
    all_mag_B = numpy.array(all_mag_B)
    all_theta_B = numpy.array(all_theta_B)
    all_perp_B = numpy.array(all_perp_B)
    
    # %% Setup for d_perp_prime and the E field magnitude fits
    
    # gamma = lambda perp_B, mag_E, d_perp_prime: rate_kHz_electric_perp_B(-1, +1, perp_B, mag_E, d_perp_prime)
    # omega = lambda perp_B, mag_E, d_perp_prime: rate_kHz_electric_perp_B(0, +1, perp_B, mag_E, d_perp_prime)
    
    # %% Level 1
    # Fit gamma as a function of B_perp
    # to d_perp_prime and the E field magnitude
    
    # p0 = (85, 17)
    # popt, pcov = curve_fit(gamma, all_perp_B, all_gamma, p0=p0,
    #                         sigma=all_gamma_err, absolute_sigma=True)
    # print(popt)
    # print(numpy.sqrt(numpy.diag(pcov)))
    
    # %% Level 2
    # Fit both gamma and omega as functions of B_perp
    # to d_perp_prime and the E field magnitude
    
    # We can trick curve_fit into doing a fit to a double-valued function by
    # using the sign of the x argument to tell us which rate to calculate - 
    # plus for gamma, minus for omega. To distinguish 0, we'll add 1 before
    # inverting for negative rates, so perp_B < 0 is for omega
    
    # p0 = (85, 17)
    # double_valued_perp_B = numpy.append(all_perp_B, -(all_perp_B+1))
    # double_valued_rates = numpy.append(all_gamma, all_omega)
    # double_valued_rates_err = numpy.append(all_gamma_err, all_omega_err)
    # popt, pcov = curve_fit(double_valued_electric_perp_B,
    #                         double_valued_perp_B, double_valued_rates, p0=p0,
    #                         sigma=double_valued_rates_err, absolute_sigma=True)
    # print(popt)
    # print(numpy.sqrt(numpy.diag(pcov)))
    
    
    # %% Plot for d_perp_prime and the E field magnitude fits
    
    # fig, axes_pack = plt.subplots(1, 2, figsize=(10,5.0))
    # fig.set_tight_layout(True)
    
    # ax = axes_pack[0]
    # ax.errorbar(all_perp_B, all_gamma, yerr=all_gamma_err,
    #             linestyle='None', ms=10)
    # perp_B_linspace = numpy.linspace(0, max(all_perp_B))
    # gammas = gamma(perp_B_linspace, *popt)
    # ax.plot(perp_B_linspace, gammas)
    
    # ax = axes_pack[1]
    # ax.errorbar(all_perp_B, all_omega, yerr=all_omega_err,
    #             linestyle='None', ms=10)
    # perp_B_linspace = numpy.linspace(0, max(all_perp_B))
    # omegas = omega(perp_B_linspace, *popt)
    # ax.plot(perp_B_linspace, omegas)
    
    
    # %% Phonon fit
    
    # gamma = lambda perp_B, popt: rate_kHz_phonon_perp_B(-1, +1, perp_B, *popt)
    # omega = lambda perp_B, popt: rate_kHz_phonon_perp_B(0, +1, perp_B, *popt)
    # if p0 is None:
    #     # p0 = [1200]*5
    #     p0 = [476, 1333, 1016, 937, 1813]
    # # popt = p0
    # popt = [476.999999999999, 1333, 1016, 937, 1813]
    # double_valued_func = double_valued_phonon_perp_B
    
    
    
    # Simplified
    gamma = lambda perp_B, popt: rate_kHz_phonon_perp_B_simp2(-1, +1, perp_B, *popt)
    omega = lambda perp_B, popt: rate_kHz_phonon_perp_B_simp2(0, +1, perp_B, *popt)
    if p0 is None:
        # p0 = [500,1300]
        p0 = [1150,1150]
    # popt = None
    popt = p0
    double_valued_func = double_valued_phonon_perp_B_simp2
    
    
    double_valued_perp_B = numpy.append(all_perp_B, -(all_perp_B+1))
    double_valued_rates = numpy.append(all_gamma, all_omega)
    double_valued_rates_err = numpy.append(all_gamma_err, all_omega_err)
    
    if popt is None:
        popt, pcov = curve_fit(double_valued_func,
                        double_valued_perp_B, double_valued_rates, p0=p0,
                        sigma=double_valued_rates_err, absolute_sigma=True)
        # popt_hz = numpy.array(popt) / Planck
        # print('popt_hz: {}'.format(popt_hz))
        print(popt)
        print(numpy.sqrt(numpy.diag(pcov)))
    
    fig, axes_pack = plt.subplots(1, 2, figsize=(10,7.0))
    fig.set_tight_layout(True)
    round_popt = [round(el) for el in popt]
    fig.suptitle('p0={}\npopt={}'.format(p0, round_popt), x=0.21, y=0.95)
    
    ax = axes_pack[0]
    ax.errorbar(all_perp_B, all_gamma, yerr=all_gamma_err,
                linestyle='None', ms=10)
    perp_B_linspace = numpy.linspace(0, max(all_perp_B))
    gammas = gamma(perp_B_linspace, popt)
    ax.plot(perp_B_linspace, gammas)
    ax.set_xlabel('B perp (G)')
    ax.set_ylabel('gamma (kHz)')
    
    ax = axes_pack[1]
    ax.errorbar(all_perp_B, all_omega, yerr=all_omega_err,
                linestyle='None', ms=10)
    perp_B_linspace = numpy.linspace(0, max(all_perp_B))
    omegas = omega(perp_B_linspace, popt)
    ax.plot(perp_B_linspace, omegas)
    ax.set_xlabel('B perp (G)')
    ax.set_ylabel('omega (kHz)')
    
    # file_name = str(p0)
    # file_name = file_name.replace(', ', '_')
    # blank_file_path = 'C:/Users/matth/Desktop/lab/bulk_dq_relaxation/anisotropy/{}.svg'
    # fig.savefig(blank_file_path.format(file_name[1:-1]))
    # plt.close(fig)
    
    
def double_valued_phonon_perp_B_simp2(perp_B, lambda_perp_prime, lambda_perp):
    
    noise_hamiltonian = calc_phonon_hamiltonian(0, 
                                        lambda_perp_prime, lambda_perp_prime, 
                                        lambda_perp, lambda_perp)
    return double_valued_noise_hamiltonian_perp_B(perp_B, noise_hamiltonian)
    
    
def double_valued_phonon_perp_B_simp3(perp_B, lambda_z,
                                      lambda_perp_prime, lambda_perp):
    
    noise_hamiltonian = calc_phonon_hamiltonian(lambda_z, 
                                        lambda_perp_prime, lambda_perp_prime,
                                        lambda_perp, lambda_perp)
    return double_valued_noise_hamiltonian_perp_B(perp_B, noise_hamiltonian)
    
    
def double_valued_phonon_perp_B(perp_B, lambda_z, lambda_xz,
                                lambda_yz, lambda_yx, lambda_xy):
    
    noise_hamiltonian = calc_phonon_hamiltonian(lambda_z, lambda_xz,
                                            lambda_yz, lambda_yx, lambda_xy)
    return double_valued_noise_hamiltonian_perp_B(perp_B, noise_hamiltonian)
    

def double_valued_electric_perp_B(perp_B, mag_E, d_perp_prime):
    
    noise_hamiltonian = calc_electric_hamiltonian([mag_E]*3, d_perp_prime)
    return double_valued_noise_hamiltonian_perp_B(perp_B, noise_hamiltonian)
        
    
def double_valued_noise_hamiltonian_perp_B(perp_B, noise_hamiltonian):
    
    gamma = lambda B_field_vec: rate(-1, +1, B_field_vec, noise_hamiltonian)/1000
    omega = lambda B_field_vec: rate(0, +1, B_field_vec, noise_hamiltonian)/1000
    
    if (type(perp_B) is list) or (type(perp_B) is numpy.ndarray):
        rates = []
        for val in perp_B:
            if val < 0:
                B_field_vec = [gmuB*-(val+1), 0.0, static_B_z]
                rates.append(omega(B_field_vec))
            else:
                B_field_vec = [gmuB*val, 0.0, static_B_z]
                rates.append(gamma(B_field_vec))
        return numpy.array(rates)
    else:
        if perp_B < 0:
            B_field_vec = [gmuB*-(perp_B+1), 0.0, static_B_z]
            return omega(B_field_vec)
        else:
            B_field_vec = [gmuB*perp_B, 0.0, static_B_z]
            return gamma(B_field_vec)
    
    
def rate_kHz_phonon_perp_B_simp2(i, j, perp_B, lambda_perp_prime, lambda_perp):
    
    noise_hamiltonian = calc_phonon_hamiltonian(0, 
                                        lambda_perp_prime, lambda_perp_prime, 
                                        lambda_perp, lambda_perp)
    return rate_kHz_noise_hamiltonian_perp_B(i, j, perp_B, noise_hamiltonian)
    
    
def rate_kHz_phonon_perp_B(i, j, perp_B, lambda_z, lambda_xz,
                           lambda_yz, lambda_yx, lambda_xy):
    
    noise_hamiltonian = calc_phonon_hamiltonian(lambda_z, lambda_xz,
                                            lambda_yz, lambda_yx, lambda_xy)
    return rate_kHz_noise_hamiltonian_perp_B(i, j, perp_B, noise_hamiltonian)
    
    
def rate_kHz_electric_perp_B(i, j, perp_B, mag_E, d_perp_prime):
    
    noise_hamiltonian = calc_electric_hamiltonian([mag_E]*3, d_perp_prime)
    return rate_kHz_noise_hamiltonian_perp_B(i, j, perp_B, noise_hamiltonian)
    
    
def rate_kHz_noise_hamiltonian_perp_B(i, j, perp_B, noise_hamiltonian):
    """
    Calculate rate (in kHz) as a function of perp_B, a fixed axial 
    B component so we can plot in 1D
    """
    
    rate_kHz = lambda B_field_vec: rate(i, j, B_field_vec, noise_hamiltonian)/1000
    if (type(perp_B) is list) or (type(perp_B) is numpy.ndarray):
        rates = []
        for val in perp_B:
            B_field_vec = [gmuB*val, 0.0, static_B_z]
            rates.append(rate_kHz(B_field_vec))
        return numpy.array(rates)
    else:
        B_field_vec = [gmuB*perp_B, 0.0, static_B_z]
        return rate_kHz(B_field_vec)
    
    
def distr(x):
    """Bose Einstein distribution"""
    return 1.0 / (exp(x) - 1.0)
    
    
def f_ij(x, x_ji):
    # For x=0 or diff=0, we get a blowup, naively at least. If we're a little
    # clever we can show that the limit at x=0 is in fact 0. 
    diff = x - x_ji
    # if (x == 0.0) or (diff == 0.0):
    #     return 0.0
    val = (x * diff)**3 * distr(x) * (distr(diff) + 1)
    # val[numpy.isnan(val)] = 0.0
    return val


def g_ijm(x, x_im, x_jm):
    return (1 / (x_im + x)) + (1 / (x_jm - x))
    
    
def int_arg(x, i, j, vecs, vals, noise_hamiltonian):
    
    i_vec = vecs[state_mapping[i]]
    i_val = vals[state_mapping[i]]
    
    j_vec = vecs[state_mapping[j]]
    j_val = vals[state_mapping[j]]

    x_ji = (Planck*(j_val-i_val)) / kT
    f_val = f_ij(x, x_ji)
    
    sum_val = 0
    for m in [-1, 0, +1]:
        
        m_ind = state_mapping[m]
        m_vec = vecs[m_ind]
        m_val = vals[m_ind]
        
        H_jm = numpy.matmul(noise_hamiltonian, m_vec)
        H_jm = numpy.matmul(conj_trans(j_vec), H_jm)
        H_jm *= 10**9
        
        H_mi = numpy.matmul(noise_hamiltonian, i_vec)
        H_mi = numpy.matmul(conj_trans(m_vec), H_mi)
        H_mi *= 10**9
        
        if i == m:
            x_im = 0.0
        else:
            x_im = (Planck*(i_val-m_val)) / kT
        
        if j == m:
            x_jm = 0.0
        else:
            x_jm = (Planck*(j_val-m_val)) / kT
        
        g_val = g_ijm(x, x_im, x_jm)
        
        sum_val += (H_jm * H_mi * g_val)
        
    return numpy.real(f_val * conj(sum_val) * sum_val)


def int_arg_fast(x, i, j, x_ij_mat, noise_H_mat):
    """
    This is a sped up version of int_arg. There's a lot of duplicated
    calculation in that function which can be done once elsewhere and passed
    in as a lookup matrix.
    """
    
    i_ind = state_mapping[i]
    j_ind = state_mapping[j]
    
    x_ji = x_ij_mat[j_ind, i_ind]
    f_val = f_ij(x, x_ji)
    
    sum_val = 0
    for m in [-1, 0, +1]:
        
        m_ind = state_mapping[m]
        
        H_jm = noise_H_mat[j_ind, m_ind]
        H_mi = noise_H_mat[m_ind, i_ind]
        
        x_im = x_ij_mat[i_ind, m_ind]
        x_jm = x_ij_mat[j_ind, m_ind]
        g_val = g_ijm(x, x_im, x_jm)
        
        sum_val += (H_jm * H_mi * g_val)
        
    return numpy.real(f_val * conj(sum_val) * sum_val)
    
    
def rate(i, j, B_field_vec, noise_hamiltonian, fast=True):
    """
    Calculate the two-phonon rate between states i and j. i and j are
    designated by m_s as integers.
    """
    
    # Fast mode does some of the int_arg processing right here so that it
    # isn't duplicated in each call to the argument.
    if fast:
        
        # B magnitude is accepted as gmuB*mag_B, everything in GHz
        vecs, vals = calc_eig_static_cartesian_B(*B_field_vec)
        
        # Construct matrices so we can just look up necessary values
        x_ij_mat = numpy.empty((3, 3), dtype=float)
        noise_H_mat = numpy.empty((3, 3), dtype=complex)
        for a in [-1, 0, +1]:
            a_ind = state_mapping[a]
            a_val = vals[a_ind]
            a_vec = vecs[a_ind]
            for b in [-1, 0, +1]:
                b_ind = state_mapping[b]
                b_val = vals[b_ind]
                b_vec = vecs[b_ind]
                if a_ind == b_ind:
                    x_ij_mat[a_ind, b_ind] = 0.0
                else:
                    x_ij_mat[a_ind, b_ind] = a_val-b_val
                el = numpy.matmul(noise_hamiltonian, b_vec)
                el = numpy.matmul(conj_trans(a_vec), el)
                noise_H_mat[a_ind, b_ind] = el
        x_ij_mat *= (10**9*Planck / kT)  # Convert from GHz to dimensionless
        noise_H_mat *= (10**9)  # Convert from GHz to Hz
        
        args = (i, j, x_ij_mat, noise_H_mat)
        int_val, int_error = quad(int_arg_fast, 0.001, x_D, args=args)
        
        return rate_coeff * int_val
    
    else:
        
        # B magnitude is accepted as gmuB*mag_B, everything in GHz
        vecs, vals = calc_eig_static_cartesian_B(*B_field_vec)
        vals *= 10**9  # Convert to Hz
        
        args = (i, j, vecs, vals, noise_hamiltonian)
        diffs = [vals[2]-vals[1], vals[2]-vals[0], vals[1]-vals[0]]
        diffs = numpy.array(diffs)
        diffs *= (Planck / kT)
        points = [0.0, *diffs]
        int_val, int_error = quad(int_arg, 0.001, x_D, args=args, points=points)
        
        
        # x_vals = numpy.linspace(0.0,6.5,100)
        # plt.plot(x_vals, int_arg(x_vals, *args))
        # print(int_arg(numpy.array([0.0, 0.000001]), *args))
        # return
        
        return rate_coeff * int_val


def calc_phonon_hamiltonian(lambda_z, lambda_xz, lambda_yz, 
                            lambda_yx, lambda_xy):
    """
    Calculate a noise Hamiltonian based on the 5 phonon coupling constants
    """
    
    sq = inv_sqrt_2*(lambda_xz-im*lambda_yz)
    dq = -lambda_yx-im*lambda_xy
    
    hamiltonian = numpy.array([[lambda_z, sq, dq],
                               [numpy.conj(sq), 0.0, -sq],
                               [numpy.conj(dq), numpy.conj(-sq), lambda_z]])
    return hamiltonian
    


def calc_electric_hamiltonian(E_field_vec, d_perp_prime=17):
    
    d_parallel = 0.35
    d_perp = 17
    
    E_x, E_y, E_z = E_field_vec
    
    diag = d_parallel*E_z
    sq = inv_sqrt_2*d_perp_prime*(E_x-im*E_y)
    dq = -d_perp*(E_x+im*E_y)
    
    hamiltonian = numpy.array([[diag, sq, dq],
                               [numpy.conj(sq), 0.0, -sq],
                               [numpy.conj(dq), numpy.conj(-sq), diag]])
    
    return hamiltonian
    

# %% Functions


def get_coupling_constants(nv_data):
    """
    Calculate the values of the coupling constants based on the B_perp=0
    measured rates gamma and Omega
    """
    
    
    # %% Get B_perp=0 points
    
    
    on_axis_omega = []
    on_axis_omega_err = []
    on_axis_gamma = []
    on_axis_gamma_err = []

    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        # name = nv['name']
        # if name in ['NVA1', 'NVA2']:
        #     continue
        # if name != 'test':
        #     continue
        
        omega = numpy.array(nv['omega'])
        omega_err = numpy.array(nv['omega_err'])
        gamma = numpy.array(nv['gamma'])
        gamma_err = numpy.array(nv['gamma_err'])
        # mag_B = numpy.array(nv['mag_B'])
        # theta_B = numpy.array(nv['theta_B'])
        perp_B = numpy.array(nv['perp_B'])
        
        # Only consider points with known B field components
        # mask = mag_B != None
        
        # Only consider points with known perp_B < 1
        mask = []
        for el in perp_B:
            if el is not None:
                if el < 1:
                    mask.append(True)
                    continue
            mask.append(False)
        
        on_axis_omega.extend(omega[mask])
        on_axis_omega_err.extend(omega_err[mask])
        on_axis_gamma.extend(gamma[mask])
        on_axis_gamma_err.extend(gamma_err[mask])
    
    # Cast to arrays
    on_axis_omega = numpy.array(on_axis_omega)
    on_axis_omega_err = numpy.array(on_axis_omega_err)
    on_axis_gamma = numpy.array(on_axis_gamma)
    on_axis_gamma_err = numpy.array(on_axis_gamma_err)
    
    num_points = len(on_axis_omega)
    
    # Work in Hz
    avg_on_axis_omega = 1000*numpy.average(on_axis_omega)
    avg_on_axis_omega_err = 1000*numpy.sqrt(numpy.sum(on_axis_omega_err**2))/num_points
    avg_on_axis_gamma = 1000*numpy.average(on_axis_gamma)
    avg_on_axis_gamma_err = 1000*numpy.sqrt(numpy.sum(on_axis_gamma_err**2))/num_points
    
    # print(num_points)
    # print(avg_on_axis_omega)
    # print(avg_on_axis_omega_err)
    # print(avg_on_axis_gamma)
    # print(avg_on_axis_gamma_err)
    
    
    # %% gamma and lambda_perp_prime
    
    # print(x_D)
    int_val, int_error = quad(on_axis_integrand, 0, x_D)
    # print(int_val)
    
    # Calculate the coefficient of lambda_perp_prime^4 in the expression
    # for gamma
    gamma_coeff = 4 * rate_coeff * x_0**2 * int_val
    lambda_perp_prime = (avg_on_axis_gamma / gamma_coeff)**(1/4)
    lambda_perp_prime_err = (lambda_perp_prime * (1/4) * avg_on_axis_gamma_err / avg_on_axis_gamma)
    
    # In J
    print('lambda_perp_prime = {} +/- {} J\n'.format(lambda_perp_prime, lambda_perp_prime_err))
    
    # In GHz
    lambda_perp_prime_GHz = lambda_perp_prime/(Planck*10**9)
    lambda_perp_prime_err_GHz = lambda_perp_prime_err/(Planck*10**9)
    print('lambda_perp_prime = {} +/- {} GHz\n'.format(lambda_perp_prime_GHz, lambda_perp_prime_err_GHz))
    
    
    # %% Omega and lambda_perp
    
    lambda_perp = lambda_perp_prime
    lambda_perp_err = lambda_perp_prime_err
    lambda_perp_GHz = lambda_perp_prime_GHz
    lambda_perp_err_GHz = lambda_perp_prime_err_GHz
    
    # # root_scalar method
    
    # # Work in GHz so the root_scalar tolerances are not too large relatively
    # omega_coeff = (1/2) * rate_coeff * x_0**2 * int_val
    # lambda_func = lambda lambda_z_GHz: (Planck*10**9)**4 * lambda_perp_prime_GHz**2 * (lambda_z_GHz**2 + (lambda_z_GHz - 2*lambda_perp_GHz)**2)
    # root_func = lambda lambda_z_GHz: avg_on_axis_omega - (omega_coeff * lambda_func(lambda_z_GHz))
    # sol = root_scalar(root_func, x0=lambda_perp_GHz/100, x1=lambda_perp_GHz/10)
    # lambda_z_GHz = sol.root
    
    # # In J
    # # print('lambda_z = {} J\n'.format(lambda_z))
    
    # # In GHz
    # # lambda_z_GHz = lambda_z/(Planck*10**9)
    # print('lambda_z = {} GHz\n'.format(lambda_z_GHz))
    
    # quadratic formula method
    
    omega_coeff = rate_coeff * x_0**2 * int_val
    a = 1
    b = -2*lambda_perp
    c = 2*lambda_perp**2 - avg_on_axis_omega/(omega_coeff*lambda_perp_prime**2)
    
    # Take the smaller root
    quadratic_formula = lambda a, b, c: (1/(2*a)) * (-b - numpy.sqrt(b**2 - 4*a*c))
    lambda_z = (1/(2*a)) * (-b - numpy.sqrt(b**2 - 4*a*c))
    
    # Propagate the error
    err_b = 2*lambda_perp_err
    err_c = (4*lambda_perp*lambda_perp_err)**2
    err_c += ((2*avg_on_axis_omega/(omega_coeff*lambda_perp_prime**3))*lambda_perp_prime_err)**2
    err_c += ((1/(omega_coeff*lambda_perp_prime**2))*avg_on_axis_omega_err)**2
    err_c = numpy.sqrt(err_c)
    
    delf_delb = (1/(2*a)) * (-1 - b * (b**2 - 4*a*c)**(-1/2))
    delf_delc = (c/a) * (b**2 - 4*a*c)**(-1/2)
    
    lambda_z_err = numpy.sqrt((delf_delb*err_b)**2 + (delf_delc*err_c)**2)
    
    # In J
    print('lambda_z = {} +/- {} J\n'.format(lambda_z, lambda_z_err))
    
    # In GHz
    lambda_z_GHz = lambda_z/(Planck*10**9)
    lambda_z_err_GHz = lambda_z_err/(Planck*10**9)
    print('lambda_z = {} +/- {} GHz\n'.format(lambda_z_GHz, lambda_z_err_GHz))
    

def on_axis_integrand(x):
    
    # Check if we have a list or array
    try:
        _ = len(x)
        x_is_multi_valued = True
    except Exception:
        x_is_multi_valued = False
        
    if x_is_multi_valued:
        ret_vals = []
        for val in x:
            if x == 0:
                ret_vals.append(1)
            else:
                result = x**2 * bose(x) * (bose(x)+1)
                ret_vals.append(result)
        ret_vals = numpy.array(ret_vals)
        return ret_vals
    else:
        if x == 0:
            return 1
        else:
            return x**2 * bose(x) * (bose(x)+1)
    
    
def bose(x):
    return (exp(x)-1)**-1


def plot_fx():
    
    fx = lambda x: x**2 * bose(x) * (bose(x)+1)
    print(x_D)
    fig, ax = plt.subplots()
    linspace_x = numpy.linspace(0.1, 2*x_D, 1000)
    ax.plot(linspace_x, fx(linspace_x))


def get_nv_data_csv(file):
    """
    Parses a csv into a list of dictionaries for each NV. Assumes the data 
    for an NV is grouped together, ie there are no gaps.
    """
    
    # Marker and color combination to distinguish NVs
    # Colors are from the Wong colorblind-safe palette
    marker_ind = 0
    markers = ['^', 'X', 'o', 's', 'D']
    colors = ['#009E73', '#E69F00', '#0072B2', '#CC79A7', '#D55E00',]
    markerfacecolors = ['#ACECDB', '#f5b11d', '#72b5db', '#f0a3cd', '#fcbd8b',]
    
    nv_data = []
    header = True
    current_name = None
    nv = None
    with open(file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Create columns from the header (first row)
            if header:
                columns = row[1:]
                header = False
                continue
            # Set up for a new NV if we're in a new block
            if row[0] != current_name: 
                # Append the current NV if there is one
                if current_name is not None:
                    nv_data.append(nv)
                current_name = row[0]
                nv = {}
                nv['name'] = current_name
                nv['marker'] = markers[marker_ind]
                nv['color'] = colors[marker_ind]
                nv['markerfacecolor'] = markerfacecolors[marker_ind]
                marker_ind += 1
                # Initialize a new list for each column
                for column in columns:
                    nv[column] = []
            for ind in range(len(columns)):
                column = columns[ind]
                val = row[ind+1]
                if val == 'None':
                    val = None
                else:
                    val = float(val)
                nv[column].append(val)
    # Don't forget the last NV!
    nv_data.append(nv)
            
    return nv_data


def weighted_corrcoeff(x, y, errors=None):
    """
    Returns Pearson correlation coefficient for dependent variable y and
    independent variable x. Optionally weighted by squared errors on y 
    (variance weights).
    """
    
    # Create a mask for elements that are None in neither array
    x_mask = x != None
    y_mask = y != None
    mask = x_mask * y_mask
    
    if errors is not None:
        cov_mat = numpy.cov(x[mask].astype(float), y[mask].astype(float),
                            aweights=errors[mask]**-2)
    else:
        cov_mat = numpy.cov(x[mask].astype(float), y[mask].astype(float))
    
    return cov_mat[0,1] / numpy.sqrt(cov_mat[0,0]*cov_mat[1,1])
    return


def linear(x, m, b):
    return m*x + b


def linear_prop(B_linspace, pcov):
    
    # Gradient as a column vector
    grad = numpy.array([[B_linspace], [1]])
    grad_T = numpy.transpose(grad)
    
    squared_errs = matmul(grad_T, matmul(pcov, grad)).flatten()[0]
    return numpy.sqrt(squared_errs)


def conf_int(ax, B_linspace, popt, pcov):
    
    # Propagation of error
    fit = linear(B_linspace, *popt)
    err = linear_prop(B_linspace, pcov)
    
    lin_color = '#009E73'
    fill_color = '#ACECDB'
    pste = numpy.sqrt(numpy.diag(pcov))
    ax.plot(B_linspace, linear(B_linspace, *popt), c=lin_color)
    ax.fill_between(B_linspace, fit - 2*err, fit + 2*err, color=fill_color)
    print('{}\n{}\n'.format(popt, pste))
    
    
def correlations(nv_data):
    """
    Return Pearson product-moment correlation coefficients for various
    combinations of measured quantities.
    """
    
    # mode = 'all'
    # mode = 'single nv'
    mode = 'ensemble'
    
    columns = ['res_minus', 'res_plus', 'splitting', 'mag_B',
                'theta_B', 'perp_B', 'contrast_minus', 'contrast_plus', 
                'rabi_minus', 'rabi_plus', 'gamma', 'omega', 'ratio', ]
    error_columns = ['gamma', 'omega', 'ratio']
    
    res_minus = []
    res_plus = []
    splitting = []
    mag_B = []
    theta_B = []
    perp_B = []
    perp_B_frac = []
    
    comp_minus = []  # |<Sz;-1|H;-1>|**2
    comp_plus = []  # |<Sz;+1|H;+1>|**2
    
    contrast_minus = []
    contrast_plus = []
    rabi_minus = []
    rabi_plus = []
    gamma = []
    gamma_error = []
    omega = []
    omega_error = []
    ratio = []
    ratio_error = []
    
    if mode == 'all':
        inclusion_check = lambda name: True
    elif mode == 'single nv':
        inclusion_check = lambda name: name not in ['NVE']
    elif mode == 'ensemble':
        inclusion_check = lambda name: name == 'NVE'
    
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        name = nv['name']
        if not inclusion_check(name):
            continue
        
        res_minus.extend(nv['res_minus'])
        res_plus.extend(nv['res_plus'])
        splitting.extend(nv['splitting'])
        mag_B.extend(nv['mag_B'])
        theta_B.extend(nv['theta_B'])
        perp_B.extend(nv['perp_B'])
        
        nv_perp_B = nv['perp_B']
        nv_mag_B = nv['mag_B']
        nv_perp_B_frac = []
        for ind in range(len(nv_perp_B)):
            if nv_perp_B[ind] is None:
               nv_perp_B_frac.append(None)
            else:
               nv_perp_B_frac.append(nv_perp_B[ind] / nv_mag_B[ind])
        perp_B_frac.extend(nv_perp_B_frac)
        
        contrast_minus.extend(nv['contrast_minus'])
        contrast_plus.extend(nv['contrast_plus'])
        rabi_minus.extend(nv['rabi_minus'])
        rabi_plus.extend(nv['rabi_plus'])
        gamma.extend(nv['gamma'])
        gamma_error.extend(nv['gamma_error'])
        omega.extend(nv['omega'])
        omega_error.extend(nv['omega_error'])
        ratio.extend(nv['ratio'])
        ratio_error.extend(nv['ratio_error'])
        
    res_minus = numpy.array(res_minus)
    res_plus = numpy.array(res_plus)
    splitting = numpy.array(splitting)
    mag_B = numpy.array(mag_B)
    theta_B = numpy.array(theta_B)
    perp_B = numpy.array(perp_B)
    perp_B_frac = numpy.array(perp_B_frac)
    contrast_minus = numpy.array(contrast_minus)
    contrast_plus = numpy.array(contrast_plus)
    rabi_minus = numpy.array(rabi_minus)
    rabi_plus = numpy.array(rabi_plus)
    gamma = numpy.array(gamma)
    gamma_error = numpy.array(gamma_error)
    omega = numpy.array(omega)
    omega_error = numpy.array(omega_error)
    ratio = numpy.array(ratio)
    ratio_error = numpy.array(ratio_error)
    
    # Calculate correlations
    corr_fun = weighted_corrcoeff
    
    for x_name in columns:
        print()
        
        for y_name in columns:
            
            x_column = eval(x_name)
            y_column = eval(y_name)
            
            x_error = None
            if x_name in error_columns:
                x_error = eval('{}_error'.format(x_name))
            y_error = None
            if y_name in error_columns:
                y_error = eval('{}_error'.format(y_name))
                
            # If we have errors on both columns, add in quadrature. This is
            # just what my intuition says is correct! I haven't checked this.
            if (x_error is not None) and (y_error is not None):
                error = numpy.sqrt(x_error**2 + y_error**2)
            elif x_error is not None:
                error = x_error
            elif y_error is not None:
                error = y_error
            else:
                error = None
                
            corrcoeff = corr_fun(x_column, y_column, error)
            # print('{}, {}: {:.2}'.format(x_name, y_name, corrcoeff))
            print(corrcoeff)
    

# %% Main
            

def main(nv_data):
    
    # fig, axes_pack = plt.subplots(4,1, figsize=(5.0625,9.0))
    fig = plt.figure(figsize=(5.0625,7.0))
    
    gs_fig = gridspec.GridSpec(2, 1, figure=fig)
    gs_top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_fig[0],
                                              hspace=0.0)
    gs_bot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_fig[1],
                                              hspace=0.0)
    
    axes_pack = []
    
    ax = fig.add_subplot(gs_top[0])
    axes_pack.append(ax)
    ax = fig.add_subplot(gs_top[1], sharex=ax)
    axes_pack.append(ax)
    
    ax = fig.add_subplot(gs_bot[0])
    axes_pack.append(ax)
    ax = fig.add_subplot(gs_bot[1], sharex=ax)
    axes_pack.append(ax)
    
    
    # %% Axes setups
    
    x_min = -1.5
    x_max = 61.5
    # x_min = -5
    # x_max = 115
    
    omega_label = r'$\Omega$ (kHz)'
    omega_min = 0.043
    omega_max = 0.077
    
    gamma_label = r'$\gamma$ (kHz)'
    gamma_min = 0.09
    gamma_max = 0.27
            
    ax = axes_pack[0]
    # ax.set_xlabel(r'$B_{\parallel}$ (G)')
    # ax.set_xlim(x_min, x_max)
    ax.set_ylabel(omega_label)
    ax.set_ylim(omega_min, omega_max)
    
    ax = axes_pack[1]
    ax.set_xlabel(r'$B_{\parallel}$ (G)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel(gamma_label)
    ax.set_ylim(gamma_min, gamma_max)
    
    ax = axes_pack[2]
    # ax.set_xlabel(r'$B_{\perp}$ (G)')
    # ax.set_xlim(x_min, x_max)
    ax.set_ylabel(omega_label)
    ax.set_ylim(omega_min, omega_max)
    
    ax = axes_pack[3]
    ax.set_xlabel(r'$B_{\perp}$ (G)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel(gamma_label)
    ax.set_ylim(gamma_min, gamma_max)
    
    all_omega = []
    all_omega_err = []
    all_gamma = []
    all_gamma_err = []
    all_par_B = []
    all_perp_B = []

    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        marker = nv['marker']
        color = nv['color'] 
        markerfacecolor = nv['markerfacecolor']
        
        name = nv['name']
        # if name in ['NVA1', 'NVA2']:
        #     continue
        # if name != 'test':
        #     continue
        
        omega = numpy.array(nv['omega'])
        omega_err = numpy.array(nv['omega_err'])
        gamma = numpy.array(nv['gamma'])
        gamma_err = numpy.array(nv['gamma_err'])
        mag_B = numpy.array(nv['mag_B'])
        par_B = numpy.array(nv['par_B'])
        perp_B = numpy.array(nv['perp_B'])
        
        # Only plot points with measured angles and
        # B small enough to fit in xlims
        mag_B_mask = []
        for val in mag_B:
            if val is None:
                mag_B_mask.append(False)
            elif val > 65:
                mag_B_mask.append(False)
            else:
                mag_B_mask.append(True)
        mag_B_mask = numpy.array(mag_B_mask)
        angle_mask = par_B != None
        mask = angle_mask * mag_B_mask
        
        # Calculate based on all measurements with components, including
        # those off axis
        all_omega.extend(omega[angle_mask])
        all_omega_err.extend(omega_err[angle_mask])
        all_gamma.extend(gamma[angle_mask])
        all_gamma_err.extend(gamma_err[angle_mask])
        all_par_B.extend(par_B[angle_mask])
        all_perp_B.extend(perp_B[angle_mask])
    
        ax = axes_pack[0]
        if True in mask:
            ax.errorbar(par_B[mask], omega[mask],
                        yerr=omega_err[mask], label=name, marker=marker, 
                        color=color, markerfacecolor=markerfacecolor, 
                        linestyle='None', ms=ms, lw=lw)
    
        ax = axes_pack[1]
        if True in mask:
            ax.errorbar(par_B[mask], gamma[mask],
                        yerr=gamma_err[mask], label=name, marker=marker, 
                        color=color, markerfacecolor=markerfacecolor, 
                        linestyle='None', ms=ms, lw=lw)
    
        ax = axes_pack[2]
        if True in mask:
            ax.errorbar(perp_B[mask], omega[mask],
                        yerr=omega_err[mask], label=name, marker=marker, 
                        color=color, markerfacecolor=markerfacecolor, 
                        linestyle='None', ms=ms, lw=lw)
    
        ax = axes_pack[3]
        if True in mask:
            ax.errorbar(perp_B[mask], gamma[mask],
                        yerr=gamma_err[mask], label=name, marker=marker, 
                        color=color, markerfacecolor=markerfacecolor, 
                        linestyle='None', ms=ms, lw=lw)
            
    # Cast to arrays
    all_omega = numpy.array(all_omega)
    all_omega_err = numpy.array(all_omega_err)
    all_gamma = numpy.array(all_gamma)
    all_gamma_err = numpy.array(all_gamma_err)
    all_par_B = numpy.array(all_par_B)
    all_perp_B = numpy.array(all_perp_B)

    # Legend
    ax = axes_pack[0]
    # ax.legend()
    # Label sorting as foretold in the good book, stack overflow
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # Switch NVE to NVC
    labels = list(labels)
    labels[labels.index('NVE')] = 'NVC'
    # for el in handles:
    #     el.set_yerr_size(0.0)
    ax.legend(handles, labels, bbox_to_anchor=(0., 1.08, 1., .102),
              loc='lower left', ncol=5, mode='expand',
              borderaxespad=0., handlelength=0.5, )
    
    # yticks
    ticks = numpy.linspace(0.05, 0.07, 3)  # omega
    axes_pack[0].set_yticks(ticks)
    axes_pack[2].set_yticks(ticks)
    ticks = numpy.linspace(0.1, 0.25, 4)  # gamma
    axes_pack[1].set_yticks(ticks)
    axes_pack[3].set_yticks(ticks)
    
    # xticks
    axes_pack[0].set_zorder(10)
    axes_pack[2].set_zorder(10)
    axes_pack[0].tick_params('x', direction='inout',
                             labelbottom=False, length=6)
    axes_pack[2].tick_params('x', direction='inout',
                             labelbottom=False, length=6)
    axes_pack[1].tick_params('x', top=True)
    axes_pack[3].tick_params('x', top=True)
    
    # Label
    fig_labels = ['(a)', '(b)', '(c)', '(d)']
    for ind in range(4):
        ax = axes_pack[ind]
        ax.text(-0.16, 0.92, fig_labels[ind], transform=ax.transAxes,
                color='black', fontsize=14)
        
        
    # Linear fits
    B_linspace = numpy.linspace(0, x_max, num=1000)
    abs_sig = True
    
    popt, pcov = curve_fit(linear, all_par_B, all_omega,
                            sigma=all_omega_err, absolute_sigma=abs_sig,
                            p0=(0.0, numpy.average(all_omega)))
    conf_int(axes_pack[0], B_linspace, popt, pcov)
    
    popt, pcov = curve_fit(linear, all_par_B, all_gamma,
                            sigma=all_gamma_err, absolute_sigma=abs_sig,
                            p0=(0.0, numpy.average(all_omega)))
    conf_int(axes_pack[1], B_linspace, popt, pcov)
    
    popt, pcov = curve_fit(linear, all_perp_B, all_omega,
                            sigma=all_omega_err, absolute_sigma=abs_sig,
                            p0=(0.0, numpy.average(all_omega)))
    conf_int(axes_pack[2], B_linspace, popt, pcov)
    
    popt, pcov = curve_fit(linear, all_perp_B, all_gamma,
                            sigma=all_gamma_err, absolute_sigma=abs_sig,
                            p0=(0.0, numpy.average(all_omega)))
    conf_int(axes_pack[3], B_linspace, popt, pcov)
        
    
    # %% Wrap up
    
    fig.tight_layout(pad=0.4, h_pad=0.4)

    
def color_scatter(nv_data):
    
    all_ratios = []
    all_ratio_errors = []
    
    # par_B, perp_B on one axis
    plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, ax = plt.subplots(1,1, figsize=(7,8))
    ax.set_xlabel(r'$B_{\parallel}$ (G)')
    ax.set_xlim(-1.5, 61.5)
    ax.set_ylabel(r'$B_{\perp}$ (G)')
    ax.set_ylim(-1.5, 61.5)

    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        marker = nv['marker']
        color = nv['color'] 
        
        name = nv['name']
        # if name in ['NVA1', 'NVA2']:
        #     continue
        # if name != 'test':
        #     continue
        
        # Calculate ratios
        ratios = numpy.array(nv['ratio'])
        ratio_errors = numpy.array(nv['ratio_err'])
        all_ratios.extend(ratios)
        all_ratio_errors.extend(ratio_errors)
        
        # Plot splitting
        # ax = axes_pack[0]
        # splittings = numpy.array(nv['splitting'])
        # mask = splittings != None
        # if True in mask:
        #     ax.errorbar(splittings[mask], ratios[mask],
        #                 yerr=ratio_errors[mask], label=name,
        #                 marker=marker, color=color, linestyle='None',
        #                 ms=9, lw=2.5)
    
        # Plot par_B
        # ax = axes_pack[1]
        par_Bs = numpy.array(nv['par_B'])
        perp_Bs = numpy.array(nv['perp_B'])
        mask = par_Bs != None
        if True in mask:
            scatter = ax.scatter(par_Bs[mask], perp_Bs[mask],
                                 c=ratios[mask], cmap='inferno')
    
    cbar = fig.colorbar(scatter)
    fig.tight_layout()


# %% Run


if __name__ == '__main__':
    
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{physics}',
        r'\usepackage{sfmath}',
        r'\usepackage{upgreek}',
        r'\usepackage{helvet}',
       ]  
    plt.rcParams.update({'font.size': 11.25})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    
    path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/papers/bulk_dq_relaxation/'
    file = path + 'compiled_data_import.csv'
    nv_data = get_nv_data_csv(file)
    # print(nv_data)
    
    # print(rate_coeff)
    
    main(nv_data)
    # phonon_fit(nv_data)
    # color_scatter(nv_data)
    # plot_gamma_omega_vs_angle(nv_data)
    # hist_gamma_omega(nv_data)
    # correlations(nv_data)
    # plot_splittings_vs_angle(nv_data)
    
    # %% Phonon fitting
    
    # B_field_vec = [0.0, 0.0, 0.1]
    
    # # k =1e10, q=4e=6e-19 for Carbon nucleus, 154e-12 bond length, a pm on 
    # # either side gives 6e7 E field in V / cm! So large fields are reasonable
    # mag_E = 100  # This number is chosen to get ~kHz rates out
    # d_perp_prime = 17
    # noise_hamiltonian = calc_electric_hamiltonian([mag_E]*3, d_perp_prime)
    
    # # val = rate(-1, +1, B_field_vec, noise_hamiltonian)
    # # print(val/1000)  # rate in kHz
    # val = rate_kHz_electric_perp_B(0, +1, 0.0, mag_E, d_perp_prime)
    # print(val)  # rate in kHz
    
    # opts = [400, 1100, 1800]
    # p0s = [[a, b, c, d, e] for a in opts for b in opts for c in opts for d in opts for e in opts]
    # print(p0s)
    
    # for p0 in p0s:
    #     phonon_fit(nv_data, p0)
    
    # plot_fx()
    
    # get_coupling_constants(nv_data)


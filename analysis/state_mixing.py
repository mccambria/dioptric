# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:28:01 2019

Mixing

First modeling additional rotating magnetic and electric field on x-y plane
(perp to NV axis)

@author: Aedan
"""

import numpy
from numpy import exp
#from numpy import pi
#from numpy.linalg import eigvals
from numpy.linalg import eig
import matplotlib.pyplot as plt


# %% Constants

# GHz
d_gs = 2.87

# numbers
inv_sqrt_2 = 1/numpy.sqrt(2)

# %% Values

# NV1
name = 'NV1'
#B_mag = 0.05 
#B_mag = .032 # GHz
B_theta = 1.2 # rad
Pi_par = -0.005 # GHz
Pi_perp = 0.010 # GHz

B_mag = 1.2 # GHz
#B_mag = 0.5 # GHz
resonant_freq = 2.4371 # GHz
contrast = 0.16
resonant_rabi_period = 235.4 # ns

# %%

def calc_single_hamiltonian_osc(mag_B, theta_B, perp_B_prime, par_Pi, perp_Pi, 
                                perp_Pi_prime, phi_B, phi_Pi):
    par_B = mag_B * numpy.cos(theta_B)
    perp_B = mag_B * numpy.sin(theta_B)
    hamiltonian = numpy.array([[d_gs + par_Pi + par_B,
                                inv_sqrt_2 * (perp_B + perp_B_prime * exp(-1j * phi_B)),
                                -perp_Pi - perp_Pi_prime * exp(1j * phi_Pi)],
                               [inv_sqrt_2 * (perp_B + perp_B_prime * exp(1j * phi_B)),
                                0,
                                inv_sqrt_2 * (perp_B + perp_B_prime * exp(-1j * phi_B))],
                               [-perp_Pi - perp_Pi_prime * exp(-1j * phi_Pi),
                                inv_sqrt_2 * (perp_B + perp_B_prime * exp(1j * phi_B)),
                                d_gs + par_Pi - par_B]])
    return hamiltonian

def calc_prob_i_state(final_hamiltonian):
    '''
    This function will compare the HIGH state of an final hamiltonian
    to the i state in the Sz basis (+1, -1, 0).
    '''
    
    # Calculate the eigenvalues and eigenvectros of the hamiltonian some time later
    eigsolution  = eig(final_hamiltonian)
    eigval_t = eigsolution[0]
    eigvec_t = eigsolution[1]
    
    # Collect the HIGH state of this hamiltonian
    high_index = numpy.argmax(numpy.abs(eigval_t))
    zero_index = numpy.argmin(numpy.abs(eigval_t))
    mid_value_list = list(set([0,1,2]) - set([zero_index, high_index]))
    low_index = mid_value_list[0]
    
    HIGH_state_t = eigvec_t[:,high_index]
#    print(HIGH_state_t)
#    print(eig(final_hamiltonian))
    
#    high_index = 0
#    zero_index = 2
#    low_index = 1
#    if high_index == 0:
#        zero_index = 2
#        low_index = 1
#    if high_index == 1:
#        zero_index = 0
#        low_index = 2
#    if high_index == 2:
#        zero_index = 1
#        low_index = 0
        
    # Calculate < i | psi(t) >
    prob_dens_plus = HIGH_state_t[0]
    prob_dens_zero = HIGH_state_t[1] # for some reason, 0 and -1 element in eigenvalue are switched from what I'd expect
    prob_dens_minus = HIGH_state_t[2]
        
    # Calculate the corresponding probs
    prob_plus = numpy.abs(numpy.dot(numpy.conj(prob_dens_plus), prob_dens_plus))
    prob_zero = numpy.abs(numpy.dot(numpy.conj(prob_dens_zero), prob_dens_zero))
    prob_minus = numpy.abs(numpy.dot(numpy.conj(prob_dens_minus), prob_dens_minus))
    
    return prob_plus, prob_zero, prob_minus,  \
            eigval_t[low_index], eigval_t[zero_index], eigval_t[high_index]

def calc_LOW_resonance(hamiltonian):
    '''
    Returns the LOW frequency of the hamiltonian passed
    '''
    
    # Calculate the eigenvalues and eigenvectros of the hamiltonian some time later
    eigval_t, eigvec_t = eig(hamiltonian)
    
    # Collect the HIGH state of this hamiltonian
    vals = numpy.sort(eigval_t)
    resonance_low = numpy.real(vals[1] - vals[0])

    return resonance_low
    
    
def rabi_contrast(freq, resonant_freq, contrast, resonant_rabi_period):
    resonant_rabi_freq = resonant_rabi_period**-1
    res_dev = freq - resonant_freq
    measured_rabi_freq = numpy.sqrt(res_dev**2 + resonant_rabi_freq**2)
#    print(resonant_rabi_freq)
    applied_pi_pulse = (resonant_rabi_period / 2) 
    
    amp = (resonant_rabi_freq / measured_rabi_freq)**2
    angle = measured_rabi_freq * 2 * numpy.pi * applied_pi_pulse / 2
    prob = amp * (numpy.sin(angle))**2
    
#    print(resonant_rabi_freq/measured_rabi_freq)

    current_contrast = (contrast * prob)

    return current_contrast

# %%
    
if __name__ == '__main__':
    plus_1_list = []
    zero_list = []
    minus_1_list = []
    eigen_LOW_list = []
    eigen_ZERO_list = []
    eigen_HIGH_list = []
    
    efficiency_list = []
    
    omega_B = .1  # rad / ms
    omega_Pi = 1 # rad / ms
    starting_Phi_Pi = 0.2 # rad
    
#    B_mag = 0
#    Pi_perp = 1
    
    B_perp_noise = B_mag * 1
    Pi_perp_noise = Pi_perp * 2
#    Pi_perp_noise = 1 * B_perp_noise
    
#    tau = numpy.linspace(10.75, 10.77, 10)
    tau = numpy.linspace(0, 100, 1000)
#    ham_0_plus_one_basis = calc_single_hamiltonian_osc(B_mag, B_theta, 0, Pi_par, 
#                                            Pi_perp, 0, 0, 0)
#    print(eig(ham_0_plus_one_basis))
    
    
    
    for t in tau:
        phi_B = omega_B * t
        phi_Pi = omega_Pi * t + starting_Phi_Pi
 
        ham_t = calc_single_hamiltonian_osc(B_mag, B_theta, B_perp_noise, 
                                            Pi_par, Pi_perp, Pi_perp_noise, 
                                            phi_B, phi_Pi)
        probs = calc_prob_i_state(ham_t)
        
        plus_1_list.append(probs[0])
        zero_list.append(probs[1])
        minus_1_list.append(probs[2])
        eigen_LOW_list.append(probs[3])
        eigen_ZERO_list.append(probs[4])
        eigen_HIGH_list.append(probs[5])
        
        low_res = calc_LOW_resonance(ham_t)
        
        new_contrast = rabi_contrast(low_res, resonant_freq, contrast, resonant_rabi_period)
#        print(new_contrast)
        efficiency = new_contrast / contrast
        efficiency_list.append(efficiency)
    total = numpy.array(plus_1_list) + numpy.array(zero_list) + numpy.array(minus_1_list)
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(tau, plus_1_list, label = '+1 component')
    ax.plot(tau, zero_list, label = '0 component')
    ax.plot(tau, minus_1_list, label = '-1 component')
#    ax.plot(tau, eigen_LOW_list, label = 'LOW eigenvalue')
#    ax.plot(tau, eigen_ZERO_list, label = 'ZERO eigenvalue')
#    ax.plot(tau, eigen_HIGH_list, label = 'HIGH eigenvalue')
#    ax.plot(tau, efficiency_list, label = 'pi pulse efficiency')
#    ax.plot(tau, total, label = 'total')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.set_title(name)
    textstr = '\n'.join((
        r'$B_{\perp, noise}=%.3f \ GHz$' % (B_perp_noise, ),
        r'$\Pi_{\perp, noise}=%.3f \ GHz$' % (Pi_perp_noise, ),
        r'$\omega_{B}=%.2f \ rad/ms$' % (omega_B, ),
        r'$\omega_{\Pi}=%.2f \ rad/ms$' % (omega_Pi, ),
        r'$\phi_{\Pi, 0}=%.3f \ rad$' % (starting_Phi_Pi, )
        ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.65, textstr, fontsize=14, transform=ax.transAxes,
            verticalalignment='top', bbox=props)
    
        
    
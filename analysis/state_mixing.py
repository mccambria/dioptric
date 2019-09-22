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
B_mag = 0.05
B_theta = 0.644
Pi_par = -0.005
Pi_perp = 0.010

# %%

def calc_single_hamiltonian(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    par_B = mag_B * numpy.cos(theta_B)
    perp_B = mag_B * numpy.sin(theta_B)
    hamiltonian = numpy.array([[d_gs + par_Pi + par_B,
                                inv_sqrt_2 * perp_B * exp(-1j * phi_B),
                                -perp_Pi * exp(1j * phi_Pi)],
                               [inv_sqrt_2 * perp_B * exp(1j * phi_B),
                                0,
                                inv_sqrt_2 * perp_B * exp(-1j * phi_B)],
                               [-perp_Pi * exp(-1j * phi_Pi),
                                inv_sqrt_2 * perp_B * exp(1j * phi_B),
                                d_gs + par_Pi - par_B]])
    return hamiltonian

def calc_single_hamiltonian_osc(mag_B, theta_B, perp_B_prime, par_Pi, perp_Pi, 
                                perp_Pi_prime, phi_B, phi_Pi):
    par_B = mag_B * numpy.cos(theta_B)
    perp_B = mag_B * numpy.sin(theta_B)
    hamiltonian = numpy.array([[d_gs + par_Pi + par_B,
                                inv_sqrt_2 * (perp_B + perp_B_prime * exp(-1j * phi_B)),
                                -perp_Pi - perp_Pi_prime * exp(1j * (phi_Pi))],
                               [inv_sqrt_2 * (perp_B + perp_B_prime * exp(1j * phi_B)),
                                0,
                                inv_sqrt_2 * (perp_B + perp_B_prime * exp(-1j * phi_B))],
                               [-perp_Pi - perp_Pi_prime * exp(-1j * (phi_Pi)),
                                inv_sqrt_2 * (perp_B + perp_B_prime * exp(1j * phi_B)),
                                d_gs + par_Pi - par_B]])
    return hamiltonian

def calc_prob_same_state(initial_hamiltonian, final_hamiltonian):
    '''
    This function will compare the HIGH state of an initial hamiltonian
    to the HIGH state of a final hamiltonian
    '''
    
    # Calculate the eigenvalues and eigenvectors of the initial hamiltonian
    eigval_0, eigvec_0 = eig(initial_hamiltonian)
    
    index_max = numpy.argmax(numpy.abs(eigval_0))
    
    # Determine what the high state is of this initial hamiltonian
    HIGH_state_0 = eigvec_0[index_max]
    
    # Calculate the eigenvalues and eigenvectros of the hamiltonian some time later
    eigval_t, eigvec_t = eig(final_hamiltonian)
    
    index_max = numpy.argmax(numpy.abs(eigval_t))
    
    # Determine what the high state is of this later hamiltonian
    HIGH_state_t = eigvec_t[index_max]
    
    # Calculate the prob of finding the final state in the initial state
    inner_product = numpy.dot(numpy.conj(HIGH_state_0), HIGH_state_t)
    prob = numpy.abs(numpy.dot(numpy.conj(inner_product), inner_product))
    
    return prob

def calc_prob_opposite_state(initial_hamiltonian, final_hamiltonian):
    '''
    This function will compare the HIGH state of an initial hamiltonian
    to the LOW state of a final hamiltonian
    '''
    
    # Calculate the eigenvalues and eigenvectors of the initial hamiltonian
    eigval_0, eigvec_0 = eig(initial_hamiltonian)
    
    index_max = numpy.argmax(numpy.abs(eigval_0))
    
    # Determine what the high state is of this initial hamiltonian
    HIGH_state_0 = eigvec_0[index_max]
    
    # Calculate the eigenvalues and eigenvectros of the hamiltonian some time later
    eigval_t, eigvec_t = eig(final_hamiltonian)
    
    index_min = numpy.argmin(numpy.abs(eigval_t))
    
    # Determine what the high state is of this later hamiltonian
    LOW_state_t = eigvec_t[index_min]
    
    # Calculate the prob of finding the final state in the initial state
    inner_product = numpy.dot(numpy.conj(HIGH_state_0), LOW_state_t)
    prob = numpy.abs(numpy.dot(numpy.conj(inner_product), inner_product))
    
    return prob
    
# %%
    
if __name__ == '__main__':
    same_prob_list = []
    opp_prob_list = []
    
    omega_B = 0.001  # rad / us
    omega_Pi = 0.009 # rad / us
    starting_Phi_Pi = 0.2
    
    B_perp_noise = 0.05 * 1
    Pi_perp_noise = 0.010 * 1
#    Pi_perp_noise = 0.01
    
    tau = numpy.linspace(0, 10000, 1000)
    
    ham_0 = calc_single_hamiltonian_osc(B_mag, B_theta, B_perp_noise, Pi_par, 
                                            Pi_perp, Pi_perp_noise, 0, starting_Phi_Pi)
    for t in tau:
        phi_B = omega_B * t
        phi_Pi = omega_Pi * t + starting_Phi_Pi
 
        ham_t = calc_single_hamiltonian_osc(B_mag, B_theta, B_perp_noise, 
                                            Pi_par, Pi_perp, Pi_perp_noise, 
                                            phi_B, phi_Pi)
        
        same_prob = calc_prob_same_state(ham_0, ham_t)
        opp_prob = calc_prob_opposite_state(ham_0, ham_t)
        
        same_prob_list.append(same_prob)
        opp_prob_list.append(opp_prob)
        
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(tau, same_prob_list, label = 'final state HIGH')
    ax.plot(tau, opp_prob_list, label = 'final state LOW')
    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.set_title(name)
    textstr = '\n'.join((
        r'$B_{\perp, noise}=%.3f \ GHz$' % (B_perp_noise, ),
        r'$\Pi_{\perp, noise}=%.3f \ GHz$' % (Pi_perp_noise, ),
        r'$\omega_{B}=%.3f \ rad/ \mu s$' % (omega_B, ),
        r'$\omega_{\Pi}=%.3f \ rad/\mu s$' % (omega_Pi, ),
        r'$\omega_{\Pi, 0}=%.3f \ rad$' % (starting_Phi_Pi, )
        ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.65, textstr, fontsize=14, transform=ax.transAxes,
            verticalalignment='top', bbox=props)
        
    
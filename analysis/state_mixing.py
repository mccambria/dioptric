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
import numpy.random as random


# %% Constants

# GHz
d_gs = 2.87

#d_gs = 1.42
# numbers
inv_sqrt_2 = 1/numpy.sqrt(2)

# %% Values

## NV1
#name = 'NV1'
#
#B_theta = 1.2 # rad
#Pi_par = -0.005 # GHz
#Pi_perp = 0.010 # GHz

# NV2
#name = 'NV2'
#
#B_theta = 0.640 # rad
#Pi_par = -0.004 # GHz
#Pi_perp = 0.007 # GHz

# NV13
#name = 'NV13'
#
#B_theta = 1.224 # rad
#Pi_par = -0.03 # GHz
#Pi_perp = 0.003 # GHz

# Test NV
name = 'NV1'

B_theta = (180-144.8) * numpy.pi / 180
Pi_par = 0 # GHz
Pi_perp = 0 # GHz

B_mag_Gauss = 50
B_mag = (B_mag_Gauss*2.8) / 1000 # GHz
#B_mag = 0.5 # GHz
resonant_freq = 2.4371 # GHz
contrast = 0.16
resonant_rabi_period = 235.4 # ns

# %%

def angle_between_vectors(v1, v2):
    theta = numpy.arccos(numpy.dot(v1,v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v1)))
    return theta

# %%

def rotate_y(A):
    return [[numpy.cos(A), 0, numpy.sin(A)], [0, 1, 0], [-numpy.sin(A), 0, numpy.cos(A)]]

# %%


def calc_single_hamiltonian_osc(mag_B, theta_B, perp_B_prime, par_Pi, perp_Pi,
                                perp_Pi_prime, phi_B, phi_Pi):
    '''
    Given values of the surround ing magnetic and Pi-field (electric and
    strain), this function calculates the Hamiltonian in the Sz basis.

    mag_B: the magnitude of the applied magnetic field
    theta_B: the angle that the applied magnetic field makes with the NV axis
    perp_B_prime: the perpendicular componenet of magnetic field noise,
    possibly changing in time
    par_Pi: the Pi field applied perpendicular to the NV axis
    perp_Pi: the Pi field applied perpendicular to the NV axis
    perp_Pi_prime: the perpendicular componenet of Pi field noise,
    possibly changing in time
    phi_B: The angle of the perpendicuclar magnetic fields
    phi_Pi: The angle of the perpendicular Pi fields

    Returns the Hamiltonian matrix as an array
    '''

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
    This function will compare the HIGH state of a final hamiltonian
    to the i state in the Sz basis (+1, -1, 0).

    Returns the various components of the HIGH states of the hamiltonian in the
    Sz basis, along with the eigenvalues of the hamiltonian
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
    '''
    Calculating the expected contrast due to how off resonance the applied
    microwaves are. Taken from major_routines.rabi.

    Needs work
    '''
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

def angle_rotation(angle):
    '''
    this function will calculate the angles between a the magnetic field from
    the permanent magnet and the 4 NV orientations. The NV orientations are
    set so that the roation axis of our rotation stage is along the y-axis
    of the coordinate system imposed. Then the program calculates the four
    angles made between the magnet at a specified angle and the NV orientations

    angle: float (radians)
    '''

    # start with the four N positions oriented in space so that the rotation
    #axis is along the y axis.

    a = [1,0,0]
    b = [-0.333478, 0.942758, 0]
    c = [-0.333261, -0.471379, -0.816541]
    d = [-0.333261, -0.471379, 0.816541]

    # define the magnetic field as pointing along x-axis, which will not rotate

    mag = [1,0,0]

    # rotate the NV system
    a_r = numpy.dot(rotate_y(angle), a)
    b_r = numpy.dot(rotate_y(angle), b)
    c_r = numpy.dot(rotate_y(angle), c)
    d_r = numpy.dot(rotate_y(angle), d)

    # Calc the angle between magnetic field and NV orientations

    theta_a = angle_between_vectors(a_r, mag)
    theta_b = angle_between_vectors(b_r, mag)
    theta_c = angle_between_vectors(c_r, mag)
    theta_d = angle_between_vectors(d_r, mag)

    return theta_a, theta_b, theta_c, theta_d

# %%

def simulate(t, num_reps):
    '''
    Similation of some B and Pi noise perpendicular to the NV axis that
    oscillates at a rate omega.
    '''

    omega_B = .1  # rad / ms
    omega_Pi = 1 # rad / ms

    B_perp_noise = B_mag * 1
    Pi_perp_noise = Pi_perp * 1

    plus_1_reps = []
    zero_reps = []
    minus_1_reps = []

    starting_Phi_Pi = random.uniform(0,2*numpy.pi, num_reps)
    starting_Phi_B = random.uniform(0,2*numpy.pi, num_reps)

    for i in range(num_reps):
        phi_B = omega_B * t + starting_Phi_B[i]
        phi_Pi = omega_Pi * t + starting_Phi_Pi[i]

        ham_t = calc_single_hamiltonian_osc(B_mag, B_theta, B_perp_noise,
                                        Pi_par, Pi_perp, Pi_perp_noise,
                                        phi_B, phi_Pi)
        probs = calc_prob_i_state(ham_t)

        plus_1_reps.append(probs[0])
        zero_reps.append(probs[1])
        minus_1_reps.append(probs[2])

    plus_1_prob = numpy.average(plus_1_reps)
    zero_prob = numpy.average(zero_reps)
    minus_1_prob = numpy.average(minus_1_reps)

    return plus_1_prob, zero_prob, minus_1_prob

# %%
if __name__ == '__main__':
    plus_1_list = []
    zero_list = []
    minus_1_list = []

    tau = numpy.linspace(0, .01, 100)
    num_reps = 10**4


#    for t in tau:
#        plus_1_prob, zero_prob, minus_1_prob = simulate(t, num_reps)
#
#        plus_1_list.append(plus_1_prob)
#        zero_list.append(zero_prob)
#        minus_1_list.append(minus_1_prob)
#
#    fig, ax = plt.subplots(figsize=(8.5, 8.5))
#    ax.plot(tau, plus_1_list, label = '+1 component')
#    ax.plot(tau, zero_list, label = '0 component')
#    ax.plot(tau, minus_1_list, label = '-1 component')
#
#    ax.set_xlabel('Time (ms)')
#    ax.set_ylabel('Probability')
#    ax.legend()
#    ax.set_title(name)


    angles = numpy.linspace(0, numpy.pi/2, 100)
    B_mag_Gauss = 50 #G
    eigensHIGH =[[],[],[],[]]
    eigensLOW = [[],[],[],[]]

    for t in range(len(angles)):
        angles_ret_vals = angle_rotation(angles[t])

        for NV in range(4):

            B_theta = angles_ret_vals[NV]

            B_mag = (B_mag_Gauss*2.8) / 1000 # GHz
            ham = calc_single_hamiltonian_osc(B_mag, B_theta, 0,
                                        Pi_par, Pi_perp, 0,
                                        0, 0)
            ret_vals = calc_prob_i_state(ham)

            eigensHIGH[NV].append(ret_vals[3])
            eigensLOW[NV].append(ret_vals[5])

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
#    print(eigensHIGH[0][48])
    ax.plot(angles*180/numpy.pi, eigensHIGH[0], label = 'initially aligned NV')
    ax.plot(angles*180/numpy.pi, eigensHIGH[1], label = 'b')
    ax.plot(angles*180/numpy.pi, eigensHIGH[2], label = 'c')
    ax.plot(angles*180/numpy.pi, eigensHIGH[3], label = 'd')
    ax.set_xlabel('Angle (degree)')
    ax.set_ylabel('Resonance (GHz)')
    ax.legend()


    angle_hopper = [0, 10, 20, 30, 40, 48, 50, 35, 52, 55, 55, 60, 70, 68, 75,
                    85, 89]

    resonances_1 = [2.719, 2.720, 2.726, 2.737, 2.751, 2.767, 2.77, 2.744,
                    2.765, 2.76, 2.759, 2.759, 2.753, 2.815, 2.755, 2.847, 2.851]

    resonances_2 = [2.828, 2.821, 2.806, 2.788, 2.772, 2.822, 2.822, 2.782,
                    2.776, 2.783, 2.783, 2.795, 2.820, 2.829, 2.8343, 2.860, 2.871]
    resonances_3 = [2.834, None, 2.817, 2.817, 2.819, None, None,2.818,  None, None,
                    2.824, 2.824, 2.830, None, None, None, None]

    print(len(resonances_3))
    print(len(angle_hopper))
    ax.plot(angle_hopper, resonances_1, 'ko')
    ax.plot(angle_hopper, resonances_2, 'ko')
    ax.plot(angle_hopper, resonances_3, 'ko')
#    ham = calc_single_hamiltonian_osc(B_mag, B_theta, 0,
#                                        Pi_par, Pi_perp, 0,
#                                        0, 0)
#
#    ret_vals = calc_prob_i_state(ham)
#    print('What is the mixture of the Sz eigenstates in the highest energy eigenstate of this Hamltonian:')
#    print(ret_vals)




#    angles=[]
#    for angle in range(0,91):
#        B_theta = angle * numpy.pi / 180
#        ham = calc_single_hamiltonian_osc(B_mag, B_theta, 0,
#                                        Pi_par, Pi_perp, 0,
#                                        0, 0)
#
#        ret_vals = calc_prob_i_state(ham)
#
#        angles.append(angle)
#        plus_1_list.append(ret_vals[0])
#        zero_list.append(ret_vals[1])
#        minus_1_list.append(ret_vals[2])
#
#    fig, ax = plt.subplots(figsize=(8.5, 8.5))
#    ax.plot(angles, plus_1_list, label = '+1 component')
#    ax.plot(angles, zero_list, label = '0 component')
#    ax.plot(angles, minus_1_list, label = '-1 component')
#
#    ax.set_xlabel('Angle (degree)')
#    ax.set_ylabel('Probability')
#    ax.legend()
#    ax.set_title('Sz eigenstate composition of highest energy eigenstate of Hamiltonian')
#
##    textstr = '\n'.join((
##        r'$B_{\perp, noise}=%.3f \ GHz$' % (B_mag, ),
##        r'$\phi_{\Pi, 0}=%.3f \ rad$' % (starting_Phi_Pi, )
##        ))
#    textstr = ( r'$B_{applied}=%.0f \ G$' % (B_mag_Gauss, ))
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    ax.text(0.05, 0.65, textstr, fontsize=14, transform=ax.transAxes,
#            verticalalignment='top', bbox=props)

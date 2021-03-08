# -*- coding: utf-8 -*-
"""
Reproduce Jarmola 2012 temperature scalings

Created on Fri Jun 26 17:40:09 2020

@author: matth
"""


# %% Imports


import numpy
import matplotlib.pyplot as plt


# %% Constants


Boltzmann = 8.617e-2  # meV / K
# from scipy.constants import Boltzmann  # J / K

# Rate coefficients in s^-1 from Jarmola
A_1 = 0.007  # Constant for S3
A_2 = 2.1e3  # Orbach
# A_2 = 2.0e3  # test
A_3 = 2.2e-11  # T^5
A_4 = 4.3e-6  # T^3
A_7 = 2.55e-20

# Quasilocalized mode activation energy
quasi = 73.0  # meV, empirical fit
# quasi = 65.0  # meV, quasilocalized resonance
# quasi = 1.17e-20  # J

ms = 7
lw = 1.75


# %% Processes and sum functions


def bose(energy, temp):
    return 1 / (numpy.exp(energy / (Boltzmann * temp)) - 1)

def orbach(temp):
    """
    This is for quasilocalized phonons interacting by a Raman process, which
    reproduces an Orbach scaling even though it's not really an Orbach.
    process. As such, the proper scaling is 
    n(omega)(n(omega)+1) approx n(omega) for omega << kT
    """
    # return A_2 * bose(quasi, temp) * (bose(quasi, temp) + 1)
    return A_2 * bose(quasi, temp)
    # return A_2 / (numpy.exp(quasi / (Boltzmann * temp)))

def raman(temp):
    return A_3 * (temp**5)

def test_T_cubed(temp):
    return A_4 * (temp**3)

def test_T_seventh(temp):
    return A_7 * (temp**7)

def omega_calc(temp):
    return (A_1 + orbach(temp) + raman(temp)) / 3
    # return A_1 + test_T_cubed(temp) + raman(temp)
    # return (A_1 + orbach(temp) + raman(temp) + test_T_seventh(temp)) / 3


# %% Main


def main():
    temp_linspace = numpy.linspace(5, 500, 1000)
    # temp_linspace = numpy.linspace(5, 5000, 1000)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_title(r'$\Omega$ from Jarmola 2012 Eq. 1 for S3')
    ax.plot(temp_linspace, omega_calc(temp_linspace), label='Total')
    ax.plot(temp_linspace, [A_1]*1000, label='Constant')
    ax.plot(temp_linspace, orbach(temp_linspace)/3, label='Orbach')
    # ax.plot(temp_linspace, test_T_cubed(temp_linspace), label='Orbach')
    ax.plot(temp_linspace, raman(temp_linspace)/3, label='Raman')
    # ax.plot(temp_linspace, test_T_seventh(temp_linspace), label='T7')
    ax.legend(loc='upper left')
    ax.set_xlabel(r'T (K)')
    ax.set_ylabel(r'$\Omega (\text{s}^{-1})$')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_ylim(2e-3, 2e3)
    ax.set_ylim(-10, 130)
    # ax.set_ylim(2e-3, None)


def main2(temps=None, omegas=None, omega_errs=None,
         gammas=None, gamma_errs=None):
    temp_linspace = numpy.linspace(5, 600, 1000)
    # temp_linspace = numpy.linspace(100, 300, 1000)
    # temp_linspace = numpy.linspace(5, 5000, 1000)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    # ax.set_title('Relaxation rates')
    ax.plot(temp_linspace, omega_calc(temp_linspace),
            label=r'$\Omega$ fit', color='#FF9933')  # from Jarmola 2012 Eq. 1 for S3
    
    ax.plot(temp_linspace, orbach(temp_linspace) * 0.7, label='Orbach')
    # ax.plot(temp_linspace, raman(temp_linspace)/3, label='Raman')
    
    ax.set_xlabel(r'T (K)')
    ax.set_ylabel(r'Relaxation rates (s$^{-1})$')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_ylim(2e-3, 2e3)
    ax.set_ylim(-10, 130)
    ax.set_ylim(-10, 800)
    # ax.set_ylim(2e-3, None)
    
    if temps is not None:
        temps = numpy.array(temps)
        omegas = numpy.array(omegas)
        omega_errs = numpy.array(omega_errs)
        gammas = numpy.array(gammas)
        gamma_errs = numpy.array(gamma_errs)
        
        mask = []
        for el in omegas:
            mask.append(el is not None)
        ax.errorbar(temps[mask],
                    omegas[mask] * 1000, yerr=omega_errs[mask] * 1000,
                    label=r'$\Omega$', marker='^',
                    color='#FF9933', markerfacecolor='#FFCC33',
                    linestyle='None', ms=ms, lw=lw)
        
        mask = []
        for el in gammas:
            mask.append(el is not None)
        ax.errorbar(temps[mask],
                    gammas[mask] * 1000, yerr=gamma_errs[mask] * 1000,
                    label= r'$\gamma$', marker='o',
                    color='#993399', markerfacecolor='#CC99CC',
                    linestyle='None', ms=ms, lw=lw)
        
    ax.legend(loc='upper left')


# %% Run the file


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
    
    temps = [295, 250, 225, 200, 175]
    omegas = [0.059, 0.025, None, None, None]
    omega_errs = [0.002, 0.002, None, None, None]
    gammas = [0.117, 0.101, 0.065, 0.028, 0.019]
    gamma_errs = [0.005, 0.013, 0.007, 0.003, 0.003]
    
    # Lin paper results
    temps.extend([300, 325, 350, 375, 400, 425, 450, 500, 550, 600])
    lin_omegas = numpy.array([0.33, 0.4, 0.5, 0.6, 0.66, 0.9, 1.1, 1.5, 2.3, 3.5]) / 4
    omegas.extend(lin_omegas.tolist())
    omega_errs.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    lin_gammas_pre = [200, None, 266, None, 333, None, 400, 475, 650, 780]
    lin_gammas = []
    for el in lin_gammas_pre:
        if el is not None:
            lin_gammas.append(el * (117/200) / 1000)
        else:
            lin_gammas.append(None)
    gammas.extend(lin_gammas)
    gamma_errs.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    

    main2(temps, omegas, omega_errs, gammas, gamma_errs)

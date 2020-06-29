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

# Rate coefficients in s^-1
A_1 = 0.007
A_2 = 2.1e3
# A_2 = 2.1e4  # test
A_3 = 2.2e-11
A_4 = 4.3e-6

# Quasilocalized mode activation energy
quasi = 73.0  # meV
# quasi = 1.17e-20  # J


# %% Processes and sum functions


def orbach(temp):
    return A_2 / (numpy.exp(quasi / (Boltzmann * temp)) - 1)

def raman(temp):
    return A_3 * (temp**5)

def test_T_cubed(temp):
    return A_4 * (temp**3)

def T_1(temp):
    return A_1 + orbach(temp) + raman(temp)
    # return A_1 + test_T_cubed(temp) + raman(temp)


# %% Main


def main():
    temp_linspace = numpy.linspace(5, 500, 1000)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_title(r'$1/T_{1}$ from Jarmola 2012 Eq. 1 for S3')
    ax.plot(temp_linspace, T_1(temp_linspace), label='Total')
    ax.plot(temp_linspace, [A_1]*1000, label='Constant')
    ax.plot(temp_linspace, orbach(temp_linspace), label='Orbach')
    # ax.plot(temp_linspace, test_T_cubed(temp_linspace), label='Orbach')
    ax.plot(temp_linspace, raman(temp_linspace), label='Raman')
    ax.legend()
    ax.set_xlabel(r'T (K)')
    ax.set_ylabel(r'$1/T_{1}$ (Hz)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(2e-3, 2e3)
    # ax.set_ylim(2e-3, None)


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
    
    main()

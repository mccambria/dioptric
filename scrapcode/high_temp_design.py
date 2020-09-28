# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:49:40 2020

@author: matth
"""


# %% Imports

import numpy as np
import matplotlib.pyplot as plt


# %% Constants


T_0 = 200  # C
T_air = 22  # C
h_air = 10  # W / (K m2)

w = 0.035  # m, PCB
l = 0.023  # m, PCB tail
feed_line_l = 0.023  # m

# k = 0.2  # W / (K m), substrate
# t = 0.000100  # m, substrate
# sides = 2  # substrate

k = 385  # W / (K m), Cu
# t = 36E-6  # m, Cu
t = 6.3E-3  # m, Cu
sides = 2  # Cu


# %% Functions 


def temp(l):
    return (T_0 - T_air) * np.exp(-np.sqrt(sides * (h_air / k) * (1/t)) * l) + T_air


# %% Main functions


def equilibrium_PCB():
    lengths = np.linspace(0, 25E-3, 1000)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.plot(lengths, temp(lengths))
    ax.set_xlabel('l (m)')
    ax.set_ylabel('T(l) (C)')


# %% Run the file


if __name__ == '__main__':
    
    equilibrium_PCB()
    
    # print(k * t * w * (T_0 - T_air) / feed_line_l)
    
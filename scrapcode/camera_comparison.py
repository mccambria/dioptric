# -*- coding: utf-8 -*-
"""
EMCCD vs qCMOS

Created on May 11th, 2023

@author: mccambria
"""


# region Import and constants

import csv
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import utils.tool_belt as tool_belt
from utils import common
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from utils.tool_belt import bose
from scipy.special import gammaln, xlogy

inv_root_2_pi = 1 / np.sqrt(2*np.pi)

# endregion

def normal(x, s):
    return (inv_root_2_pi / s) * np.exp(-((x/s)**2) / 2)

def poisson(x, l):
    return np.exp(xlogy(x, l) - l - gammaln(x + 1))



def main():
    


if __name__ == "__main__":
    kpl.init_kplotlib()
    main()
    kpl.show(block=True)

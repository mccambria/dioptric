# -*- coding: utf-8 -*-
"""
Main text fig 3

Created on June 5th, 2024

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig

def main(data):
    


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=)

    main(data)

    plt.show(block=True)
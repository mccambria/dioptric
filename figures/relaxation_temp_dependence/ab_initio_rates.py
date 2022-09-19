import errno
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as patches
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
import pandas as pd
import utils.tool_belt as tool_belt
from utils.kplotlib import color_mpl_to_color_hex, lighten_color_hex
import utils.common as common
from scipy.odr import ODR, Model, RealData
import sys
from pathlib import Path
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
import temp_dependence_fitting
from temp_dependence_fitting import gamma_face_color, gamma_edge_color, omega_face_color, omega_edge_color, ratio_face_color, ratio_edge_color
import csv
import utils.kplotlib as kpl
from utils.kplotlib import (
    marker_size,
    line_width,
    marker_size_inset,
    line_width_inset,
)


marker_edge_width = line_width
marker_edge_width_inset = line_width_inset


def main():
    
    ### Params
    
    rates_y_range = [5e-3, 1000]
    rates_yscale = "log"
    ratio_y_range = [0,20]
    ratio_yscale = "linear"
    temp_range = [0, 480]
    xscale = "linear"
    
    ### Setup
    
    home = common.get_nvdata_dir()
    path = home / "paper_materials/relaxation_temp_dependence"

    # Fit to Omega and gamma simultaneously
    data_file_name = "compiled_data"
    data_points = temp_dependence_fitting.get_data_points(
        path, data_file_name, temp_range
    )
    (
        popt,
        pvar,
        beta_desc,
        omega_hopper_fit_func,
        omega_wu_fit_func,
        gamma_hopper_fit_func,
        gamma_wu_fit_func,
    ) = temp_dependence_fitting.fit_simultaneous(data_points, "double_orbach")
    omega_hopper_lambda = lambda temp: omega_hopper_fit_func(temp, popt)
    omega_wu_lambda = lambda temp: omega_wu_fit_func(temp, popt)
    gamma_hopper_lambda = lambda temp: gamma_hopper_fit_func(temp, popt)
    gamma_wu_lambda = lambda temp: gamma_wu_fit_func(temp, popt)
    
    sim_file_name = "Tdep_512_PBE.dat"

    min_temp = temp_range[0]
    max_temp = temp_range[1]
    linspace_min_temp = max(0, min_temp)
    temp_linspace = np.linspace(linspace_min_temp, max_temp, 1000)
    
    ### Figure prep
    
    fig, ax_rates, ax_ratio = plt.subplots(1, 2, figsize=kpl.double_figsize)
    
    for ax in [ax_rates, ax_ratio]:
        ax.set_xlabel(r"Temperature $\mathit{T}$ (K)")
        ax.set_xscale(xscale)
        ax.set_xlim(min_temp, max_temp)
        
    ax_rates.set_yscale(rates_yscale)
    ax_rates.set_ylim(rates_y_range[0], rates_y_range[1])
    ax_rates.set_ylabel(r"Relaxation rates (s$^{-1}$)")
    
    ax_ratio.set_yscale(ratio_yscale)
    ax_ratio.set_ylim(ratio_y_range[0], ratio_y_range[1])
    ax_ratio.set_ylabel(r"Predicted / model")
    


if __name__ == "__main__":

    kpl.init_kplotlib()
    main()
    plt.show(block=True)

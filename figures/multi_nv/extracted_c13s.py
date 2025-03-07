# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""

import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from numba import njit
from scipy.optimize import brute

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.tool_belt import curve_fit


def main(hfs_res, hfs_err_res, hfs_echo, hfs_err_echo):
    res_order = np.argsort(hfs_res)
    hfs_res = 1000 * np.array(hfs_res)[res_order]
    hfs_err_res = 1000 * np.array(hfs_err_res)[res_order]
    echo_order = np.argsort(hfs_echo)
    hfs_echo = np.array(hfs_echo)[echo_order]
    hfs_err_echo = np.array(hfs_err_echo)[echo_order]

    num_nvs = len(hfs_res) + len(hfs_echo)

    fig, ax = plt.subplots()
    nv_ind = 1
    for hfs_list, hfs_err_list, color, label in zip(
        (hfs_echo, hfs_res),
        (hfs_err_echo, hfs_err_res),
        (kpl.KplColors.BLUE, kpl.KplColors.RED),
        ("Spin echo", "ESR"),
    ):
        for ind in range(len(hfs_list)):
            kpl.plot_points(
                ax, nv_ind, hfs_list[ind], hfs_err_list[ind], color=color, label=label
            )
            nv_ind += 1
            label = None

    # From Smeltzer 2011
    for theory_val, theory_err in zip(
        [14.8, 13.9, 7.5, 5.7, 4.6, 4.67, 2.63, 2.27],
        [0.1, 0.1, 0.1, 0.2, 0.1, 0.04, 0.07, 0.04],
    ):
        # ax.axhline(theory_val, color=kpl.KplColors.LIGHT_GRAY, zorder=-50)
        ax.fill_between(
            [-1, num_nvs + 10],
            theory_val - theory_err,
            theory_val + theory_err,
            color=kpl.KplColors.LIGHT_GRAY,
            zorder=-50,
        )

    ax.set_xlabel("NV index")
    ax.set_ylabel("$^{13}$C hyperfine coupling (MHz)")
    ax.legend(loc=kpl.Loc.LOWER_RIGHT)
    margin = 0.8
    ax.set_xlim(-margin, num_nvs + margin)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


if __name__ == "__main__":
    kpl.init_kplotlib()

    # fmt: off
    # From ./resonance.py, in GHz
    hfs_res = [0.008270982638238914, 0.015881063467104776, 0.014010042750685282, 0.015391657472928187, 0.012955566280101407, 0.016983227280784243]
    hfs_err_res = [0.0016409452584717822, 0.0010983852602745553, 0.0004848082620682548, 0.0007214312144406817, 0.0006541485039380769, 0.0012675922107645444]
    # fmt: on
    # From ./spin_echo/spin_echo-mcc.py, in MHz
    data = dm.get_raw_data(file_id=1732403187814)
    popts = np.array(data["popts"])
    pcovs = np.array(data["pcovs"])
    pstes = np.array([np.sqrt(np.diag(pcovs[ind])) for ind in range(len(pcovs))])
    hfs_echo = popts[:, -2]
    hfs_err_echo = pstes[:, -2]

    main(hfs_res, hfs_err_res, hfs_echo, hfs_err_echo)

    plt.show(block=True)

# -*- coding: utf-8 -*-
"""
Take a picture of the sample by snake scan

Created on July 11th, 2023

@author: mccambria
"""


# region Import and constants

import numpy as np
import utils.common as common
import utils.tool_belt as tb
import utils.kplotlib as kpl
import utils.positioning as positioning
import matplotlib.pyplot as plt
import labrad
import time


# endregion
# region Functions


# endregion
# region Main


def main(nv_sig, x_range, y_range, num_steps, nv_minus_init=False):
    with labrad.connect(username="", password="") as cxn:
        return main_with_cxn(cxn, nv_sig, x_range, y_range, num_steps, nv_minus_init)


def main_with_cxn(cxn, nv_sig, x_range, y_range, num_steps, nv_minus_init):
    ### Basic setup
    opx = cxn.QM_opx

    # Just square for now
    x_num_steps = num_steps
    y_num_steps = num_steps

    tb.reset_cfm(cxn)
    coords = nv_sig["coords"]
    x_center, y_center, z_center = positioning.adjust_coords_for_drift(coords, cxn)

    ### Laser details
    xy_delay = 0
    laser_key = "imaging_laser"
    laser_name = nv_sig[laser_key]
    tb.set_filter(cxn, nv_sig, laser_key)
    time.sleep(1)
    readout_power = tb.set_laser_power(cxn, nv_sig, laser_key)
    readout = nv_sig["imaging_readout_dur"]

    # Prepare the sequence
    ret_vals = positioning.get_scan_grid_2d(
        x_center, y_center, x_range, y_range, x_num_steps, y_num_steps
    )
    seq_args = [xy_delay, readout, laser_name, readout_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "image_sample.py"

    # Stream it
    opx.stream_immediate(seq_file, seq_args_string)

    # tb.poll_safe_stop()


# endregion

if __name__ == "__main__":
    kpl.init_kplotlib()

    # plt.show(block=True)

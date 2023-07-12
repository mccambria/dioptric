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
    # print("test")

    ### Laser details
    laser_key = "imaging_laser"
    laser_name = nv_sig[laser_key]
    tb.set_filter(cxn, nv_sig, laser_key)
    time.sleep(1)
    # readout_power = tb.set_laser_power(cxn, nv_sig, laser_key)
    readout_power = 0.45
    readout = nv_sig["imaging_readout_dur"]

    # Prepare the sequence
    ret_vals = positioning.get_scan_grid_2d(
        x_center, y_center, x_range, y_range, x_num_steps, y_num_steps, dtype=int
    )
    # print(ret_vals[0])
    # print(ret_vals[1])

    # x_freqs, y_freqs, _, _, _ = positioning.get_scan_grid_2d(
    #     x_center, y_center, x_range, y_range, x_num_steps, y_num_steps, dtype=int
    # )
    # x_freqs = x_freqs.tolist()
    # y_freqs = y_freqs.tolist()
    # for x_freq, y_freq in zip(x_freqs, y_freqs):
    #     print(f"{x_freq}, {y_freq}")
    # return

    seq_args = [
        # Positioning
        x_center,
        y_center,
        x_range,
        y_range,
        x_num_steps,
        y_num_steps,
        # Laser
        readout,
        laser_name,
        readout_power,
    ]
    # print(seq_args)
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "image_sample.py"

    # Stream it
    print("stream start")
    opx.stream_immediate(seq_file, seq_args_string, -1)
    print("streaming")


# endregion

if __name__ == "__main__":
    kpl.init_kplotlib()

    # plt.show(block=True)

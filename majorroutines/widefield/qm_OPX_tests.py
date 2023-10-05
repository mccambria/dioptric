# -*- coding: utf-8 -*-
"""
Sequencing test with QM OPX

Created on June 19th, 2023

@author: mccambria
"""


# region Import and constants

import numpy as np
import utils.common as common
import utils.tool_belt as tb
import utils.kplotlib as kpl
import matplotlib.pyplot as plt
import labrad


# endregion
# region Functions


def poisson(val, param):
    return 1 + val


# endregion
# region Main


def main(nv_sig):
    with labrad.connect(username="", password="") as cxn:
        return main_with_cxn(cxn, nv_sig)


def main_with_cxn(cxn, nv_sig):
    opx = cxn.QM_opx

    # # Single RF tone
    # laser_name = "laserglow_589"
    # # Channel, freq, amplitude, duration
    # seq_args = [f"{laser_name}_x", 10, 0.4, 1000]
    # seq_args = ["ao1", 10, 0.4, 100]
    # seq_args_string = tb.encode_seq_args(seq_args)
    # seq_file = "rf_test.py"
    # opx.stream_immediate(seq_file, seq_args_string, -1)

    # Digital channels, analog channels, analog voltages, analog frequencies
    # v = 0.34  # 0.7 W, 520 nm
    v = 0.52  # 0.41 V = 1 W, 638 nm
    opx.constant_ac([], [1, 2, 3], [v, v, v], [75e6, 75e6, 75e6])
    # opx.constant_ac([], [3], [0.34], [110e6])
    # opx.constant_ac([], [1,], [0.1], [0])

    # x_freqs = np.arange(55, 96, 10)
    # y_freqs = np.arange(55, 96, 10)
    # # x_freqs = [65, 75, 85]
    # # y_freqs = [65, 75, 85]
    # for x_freq in x_freqs:
    #     for y_freq in y_freqs:
    #         print()
    #         print(f"coords in MHz: {x_freq}, {y_freq}")
    #         opx.constant_ac([], [2, 3], [v, v], [x_freq * 1e6, y_freq * 1e6])
    #         stop = input("Enter to advance or c to stop: ") == "c"
    #         if stop:
    #             break
    #     if stop:
    #         break

    # x_freq = 75
    # y_freq = 75
    # # powers = np.arange(0.5, 1.4, 0.1)
    # powers = [1.6, 1.7, 1.6, 1.8, 1.9, 2.0]
    # for p in powers:
    #     print()
    #     print(f"RF power: {p} W")
    #     v = 0.41 * np.sqrt(p)
    #     print(f"Voltage: {v} V")
    #     opx.constant_ac([], [2, 3], [v, v], [x_freq * 1e6, y_freq * 1e6])
    #     stop = input("Enter to advance or c to stop: ") == "c"
    #     if stop:
    #         break


# endregion

if __name__ == "__main__":
    config = common.get_config_dict()
    # print(config)
    # kpl.init_kplotlib()

    # plt.show(block=True)

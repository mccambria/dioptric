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
    with labrad.connect() as cxn:
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
    opx.constant_ac([], [1, 2, 3], [0.1, 0.4, 0.4], [0, 110e6, 110e6])
    # opx.constant_ac([], [1,], [0.1], [0])

    tb.poll_safe_stop()


# endregion

if __name__ == "__main__":
    config = common.get_config_dict()
    config = common.get_config_dict()
    print(config)
    # kpl.init_kplotlib()

    # plt.show(block=True)

# -*- coding: utf-8 -*-
"""
QM OPX sequence utils

Created June 25th, 2023

@author: mccambria
"""


from qm.qua import align, declare, assign, infinite_loop_, while_
from qm import qua
from utils import common
from utils.constants import ModMode


def handle_reps(one_rep, num_reps, wait_for_trigger=False):
    """Handle repetitions of a given sequence - you just have to pass
    a function defining the behavior for a single loop. Optionally
    waits for trigger pulse between loops.

    Parameters
    ----------
    one_rep : function
        QUA "macro" to be repeated
    num_reps : int
        Number of times to repeat, -1 for infinite loop
    wait_for_trigger : bool, optional
        Whether or not to pause execution between loops until a trigger
        pulse is received by the OPX, by default False
    """

    dummy_element = "do1"  # Just necessary for wait_for_trigger

    if num_reps == -1:
        with infinite_loop_():
            one_rep()
            if wait_for_trigger:
                qua.wait_for_trigger(dummy_element)
                align()
    else:
        handle_reps_ind = declare(int, value=0)
        with while_(handle_reps_ind < num_reps):
            one_rep()
            assign(handle_reps_ind, handle_reps_ind + 1)
            if wait_for_trigger:
                qua.wait_for_trigger(dummy_element)
                align()


def convert_ns_to_clock_cycles(duration_ns):
    return round(duration_ns / 4)


def get_laser_mod_element(laser_name):
    config = common.get_config_dict()
    mod_mode = config["Optics"][laser_name]["mod_mode"]
    if mod_mode == ModMode.ANALOG:
        laser_mod_element = f"ao_{laser_name}_am"
    elif mod_mode == ModMode.DIGITAL:
        laser_mod_element = f"do_{laser_name}_dm"
    return laser_mod_element

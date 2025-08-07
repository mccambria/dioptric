# -*- coding: utf-8 -*-
"""
Charge state readout after polarization/ionization, no spin manipulation

Created on October 13th, 2023

@author: mccambria
@author: schand
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils


def macro(
    pol_coords_list,
    pol_duration_list,
    pol_amp_list,
    ion_coords_list,
    num_reps,
    ion_do_target_list=None,
    verify_charge_states=False,
    pol_duration_override=None,
    pol_amp_override=None,
    readout_duration_override=None,
    readout_amp_override=None,
):
    if num_reps is None:
        num_reps = 1

    def one_exp(do_ionize):
        seq_utils.macro_polarize(
            pol_coords_list,
            pol_duration_list,
            pol_amp_list,
            pol_duration_override,
            pol_amp_override,
            targeted_polarization=verify_charge_states,
            verify_charge_states=verify_charge_states,
            spin_pol=False,
        )

        if do_ionize:
            seq_utils.macro_ionize(ion_coords_list, do_target_list=ion_do_target_list)

        seq_utils.macro_charge_state_readout(
            readout_duration_override, readout_amp_override
        )
        seq_utils.macro_wait_for_trigger()

    def one_rep(rep_ind=None):
        for do_ionize in [True, False]:
            one_exp(do_ionize)

    seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
    seq_utils.macro_pause()

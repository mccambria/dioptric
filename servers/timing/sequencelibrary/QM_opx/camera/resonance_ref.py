# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(
    base_scc_seq_args,
    step_vals=None,
    num_reps=1,
    reference=True,
    pol_duration_ns=None,
    ion_duration_ns=None,
    readout_duration_ns=None,
    phase=None,
):
    # if phase is not None:
    #     i_el, q_el = seq_utils.get_iq_mod_elements(uwave_ind)
    # phase_rad = phase * (np.pi / 180)
    # i_comp = 0.5 * np.cos(phase_rad)
    # q_comp = 0.5 * np.sin(phase_rad)
    # iq_pulse_dict = {0: , 90:}

    with qua.program() as seq:

        def uwave_macro_sig(uwave_ind_list, step_val):
            seq_utils.macro_pi_pulse(uwave_ind_list)

        base_scc_sequence.macro(
            base_scc_seq_args,
            uwave_macro_sig,
            step_vals,
            num_reps,
            pol_duration_ns,
            ion_duration_ns,
            readout_duration_ns,
            reference=reference,
        )

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [
                    [108.48124282165938, 109.79869381786162],
                    [108.92124282165938, 110.04969381786162],
                    [109.17324282165939, 110.39769381786162],
                ],
                [
                    [73.16298031205457, 75.08589052467828],
                    [73.43898031205457, 75.23289052467827],
                    [73.69798031205457, 75.49989052467826],
                ],
                [
                    100,
                    100,
                    100,
                ],
                [],
                [0, 1],
            ],
            [0],
        )

        sim_config = SimulationConfig(duration=int(500e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

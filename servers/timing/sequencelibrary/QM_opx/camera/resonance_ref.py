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
from servers.timing.sequencelibrary.QM_opx.camera import base_sequence


def get_seq(
    pol_coords_list,
    ion_coords_list,
    spin_flip_ind_list,
    uwave_ind,
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

    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind)
    buffer = seq_utils.get_widefield_operation_buffer()

    with qua.program() as seq:

        def uwave_macro_sig(step_val):
            qua.align()
            qua.play("pi_pulse", sig_gen_el)
            # if phase is not None:
            #     qua.play("pi_pulse", i_el)
            #     qua.play("pi_pulse", q_el)
            qua.wait(buffer, sig_gen_el)

        base_sequence.macro(
            pol_coords_list,
            ion_coords_list,
            spin_flip_ind_list,
            uwave_ind,
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
                [108.63547773676507, 108.73446819207585],
                [109.45547773676506, 110.64046819207584],
            ],
            [
                [73.55849816673624, 74.04886961198135],
                [74.19849816673624, 75.56986961198135],
            ],
            [],
            0,
            [0],
            1,
        )

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

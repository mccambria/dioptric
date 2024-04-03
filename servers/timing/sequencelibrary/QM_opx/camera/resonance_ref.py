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
    uwave_ind,
    step_vals=None,
    num_reps=1,
    reference=True,
    pol_duration_ns=None,
    uwave_duration_ns=None,
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
    uwave_duration = seq_utils.convert_ns_to_cc(uwave_duration_ns, allow_zero=True)
    buffer = seq_utils.get_widefield_operation_buffer()

    def uwave_macro_sig(step_val):
        if uwave_duration is None:
            qua.play("pi_pulse", sig_gen_el)
            # if phase is not None:
            #     qua.play("pi_pulse", i_el)
            #     qua.play("pi_pulse", q_el)
        else:
            if uwave_duration != 0:
                qua.play("on", sig_gen_el, duration=uwave_duration)
        qua.wait(buffer, sig_gen_el)

    seq = base_sequence.get_seq(
        pol_coords_list,
        ion_coords_list,
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
                [109.05560372660722, 110.77022466032236],
                [109.2856037266072, 111.45022466032236],
            ],
            [
                [73.78442169604547, 75.67270342527479],
                [74.02642169604547, 76.07570342527478],
            ],
            0,
            [
                2.784615384615385,
                2.8123076923076926,
                2.96,
                2.9323076923076923,
                2.9,
                2.8400000000000003,
                2.9184615384615387,
                2.9276923076923076,
            ],
            10,
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

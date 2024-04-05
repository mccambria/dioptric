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
    repol_coords_list,
    ion_coords_list,
    uwave_ind,
    step_vals,
    num_reps=1,
):
    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind)
    step_vals = [seq_utils.convert_ns_to_cc(el) for el in step_vals]
    buffer = seq_utils.get_widefield_operation_buffer()

    with qua.program() as seq:

        def uwave_macro_sig(step_val):
            seq_utils.macro_anticorrelate(repol_coords_list, uwave_ind)
            qua.align()
            qua.play("on", sig_gen_el, duration=step_val)
            qua.wait(buffer, sig_gen_el)

        base_sequence.macro(
            pol_coords_list, ion_coords_list, uwave_macro_sig, step_vals, num_reps
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
                [109.05560372660722, 110.77022466032236],
            ],
            [
                [73.78442169604547, 75.67270342527479],
                [74.02642169604547, 76.07570342527478],
            ],
            0,
            [1, 2, 3, 4],
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

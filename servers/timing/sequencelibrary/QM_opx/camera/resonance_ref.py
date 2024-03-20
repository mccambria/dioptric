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
    args,
    num_reps,
    reference=True,
    pol_duration_ns=None,
    uwave_duration_ns=None,
    ion_duration_ns=None,
    readout_duration_ns=None,
    phase=None,
):
    (pol_coords_list, ion_coords_list, uwave_ind) = args

    # if phase is not None:
    #     i_el, q_el = seq_utils.get_iq_mod_elements(uwave_ind)
    # phase_rad = phase * (np.pi / 180)
    # i_comp = 0.5 * np.cos(phase_rad)
    # q_comp = 0.5 * np.sin(phase_rad)
    # iq_pulse_dict = {0: , 90:}

    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind)
    uwave_duration = seq_utils.convert_ns_to_cc(uwave_duration_ns, allow_zero=True)
    buffer = seq_utils.get_widefield_operation_buffer()

    def uwave_macro_sig():
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
        num_reps,
        uwave_macro_sig,
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
        args = [
            [
                [112.8143831410256, 110.75435400118901],
                [112.79838314102561, 110.77035400118902],
            ],
            [
                [76.56091979499166, 75.8487161634141],
                [76.30891979499165, 75.96071616341409],
            ],
            0,
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(100e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

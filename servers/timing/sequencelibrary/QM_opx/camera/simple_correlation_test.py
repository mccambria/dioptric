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

# def get_seq(base_scc_seq_args, random_seed, num_reps=1):
#     with qua.program() as seq:
#         seq_utils.init()
#         seq_utils.macro_run_aods()
#         rand_val = qua.declare(int)
#         qua_random = qua.Random()
#         qua_random.set_seed(random_seed)

#         def uwave_macro_sig1(uwave_ind_list, step_val):
#             # Parity check for pi pulse
#             qua.assign(rand_val, qua_random.rand_int(2))
#             with qua.if_(qua.Cast.unsafe_cast_bool(rand_val)):
#                 seq_utils.macro_pi_pulse(uwave_ind_list)

#         def uwave_macro_sig2(uwave_ind_list, step_val):
#             # Parity check for pi pulse
#             qua.assign(rand_val, qua_random.rand_int(2))
#             with qua.if_(qua.Cast.unsafe_cast_bool(rand_val)):
#                 seq_utils.macro_pi_pulse(uwave_ind_list)
#             seq_utils.macro_pi_pulse(uwave_ind_list)

#         base_scc_sequence.macro(
#             base_scc_seq_args,
#             # uwave_macro_sig,
#             [uwave_macro_sig1, uwave_macro_sig2],
#             num_reps=num_reps,
#             reference=False,
#         )

#     seq_ret_vals = []
#     return seq, seq_ret_vals


# def get_seq(base_scc_seq_args, random_seed, num_reps=1):
#     with qua.program() as seq:
#         seq_utils.init()
#         seq_utils.macro_run_aods()
#         rand_val = qua.declare(int)
#         qua_random = qua.Random()
#         qua_random.set_seed(random_seed)

#         def uwave_macro_sig(uwave_ind_list, step_val):
#             # Parity check for pi pulse
#             qua.assign(rand_val, qua_random.rand_int(2))
#             with qua.if_(qua.Cast.unsafe_cast_bool(rand_val)):
#                 seq_utils.macro_pi_pulse(uwave_ind_list)

#         base_scc_sequence.macro(
#             base_scc_seq_args,
#             uwave_macro_sig,
#             # [uwave_macro_sig1, uwave_macro_sig2],
#             num_reps=num_reps,
#             reference=True,
#         )

#     seq_ret_vals = []
#     return seq, seq_ret_vals


def get_seq(base_scc_seq_args, num_reps=1):
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        qua_random = qua.Random()

        def uwave_macro_sig(uwave_ind_list, step_val):
            step_val = qua.declare(int)
            qua.assign(step_val, qua_random.rand_int(2))
            with qua.if_(step_val == 1):
                seq_utils.macro_pi_pulse(uwave_ind_list)

        def uwave_macro_ref(uwave_ind_list, step_val):
            return True

        base_scc_sequence.macro(
            base_scc_seq_args,
            [uwave_macro_sig, uwave_macro_ref],
            num_reps=num_reps,
            reference=False,
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
                [[109.114, 107.084], [110.232, 105.466], [110.468, 108.724]],
                [10000, 10000, 10000],
                [1.0, 1.0, 1.0],
                [[73.686, 72.605], [74.542, 71.289], [74.759, 73.921]],
                [116, 108, 108],
                [1.0, 1.0, 1.0],
                [False, True, False],
                [0, 1],
            ],
            5,
        )

        sim_config = SimulationConfig(duration=int(300e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

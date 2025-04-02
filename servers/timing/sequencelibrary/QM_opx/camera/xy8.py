# -*- coding: utf-8 -*-
"""
Widefield XY8 Coherence Sequence

Created on October 13th, 2023
Saroj Chand on March 22nd, 2025
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
import utils.tool_belt as tb
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, step_vals, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    uwave_ind_list = base_scc_seq_args[-1]
    macro_pi_pulse_duration = seq_utils.get_macro_pi_pulse_duration(uwave_ind_list)
    macro_pi_on_2_pulse_duration = seq_utils.get_macro_pi_on_2_pulse_duration(
        uwave_ind_list
    )

    # Adjust step values to compensate for internal delays
    step_vals = [
        seq_utils.convert_ns_to_cc(el) - macro_pi_pulse_duration for el in step_vals
    ]
    # Define XY8 pulse phase sequence (in radians)
    xy8_phases = [
        0,  # π_X
        np.pi / 2,  # π_Y
        0,  # π_X
        np.pi / 2,  # π_Y
        np.pi / 2,  # π_Y
        0,  # π_X
        np.pi / 2,  # π_Y
        0,  # π_X
    ]
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_val = qua.declare(int)

        def uwave_macro_sig(uwave_ind_list, step_val):
            qua.align()
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.wait(step_val)

            for i, phase in enumerate(xy8_phases):
                seq_utils.macro_pi_pulse(uwave_ind_list, phase=phase)
                if i < len(xy8_phases) - 1:
                    qua.wait(2 * step_val)  # 2τ between πs
                else:
                    qua.wait(step_val)  # τ after last π

            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.wait(buffer)

        # SBC this test to compare with other dual rail ref
        def uwave_macro_ref(uwave_ind_list, step_val):
            qua.align()
            qua.wait(step_val)
            for i in range(len(xy8_phases)):
                if i < len(xy8_phases) - 1:
                    qua.wait(2 * step_val)  # 2τ between πs
                else:
                    qua.wait(step_val)  # τ after last would-be π

            qua.wait(buffer)

        with qua.for_each_(step_val, step_vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig, uwave_macro_ref],
                step_val,
                num_reps,
                reference=False,
            )

    seq_ret_vals = []
    return seq, seq_ret_vals


# def get_seq(args, num_reps):
#     (pol_coords_list, ion_coords_list, tau_ns) = args

#     tau = seq_utils.convert_ns_to_cc(tau_ns)
#     half_tau = seq_utils.convert_ns_to_cc(tau_ns / 2)
#     buffer = seq_utils.get_widefield_operation_buffer()
#     sig_gen_el = seq_utils.get_sig_gen_element()
#     i_el, q_el = seq_utils.get_iq_mod_elements()
#     rabi_period = seq_utils.get_rabi_period()
#     pi_pulse_duration = int(rabi_period / 2)
#     pi_on_2_pulse_duration = int(rabi_period / 4)
#     adj_tau = tau - pi_on_2_pulse_duration
#     adj_2_tau = 2 * adj_tau

#     def y_pi_on_2_pulse():
#         qua.play("off", i_el)
#         qua.play("on", q_el)
#         qua.play("pi_on_2_pulse", sig_gen_el)

#     def x_pi_pulse():
#         qua.play("on", i_el)
#         qua.play("off", q_el)
#         qua.play("pi_pulse", sig_gen_el)

#     def y_pi_pulse():
#         qua.play("off", i_el)
#         qua.play("on", q_el)
#         qua.play("pi_pulse", sig_gen_el)

#     def uwave_macro():
#         y_pi_on_2_pulse()
#         qua.wait(adj_tau)

#         x_pi_pulse()
#         qua.wait(adj_2_tau)
#         y_pi_pulse()
#         qua.wait(adj_2_tau)
#         x_pi_pulse()
#         qua.wait(adj_2_tau)
#         y_pi_pulse()

#         qua.wait(adj_2_tau)

#         y_pi_pulse()
#         qua.wait(adj_2_tau)
#         x_pi_pulse()
#         qua.wait(adj_2_tau)
#         y_pi_pulse()
#         qua.wait(adj_2_tau)
#         x_pi_pulse()

#         qua.wait(adj_tau)
#         y_pi_on_2_pulse()

#         qua.align()

#     seq = base_scc_sequence.get_seq(
#         pol_coords_list, ion_coords_list, num_reps, uwave_macro
#     )

#     seq_ret_vals = []
#     return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    tb.set_delays_to_zero(opx_config)
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 10e3

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [[107.721, 107.702], [107.439, 105.963]],
                [164, 144],
                [1.0, 1.0],
                [[72.443, 73.218], [72.175, 71.806]],
                [88, 80],
                [1.0, 1.0],
                [False, False],
                [1],
            ],
            [
                9220,
                18796,
                312752,
                42920,
                1000,
                147776,
            ],
            1,
        )

        sim_config = SimulationConfig(duration=int(240e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(f"An error occurred: {exc}")
    finally:
        qmm.close_all_quantum_machines()

# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""


import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(args, num_reps):
    (pol_coords_list, ion_coords_list, tau_ns) = args

    tau = seq_utils.convert_ns_to_cc(tau_ns)
    half_tau = seq_utils.convert_ns_to_cc(tau_ns / 2)
    buffer = seq_utils.get_widefield_operation_buffer()
    sig_gen_el = seq_utils.get_sig_gen_element()
    i_el, q_el = seq_utils.get_iq_mod_elements()
    rabi_period = seq_utils.get_rabi_period()
    pi_pulse_duration = int(rabi_period / 2)
    pi_on_2_pulse_duration = int(rabi_period / 4)
    adj_tau = tau - pi_on_2_pulse_duration
    adj_2_tau = 2 * adj_tau

    def y_pi_on_2_pulse():
        qua.play("off", i_el)
        qua.play("on", q_el)
        qua.play("pi_on_2_pulse", sig_gen_el)

    def x_pi_pulse():
        qua.play("on", i_el)
        qua.play("off", q_el)
        qua.play("pi_pulse", sig_gen_el)

    def y_pi_pulse():
        qua.play("off", i_el)
        qua.play("on", q_el)
        qua.play("pi_pulse", sig_gen_el)

    def uwave_macro():
        y_pi_on_2_pulse()
        qua.wait(adj_tau)

        x_pi_pulse()
        qua.wait(adj_2_tau)
        y_pi_pulse()
        qua.wait(adj_2_tau)
        x_pi_pulse()
        qua.wait(adj_2_tau)
        y_pi_pulse()

        qua.wait(adj_2_tau)

        y_pi_pulse()
        qua.wait(adj_2_tau)
        x_pi_pulse()
        qua.wait(adj_2_tau)
        y_pi_pulse()
        qua.wait(adj_2_tau)
        x_pi_pulse()

        qua.wait(adj_tau)
        y_pi_on_2_pulse()

        qua.align()

    seq = base_scc_sequence.get_seq(
        pol_coords_list, ion_coords_list, num_reps, uwave_macro
    )

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(host=ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        args = [
            "laser_INTE_520",
            1000.0,
            [
                [112.8143831410256, 110.75435400118901],
                [112.79838314102561, 110.77035400118902],
            ],
            "laser_COBO_638",
            200,
            [
                [76.56091979499166, 75.8487161634141],
                [76.30891979499165, 75.96071616341409],
            ],
            "laser_OPTO_589",
            3500.0,
            "sig_gen_STAN_sg394",
            96 / 2,
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(500e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

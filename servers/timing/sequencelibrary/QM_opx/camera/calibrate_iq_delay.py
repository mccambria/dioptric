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
    (pol_coords_list, ion_coords_list, uwave_ind, i_or_q, tau_ns) = args

    tau = seq_utils.convert_ns_to_cc(tau_ns, raise_error=False)
    buffer = seq_utils.get_widefield_operation_buffer()
    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind)
    i_el, q_el = seq_utils.get_iq_mod_elements(uwave_ind)
    iq_el = i_el if i_or_q else q_el

    def uwave_macro():
        # IQ
        # qua.ramp_to_zero(iq_el)
        # qua.play("on", iq_el)
        qua.wait(buffer - tau, iq_el)
        qua.play("on", iq_el)
        # qua.play("pi_pulse", iq_el)
        # qua.wait(buffer + tau, iq_el)
        # Pi pulse
        qua.wait(buffer, sig_gen_el)
        qua.play("pi_pulse", sig_gen_el)
        qua.wait(buffer, sig_gen_el)
        #
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
            [
                [112.8143831410256, 110.75435400118901],
                [112.79838314102561, 110.77035400118902],
            ],
            [
                [76.56091979499166, 75.8487161634141],
                [76.30891979499165, 75.96071616341409],
            ],
            0,
            False,
            10,
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

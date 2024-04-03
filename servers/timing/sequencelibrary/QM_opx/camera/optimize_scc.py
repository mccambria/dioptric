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


def get_seq(
    pol_coords_list, ion_coords_list, uwave_ind, ion_duration_ns_list, num_reps
):
    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind)
    buffer = seq_utils.get_widefield_operation_buffer()

    def sig_exp():
        qua.align()
        qua.play("pi_pulse", sig_gen_el)
        qua.wait(buffer, sig_gen_el)

    def ref_exp():
        pass

    uwave_macro_list = [sig_exp, ref_exp]
    num_exps_per_rep = len(uwave_macro_list)

    ion_duration_list = [seq_utils.convert_ns_to_cc(el) for el in ion_duration_ns_list]

    with qua.program() as seq:
        seq_utils.init()
        ion_duration = qua.declare(int)

        def one_exp(exp_ind):
            seq_utils.macro_polarize(pol_coords_list)
            uwave_macro_list[exp_ind]()
            seq_utils.macro_scc(ion_coords_list, ion_duration, pol_coords_list)
            seq_utils.macro_charge_state_readout()
            seq_utils.macro_wait_for_trigger()

        def one_rep():
            for exp_ind in range(num_exps_per_rep):
                one_exp(exp_ind)

        def one_step():
            seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
            seq_utils.macro_pause()

        with qua.for_each_(ion_duration, ion_duration_list):
            one_step()

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
                [112.21219579120823, 110.40003798562638],
                [112.10719579120823, 110.9080379856264],
            ],
            [
                [75.99059786642306, 75.34468901215536],
                [75.64159786642307, 76.07968901215536],
            ],
            0,
            [1000, 200, 16],
            5,
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

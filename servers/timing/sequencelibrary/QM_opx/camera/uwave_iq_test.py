"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(seq_args, num_reps=-1):
    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind=1)
    i_el = seq_utils.get_sig_gen_i_element(uwave_ind=1)
    q_el = seq_utils.get_sig_gen_q_element(uwave_ind=1)

    with qua.program() as seq:

        def one_rep_macro():
            # I=0.5 for 1 us
            qua.play("iq_test", sig_gen_el)
            qua.play("iq_test", i_el)
            # qua.play("iq_test", q_el)
            qua.align()

            # I=-0.5 for 1 us
            qua.play("iq_test" * qua.amp(-1.0), i_el)
            # qua.play("iq_test" * qua.amp(-1.0), q_el)
            qua.align()

            # # Q=0.5 for 1 us
            # qua.play("iq_test", q_el)
            # qua.align()

            # # Q=-0.5 for 1 us
            # qua.play("iq_test" * qua.amp(-1.0), q_el)
            # qua.align()

        seq_utils.handle_reps(one_rep_macro, num_reps, wait_for_trigger=False)

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    # test = seq_utils.get_sig_gen_i_element(uwave_ind=1)
    # print(test)
    # sys.exit()

    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq([])

        sim_config = SimulationConfig(duration=int(20e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(f"An error occurred: {exc}")
    finally:
        qmm.close_all_quantum_machines()

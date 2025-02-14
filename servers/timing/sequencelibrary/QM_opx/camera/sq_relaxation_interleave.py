# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence
from utils import common
from utils import tool_belt as tb
from utils.constants import NVSpinState


def get_seq(
    base_scc_seq_args,
    step_vals,
    num_reps=1,
):
    # step_vals = [seq_utils.convert_ns_to_cc(el) for el in step_vals]
    print(step_vals)
    with qua.program() as seq:
        ### init
        seq_utils.init()
        seq_utils.macro_run_aods()

        step_val = qua.declare(int)

        def uwave_macro_0(uwave_ind_list, step_val):
            qua.align()
            qua.wait(step_val)

        def uwave_macro_1(uwave_ind_list, step_val):
            qua.align()
            qua.wait(step_val)
            seq_utils.macro_pi_pulse(uwave_ind_list)

        # base_scc_sequence.macro(
        #     base_scc_seq_args,
        #     [uwave_macro_0, uwave_macro_1],
        #     step_vals,
        #     num_reps,
        #     reference=False,
        # )

        def one_step():
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_0, uwave_macro_1],
                step_vals=step_vals,
                num_reps=num_reps,
                reference=False,
            )

        with qua.for_each_(step_val, step_vals):
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
        args = [
            [
                [[108.826, 106.773], [110.002, 105.138], [110.183, 108.395]],
                [1000, 1000, 1000],
                [1.0, 1.0, 1.0],
                [[73.347, 72.222], [74.196, 70.89], [74.442, 73.533]],
                [240, 240, 240],
                [1.0, 1.0, 1.0],
                [False, False, False],
                [0, 1],
            ],
            [
                1493672.0,
                3583572.0,
                579820.0,
                2380772.0,
                12302064.0,
                1141692.0,
                356880.0,
                20001000.0,
                2936488.0,
                6236064.0,
                7425680.0,
                14489072.0,
                1000.0,
            ],
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(400e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

"""
Widefield Ramsey Phase Scan Test

Created on March 25th, 2025

@author: mccambria
@author: sbchand
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

    phi_vals = step_vals  # If looping inside QUA
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        phi = qua.declare(qua.fixed)

        def uwave_macro_sig(uwave_ind_list, step_val):
            phi = step_val
            qua.align()
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
            # qua.wait(24)  # wait for 24 clock cycle corresponding to 96ns
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=phi)
            qua.wait(buffer)

        with qua.for_each_(phi, phi_vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig],
                step_val=phi,
                num_reps=num_reps,
            )

    seq_ret_vals = []
    return seq, seq_ret_vals


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
                [[107.715, 107.718], [107.433, 105.978]],
                [164, 144],
                [1.0, 1.0],
                [[72.438, 73.231], [72.171, 71.818]],
                [88, 80],
                [1.0, 1.0],
                [False, False],
                [1],
            ],
            [
                0.6283185307179586,
                1.0471975511965976,
                2.5132741228718345,
                3.7699111843077517,
                5.654866776461628,
            ],
        )

        sim_config = SimulationConfig(duration=int(50e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(f"An error occurred: {exc}")
    finally:
        qmm.close_all_quantum_machines()

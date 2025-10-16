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

def emit_phase_switch(uwave_ind_list, phi_var, allowed_phases):
    # Build a QUA switch with the compile-time list of allowed phases
    with qua.switch_(phi_var):
        for ph in allowed_phases:
            with qua.case_(int(ph)):
                seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=int(ph))
# tau = 16 ## ns
# tau = 19.6e3 ## revival
# tau_ticks = seq_utils.convert_ns_to_cc(tau)
# two_tau_ticks = 2 * tau_ticks 

def get_seq(base_scc_seq_args, step_vals, evol_time, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    allowed_cases = sorted({int(x % 360) for x in step_vals if int(x % 360) != 360})
    tau_ticks = seq_utils.convert_ns_to_cc(evol_time)
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_val = qua.declare(int)
        
        def uwave_macro_sig(uwave_ind_list, phi):
            qua.align()
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.wait(tau_ticks)
            seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
            qua.wait(tau_ticks)
            emit_phase_switch(uwave_ind_list, phi, allowed_cases)
            qua.wait(buffer)


        # def uwave_macro_sig(uwave_ind_list, phi):
        #     qua.align()
        #     seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
        #     qua.wait(tau_ticks)
        #     seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
        #     qua.wait(two_tau_ticks)
        #     seq_utils.macro_pi_pulse(uwave_ind_list, phase=90)
        #     qua.wait(two_tau_ticks)
        #     seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
        #     qua.wait(two_tau_ticks)
        #     seq_utils.macro_pi_pulse(uwave_ind_list, phase=90)
        #     qua.wait(two_tau_ticks)
        #     seq_utils.macro_pi_pulse(uwave_ind_list, phase=90)
        #     qua.wait(two_tau_ticks)
        #     seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
        #     qua.wait(two_tau_ticks)
        #     seq_utils.macro_pi_pulse(uwave_ind_list, phase=90)
        #     qua.wait(two_tau_ticks)
        #     seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
        #     qua.wait(tau_ticks)
        #     emit_phase_switch(uwave_ind_list, phi, allowed_cases)
        #     qua.wait(buffer)
        
        
        
        # def uwave_macro_sig(uwave_ind_list, phi):
        #     qua.align()
        #     seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
        #     seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
        #     # seq_utils.macro_pi_pulse(uwave_ind_list, phase=90)
        #     # seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
        #     # seq_utils.macro_pi_pulse(uwave_ind_list, phase=90)
        #     # seq_utils.macro_pi_pulse(uwave_ind_list, phase=90)
        #     # seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
        #     # seq_utils.macro_pi_pulse(uwave_ind_list, phase=90)
        #     # seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
        #     with qua.switch_(phi):
        #         with qua.case_(0):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
        #         with qua.case_(18):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=18)
        #         with qua.case_(36):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=36)
        #         with qua.case_(54):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=54)
        #         with qua.case_(72):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=72)
        #         with qua.case_(90):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=90)
        #         with qua.case_(108):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=108)
        #         with qua.case_(126):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=126)
        #         with qua.case_(144):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=144)
        #         with qua.case_(162):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=162)
        #         with qua.case_(180):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=180)
        #         with qua.case_(198):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=198)
        #         with qua.case_(216):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=216)
        #         with qua.case_(234):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=234)
        #         with qua.case_(252):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=252)
        #         with qua.case_(270):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=270)
        #         with qua.case_(288):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=288)
        #         with qua.case_(306):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=306)
        #         with qua.case_(324):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=324)
        #         with qua.case_(342):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=342)
        #         with qua.case_(360):
        #             seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=360)
        #     qua.wait(buffer)

        with qua.for_each_(step_val, step_vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig],
                step_val=step_val,
                num_reps=num_reps,
            )

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    tb.set_delays_to_zero(opx_config)
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 2e3

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
            [54],
        )

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(f"An error occurred: {exc}")
    finally:
        qmm.close_all_quantum_machines()

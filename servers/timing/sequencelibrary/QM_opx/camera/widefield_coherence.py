# -*- coding: utf-8 -*-
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

def get_seq(base_scc_seq_args, step_vals, evol_time, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    allowed_cases = sorted({int(x % 360) for x in step_vals if int(x % 360) != 360})
    tau_ticks = seq_utils.convert_ns_to_cc(evol_time)
    two_tau_ticks = 2 * tau_ticks 

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_val = qua.declare(int)
        
        # def uwave_macro_sig(uwave_ind_list, phi):
        #     qua.align()
        #     seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
        #     qua.wait(tau_ticks)
        #     seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
        #     qua.wait(tau_ticks)
        #     emit_phase_switch(uwave_ind_list, phi, allowed_cases)
        #     qua.wait(buffer)

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
            
            
        def uwave_macro_sig(uwave_ind_list, phi):
            qua.align()
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.wait(tau_ticks)
            seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
            qua.wait(two_tau_ticks)
            seq_utils.macro_pi_pulse(uwave_ind_list, phase=90)
            qua.wait(two_tau_ticks)
            seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
            qua.wait(two_tau_ticks)
            seq_utils.macro_pi_pulse(uwave_ind_list, phase=90)
            qua.wait(tau_ticks)
            emit_phase_switch(uwave_ind_list, phi, allowed_cases)
            qua.wait(buffer)
        
        
        
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
            200,
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
        
        
        
# """
# Widefield Coherence Sequences (single file)
# Includes: Ramsey, Spin Echo, XY4, XY8, XY16 (optionally phase scan)

# Upated: 2026-01-05 (Saroj Chand)
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from qm import QuantumMachinesManager, qua
# from qm.simulate import SimulationConfig

# import utils.common as common
# import utils.tool_belt as tb
# from servers.timing.sequencelibrary.QM_opx import seq_utils
# from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


# # -------------------------
# # Helpers
# # -------------------------
# def xy_phases(order: int):
#     """Return list of pi-pulse phases (deg) for one XY cycle."""
#     if order == 4:
#         return [0, 90, 0, 90]
#     if order == 8:
#         return [0, 90, 0, 90, 90, 0, 90, 0]
#     if order == 16:
#         # common XY16: XYXYYXYX YXYXXYXY (phases in deg)
#         return [0,90,0,90, 90,0,90,0,  90,0,90,0,  0,90,0,90]
#     raise ValueError("XY order must be 4, 8, or 16")


# def emit_phase_switch(uwave_ind_list, phi_var, allowed_phases_deg):
#     """Final π/2 with selectable phase via compile-time switch cases."""
#     with qua.switch_(phi_var):
#         for ph in allowed_phases_deg:
#             with qua.case_(int(ph)):
#                 seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=int(ph))


# def make_uwave_macro(seq_type: str, tau_ticks: int, buffer_ticks: int, allowed_phases_deg, xy_order=8, n_cycles=1):
#     """
#     Returns a function uwave_macro_sig(uwave_ind_list, phi) that base_scc_sequence.macro can call.
#     seq_type: "ramsey", "spin_echo", "xy4", "xy8", "xy16"
#     """
#     seq_type = seq_type.lower()
#     if seq_type.startswith("xy"):
#         # parse order if passed like "xy8"
#         try:
#             xy_order = int(seq_type.replace("xy", ""))
#         except Exception:
#             pass
#         seq_type = "xy"

#     xy_list = xy_phases(xy_order) if seq_type == "xy" else None

#     def uwave_macro_sig(uwave_ind_list, phi):
#         qua.align()

#         # Start: π/2 about X
#         seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)

#         if seq_type == "ramsey":
#             # π/2 - τ - π/2(phase)
#             qua.wait(tau_ticks)

#         elif seq_type == "spin_echo":
#             # π/2 - τ - π - τ - π/2(phase)
#             qua.wait(tau_ticks)
#             seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
#             qua.wait(tau_ticks)

#         elif seq_type == "xy":
#             # π/2 - [XYk]x n_cycles - π/2(phase)
#             # Implement as τ - π(phase) - 2τ - π(...) ... - τ
#             # (You can tweak this timing convention to match your lab standard.)
#             # π/2 - [XYk]x n_cycles - π/2(phase)
#             for cyc in range(n_cycles):
#                 for j, ph in enumerate(xy_list):
#                     qua.wait(tau_ticks)
#                     seq_utils.macro_pi_pulse(uwave_ind_list, phase=int(ph))

#                     is_last_pi = (cyc == n_cycles - 1) and (j == len(xy_list) - 1)
#                     qua.wait(tau_ticks if is_last_pi else 2 * tau_ticks)

#         else:
#             raise ValueError(f"Unknown seq_type: {seq_type}")

#         # End: π/2 with selectable phase (phase scan)
#         emit_phase_switch(uwave_ind_list, phi, allowed_phases_deg)

#         # buffer for camera / next ops
#         qua.wait(buffer_ticks)

#     return uwave_macro_sig


# def get_seq(base_scc_seq_args,
#             step_vals,
#             evol_time,
#             seq_type="xy8",
#             n_cycles=1,
#             num_reps=1):
#     """
#     step_vals_deg: list of phases (deg) for final π/2 (compile-time switch cases)
#     evol_time_ns: tau (ns)
#     seq_type: "ramsey", "spin_echo", "xy4", "xy8", "xy16"
#     """
#     buffer_ticks = seq_utils.get_widefield_operation_buffer()
#     tau_ticks = seq_utils.convert_ns_to_cc(evol_time)

#     allowed_cases = sorted({int(x % 360) for x in step_vals if int(x % 360) != 360})

#     with qua.program() as seq:
#         seq_utils.init()
#         seq_utils.macro_run_aods()

#         step_val = qua.declare(int)

#         uwave_macro_sig = make_uwave_macro(
#             seq_type=seq_type,
#             tau_ticks=tau_ticks,
#             buffer_ticks=buffer_ticks,
#             allowed_phases_deg=allowed_cases,
#             n_cycles=n_cycles,
#         )

#         with qua.for_each_(step_val, step_vals):
#             base_scc_sequence.macro(
#                 base_scc_seq_args,
#                 [uwave_macro_sig], 
#                 step_val=step_val,
#                 num_reps=num_reps,
#             )

#     return seq, []

# -------------------------
#  simulation
# -------------------------
# if __name__ == "__main__":
#     config_module = common.get_config_module()
#     config = config_module.config
#     opx_config = config_module.opx_config
#     tb.set_delays_to_zero(opx_config)
#     opx_config["pulses"]["yellow_spin_pol"]["length"] = 2e3

#     qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
#     qmm = QuantumMachinesManager(**qm_opx_args)
#     opx = qmm.open_qm(opx_config)
#     # Example base args (same style as yours)
#     base_args = [
#         [[107.715, 107.718], [107.433, 105.978]],
#         [164, 144],
#         [1.0, 1.0],
#         [[72.438, 73.231], [72.171, 71.818]],
#         [88, 80],
#         [1.0, 1.0],
#         [False, False],
#         [0],  # uwave_ind_list, etc
#     ]

#     try:
#         seq_type = "xy4"      # "ramsey", "spin_echo", "xy4", "xy8", "xy16"
#         tau_ns   = 200
#         phases = [0, 90, 180, 270]
#         seq, _ = get_seq(base_args, phases, tau_ns, seq_type=seq_type, n_cycles=1, num_reps=1)

#         sim_config = SimulationConfig(duration=int(200e3 / 4))
#         sim = opx.simulate(seq, sim_config)
#         samples = sim.get_simulated_samples()
#         samples.con1.plot()
#         plt.show(block=True)
#     finally:
#         qmm.close_all_quantum_machines()

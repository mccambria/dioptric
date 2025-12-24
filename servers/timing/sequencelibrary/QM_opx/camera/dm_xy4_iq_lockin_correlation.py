# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: Saroj Chand
"""

# -*- coding: utf-8 -*-
"""
DM lock-in search sequence (widefield) using XY4-N and 4-shot I/Q phase cycling.

Experiments per rep (in this order):
  0: I+  (final pi/2 phase = +X)
  1: I-  (final pi/2 phase = -X)
  2: Q+  (final pi/2 phase = +Y)
  3: Q-  (final pi/2 phase = -Y)
  4: optional reference (MW off)

Input args (encoded by tb.encode_seq_args):
  base_scc_seq_args, tau_ns, n_xy4_blocks, include_ref

tau_ns is the Hahn-tau convention you already use:
  - Hahn total evolution = 2*tau
  - lock-in center freq ~ 1/(2*tau)

This implementation uses XY4 with pulses separated by 2*tau:
  (pi/2)_X - tau - [pi_X - 2tau - pi_Y - 2tau - pi_X - 2tau - pi_Y] - tau - (pi/2)_phi
Repeated n_xy4_blocks times with continuous 2tau spacing between pi pulses.

@author: Saroj Chand (DM lock-in)
"""

import numpy as np
from qm import qua
import utils.tool_belt as tb
import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


# -----------------
# Phase conventions
# -----------------
# Your code previously used phase=99 to mean ~+Y (quadrature).
# Here we use 0/90/180/270 degrees by default.
# If your seq_utils expects different numeric codes, change these 4 constants.
PH_X   = 0
PH_Y   = 90
PH_mX  = 180
PH_mY  = 270


def _xy4_core(uwave_ind_list, tau_cc, final_phase_deg):
    """
    XY4-N core with interpulse spacing 2*tau and endcaps tau.
    Returns False (do not skip spin flip).
    """
    two_tau_cc = 2 * tau_cc

    # Entry pi/2 around X
    seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=PH_X)

    # Endcap
    qua.wait(tau_cc)

    # Total pi pulses: 4*n_blocks (pattern X, Y, X, Y repeated)
    total_pis = 4 
    for k in range(total_pis):
        # XYXY...
        phase = PH_X if (k % 2 == 0) else PH_Y
        seq_utils.macro_pi_pulse(uwave_ind_list, phase=phase)

        # Wait: 2tau between pi pulses; after the last pi, wait tau
        if k < total_pis - 1:
            qua.wait(two_tau_cc)
        else:
            qua.wait(tau_cc)

    # Exit pi/2 in requested quadrature/readout phase
    seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=final_phase_deg)

    return False


def get_seq(base_scc_seq_args, tau,num_reps=1):
    tau_cc = seq_utils.convert_ns_to_cc(tau)
    include_ref = False
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        # --- Define 4 signal experiments (I+/I-/Q+/Q-) ---
        def uwave_Ip(uwave_ind_list, step_val):
            return _xy4_core(uwave_ind_list, tau_cc,PH_X)

        def uwave_Im(uwave_ind_list, step_val):
            return _xy4_core(uwave_ind_list, tau_cc, PH_mX)

        def uwave_Qp(uwave_ind_list, step_val):
            return _xy4_core(uwave_ind_list, tau_cc, PH_Y)

        def uwave_Qm(uwave_ind_list, step_val):
            return _xy4_core(uwave_ind_list, tau_cc, PH_mY)

        def uwave_ref(uwave_ind_list, step_val):
            # MW off reference (keeps optical/SCC timing identical)
            return True

        uwave_macros = [uwave_Ip, uwave_Im, uwave_Qp, uwave_Qm]
        if include_ref:
            uwave_macros.append(uwave_ref)

        base_scc_sequence.macro(
            base_scc_seq_args,
            uwave_macros,
            num_reps=num_reps,
            reference=False,  # do NOT auto-append another reference
        )

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    # Local simulation hook (optional)
    import matplotlib.pyplot as plt
    from qm import QuantumMachinesManager
    from qm.simulate import SimulationConfig

    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    tb.set_delays_to_zero(opx_config)

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, _ = get_seq(
            [
                [[109.114, 107.084], [110.468, 108.724]],
                [1000, 1000],
                [1.0, 1.0],
                [[73.686, 72.605], [74.759, 73.921]],
                [116, 108],
                [1.0, 1.0],
                [False, False],
                [0, 1],
            ],
            tau=7.5e3,            # ns
            num_reps=1,
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

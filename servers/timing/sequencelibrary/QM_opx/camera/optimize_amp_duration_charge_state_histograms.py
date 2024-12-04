# -*- coding: utf-8 -*-
"""
Charge state readout after polarization/ionization, no spin manipulation

Created on October 13th, 2024

@author: Saroj Chand
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_charge_state_histograms


def get_seq(
    pol_coords_list,
    pol_duration_list,
    pol_amp_list,
    ion_coords_list,
    step_vals,
    optimize_pol_or_readout,
    optimize_type,
    num_reps,
):
    if optimize_type in ["duration", "both"]:
        step_vals[:, 0] = [seq_utils.convert_ns_to_cc(el) for el in step_vals[:, 0]]

    with qua.program() as seq:
        num_nvs = len(pol_coords_list)
        seq_utils.init(num_nvs)
        seq_utils.macro_run_aods()

        # Declare override variables
        if optimize_type == "both":
            duration_override = qua.declare(int)
            amp_override = qua.declare(qua.fixed)
        elif optimize_type == "duration":
            override_var = qua.declare(int)
        else:
            override_var = qua.declare(qua.fixed)

        # Determine which variables to override
        pol_duration_override = None
        pol_amp_override = None
        readout_duration_override = None
        readout_amp_override = None
        if optimize_pol_or_readout:
            if optimize_type == "both":
                pol_duration_override = duration_override
                pol_amp_override = amp_override
            elif optimize_type == "duration":
                pol_duration_override = override_var
            else:
                pol_amp_override = override_var
        else:
            if optimize_type == "both":
                readout_duration_override = duration_override
                readout_amp_override = amp_override
            elif optimize_type == "duration":
                readout_duration_override = override_var
            else:
                readout_amp_override = override_var

        def one_step():
            base_charge_state_histograms.macro(
                pol_coords_list,
                pol_duration_list,
                pol_amp_list,
                ion_coords_list,
                num_reps,
                pol_duration_override=pol_duration_override,
                pol_amp_override=pol_amp_override,
                readout_duration_override=readout_duration_override,
                readout_amp_override=readout_amp_override,
            )

        if optimize_type == "both":
            with qua.for_each_(duration_override, step_vals[:, 0]):
                with qua.for_each_(amp_override, step_vals[:, 1]):
                    one_step()
        else:
            with qua.for_each_(override_var, step_vals):
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
        seq_args = [
            [
                [108.793, 106.879],
                [109.929, 105.244],
                [110.15, 108.507],
                [107.701, 104.991],
                [106.534, 106.305],
            ],
            [1000, 1000, 1000, 1000, 1000],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [
                [73.495, 72.409],
                [74.355, 71.074],
                [74.614, 73.732],
                [72.581, 70.875],
                [71.689, 71.95],
            ],
            np.array(
                [
                    [4000, 0.2],
                    [8000, 0.4],
                    [12000, 0.6],
                    [16000, 0.8],
                    [20000, 1.0],
                ]
            ),
            True,
            "both",
            4,
        ]
        seq, seq_ret_vals = get_seq(*seq_args)

        sim_config = SimulationConfig(duration=int(150e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
        plt.show(block=True)

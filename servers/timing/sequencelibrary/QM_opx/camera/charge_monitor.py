# -*- coding: utf-8 -*-
"""
Charge state readout after polarization/ionization, no spin manipulation

Created on October 13th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils


def get_seq(pol_coords_list, charge_prep, dark_time_ns, num_reps):
    if num_reps is None:
        num_reps = 1
    num_nvs = len(pol_coords_list)

    dark_time = seq_utils.convert_ns_to_cc(dark_time_ns, allow_zero=True)

    with qua.program() as seq:
        seq_utils.init(num_nvs)
        seq_utils.macro_run_aods()

        def one_rep():
            if charge_prep:
                seq_utils.macro_polarize(
                    pol_coords_list, spin_pol=False, targeted_polarization=True
                )
            seq_utils.macro_charge_state_readout()
            seq_utils.macro_wait_for_trigger()
            if dark_time > 0:
                qua.wait(dark_time)

        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
        seq_utils.macro_pause()

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
                [110, 109.51847988358679],
                [112, 110.70156405156148],
            ],
            [
                [75.42725784791932, 75.65982013416432],
                [75.98725784791932, 74.74382013416432],
            ],
            False,
            True,
        ]
        seq, seq_ret_vals = get_seq(*args, 5)

        sim_config = SimulationConfig(duration=int(400e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
        plt.show(block=True)

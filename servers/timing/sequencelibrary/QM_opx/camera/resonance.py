# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import cProfile
import time

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig

from servers.timing.sequencelibrary.QM_opx.camera import resonance_ref
from utils import common


def get_seq(
    pol_coords_list,
    repol_coords_list,
    ion_coords_list,
    uwave_ind,
    uwave_duration_ns,
    num_reps,
):
    return resonance_ref.get_seq(
        pol_coords_list,
        repol_coords_list,
        ion_coords_list,
        uwave_ind,
        num_reps=num_reps,
        reference=False,
        uwave_duration_ns=uwave_duration_ns,
    )


# def get_seq(args, num_reps):
#     # (pol_coords_list, ion_coords_list, uwave_ind) = args
#     return resonance_ref.get_seq(args, num_reps, reference=False)


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
                [112.21219579120823, 110.40003798562638],
                [112.10719579120823, 110.9080379856264],
            ],
            [
                [75.99059786642306, 75.34468901215536],
                [75.64159786642307, 76.07968901215536],
            ],
            0,
        ]

        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    # finally:
    #     qmm.close_all_quantum_machines()

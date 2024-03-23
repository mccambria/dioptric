# -*- coding: utf-8 -*-
"""
Scanning illumination and widefield collection

Created on October 13th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy
from qm import QuantumMachinesManager, generate_qua_script, qua
from qm.simulate import SimulationConfig

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_sequence
from utils import common
from utils.constants import IonPulseType


def get_seq(args, num_reps):
    (pol_coords_list, ion_coords_list) = args

    def uwave_macro():
        pass

    seq = base_sequence.get_seq(
        pol_coords_list,
        ion_coords_list,
        num_reps,
        uwave_macro,
        ion_pulse_type=IonPulseType.ION,
    )

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(host=ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        args = [
            5000.0,
            "laser_OPTO_589",
            True,
            "laser_INTE_520",
            [111.202, 109.801],
            1000,
            False,
            "laser_COBO_638",
            [75, 75],
            2000.0,
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

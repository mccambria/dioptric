# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""


from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig
from servers.timing.sequencelibrary.QM_opx import seq_utils as seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import resonance_ref
import utils.common as common
import matplotlib.pyplot as plt


def get_seq(args, num_reps):
    return resonance_ref.get_seq(args, num_reps, reference=False)


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(host=ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        args = [
            "laser_INTE_520",
            10000.0,
            [
                [112.18352668291094, 110.34002169977755],
                [112.07852668291093, 110.84802169977756],
                [110.83652668291093, 111.82802169977755],
                [111.27252668291094, 109.38302169977756],
                [112.20752668291094, 109.49302169977756],
                [112.92852668291094, 110.40002169977755],
                [111.47052668291093, 107.64302169977756],
                [111.55952668291094, 107.65202169977756],
                [112.06852668291094, 107.02702169977756],
                [110.99552668291093, 110.24502169977755],
            ],
            "laser_COBO_638",
            220,
            [
                [75.97049095236301, 75.29800391512516],
                [75.62149095236302, 76.03300391512516],
                [75.03949095236301, 76.68000391512516],
                [75.18649095236302, 74.79700391512516],
                [75.93349095236302, 74.83800391512516],
                [76.54149095236302, 75.46700391512516],
                [75.42949095236301, 73.32500391512517],
                [75.28149095236301, 73.24000391512516],
                [75.80649095236302, 72.78900391512516],
                [75.00549095236302, 75.30800391512516],
            ],
            "laser_OPTO_589",
            35000000.0,
            "sig_gen_STAN_sg394",
            64,
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(2000e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

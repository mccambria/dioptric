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
    uwave_duration_ns = args.pop()
    uwave_ind = args.pop()
    return resonance_ref.get_seq(
        args, num_reps, uwave_duration_ns=uwave_duration_ns, reference=False
    )


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(host=ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        args = [
            None,
            [
                [112.21219579120823, 110.40003798562638],
                [112.10719579120823, 110.9080379856264],
                [110.86519579120822, 111.88803798562638],
                [111.30119579120823, 109.44303798562639],
                [112.23619579120823, 109.55303798562639],
                [112.95719579120824, 110.46003798562639],
                [111.49919579120822, 107.7030379856264],
                [111.58819579120824, 107.7120379856264],
                [112.09719579120824, 107.0870379856264],
                [111.02419579120823, 110.30503798562638],
            ],
            None,
            [
                [75.99059786642306, 75.34468901215536],
                [75.64159786642307, 76.07968901215536],
                [75.05959786642306, 76.72668901215535],
                [75.20659786642307, 74.84368901215535],
                [75.95359786642307, 74.88468901215535],
                [76.56159786642307, 75.51368901215535],
                [75.44959786642306, 73.37168901215536],
                [75.30159786642307, 73.28668901215535],
                [75.82659786642307, 72.83568901215536],
                [75.02559786642307, 75.35468901215535],
            ],
            None,
            0,
            None,
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

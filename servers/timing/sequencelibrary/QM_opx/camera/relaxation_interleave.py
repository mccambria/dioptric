# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""


from qm import qua
from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_sequence
from utils import common
from utils import tool_belt as tb
import matplotlib.pyplot as plt

from utils.constants import NVSpinState


def get_seq(args, num_reps):
    pol_coords_list, ion_coords_list, tau_ns = args[:3]
    states = args[3:]

    states = [eval(el) for el in states]
    init_state_0, readout_state_0, init_state_1, readout_state_1 = states

    buffer = seq_utils.get_widefield_operation_buffer()
    tau = seq_utils.convert_ns_to_cc(tau_ns)

    sig_gen_el_dict = {
        NVSpinState.ZERO: None,
        NVSpinState.LOW: seq_utils.get_sig_gen_element(0),
        NVSpinState.HIGH: seq_utils.get_sig_gen_element(1),
    }

    # Set up microwave sequence for experiment 0

    init_sig_gen_el_0 = sig_gen_el_dict[init_state_0]
    readout_sig_gen_el_0 = sig_gen_el_dict[readout_state_0]

    def uwave_macro_0():
        if init_sig_gen_el_0 is not None:
            qua.play("pi_pulse", init_sig_gen_el_0)
        qua.wait(tau)
        if readout_sig_gen_el_0 is not None:
            qua.play("pi_pulse", readout_sig_gen_el_0)
        qua.wait(buffer)
        qua.align()

    # Set up microwave sequence for experiment 1

    init_sig_gen_el_1 = sig_gen_el_dict[init_state_1]
    readout_sig_gen_el_1 = sig_gen_el_dict[readout_state_1]

    def uwave_macro_1():
        if init_sig_gen_el_1 is not None:
            qua.play("pi_pulse", init_sig_gen_el_1)
        qua.wait(tau)
        if readout_sig_gen_el_1 is not None:
            qua.play("pi_pulse", readout_sig_gen_el_1)
        qua.wait(buffer)
        qua.align()

    # Call the base sequence

    uwave_macro = (uwave_macro_0, uwave_macro_1)
    seq = base_sequence.get_seq(pol_coords_list, ion_coords_list, num_reps, uwave_macro)

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
            [
                [112.03744137001495, 109.50814699059372],
                [112.22844137001495, 108.72114699059371],
                [111.05444137001496, 108.90314699059371],
            ],
            [
                [76.0990499534296, 75.08248628148773],
                [76.2550499534296, 74.39048628148774],
                [75.4050499534296, 74.55748628148774],
            ],
            1000.0,
            "NVSpinState.HIGH",
            "NVSpinState.LOW",
            "NVSpinState.ZERO",
            "NVSpinState.LOW",
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(400e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

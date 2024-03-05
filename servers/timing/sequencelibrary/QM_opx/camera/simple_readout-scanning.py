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

import utils.common as common
import utils.kplotlib as kpl
import utils.tool_belt as tb
from servers.timing.sequencelibrary.QM_opx import seq_utils


def get_seq(args, num_reps):
    readout_duration, readout_laser, coords_1, coords_2 = args
    if num_reps is None:
        num_reps = 1

    laser_element = f"do_{readout_laser}_dm"
    camera_element = "do_camera_trigger"
    x_element = f"ao_{readout_laser}_x"
    y_element = f"ao_{readout_laser}_y"
    readout_duration_cc = readout_duration / 4  # * 4 ns / clock_cycle = 1 us
    coords_1_hz = [round(el * 10**6) for el in coords_1]
    coords_2_hz = [round(el * 10**6) for el in coords_2]
    with qua.program() as seq:
        x_freq = qua.declare(int)
        y_freq = qua.declare(int)

        seq_utils.turn_on_aods([readout_laser])

        ### Define one rep here
        def one_rep():
            qua.play("on", camera_element)
            with qua.for_each_((x_freq, y_freq), (coords_1_hz, coords_2_hz)):
                qua.update_frequency(x_element, x_freq)
                qua.update_frequency(y_element, y_freq)
                qua.play("continue", x_element)
                qua.play("continue", y_element)
                qua.play("on", laser_element, duration=readout_duration_cc)
            qua.align()
            qua.ramp_to_zero(camera_element)

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps)

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
        # readout, readout_laser, coords_1, coords_2
        args = [
            2000.0,
            "laser_INTE_520",
            [
                108.21143419610962,
                109.68543419610963,
                110.06143419610963,
            ],
            [
                108.60157559559875,
                109.96457559559875,
                108.84157559559875,
            ],
        ]
        seq, seq_ret_vals = get_seq(args, 4)

        sim_config = SimulationConfig(duration=round(40e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

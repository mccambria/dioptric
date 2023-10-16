# -*- coding: utf-8 -*-
"""
Scanning illumination and widefield collection

Created on October 13th, 2023

@author: mccambria
"""


import numpy
from qm import qua
from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig
from qm.qua import program, declare, strict_timing_
from qm.qua import wait, update_frequency, play, align, fixed, for_each_
import servers.timing.sequencelibrary.QM_opx.seq_utils as seq_utils
import utils.common as common
import utils.tool_belt as tb
import utils.kplotlib as kpl
import matplotlib.pyplot as plt
from qm import generate_qua_script


def qua_program(coords_1, coords_2, readout, readout_laser, readout_power, num_reps):
    laser_element = f"do_{readout_laser}_dm"
    camera_element = f"do_camera_trigger"
    x_element = f"ao_{readout_laser}_x"
    y_element = f"ao_{readout_laser}_y"
    clock_cycles = readout / 4  # * 4 ns / clock_cycle = 1 us
    coords_1_hz = [round(el * 10**6) for el in coords_1]
    coords_2_hz = [round(el * 10**6) for el in coords_2]
    with program() as seq:
        x_freq = declare(int)
        y_freq = declare(int)

        # play("on", laser_element, duration=clock_cycles)
        ### Define one rep here
        def one_rep():
            with for_each_((x_freq, y_freq), (coords_1_hz, coords_2_hz)):
                update_frequency(x_element, x_freq)
                update_frequency(y_element, y_freq)
                play("aod_cw", x_element, duration=clock_cycles)
                play("aod_cw", y_element, duration=clock_cycles)
                play("on", laser_element, duration=clock_cycles)
                play("on", camera_element, duration=clock_cycles)

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps)

        play("off", camera_element, duration=20)
        # play("on", laser_element, duration=clock_cycles)

    return seq


def get_seq(opx_config, config, args, num_reps=-1):
    coords_1, coords_2, readout, readout_laser, readout_power = args
    seq = qua_program(
        coords_1, coords_2, readout, readout_laser, readout_power, num_reps
    )
    final = ""
    # specify what one 'sample' means for  readout
    sample_size = "all_reps"
    num_gates = 0
    return seq, final, [], num_gates, sample_size


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(host=ip_address)
    # qmm = QuantumMachinesManager()
    # qmm.close_all_quantum_machines()
    # print(qmm.list_open_quantum_machines())
    opx = qmm.open_qm(opx_config)

    try:
        # coords_1, coords_2, readout, readout_laser, readout_power
        args = [
            [105.0, 110.0, 115.0, 115.0],
            [105.0, 105.0, 105.0, 110.0],
            10000.0,
            "laser_INTE_520",
            None,
        ]
        args = [[110.0], [110.0], 10000.0, "laser_INTE_520", None]
        ret_vals = get_seq(opx_config, config, args, 4)
        seq, final, ret_vals, _, _ = ret_vals

        sim_config = SimulationConfig(duration=round(10e4 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

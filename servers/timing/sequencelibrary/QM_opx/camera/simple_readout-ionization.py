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
from qm.qua import program, declare, strict_timing_, while_, assign, wait
from qm.qua import wait, update_frequency, play, align, fixed, for_each_
import servers.timing.sequencelibrary.QM_opx.seq_utils as seq_utils
import utils.common as common
import utils.tool_belt as tb
import utils.kplotlib as kpl
from utils.constants import ModMode
import matplotlib.pyplot as plt
from qm import generate_qua_script


# def qua_program(
#     readout, readout_laser, ionization_laser, coords_1, coords_2, mod_mode, num_reps
# ):
#     if mod_mode == ModMode.ANALOG:
#         readout_laser_element = f"ao_{readout_laser}_am"
#     elif mod_mode == ModMode.DIGITAL:
#         readout_laser_element = f"do_{readout_laser}_dm"
#     ionization_laser_element = f"do_{ionization_laser}_dm"
#     camera_element = f"do_camera_trigger"
#     x_element = f"ao_{ionization_laser}_x"
#     y_element = f"ao_{ionization_laser}_y"
#     clock_cycles = readout / 4  # * 4 ns / clock_cycle = 1 us
#     coords_1_hz = [round(el * 10**6) for el in coords_1]
#     coords_2_hz = [round(el * 10**6) for el in coords_2]
#     num_reps = num_reps * readout / 1000  # Num of us cycles
#     clock_cycles = 250  # * 4 ns / clock_cycle = 1 us
#     with program() as seq:
#         x_freq = declare(int)
#         y_freq = declare(int)

#         ### Define one rep here
#         def one_rep():
#             with for_each_((x_freq, y_freq), (coords_1_hz, coords_2_hz)):
#                 update_frequency(x_element, x_freq)
#                 update_frequency(y_element, y_freq)
#                 play("aod_cw", x_element, duration=clock_cycles)
#                 play("aod_cw", y_element, duration=clock_cycles)
#                 play("on", readout_laser_element, duration=clock_cycles)
#                 play("on", ionization_laser_element, duration=clock_cycles)
#                 play("on", camera_element, duration=clock_cycles)

#         ### Handle the reps in the utils code
#         seq_utils.handle_reps(one_rep, num_reps)

#         play("off", camera_element, duration=20)
#         # play("on", readout_laser_element, duration=clock_cycles)

#     return seq


def qua_program(
    readout, readout_laser, ionization_laser, coords_1, coords_2, mod_mode, num_reps
):
    if mod_mode == ModMode.ANALOG:
        readout_laser_element = f"ao_{readout_laser}_am"
    elif mod_mode == ModMode.DIGITAL:
        readout_laser_element = f"do_{readout_laser}_dm"
    ionization_laser_element = f"do_{ionization_laser}_dm"
    camera_element = f"do_camera_trigger"
    x_element = f"ao_{ionization_laser}_x"
    y_element = f"ao_{ionization_laser}_y"
    coords_1_hz = [round(el * 10**6) for el in coords_1]
    coords_2_hz = [round(el * 10**6) for el in coords_2]
    # num_reps = num_reps * readout / 1000  # Num of us cycles
    ionization_duration = int(1000 / 4)
    pulsed_readout = int(10e6 / 4)
    one_rep_duration = ionization_duration + pulsed_readout
    num_reps = num_reps * readout / (one_rep_duration * 4)
    default_len_clock_cycles = int(1000 / 4)
    with program() as seq:
        x_freq = declare(int, value=coords_1_hz[0])
        y_freq = declare(int, value=coords_2_hz[0])
        update_frequency(x_element, x_freq)
        update_frequency(y_element, y_freq)

        # AODs
        play("aod_cw", x_element)
        play("aod_cw", y_element)

        # Camera
        play("on", camera_element)

        ### Define one rep here
        def one_rep():
            # Ionization laser
            wait(int(400 / 4), ionization_laser_element)
            play("ionize", ionization_laser_element)

            wait(default_len_clock_cycles, readout_laser_element)
            play("charge_state_readout", readout_laser_element)
            # assign(ind, 0)
            # with while_(ind < round(pulsed_readout / default_len_clock_cycles)):
            #     play("on", readout_laser_element)
            #     assign(ind, ind + 1)
            align()

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps)

        play("off", camera_element)
        # play("on", readout_laser_element, duration=clock_cycles)

    return seq


def get_seq(opx_config, config, args, num_reps=-1):
    readout_laser = args[1]
    mod_mode = config["Optics"][readout_laser]["mod_mode"]
    seq = qua_program(*args, mod_mode, num_reps)
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
        args = [100e6, "laser_OPTO_589", "laser_COBO_638", [75.0], [75.0]]
        ret_vals = get_seq(opx_config, config, args, 1)
        seq, final, ret_vals, _, _ = ret_vals

        sim_config = SimulationConfig(duration=int(1e6 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

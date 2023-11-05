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


def qua_program(
    readout,
    readout_laser,
    do_polarize,
    do_ionize,
    ionization_laser,
    ionization_coords,
    polarization_laser,
    polarization_coords,
    mod_mode,
    num_reps,
):
    if mod_mode == ModMode.ANALOG:
        readout_laser_element = f"ao_{readout_laser}_am"
    elif mod_mode == ModMode.DIGITAL:
        readout_laser_element = f"do_{readout_laser}_dm"

    camera_element = f"do_camera_trigger"

    # Ionization
    ionization_laser_element = f"do_{ionization_laser}_dm"
    ionization_x_element = f"ao_{ionization_laser}_x"
    ionization_y_element = f"ao_{ionization_laser}_y"

    # Polarization
    polarization_laser_element = f"do_{polarization_laser}_dm"
    polarization_x_element = f"ao_{polarization_laser}_x"
    polarization_y_element = f"ao_{polarization_laser}_y"

    access_time = int(20e3 / 4)
    polarization_duration = int(1e6 / 4)
    setup_duration = access_time + polarization_duration + int(10e3 / 4)
    us_clock_cycles = int(1e3 / 4)
    readout_clock_cycles = int(readout / 4)
    total_duration = setup_duration + readout_clock_cycles
    # print(((setup_duration * 4) + readout) * 10**-9)

    with program() as seq:
        # Polarization
        polarization_x_freq = declare(
            int, value=round(polarization_coords[0] * 10**6)
        )
        polarization_y_freq = declare(
            int, value=round(polarization_coords[1] * 10**6)
        )
        update_frequency(polarization_x_element, polarization_x_freq)
        update_frequency(polarization_y_element, polarization_y_freq)
        play("aod_cw", polarization_x_element, duration=total_duration)
        play("aod_cw", polarization_y_element, duration=total_duration)
        if do_polarize:
            wait(access_time, polarization_laser_element)
            play(
                "on",
                polarization_laser_element,
                duration=polarization_duration,
                # duration=total_duration
            )

        # Ionization
        ionization_x_freq = declare(int, value=round(ionization_coords[0] * 10**6))
        ionization_y_freq = declare(int, value=round(ionization_coords[1] * 10**6))
        update_frequency(ionization_x_element, ionization_x_freq)
        update_frequency(ionization_y_element, ionization_y_freq)
        play("aod_cw", ionization_x_element, duration=total_duration)
        play("aod_cw", ionization_y_element, duration=total_duration)
        if do_ionize:
            wait(access_time + int(5e3 / 4), ionization_laser_element)
            play("on", ionization_laser_element, duration=int(3e3 / 4))

        # Yellow readout
        wait(setup_duration, readout_laser_element)
        wait(setup_duration, camera_element)
        play("on", readout_laser_element, duration=readout_clock_cycles)
        play("on", camera_element, duration=readout_clock_cycles)
        play("off", camera_element, duration=25)

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
        # readout, readout_laser, do_ionize, ionization_laser, ionization_coords, polarization_laser, polarization_coords
        args = [
            1000000.0,
            "laser_OPTO_589",
            False,
            "laser_COBO_638",
            [74.45, 75.25],
            "laser_INTE_520",
            [111.695, 108.75],
        ]
        ret_vals = get_seq(opx_config, config, args, 1)
        seq, final, ret_vals, _, _ = ret_vals

        sim_config = SimulationConfig(duration=int(2e6 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

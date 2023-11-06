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


def get_seq(args, num_reps=1):
    (
        readout,
        readout_laser,
        do_polarize,
        do_ionize,
        ionization_laser,
        ionization_coords,
        polarization_laser,
        polarization_coords,
    ) = args

    readout_laser_el = seq_utils.get_laser_mod_element(readout_laser)
    camera_el = f"do_camera_trigger"

    # Ionization
    ionization_laser_el = f"do_{ionization_laser}_dm"
    ionization_x_el = f"ao_{ionization_laser}_x"
    ionization_y_el = f"ao_{ionization_laser}_y"

    # Polarization
    polarization_laser_el = f"do_{polarization_laser}_dm"
    polarization_x_el = f"ao_{polarization_laser}_x"
    polarization_y_el = f"ao_{polarization_laser}_y"

    access_time = int(20e3 / 4)
    polarization_duration = int(1e6 / 4)
    setup_duration = access_time + polarization_duration + int(10e3 / 4)
    us_clock_cycles = int(1e3 / 4)
    readout_clock_cycles = int(readout / 4)
    total_duration = setup_duration + readout_clock_cycles
    # print(((setup_duration * 4) + readout) * 10**-9)

    with program() as seq:

        def one_rep():
            # Polarization
            polarization_x_freq = declare(
                int, value=round(polarization_coords[0] * 10**6)
            )
            polarization_y_freq = declare(
                int, value=round(polarization_coords[1] * 10**6)
            )
            update_frequency(polarization_x_el, polarization_x_freq)
            update_frequency(polarization_y_el, polarization_y_freq)
            play("aod_cw", polarization_x_el, duration=total_duration)
            play("aod_cw", polarization_y_el, duration=total_duration)
            if do_polarize:
                wait(access_time, polarization_laser_el)
                play(
                    "on",
                    polarization_laser_el,
                    # duration=polarization_duration,
                    duration=total_duration - access_time,
                )

            # Ionization
            ionization_x_freq = declare(
                int, value=round(ionization_coords[0] * 10**6)
            )
            ionization_y_freq = declare(
                int, value=round(ionization_coords[1] * 10**6)
            )
            update_frequency(ionization_x_el, ionization_x_freq)
            update_frequency(ionization_y_el, ionization_y_freq)
            play("aod_cw", ionization_x_el, duration=total_duration)
            play("aod_cw", ionization_y_el, duration=total_duration)
            if do_ionize:
                wait(access_time + int(5e3 / 4), ionization_laser_el)
                play("on", ionization_laser_el, duration=int(3e3 / 4))

            # Yellow readout
            wait(setup_duration, readout_laser_el)
            wait(setup_duration, camera_el)
            play("on", readout_laser_el, duration=readout_clock_cycles)
            play("on", camera_el, duration=readout_clock_cycles)
            play("off", camera_el, duration=25)

        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=True)

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
        seq, seq_ret_vals = get_seq(args, 1)

        sim_config = SimulationConfig(duration=int(2e6 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

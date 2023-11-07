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
import servers.timing.sequencelibrary.QM_opx.seq_utils as seq_utils
import utils.common as common
import matplotlib.pyplot as plt
from qm import generate_qua_script


def get_seq(args, num_reps):
    (
        readout_duration,
        readout_laser,
        do_polarize,
        do_ionize,
        ionization_laser,
        ionization_coords,
        polarization_laser,
        polarization_coords,
    ) = args
    if num_reps == None:
        num_reps = 1

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

    access_time_cc = int(20e3 / 4)
    polarization_duration_cc = int(1e6 / 4)
    # polarization_duration_cc = int(10e3 / 4)
    setup_duration_cc = access_time_cc + polarization_duration_cc + int(10e3 / 4)
    readout_duration_cc = int(readout_duration / 4)
    total_duration_cc = setup_duration_cc + readout_duration_cc
    # print(((setup_duration * 4) + readout) * 10**-9)
    camera_pad_cc = seq_utils.calc_camera_pad(total_duration_cc)

    with qua.program() as seq:

        def one_rep():
            # Polarization
            polarization_x_freq = qua.declare(
                int, value=round(polarization_coords[0] * 10**6)
            )
            polarization_y_freq = qua.declare(
                int, value=round(polarization_coords[1] * 10**6)
            )
            qua.update_frequency(polarization_x_el, polarization_x_freq)
            qua.update_frequency(polarization_y_el, polarization_y_freq)
            qua.play("aod_cw", polarization_x_el, duration=total_duration_cc)
            qua.play("aod_cw", polarization_y_el, duration=total_duration_cc)
            if do_polarize:
                qua.wait(access_time_cc, polarization_laser_el)
                qua.play(
                    "on",
                    polarization_laser_el,
                    duration=polarization_duration_cc,
                    # duration=total_duration_cc - access_time_cc,
                )

            # Ionization
            # ionization_x_freq = qua.declare(
            #     int, value=round(ionization_coords[0] * 10**6)
            # )
            # ionization_y_freq = qua.declare(
            #     int, value=round(ionization_coords[1] * 10**6)
            # )
            # qua.update_frequency(ionization_x_el, ionization_x_freq)
            # qua.update_frequency(ionization_y_el, ionization_y_freq)
            # qua.play("aod_cw", ionization_x_el, duration=total_duration_cc)
            # qua.play("aod_cw", ionization_y_el, duration=total_duration_cc)
            # if do_ionize:
            #     qua.wait(access_time_cc + int(5e3 / 4), ionization_laser_el)
            #     qua.play("on", ionization_laser_el, duration=int(3e3 / 4))

            # Yellow readout
            qua.wait(setup_duration_cc, readout_laser_el)
            qua.wait(setup_duration_cc, camera_el)
            qua.play("on", readout_laser_el, duration=readout_duration_cc)
            qua.play("on", camera_el)
            qua.align()
            qua.play("off", camera_el)

        seq_utils.handle_reps(one_rep, num_reps, post_trigger_pad=camera_pad_cc)
        # seq_utils.handle_reps(one_rep, num_reps)

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
        # readout_duration, readout_laser, do_polarize, do_ionize, ionization_laser, ionization_coords, polarization_laser, polarization_coords,
        args = [
            5e6,
            "laser_OPTO_589",
            True,
            False,
            "laser_COBO_638",
            [74.45, 75.25],
            "laser_INTE_520",
            [111.695, 108.75],
        ]
        seq, seq_ret_vals = get_seq(args, 1)

        sim_config = SimulationConfig(duration=int(50e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

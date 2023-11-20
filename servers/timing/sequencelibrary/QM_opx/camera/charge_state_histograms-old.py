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
        pol_laser,
        pol_duration_ns,
        pol_coords_list,
        ion_laser,
        ion_duration_ns,
        ion_coords_list,
        readout_laser,
        readout_duration_ns,
        diff_polarize,
        diff_ionize,
    ) = args

    if num_reps == None:
        num_reps = 1

    # Polarization
    pol_laser_el = seq_utils.get_laser_mod_element(pol_laser)
    pol_x_el = f"ao_{pol_laser}_x"
    pol_y_el = f"ao_{pol_laser}_y"

    # Ionization
    ion_laser_el = seq_utils.get_laser_mod_element(ion_laser)
    ion_x_el = f"ao_{ion_laser}_x"
    ion_y_el = f"ao_{ion_laser}_y"

    # Readout
    readout_laser_el = seq_utils.get_laser_mod_element(readout_laser)
    camera_el = f"do_camera_trigger"

    if diff_polarize and not diff_ionize:
        do_polarize_sig = True
        do_polarize_ref = False
        do_ionize_sig = False
        do_ionize_ref = False
    elif not diff_polarize and diff_ionize:
        do_polarize_sig = True
        do_polarize_ref = True
        do_ionize_sig = True
        do_ionize_ref = False

    access_time = seq_utils.get_aod_access_time()
    pol_duration = seq_utils.convert_ns_to_cc(pol_duration_ns)
    ion_duration = seq_utils.convert_ns_to_cc(ion_duration_ns)
    buffer = seq_utils.get_widefield_operation_buffer()
    setup_duration = access_time + pol_duration + buffer + ion_duration + buffer
    readout_duration = seq_utils.convert_ns_to_cc(readout_duration_ns)

    with qua.program() as seq:
        pol_x_freq = qua.declare(int, value=round(pol_coords[0] * 10**6))
        pol_y_freq = qua.declare(int, value=round(pol_coords[1] * 10**6))
        qua.update_frequency(pol_x_el, pol_x_freq)
        qua.update_frequency(pol_y_el, pol_y_freq)
        qua.play("aod_cw", pol_x_el)
        qua.play("aod_cw", pol_y_el)

        ion_x_freq = qua.declare(int, value=round(ion_coords[0] * 10**6))
        ion_y_freq = qua.declare(int, value=round(ion_coords[1] * 10**6))
        qua.update_frequency(ion_x_el, ion_x_freq)
        qua.update_frequency(ion_y_el, ion_y_freq)
        qua.play("aod_cw", ion_x_el)
        qua.play("aod_cw", ion_y_el)

        def half_rep(do_polarize_sub, do_ionize_sub):
            # Polarization
            if do_polarize_sub:
                qua.wait(access_time, pol_laser_el)
                qua.play("on", pol_laser_el, duration=pol_duration)

            # Ionization
            if do_ionize_sub:
                qua.wait(access_time + pol_duration + buffer, ion_laser_el)
                qua.play("on", ion_laser_el, duration=ion_duration)

            # Yellow readout
            qua.wait(setup_duration, readout_laser_el)
            qua.wait(setup_duration, camera_el)
            qua.play("on", readout_laser_el, duration=readout_duration)
            qua.play("on", camera_el)
            qua.align()
            qua.play("off", camera_el)
            qua.align()

        def one_rep():
            for half_rep_args in [
                [do_polarize_sig, do_ionize_sig],
                [do_polarize_ref, do_ionize_ref],
            ]:
                half_rep(*half_rep_args)
                qua.wait_for_trigger(camera_el)
                qua.align()

        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)

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
            False,
            "laser_INTE_520",
            [111.326, 109.79],
            1000.0,
            True,
            "laser_COBO_638",
            [75.02, 75.425],
            1000.0,
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

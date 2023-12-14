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
from servers.timing.sequencelibrary.QM_opx import seq_utils
import utils.common as common
import matplotlib.pyplot as plt
from qm import generate_qua_script


def get_seq(args, num_reps):
    (
        readout_duration_ns,
        readout_laser,
        pol_laser,
        pol_coords,
        pol_duration_ns,
        ion_laser,
        ion_coords,
        ion_duration_ns,
    ) = args
    if num_reps == None:
        num_reps = 1

    readout_laser_el = seq_utils.get_laser_mod_element(readout_laser, sticky=True)
    camera_el = f"do_camera_trigger"

    # Polarization
    pol_laser_el = f"do_{pol_laser}_dm"
    pol_x_el = f"ao_{pol_laser}_x"
    pol_y_el = f"ao_{pol_laser}_y"

    # Ionization
    ion_laser_el = f"do_{ion_laser}_dm"
    ion_x_el = f"ao_{ion_laser}_x"
    ion_y_el = f"ao_{ion_laser}_y"

    access_time = seq_utils.get_aod_access_time()
    pol_duration = seq_utils.convert_ns_to_cc(pol_duration_ns)
    ion_duration = seq_utils.convert_ns_to_cc(ion_duration_ns)
    default_pulse_duration = seq_utils.get_default_pulse_duration()
    operation_gap = seq_utils.convert_ns_to_cc(10e3)
    setup_duration = (
        access_time + pol_duration + operation_gap + ion_duration + operation_gap
    )
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

        def one_rep():
            # Polarization
            qua.wait(access_time, pol_laser_el)
            qua.play("on", pol_laser_el, duration=pol_duration)

            # Ionization
            qua.wait(access_time + pol_duration + operation_gap, ion_laser_el)
            qua.play("long_ionize", ion_laser_el)

            # Yellow readout
            qua.wait(setup_duration, (readout_laser_el, camera_el))
            qua.play("charge_readout", readout_laser_el)
            qua.play("on", camera_el)
            qua.wait(
                readout_duration - default_pulse_duration, (readout_laser_el, camera_el)
            )
            qua.align()
            qua.ramp_to_zero(readout_laser_el)
            qua.ramp_to_zero(camera_el)
            qua.align()

        seq_utils.handle_reps(one_rep, num_reps)

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
            True,
            "laser_INTE_520",
            [111.202, 109.801],
            1000,
            False,
            "laser_COBO_638",
            [75, 75],
            2000.0,
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

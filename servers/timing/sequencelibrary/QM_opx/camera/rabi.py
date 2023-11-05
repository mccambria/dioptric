# -*- coding: utf-8 -*-
"""
Polarization --> Microwave --> Ionization --> Readout
Created on October 13th, 2023

@author: mccambria
"""

# Import required libraries

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
from servers.timing.sequencelibrary.QM_opx import seq_utils
from utils.constants import ModMode


# Function to define the sequence
def define_sequence(
    polarization_laser,
    microwave_sig_gen,
    ionization_laser,
    readout_laser,
    durations,
    coords,
    config,
):
    uwave_buffer = config["CommonDurations"]["uwave_buffer"]
    aod_rise_time = config["CommonDurations"]["aod_rise_time"]
    aod_end_buffer = config["CommonDurations"]["aod_end_buffer"]
    readout_laser_mod_mode = config["Optics"][readout_laser]["mod_mode"]
    if readout_laser_mod_mode == ModMode.ANALOG:
        readout_element = f"ao_{readout_laser}_am"
    elif readout_laser_mod_mode == ModMode.DIGITAL:
        readout_element = f"do_{readout_laser}_dm"

    # Define the elements for lasers, AODs, and cameras
    polarization_element = f"do_{polarization_laser}_dm"
    x_element_pol = f"ao_{polarization_laser}_x"
    y_element_pol = f"ao_{polarization_laser}_y"

    microwave_element = f"do_{microwave_sig_gen}_dm"

    ionization_element = f"do_{ionization_laser}_dm"
    x_element_ion = f"ao_{ionization_laser}_x"
    y_element_ion = f"ao_{ionization_laser}_y"

    camera_element = "do_camera_trigger"

    uwave_buffer_cc = seq_utils.convert_ns_to_clock_cycles(uwave_buffer)
    aod_rise_time_cc = seq_utils.convert_ns_to_clock_cycles(aod_rise_time)
    aod_end_buffer_cc = seq_utils.convert_ns_to_clock_cycles(aod_end_buffer)

    durations = [round(el / 4) for el in durations]
    (
        polarization_duration_cc,
        microwave_duration_cc,
        ionization_duration_cc,
        readout_duration_cc,
        camera_duration_cc
    ) = durations
    
    aod_duration_cc = aod_rise_time_cc + polarization_duration_cc + aod_end_buffer_cc

    # Extract coordinates
    x_freqs_pol, y_freqs_pol, x_freqs_ion, y_freqs_ion = coords

    # Convert frequencies from MHz to Hz
    x_freqs_pol = [int(el * 1e6) for el in x_freqs_pol]
    y_freqs_pol = [int(el * 1e6) for el in y_freqs_pol]
    x_freqs_ion = [int(el * 1e6) for el in x_freqs_ion]
    y_freqs_ion = [int(el * 1e6) for el in y_freqs_ion]
    

    with program() as seq:
        x_freq_pol = declare(int)
        y_freq_pol = declare(int)
        x_freq_ion = declare(int)
        y_freq_ion = declare(int)

        with for_each_((x_freq_pol, y_freq_pol), (x_freqs_pol, y_freqs_pol)):
            # Update frequencies for AODs
            update_frequency(x_element_pol, x_freq_pol)
            update_frequency(y_element_pol, y_freq_pol)

            # Play AODs for polarization (duration in ns)
            play("aod_cw", x_element_pol, duration=aod_duration_cc)
            play("aod_cw", y_element_pol, duration=aod_duration_cc)
            wait(aod_rise_time, polarization_element)
            play("on", polarization_element, duration=polarization_duration_cc)
            wait(aod_end_buffer, polarization_element)

        # Wait for microwave setup
        wait(setup_duration + uwave_buffer_cc, microwave_element)

        # Play microwave pulse
        play("on", microwave_element, duration=microwave_duration)

        # Wait for ionization setup
        wait(
            setup_duration + uwave_buffer_cc + microwave_duration + uwave_buffer_cc,
            ionization_element,
        )

        with for_each_((x_freq_ion, y_freq_ion), (x_freqs_ion, y_freqs_ion)):
            update_frequency(x_element_ion, x_freq_ion)
            update_frequency(y_element_ion, y_freq_ion)

            # Play AODs for ionization (duration in ns)
            play("aod_cw", x_element_ion, duration=aod_duration)
            play("aod_cw", y_element_ion, duration=aod_duration)
            play("on", ionization_element, duration=ionization_duration)

        # Wait for readout setup
        wait(
            setup_duration
            + uwave_buffer_cc
            + microwave_duration
            + uwave_buffer_cc
            + ionization_duration
            + uwave_buffer_cc,
            readout_element,
        )

        wait(
            setup_duration
            + uwave_buffer_cc
            + microwave_duration
            + uwave_buffer_cc
            + ionization_duration
            + uwave_buffer_cc,
            camera_element,
        )

        # Readout sequence
        play("on", readout_element, duration=readout_duration)
        play("on", camera_element, duration=camera_duration)

    return seq


# Function to get the sequence
def get_sequence(opx_config, config, args, num_reps=-1):
    seq = define_sequence(*args, config)
    final = ""
    sample_size = "all_reps"
    num_gates = 0
    return seq, final, [], num_gates, sample_size


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(host=ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        # Define the parameters for polarization, microwave, ionization, readout, and coordinates
        durations = [1000, 100, 200, 10e6, 10e6, 15e3, 20e3]
        args = [
            "laser_INTE_520",  # polarization_laser,
            "sig_gen_STAN_sg394",  # microwave_sig_gen,
            "laser_COBO_638",  # ionization_laser,
            "laser_OPTO_589",  # readout_laser,
            durations,  # durations,
            [[75], [75], [110], [110]],  # coords,
        ]

        seq = define_sequence(*args, config)

        sim_config = SimulationConfig(duration=int(1e6 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

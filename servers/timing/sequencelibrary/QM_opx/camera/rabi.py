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


# Function to define the sequence
def define_sequence(
    polarization_laser,
    microwave_sig_gen,
    ionization_laser,
    readout_laser,
    durations,
    coords,
):
    # Define the elements for lasers, AODs, and cameras
    polarization_element = f"do_{polarization_laser}_dm"
    x_element_pol = f"ao_{polarization_laser}_x"
    y_element_pol = f"ao_{polarization_laser}_y"

    microwave_element = f"do_{microwave_sig_gen}_dm"

    ionization_element = f"do_{ionization_laser}_dm"
    x_element_ion = f"ao_{ionization_laser}_x"
    y_element_ion = f"ao_{ionization_laser}_y"

    readout_element = f"do_{readout_laser}_dm"
    camera_element = "do_camera_trigger"

    x_freqs_pol, y_freqs_pol, x_freqs_ion, y_freqs_ion = coords

    (
        polarization_duration,
        microwave_duration,
        readout_duration,
        camera_duration,
    ) = durations

    ns_per_clock_cycle = 4

    with program() as seq:
        x_freq_pol = declare(int)
        y_freq_pol = declare(int)
        x_freq_ion = declare(int)
        y_freq_ion = declare(int)

        with for_each_((x_freq_pol, y_freq_pol), (x_freqs_pol, y_freqs_pol)):
            # Update frequencies for AODs
            update_frequency(x_element_pol, x_freq_pol)
            update_frequency(y_element_pol, y_freq_pol)

            # Play laser polarization (duration in ns)
            play("aod_cw", x_element_pol, duration=clock_cycles)
            play("aod_cw", y_element_pol, duration=clock_cycles)
            play("on", polarization_element, duration=polarization_duration)

        # Play microwave pulse
        play("on", microwave_element, duration=microwave_duration)

        with for_each_((x_freq_ion, y_freq_ion), (x_freqs_ion, y_freqs_ion)):
            update_frequency(x_element_ion, x_freq_ion)
            update_frequency(y_element_ion, y_freq_ion)
            # Play ionization laser
            play("aod_cw", x_element_ion, duration=clock_cycles)
            play("aod_cw", y_element_ion, duration=clock_cycles)
            play("on", ionization_element, duration=200)

        # Readout sequence
        play("on", readout_element, duration=readout_duration)
        play("on", camera_element, duration=camera_duration)

    return seq


# Function to get the sequence
def get_sequence(opx_config, config, args, num_reps=-1):
    seq = define_sequence(*args)
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
        args = [
            "polarization",
            "microwave",
            "ionization",
            "readout",
            [105.0, 110.0, 115.0, 115.0],  # the coordinates for coords
            4,  # Number of repetitions
        ]
        durations = (1000, 100, 20, 20, 1000 / 4)  # Durations in ns
        args.append(durations)

        ret_vals = get_sequence(opx_config, config, args, 4)
        seq, final, ret_vals, _, _ = ret_vals

        sim_config = SimulationConfig(
            duration=int(10e4 / durations[-1])
        )  # Simulate based on the clock cycle duration
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()

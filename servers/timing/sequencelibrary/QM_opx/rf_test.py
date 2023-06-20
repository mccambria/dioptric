#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

rabi sequence for the opx

"""


import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import qua
from qm import SimulationConfig
from qm.qua import program, declare, declare_stream, stream_processing
from qm.qua import measure, wait, save, play, align
from utils.tool_belt import States


def qua_program(opx, config, args, num_reps):
    ### get inputted parameters
    freq, amp, duration = args

    with program() as seq:
        
        align()
        qua.update_frequency(element, new_frequency)
        qua.play(
            qua.laser_pulse * amp(laser_amplitude), laser_name, duration=polarization_cc
        )

        align()

        with for_(n, 0, n < num_reps, n + 1):
            align()

            play(
                laser_pulse * amp(laser_amplitude), laser_name, duration=polarization_cc
            )

            align()
            wait(laser_m_uwave_delay_cc)
            wait(signal_wait_time_cc)

            with if_(tau_cc_qua >= 4):
                play("uwave_ON", sig_gen, duration=tau_cc)
                align()
            with elif_(tau_cc_qua <= 3):
                align()

            wait(uwave_m_laser_delay_cc)
            wait(signal_wait_time_cc)

            play(
                laser_pulse * amp(laser_amplitude), laser_name, duration=polarization_cc
            )

            if num_apds == 2:
                wait(laser_delay_time_cc, "do_apd_0_gate", "do_apd_1_gate")
                measure(
                    "readout",
                    "do_apd_0_gate",
                    None,
                    time_tagging.analog(
                        times_gate1_apd_0, readout_time, counts_gate1_apd_0
                    ),
                )
                measure(
                    "readout",
                    "do_apd_1_gate",
                    None,
                    time_tagging.analog(
                        times_gate1_apd_1, readout_time, counts_gate1_apd_1
                    ),
                )
                save(counts_gate1_apd_0, counts_st_apd_0)
                save(counts_gate1_apd_1, counts_st_apd_1)

            if num_apds == 1:
                wait(laser_delay_time_cc, "do_apd_{}_gate".format(apd_indices[0]))
                measure(
                    "readout",
                    "do_apd_{}_gate".format(apd_indices[0]),
                    None,
                    time_tagging.analog(
                        times_gate1_apd_0, readout_time, counts_gate1_apd_0
                    ),
                )
                save(counts_gate1_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)

            align()
            wait(mid_duration_cc)
            align()

            play(
                laser_pulse * amp(laser_amplitude),
                laser_name,
                duration=reference_laser_on_cc,
            )

            if num_apds == 2:
                wait(laser_delay_time_cc, "do_apd_0_gate", "do_apd_1_gate")
                measure(
                    "readout",
                    "do_apd_0_gate",
                    None,
                    time_tagging.analog(
                        times_gate2_apd_0, readout_time, counts_gate2_apd_0
                    ),
                )
                measure(
                    "readout",
                    "do_apd_1_gate",
                    None,
                    time_tagging.analog(
                        times_gate2_apd_1, readout_time, counts_gate2_apd_1
                    ),
                )
                save(counts_gate2_apd_0, counts_st_apd_0)
                save(counts_gate2_apd_1, counts_st_apd_1)

            if num_apds == 1:
                wait(laser_delay_time_cc, "do_apd_{}_gate".format(apd_indices[0]))
                measure(
                    "readout",
                    "do_apd_{}_gate".format(apd_indices[0]),
                    None,
                    time_tagging.analog(
                        times_gate2_apd_0, readout_time, counts_gate2_apd_0
                    ),
                )
                save(counts_gate2_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)

            align()

        play("clock_pulse", "do_sample_clock")

        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0")
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")

    return seq, period, num_gates


def get_seq(opx, config, args, num_repeat):
    seq, period, num_gates = qua_program(opx, config, args, num_repeat)
    final = ""
    # specify what one 'sample' means for  readout
    sample_size = "all_reps"
    return seq, final, [period], num_gates, sample_size


if __name__ == "__main__":
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time

    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117", port="80")
    qm = qmm.open_qm(config_opx)

    simulation_duration = 35000 // 4  # clock cycle units - 4ns

    num_repeat = 3

    args = [100, 1000.0, 350, 100, 3, "cobolt_515", 1]
    seq, f, p, ns, ss = get_seq([], config, args, num_repeat)

    plt.figure()

    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    plt.show()

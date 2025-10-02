#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

simple readout sequence for the opx in qua

"""

import numpy
from opx_configuration_file import *
from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager

import utils.tool_belt as tool_belt
from utils.tool_belt import Mod_types


def qua_program(opx, config, args, num_reps):
    (
        first_init_pulse_time,
        init_pulse_time,
        readout_time,
        first_init_laser_key,
        init_laser_key,
        readout_laser_key,
        first_init_laser_power,
        init_laser_power,
        read_laser_power,
        readout_on_pulse_ind,
        apd_index,
    ) = args

    init_laser_mod_type = config["Optics"][init_laser_key]["mod_type"]
    init_laser_pulse = "laser_ON_{}".format(eval(init_laser_mod_type).name)
    first_init_laser_mod_type = config["Optics"][first_init_laser_key]["mod_type"]
    first_init_laser_pulse = "laser_ON_{}".format(eval(first_init_laser_mod_type).name)
    readout_laser_mod_type = config["Optics"][readout_laser_key]["mod_type"]
    readout_laser_pulse = "laser_ON_{}".format(eval(readout_laser_mod_type).name)
    first_init_laser_delay_time = config["Optics"][first_init_laser_key]["delay"]
    init_laser_delay_time = config["Optics"][init_laser_key]["delay"]
    readout_laser_delay_time = config["Optics"][readout_laser_key]["delay"]
    intra_pulse_delay = config["CommonDurations"]["scc_ion_readout_buffer"]

    if eval(init_laser_mod_type).name == "ANALOG":
        init_laser_amplitude = init_laser_power
    if eval(init_laser_mod_type).name == "DIGITAL":
        init_laser_amplitude = 1

    if eval(first_init_laser_mod_type).name == "ANALOG":
        first_init_laser_amplitude = first_init_laser_power
    if eval(first_init_laser_mod_type).name == "DIGITAL":
        first_init_laser_amplitude = 1

    if eval(readout_laser_mod_type).name == "ANALOG":
        readout_laser_amplitude = read_laser_power
    if eval(readout_laser_mod_type).name == "DIGITAL":
        readout_laser_amplitude = 1

    apd_indices = config["apd_indices"]
    positioning = config["Positioning"]
    if "xy_small_response_delay" in positioning:
        pos_move_time = positioning["xy_small_response_delay"]
    else:
        pos_move_time = positioning["xy_delay"]

    num_apds = len(apd_indices)
    num_gates = 1
    timetag_list_size = int(15900 / num_gates / 2)

    max_readout_time = 500  # config['PhotonCollection']['qm_opx_max_readout_time']
    delay_between_readouts_iterations = 200  # simulated - conservative estimate

    # for now lets assume we use two different lasers.

    if readout_time > max_readout_time:
        num_readouts = int(readout_time / max_readout_time)
        apd_readout_time = max_readout_time

    elif readout_time <= max_readout_time:
        num_readouts = 1
        apd_readout_time = readout_time

    init_laser_on_time = init_pulse_time
    readout_laser_on_time = num_readouts * (apd_readout_time) + (num_readouts - 1) * (
        delay_between_readouts_iterations
    )

    first_init_laser_on_time = first_init_pulse_time
    period = (
        pos_move_time
        + init_laser_on_time
        + intra_pulse_delay
        + readout_laser_on_time
        + 300
    )

    with program() as seq:
        counts_gate1_apd_0 = declare(int)
        counts_gate1_apd_1 = declare(int)
        times_gate1_apd_0 = declare(int, size=timetag_list_size)
        times_gate1_apd_1 = declare(int, size=timetag_list_size)
        counts_st_apd_0 = declare_stream()
        counts_st_apd_1 = declare_stream()

        times_st_apd_0 = declare_stream()
        times_st_apd_1 = declare_stream()

        save(0, times_st_apd_0)
        save(0, times_st_apd_1)

        n = declare(int)
        i = declare(int)
        j = declare(int)
        k = declare(int)

        with for_(n, 0, n < num_reps, n + 1):
            align()
            play(
                first_init_laser_pulse * amp(first_init_laser_amplitude),
                first_init_laser_key,
                duration=first_init_laser_on_time // 4,
            )

            align()
            wait(pos_move_time // 4)

            play(
                init_laser_pulse * amp(init_laser_amplitude),
                init_laser_key,
                duration=init_laser_on_time // 4,
            )

            align()
            wait(intra_pulse_delay // 4)
            align()

            with for_(i, 0, i < num_readouts, i + 1):
                align()
                play(
                    readout_laser_pulse * amp(readout_laser_amplitude),
                    readout_laser_key,
                    duration=apd_readout_time // 4,
                )

                wait(readout_laser_delay_time // 4, "do_apd_0_gate", "do_apd_1_gate")

                if num_apds == 2:
                    measure(
                        "readout",
                        "do_apd_0_gate",
                        None,
                        time_tagging.analog(
                            times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0
                        ),
                    )
                    measure(
                        "readout",
                        "do_apd_1_gate",
                        None,
                        time_tagging.analog(
                            times_gate1_apd_1, apd_readout_time, counts_gate1_apd_1
                        ),
                    )
                    save(counts_gate1_apd_0, counts_st_apd_0)
                    save(counts_gate1_apd_1, counts_st_apd_1)
                    assign(counts_gate1_apd_0, 10)
                    assign(counts_gate1_apd_1, 10)
                    align("do_apd_0_gate", "do_apd_1_gate")

                with for_(j, 0, j < counts_gate1_apd_0, j + 1):
                    save(1, times_st_apd_0)

                with for_(k, 0, k < counts_gate1_apd_1, k + 1):
                    save(2, times_st_apd_1)

            ##clock pulse that advances piezos and ends a sample in the tagger
            align()
            wait(25)
            play("clock_pulse", "do_sample_clock")
            wait(25)

        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0")
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")
            times_st_apd_0.save_all("times_apd0")
            times_st_apd_1.save_all("times_apd1")

    return seq, period, num_gates


def get_seq(
    opx, config, args, num_repeat
):  # so this will give the full desired sequence, with however many repeats are intended repeats
    seq, period, num_gates = qua_program(opx, config, args, num_repeat)
    final = ""
    sample_size = "one_rep"  # 'all_reps

    return seq, final, [period], num_gates, sample_size


if __name__ == "__main__":
    import time

    import matplotlib.pylab as plt
    from qualang_tools.results import fetching_tool, progress_counter

    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117", port="80")

    # readout_time = 3e3
    max_readout_time = config["PhotonCollection"]["qm_opx_max_readout_time"]

    qm = qmm.open_qm(config_opx)
    simulation_duration = 80000 // 4  # clock cycle units - 4ns
    num_repeat = 2
    # init_pulse_time, readout_time, init_laser_key, readout_laser_key,\
    # init_laser_power, read_laser_power, readout_on_pulse_ind, apd_index  = args

    # start_t = time.time()
    # compilied_program_id = qm.compile(seq)
    # t1 = time.time()
    # print(t1 - start_t)

    # program_job = qm.queue.add_compiled(compilied_program_id)
    # job = program_job.wait_for_execution()
    # print(time.time()-t1)

    # job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    # job_sim.get_simulated_samples().con1.plot()
    # plt.show()
    #
    # print(time.time())
    # print(time.time())
    # job = qm.execute(seq)
    # st = time.time()
    args = [
        1000,
        300,
        2000,
        "cobolt_515",
        "cobolt_638",
        "laserglow_589",
        1,
        1,
        0.4,
        2,
        0,
    ]
    seq, f, p, ng, ss = get_seq([], config, args, num_repeat)
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    plt.show()
    # job = qm.execute(seq)

    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1","times_apd0","times_apd1"], mode="wait_for_all")
    # counts_apd0, counts_apd1, times_apd0, times_apd1 = results.fetch_all()
    # # print(counts_apd0)
    # print(np.sum(counts_apd0))

    # args = [10000, 200e6, 'cobolt_638', 'laserglow_589',1,1,2,0]
    # seq , f, p, ng, ss = get_seq([],config, args, num_repeat)
    # job = qm.execute(seq)
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1","times_apd0","times_apd1"], mode="wait_for_all")
    # counts_apd0, counts_apd1, times_apd0, times_apd1 = results.fetch_all()
    # # print(counts_apd0)
    # print(np.sum(counts_apd0))
    # print(times_apd0)
    # print(time.time() - st)

    # print('')
    # print(np.shape(counts_apd0.tolist()))
    # # print('')
    # print(np.shape(counts_apd1.tolist()))
    # time.sleep(2)
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="live")
    # counts_apd0, counts_apd1 = results.fetch_all()

    # # print('')
    # print(np.shape(counts_apd0.tolist()))
    # # print('')
    # print(np.shape(counts_apd1.tolist()))
    # print(counts_apd0.tolist())

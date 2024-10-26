# -*- coding: utf-8 -*-
"""
Plot the counts obtained by moving the AOM on time so that we can
determine the delay of the AOM relative to the APD gating.

The apd gate is delayed by tau, which is swept through a range of times. The
apd gate is offset from the end of the laser pulse by 500 ns. As tau is
increased, the apd gate moves closer (and eventually past) the laser pulse.
So if there were no delays betwee nthe laser and apd, the apd gate would just
stop overlapping with the laser pulse at 500 ns.

For laser delays, the end of the tail of the pulse shoudl be at 500 ns. If it occurs
later thatn 500 ns, the difference is the delay added at the beginning of
all other sequence trains.

Created on Fri Jul 12 13:53:45 2019

@author: mccambria
"""


# %% Imports


import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy

import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt
from utils.tool_belt import States

# %% Functions


def measure_delay(
    cxn,
    nv_sig,
    delay_range,
    num_steps,
    num_reps,
    seq_file,
    state=States.LOW,
    laser_name=None,
    laser_power=None,
):
    taus = numpy.linspace(delay_range[0], delay_range[1], num_steps)
    max_tau = delay_range[1]
    tau_ind_list = list(range(num_steps))
    shuffle(tau_ind_list)

    sig_counts = numpy.empty(num_steps)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)

    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)

    tool_belt.reset_cfm(cxn)

    if "charge_readout_laser_filter" in nv_sig:
        tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")

    tool_belt.init_safe_stop()

    n = 0
    for tau_ind in tau_ind_list:
        if tool_belt.safe_stop():
            break

        st = time.time()
        # optimize.main_with_cxn(cxn, nv_sig)
        targeting.main_with_cxn(cxn, nv_sig)

        # charge_readout_laser_server = tool_belt.get_server_charge_readout_laser(cxn)
        # charge_readout_laser_server.load_feedthrough(1.0)

        # Turn on the microwaves for determining microwave delay
        sig_gen = None
        if seq_file == "uwave_delay.py":
            delayed_element = "uwave"
            sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)
            sig_gen_cxn.set_freq(nv_sig["resonance_{}".format(state.name)])
            sig_gen_cxn.set_amp(nv_sig["uwave_power_{}".format(state.name)])
            sig_gen_cxn.uwave_on()

            pi_pulse = round(nv_sig["rabi_{}".format(state.name)] / 2)

        if seq_file == "iq_delay.py":
            delayed_element = "iq"
            sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)
            sig_gen_cxn.set_freq(nv_sig["resonance_{}".format(state.name)])
            sig_gen_cxn.set_amp(nv_sig["uwave_power_{}".format(state.name)])
            sig_gen_cxn.load_iq()
            sig_gen_cxn.uwave_on()
            awg_cxn = tool_belt.get_server_arb_wave_gen(cxn)
            awg_cxn.load_arb_phases([0, numpy.pi / 2])
            # awg_cxn.load_arb_phases([numpy.pi/2, 0])
            pi_pulse = round(nv_sig["rabi_{}".format(state.name)] / 2)

        if laser_name == "laser_LGLO_589":
            laser_server = cxn.laser_LGLO_589
            laser_server.load_feedthrough(laser_power)

        counter_server.start_tag_stream()
        ###########

        # Break out of the while if the user says stop
        # if tool_belt.safe_stop():
        #     break

        tau = taus[tau_ind]
        print("Index #{}/{}: {} ns".format(n, num_steps - 1, tau))
        n += 1
        # print(tau)
        if seq_file == "aom_delay.py":
            delayed_element = "laser"
            readout = 5e3  # nv_sig["imaging_readout_dur"]
            seq_args = [
                tau,
                max_tau,
                readout,
                laser_name,
                laser_power,
            ]
            print(seq_args)
        elif seq_file == "uwave_delay.py" or seq_file == "iq_delay.py":
            laser_key = "spin_laser"
            laser_name = nv_sig[laser_key]
            laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
            readout = nv_sig["spin_readout_dur"]
            polarization = nv_sig["spin_pol_dur"]
            seq_args = [
                tau,
                max_tau,
                readout,
                pi_pulse,
                polarization,
                state.value,
                laser_name,
                laser_power,
            ]
            # print(seq_args)
        # elif seq_file == "iq_delay.py":
        #     laser_key = "spin_laser"
        #     laser_name = nv_sig[laser_key]
        #     laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        #     readout = nv_sig["spin_readout_dur"]
        #     polarization = nv_sig["spin_pol_dur"]
        #     seq_args = [
        #         tau,
        #         max_tau,
        #         readout,
        #         pi_pulse,
        #         polarization,
        #         state.value,
        #         laser_name,
        #         laser_power,
        #     ]

        # print(seq_args)
        # return
        # Clear the counter_server buffer of any excess counts
        counter_server.clear_buffer()
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        # print(seq_args_string)
        # print(seq_file)
        # print(num_reps)
        pulsegen_server.stream_immediate(seq_file, num_reps, seq_args_string)
        # print('here')
        # complete_counts = counter_server.read_counter_complete()

        # new_counts = counter_server.read_counter_separate_gates(1)
        new_counts = counter_server.read_counter_modulo_gates(2, 1)

        # print('here2')
        sample_counts = new_counts[0]
        # print(sample_counts)
        # if len(sample_counts) != 2 * num_reps:
        #     print("Error!")
        ref_counts[tau_ind] = sample_counts[0]  # sum(sample_counts[0::2])
        sig_counts[tau_ind] = sample_counts[1]  # sum(sample_counts[1::2])

        print("run time:", time.time() - st)

    counter_server.stop_tag_stream()

    tool_belt.reset_cfm(cxn)

    # kcps
    #    sig_count_rates = (sig_counts / (num_reps * 1000)) / (readout / (10**9))
    #    ref_count_rates = (ref_counts / (num_reps * 1000)) / (readout / (10**9))
    norm_avg_sig = sig_counts / numpy.average(ref_counts)
    # norm_avg_sig = sig_counts / ref_counts

    fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    ax = axes_pack[0]
    ax.plot(taus, sig_counts, "r-", label="signal")
    ax.plot(taus, ref_counts, "g-", label="reference")
    ax.set_title("Counts vs Delay Time")
    ax.set_xlabel("{} Delay time (ns)".format(delayed_element))
    ax.set_ylabel("Counts")
    ax.legend()
    ax = axes_pack[1]
    ax.plot(taus, norm_avg_sig, "b-")
    ax.set_title("Contrast vs Delay Time")
    ax.set_xlabel("Delay time (ns)")
    ax.set_ylabel("Contrast (arb. units)")
    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()

    timestamp = tool_belt.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "sequence": seq_file,
        "laser_name": laser_name,
        "state": state.name,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(cxn),
        "delay_range": delay_range,
        "delay_range-units": "ns",
        "num_steps": num_steps,
        "num_reps": num_reps,
        "sig_counts": sig_counts.astype(int).tolist(),
        "sig_counts-units": "counts",
        "ref_counts": ref_counts.astype(int).tolist(),
        "ref_counts-units": "counts",
        "norm_avg_sig": norm_avg_sig.astype(float).tolist(),
        "norm_avg_sig-units": "arb",
    }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)

    # if tool_belt.check_safe_stop_alive():
    #     print("\n\nRoutine complete. Press enter to exit.")
    #     tool_belt.poll_safe_stop()


# %% Mains


def aom_delay(
    cxn,
    nv_sig,
    delay_range,
    num_steps,
    num_reps,
    laser_name,
    laser_power,
):
    """
    This will repeatedly run the same sequence with different passed laser
    delays. If there were no delays, the sequence would look like this
    laser ________|--------|________|--------|___
    APD   ___________|--|_________________|--|___
    The first readout is a reference - the laser should be on long enough such
    that the readout is roughly in the middle of the laser pulse regardless of
    of the actual laser delay. The second readout is a signal. We should see
    a normalized signal consistent with unity. If there is a delay we'll get
    this sequence
    laser __________|--------|________|--------|_
    APD   ___________|--|_________________|--|___
    and the normalized signal will still be unity. If the passed delay is
    excessive then we'll get this
    laser ______|--------|________|--------|_____
    APD   ___________|--|_________________|--|___
    and the normalized signal will be below unity. So we need to find the
    maximum passed delay that brings the normalized signal to unity before it
    starts to fall off.
    """

    seq_file = "aom_delay.py"

    measure_delay(
        cxn,
        nv_sig,
        delay_range,
        num_steps,
        num_reps,
        seq_file,
        laser_name=laser_name,
        laser_power=laser_power,
    )


def uwave_delay(cxn, nv_sig, state, delay_range, num_steps, num_reps):
    """
    This will repeatedly run the same sequence with different passed microwave
    delays. If there were no delays, the sequence would look like this
    uwave ______________________|---|____________
    laser ________|--------|________|--------|___
    APD   ________|----|____________|----|_______
    The first readout is a reference, the second is a signal. We should see
    a normalized signal consistent with the full pi pulse contrast. If there is
    a delay we'll get this sequence
    uwave ________________________|---|__________
    laser ________|--------|________|--------|___
    APD   ________|----|____________|----|_______
    and the normalized signal will be higher than the full pi pulse contrast.
    We need to find the minimum passed delay that recovers the full contrast.
    (This function assumes the laser delay is properly set!)
    """

    seq_file = "uwave_delay.py"

    measure_delay(
        cxn,
        nv_sig,
        delay_range,
        num_steps,
        num_reps,
        seq_file,
        state=state,
    )


def iq_delay(cxn, nv_sig, state, delay_range, num_steps, num_reps):
    """
    This will repeatedly run the same sequence with different passed iq
    delays. If there were no delays, the sequence would look like this

    iq    |-|_________________|-|________________
    uwave ____________________|---|______________
    laser ________|--------|________|--------|___
    APD   ________|----|____________|----|_______

    The first readout is a reference, the second is a signal. The iq modulation
    initially is at 0 degrees, and the second pulse changes it to pi/2.
    We should see a normalized signal consistent with the full pi pulse contrast.
    If there is a positive delay we'll get this sequence

    iq    __|-|_________________|-|______________
    uwave ____________________|---|______________
    laser ________|--------|________|--------|___
    APD   ________|----|____________|----|_______

    and the normalized signal will be higher than the full pi pulse contrast.
    The signal will reduce in contrast as the iq trigger passes over the pi pulse.
    The correct delay is when the counts return to their full contrast.

    |      __
    |     /  \
    |____/    \___
    -----------------
              * This is the value of the correct delay
    (This function assumes the laser delay and uwave delay are properly set!)
    """

    seq_file = "iq_delay.py"

    measure_delay(
        cxn,
        nv_sig,
        delay_range,
        num_steps,
        num_reps,
        seq_file,
        state=state,
    )


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == "__main__":
    # Carr parameters

    # Rabi parameters

    apd_indices = [1]
    sample_name = "siena"
    # sample_name = 'ayrton12'
    green_power = 8000
    nd_green = "ND_1.1"
    green_laser = "integrated_520"
    # green_laser = "cobolt_515"
    yellow_laser = "laser_LGLO_589"
    red_laser = "cobolt_638"

    nv_sig = {
        "coords": [0.032, -0.148, 7.44],
        "name": "{}-nv0_2023_03_20".format(
            sample_name,
        ),
        "disable_opt": False,
        "ramp_voltages": False,
        "expected_count_rate": 13,
        "spin_laser": green_laser,
        "spin_laser_power": green_power,
        "spin_laser_filter": nd_green,
        "spin_readout_dur": 400,
        "spin_pol_dur": 1000.0,
        "imaging_laser": green_laser,
        "imaging_laser_power": green_power,
        "imaging_laser_filter": nd_green,
        "imaging_readout_dur": 1e7,
        "charge_readout_laser": yellow_laser,
        "charge_readout_laser_filter": "nd_0.5",
        "collection_filter": "715_sp+630_lp",  # NV band only
        "magnet_angle": 163,
        "resonance_LOW": 2.82309,
        "rabi_LOW": 39.33 * 2,
        "uwave_power_LOW": -2,
        "resonance_HIGH": 2.91872,
        "rabi_HIGH": 41.91 * 2,
        "uwave_power_HIGH": -2,
    }

    # Hahn parameters
    # apd_indices = [1]
    # sample_name = 'johnson'
    # green_laser = "integrated_520"
    # yellow_laser = "laserglow_589"
    # red_laser = "cobolt_638"

    # nv_sig = { 'coords': [-0.156, 0.030, 5.7], 'name': '{}-nv0_2022_04_06'.format(sample_name),
    #         'disable_opt': False, "disable_z_opt": False, 'expected_count_rate': 13,

    #         # 'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1E7,
    #         # 'imaging_laser': yellow_laser, 'imaging_laser_power': 1.0, 'imaging_readout_dur': 1e8,
    #         'imaging_laser': green_laser, 'imaging_readout_dur': 1e4,
    #         'spin_laser': green_laser, 'spin_laser_filter': 'nd_0.5', 'spin_pol_dur': 1E5, 'spin_readout_dur': 350,
    #         # 'spin_laser': green_laser, 'spin_laser_filter': 'nd_0', 'spin_pol_dur': 1E4, 'spin_readout_dur': 300,

    #         'nv-_reionization_laser': green_laser, 'nv-_reionization_dur': 1E6, 'nv-_reionization_laser_filter': 'nd_1.0',
    #         # 'nv-_reionization_laser': green_laser, 'nv-_reionization_dur': 1E5, 'nv-_reionization_laser_filter': 'nd_0.5',
    #         'nv-_prep_laser': green_laser, 'nv-_prep_laser_dur': 1E6, 'nv-_prep_laser_filter': 'nd_1.0',

    #         'nv0_ionization_laser': red_laser, 'nv0_ionization_dur': 200,
    #         'nv0_prep_laser': red_laser, 'nv0_prep_laser_dur': 1e3,

    #         'spin_shelf_laser': yellow_laser, 'spin_shelf_dur': 0, 'spin_shelf_laser_power': 1.0,
    #         # 'spin_shelf_laser': green_laser, 'spin_shelf_dur': 50,
    #         "initialize_laser": green_laser, "initialize_dur": 1e4,
    #         # "charge_readout_laser": yellow_laser, "charge_readout_dur": 700e6, "charge_readout_laser_power": 0.71,
    #         "charge_readout_laser": yellow_laser, "charge_readout_dur": 32e6, "charge_readout_laser_power": 1.0,

    #         "collection_filter": "630_lp", 'magnet_angle': None,
    #         'resonance_LOW': 2.8000, 'rabi_LOW': 133.6, 'uwave_power_LOW': 16.5,
    #         'resonance_HIGH': 2.9416, 'rabi_HIGH': 181.0, 'uwave_power_HIGH': 16.5}

    # laser delay
    num_steps = 51
    num_reps = int(5e4)
    # laser_name = 'laserglow_532'
    # delay_range = [0, 500]
    # num_reps = int(1e5)
    laser_name = "laser_LGLO_589"
    delay_range = [500, 2500]
    laser_power = 1.0
    # num_reps = int(1e4)
    # laser_name = 'integrated_520'
    # laser_name = 'cobolt_515'
    # laser_power = 0.65
    # laser_name = 'cobolt_638'
    # laser_power = None
    # laser_name = 'laserglow_589'
    # laser_power = 0.6
    # delay_range = [0,500]
    with labrad.connect() as cxn:
        aom_delay(
            cxn, nv_sig, delay_range, num_steps, num_reps, laser_name, laser_power
        )

    # uwave_delay
    num_reps = int(2e5)
    # num_reps = int(1e4)
    # delay_range = [-310, 490]
    delay_range = [555, 655]
    num_steps = 101
    # bnc 835
    # state = States.LOW
    # sg394
    state = States.HIGH
    # with labrad.connect() as cxn:
    #     iq_delay(
    #           cxn,
    #           nv_sig,
    #           state,
    #           delay_range,
    #           num_steps,
    #           num_reps,
    #       )
    # uwave_delay(
    #     cxn,
    #     nv_sig,
    #     state,
    #     delay_range,
    #     num_steps,
    #     num_reps,
    # )
    # uwave_delay(
    #    cxn,
    #     nv_sig,
    #     States.LOW,
    #     delay_range,
    #     num_steps,
    #     num_reps,
    #     green_laser,
    #     1,
    # )
    # fm_delay(
    #     cxn,
    #     nv_sig,
    #     state,
    #     delay_range,
    #     num_steps,
    #     num_reps,
    # )

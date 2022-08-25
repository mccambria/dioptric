# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:17:35 2022

based off this paper: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601

@author: kolkowitz
"""


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
from scipy.optimize import minimize_scalar
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from random import shuffle
import labrad
from utils.tool_belt import States
from scipy.optimize import curve_fit


def test_1():
    '''
    This pulse sequence consists of pi/2 pulses with the same phase:
        pi/2 - pi/2
    '''
    
    
    return
def test_iq(
    cxn,
    nv_sig,
    apd_indices,
    num_pi_pulse_steps,
    num_reps,
    state=States.HIGH,
):

    # pi_indices = numpy.linspace(0, num_pi_pulse_steps, num_pi_pulse_steps+1)
    # max_pi_ind = pi_indices[1]
    pi_ind_list = list(range(num_pi_pulse_steps))
    shuffle(pi_ind_list)

    sig_counts = numpy.empty(num_pi_pulse_steps)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)


    tool_belt.reset_cfm(cxn)
        
    if 'charge_readout_laser_filter' in nv_sig:
        tool_belt.set_filter(cxn, nv_sig, 'charge_readout_laser')
    seq_file = 'test_iq_mod.py'

    # tool_belt.init_safe_stop()
    
    n= 0
    for pi_ind in pi_ind_list:
        
        optimize.main_with_cxn(cxn, nv_sig, apd_indices)
        # Turn on the microwaves for determining microwave delayabi_{}".format(state.name)] / 2)
        
        if pi_ind == 0:
            phases = [0,0,0]
        else:
            phases = [0, 0] + [numpy.pi/2]*pi_ind + [0]
        
        # if seq_file == "iq_delay.py":
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_freq(nv_sig["resonance_{}".format(state.name)])
        sig_gen_cxn.set_amp(nv_sig["uwave_power_{}".format(state.name)])
        sig_gen_cxn.load_iq()
        sig_gen_cxn.uwave_on()
        cxn.arbitrary_waveform_generator.load_arb_phases(phases)
        pi_pulse = round(nv_sig["rabi_{}".format(state.name)] / 2)
        pi_on_2_pulse = round(nv_sig["rabi_{}".format(state.name)] / 4)

        cxn.apd_tagger.start_tag_stream(apd_indices)
        ###########
    
        print('Index #{}/{}'.format(n, num_pi_pulse_steps-1))
        n+=1
        
        laser_key = "spin_laser"
        laser_name = nv_sig[laser_key]
        laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        readout = nv_sig["spin_readout_dur"]
        polarization = nv_sig["spin_pol_dur"]
        seq_args = [
            readout,
            pi_pulse,
            pi_on_2_pulse,
            polarization,
            pi_ind,
            state.value,
            apd_indices[0],
            laser_name,
            laser_power,
        ]

        # print(seq_args)
        # return
        # Clear the tagger buffer of any excess counts
        cxn.apd_tagger.clear_buffer()
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate(
            seq_file, num_reps, seq_args_string
        )

        new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
        sample_counts = new_counts[0]
        if len(sample_counts) != 2 * num_reps:
            print("Error!")
        ref_counts[pi_ind] = sum(sample_counts[0::2])
        sig_counts[pi_ind] = sum(sample_counts[1::2])

    cxn.apd_tagger.stop_tag_stream()

    tool_belt.reset_cfm(cxn)

    # kcps
    #    sig_count_rates = (sig_counts / (num_reps * 1000)) / (readout / (10**9))
    #    ref_count_rates = (ref_counts / (num_reps * 1000)) / (readout / (10**9))
    norm_avg_sig = sig_counts / ref_counts

    fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    ax = axes_pack[0]
    ax.plot(pi_ind_list, sig_counts, "r-", label="signal")
    ax.plot(pi_ind_list, ref_counts, "g-", label="reference")
    # ax.set_title("Num")
    ax.set_xlabel("Number of pi pulses")
    ax.set_ylabel("Counts")
    ax.legend()
    ax = axes_pack[1]
    ax.plot(pi_ind_list, norm_avg_sig, "b-")
    # ax.set_title("Contrast vs Delay Time")
    ax.set_xlabel("Number of pi pulses")
    ax.set_ylabel("Contrast (arb. units)")
    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()

    timestamp = tool_belt.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "sequence": seq_file,
        "laser_name": laser_name,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "num_pi_pulse_steps": num_pi_pulse_steps,
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

    return
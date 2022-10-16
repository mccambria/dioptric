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


import labrad
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import majorroutines.optimize_digital as optimize_digital
from random import shuffle
import numpy
import matplotlib.pyplot as plt
from utils.tool_belt import States
import time


# %% Functions


def measure_delay(
    cxn,
    nv_sig,
    apd_indices,
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
    
    counter_server = tool_belt.get_counter_server(cxn)
    pulsegen_server = tool_belt.get_pulsegen_server(cxn)


    tool_belt.reset_cfm(cxn)
        
    if 'charge_readout_laser_filter' in nv_sig:
        tool_belt.set_filter(cxn, nv_sig, 'charge_readout_laser')
        

    # tool_belt.init_safe_stop()
    
    n= 0
    for tau_ind in tau_ind_list:
        
        # optimize.main_with_cxn(cxn, nv_sig, apd_indices)
        optimize_digital.main_with_cxn(cxn, nv_sig, apd_indices)
        # Turn on the microwaves for determining microwave delay
        sig_gen = None
        if seq_file == "uwave_delay.py":
            sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
            sig_gen_cxn.set_freq(nv_sig["resonance_{}".format(state.name)])
            sig_gen_cxn.set_amp(nv_sig["uwave_power_{}".format(state.name)])
            sig_gen_cxn.uwave_on()
            pi_pulse = round(nv_sig["rabi_{}".format(state.name)] / 2)
            
        if seq_file == "iq_delay.py":
            sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
            sig_gen_cxn.set_freq(nv_sig["resonance_{}".format(state.name)])
            sig_gen_cxn.set_amp(nv_sig["uwave_power_{}".format(state.name)])
            sig_gen_cxn.load_iq()
            sig_gen_cxn.uwave_on()
            cxn.arbitrary_waveform_generator.load_arb_phases([0, numpy.pi/2])
            pi_pulse = round(nv_sig["rabi_{}".format(state.name)] / 2)

        counter_server.start_tag_stream(apd_indices)
        ###########
    
        # Break out of the while if the user says stop
        # if tool_belt.safe_stop():
        #     break
        
        tau = taus[tau_ind]
        print('Index #{}/{}: {} ns'.format(n, num_steps-1,tau))
        n+=1
        # print(tau)
        if seq_file == "aom_delay.py":
            readout = 5e3#,nv_sig["imaging_readout_dur"]
            seq_args = [
                tau,
                max_tau, 
                readout,
                apd_indices[0],
                laser_name,
                laser_power,
            ]
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
                apd_indices[0],
                laser_name,
                laser_power,
            ]
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
        #         apd_indices[0],
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
        pulsegen_server.stream_immediate(
            seq_file, num_reps, seq_args_string
        )
        # print('here')
        # complete_counts = counter_server.read_counter_complete()

        new_counts = counter_server.read_counter_separate_gates(1)
        # print('here2')
        sample_counts = new_counts[0]
        # print(sample_counts)
        if len(sample_counts) != 2 * num_reps:
            print("Error!")
        ref_counts[tau_ind] = sum(sample_counts[0::2])
        sig_counts[tau_ind] = sum(sample_counts[1::2])

    counter_server.stop_tag_stream()

    tool_belt.reset_cfm(cxn)

    # kcps
    #    sig_count_rates = (sig_counts / (num_reps * 1000)) / (readout / (10**9))
    #    ref_count_rates = (ref_counts / (num_reps * 1000)) / (readout / (10**9))
    norm_avg_sig = sig_counts / ref_counts

    fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    ax = axes_pack[0]
    ax.plot(taus, sig_counts, "r-", label="signal")
    ax.plot(taus, ref_counts, "g-", label="reference")
    ax.set_title("Counts vs Delay Time")
    ax.set_xlabel("Delay time (ns)")
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
        "sig_gen": sig_gen,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
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
    apd_indices,
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
        apd_indices,
        delay_range,
        num_steps,
        num_reps,
        seq_file,
        laser_name=laser_name,
        laser_power=laser_power,
    )


def uwave_delay(
    cxn, nv_sig, apd_indices, state, delay_range, num_steps, num_reps
):

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
        apd_indices,
        delay_range,
        num_steps,
        num_reps,
        seq_file,
        state=state,
    )

def iq_delay(
    cxn, nv_sig, apd_indices, state, delay_range, num_steps, num_reps
):

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
    If there is a delay we'll get this sequence
    
    iq    __|-|_________________|-|______________
    uwave ____________________|---|______________
    laser ________|--------|________|--------|___
    APD   ________|----|____________|----|_______
    
    and the normalized signal will be higher than the full pi pulse contrast.
    We need to find the minimum passed delay that recovers the full contrast.
    (This function assumes the laser delay and uwave delay are properly set!)
    """

    seq_file = "iq_delay.py"

    measure_delay(
        cxn,
        nv_sig,
        apd_indices,
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
    
    with labrad.connect() as cxn:
        apd_indices = tool_belt.get_registry_entry(cxn, "apd_indices", ["","Config"])
        apd_indices = apd_indices.astype(list).tolist()
        

    sample_name = "johnson"
    green_laser = "cobolt_515"

    nv_sig = {
        'coords': [84.605, 37.951, 69.28], 'name': '{}-search'.format(sample_name),
        'ramp_voltages': False,
        "only_z_opt": False,
        'disable_opt': False, "disable_z_opt": False, 'expected_count_rate': 47,
        "imaging_laser": green_laser, "imaging_laser_filter": "nd_0", "imaging_readout_dur": 1e7,
        "spin_laser": green_laser,
        "spin_laser_filter": "nd_0",
        "spin_pol_dur": 100e3,
        "spin_readout_dur": 2e3,
        "nv-_reionization_laser": green_laser,
        "nv-_reionization_dur": 1e6,
        "nv-_reionization_laser_filter": "nd_0",
        "nv-_prep_laser": green_laser,
        "nv-_prep_laser_dur": 1e6,
        "nv-_prep_laser_filter": "nd_0",
        "initialize_laser": green_laser,
        "initialize_dur": 1e4,
        'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': 2.8059, 'rabi_LOW': 226.9, 'uwave_power_LOW': 16.5,
        'resonance_HIGH': 2.9363, 'rabi_HIGH': 300, 'uwave_power_HIGH': 16.5,
        }
    
    # laser_delay
    num_reps = int(5e6)
    delay_range = [100, 550]
    num_steps = 21
    # bnc 835
    # state = States.LOW
    #  sg394
    state = States.HIGH
    with labrad.connect() as cxn:
        aom_delay(
            cxn,
            nv_sig,
            apd_indices,
            delay_range,
            num_steps,
            num_reps,
            green_laser,
            1,
        )

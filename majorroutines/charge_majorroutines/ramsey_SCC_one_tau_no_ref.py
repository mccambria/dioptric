# -*- coding: utf-8 -*-
"""
Ramsey measruement.

This routine polarizes the nv state into 0, then applies a pi/2 pulse to
put the state into a superposition between the 0 and + or - 1 state. The state
then evolves for a time, tau, of free precesion, and then a second pi/2 pulse
is applied. The amount of population in 0 is read out by collecting the
fluorescence during a readout.

It then takes a fast fourier transform of the time data to attempt to extract
the frequencies in the ramsey experiment. If the funtion can't determine the
peaks in the fft, then a detuning is used.

Lastly, this file curve_fits the data to a triple sum of cosines using the
found frequencies.

Created on Wed Apr 24 15:01:04 2019

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
from scipy.signal import find_peaks
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from random import shuffle
import os
import labrad
from utils.tool_belt import States
from scipy.optimize import curve_fit
optimization_type = tool_belt.get_optimization_style()
if optimization_type == 'DISCRETE':
    import majorroutines.optimize_digital as optimize
if optimization_type == 'CONTINUOUS':
    import majorroutines.optimize as optimize



def main(
    nv_sig,
    apd_indices,
    detuning,
    precession_time,
    num_reps,
    state=States.LOW,
    conditional_logic=False,
    photon_threshold=None,
    chop_factor=None
):

    with labrad.connect() as cxn:
        main_with_cxn(
            cxn,
            nv_sig,
            apd_indices,
            detuning,
            precession_time,
            num_reps,
            state,
            conditional_logic,
            photon_threshold,
            chop_factor
        )


def main_with_cxn(
    cxn,
    nv_sig,
    apd_indices,
    detuning,
    precession_time,
    num_reps,
    state=States.LOW,
    conditional_logic=False,
    photon_threshold=None,
    chop_factor=None
):
    
    counter_server = tool_belt.get_counter_server(cxn)
    pulsegen_server = tool_belt.get_pulsegen_server(cxn)
    

    tool_belt.reset_cfm(cxn)

    # %% Sequence setup

    green_laser_key = "nv-_reionization_laser"
    green_laser_name = nv_sig[green_laser_key]
    red_laser_key = "nv0_ionization_laser"
    red_laser_name = nv_sig[red_laser_key]
    yellow_laser_key = "charge_readout_laser"
    yellow_laser_name = nv_sig[yellow_laser_key]
    tool_belt.set_filter(cxn, nv_sig, green_laser_key)
    green_laser_power = tool_belt.set_laser_power(cxn, nv_sig, green_laser_key)
    tool_belt.set_filter(cxn, nv_sig, red_laser_key)
    red_laser_power = tool_belt.set_laser_power(cxn, nv_sig, red_laser_key)
    tool_belt.set_filter(cxn, nv_sig, yellow_laser_key)
    yellow_laser_power = tool_belt.set_laser_power(cxn, nv_sig, yellow_laser_key)
    
    polarization_time = nv_sig["nv-_reionization_dur"]
    ion_time = nv_sig['nv0_ionization_dur']
    gate_time = nv_sig["charge_readout_dur"]

    rabi_period = nv_sig["rabi_{}".format(state.name)]
    uwave_freq = nv_sig["resonance_{}".format(state.name)]
    uwave_power = nv_sig["uwave_power_{}".format(state.name)]
    # Detune the pi/2 pulse frequency
    uwave_freq_detuned = uwave_freq + detuning / 10**3

    # Get pulse frequencies
    uwave_pi_pulse = 0
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)
    
    if conditional_logic:
        seq_file_name = "ramsey_scc_noref_onetau_conditional.py"
        
    else:
        seq_file_name = "ramsey_scc_noref_onetau.py"
        
    precession_time = numpy.int32(precession_time)

    sig_counts = numpy.zeros([num_reps])
    sig_counts[:] = numpy.nan
    # ref_counts = numpy.copy(sig_counts)

    # %% Make some lists and variables to save at the end

    opti_coords_list = []

    # %% Analyze the sequence
    if conditional_logic:
        seq_args = [
            precession_time/2,
            polarization_time,
            ion_time,
            gate_time,
            uwave_pi_pulse,
            uwave_pi_on_2_pulse,
            apd_indices[0],
            state.value,
            green_laser_name, red_laser_name, yellow_laser_name,
            green_laser_power, red_laser_power, yellow_laser_power,
            photon_threshold,chop_factor]
    else:
        seq_args = [
            precession_time/2,
            polarization_time,
            ion_time,
            gate_time,
            uwave_pi_pulse,
            uwave_pi_on_2_pulse,
            apd_indices[0],
            state.value,
            green_laser_name, red_laser_name, yellow_laser_name,
            green_laser_power, red_laser_power, yellow_laser_power
            ]
    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(seq_file_name, seq_args_string)
    seq_time = ret_vals[0]


    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()


    # Break out of the while if the user says stop
    # Optimize and save the coords we found
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
    opti_coords_list.append(opti_coords)

    # Set up the microwaves
    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
    sig_gen_cxn.set_freq(uwave_freq_detuned)
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.uwave_on()

    # Set up the laser
    tool_belt.set_filter(cxn, nv_sig, green_laser_key)
    green_laser_power = tool_belt.set_laser_power(cxn, nv_sig, green_laser_key)
    tool_belt.set_filter(cxn, nv_sig, red_laser_key)
    red_laser_power = tool_belt.set_laser_power(cxn, nv_sig, red_laser_key)
    tool_belt.set_filter(cxn, nv_sig, yellow_laser_key)
    yellow_laser_power = tool_belt.set_laser_power(cxn, nv_sig, yellow_laser_key)

    # Load the APD
    counter_server.start_tag_stream(apd_indices)

    if conditional_logic:
        seq_args = [
            precession_time/2,
            polarization_time,
            ion_time,
            gate_time,
            uwave_pi_pulse,
            uwave_pi_on_2_pulse,
            apd_indices[0],
            state.value,
            green_laser_name, red_laser_name, yellow_laser_name,
            green_laser_power, red_laser_power, yellow_laser_power,
            photon_threshold, chop_factor]
    else:
        seq_args = [
            precession_time/2,
            polarization_time,
            ion_time,
            gate_time,
            uwave_pi_pulse,
            uwave_pi_on_2_pulse,
            apd_indices[0],
            state.value,
            green_laser_name, red_laser_name, yellow_laser_name,
            green_laser_power, red_laser_power, yellow_laser_power
            ]
    
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    # Clear the counter/tagger buffer of any excess counts
    counter_server.clear_buffer()
    print(seq_args)
    pulsegen_server.stream_immediate(seq_file_name, num_reps, seq_args_string)

    new_counts = counter_server.read_counter_separate_gates(1)
    sample_counts = new_counts[0]
    sig_counts = sample_counts
    
    if conditional_logic:
        num_readouts_per_rep, reinit_state_st = cxn.qm_opx.get_cond_logic_num_ops(2)
    else:
        num_readouts_per_rep = numpy.ones(num_reps)
        reinit_state_st = numpy.ones(num_reps)

    counter_server.stop_tag_stream()
        

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)


    # %% Plot the final data
    plot_figure=False
    if plot_figure:
        raw_fig, ax = plt.subplots(1, 1, figsize=(17, 8.5))
        ax.cla()
        ax.plot(sig_counts, "r-", label="signal")
        ax.set_xlabel(r"$\tau"+" = {}".format(precession_time)+" ($\mathrm{\mu s}$)")
        ax.set_ylabel("Counts")
        ax.legend()
        
        raw_fig.canvas.draw()
        raw_fig.set_tight_layout(True)
        raw_fig.canvas.flush_events()

    # %% Save the data

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "timeElapsed": timeElapsed,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        'detuning': detuning,
        'detuning-units': 'MHz',
        "gate_time": gate_time,
        "gate_time-units": "ns",
        "uwave_freq": uwave_freq_detuned,
        "uwave_freq-units": "GHz",
        "uwave_power": uwave_power,
        "uwave_power-units": "dBm",
        "rabi_period": rabi_period,
        "rabi_period-units": "ns",
        "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
        "uwave_pi_on_2_pulse-units": "ns",
        "precession_time": int(precession_time),
        "precession_time-units": "ns",
        "state": state.name,
        "num_reps": num_reps,
        "sig_counts": sig_counts.astype(int).tolist(),
        "sig_counts-units": "counts",
        "num_readouts_per_rep": num_readouts_per_rep.astype(int).tolist(),
        "reinit_state_true_false_each_rep": reinit_state_st.astype(int).tolist(),
        'chop_factor':chop_factor
    }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    if plot_figure:
        tool_belt.save_figure(raw_fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)

    # %% Fit and save figs
    
    return 


# %% Run the file


if __name__ == "__main__":

    
    
    path = "E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_Carr/branch_opx-setup/ramsey_scc_one_tau_no_ref/2022_11/readout_time+power_sweep"
    timestamp = tool_belt.get_time_stamp()
    # save_path1 = tool_belt.get_file_path(__file__, timestamp,'-timetraces_dur')
    # save_path2 = tool_belt.get_file_path(__file__, timestamp,'-hists_dur')
    
    # for filename in os.listdir(path):
        
    for filename in ["2022_11_22-16_23_02-johnson-search"]:
        
        data = tool_belt.get_raw_data(filename)
        nv_sig = data['nv_sig']
        readout_time = nv_sig['charge_readout_dur']
        readout_dur = int(nv_sig['charge_readout_dur']/1e3)
        readout_power = int(nv_sig['charge_readout_laser_power']*1000)
        init_time = int(nv_sig['nv-_reionization_dur'])
        averaging_times = numpy.array([75e3,300e3,670e3]) #us
        nbins = numpy.array(averaging_times/readout_dur,dtype=numpy.int32)
        sig_counts = numpy.array(data['sig_counts'])
        precession_time = data['precession_time']
        chop_factor = data['chop_factor']
        num_readouts_each_rep = numpy.array(data['num_readouts_per_rep'])
        did_we_initialize_each_rep = numpy.array(data["reinit_state_true_false_each_rep"])
        num_reps = data['num_reps']
        time_per_rep = numpy.zeros(num_reps)
        time_per_rep = num_readouts_each_rep*(readout_time+2000)/chop_factor \
            + init_time*did_we_initialize_each_rep \
                + 340 + 3000
        
        
        

        if True:
            timestamp = tool_belt.get_time_stamp()
            save_path1 = tool_belt.get_file_path(__file__, timestamp,'-timetraces_'+'_{}'.format(readout_dur)+
                                                 'ms_{}'.format(readout_power)+'mV_{}'.format(init_time)+'ns')
            save_path2 = tool_belt.get_file_path(__file__, timestamp,'-hists_'+'_{}'.format(readout_dur)+
                                                 'ms_{}'.format(readout_power)+'mV_{}'.format(init_time)+'ns')
            
            raw_fig1, axes = plt.subplots(3, 1, figsize=(17, 8.5),sharex=True)
            # axes.cla()
            
            nb=0
            for ax in axes:
                width=nbins[nb]
                binned_data = sig_counts[:(sig_counts.size // width) * width].reshape(-1, width).sum(axis=1)
                ts = numpy.linspace(readout_dur,readout_dur*len(sig_counts),len(binned_data))/1000000
                ax.plot(ts,binned_data, "r-",label='{} ms ({}$\sigma$)'.format(averaging_times[nb]/1e3,nb+1))
                ax.set_ylabel("Binned Counts (summed)")
                # plt.xlim(0,6)
                nb=nb+1
                ax.legend(loc='upper right')
            axes[2].set_xlabel('Time (s)')
            axes[0].set_title('readout = {} $\mu$s   power = {} V   init_time = {} ns'.format(readout_dur,readout_power/1000,init_time))
            plt.show()
            tool_belt.save_figure(raw_fig1, save_path1)
            
            raw_fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
            nb=0
            for ax2 in axes2:
                width=nbins[nb]
                binned_data = sig_counts[:(sig_counts.size // width) * width].reshape(-1, width).sum(axis=1)
                ax2.hist(binned_data,bins=10,histtype='step',linewidth=2,label='{} ms ({}$\sigma$)'.format(averaging_times[nb]/1e3,nb+1))
                ax2.set_xlabel('Binned Counts (summed)')
                ax2.legend(loc='upper right')
                nb=nb+1
            axes2[1].set_title('readout = {} $\mu$s   power = {} V   init_time = {} ns'.format(readout_dur,readout_power/1000,init_time))
            plt.show()
            
            tool_belt.save_figure(raw_fig2, save_path2)
    
    
    
        

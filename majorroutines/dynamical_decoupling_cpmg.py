# -*- coding: utf-8 -*-
"""
Dynamical decoupling CPMG



Created on Fri Aug 5 2022

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import majorroutines.optimize as optimize
from scipy.optimize import minimize_scalar
from utils.tool_belt import NormStyle
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from random import shuffle
import labrad
from utils.tool_belt import States
from scipy.optimize import curve_fit
from numpy.linalg import eigvals


# %% Constants


im = 0 + 1j
inv_sqrt_2 = 1 / numpy.sqrt(2)
gmuB = 2.8e-3  # gyromagnetic ratio in GHz / G

def create_fit_figure(
    taus,
    num_steps,
    norm_avg_sig,
    norm_avg_sig_ste,
    pi_pulse_reps,
    do_dq,
    fit_func,
    popt,
    pcov
):

    tau_T = 2*taus*pi_pulse_reps/1000

    smooth_taus = numpy.linspace(
        tau_T[0], tau_T[-1], num=1000
    )

    fit_fig, ax = plt.subplots()
    kpl.plot_points(ax, tau_T, norm_avg_sig, yerr=norm_avg_sig_ste, color = KplColors.BLUE, label = 'data')
    kpl.plot_line(
        ax,
        smooth_taus,
        fit_func(smooth_taus, *popt),
        color=KplColors.RED,
        label="fit",
    )
    ax.set_xlabel(r"$T = 2 \tau$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Contrast (arb. units)")
    ax.set_title("CPMG-{}".format(pi_pulse_reps))
    # ax.legend()
    
    t2_us = popt[1]/1000
    t2_unc_us = numpy.sqrt(pcov[1][1])/1000
    text_popt = "\n".join(
        (
            r"$S(T) = e^{(- (T/T_2)^3)}$",
            r"$T_2=$%.2f $\pm$ %.2f ms" % (t2_us, t2_unc_us),
        )
    )
    kpl.anchored_text(ax, text_popt, kpl.Loc.UPPER_RIGHT, size=kpl.Size.SMALL)
    
    if do_dq:
        ax.set_title("CPMG-{} DQ basis".format(pi_pulse_reps))
    else:
        ax.set_title("CPMG-{} SQ basis".format(pi_pulse_reps))

    fit_fig.canvas.draw()
    fit_fig.set_tight_layout(True)
    fit_fig.canvas.flush_events()

    return fit_fig

def fit_t2_12C(data, fixed_offset = None, incremental=False):
    '''
    for isotopically pure samples, this will fit t2
    '''
    
    if incremental:
        run_ind = data['run_ind']
        sig_counts = data['sig_counts']
        ref_counts = data['ref_counts']
        nv_sig = data['nv_sig']
        gate_time = nv_sig['spin_readout_dur']
        norm_style = NormStyle.SINGLE_VALUED
        num_reps = data['num_reps']
        ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, gate_time, norm_style)
        (
             sig_counts_avg_kcps,
             ref_counts_avg_kcps,
             norm_avg_sig,
             norm_avg_sig_ste,
        ) = ret_vals
    
    else:
        norm_avg_sig = data['norm_avg_sig']
        norm_avg_sig_ste = data['norm_avg_sig_ste']
    
    
    plot_taus = data['plot_taus']
    num_steps = data['num_steps']
    pi_pulse_reps = data['pi_pulse_reps']
    do_dq=data['do_dq']
    taus = numpy.array(data['taus'])
    
    if fixed_offset:
        fit_func = lambda x, amp, decay:tool_belt.exp_stretch_decay(x, amp, decay, fixed_offset, 3)
        init_params = [ -0.01, 1000]
    else:
        fit_func = lambda x, amp, decay, offset:tool_belt.exp_stretch_decay(x, amp, decay, offset, 3)
        init_params = [ -0.01, 3000, 1.01]
    
    popt, pcov = curve_fit(
        fit_func,
        plot_taus,
        norm_avg_sig,
        p0=init_params,
        absolute_sigma = True,
        sigma=norm_avg_sig_ste
    )
    print(popt)
    # popt = [-0.01, 1000]
    fig = create_fit_figure(
        taus,
        num_steps,
        norm_avg_sig,
        norm_avg_sig_ste,
        pi_pulse_reps,
        do_dq,
        fit_func,
        popt,
        pcov)
    
    return fig
    

def compile_12C_data(file_list, do_save = True):
    '''
    for a bunch of 12C files of the same type, combine data
    '''
    sig_counts = []
    ref_counts = []
    num_runs = 0
    for file in file_list:
        data = tool_belt.get_raw_data(file)
        sig_counts_i = data['sig_counts']
        ref_counts_i = data['ref_counts']
        num_runs_i = data['num_runs']
        
        sig_counts = sig_counts+sig_counts_i
        ref_counts = ref_counts+ref_counts_i
        num_runs  = num_runs + num_runs_i
        
    nv_sig = data['nv_sig']
    norm_style = NormStyle.SINGLE_VALUED
    # norm_style = NormStyle.POINT_TO_POINT
    gate_time = nv_sig['spin_readout_dur']
    num_reps = data['num_reps']
    ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, gate_time, norm_style)
    (
         sig_counts_avg_kcps,
         ref_counts_avg_kcps,
         norm_avg_sig,
         norm_avg_sig_ste,
    ) = ret_vals
    
    data['sig_counts']= sig_counts
    data['ref_counts']= ref_counts
    data['norm_avg_sig']= norm_avg_sig.tolist()
    data['norm_avg_sig_ste']= norm_avg_sig_ste.tolist()
    nv_sig['norm_style'] = norm_style
    data['nv_sig'] = nv_sig
    data['num_runs']= num_runs
    data['file_list']= file_list
    
    fig = fit_t2_12C(data)
    
    if do_save:
        timestamp = tool_belt.get_time_stamp()
        nv_name = nv_sig["name"]
        file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
        tool_belt.save_figure(fig, file_path)
        tool_belt.save_raw_data(data, file_path)
    

# %% Main


def main(
    nv_sig,
    precession_dur_range,
    pi_pulse_reps,
    num_steps,
    num_reps,
    num_runs,
    taus=[],
    state=States.HIGH,
    do_dq = False,
    do_scc = False,
    comp_wait_time = 80,
    dd_wait_time = 200,
    do_plot = True,
    do_save = True
):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(
            cxn,
            nv_sig,
            precession_dur_range,
            pi_pulse_reps,
            num_steps,
            num_reps,
            num_runs,
            taus,
            state,
            do_dq,
            do_scc,
            comp_wait_time,
            dd_wait_time,
            do_plot,
            do_save
        )
        return sig_counts, ref_counts


def main_with_cxn(
    cxn,
    nv_sig,
    precession_time_range,
    pi_pulse_reps,
    num_steps,
    num_reps,
    num_runs,
    taus=[],
    state=States.HIGH,
    do_dq = False,
    do_scc = False,
    comp_wait_time = 80,
    dd_wait_time = 200,
    do_plot = True,
    do_save = True
):

    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    arbwavegen_server = tool_belt.get_server_arb_wave_gen(cxn)
    
    tool_belt.reset_cfm(cxn)
    kpl.init_kplotlib()

    # %% Sequence setup

    
    if do_scc:
        laser_tag = "nv-_reionization"
        laser_key = "{}_laser".format(laser_tag)
        pol_laser_name = nv_sig[laser_key]
        pol_laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        polarization_dur = nv_sig["{}_dur".format(laser_tag)]

        laser_tag = "nv0_ionization"
        laser_key = "{}_laser".format(laser_tag)
        ion_laser_name = nv_sig[laser_key]
        ion_laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        ionization_dur = nv_sig["{}_dur".format(laser_tag)]

        laser_tag = "charge_readout"
        laser_key = "{}_laser".format(laser_tag)
        readout_laser_name = nv_sig[laser_key]
        readout_laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        readout = nv_sig["{}_dur".format(laser_tag)]
    else:
        laser_key = "spin_laser"
        laser_name = nv_sig[laser_key]
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        polarization_time = nv_sig["spin_pol_dur"]
        readout = nv_sig["spin_readout_dur"]
        
    norm_style = nv_sig['norm_style']
        

    rabi_period = nv_sig["rabi_{}".format(state.name)]
    uwave_freq = nv_sig["resonance_{}".format(state.name)]
    uwave_power = nv_sig["uwave_power_{}".format(state.name)]

    # Get pulse frequencies
    # uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    # uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)
    uwave_pi_pulse = nv_sig["pi_pulse_{}".format(state.name)]
    uwave_pi_on_2_pulse = nv_sig["pi_on_2_pulse_{}".format(state.name)]
    

    # set up to drive transition through zero
    if do_dq:
        
       # rabi_period_low = nv_sig["rabi_{}".format(States.LOW.name)]
        uwave_freq_low = nv_sig["resonance_{}".format(States.LOW.name)]
        uwave_power_low = nv_sig["uwave_power_{}".format(States.LOW.name)]
       # uwave_pi_pulse_low = tool_belt.get_pi_pulse_dur(rabi_period_low)
       # uwave_pi_on_2_pulse_low = tool_belt.get_pi_on_2_pulse_dur(rabi_period_low)
        uwave_pi_pulse_low = nv_sig["pi_pulse_{}".format(States.LOW.name)]
        uwave_pi_on_2_pulse_low = nv_sig["pi_on_2_pulse_{}".format(States.LOW.name)]
        #rabi_period_high = nv_sig["rabi_{}".format(States.HIGH.name)]
        uwave_freq_high = nv_sig["resonance_{}".format(States.HIGH.name)]
        uwave_power_high = nv_sig["uwave_power_{}".format(States.HIGH.name)]
        #uwave_pi_pulse_high = tool_belt.get_pi_pulse_dur(rabi_period_high)
       # uwave_pi_on_2_pulse_high = tool_belt.get_pi_on_2_pulse_dur(rabi_period_high)
        uwave_pi_pulse_high = nv_sig["pi_pulse_{}".format(States.HIGH.name)]
        uwave_pi_on_2_pulse_high = nv_sig["pi_on_2_pulse_{}".format(States.HIGH.name)]
        
        
        if state.value == States.LOW.value:
            state_activ = States.LOW
            state_proxy = States.HIGH
            
            coh_pulse_dur = uwave_pi_on_2_pulse_low + uwave_pi_pulse_high
            echo_pulse_dur = uwave_pi_pulse_high + uwave_pi_pulse_low + uwave_pi_pulse_high
        elif state.value == States.HIGH.value:
            state_activ = States.HIGH
            state_proxy = States.LOW
            
            coh_pulse_dur = uwave_pi_on_2_pulse_high + uwave_pi_pulse_low
            echo_pulse_dur = uwave_pi_pulse_low + uwave_pi_pulse_high + uwave_pi_pulse_low
    
    # %% Create the array of relaxation times

    # Array of times to sweep through
    # Must be ints
    min_precession_time = int(precession_time_range[0])
    max_precession_time = int(precession_time_range[1])

    if len(taus) == 0:
        taus = numpy.linspace(
            min_precession_time,
            max_precession_time,
            num=num_steps,
            dtype=numpy.int32,
        )
    # taus = taus + 500
    print(taus)
    # return
    # Convert to ms
    plot_taus = (taus * 2 * pi_pulse_reps) / 1000
    #if do_dq:
    #    plot_taus = (2* coh_pulse_dur + (2*taus + echo_pulse_dur)  * 2 *4 * pi_pulse_reps) / 1000

    # %% Fix the length of the sequence to account for odd amount of elements

    # Our sequence pairs the longest time with the shortest time, and steps
    # toward the middle. This means we only step through half of the length
    # of the time array.

    # That is a problem if the number of elements is odd. To fix this, we add
    # one to the length of the array. When this number is halfed and turned
    # into an integer, it will step through the middle element.

    if len(taus) % 2 == 0:
        half_length_taus = int(len(taus) / 2)
    elif len(taus) % 2 == 1:
        half_length_taus = int((len(taus) + 1) / 2)

    # Then we must use this half length to calculate the list of integers to be
    # shuffled for each run

    tau_ind_list = list(range(0, half_length_taus))

    # %% Create data structure to save the counts

    # We create an array of NaNs that we'll fill
    # incrementally for the signal and reference counts.
    # NaNs are ignored by matplotlib, which is why they're useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.

    sig_counts = numpy.zeros([num_runs, num_steps])
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)

    # %% Make some lists and variables to save at the end

    opti_coords_list = []
    tau_index_master_list = [[] for i in range(num_runs)]

    # %% Analyze the sequence
    
    num_reps = int(num_reps)

    if do_dq:
        if do_scc:
            seq_file_name = "dynamical_decoupling_dq_scc.py"
            seq_args = [
                taus[0],
                polarization_dur,
                ionization_dur,
                readout,
                uwave_pi_pulse_low,
                uwave_pi_on_2_pulse_low,
                uwave_pi_pulse_high,
                uwave_pi_on_2_pulse_high,
                taus[-1],
                comp_wait_time,
                pi_pulse_reps,
                state_activ.value,
                state_proxy.value,
                pol_laser_name,
                pol_laser_power,
                ion_laser_name,
                ion_laser_power,
                readout_laser_name,
                readout_laser_power,
            ]
        else:
            seq_file_name = "dynamical_decoupling_dq.py"
            seq_args = [
                  taus[0],
                polarization_time,
                readout,
                uwave_pi_pulse_low,
                uwave_pi_on_2_pulse_low,
                uwave_pi_pulse_high,
                uwave_pi_on_2_pulse_high,
                taus[-1],
                comp_wait_time,
                pi_pulse_reps,
                state_activ.value,
                state_proxy.value,
                laser_name, 
                laser_power
            ]
    else:
        if do_scc:    
            seq_file_name = "dynamical_decoupling_scc.py"
            seq_args = [
                  taus[0],
                  polarization_dur,
                  ionization_dur,
                  readout,
                  uwave_pi_pulse,
                  uwave_pi_on_2_pulse,
                  taus[-1],
                  dd_wait_time,
                  pi_pulse_reps,
                  state.value,
                  pol_laser_name,
                  pol_laser_power,
                  ion_laser_name,
                  ion_laser_power,
                  readout_laser_name,
                  readout_laser_power,
              ]
        else:
            seq_file_name = "dynamical_decoupling.py"
            seq_args = [
                  taus[0],
                  polarization_time,
                  readout,
                  uwave_pi_pulse,
                  uwave_pi_on_2_pulse,
                  taus[-1],
                  dd_wait_time,
                  pi_pulse_reps,
                  state.value,
                  laser_name,
                  laser_power,
              ]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(seq_file_name, seq_args_string)
    seq_time = ret_vals[0]
    print(seq_file_name)
    print(seq_args)
    # return
        # print(seq_time)

    # %% Let the user know how long this will take

    seq_time_s = seq_time / (10 ** 9)  # to seconds
    expected_run_time_s = (
        (num_steps / 2) * num_reps * num_runs * seq_time_s
    )  # s
    expected_run_time_m = expected_run_time_s / 60  # to minutes

    print(" \nExpected run time: {:.1f} minutes. ".format(expected_run_time_m))
    # return
    
    # create figure
    if do_plot:
        raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    
    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):

        print(" \nRun index: {}".format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize
        opti_coords = optimize.main_with_cxn(cxn, nv_sig)
        opti_coords_list.append(opti_coords)

        # Set up the microwaves
        
        if do_dq:
            sig_gen_low_cxn = tool_belt.get_server_sig_gen(cxn, States.LOW)
            sig_gen_low_cxn.set_freq(uwave_freq_low)
            sig_gen_low_cxn.set_amp(uwave_power_low)
            sig_gen_low_cxn.uwave_on()
            sig_gen_high_cxn = tool_belt.get_server_sig_gen(cxn, States.HIGH)
            sig_gen_high_cxn.set_freq(uwave_freq_high)
            sig_gen_high_cxn.set_amp(uwave_power_high)
            sig_gen_high_cxn.load_iq()
            sig_gen_high_cxn.uwave_on()
        else:
            sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)
            sig_gen_cxn.set_freq(uwave_freq)
            sig_gen_cxn.set_amp(uwave_power)
            sig_gen_cxn.load_iq()
            sig_gen_cxn.uwave_on()
        if pi_pulse_reps == 1:
              arbwavegen_server.load_arb_phases([0, 0, 0, 0, 0, 0]) 
        else:
              arbwavegen_server.load_cpmg(pi_pulse_reps)
            
        

        # Set up the laser
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        
        if do_scc:
            charge_readout_laser_server = tool_belt.get_server_charge_readout_laser(cxn)
            charge_readout_laser_server.load_feedthrough(nv_sig["charge_readout_laser_power"])

        # Load the APD
        counter_server.start_tag_stream()

        # Shuffle the list of tau indices so that it steps thru them randomly
        shuffle(tau_ind_list)

        for tau_ind in tau_ind_list:

            # 'Flip a coin' to determine which tau (long/shrt) is used first
            rand_boolean = numpy.random.randint(0, high=2)
            #rand_boolean = 1
            

            if rand_boolean == 1:
                tau_ind_first = tau_ind
                tau_ind_second = -tau_ind - 1
            elif rand_boolean == 0:
                tau_ind_first = -tau_ind - 1
                tau_ind_second = tau_ind

            # add the tau indexxes used to a list to save at the end
            tau_index_master_list[run_ind].append(tau_ind_first)
            tau_index_master_list[run_ind].append(tau_ind_second)

            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break

            print(" \nFirst relaxation time: {}".format(taus[tau_ind_first]))
            print("Second relaxation time: {}".format(taus[tau_ind_second]))

            
            if do_dq:
                if do_scc:
                    seq_args = [
                        taus[tau_ind_first],
                        polarization_dur,
                        ionization_dur,
                        readout,
                        uwave_pi_pulse_low,
                        uwave_pi_on_2_pulse_low,
                        uwave_pi_pulse_high,
                        uwave_pi_on_2_pulse_high,
                        taus[tau_ind_second],
                        comp_wait_time,
                        pi_pulse_reps,
                        state_activ.value,
                        state_proxy.value,
                        pol_laser_name,
                        pol_laser_power,
                        ion_laser_name,
                        ion_laser_power,
                        readout_laser_name,
                        readout_laser_power,
                    ]
                else:
                    seq_args = [
                        taus[tau_ind_first],
                        polarization_time,
                        readout,
                        uwave_pi_pulse_low,
                        uwave_pi_on_2_pulse_low,
                        uwave_pi_pulse_high,
                        uwave_pi_on_2_pulse_high,
                        taus[tau_ind_second],
                        comp_wait_time,
                        pi_pulse_reps,
                        state_activ.value,
                        state_proxy.value,
                        laser_name,
                        laser_power, 
                    ]
            else:
                if do_scc:    
                    # seq_file_name = "dynamical_decoupling_scc.py"
                    seq_args = [
                          taus[tau_ind_first],
                          polarization_dur,
                          ionization_dur,
                          readout,
                          uwave_pi_pulse,
                          uwave_pi_on_2_pulse,
                          taus[tau_ind_second],
                          dd_wait_time,
                          pi_pulse_reps,
                          state.value,
                          pol_laser_name,
                          pol_laser_power,
                          ion_laser_name,
                          ion_laser_power,
                          readout_laser_name,
                          readout_laser_power,
                      ]
                else:
                    # seq_file_name = "dynamical_decoupling.py"
                    seq_args = [
                          taus[tau_ind_first],
                          polarization_time,
                          readout,
                          uwave_pi_pulse,
                          uwave_pi_on_2_pulse,
                          taus[tau_ind_second],
                          dd_wait_time,
                          pi_pulse_reps,
                          state.value,
                          laser_name,
                          laser_power,
                      ]
            # print(seq_args)
            # return
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # Clear the tagger buffer of any excess counts
            # counter_server.clear_buffer()
            pulsegen_server.stream_immediate(
                seq_file_name, num_reps, seq_args_string
            )

            # Each sample is of the form [*(<sig_shrt>, <ref_shrt>, <sig_long>, <ref_long>)]
            # So we can sum on the values for similar index modulus 4 to
            # parse the returned list into what we want.
            new_counts = counter_server.read_counter_separate_gates(1)
            sample_counts = new_counts[0]
            # print(new_counts)

            count = sum(sample_counts[0::4])
            sig_counts[run_ind, tau_ind_first] = count
            print("First signal = " + str(count))

            count = sum(sample_counts[1::4])
            ref_counts[run_ind, tau_ind_first] = count
            print("First Reference = " + str(count))

            count = sum(sample_counts[2::4])
            sig_counts[run_ind, tau_ind_second] = count
            print("Second Signal = " + str(count))

            count = sum(sample_counts[3::4])
            ref_counts[run_ind, tau_ind_second] = count
            print("Second Reference = " + str(count))

        counter_server.stop_tag_stream()

        # %% incremental plotting
      
        # Average the counts over the iterations
        inc_sig_counts = sig_counts[: run_ind + 1]
        inc_ref_counts = ref_counts[: run_ind + 1]
        ret_vals = tool_belt.process_counts(
            inc_sig_counts, inc_ref_counts, num_reps, readout, norm_style
        )
        (
            sig_counts_avg_kcps,
            ref_counts_avg_kcps,
            norm_avg_sig,
            norm_avg_sig_ste,
        ) = ret_vals
        
        
        if do_plot:
            ax = axes_pack[0]
            ax.cla()
            ax.plot(plot_taus, sig_counts_avg_kcps, "r-", label="signal")
            ax.plot(plot_taus, ref_counts_avg_kcps, "g-", label="reference")
            ax.set_xlabel(r"Precession time, $T = 2 N \tau (\mathrm{\mu s}$)")
            ax.set_ylabel("kcps")
            ax.legend()
            
            ax = axes_pack[1]
            ax.cla()
            
            kpl.plot_points(ax, plot_taus, norm_avg_sig, yerr=norm_avg_sig_ste, color = KplColors.BLUE)
            # ax.errorbar(plot_taus, norm_avg_sig,yerr= norm_avg_sig_ste, color = 'b')
            # ax.plot(plot_taus, norm_avg_sig, "b-")
            if do_dq:
                dq_text = 'DQ'
            else:
                dq_text = 'SQ'
            if do_scc:
                ax.set_title("CPMG-{} {} SCC Measurement".format(pi_pulse_reps, dq_text))
            else:
                ax.set_title("CPMG-{} {} Measurement".format(pi_pulse_reps, dq_text))
                
            ax.set_xlabel(r"Precession time, $T = 2 N \tau (\mathrm{\mu s}$)")
            ax.set_ylabel("Contrast (arb. units)")
            
            text_popt = 'Run # {}/{}'.format(run_ind+1,num_runs)
    
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.8, 0.9, text_popt,transform=ax.transAxes,
                    verticalalignment='top', bbox=props)
            
            raw_fig.canvas.draw()
            raw_fig.set_tight_layout(True)
            raw_fig.canvas.flush_events()
        
        # %% Save the data we have incrementally for long T1s
        if do_save:
            raw_data = {
                "start_timestamp": start_timestamp,
                "nv_sig": nv_sig,
                "nv_sig-units": tool_belt.get_nv_sig_units(cxn),
                'pi_pulse_reps': pi_pulse_reps,
                "do_dq": do_dq,
                "do_scc": do_scc,
                "uwave_freq": uwave_freq,
                "uwave_freq-units": "GHz",
                "uwave_power": uwave_power,
                "uwave_power-units": "dBm",
                "uwave_pi_pulse": uwave_pi_pulse,
                "uwave_pi_pulse-units": "ns",
                "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
                "uwave_pi_on_2_pulse-units": "ns",
                "precession_time_range": precession_time_range,
                "precession_time_range-units": "ns",
                "state": state.name,
                "num_steps": num_steps,
                "num_reps": num_reps,
                "run_ind": run_ind,	
                "taus": taus.tolist(),	
                "plot_taus":plot_taus.tolist(),	
                "taus-units": "ns",
                "tau_index_master_list": tau_index_master_list,
                "opti_coords_list": opti_coords_list,
                "opti_coords_list-units": "V",
                "sig_counts": sig_counts.astype(int).tolist(),
                "sig_counts-units": "counts",
                "ref_counts": ref_counts.astype(int).tolist(),
                "ref_counts-units": "counts",
                "norm_avg_sig_ste": norm_avg_sig_ste.tolist()
            }
    
            # This will continuously be the same file path so we will overwrite
            # the existing file with the latest version
            file_path = tool_belt.get_file_path(
                __file__, start_timestamp, nv_sig["name"], "incremental"
            )
            tool_belt.save_raw_data(raw_data, file_path)
            if do_plot:
                tool_belt.save_figure(raw_fig, file_path)

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Plot the data

    ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, norm_style)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals
    
    if do_plot:
        ax = axes_pack[0]
        ax.cla()
        ax.plot(plot_taus, sig_counts_avg_kcps, "r-", label="signal")
        ax.plot(plot_taus, ref_counts_avg_kcps, "g-", label="reference")
        ax.set_xlabel(r"Precession time, $T = 2 N \tau (\mathrm{\mu s}$)")
        ax.set_ylabel("kcps")
        ax.legend()
    
        ax = axes_pack[1]
        ax.cla()
        #ax.plot(plot_taus, norm_avg_sig, "b-")
        kpl.plot_points(ax, plot_taus, norm_avg_sig, yerr=norm_avg_sig_ste, color = KplColors.BLUE)
        ax.set_xlabel(r"Precession time, $T = 2 N \tau (\mathrm{\mu s}$)")
        ax.set_ylabel("Contrast (arb. units)")
    
        if do_dq:
            dq_text = 'DQ'
        else:
            dq_text = 'SQ'
        if do_scc:
            ax.set_title("CPMG-{} {} SCC Measurement".format(pi_pulse_reps, dq_text))
        else:
            ax.set_title("CPMG-{} {} Measurement".format(pi_pulse_reps, dq_text))
                
        raw_fig.canvas.draw()
        raw_fig.set_tight_layout(True)
        raw_fig.canvas.flush_events()

    # %% Save the data

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    if do_save:
        raw_data = {
            "timestamp": timestamp,
            "timeElapsed": timeElapsed,
            "nv_sig": nv_sig,
            "nv_sig-units": tool_belt.get_nv_sig_units(cxn),
            'pi_pulse_reps': pi_pulse_reps,
            "do_dq": do_dq,
            "do_scc": do_scc,
            'comp_wait_time': comp_wait_time,
            "uwave_freq": uwave_freq,
            "uwave_freq-units": "GHz",
            "uwave_power": uwave_power,
            "uwave_power-units": "dBm",
            "uwave_pi_pulse": uwave_pi_pulse,
            "uwave_pi_pulse-units": "ns",
            "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
            "uwave_pi_on_2_pulse-units": "ns",
            "precession_time_range": precession_time_range,
            "precession_time_range-units": "ns",
            "state": state.name,
            "num_steps": num_steps,
            "num_reps": num_reps,
            "num_runs": num_runs,	
            "taus": taus.tolist(),	
            "plot_taus":plot_taus.tolist(),	
            "taus-units": "ns",
            "tau_index_master_list": tau_index_master_list,
            "opti_coords_list": opti_coords_list,
            "opti_coords_list-units": "V",
            "sig_counts": sig_counts.astype(int).tolist(),
            "sig_counts-units": "counts",
            "ref_counts": ref_counts.astype(int).tolist(),
            "ref_counts-units": "counts",
            "norm_avg_sig": norm_avg_sig.astype(float).tolist(),
            "norm_avg_sig-units": "arb",
            "norm_avg_sig_ste": norm_avg_sig_ste.tolist()
        }
    
        nv_name = nv_sig["name"]
        file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
        tool_belt.save_figure(raw_fig, file_path)
        if do_plot:
            tool_belt.save_raw_data(raw_data, file_path)

    # Fit and save figs


    return sig_counts, ref_counts


    
# %% Run the file


if __name__ == "__main__":
    kpl.init_kplotlib()

    folder = 'pc_rabi/branch_master/dynamical_decoupling_cpmg/2022_12'   
    
    file1 = '2022_11_14-11_02_50-siena-nv1_2022_10_27'
    file2 = '2022_11_14-11_02_59-siena-nv1_2022_10_27'
    file4 = '2022_11_14-11_03_05-siena-nv1_2022_10_27'
    file8 = '2022_11_14-11_00_01-siena-nv1_2022_10_27'
    file16 = '2022_12_19-15_48_05-siena-nv1_2022_10_27'
    
    folder_relaxation = 'pc_rabi/branch_master/t1_dq_main/2022_11'
    file_t1 = '2022_11_22-08_15_49-siena-nv1_2022_10_27'
    
    data = tool_belt.get_raw_data(file16, folder)
    # fit_t2_decay(data)
    
    # file_list = [file16_1, file16_2, file16_3]
    # folder_list = [folder, folder, folder]
    # tool_belt.save_combine_data(file_list, folder_list, 'dynamical_decoupling_cpmg.py')
    
    # file_list = [
    #             file1, 
    #               file2, 
    #               file4, 
    #               file8, 
    #                 file16, 
    #               file_t1
    #              ]
    # color_list = ['red', 
    #                'blue', 
    #               'orange', 
    #                 'green',
    #                 'purple', 
    #                'black'
    #               ]
    
    
    # if True:
    # # if False:
    #     fig, ax = plt.subplots(figsize=(8.5, 8.5))
    #     # amplitude = 0.069
    #     # offset = 0.931
    #     for f in range(len(file_list)):
    #         file = file_list[f]
             
    #         # if f == 10:
    #         #     w = 1
    #         if f == len(file_list)-1: 
    #             data = tool_belt.get_raw_data(file, folder_relaxation)  
    #             relaxation_time_range = data['relaxation_time_range']
    #             min_relaxation_time = int(relaxation_time_range[0])
    #             max_relaxation_time = int(relaxation_time_range[1])
    #             num_steps = data['num_steps']
    #             tau_T = numpy.linspace(
    #                 min_relaxation_time,
    #                 max_relaxation_time,
    #                 num=num_steps,
    #               )  
    #             tau_T_us = tau_T / 1000
    #             norm_avg_sig = data['norm_avg_sig']
    #             ax.plot([],[],"-o", color= color_list[f], label = "T1")
                
    #             A0 = 0.098
    #             amplitude = 2/3 * 2*A0
    #             offset = 1 - amplitude
                
    #             fit_func = lambda x, amp, decay: tool_belt.exp_decay(x, amp, decay, offset)
    #             init_params = [0.069, 5000]
                
    #             popt, pcov = curve_fit(
    #                 fit_func,
    #                 tau_T_us,
    #                 norm_avg_sig,
    #                 # sigma=norm_avg_sig_ste,
    #                 # absolute_sigma=True,
    #                 p0=init_params,
    #             )
    #             print(popt)
    #             print(numpy.sqrt(numpy.diag(pcov)))
                
    #         else:  
    #             data = tool_belt.get_raw_data(file, folder)  
    #             popt, fit_func = fit_t2_decay(data, do_plot= False)
            
    #             taus = numpy.array(data['taus'])
    #             num_steps = data['num_steps']
    #             norm_avg_sig = data['norm_avg_sig']
    #             pi_pulse_reps = data['pi_pulse_reps']
            
    #             tau_T = 2*taus*pi_pulse_reps
                   
               
    #             # for legend
    #             ax.plot([],[],"-o", color= color_list[f], label = "CPMG-{}".format(pi_pulse_reps))
            
    #         # linspace_T = numpy.linspace(
    #         #     tau_T[0], tau_T[-1], num=1000
    #         linspace_T = numpy.linspace(
    #                 tau_T[0], tau_T[-1], num=1000
    #         )
    #         ax.plot(tau_T / 1000, norm_avg_sig, "o", color= color_list[f])
    #         # ax.errorbar(taus, norm_avg_sig, yerr=norm_avg_sig_ste,\
    #         #             fmt='bo', label='data')
    #         ax.plot(
    #             linspace_T / 1000,
    #             fit_func(linspace_T/1000, *popt),
    #             "-", color= color_list[f]
    #         )
            
    #     ax.set_xlabel(r"$T = 2 \tau$ ($\mathrm{\mu s}$)")
    #     ax.set_ylabel("Contrast (arb. units)")
    #     ax.set_title("CPMG-N")
    #     ax.legend()
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
  
    
    # file_name_sq = "2023_03_24-23_02_52-siena-nv0_2023_03_20" #SQ
    # file_name_dq = "2023_03_26-14_36_28-siena-nv0_2023_03_20" #DQ
    
    # file_list = [file_name_sq, file_name_dq]
    # label_list=['SQ', 'DQ']
    # fit_fig, ax = plt.subplots()
    # for f in range(len(file_list)):
    #     file = file_list[f]
    #     data = tool_belt.get_raw_data(file)
    #     norm_avg_sig = data['norm_avg_sig']
    #     norm_avg_sig_ste = data['norm_avg_sig_ste']
    #     plot_taus = data['plot_taus']
    #     num_steps = data['num_steps']
    #     pi_pulse_reps = data['pi_pulse_reps']
    #     do_dq=data['do_dq']
    #     taus = numpy.array(data['taus'])
        
    #     tau_T = 2*taus*pi_pulse_reps/1000

    #     kpl.plot_points(ax, tau_T, norm_avg_sig, yerr=norm_avg_sig_ste,  label = label_list[f])
    #     ax.set_xlabel(r"$T = 2 \tau$ ($\mathrm{\mu s}$)")
    #     ax.set_ylabel("Contrast (arb. units)")
    #     ax.set_title("CPMG-{}".format(pi_pulse_reps))
    #     ax.legend()
        
    
    file_name = '2023_03_29-12_03_15-siena-nv0_2023_03_20'
    data = tool_belt.get_raw_data(file_name)
    fit_t2_12C(data, fixed_offset = 1.018)
    
    
    # file_list = ['2023_03_24-23_02_52-siena-nv0_2023_03_20', # SQ CPMG-2
    #               '2023_03_27-21_12_39-siena-nv0_2023_03_20'
    #     ]
    # file_list = ['2023_03_26-14_36_28-siena-nv0_2023_03_20', # DQ CPMG-2
    #               '2023_03_27-18_14_10-siena-nv0_2023_03_20'
    #         ]
    # file_list = ['2023_03_27-02_35_40-siena-nv0_2023_03_20', # DQ CPMG-4
    #               '2023_03_28-03_47_24-siena-nv0_2023_03_20'
    #                 ]
    file_list = ['2023_03_27-05_34_19-siena-nv0_2023_03_20', # DQ CPMG-8
                  '2023_03_28-09_44_49-siena-nv0_2023_03_20'
                    ]
    
    # compile_12C_data(file_list, 
    #                   do_save = True
    #                     )
    

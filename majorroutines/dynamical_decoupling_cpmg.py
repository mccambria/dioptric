# -*- coding: utf-8 -*-
"""
Dynamical decoupling CPMG



Created on Fri Aug 5 2022

@author: agardill
"""

# %% Imports


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
    fit_func,
    popt,
):

    tau_T = 2*taus*pi_pulse_reps

    linspace_taus = numpy.linspace(
        tau_T[0], tau_T[-1], num=1000
    )

    fit_fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fit_fig.set_tight_layout(True)
    ax.plot(tau_T / 1000, norm_avg_sig, "bo", label="data")
    # ax.errorbar(taus, norm_avg_sig, yerr=norm_avg_sig_ste,\
    #             fmt='bo', label='data')
    ax.plot(
        linspace_taus / 1000,
        fit_func(linspace_taus/1000, *popt),
        "r-",
        label="fit",
    )
    ax.set_ylabel("Contrast (arb. units)")
    ax.set_title("CPMG-{}".format(pi_pulse_reps))
    ax.legend()
    
    ax.set_xlabel(r"$T = 2 \tau$ ($\mathrm{\mu s}$)")
    t2_time = popt[-1]
    text_popt = "\n".join(
        (
            r"$S(T) = exp(- (T/T_2)^3)$",
            r"$T_2=$%.3f us" % (t2_time),
        )
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.70,
        0.85,
        text_popt,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

     



    fit_fig.canvas.draw()
    fit_fig.set_tight_layout(True)
    fit_fig.canvas.flush_events()

    return fit_fig

def fit_t2_decay(data, do_plot = True):
    '''
    either pass in only the revivals, or for isotopically pure samples, this will fit t2
    '''
    
    precession_dur_range = data["precession_time_range"]
    sig_counts = data["sig_counts"]
    ref_counts = data["ref_counts"]
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    pi_pulse_reps = data['pi_pulse_reps']
    taus = numpy.array(data['taus'])
    
    tau_T = 2*taus*pi_pulse_reps


    #  Normalization and uncertainty

    avg_sig_counts = numpy.average(sig_counts[::], axis=0)
    ste_sig_counts = numpy.std(sig_counts[::], axis=0, ddof=1) / numpy.sqrt(
        num_runs
    )

    # Assume reference is constant and can be approximated to one value
    avg_ref = numpy.average(ref_counts[::])

    # Divide signal by reference to get normalized counts and st error
    norm_avg_sig = avg_sig_counts / avg_ref
    # print(list(norm_avg_sig))
    norm_avg_sig_ste = ste_sig_counts / avg_ref

    # Hard guess
    A0 = 0.098
    offset = 0.902
    amplitude = 2/3 * 2*A0
    offset = 1 - amplitude
    decay_time = 6e6
    # dominant_freqs = [1 / (1000*revival_time)]

    #  Fit

    # The fit doesn't like dealing with vary large numbers. We'll convert to
    # us here and then convert back to ns after the fit for consistency.

    tau_T_us = tau_T / 1000
    decay_time_us = decay_time / 1000

    init_params = [
        amplitude,
        # offset,
        decay_time_us,
    ]
    
    fit_func = tool_belt.t2_func
    fit_func = lambda t, amplitude, t2: tool_belt.t2_func(t, amplitude, offset, t2)
    popt, pcov = curve_fit(
        fit_func,
        tau_T_us,
        norm_avg_sig,
        sigma=norm_avg_sig_ste,
        absolute_sigma=True,
        p0=init_params,
    )
    print(popt)
    print(numpy.sqrt(numpy.diag(pcov)))
    if do_plot:
        create_fit_figure(
            taus,
            num_steps,
            norm_avg_sig,
            norm_avg_sig_ste,
            pi_pulse_reps,
            fit_func,
        popt,)
    
    return popt, fit_func



# %% Main


def main(
    nv_sig,
    apd_indices,
    precession_dur_range,
    pi_pulse_reps,
    num_steps,
    num_reps,
    num_runs,
    taus=[],
    state=States.LOW,
):

    with labrad.connect() as cxn:
        angle = main_with_cxn(
            cxn,
            nv_sig,
            apd_indices,
            precession_dur_range,
            pi_pulse_reps,
            num_steps,
            num_reps,
            num_runs,
            taus,
            state,
        )
        return angle


def main_with_cxn(
    cxn,
    nv_sig,
    apd_indices,
    precession_time_range,
    pi_pulse_reps,
    num_steps,
    num_reps,
    num_runs,
    taus=[],
    state=States.LOW,
):

    tool_belt.reset_cfm(cxn)

    # %% Sequence setup

    laser_key = "spin_laser"
    laser_name = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    polarization_time = nv_sig["spin_pol_dur"]
    gate_time = nv_sig["spin_readout_dur"]

    rabi_period = nv_sig["rabi_{}".format(state.name)]
    uwave_freq = nv_sig["resonance_{}".format(state.name)]
    uwave_power = nv_sig["uwave_power_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    seq_file_name = "dynamical_decoupling.py"

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

    seq_args = [
          taus[0],
          polarization_time,
          gate_time,
          uwave_pi_pulse,
          uwave_pi_on_2_pulse,
          taus[-1],
          pi_pulse_reps,
          apd_indices[0],
          state.value,
          laser_name,
          laser_power,
      ]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(seq_file_name, seq_args_string)
    seq_time = ret_vals[0]
    # print(seq_args)
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
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
        opti_coords_list.append(opti_coords)

        # Set up the microwaves
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_freq(uwave_freq)
        sig_gen_cxn.set_amp(uwave_power)
        sig_gen_cxn.load_iq()
        sig_gen_cxn.uwave_on()
        
        cxn.arbitrary_waveform_generator.load_cpmg(pi_pulse_reps)
        

        # Set up the laser
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)

        # Shuffle the list of tau indices so that it steps thru them randomly
        shuffle(tau_ind_list)

        for tau_ind in tau_ind_list:

            # 'Flip a coin' to determine which tau (long/shrt) is used first
            rand_boolean = numpy.random.randint(0, high=2)

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

            seq_args = [
                  taus[tau_ind_first],
                  polarization_time,
                  gate_time,
                  uwave_pi_pulse,
                  uwave_pi_on_2_pulse,
                  taus[tau_ind_second],
                  pi_pulse_reps,
                  apd_indices[0],
                  state.value,
                  laser_name,
                  laser_power,
              ]
            # print(seq_args)
            # return
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # Clear the tagger buffer of any excess counts
            # cxn.apd_tagger.clear_buffer()
            cxn.pulse_streamer.stream_immediate(
                seq_file_name, num_reps, seq_args_string
            )

            # Each sample is of the form [*(<sig_shrt>, <ref_shrt>, <sig_long>, <ref_long>)]
            # So we can sum on the values for similar index modulus 4 to
            # parse the returned list into what we want.
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
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

        cxn.apd_tagger.stop_tag_stream()

        # %% incremental plotting
        
        #Average the counts over the iterations
        avg_sig_counts = numpy.average(sig_counts[:(run_ind+1)], axis=0)
        avg_ref_counts = numpy.average(ref_counts[:(run_ind+1)], axis=0)
        try:
            norm_avg_sig = avg_sig_counts / numpy.average(avg_ref_counts)
        except RuntimeWarning as e:
            print(e)
            inf_mask = numpy.isinf(norm_avg_sig)
            # Assign to 0 based on the passed conditional array
            norm_avg_sig[inf_mask] = 0
        
        
        ax = axes_pack[0]
        ax.cla()
        ax.plot(plot_taus, avg_sig_counts, "r-", label="signal")
        ax.plot(plot_taus, avg_ref_counts, "g-", label="reference")
        ax.set_xlabel(r"Precession time, $T = 2 N \tau (\mathrm{\mu s}$)")
        ax.set_ylabel("Counts")
        ax.legend()
        
        ax = axes_pack[1]
        ax.cla()
        ax.plot(plot_taus, norm_avg_sig, "b-")
        ax.set_title("CPMG-{} Measurement".format(pi_pulse_reps))
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

        raw_data = {
            "start_timestamp": start_timestamp,
            "nv_sig": nv_sig,
            "nv_sig-units": tool_belt.get_nv_sig_units(),
            'pi_pulse_reps': pi_pulse_reps,
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
        }

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(
            __file__, start_timestamp, nv_sig["name"], "incremental"
        )
        tool_belt.save_raw_data(raw_data, file_path)
        tool_belt.save_figure(raw_fig, file_path)

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Plot the data

    ax = axes_pack[0]
    ax.cla()
    ax.plot(plot_taus, avg_sig_counts, "r-", label="signal")
    ax.plot(plot_taus, avg_ref_counts, "g-", label="reference")
    ax.set_xlabel(r"Precession time, $T = 2 N \tau (\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")
    ax.legend()

    ax = axes_pack[1]
    ax.cla()
    ax.plot(plot_taus, norm_avg_sig, "b-")
    ax.set_title("CPMG -{} Measurement".format(pi_pulse_reps))
    ax.set_xlabel(r"Precession time, $T = 2 N \tau (\mathrm{\mu s}$)")
    ax.set_ylabel("Contrast (arb. units)")

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
        'pi_pulse_reps': pi_pulse_reps,
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
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(raw_fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)

    # %% Fit and save figs


    return 


    
# %% Run the file


if __name__ == "__main__":

    folder = 'pc_rabi/branch_master/dynamical_decoupling_cpmg/2022_11'   
    
    file1 = '2022_11_14-11_02_50-siena-nv1_2022_10_27'
    file2 = '2022_11_14-11_02_59-siena-nv1_2022_10_27'
    file4 = '2022_11_14-11_03_05-siena-nv1_2022_10_27'
    file8 = '2022_11_14-11_00_01-siena-nv1_2022_10_27'
    file16 = '2022_11_14-11_03_13-siena-nv1_2022_10_27'
    
    folder_relaxation = 'pc_rabi/branch_master/t1_dq_main/2022_11'
    file_t1 = '2022_11_12-22_17_47-siena-nv1_2022_10_27'
    
    # data = tool_belt.get_raw_data(file16, folder)
    # fit_t2_decay(data)
    
    # file_list = [file16_1, file16_2, file16_3]
    # folder_list = [folder, folder, folder]
    # tool_belt.save_combine_data(file_list, folder_list, 'dynamical_decoupling_cpmg.py')
    
    file_list = [
                file1, 
                  file2, 
                  file4, 
                  file8, 
                    file16, 
                  file_t1
                 ]
    color_list = ['red', 
                   'blue', 
                  'orange', 
                    'green',
                    'purple', 
                   'black'
                  ]
    
    
    if True:
    # if False:
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        # amplitude = 0.069
        # offset = 0.931
        for f in range(len(file_list)):
            file = file_list[f]
             
            # if f == 10:
            #     w = 1
            if f == len(file_list)-1: 
                data = tool_belt.get_raw_data(file, folder_relaxation)  
                relaxation_time_range = data['relaxation_time_range']
                min_relaxation_time = int(relaxation_time_range[0])
                max_relaxation_time = int(relaxation_time_range[1])
                num_steps = data['num_steps']
                tau_T = numpy.linspace(
                    min_relaxation_time,
                    max_relaxation_time,
                    num=num_steps,
                  )  
                tau_T_us = tau_T / 1000
                norm_avg_sig = data['norm_avg_sig']
                ax.plot([],[],"-o", color= color_list[f], label = "T1")
                
                A0 = 0.098
                amplitude = 2/3 * 2*A0
                offset = 1 - amplitude
                
                fit_func = lambda x, amp, decay: tool_belt.exp_decay(x, amp, decay, offset)
                init_params = [0.069, 5000]
                
                popt, pcov = curve_fit(
                    fit_func,
                    tau_T_us,
                    norm_avg_sig,
                    # sigma=norm_avg_sig_ste,
                    # absolute_sigma=True,
                    p0=init_params,
                )
                print(popt)
                print(numpy.sqrt(numpy.diag(pcov)))
                
            else:  
                data = tool_belt.get_raw_data(file, folder)  
                popt, fit_func = fit_t2_decay(data, do_plot= False)
            
                taus = numpy.array(data['taus'])
                num_steps = data['num_steps']
                norm_avg_sig = data['norm_avg_sig']
                pi_pulse_reps = data['pi_pulse_reps']
            
                tau_T = 2*taus*pi_pulse_reps
                   
               
                # for legend
                ax.plot([],[],"-o", color= color_list[f], label = "CPMG-{}".format(pi_pulse_reps))
            
            # linspace_T = numpy.linspace(
            #     tau_T[0], tau_T[-1], num=1000
            linspace_T = numpy.linspace(
                    tau_T[0], tau_T[-1], num=1000
            )
            ax.plot(tau_T / 1000, norm_avg_sig, "o", color= color_list[f])
            # ax.errorbar(taus, norm_avg_sig, yerr=norm_avg_sig_ste,\
            #             fmt='bo', label='data')
            ax.plot(
                linspace_T / 1000,
                fit_func(linspace_T/1000, *popt),
                "-", color= color_list[f]
            )
            
        ax.set_xlabel(r"$T = 2 \tau$ ($\mathrm{\mu s}$)")
        ax.set_ylabel("Contrast (arb. units)")
        ax.set_title("CPMG-N")
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        
    
    
    
    

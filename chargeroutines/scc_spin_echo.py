# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:52:28 2021

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import matplotlib.pyplot as plt
import time
import labrad
import majorroutines.spin_echo as spin_echo
from utils.tool_belt import States
from random import shuffle
from scipy.optimize import curve_fit
 



def quartic(tau, offset, revival_time, decay_time, amplitudes):
    tally = offset
    exp_part = numpy.exp(-(((tau - revival_time) / decay_time) ** 4))
    tally += amplitudes * exp_part
    return tally


def t2(tau,  a, C, decay_time):
    exp_part = numpy.exp(-((tau / decay_time) ** 4))
    sin_part = 1 - C * numpy.sin(a * tau)**2 
    return  exp_part * sin_part

def combine_revivals(file_list, folder):
    
    norm_counts_tot = []
    taus_tot = []
    for file in file_list:
        data = tool_belt.get_raw_data(file, folder)
        taus = data['taus']
        norm_avg_sig = data['norm_avg_sig']
        norm_counts_tot = norm_counts_tot + norm_avg_sig
        taus_tot = taus_tot + taus
    nv_sig = data['nv_sig']
    uwave_pi_on_2_pulse = data['uwave_pi_on_2_pulse']
    uwave_pi_pulse = data['uwave_pi_pulse']
    state = data['state']
    num_reps = data['num_reps']
    num_runs = ['num_runs']
    
    timestamp = tool_belt.get_time_stamp()
    rawData = {'timestamp': timestamp,
        'nv_sig': nv_sig,
        'nv_sig-units': tool_belt.get_nv_sig_units(),
        "uwave_pi_pulse": uwave_pi_pulse,
        "uwave_pi_pulse-units": "ns",
        "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
        "uwave_pi_on_2_pulse-units": "ns",
        'state': state,
        'num_reps': num_reps,
        'num_runs': num_runs,
        'taus': taus_tot,
        "norm_avg_sig": norm_counts_tot,
        "norm_avg_sig-units": "arb",
        }
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(taus_tot, norm_counts_tot, 'bo')
    ax.set_xlabel('Taus (us)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend(loc='lower right')
    
    
    name = nv_sig['name']
    filePath = tool_belt.get_file_path(__file__, timestamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
            
    return

# %% Main


def main(nv_sig, opti_nv_sig, apd_indices, precession_time_range,
         num_steps, num_reps, num_runs,  
         state=States.LOW):

    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig,opti_nv_sig, apd_indices, precession_time_range,
                  num_steps, num_reps, num_runs, 
                  state)
    return 


def main_with_cxn(cxn, nv_sig, opti_nv_sig,apd_indices, precession_time_range,
              num_steps, num_reps, num_runs,  
              state=States.LOW):

    #  Initial calculations and setup

    tool_belt.reset_cfm(cxn)
    seq_file_name = 'scc_spin_echo.py'

    # Create the array of relaxation times

    # Array of times to sweep through
    # Must be ints
    min_precession_time = int(precession_time_range[0])
    max_precession_time = int(precession_time_range[1])

    taus = numpy.linspace(
        min_precession_time,
        max_precession_time,
        num=num_steps,
        dtype=numpy.int32,
    )
    # print(taus)
    #return
    
    # Fix the length of the sequence to account for odd amount of elements

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
    
    
    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    ref_counts = numpy.empty([num_runs, num_steps])
    ref_counts[:] = numpy.nan
    sig_counts = numpy.copy(ref_counts)
    
    drift_list = []
    tau_index_master_list = [[] for i in range(num_runs)]
    
    
    # Set the charge readout (assumed to be yellow here) to the correct filter
    if 'charge_readout_laser_filter' in nv_sig:
        tool_belt.set_filter(cxn, nv_sig, 'charge_readout_laser')
    
    
    
    opti_interval = 4 # min
    
    nv_coords = nv_sig['coords']
    rabi_period = nv_sig["rabi_{}".format(state.name)]
    uwave_freq = nv_sig["resonance_{}".format(state.name)]
    uwave_power = nv_sig["uwave_power_{}".format(state.name)]
    
    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)


    readout_time = nv_sig['charge_readout_dur']
    readout_power = nv_sig['charge_readout_laser_power']
    ionization_time = nv_sig['nv0_ionization_dur']
    reionization_time = nv_sig['nv-_reionization_dur']
    shelf_time = nv_sig['spin_shelf_dur']
    shelf_power = nv_sig['spin_shelf_laser_power']
    
    
    green_laser_name = nv_sig['nv-_reionization_laser']
    red_laser_name = nv_sig['nv0_ionization_laser']
    yellow_laser_name = nv_sig['charge_readout_laser']
    sig_gen_name = tool_belt.get_signal_generator_name_no_cxn(state)    
            
            
    seq_args = [readout_time, reionization_time, ionization_time, shelf_time ,
          min_precession_time, max_precession_time, uwave_pi_pulse, uwave_pi_on_2_pulse, 
          green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name,
        apd_indices[0], readout_power, shelf_power]
    
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(seq_file_name, seq_args_string)
        
    period = ret_vals[0]
    
    period_s = period/10**9
    period_s_total = (period_s*num_reps*num_steps*num_runs/2 + 1)
    period_m_total = period_s_total/60
    print('Expected time: {:.1f} m'.format(period_m_total))
    # return

    # %% Get the starting time of the function

    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()
    
    start_time = time.time()
    start_function_time = start_time
    
    
    for run_ind in range(num_runs):
        print('Run index: {}'. format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize and save the coords we found
        optimize.main_with_cxn(cxn, opti_nv_sig, apd_indices)
        drift = tool_belt.get_drift()
        drift_list.append(drift)
        
        adjusted_nv_coords = numpy.array(nv_coords) + drift
        
        
        # Set up the microwaves and laser. Then load the pulse streamer 
        # (must happen after optimize and iq_switch since run their
        # own sequences)
        # Set up the microwaves
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_freq(uwave_freq)
        sig_gen_cxn.set_amp(uwave_power)
        sig_gen_cxn.uwave_on()

        

        # Shuffle the taus we step thru
        shuffle(tau_ind_list)
        
        
        # Take a sample and increment the frequency
        for tau_ind in tau_ind_list:
             
            # Check if we should optimize
            time_current = time.time()
            if time_current - start_time > opti_interval * 60:
                optimize.main_with_cxn(cxn, opti_nv_sig, apd_indices)
                drift = tool_belt.get_drift()
                drift_list.append(drift)
                
                adjusted_nv_coords = numpy.array(nv_coords) + drift
                
                start_time = time_current
            tool_belt.set_xyz(cxn, adjusted_nv_coords)
            
            cxn.apd_tagger.start_tag_stream(apd_indices)
            
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
           
            seq_args = [readout_time, reionization_time, ionization_time, shelf_time ,
          taus[tau_ind_first], taus[tau_ind_second], uwave_pi_pulse, uwave_pi_on_2_pulse, 
          green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name,
          apd_indices[0], readout_power, shelf_power]
            
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # Clear the tagger buffer of any excess counts
            cxn.apd_tagger.clear_buffer()
            
            cxn.pulse_streamer.stream_immediate(
                seq_file_name, num_reps, seq_args_string
            )

        
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
            # print(new_counts[0])
            sample_counts = new_counts[0]
        
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

        # %% Save the data we have incrementally for long measurements

        rawData = {'start_timestamp': start_timestamp,
                   'nv_sig': nv_sig,
                   'nv_sig-units': tool_belt.get_nv_sig_units(),
                    "uwave_freq": uwave_freq,
                    "uwave_freq-units": "GHz",
                    "uwave_power": uwave_power,
                    "uwave_power-units": "dBm",
                    "rabi_period": rabi_period,
                    "rabi_period-units": "ns",
                    "uwave_pi_pulse": uwave_pi_pulse,
                    "uwave_pi_pulse-units": "ns",
                    "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
                    "uwave_pi_on_2_pulse-units": "ns",
                    "precession_time_range": precession_time_range,
                    "precession_time_range-units": "ns",
                   'state': state.name,
                   'num_steps': num_steps,
                   'run_ind': run_ind,
                   'taus': taus.tolist(),
                   'drift_list': drift_list,
                   'opti_interval': opti_interval,
                   'sig_counts': sig_counts.astype(int).tolist(),
                   'sig_counts-units': 'counts',
                   'ref_counts': ref_counts.astype(int).tolist(),
                   'ref_counts-units': 'counts'}

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(rawData, file_path)


    # %% Process and plot the data
    end_function_time = time.time()
    time_elapsed = end_function_time - start_function_time
    # print(time_elapsed)
    
    # %% Average the counts over the iterations

    avg_sig_counts = numpy.average(sig_counts, axis=0)
    avg_ref_counts = numpy.average(ref_counts, axis=0)

    # %% Calculate the ramsey data, signal / reference over different
    # relaxation times

    # Replace x/0=inf with 0
    try:
        norm_avg_sig = avg_sig_counts / avg_ref_counts
    except RuntimeWarning as e:
        print(e)
        inf_mask = numpy.isinf(norm_avg_sig)
        # Assign to 0 based on the passed conditional array
        norm_avg_sig[inf_mask] = 0


    # %% Plot the data

    raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

    ax = axes_pack[0]
    # Account for the pi/2 pulse on each side of a tau
    plot_taus = (taus + uwave_pi_pulse) / 1000
    ax.plot(plot_taus, avg_sig_counts, "r-", label="signal")
    ax.plot(plot_taus, avg_ref_counts, "g-", label="reference")
    ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")
    ax.legend()

    ax = axes_pack[1]
    ax.plot(plot_taus, norm_avg_sig, "b-")
    ax.set_title("Spin Echo Measurement")
    ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Contrast (arb. units)")

    raw_fig.canvas.draw()
    raw_fig.set_tight_layout(True)
    raw_fig.canvas.flush_events()
    
    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'time_elapsed': time_elapsed,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                "uwave_freq": uwave_freq,
                "uwave_freq-units": "GHz",
                "uwave_power": uwave_power,
                "uwave_power-units": "dBm",
                "rabi_period": rabi_period,
                "rabi_period-units": "ns",
                "uwave_pi_pulse": uwave_pi_pulse,
                "uwave_pi_pulse-units": "ns",
                "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
                "uwave_pi_on_2_pulse-units": "ns",
                "precession_time_range": precession_time_range,
                "precession_time_range-units": "ns",
                'state': state.name,
                'num_steps': num_steps,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'taus': taus.tolist(),
                'drift_list': drift_list,
                'opti_interval': opti_interval,
                'sig_counts': sig_counts.astype(int).tolist(),
                'sig_counts-units': 'counts',
                'ref_counts': ref_counts.astype(int).tolist(),
                'ref_counts-units': 'counts',
                "norm_avg_sig": norm_avg_sig.astype(float).tolist(),
                "norm_avg_sig-units": "arb",
                }

        
    name = nv_sig['name']
    filePath = tool_belt.get_file_path(__file__, timestamp, name)
    tool_belt.save_figure(raw_fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)

    return 

    

# %%

if __name__ == '__main__':

    apd_indices = [0]
    sample_name = 'johnson'    
    
    green_laser = "cobolt_515"
    yellow_laser = 'laserglow_589'
    red_laser = 'cobolt_638'
    
    green_power = 7
    red_power = 120
    nd_yellow = "nd_0.5"
    
    opti_nv_sig = {
        "coords": [-0.037, 0.273, 4.85],
        "name": "{}-nv0_2021_09_23".format(sample_name,),
        "disable_opt": False,
        "expected_count_rate": 30,
        "imaging_laser":green_laser,
        "imaging_laser_power": green_power,
        "imaging_readout_dur": 1e7,
        "collection_filter": "630_lp",
        "magnet_angle": None,
    }  # 14.5 max
    
    
    
    nv_sig = {
        "coords":[0.16108189, 0.13252713, 4.79 ],
        
        "name": "{}-dnv5_2021_09_23".format(sample_name,),
        "disable_opt": False,
        "expected_count_rate": 75,
            'imaging_laser': green_laser, 'imaging_laser_power': green_power,
            'imaging_readout_dur': 1E7,
            
            'nv-_reionization_laser': green_laser, 'nv-_reionization_laser_power': green_power, 
            'nv-_reionization_dur': 1E5,
        
            'nv0_ionization_laser': red_laser, 'nv0_ionization_laser_power': red_power,
            'nv0_ionization_dur':300,
            
            'spin_shelf_laser': yellow_laser, 'spin_shelf_laser_filter': nd_yellow, 
            'spin_shelf_laser_power': 0.4, 'spin_shelf_dur':0,
            
            'charge_readout_laser': yellow_laser, 'charge_readout_laser_filter': nd_yellow, 
            'charge_readout_laser_power': 0.3, 'charge_readout_dur':0.5e6,
            
            'collection_filter': '630_lp', 'magnet_angle': 114,
            
        "resonance_LOW":2.7926,"rabi_LOW": 143.1,"uwave_power_LOW": 15.5, 
        "resonance_HIGH": 2.9496,"rabi_HIGH": 215,"uwave_power_HIGH": 14.5} 
    
    
    
    
    # max_time = 200  # us
    # num_steps = int(max_time + 1)  # 1 point per us
    # precession_time_range = [0, max_time * 10 ** 3]
    
    start = 0
    stop = 190
    num_steps = int(stop - start + 1)  # 1 point per us
    precession_time_range = [start *1e3, stop *1e3]
    
    num_reps = int(1e3)
    num_runs = 30 #60
    
    do_plot = True
    
    try:
        if not do_plot:
            main(nv_sig, opti_nv_sig, apd_indices,precession_time_range,
                  num_steps, num_reps, num_runs)
            
            # for i in []:
            #     t = (i ) * 30.5
            #     start = t-10
            #     if t < 10:
            #         start = 0
            #     stop = t+10
            #     num_steps = int((stop - start)*2 + 1)  # 2 point per us
            #     precession_time_range = [start *1e3, stop *1e3]
                
            #     main(nv_sig, opti_nv_sig, apd_indices,precession_time_range,
            #       num_steps, num_reps, num_runs)
                
          # %%  
        else:
            # folder = 'pc_rabi/branch_master/scc_spin_echo/2021_10'
            folder = 'pc_rabi/branch_master/spin_echo/2021_10'
            # ++++ COMPARE +++++
            # file = '2021_10_06-10_46_34-johnson-dnv5_2021_09_23'
            file = '2021_10_02-11_27_24-johnson-dnv5_2021_09_23'
                
            fig, ax = plt.subplots(figsize=(8.5, 8.5))
                
            data = tool_belt.get_raw_data(file, folder)
            # taus = data['taus']
            
            precession_time_range = data['precession_time_range']
            num_steps = data['num_steps']
            min_precession_time = int(precession_time_range[0])
            max_precession_time = int(precession_time_range[1])
        
            taus = numpy.linspace(
                min_precession_time,
                max_precession_time,
                num=num_steps,
                                )
            
            norm_avg_sig = numpy.array(data['norm_avg_sig'])
            uwave_pi_pulse = data['uwave_pi_pulse']
                
            plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
            
            lin_taus = numpy.linspace(plot_taus[0], plot_taus[-1], 500)
            
            fit_func = t2
            
            init_params = [1/60*2*numpy.pi, 0.2, 1500]
            
            popt, pcov = curve_fit(
                fit_func,
                plot_taus,
                norm_avg_sig ,
                p0=init_params,
            )
    
            print(popt)
            ax.plot(lin_taus, fit_func(lin_taus, *popt), 'r-')
            
            # text_eq = r"O + A * e$^{((\tau - \tau_r) / d)^4}$"
            
            # text_popt = "\n".join(
            #     (
            #         r"O=%.3f (contrast)" % (popt[0]),
            #         r"$\tau_r$=%.3f $\mathrm{\mu s}$" % (popt[1]),
            #         r"d=%.3f $\mathrm{\mu s}$" % (popt[2]),
            #         r"A=%.3f (contrast)" % (popt[3]),
            #     )
            # )
        
            # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            # ax.text(
            #     0.05,
            #     0.15,
            #     text_popt,
            #     transform=ax.transAxes,
            #     fontsize=12,
            #     verticalalignment="top",
            #     bbox=props,
            # )
            # ax.text(
            #     0.7,
            #     0.9,
            #     text_eq,
            #     transform=ax.transAxes,
            #     fontsize=12,
            #     verticalalignment="top",
            #     bbox=props,
            # )
            
            ax.plot(plot_taus,norm_avg_sig , 'bo')
            ax.set_xlabel('Taus (us)')
            ax.set_ylabel('Contrast (arb. units)')
            ax.legend(loc='lower right')
            
            
            
            # #++++++++++++++++++++ Combine individual files
            # file_list = ['2021_10_04-20_03_19-johnson-dnv5_2021_09_23',
            #               '2021_10_04-20_36_53-johnson-dnv5_2021_09_23',
            #               '2021_10_04-21_13_01-johnson-dnv5_2021_09_23',
            #               '2021_10_04-21_51_48-johnson-dnv5_2021_09_23',
            #               '2021_10_04-22_50_40-johnson-dnv5_2021_09_23',
            #               '2021_10_04-23_34_51-johnson-dnv5_2021_09_23',
            #               '2021_10_05-00_21_25-johnson-dnv5_2021_09_23'
            #               ]
            # combine_revivals(file_list, folder)
         
    
    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        # tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()

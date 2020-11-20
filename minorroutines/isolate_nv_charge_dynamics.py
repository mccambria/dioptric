# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:12:42 2020

@author: gardill
"""

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import time
import matplotlib.pyplot as plt
import labrad
import majorroutines.image_sample as image_sample
import copy
import scipy.stats as stats
import minorroutines.time_resolved_readout as time_resolved_readout
# %%

def red_scan(nv_sig, apd_indices, scan_range_ratio):
    image_sample.main(nv_sig,  0.1*scan_range_ratio, 0.1*scan_range_ratio,
                      int(20*scan_range_ratio), apd_indices, 638,
                      save_data=False, plot_data=False, readout =10**3)
    return


def green_scan(nv_sig, apd_indices,scan_range_ratio):
    image_sample.main(nv_sig,  0.1*scan_range_ratio, 0.1*scan_range_ratio,
                      int(20*scan_range_ratio), apd_indices, 532,
                      save_data=False, plot_data=False, readout =10**4)
    return

def green_pulse(nv_coords):

    drift =tool_belt.get_drift()
    adj_coords = numpy.array(nv_coords) + numpy.array(drift)
    with labrad.connection() as cxn:
            tool_belt.set_xyz(cxn, adj_coords)
            cxn.pulse_streamer.constant([3],0,0)
            time.sleep(5)
            cxn.pulse_streamer.constant([],0,0)
    return

def main(target_sig, readout_sig, target_color, readout_color, apd_indices):

    with labrad.connect() as cxn:
        counts = main_with_cxn(cxn, target_sig, readout_sig, target_color, readout_color, apd_indices)

    return counts
def main_with_cxn(cxn, target_sig, readout_sig, target_color, readout_color, apd_indices):
    apd_index= apd_indices[0]
    readout_file_name = 'simple_readout.py'
    target_file_name = 'simple_pulse.py'
    #delay of aoms and laser
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    pulser_wiring_red = wiring['do_638_laser']
    pulser_wiring_green = wiring['do_532_aom']

    drift = tool_belt.get_drift()
    target_coords = target_sig['coords']
    target_coords_drift = numpy.array(target_coords) + numpy.array(drift)
    readout_coords = readout_sig['coords']
    readout_coords_drift = numpy.array(readout_coords) + numpy.array(drift)

    # Short green pulse on readout NV before readout

    # move the galvo to the readout NV
#    tool_belt.set_xyz(cxn, readout_coords_drift)
#    time.sleep(0.01)
#    # Pusle the green laser on NV_readout
#    seq_args = [laser_515_delay, 10**3, 0.0, 532]
#    seq_args_string = tool_belt.encode_seq_args(seq_args)
#    cxn.pulse_streamer.stream_immediate(target_file_name, 1, seq_args_string)

    # Target
    tool_belt.set_xyz(cxn,target_coords_drift)
    time.sleep(0.01)
    #If we are pulsing an initial laser:
    if target_color:
        # Pusle the laser
        if target_color == 532:
            target_pulse_time = target_sig['pulsed_reionization_dur']
            laser_delay = laser_515_delay
            pulser_wiring_value = pulser_wiring_green
        elif target_color == 638:
            target_pulse_time = target_sig['pulsed_ionization_dur']
            laser_delay = laser_638_delay
            pulser_wiring_value = pulser_wiring_red
        # pulse the laser, different techniques based on length of pulse
        if target_pulse_time >= 10**9:
            cxn.pulse_streamer.constant([pulser_wiring_value], 0.0, 0.0)
            time.sleep(target_pulse_time/10**9)
            cxn.pulse_streamer.constant([], 0.0, 0.0)
        else:
            seq_args = [laser_delay, int(target_pulse_time), 0.0, target_color]
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            cxn.pulse_streamer.stream_immediate(target_file_name, 1, seq_args_string)
    # for the control, still wait the time we would spend pulsing the green laser
    elif not target_color:
        target_pulse_time = target_sig['pulsed_reionization_dur']
        if target_pulse_time >= 10**9:
            cxn.pulse_streamer.constant([], 0.0, 0.0)
            time.sleep(target_pulse_time/10**9)
            cxn.pulse_streamer.constant([], 0.0, 0.0)
        else:
            seq_args = [laser_515_delay + target_pulse_time, 0, 0.0, 589]
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            cxn.pulse_streamer.stream_immediate(target_file_name, 1, seq_args_string)


    # then readout with yellow
    aom_ao_589_pwr = readout_sig['am_589_power']
    nd_filter = readout_sig['nd_filter']
    readout_pulse_time = readout_sig['pulsed_SCC_readout_dur']
    laser_delay = aom_589_delay

    cxn.filter_slider_ell9k.set_filter(nd_filter)
   # move the galvo to the readout
    tool_belt.set_xyz(cxn, readout_coords_drift)
    time.sleep(0.01)

#    seq_args = [0, laser_515_delay,  laser_delay,10**3, int(readout_pulse_time),
#                aom_ao_589_pwr, apd_index,532,  readout_color]      #two pulse
    seq_args = [laser_delay, int(readout_pulse_time), aom_ao_589_pwr, apd_index, readout_color]    #simple readout
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(readout_file_name, seq_args_string)
    # collect the counts
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Clear the buffer
    cxn.apd_tagger.clear_buffer()
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(readout_file_name, 1, seq_args_string)

    new_counts = cxn.apd_tagger.read_counter_simple(1)
#    sample_counts = new_counts[0]
#    print(sample_counts)

    # signal counts are even - get every second element starting from 0
    sig_counts = new_counts[0]


    cxn.apd_tagger.stop_tag_stream()

    return sig_counts

def readout_list(readout_coords_list, parameter_sig, apd_indices, initial_pulse=None):

    with labrad.connect() as cxn:
        signal_counts_list = readout_list_with_cxn(cxn,readout_coords_list,
                                           parameter_sig, apd_indices, initial_pulse)

    return signal_counts_list
def readout_list_with_cxn(cxn, readout_coords_list, parameters_sig, apd_indices, initial_pulse=None):
    apd_index= apd_indices[0]
    readout_file_name = 'simple_readout.py'
    pulse_file_name = 'simple_pulse.py'

    signal_counts_list = []

    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    aom_589_delay = shared_params['589_aom_delay']
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = shared_params['large_angle_galvo_delay']

    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    pulser_wiring_red = wiring['do_638_laser']
    pulser_wiring_green = wiring['do_532_aom']
    aom_ao_589_pwr = parameters_sig['am_589_power']

    nd_filter = parameters_sig['nd_filter']
    readout_pulse_time = parameters_sig['pulsed_SCC_readout_dur']
    cxn.filter_slider_ell9k.set_filter(nd_filter)

    drift = tool_belt.get_drift()

    for coords in readout_coords_list:
        readout_coords_drift = numpy.array(coords) + numpy.array(drift)
       # move the galvo to the readout
        tool_belt.set_xyz(cxn, readout_coords_drift)

        # if we want to have an initial pulseo on the NV, pulse green or red
        if initial_pulse:
            # Short green pulse on readout NV before readout
            if initial_pulse == 532:
                pulse_time = parameters_sig['pulsed_reionization_dur']
                laser_delay = laser_515_delay
                pulser_wiring_value = pulser_wiring_green
            elif initial_pulse == 638:
                pulse_time = parameters_sig['pulsed_ionization_dur']
                laser_delay = laser_638_delay
                pulser_wiring_value = pulser_wiring_red
            # based on length scale, pusle in two different ways
            if pulse_time >= 10**9:
                time.sleep(0.002)
                cxn.pulse_streamer.constant([pulser_wiring_value], 0.0, 0.0)
                time.sleep(pulse_time/10**9)
                cxn.pulse_streamer.constant([], 0.0, 0.0)
            else:
                # Pusle the green laser on NV_readout
                seq_args = [laser_delay+galvo_delay,int( pulse_time), 0.0, initial_pulse]
                seq_args_string = tool_belt.encode_seq_args(seq_args)
                cxn.pulse_streamer.stream_immediate(pulse_file_name, 1, seq_args_string)


        # readout on NV in yellow
        
#        seq_args = [aom_589_delay, int(readout_pulse_time), aom_ao_589_pwr, apd_index, 589]    #simple readout
#        seq_args_string = tool_belt.encode_seq_args(seq_args)
#        
#        cxn.pulse_streamer.stream_load(readout_file_name, seq_args_string) # 0.01 s
#                
#        # collect the counts
#        cxn.apd_tagger.start_tag_stream(apd_indices) # 0.2 s
#        # Clear the buffer
#        cxn.apd_tagger.clear_buffer() #0.12 s
#        # Run the sequence
#        cxn.pulse_streamer.stream_immediate(readout_file_name, 1, seq_args_string) #0.05 s
#
#        new_counts = cxn.apd_tagger.read_counter_simple(1) # 0.25 s
#        sig_counts = new_counts[0]
#        signal_counts_list.append(int(sig_counts))
#
#        cxn.apd_tagger.stop_tag_stream() # 0.01 s
                
        ############### try time resolved readout
        
        processed_tags = []
        num_reps=1
        # Expose the stream
        cxn.apd_tagger.start_tag_stream(apd_indices, apd_indices, False)
    
        # Find the gate channel
        # The order of channel_mapping is APD, APD gate open, APD gate close
        channel_mapping = cxn.apd_tagger.get_channel_mapping()
        gate_open_channel = channel_mapping[1]
        gate_close_channel = channel_mapping[2]
        # Get the channel number for the clock on the tagger
        clock_wiring = tool_belt.get_time_tagger_wiring(cxn)
        clock_channel = clock_wiring['di_clock']
            
        # Stream the sequence
        seq_args = [readout_pulse_time, aom_589_delay ,
                aom_ao_589_pwr, apd_index,
                589]
#        print(seq_args)
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate('time_resolved_readout_clock_in_seq.py', int(num_reps),
                                            seq_args_string)
        time.sleep(1)
#        time.sleep(10)
        # Initialize state
        current_tags = []
        current_channels = []
        num_processed_reps = 0

        while num_processed_reps < num_reps:
            
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
            
#            ret_vals_string = cxn.apd_tagger.read_tag_stream()
#            new_tags,new_channels = tool_belt.decode_time_tags(ret_vals_string)
            new_tags, new_channels = cxn.apd_tagger.read_tag_stream_master()
            new_tags = numpy.array(new_tags, dtype=numpy.int64)
            print(new_tags)
            print(new_channels)
            if new_tags == []:
                continue
                        
            # MCC test
            if len(new_tags) > 750000:
                print()
                print('Received {} tags out of 10^6 max'.format(len(new_tags)))
                print('Turn down the reps and turn up the runs so that the Time Tagger can catch up!')
            
            ret_vals = time_resolved_readout.process_raw_buffer(new_tags, new_channels,
                                   current_tags, current_channels,
                                   gate_open_channel, gate_close_channel)
            new_processed_tags, num_new_processed_reps = ret_vals
            
            num_processed_reps += num_new_processed_reps
            
            processed_tags.extend(new_processed_tags)
            
            
        initial_clock = numpy.where(new_channels==clock_channel )[0][0]
        initial_gate = numpy.where(new_channels==gate_open_channel )[0][0]
        print(initial_clock)
        time_passed_ps = processed_tags[initial_clock] - processed_tags[initial_gate]
        print(time_passed_ps/10**12)
        
        processed_tags = [int(el) for el in processed_tags]
        readout_time_ps = 1000*readout_pulse_time
        num_bins = 100
        binned_samples, bin_edges = numpy.histogram(processed_tags, num_bins,
                                    (0, readout_time_ps))
        signal_counts_list.append(int(sum(binned_samples)))
        
        # Compute the centers of the bins
    bin_size = readout_pulse_time / num_bins
    bin_center_offset = bin_size / 2
    bin_centers = numpy.linspace(0, readout_pulse_time, num_bins) + bin_center_offset
#    print(bin_centers)
#
#    # Plot
#
#    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
#    
#    ax.plot(bin_centers, binned_samples, 'r-')
##    ax.set_title('Lifetime')
#    ax.set_xlabel('Readout time (ns)')
#    ax.set_ylabel('Counts')

    return signal_counts_list

def simple_pulse(target_sig, pulse_color):
    with labrad.connect() as cxn:
        simple_pulse_with_cxn(cxn, target_sig, pulse_color)

    return
def simple_pulse_with_cxn(cxn, target_sig, pulse_color):
    pulse_file_name = 'simple_pulse.py'

    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params['515_laser_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']

    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    pulser_wiring_red = wiring['do_638_laser']
    pulser_wiring_green = wiring['do_532_aom']

    drift = tool_belt.get_drift()
    target_coords = target_sig['coords']

    target_coords_drift = numpy.array(target_coords) + numpy.array(drift)

    # move the galvo to the readout
    tool_belt.set_xyz(cxn, target_coords_drift)
    time.sleep(0.001)

    # Short green pulse on readout NV before readout
    if pulse_color == 532:
        pulse_time = target_sig['pulsed_reionization_dur']
        laser_delay = laser_515_delay
        pulser_wiring_value = pulser_wiring_green
    elif pulse_color == 638:
        pulse_time = target_sig['pulsed_ionization_dur']
        laser_delay = laser_638_delay
        pulser_wiring_value = pulser_wiring_red
        
#    print(pulse_time/10**9)
    # based on length scale, pusle in two different ways
    if pulse_time >= 10**9:
        cxn.pulse_streamer.constant([pulser_wiring_value], 0.0, 0.0)
        time.sleep(pulse_time/10**9)
        cxn.pulse_streamer.constant([], 0.0, 0.0)
    else:
        # Pusle the green laser on NV_readout
        seq_args = [laser_delay,int( pulse_time), 0.0, pulse_color]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate(pulse_file_name, 1, seq_args_string)

    return

def simple_pulse_list(coords_list, init_pulse_time,pulse_color):
    with labrad.connect() as cxn:
        simple_pulse_list_with_cxn(cxn, coords_list, init_pulse_time, pulse_color)

    return
def simple_pulse_list_with_cxn(cxn, coords_list, init_pulse_time, pulse_color):
    pulse_file_name = 'simple_pulse.py'

    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    # aom_589_delay = shared_params['589_aom_delay']
    laser_515_delay = shared_params['515_laser_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = shared_params['large_angle_galvo_delay']

    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    pulser_wiring_red = wiring['do_638_laser']
    pulser_wiring_green = wiring['do_532_aom']
    # aom_ao_589_pwr = parameters_sig['am_589_power']

    # nd_filter = parameters_sig['nd_filter']
    # readout_pulse_time = parameters_sig['pulsed_SCC_readout_dur']
    # cxn.filter_slider_ell9k.set_filter(nd_filter)

    drift = tool_belt.get_drift()
    for coords in coords_list:
        coords_drift = numpy.array(coords) + numpy.array(drift)
       # move the galvo to the readout
        tool_belt.set_xyz(cxn, coords_drift)

        # pulse on each NV
        # These parameters might not be used, but let's just keep them in
        if pulse_color == 532:
            laser_delay = laser_515_delay
            pulser_wiring_value = pulser_wiring_green
        elif pulse_color == 638:
            laser_delay = laser_638_delay
            pulser_wiring_value = pulser_wiring_red

        # based on length scale, pusle in two different ways
        if init_pulse_time >= 10**9:
            time.sleep(0.002)
            cxn.pulse_streamer.constant([pulser_wiring_value], 0.0, 0.0)
            time.sleep(init_pulse_time/10**9)
            cxn.pulse_streamer.constant([], 0.0, 0.0)
        else:
            # Pusle the laser on NV_readout
            seq_args = [laser_delay+galvo_delay,int( init_pulse_time), 0.0, pulse_color]
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            cxn.pulse_streamer.stream_immediate(pulse_file_name, 1, seq_args_string)

    return

#%%
def charge_spot(readout_coords,target_A_coords, target_B_coords, parameters_sig, num_runs, init_scan):
    startFunctionTime = time.time()
    with labrad.connect() as cxn:
        tool_belt.reset_cfm(cxn)
    apd_indices = [0]
#    num_runs = 25
    # add the coords to the dictionry of measurement paramters
    readout_sig = copy.deepcopy(parameters_sig)
    readout_sig['coords'] = readout_coords

    target_A_sig = copy.deepcopy(parameters_sig)
    target_A_sig['coords'] = target_A_coords

    target_B_sig = copy.deepcopy(parameters_sig)
    target_B_sig['coords'] = target_B_coords

    # calculate the point between the readout and target_A, and use that as center of readout.
    scan_coords = (numpy.array(readout_coords) + numpy.array(target_A_coords))/2
#    print(scan_coords)
    scan_sig = copy.deepcopy(parameters_sig)
    scan_sig['coords'] = scan_coords

    readout_sec = readout_sig['pulsed_SCC_readout_dur'] / 10**9

    # calculate the distance between the NV and scan center spot,
    x_dif = readout_coords[0] - scan_coords[0]
    y_dif = readout_coords[1] - scan_coords[1]
#    z_dif = readout_coords[2] - scan_coords[2]
    distance_V = numpy.sqrt(x_dif**2 + y_dif**2)
    # ratio between this distance and 0.05 V (which I was previously using 0.1 V as the scan range)
    scan_range_scaling = distance_V/0.05
#    print(scan_range_scaling)
    # create some lists for dataopti_coords_list
    opti_coords_list = []
    control = []
    green_readout = []
    red_readout = []
    green_target_A = []
    red_target_A = []
    green_target_B = []
    red_target_B = []

    start_timestamp = tool_belt.get_time_stamp()
#    green_scan(readout_sig, apd_indices)
#    with labrad.connect() as cxn:
#        opti_coords = optimize.main_with_cxn(cxn, readout_sig, apd_indices, 532, disable=False)
#        opti_coords_list.append(opti_coords)

    for run in range(num_runs):
        print('run {}'.format(run))
        #optimize
        if run % 5 == 0:
            with labrad.connect() as cxn:
                opti_coords = optimize.main_with_cxn(cxn, readout_sig, apd_indices, 532, disable=False)
                opti_coords_list.append(opti_coords)

        # Step through the experiments

#         control: readout NV_readout
        if init_scan == 532:
            green_scan(scan_sig, apd_indices, scan_range_scaling)
        elif init_scan == 638:
            red_scan(scan_sig, apd_indices, scan_range_scaling)
        sig_count =  main(target_A_sig, readout_sig, None, 589, apd_indices)
#        control_kcps = (sig_count  / 10**3) / readout_sec
        control.append(int(sig_count))
        print('control: {} counts'.format(sig_count) )
#        print('control: {} kcps'.format(control_kcps) )

        # green_readout: measure NV after green pulse on readout NV
        if init_scan == 532:
            green_scan(scan_sig, apd_indices, scan_range_scaling)
        elif init_scan == 638:
            red_scan(scan_sig, apd_indices, scan_range_scaling)
        sig_count =  main(readout_sig, readout_sig, 532, 589, apd_indices)
#        green_readout_kcps = (sig_count  / 10**3) / readout_sec
        green_readout.append(int(sig_count))
        print('green_readout: {} counts'.format(sig_count) )
#        print('green readout: {} kcps'.format(green_readout_kcps) )

        # red_readout: measure NV after red pulse on readout NV
        if init_scan == 532:
            green_scan(scan_sig, apd_indices, scan_range_scaling)
        elif init_scan == 638:
            red_scan(scan_sig, apd_indices, scan_range_scaling)
        sig_count =  main(readout_sig, readout_sig, 638, 589, apd_indices)
#        red_readout_kcps = (sig_count  / 10**3) / readout_sec
        red_readout.append(int(sig_count))
        print('red_readout: {} counts'.format(sig_count) )
#        print('red readout: {} kcps'.format(red_readout_kcps) )

        # green_target: measure NV after green pulse on target NV
        if init_scan == 532:
            green_scan(scan_sig, apd_indices, scan_range_scaling)
        elif init_scan == 638:
            red_scan(scan_sig, apd_indices, scan_range_scaling)
        sig_count =  main(target_A_sig, readout_sig, 532, 589, apd_indices)
#        green_target_A_kcps = (sig_count  / 10**3) / readout_sec
        green_target_A.append(int(sig_count))
        print('green_target_A: {} counts'.format(sig_count) )
#        print('green target: {} kcps'.format(green_target_kcps) )

        # red_target: measure NV after red pulse on target NV
        if init_scan == 532:
            green_scan(scan_sig, apd_indices, scan_range_scaling)
        elif init_scan == 638:
            red_scan(scan_sig, apd_indices, scan_range_scaling)
        sig_count =  main(target_A_sig, readout_sig, 638, 589, apd_indices)
#        red_target_A_kcps = (sig_count  / 10**3) / readout_sec
        red_target_A.append(int(sig_count))
        print('red_target_A: {} counts'.format(sig_count) )
#        print('red target: {} kcps'.format(red_target_kcps) )
#
#        # green_B: measure NV after green pulse on dark spot
#        if init_scan == 532:
#            green_scan(scan_sig, apd_indices, scan_range_scaling)
#        elif init_scan == 638:
#            red_scan(scan_sig, apd_indices, scan_range_scaling)
#        sig_count =  main(target_B_sig, readout_sig, 532, 589, apd_indices)
#        green_target_B_kcps = (sig_count  / 10**3) / readout_sec
#        green_target_B.append(green_target_B_kcps)
#        print('green_target_B: {} counts'.format(sig_count) )
##        print('green dark: {} kcps'.format(green_dark_kcps) )
#
#        # red_B: measure NV after red pulse on dark spot
#        if init_scan == 532:
#            green_scan(scan_sig, apd_indices, scan_range_scaling)
#        elif init_scan == 638:
#            red_scan(scan_sig, apd_indices, scan_range_scaling)
#        sig_count =  main(target_B_sig, readout_sig, 638, 589, apd_indices)
#        red_target_B_kcps = (sig_count  / 10**3) / readout_sec
#        red_target_B.append(red_target_B_kcps)
#        print('red_target_B: {} counts'.format(sig_count) )
##        print('red dark: {} kcps'.format(red_dark_kcps) )

        raw_data = {'start_time': start_timestamp,
                'readout_coords': readout_coords,
                'target_A_coords': target_A_coords,
                'target_B_coords': target_B_coords,
                'parameters_sig': parameters_sig,
                'parameters_sig-units': tool_belt.get_nv_sig_units(),
                'num_runs':num_runs,
                'opti_coords_list': opti_coords_list,
                'control': control,
                'control-units': 'counts',
                'green_readout': green_readout,
                'green_readout-units': 'counts',
                'red_readout': red_readout,
                'red_readout-units': 'counts',
                'green_target_A': green_target_A,
                'green_target_A-units': 'counts',
                'red_target_A': red_target_A,
                'red_target_A-units': 'counts',
                'green_target_B': green_target_B,
                'green_target_B-units': 'counts',
                'red_target_B': red_target_B,
                'red_target_B-units': 'counts',
                }

        file_path = tool_belt.get_file_path(__file__, start_timestamp, parameters_sig['name'], 'incremental')
        if init_scan == 532:
            tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge-green_init')
        elif init_scan == 638:
            tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge-red_init')

    control_avg = numpy.average(control)
    control_ste = stats.sem(control)
    print('control measurement avg: {} +/- {} counts'.format(control_avg, control_ste))
    green_readout_avg = numpy.average(green_readout)
    green_readout_ste = stats.sem(green_readout)
    print('green readout measurement avg: {} +/- {} counts'.format(green_readout_avg, green_readout_ste))
    red_readout_avg = numpy.average(red_readout)
    red_readout_ste = stats.sem(red_readout)
    print('red readout measurement avg: {} +/- {} counts'.format(red_readout_avg, red_readout_ste))
    green_target_A_avg = numpy.average(green_target_A)
    green_target_A_ste = stats.sem(green_target_A)
    print('green target measurement avg: {} +/- {} counts'.format(green_target_A_avg, green_target_A_ste))
    red_target_A_avg = numpy.average(red_target_A)
    red_target_A_ste = stats.sem(red_target_A)
    print('red target measurement avg: {} +/- {} counts'.format(red_target_A_avg, red_target_A_ste))
#    green_target_B_avg = numpy.average(green_target_B)
#    green_target_B_ste =  stats.sem(green_target_B)
#    print('green dark measurement avg: {} +/- {} kcps'.format(green_target_B_avg, green_target_B_ste))
#    red_target_B_avg = numpy.average(red_target_B)
#    red_target_B_ste = stats.sem(red_target_B)
#    print('red dark measurement avg: {} +/- {} kcps'.format(red_target_B_avg, red_target_B_ste))

    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              readout_sig['am_589_power'], parameters_sig['nd_filter'])

    # Save

    endFunctionTime= time.time()
    timeElapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'timeElapsed':timeElapsed,
            'readout_coords': readout_coords,
            'target_A_coords': target_A_coords,
            'target_B_coords': target_B_coords,
            'parameters_sig': parameters_sig,
            'parameters_sig-units': tool_belt.get_nv_sig_units(),
            'green_optical_power_pd': green_optical_power_pd,
            'green_optical_power_pd-units': 'V',
            'green_optical_power_mW': green_optical_power_mW,
            'green_optical_power_mW-units': 'mW',
            'red_optical_power_pd': red_optical_power_pd,
            'red_optical_power_pd-units': 'V',
            'red_optical_power_mW': red_optical_power_mW,
            'red_optical_power_mW-units': 'mW',
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
            'num_runs':num_runs,
            'opti_coords_list': opti_coords_list,
            'control': control,
            'control-units': 'counts',
            'green_readout': green_readout,
            'green_readout-units': 'counts',
            'red_readout': red_readout,
            'red_readout-units': 'counts',
            'green_target_A': green_target_A,
            'green_target_A-units': 'counts',
            'red_target_A': red_target_A,
            'red_target_A-units': 'counts',
            'green_target_B': green_target_B,
            'green_target_B-units': 'counts',
            'red_target_B': red_target_B,
            'red_target_B-units': 'counts',

            'control_avg': control_avg,
            'control_avg-units': 'counts',
            'green_readout_avg': green_readout_avg,
            'green_readout_avg-units': 'counts',
            'red_readout_avg': red_readout_avg,
            'red_readout_avg-units': 'counts',
            'green_target_A_avg': green_target_A_avg,
            'green_target_A_avg-units': 'counts',
            'red_target_A_avg': red_target_A_avg,
            'red_target_A_avg-units': 'counts',
#            'green_target_B_avg': green_target_B_avg,
#            'green_target_B_avg-units': 'kcps',
#            'red_target_B_avg': red_target_B_avg,
#            'red_target_B_avg-units': 'kcps',

            'control_ste': control_ste,
            'control_ste-units': 'counts',
            'green_readout_ste': green_readout_ste,
            'green_readout_ste-units': 'counts',
            'red_readout_ste': red_readout_ste,
            'red_readout_ste-units': 'counts',
            'green_target_A_ste': green_target_A_ste,
            'green_target_A_ste-units': 'counts',
            'red_target_A_ste': red_target_A_ste,
            'red_target_A_ste-units': 'counts',
#            'green_target_B_ste': green_target_B_ste,
#            'green_target_B_ste-units': 'kcps',
#            'red_target_B_ste': red_target_B_ste,
#            'red_target_B_ste-units': 'kcps',
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, parameters_sig['name'])
    if init_scan == 532:
        tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge-green_init')
    elif init_scan == 638:
        tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge-red_init')


    print(' \nRoutine complete!')
    return

#%%
def charge_spot_list(target_coords, readout_coords_list, parameters_sig, pulse_time, num_runs, init_color, pulse_color):
    startFunctionTime = time.time()
    init_pulse_time = 5 * 10**3 # initial time to pulse laser to initialize NVs

    with labrad.connect() as cxn:
        tool_belt.reset_cfm(cxn)

    apd_indices = [0]
#    readout_sec = readout_sig['pulsed_SCC_readout_dur'] / 10**9
    
    # put the pulse time in the correct parameter value in parameters_sig
    if pulse_color == 532:
        parameters_sig['pulsed_reionization_dur'] = pulse_time
    elif pulse_color == 638:
        parameters_sig['pulsed_ionization_dur'] = pulse_time

    # add the coords to the dictionry of measurement paramters
    target_sig = copy.deepcopy(parameters_sig)
    target_sig['coords'] = target_coords

#    optimize_sig = copy.deepcopy(parameters_sig)
#    optimize_sig['coords'] = optimize_coords


    # Have a list of all coords to use with initialization
    all_coords_list = readout_coords_list + [target_coords]

    #calculate the distances from the target to each readout:
    rad_dist_list = []
    for coords in readout_coords_list:
        coords_diff = numpy.array(target_coords) - numpy.array(coords)
        coords_diff_sqrd = coords_diff**2
        rad_dist = numpy.sqrt(sum(coords_diff_sqrd))
        rad_dist_list.append(rad_dist)

    rad_dist_list_sort =  copy.deepcopy(rad_dist_list)
    rad_dist_list_sort.sort()
    # pick out the largest distance, this will define our red scan area
    max_rad_dist = rad_dist_list_sort[-1]

    #convert the radial distances to um
    rad_dist_list_um = numpy.array(rad_dist_list)*35

    # ratio between this distance and 0.05 V (which I was previously using 0.1 V as the scan range)
    scan_range_scaling = (max_rad_dist*1.05)/0.05

#    fig, ax = plt.subplots(1, 1, figsize=(17, 8.5))
#    ax.plot(rad_dist_list_um, rad_dist_list_um, 'o')
#    ax.set_xlabel('Distance from central target NV (um)')
#    ax.legend()
#    return

    # create some lists for dataopti_coords_list
    opti_coords_list = []
    control_array = []
    green_readout_array = []
    green_target_array = []

    start_timestamp = tool_belt.get_time_stamp()
    #optimize at the beginning of the measurement
    with labrad.connect() as cxn:
        opti_coords = optimize.main_with_cxn(cxn, target_sig, apd_indices, 532, disable=False)
        opti_coords_list.append(opti_coords)
    # record the time starting at the beginning of the runs
    run_start_time = time.time()

    for run in range(num_runs):
        print('run {}'.format(run))

        #optimize every 5 min or so
        # So first check the time. If the time that has passed since the last
        # optimize is longer that 5 min, optimize again
        current_time = time.time()
        if current_time - run_start_time >= 5*60:
#        if run % 3 == 0:
            with labrad.connect() as cxn:
                opti_coords = optimize.main_with_cxn(cxn, target_sig, apd_indices, 532, disable=False)
                opti_coords_list.append(opti_coords)
            run_start_time = current_time

#        start_time = time.time()
        ############## Control measurement on each NV
        # initialize pulses on all NVs
        simple_pulse_list(all_coords_list, init_pulse_time,init_color)

        counts_list = readout_list(readout_coords_list, parameters_sig, apd_indices)
        control_array.append(counts_list)
#        end_time = time.time()
#        print('control: ' + str(end_time - start_time))

#        start_time = time.time()
        ############## pulse on each NV
        # initialize pulses on all NVs
        simple_pulse_list(all_coords_list, init_pulse_time,init_color)

        counts_list = readout_list(readout_coords_list, parameters_sig, apd_indices, initial_pulse = pulse_color)
        green_readout_array.append(counts_list)
#        end_time = time.time()
#        print('readout: ' + str(end_time - start_time))

#        start_time = time.time()
        ############### pusle on target
        # initialize pulses on all NVs
        simple_pulse_list(all_coords_list, init_pulse_time,init_color)

        simple_pulse(target_sig, pulse_color)

        counts_list = readout_list(readout_coords_list, parameters_sig, apd_indices)
        green_target_array.append(counts_list)
#        end_time = time.time()
#        print('target: ' + str(end_time - start_time))

        raw_data = {'start_time': start_timestamp,
                'init_color': init_color,
                'pulse_color': pulse_color,
                'readout_coords_list': readout_coords_list,
                'target_coords': target_coords,
#            'optimize_sig': optimize_sig,
                'parameters_sig': parameters_sig,
                'parameters_sig-units': tool_belt.get_nv_sig_units(),
            'init_pulse_time': init_pulse_time,
            'init_pulse_time-units': 'ns',
            'final_pulse_time': pulse_time,
            'final_pulse_time-units': 'ns',
                'num_runs':num_runs,
                'opti_coords_list': opti_coords_list,
                'rad_dist_list': rad_dist_list,
                'rad_dist_list-units': 'V',
                'control_array': control_array,
                'control_array-units': 'counts',
                'green_readout_array': green_readout_array,
                'green_readout_array-units': 'counts',
                'green_target_array': green_target_array,
                'green_target_array-units': 'counts',
                }

        file_path = tool_belt.get_file_path(__file__, start_timestamp, parameters_sig['name'], 'incremental')
        if init_color == 532:
            tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge_list-green_init')
        elif init_color == 638:
            tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge_list-red_init')

    control_avg = numpy.average(control_array, axis = 0)
    control_ste = stats.sem(control_array, axis = 0)
    green_readout_avg = numpy.average(green_readout_array, axis = 0)
    green_readout_ste = stats.sem(green_readout_array, axis = 0)
    green_target_avg = numpy.average(green_target_array, axis = 0)
    green_target_ste = stats.sem(green_target_array, axis = 0)

    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              parameters_sig['am_589_power'], parameters_sig['nd_filter'])

    # Save

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
                'init_color': init_color,
                'pulse_color': pulse_color,
            'readout_coords_list': readout_coords_list,
            'target_coords': target_coords,
#            'optimize_sig': optimize_sig,
            'parameters_sig': parameters_sig,
            'parameters_sig-units': tool_belt.get_nv_sig_units(),
            'init_pulse_time': init_pulse_time,
            'init_pulse_time-units': 'ns',
            'final_pulse_time': pulse_time,
            'final_pulse_time-units': 'ns',
            'green_optical_power_pd': green_optical_power_pd,
            'green_optical_power_pd-units': 'V',
            'green_optical_power_mW': green_optical_power_mW,
            'green_optical_power_mW-units': 'mW',
            'red_optical_power_pd': red_optical_power_pd,
            'red_optical_power_pd-units': 'V',
            'red_optical_power_mW': red_optical_power_mW,
            'red_optical_power_mW-units': 'mW',
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
            'num_runs':num_runs,
            'opti_coords_list': opti_coords_list,
            'rad_dist_list': rad_dist_list,
            'rad_dist_list-units': 'V',
            'control_array': control_array,
            'control_array-units': 'counts',
            'green_readout_array': green_readout_array,
            'green_readout_array-units': 'counts',
            'green_target_array': green_target_array,
            'green_target_array-units': 'counts',

            'control_avg': control_avg.tolist(),
            'control_avg-units': 'counts',
            'green_readout_avg': green_readout_avg.tolist(),
            'green_readout_avg-units': 'counts',
            'green_target_avg': green_target_avg.tolist(),
            'green_target_avg-units': 'counts',

            'control_ste': control_ste.tolist(),
            'control_ste-units': 'counts',
            'green_readout_ste': green_readout_ste.tolist(),
            'green_readout_ste-units': 'counts',
            'green_target_ste': green_target_ste.tolist(),
            'green_target_ste-units': 'counts',
            }


    fig, ax = plt.subplots(1, 1, figsize=(17, 8.5))
    ax.errorbar(rad_dist_list_um, control_avg, yerr = control_ste,fmt = 'ko', label = 'control measurement (no initial pulse)')
    ax.errorbar(rad_dist_list_um, green_readout_avg, yerr = green_readout_ste,fmt = 'go', label = 'initial {} pulse on individual NV'.format(pulse_color))
    ax.errorbar(rad_dist_list_um, green_target_avg, yerr = green_target_ste, fmt = 'bo',label = 'initial {} pulse on single target NV'.format(pulse_color))
    ax.set_title('Pulsed charge measurements on multiple NVs ({} init, {} s {} pulse)'.format(init_color, pulse_time/10**9, pulse_color))
    ax.set_xlabel('Distance from central target NV (um)')
    ax.set_ylabel('Average counts')
    ax.legend()


    file_path = tool_belt.get_file_path(__file__, timestamp, parameters_sig['name'])
    if init_color == 532:
        tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge_list-green_init')
        tool_belt.save_figure(fig, file_path + '-isoalted_nv_charge_list-green_init')
    elif init_color == 638:
        tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge_list-red_init')
        tool_belt.save_figure(fig, file_path + '-isoalted_nv_charge_list-red_init')


    print(' \nRoutine complete!')
    return
# %% Run the files

if __name__ == '__main__':
    sample_name = 'johnson'
#    NVA_coords = [0.169, 0.014, 5.2]
#    NVB_coords = [0.396, -0.081, 5.2]
#    dark_spot_1_coords = [0.392, -0.110,  5.2]
#    dark_spot_2_coords = [0.108, 0.007, 5.2]

#    NV_target = [0.078- 0.018, 0.054 - 0.003, 5.22 + 0.04] #account for the drift on 11/16
    NV_target = [0.047, 0.030, 5.22]
#    NV_target = [-0.971, 0.064, 5.22]

    nv_readout_list = [ [0.047, 0.031, 5.21],
    [0.051, 0.007, 5.21],
    [0.020, 0.066, 5.22],
    [0.072, 0.076, 5.22],
    [0.117, 0.023, 5.21],
    [0.084, -0.016, 5.21],
    [0.009, 0.100, 5.24],
#    [-0.016, 0.115, 5.22],
#    [0.110, 0.102, 5.21],
#    [0.174, 0.010, 5.24],
#    [0.156, 0.143, 5.24],
#    [0.143, 0.119, 5.21],
#    [0.170, 0.104, 5.21],
#    [0.055, -0.126, 5.21],
#    [-0.037, -0.143, 5.21],
#    [0.157, 0.143, 5.23],
#    [0.179, 0.247, 5.26],
#    [0.174, 0.010, 5.23],
#    [0.265, -0.032, 5.24],
#    [0.291, 0.066, 5.22],
#    [0.291, -0.142, 5.21],
#    [0.363, 0.066, 5.22],
#    [0.402, -0.085, 5.21],
    [0.402, -0.169, 5.21],
    [0.354, 0.198, 5.28],
    [0.264, -0.032, 5.25],]

#    nv_readout_list = [[-0.955, 0.024, 5.24],
#                       [-0.970, 0.060, 5.20],
#[-0.880, 0.005, 5.17],
#[-1.151, 0.019, 5.18],
#[-1.070, -0.053, 5.19],
#[-0.852, -0.048, 5.15],
#[-0.898, -0.105, 5.18],
#[-1.139, -0.147, 5.20],
#[-0.913, -0.169, 5.17],
#[-0.973, -0.285, 5.25],
#[-1.141, -0.316, 5.21],
#[-0.833, -0.344, 5.22],
#[-0.698, -0.220, 5.17],
#[-1.033, -0.467, 5.17],
#[-0.963, -0.468, 5.22],
#[-1.143, -0.494, 5.25],
#[-0.796, -0.561, 5.15],
#[-0.806, -0.629, 5.25],
#[-1.070, -0.682, 5.18],
#[-0.796, -0.703, 5.20],
#[-0.979, -0.773, 5.15],
#[-1.174, -0.722, 5.19],
#[-1.008, -0.831, 5.20],
#[-1.102, -0.875, 5.15],
#[-0.994, -0.925, 5.20],
#[-0.954, -0.951, 5.18],
#[-0.995, -1.035, 5.21],
#[-1.008, -1.094, 5.19],
#[-1.168, -1.170, 5.22],
#[-1.250, -1.108, 5.19],
#[-0.821, -1.187, 5.22],
#[-0.844, -1.264, 5.25],
#[-1.102, -1.304, 5.21],
#[-0.996, -1.356, 5.19],
#[-0.945, -1.391, 5.25],
#[-1.318, -1.308, 5.25],
#
#            ]

    num_runs =2#60
    init_color = 532
    pulse_color = 638

    # The parameters that we want to run these measurements with
#    base_nv_sig  = { 'coords':None,
#            'name': '{}-NVA'.format(sample_name),
#            'expected_count_rate': 45, 'nd_filter': 'nd_0',
#            'pulsed_SCC_readout_dur': 4*10**6, 'am_589_power': 0.2,
#            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120,
#            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':20,
#            'magnet_angle': 0}


    # run the measurements!
#    for t_g in [10**5, 10**6, 10**7, 10**8, 10**9, 10**10]:
    for t_g in [10**5]:
        base_nv_sig  = { 'coords':None,
                'name': '{}-NVA'.format(sample_name),
                'expected_count_rate': 60, 'nd_filter': 'nd_0',
                'pulsed_SCC_readout_dur': 4*10**6, 'am_589_power': 0.2,
                'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120,
                'pulsed_reionization_dur': 10**5, 'cobalt_532_power':20,
                'magnet_angle': 0}
#        charge_spot(NVA_coords, NVB_coords, dark_spot_1_coords, base_nv_sig, num_runs, 638)
#        charge_spot(dark_spot_1_coords, NVA_coords, dark_spot_2_coords,base_nv_sig, num_runs, 638)
        charge_spot_list(NV_target, nv_readout_list, base_nv_sig, t_g,  num_runs, init_color, pulse_color)

#    norm_sub_folder = 'collect_charge_counts/branch_Spin_to_charge/2020_11'
#    norm_file = '2020_11_16-16_04_47-johnson-2020_11_10-nv_list' # 11/10 - 15 um
#    # norm_file = '2020_11_16-15_49_59-johnson-2020_11_13-nv_list' # 11/13 - 50 um
#
#
#    file_10s_dark = '2020_11_17-10_17_37-johnson-NVA-isoalted_nv_charge_list-red_init'
#    file_10s_ggNV ='2020_11_12-18_02_49-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_10s_grNV ='2020_11_13-09_17_25-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_10s_rgNV_extent = '2020_11_14-08_53_17-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_10s_rrNV = '2020_11_18-00_58_36-johnson-NVA-isoalted_nv_charge_list-red_init'
#
#    file_1s_NV = '2020_11_11-01_07_19-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_1s_dark = '2020_11_17-04_23_50-johnson-NVA-isoalted_nv_charge_list-red_init'
#    file_1s_ggNV ='2020_11_12-01_30_37-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_1s_grNV ='2020_11_12-21_16_25-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_1s_rgNV_extent = '2020_11_14-00_46_37-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_1s_rrNV = '2020_11_17-23_35_21-johnson-NVA-isoalted_nv_charge_list-red_init'
#
#    file_100ms_NV = '2020_11_11-12_09_13-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_100ms_dark = '2020_11_17-02_35_42-johnson-NVA-isoalted_nv_charge_list-red_init'
#    file_100ms_ggNV ='2020_11_12-11_05_05-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_100ms_grNV ='2020_11_13-06_52_53-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_100ms_rgNV_extent = '2020_11_15-18_33_43-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_100ms_rrNV = '2020_11_17-22_12_11-johnson-NVA-isoalted_nv_charge_list-red_init'
#
#    file_10ms_NV = '2020_11_11-08_15_52-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_10ms_dark = '2020_11_17-01_14_10-johnson-NVA-isoalted_nv_charge_list-red_init'
#    file_10ms_ggNV ='2020_11_12-08_42_14-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_10ms_grNV ='2020_11_13-04_28_42-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_10ms_rgNV_extent = '2020_11_15-16_39_55-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_10ms_rrNV = '2020_11_17-20_49_00-johnson-NVA-isoalted_nv_charge_list-red_init'
#
#    file_1ms_NV = '2020_11_11-05_53_03-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_1ms_dark = '2020_11_16-23_52_36-johnson-NVA-isoalted_nv_charge_list-red_init'
#    file_1ms_ggNV ='2020_11_12-06_18_47-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_1ms_grNV ='2020_11_13-02_04_51-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_1ms_rgNV_extent = '2020_11_15-14_46_07-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_1ms_rrNV = '2020_11_17-19_25_37-johnson-NVA-isoalted_nv_charge_list-red_init'
#
#    file_100us_NV = '2020_11_11-03_30_38-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_100us_dark = '2020_11_16-22_30_50-johnson-NVA-isoalted_nv_charge_list-red_init'
#    file_100us_ggNV= '2020_11_12-03_54_57-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_100us_grNV ='2020_11_12-23_40_33-goeppert-mayer-NVA-isoalted_nv_charge_list-green_init'
#    file_100us_rgNV_extent = '2020_11_15-12_51_40-goeppert-mayer-NVA-isoalted_nv_charge_list-red_init'
#    file_100us_rrNV = '2020_11_17-18_02_27-johnson-NVA-isoalted_nv_charge_list-red_init'
#
#    file_list = [file_10s_dark,file_1s_dark,file_100ms_dark,
#                 file_10ms_dark, file_1ms_dark, file_100us_dark]
#    fmt_list = ['D',
#                'o',
#                '^', 's','x',
#                   '+']
#    # label_list = ['target data', 'nv0', 'nv-']
#
#    sub_folder = 'isolate_nv_charge_dynamics/branch_Spin_to_charge/2020_11'
#
#
#    fig, ax = plt.subplots(1, 1, figsize=(17, 8.5))
#    norm_data = tool_belt.get_raw_data(norm_sub_folder, norm_file)
#
#    control_avg = numpy.array(norm_data['nv0_avg_list'])
#    control_ste = numpy.array(norm_data['nv0_ste_list'])
#    green_readout_avg = numpy.array(norm_data['nvm_avg_list'])
#    green_readout_ste = numpy.array(norm_data['nvm_ste_list'])
#
#    i = 0
#    for f in file_list:
#        data = tool_belt.get_raw_data(sub_folder, f)
#
#        parameters_sig = data['parameters_sig']
#        readout_coords_list = data['readout_coords_list']
#        target_coords = data['target_coords']
#        # control_avg = numpy.array(data['control_avg'])
#        # control_ste = numpy.array(data['control_ste'])
#        # green_readout_avg = numpy.array(data['green_readout_avg'])
#        # green_readout_ste = numpy.array(data['green_readout_ste'])
#        green_target_avg = numpy.array(data['green_target_avg'])
#        green_target_ste = numpy.array(data['green_target_ste'])
#        # rad_dist_list = data['rad_dist_list']
#        pulse_time = data['final_pulse_time']
#
#        rad_dist_list = []
#        for coords in readout_coords_list:
#            coords_diff = numpy.array(target_coords) - numpy.array(coords)
#            coords_diff_sqrd = coords_diff**2
#            rad_dist = numpy.sqrt(sum(coords_diff_sqrd))
#            rad_dist_list.append(rad_dist)
#
#        rad_dist_list_um = numpy.array(rad_dist_list)*35
#        pulse_time = parameters_sig['pulsed_reionization_dur']
#
#        normalized_counts = (green_target_avg - control_avg) / (green_readout_avg - control_avg)
#
#        # calculating uncertainty
#        n = green_target_avg - control_avg
#        d = green_readout_avg - control_avg
#        term_1 = numpy.sqrt(green_target_ste**2 + control_ste**2)/n
#        term_2 = numpy.sqrt(green_readout_ste**2 + control_ste**2)/d
#        normalized_unc = normalized_counts*numpy.sqrt(term_1**2 + term_2**2)
#
#        # sorting the list based on the radial distance so we can do a line plot
#        paired_data = list(zip(rad_dist_list_um, normalized_counts, normalized_unc))
#        sorted_paired_data = sorted(paired_data, key=lambda x: x[0])
#
#        sorted_rad_dist_list_um = [x[0] for x in sorted_paired_data]
#        sorted_normalized_counts = [x[1] for x in sorted_paired_data]
#        sorted_normalized_unc = [x[2] for x in sorted_paired_data]
#
#        # For plotting things a little clearer, exclude NVs withlarge uncertainty
#        del_list = []
#        for u in range(len(sorted_normalized_unc)):
#            unc= abs(sorted_normalized_unc[u])
#            if unc > 1:
#                del_list.append(u)
#        for d in sorted(del_list, reverse=True):
#            del sorted_rad_dist_list_um[d]
#            del sorted_normalized_counts[d]
#            del sorted_normalized_unc[d]
#
#        ax.errorbar(sorted_rad_dist_list_um, sorted_normalized_counts, fmt = fmt_list[i],
#                      yerr = sorted_normalized_unc,
#                # label = label_list[i])
#                        label = '{} ms green pulse'.format(pulse_time/10**6))
#
#        # ax.plot(rad_dist_list_um, normalized_counts, fmt_list[i],
#        #         # label = label_list[i])
#        #                 label = '{} ms green pulse'.format(pulse_time/10**6))
#
#        # ax.errorbar(rad_dist_list_um, green_target_avg, yerr=green_target_ste, fmt ='o', label = 'target')
#        # ax.errorbar(rad_dist_list_um, green_readout_avg, yerr=green_readout_ste, fmt='o', label = 'nv-')
#        # ax.errorbar(rad_dist_list_um, control_avg, yerr=control_ste, fmt='o', label = 'nv0')
#        # ax.set_ylim([-1.5,2])
#        i += 1
#
#    ax.set_title('Pulsed charge measurements on multiple NVs (red initialization, green pulse), on dark spot')
#    ax.set_xlabel('Distance from central target NV (um)')
#    ax.set_ylabel('Normalized counts')
#    ax.legend()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:39:55 2019

Collect the photon counts under yellow illumination, after reionizing NV into
NV- with green light.

@author: yanfeili
"""

# %% import
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import matplotlib.pyplot as plt
import labrad
import time
from scipy.optimize import curve_fit
import scipy.stats
import scipy.special
import math
import minorroutines.photonstatistics as ps

#%% function

#get photon distribution
def get_Probability_distribution(aList):

    def get_unique_value(aList):
        unique_value_list = []
        for i in range(0,len(aList)):
            if aList[i] not in unique_value_list:
                unique_value_list.append(aList[i])
        return unique_value_list
    unique_value = get_unique_value(aList)
    relative_frequency = []
    for i in range(0,len(unique_value)):
        relative_frequency.append(aList.count(unique_value[i])/ (len(aList)))

    return unique_value, relative_frequency

#quick double poisson curve fit

#given the data, return the fit
def get_poisson_distribution_fit(readout_time,unique_value, relative_frequency):
    tR = readout_time
    number_of_photons = unique_value
    def PoissonDistribution(number_of_photons, a, b, numbla1, numbla2):
        #numbla1 and numbla2 represent the fluorescence rate
        poissonian =[]
        for i in range(len(number_of_photons)):
            n = number_of_photons[i]
            poissonian.append((a*(numbla1*tR)**n) * (math.e ** (-numbla1*tR)) /math.factorial(n) + b*((numbla2*tR)**n) * (math.e ** (-numbla2*tR)) /math.factorial(n))
        return poissonian
    popt, pcov = curve_fit(PoissonDistribution, number_of_photons,  relative_frequency)
    print(popt, pcov)
    return popt
#given the fit, return the curve
def get_poisson_distribution_curve(number_of_photons,readout_time, a, b, numbla1, numbla2):
    poissonian_curve =[]
    tR = readout_time
    for i in range(len(number_of_photons)):
        n = number_of_photons[i]
        poissonian_curve.append((a*(numbla1*tR)**n) * (math.e ** (-numbla1*tR)) /math.factorial(n) + b*((numbla2*tR)**n) * (math.e ** (-numbla2*tR)) /math.factorial(n))
    return poissonian_curve


#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, aom_ao_589_pwr,readout_time, num_runs, num_reps):

    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, apd_indices, aom_ao_589_pwr,readout_time, num_runs, num_reps)

def main_with_cxn(cxn, nv_sig, apd_indices, aom_ao_589_pwr,readout_time,num_runs, num_reps):

    tool_belt.reset_cfm(cxn)

# %% Initial Calculation and setup
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #Define some parameters

    #delay of aoms
    aom_delay = shared_params['515_laser_delay']

    #readout_power in unit of microwatts
    # aom_power = numpy.sqrt((readout_power - 0.432)/1361.811) #uW
    #'TBD'
#    readout_power = 0

    reionization_time = 1*10**6
    illumination_time = readout_time + 10**3

    # Set up our data structure, an array of NaNs that we'll fill
    # we repeatively collect photons for tR

    sig_counts=[]
    opti_coords_list = []

    # %% Read the optical power for yellow and green light

    green_optical_power_pd = tool_belt.opt_power_via_photodiode(532)

    yellow_optical_power_pd = tool_belt.opt_power_via_photodiode(589,
           AO_power_settings = aom_ao_589_pwr, nd_filter = nv_sig['nd_filter'])

    # Convert V to mW optical power
    green_optical_power_mW = \
            tool_belt.calc_optical_power_mW(532, green_optical_power_pd)

    yellow_optical_power_mW = \
            tool_belt.calc_optical_power_mW(589, yellow_optical_power_pd)



#%% Estimate the lenth of the sequance

    seq_args = [readout_time, reionization_time, illumination_time, aom_delay,
                apd_indices[0], aom_ao_589_pwr]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load('determine_n_thresh.py', seq_args_string)

    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * num_runs * seq_time_s + (0.5 * num_runs)  # s
    expected_run_time_m = expected_run_time / 60 # m

    # Ask to continue and timeout if no response in 2 seconds?

    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))

#    return

#%% Collect data
    tool_belt.init_safe_stop()


    for run_ind in range(num_runs):

        print('Run index: {}'. format(run_ind))

        # Optimize
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable = True)
        opti_coords_list.append(opti_coords)
#
#        drift = numpy.array(tool_belt.get_drift())
#        coords = numpy.array(nv_sig['coords'])
#
#        coords_drift = coords - drift
#
#        cxn.galvo.write(coords_drift[0], coords_drift[1])
#        cxn.objective_piezo.write(coords_drift[2])

        #  set filter slider according to nv_sig
        ND_filter = nv_sig['nd_filter']
        cxn.filter_slider_ell9k.set_filter(ND_filter)
        time.sleep(0.1)


        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)

        seq_args = [readout_time, reionization_time, illumination_time,
                    aom_delay ,apd_indices[0], aom_ao_589_pwr]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate('determine_n_thresh.py', num_reps, seq_args_string)

        # Get the counts
        new_counts = cxn.apd_tagger.read_counter_simple(num_reps)

        sig_counts.extend(new_counts)

    cxn.apd_tagger.stop_tag_stream()


#%% plot the data and fit

    unique_value, relative_frequency = get_Probability_distribution(list(sig_counts))

    #double poisson fit
    a, b, numbla1, numbla2 = get_poisson_distribution_fit(readout_time*10**-9,unique_value, relative_frequency)
    number_of_photons = list(range(max(unique_value)+1))
    curve = get_poisson_distribution_curve(number_of_photons,readout_time*10**-9, a, b, numbla1, numbla2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))

    ax.plot(unique_value, relative_frequency, 'bo')
    ax.plot(number_of_photons, curve,'r')
    ax.set_xlabel('number of photons (n)')
    ax.set_ylabel('P(n)')

    text = '\n'.join(('Reionization time (532 nm)' + '%.3f'%(reionization_time/10**3) + 'us',
                      'Illumination time (589 nm)' + '%.3f'%(illumination_time/10**3) + 'us'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.55, 0.6, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

  #%% monitor photon counts
    fig2, ax2 = plt.subplots(1,1,figsize = (10,8.5))

    time_axe = ps.get_time_axe(seq_time_s, readout_time*10**-9,sig_counts)
    photon_counts = ps.get_photon_counts(readout_time*10**-9, sig_counts)

    ax2.plot(time_axe,photon_counts)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('photon counts (cps)')

    text = '\n'.join(('Readout time (589 nm)'+'%.3f'%(readout_time/10**3) + 'us',
                     'Readout power (589 nm)'+'%.3f'%(yellow_optical_power_mW * 1000) + 'uW'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.55, 0.6, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)


#%% Save data
    timestamp = tool_belt.get_time_stamp()

    # turn the list of unique_values into pure integers, for saving
    unique_value = [int(el) for el in unique_value]
    sig_counts = [int(el) for el in sig_counts]

    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'aom_ao_589_pwr': aom_ao_589_pwr,
            'aom_ao_589_pwr-units':'V',
            'green_optical_power_pd': green_optical_power_pd,
            'green_optical_power_pd-units': 'V',
            'green_optical_power_mW': green_optical_power_mW,
            'green_optical_power_mW-units': 'mW',
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
            'readout_time':readout_time,
            'readout_time_unit':'ns',
            'reionization_time': reionization_time,
            'reionization_time-units': 'ns',
            'illumination_time': illumination_time,
            'illumination_time-units': 'ns',
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'num_runs': num_runs,
            'num_reps':num_reps,
            'sig_counts': sig_counts,
            'sig_counts-units': 'counts',
            'unique_valuesNV-': unique_value,
            'unique_values-units': 'num of photons',
            'relative_frequencyNV-': relative_frequency,
            'relative_frequency-units': 'occurrences'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)

    tool_belt.save_figure(fig, file_path)

# -*- coding: utf-8 -*-
"""
Electron spin resonance routine. Scans the microwave frequency, taking counts
at each point. This version takes signal counts with a repeated ionization sub-
sequence to isolate one orientation of NVs. The reference counts are taken 
without ionizing the NVs.

Created on Wed Apr 15 15:39:23 2020

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import labrad
from utils.tool_belt import States


# %% Figure functions


def create_fit_figure(freq_range, freq_center, num_steps,
                      norm_avg_sig, fit_func, popt):

    freqs = calculate_freqs(freq_range, freq_center, num_steps)
    smooth_freqs = calculate_freqs(freq_range, freq_center, 1000)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(freqs, norm_avg_sig, 'b', label='data')
    ax.plot(smooth_freqs, fit_func(smooth_freqs, *popt), 'r-', label='fit')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend()

    text = '\n'.join(('Contrast = {:.3f}',
                      'Standard deviation = {:.4f} GHz',
                      'Frequency = {:.4f} GHz'))
    if fit_func == single_gaussian_dip:
        low_text = text.format(*popt[0:3])
        high_text = None
    elif fit_func == double_gaussian_dip:
        low_text = text.format(*popt[0:3])
        high_text = text.format(*popt[3:6])

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.05, 0.15, low_text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=props)
    if high_text is not None:
        ax.text(0.55, 0.15, high_text, transform=ax.transAxes, fontsize=12,
                verticalalignment="top", bbox=props)

    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()

    return fig


# %% Functions


def calculate_freqs(freq_range, freq_center, num_steps):
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    return numpy.linspace(freq_low, freq_high, num_steps)

def gaussian(freq, constrast, sigma, center):
    return constrast * numpy.exp(-((freq-center)**2) / (2 * (sigma**2)))

def double_gaussian_dip(freq, low_constrast, low_sigma, low_center,
                        high_constrast, high_sigma, high_center):
    low_gauss = gaussian(freq, low_constrast, low_sigma, low_center)
    high_gauss = gaussian(freq, high_constrast, high_sigma, high_center)
    return 1.0 - low_gauss - high_gauss

def single_gaussian_dip(freq, constrast, sigma, center):
    return 1.0 - gaussian(freq, constrast, sigma, center)

def fit_resonance(freq_range, freq_center, num_steps,
                  norm_avg_sig, ref_counts):

    # %% Calculate freqs

    freqs = calculate_freqs(freq_range, freq_center, num_steps)

    # %% Guess the locations of the minimums

    contrast = 0.10  # Arb
    sigma = 0.005  # MHz
#    sigma = 0.010  # MHz
    fwhm = 2.355 * sigma

    # Convert to index space
    fwhm_ind = fwhm * (num_steps / freq_range)
    if fwhm_ind < 1:
        fwhm_ind = 1

    # Bit of processing
    inverted_norm_avg_sig = 1 - norm_avg_sig
    ref_std = numpy.std(ref_counts)
    rel_ref_std = ref_std / numpy.average(ref_counts)
    height = max(rel_ref_std, contrast/4)
#    height = 0.2

    # Peaks must be separated from each other by the estimated fwhm (rayleigh
    # criteria), have a contrast of at least the noise or 5% (whichever is
    # greater), and have a width of at least two points
    peak_inds, details = find_peaks(inverted_norm_avg_sig, distance=fwhm_ind,
                                    height=height, width=2)
    peak_inds = peak_inds.tolist()
    peak_heights = details['peak_heights'].tolist()

    if len(peak_inds) > 1:
        # Find the location of the highest peak
        max_peak_height = max(peak_heights)
        max_peak_peak_inds = peak_heights.index(max_peak_height)
        max_peak_freqs = peak_inds[max_peak_peak_inds]

        # Remove what we just found so we can find the second highest peak
        peak_inds.pop(max_peak_peak_inds)
        peak_heights.pop(max_peak_peak_inds)

        # Find the location of the next highest peak
        next_max_peak_height = max(peak_heights)
        next_max_peak_peak_inds = peak_heights.index(next_max_peak_height)  # Index in peak_inds
        next_max_peak_freqs = peak_inds[next_max_peak_peak_inds]  # Index in freqs

        # List of higest peak then next highest peak
        peaks = [max_peak_freqs, next_max_peak_freqs]

        # Only keep the smaller peak if it's > 1/3 the height of the larger peak
        if next_max_peak_height > max_peak_height / 3:
            # Sort by frequency
            peaks.sort()
            low_freq_guess = freqs[peaks[0]]
            high_freq_guess = freqs[peaks[1]]
        else:
            low_freq_guess = freqs[peaks[0]]
            high_freq_guess = None

    elif len(peak_inds) == 1:
        low_freq_guess = freqs[peak_inds[0]]
        high_freq_guess = None
    else:
        print('Could not locate peaks')
        return None, None
    
#    low_freq_guess = 2.832
#    high_freq_guess = 2.849

    # %% Fit!

    if high_freq_guess is None:
        fit_func = single_gaussian_dip
        guess_params = [contrast, sigma, low_freq_guess]
    else:
        fit_func = double_gaussian_dip
        guess_params=[contrast, sigma, low_freq_guess,
                      contrast, sigma, high_freq_guess]

    try:
        popt, pcov = curve_fit(fit_func, freqs, norm_avg_sig, p0=guess_params)
    except Exception:
        print('Something went wrong!')
        popt = guess_params

    # Return the resonant frequencies
    return fit_func, popt

def simulate(res_freq, freq_range, contrast,
             rabi_period, uwave_pulse_dur):

    rabi_freq = rabi_period**-1

    smooth_freqs = calculate_freqs(freq_range, res_freq, 1000)

    omega = numpy.sqrt((smooth_freqs-res_freq)**2 + rabi_freq**2)
    amp = (rabi_freq / omega)**2
    angle = omega * 2 * numpy.pi * uwave_pulse_dur / 2
    prob = amp * (numpy.sin(angle))**2

    rel_counts = 1.0 - (contrast * prob)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(smooth_freqs, rel_counts)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Contrast (arb. units)')


# %% User functions


def state(nv_sig, apd_indices, state, freq_range,
          num_steps, num_reps, num_runs):

    freq_center = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    uwave_pulse_dur = nv_sig['rabi_{}'.format(state.name)] // 2

    resonance_list = main(nv_sig, apd_indices, freq_center, freq_range,
         num_steps, num_reps, num_runs, uwave_power, uwave_pulse_dur, state)

    return resonance_list

# %% Main


def main(nv_sig, apd_indices, freq_center, freq_range,
         num_steps, num_reps, num_runs, test_uwave_power, test_uwave_pulse_dur,
         test_state=States.HIGH):

    with labrad.connect() as cxn:
        resonance_list = main_with_cxn(cxn, nv_sig, apd_indices, freq_center, freq_range,
                  num_steps, num_reps, num_runs, test_uwave_power, test_uwave_pulse_dur,
                  test_state)
    return resonance_list
def main_with_cxn(cxn, nv_sig, apd_indices, freq_center, freq_range,
              num_steps, num_reps, num_runs, test_uwave_power, test_uwave_pulse_dur,
              test_state=States.HIGH):

    # %% Initial calculations and setup

    tool_belt.reset_cfm(cxn)

    # Calculate the frequencies we need to set
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    freqs = numpy.linspace(freq_low, freq_high, num_steps)

    # define the target state to be opposite of the test state, so they use 
    # different sig gens
    if test_state == States.HIGH:
        target_state = States.LOW
    elif test_state == States.LOW:
        target_state = States.HIGH
    
    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    ref_counts = numpy.empty([num_runs, num_steps])
    ref_counts[:] = numpy.nan
    sig_counts = numpy.copy(ref_counts)

    # Define some times for the sequence (in ns)
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    # info from the nv_sig
    num_ionizations = nv_sig['ionization_rep']
    yellow_pol_time =  nv_sig['yellow_pol_dur']
    yellow_pol_pwr = nv_sig['am_589_pol_power']
    shelf_time = nv_sig['pulsed_shelf_dur']
    shelf_pwr = nv_sig['am_589_shelf_power']
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    init_ion_time = nv_sig['pulsed_initial_ion_dur']
    ionization_time = nv_sig['pulsed_ionization_dur']
    reionization_time = nv_sig['pulsed_reionization_dur']
 
    # infor from shared params
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    rf_delay = shared_params['uwave_delay']   
    wait_time = shared_params['post_polarization_wait_dur']

    # target uwave info
    target_uwave_freq = nv_sig['resonance_{}'.format(target_state.name)]
    target_uwave_power = nv_sig['uwave_power_{}'.format(target_state.name)]
    rabi_period = nv_sig['rabi_{}'.format(target_state.name)]
    target_uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    
    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    readout_sec = readout_time / (10**9)
    seq_args = [test_uwave_pulse_dur, readout_time, yellow_pol_time, 
            shelf_time, init_ion_time, reionization_time, ionization_time, 
            target_uwave_pi_pulse, wait_time, num_ionizations, 
            laser_515_delay, aom_589_delay, laser_638_delay, rf_delay, 
            apd_indices[0], aom_ao_589_pwr, yellow_pol_pwr, shelf_pwr, 
            target_state.value, test_state.value]
    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)

    opti_coords_list = []

    # %% Turn on the taget sig generator

    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, target_state)
    sig_gen_cxn.set_freq(target_uwave_freq)
    sig_gen_cxn.set_amp(target_uwave_power)
    sig_gen_cxn.uwave_on()
    
    # %% Get the starting time of the function

    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):
        print('Run index: {}'. format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize and save the coords we found
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable = True)
        opti_coords_list.append(opti_coords)

        # Load the pulse streamer (must happen after optimize since optimize
        # loads its own sequence)
        cxn.pulse_streamer.stream_load('pulsed_resonance_isolate_orientation.py', seq_args_string)

        # Start the tagger stream
        cxn.apd_tagger.start_tag_stream(apd_indices)

        # Take a sample and increment the frequency
        for step_ind in range(num_steps):

            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break

            # Just assume the low state
            sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, test_state)
            sig_gen_cxn.set_freq(freqs[step_ind])
            sig_gen_cxn.set_amp(test_uwave_power)
            sig_gen_cxn.uwave_on()

            # It takes 400 us from receipt of the command to
            # switch frequencies so allow 1 ms total
#            time.sleep(0.001)

            # Start the timing stream
            cxn.pulse_streamer.stream_start(num_reps)

            # Get the counts
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)

            sample_counts = new_counts[0]

            # signal counts are even - get every second element starting from 0
            sig_gate_counts = sample_counts[0::2]
            sig_counts[run_ind, step_ind] = sum(sig_gate_counts)

            # ref counts are odd - sample_counts every second element starting from 1
            ref_gate_counts = sample_counts[1::2]
            ref_counts[run_ind, step_ind] = sum(ref_gate_counts)

        cxn.apd_tagger.stop_tag_stream()

        # %% Save the data we have incrementally for long measurements

        rawData = {'start_timestamp': start_timestamp,
                   'nv_sig': nv_sig,
                   'nv_sig-units': tool_belt.get_nv_sig_units(),
                   'freq_center': freq_center,
                   'freq_center-units': 'GHz',
                   'freq_range': freq_range,
                   'freq_range-units': 'GHz',
                   'test_uwave_pulse_dur': test_uwave_pulse_dur,
                   'test_uwave_pulse_dur-units': 'ns',
                   'test_state': test_state.name,
                   'target_state': target_state.name,
                   'num_steps': num_steps,
                   'run_ind': run_ind,
                   'test_uwave_power': test_uwave_power,
                   'test_uwave_power-units': 'dBm',
                   'opti_coords_list': opti_coords_list,
                   'opti_coords_list-units': 'V',
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

    # Find the averages across runs
    avg_ref_counts = numpy.average(ref_counts, axis=0)
    avg_sig_counts = numpy.average(sig_counts, axis=0)
    norm_avg_sig = avg_sig_counts / avg_ref_counts

    # Convert to kilocounts per second

    kcps_uwave_off_avg = (avg_ref_counts / (num_reps * 1000)) / readout_sec
    kcpsc_uwave_on_avg = (avg_sig_counts / (num_reps * 1000)) / readout_sec

    # Create an image with 2 plots on one row, with a specified size
    # Then draw the canvas and flush all the previous plots from the canvas
    fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

    # The first plot will display both the uwave_off and uwave_off counts
    ax = axes_pack[0]
    ax.plot(freqs, kcps_uwave_off_avg, 'r-', label = 'Reference (w/out ionization)')
    ax.plot(freqs, kcpsc_uwave_on_avg, 'g-', label = 'Signal (w/ ionization)')
    ax.set_title('Non-normalized Count Rate Versus Frequency')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Count rate (kcps)')
    ax.legend()
    # The second plot will show their subtracted values
    ax = axes_pack[1]
    ax.plot(freqs, norm_avg_sig, 'b-')
    ax.set_title('Normalized Count Rate vs Frequency')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Contrast (arb. units)')

    fig.canvas.draw()
    fig.tight_layout()
    fig.canvas.flush_events()

    # %% Fit the data

    fit_func, popt = fit_resonance(freq_range, freq_center, num_steps,
                                   norm_avg_sig, ref_counts)
    if (fit_func is not None) and (popt is not None):
        fit_fig = create_fit_figure(freq_range, freq_center, num_steps,
                                    norm_avg_sig, fit_func, popt)
    else:
        fit_fig = None

    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'opti_coords_list': opti_coords_list,
               'opti_coords_list-units': 'V',
               'freq_center': freq_center,
               'freq_center-units': 'GHz',
               'freq_range': freq_range,
               'freq_range-units': 'GHz',
               'test_uwave_pulse_dur': test_uwave_pulse_dur,
               'test_uwave_pulse_dur-units': 'ns',
               'test_state': test_state.name,
               'target_state': target_state.name,
               'num_steps': num_steps,
               'num_reps': num_reps,
               'num_runs': num_runs,
               'test_uwave_power': test_uwave_power,
               'test_uwave_power-units': 'dBm',
               'sig_counts': sig_counts.astype(int).tolist(),
               'sig_counts-units': 'counts',
               'ref_counts': ref_counts.astype(int).tolist(),
               'ref_counts-units': 'counts',
               'norm_avg_sig': norm_avg_sig.astype(float).tolist(),
               'norm_avg_sig-units': 'arb'}

    name = nv_sig['name']
    filePath = tool_belt.get_file_path(__file__, timestamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
    filePath = tool_belt.get_file_path(__file__, timestamp, name + '-fit')
    if fit_fig is not None:
        tool_belt.save_figure(fit_fig, filePath)

    # %% Return

    if fit_func == single_gaussian_dip:
        print('Single resonance at {:.4f} GHz'.format(popt[2]))
        print('\n')
        return popt[2], None
    elif fit_func == double_gaussian_dip:
        print('Resonances at {:.4f} GHz and {:.4f} GHz'.format(popt[2], popt[5]))
        print('Splitting of {:d} MHz'.format(int((popt[5] - popt[2]) * 1000)))
        print('\n')
        return popt[2], popt[5]
    else:
        print('No resonances found')
        print('\n')
        return None, None


# %% Run the file


if __name__ == '__main__':

#    file = '2019_10/2019_10_13-23_44_37-ayrton12-NV13_2019_06_10'
#    data = tool_belt.get_raw_data('pulsed_resonance.py', file)
##    data = tool_belt.get_raw_data('resonance.py', file)
#
#    freq_center = data['freq_center']
#    freq_range = data['freq_range']
#    num_steps = data['num_steps']
#    norm_avg_sig = numpy.array(data['norm_avg_sig'])
#    ref_counts = numpy.array(data['ref_counts'])
#    sig_counts = numpy.array(data['sig_counts'])
#    avg_ref_counts = numpy.average(ref_counts, axis=0)
#    avg_sig_counts = numpy.average(sig_counts, axis=0)
#    norm_avg_sig = avg_sig_counts / avg_ref_counts
#
#
#    fit_func, popt = fit_resonance(freq_range, freq_center, num_steps,
#                                   norm_avg_sig, ref_counts)
##    if (fit_func is not None) and (popt is not None):
#    create_fit_figure(freq_range, freq_center, num_steps,
#                      norm_avg_sig, fit_func, popt)
    
    # res_freq, freq_range, contrast, rabi_period, uwave_pulse_dur
#    simulate(2.87, 0.2, 0.1, 100, 50)
    
    apd_indices = [0]
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0.5',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.3, 
            'yellow_pol_dur': 2*10**3, 'am_589_pol_power': 0.3,
            'pulsed_initial_ion_dur': 200*10**3,
            'pulsed_shelf_dur': 50, 'am_589_shelf_power': 0.3,
            'pulsed_ionization_dur': 450, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 200*10**3, 'cobalt_532_power': 8,
            'ionization_rep': 13,
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 187.8, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}   
    nv_sig = ensemble
    
    main(nv_sig, apd_indices, 2.87, 0.1,
         51, 10**4, 1, 10.0, 125,
         test_state=States.HIGH)

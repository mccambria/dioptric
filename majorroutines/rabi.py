# -*- coding: utf-8 -*-
"""
Rabi flopping routine. Sweeps the pulse duration of a fixed uwave frequency.

Created on Tue Apr 23 11:49:23 2019

@author: mccambria
"""


# %% Imports


import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import utils.positioning as positioning
import numpy
import os
import time
import matplotlib.pyplot as plt
from random import shuffle
from scipy.optimize import curve_fit
import labrad
import majorroutines.optimize as optimize
from utils.tool_belt import NormStyle
from utils.tool_belt import States



# %% Functions

def fit_data(uwave_time_range, num_steps, fit_func, norm_avg_sig, norm_avg_sig_ste = None):

    #  Set up

    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus, tau_step = numpy.linspace(min_uwave_time, max_uwave_time,
                            num=num_steps, dtype=numpy.int32, retstep=True)

    # fit_func = tool_belt.cosexp_1_at_0

    #  Estimated fit parameters

    offset = numpy.average(norm_avg_sig)
    decay = 1000

    # To estimate the frequency let's find the highest peak in the FFT
    transform = numpy.fft.rfft(norm_avg_sig)
    freqs = numpy.fft.rfftfreq(num_steps, d=tau_step)
    transform_mag = numpy.absolute(transform)
    # [1:] excludes frequency 0 (DC component)
    max_ind = numpy.argmax(transform_mag[1:])
    frequency = freqs[max_ind + 1]

    #  Fit

    init_params = [offset, frequency, decay]

    try:
        popt, pcov = curve_fit(fit_func, taus, norm_avg_sig,
                            p0=init_params,
                            sigma=norm_avg_sig_ste,
                            absolute_sigma=True)
    except Exception as e:
        print(e)
        popt = None
        pcov = None

    return fit_func, popt, pcov

def create_cos_fit_figure(uwave_time_range, num_steps, uwave_freq, norm_avg_sig,
                      norm_avg_sig_ste, fit_func=None, popt=None):

    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus, tau_step = numpy.linspace(min_uwave_time, max_uwave_time,
                            num=num_steps, dtype=numpy.int32, retstep=True)
    smooth_taus = numpy.linspace(min_uwave_time, max_uwave_time, num=1000)
    fig = None
    # Fitting
    if (fit_func is None) or (popt is None):
        fit_func, popt, pcov = fit_data(
            uwave_time_range,
            num_steps,
            fit_func,
            norm_avg_sig,
            norm_avg_sig_ste,
        )
    if (popt is not None):
        # Plot setup
        fig, ax = plt.subplots()
        ax.set_xlabel('Microwave duration (ns)')
        ax.set_ylabel("Normalized fluorescence")
        ax.set_title('Rabi Oscillation Of NV Center Electron Spin')
        
        # Plotting
        if norm_avg_sig_ste is not None:
            kpl.plot_points(ax, taus, norm_avg_sig, yerr=norm_avg_sig_ste)
        else:
            kpl.plot_line(ax, taus, norm_avg_sig)
        kpl.plot_line(
            ax,
            smooth_taus,
            fit_func(smooth_taus, *popt),
            color=KplColors.RED,
        )
        Amp = 1- popt[0]
        base_text = "Offset = {:.3f} \nAmp = {:.3f} \n1/v = {:.1f} ns \nd = {:.1f} ns"
        size = kpl.Size.SMALL
        text = base_text.format(popt[0], Amp, 1/popt[1], popt[2])
        kpl.anchored_text(ax, text, kpl.Loc.LOWER_LEFT, size=size)
    
    return fig, ax, fit_func, popt, pcov


from scipy.optimize import fsolve

def solve_linear(m, b, y, z):
   x = z[0]

   F = numpy.empty((1))
   
   F[0] = m*x + b - y
   return F

def create_piecewise_fit_figure(uwave_time_range, num_steps, uwave_freq, norm_avg_sig,
                      norm_avg_sig_ste, fit_func=None, popt=None):

    # instead of fitting to a decay cosine, this function fits the initial slope
    # to a linear line to find the pi/2 pulse duration and a quadratic to 
    # the first dip to find the pi pulse duration.
    
    # this should accurately find the pi/2 and pi pulse times, which are not always 
    # even fractions of the total period (possibly due to rise times on the MW switches)
        
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus, tau_step = numpy.linspace(min_uwave_time, max_uwave_time,
                            num=num_steps, dtype=numpy.int32, retstep=True)
    smooth_taus = numpy.linspace(min_uwave_time, max_uwave_time, num=1000)
    fig = None
    # Fitting
    if (fit_func is None) or (popt is None):
        fit_func, popt, pcov = fit_data(
            uwave_time_range,
            num_steps,
            fit_func,
            norm_avg_sig,
            norm_avg_sig_ste,
        )
    if (popt is not None):
        # Plot setup
        fig, ax = plt.subplots()
        ax.set_xlabel('Microwave duration (ns)')
        ax.set_ylabel("Normalized fluorescence")
        ax.set_title('Rabi Oscillation Of NV Center Electron Spin')
        
        # Plotting
        if norm_avg_sig_ste is not None:
            kpl.plot_points(ax, taus, norm_avg_sig, yerr=norm_avg_sig_ste)
        else:
            kpl.plot_line(ax, taus, norm_avg_sig)
            
        # kpl.plot_line(
        #     ax,
        #     smooth_taus,
        #     fit_func(smooth_taus, *popt),
        #     color=KplColors.RED,
        # )
        
        ax.axhline(popt[0])
        
        #prep for fitting
        period = 1/popt[1]
        print(period)
        rabi_offset = popt[0]
        Amp = 1- popt[0]
        q=period/4# + period/25
        q_ind=q/tau_step
        q_range = 0.1 *period/tau_step
        q_ind_low = int(numpy.round(q_ind-q_range)) 
        q_ind_high = int(numpy.round(q_ind+q_range) )
        
        h=period/2
        h_ind=h/tau_step
        h_range = 0.2 *period/tau_step
        h_ind_low = int(numpy.round(h_ind-h_range) )
        h_ind_high = int(numpy.round(h_ind+h_range) )
        
        ### fit linear line to initial slope
        fit_func_q = tool_belt.linear
        init_params = [-Amp/(h), rabi_offset]
        popt_q, pcov = curve_fit(fit_func_q, taus[q_ind_low : q_ind_high], norm_avg_sig[q_ind_low : q_ind_high],
                            p0=init_params,
                            sigma=norm_avg_sig_ste[q_ind_low : q_ind_high],
                            absolute_sigma=True)
        
        # find intersection of linear line and offset (half of full range)        
        solve_linear_func = lambda z: solve_linear(popt_q[0], popt_q[1], rabi_offset, z)
        zGuess = numpy.array([q])
        solve= fsolve(solve_linear_func,zGuess)
        pi_on_2_pulse =  solve[0]
        # print(pi_on_2_pulse)
        
        # plot linear line
        smooth_taus_q = numpy.linspace(taus[q_ind_low], taus[q_ind_high], num=1000)
        kpl.plot_line(
            ax,
            smooth_taus_q,
            fit_func_q(smooth_taus_q, *popt_q),
            color=KplColors.GREEN,
        )
        
        
        ### fit quadratic to initial slope
        fit_func_h = lambda x, a, c, x_offset: tool_belt.quadratic(x, a, 0, c, x_offset)
        init_params = [1e-5, 1-2*Amp, h]
        popt_h, pcov = curve_fit(fit_func_h, taus[h_ind_low : h_ind_high], norm_avg_sig[h_ind_low : h_ind_high],
                            p0=init_params,
                            sigma=norm_avg_sig_ste[h_ind_low : h_ind_high],
                            absolute_sigma=True)
        
        # find when quadratic is min (the x_offset param)        
        pi_pulse =  popt_h[2]
        # print(popt_h)
        
        # plot quadratic line
        smooth_taus_h = numpy.linspace(taus[h_ind_low], taus[h_ind_high], num=1000)
        kpl.plot_line(
            ax,
            smooth_taus_h,
            fit_func_h(smooth_taus_h, *popt_h),
            color=KplColors.RED,
        )
        Amp = 1- popt[0]
        base_text = "pi pulse = {:.2f} ns \npi_on_2 pulse = {:.2f} ns"
        size = kpl.Size.SMALL
        text = base_text.format(pi_pulse, pi_on_2_pulse)
        kpl.anchored_text(ax, text, kpl.Loc.LOWER_LEFT, size=size)
        
        # print(popt[0])
        # ax.axhline(1-(1-popt[0])*2)
    
    return fig, ax, fit_func, popt, pcov, pi_pulse,pi_on_2_pulse 

def create_raw_data_figure(
    taus,
    avg_sig_counts=None,
    avg_ref_counts=None,
    norm_avg_sig=None,
):
    num_steps = len(taus)
    # Plot setup
    fig, axes_pack = plt.subplots(1, 2, figsize=kpl.double_figsize)
    ax_sig_ref, ax_norm = axes_pack
    ax_sig_ref.set_xlabel('Microwave duration (ns)')
    ax_sig_ref.set_ylabel("Count rate (kcps)")
    ax_norm.set_xlabel('Microwave duration (ns)')
    ax_norm.set_ylabel("Normalized fluorescence")

    # Plotting
    if avg_sig_counts is None:
        avg_sig_counts = numpy.empty(num_steps)
        avg_sig_counts[:] = numpy.nan
    kpl.plot_line(
        ax_sig_ref, taus, avg_sig_counts, label="Signal", color=KplColors.GREEN
    )
    if avg_ref_counts is None:
        avg_ref_counts = numpy.empty(num_steps)
        avg_ref_counts[:] = numpy.nan
    kpl.plot_line(
        ax_sig_ref, taus, avg_ref_counts, label="Reference", color=KplColors.RED
    )
    ax_sig_ref.legend(loc=kpl.Loc.LOWER_RIGHT)
    if norm_avg_sig is None:
        norm_avg_sig = numpy.empty(num_steps)
        norm_avg_sig[:] = numpy.nan
    kpl.plot_line(ax_norm, taus, norm_avg_sig, color=KplColors.BLUE)

    return fig, ax_sig_ref, ax_norm


def simulate(uwave_time_range, freq, resonant_freq, contrast,
             measured_rabi_period=None, resonant_rabi_period=None):

    if measured_rabi_period is None:
        resonant_rabi_freq = resonant_rabi_period**-1
        res_dev = freq - resonant_freq
        measured_rabi_freq = numpy.sqrt(res_dev**2 + resonant_rabi_freq**2)
        measured_rabi_period = measured_rabi_freq**-1
        print('measured_rabi_period: {} ns'.format(measured_rabi_period))
    elif resonant_rabi_period is None:
        measured_rabi_freq = measured_rabi_period**-1
        res_dev = freq-resonant_freq
        resonant_rabi_freq = numpy.sqrt(measured_rabi_freq**2 - res_dev**2)
        resonant_rabi_period = resonant_rabi_freq**-1
        print('resonant_rabi_period: {} ns'.format(resonant_rabi_period))
    else:
        raise RuntimeError('Pass either a measured_rabi_period or a ' \
                           'resonant_rabi_period, not both/neither.')

    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    smooth_taus = numpy.linspace(min_uwave_time, max_uwave_time,
                          num=1000, dtype=numpy.int32)
    amp = (resonant_rabi_freq / measured_rabi_freq)**2
    angle = measured_rabi_freq * 2 * numpy.pi * smooth_taus / 2
    prob = amp * (numpy.sin(angle))**2

    rel_counts = 1.0 - (contrast * prob)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(smooth_taus, rel_counts)
    ax.set_xlabel('Tau (ns)')
    ax.set_ylabel('Contrast (arb. units)')

def hyperfine_rabi_func(t, offset, freq, decay):
    two_pi = 2 * numpy.pi
    amp = offset - 1
    v= 2.2e-3
    omega_pm= numpy.sqrt((v)**2 + freq**2)
    omega_0= freq
    term_0 = numpy.cos((two_pi * omega_0 * t)) 
    term_pm = numpy.cos((two_pi * omega_pm * t))
    
    return offset + (numpy.exp(-t / abs(decay)) * abs(amp/3)* ( term_0  +2*term_pm) )




# %% Main


def main(nv_sig, uwave_time_range, state,
         num_steps, num_reps, num_runs,
         opti_nv_sig = None,
         return_popt=False,
         do_scc = False,
         do_dq = False,
         do_cos_fit = True):

    with labrad.connect() as cxn:
        rabi_per, sig_counts, ref_counts, popt = main_with_cxn(cxn, nv_sig,
                                         uwave_time_range, state,
                                         num_steps, num_reps, num_runs,
                                         opti_nv_sig,do_scc, do_dq, do_cos_fit)

        if return_popt:
            return rabi_per, popt
        if not return_popt:
            return rabi_per


def main_with_cxn(cxn, nv_sig,  uwave_time_range, state,
                  num_steps, num_reps, num_runs,
                  opti_nv_sig = None,
                  do_scc = False,
                  do_dq = False,
                  do_cos_fit = True):
    do_iq = False 
    
    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    arbwavegen_server = tool_belt.get_server_arb_wave_gen(cxn)

    tool_belt.reset_cfm(cxn)
    kpl.init_kplotlib()

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Initial calculations and setup

    uwave_freq = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]

    laser_key = 'spin_laser'
    laser_name = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    
    if do_dq:
        rabi_period_low = nv_sig["rabi_{}".format(States.LOW.name)]
        uwave_pi_pulse_low = tool_belt.get_pi_pulse_dur(rabi_period_low)
        uwave_freq_low = nv_sig["resonance_{}".format(States.LOW.name)]
        uwave_power_low = nv_sig["uwave_power_{}".format(States.LOW.name)]
        
        rabi_period_high = nv_sig["rabi_{}".format(States.HIGH.name)]
        uwave_pi_pulse_high = tool_belt.get_pi_pulse_dur(rabi_period_high)
        uwave_freq_high = nv_sig["resonance_{}".format(States.HIGH.name)]
        uwave_power_high = nv_sig["uwave_power_{}".format(States.HIGH.name)]
        
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
            polarization_time = nv_sig['spin_pol_dur']
            readout = nv_sig['spin_readout_dur']
            readout_sec = readout / (10**9)
        
        
    norm_style = nv_sig["norm_style"]

    # Array of times to sweep through
    # Must be ints since the pulse streamer only works with int64s
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(min_uwave_time, max_uwave_time,
                          num=num_steps, dtype=numpy.int32)


    # Analyze the sequence
    num_reps = int(num_reps)
    if do_dq:
        if do_scc:
            file_name = 'rabi_dq_scc.py'    
            seq_args = [
                    taus[0],
                    readout,
                    polarization_dur,
                    ionization_dur,
                    uwave_pi_pulse_low,
                    uwave_pi_pulse_high,
                    max_uwave_time,
                    state.value,
                    pol_laser_name,
                    readout_laser_name,
                    ion_laser_name,
                    pol_laser_power,
                    ion_laser_power,
                    readout_laser_power,]
        else:
                
            file_name = 'rabi_dq.py'    
            seq_args = [taus[0], polarization_time,
                        readout,  uwave_pi_pulse_low, uwave_pi_pulse_high, max_uwave_time,
                        state.value, laser_name, laser_power]

    else:    
        if do_scc:
            file_name = 'rabi_scc.py'
            seq_args = [
                    readout,
                    polarization_dur,
                    ionization_dur,
                    taus[0],
                    max_uwave_time,
                    pol_laser_name,
                    readout_laser_name,
                    ion_laser_name,
                    state.value, 
                    pol_laser_power,
                    ion_laser_power,
                    readout_laser_power,]
        else:
            file_name = 'rabi.py'
            seq_args = [taus[0], polarization_time,
                        readout, max_uwave_time,
                        state.value, laser_name, laser_power]
#    for arg in seq_args:
#        print(type(arg))
    # print(file_name)
    # print(seq_args)
    # return
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    pulsegen_server.stream_load(file_name, seq_args_string)

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    sig_counts = numpy.empty([num_runs, num_steps], dtype=numpy.float32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    # norm_avg_sig = numpy.empty([num_runs, num_steps])

    # %% Make some lists and variables to save at the end

    opti_coords_list = []
    tau_index_master_list = [[] for i in range(num_runs)]

    # Create a list of indices to step through the taus. This will be shuffled
    tau_ind_list = list(range(0, num_steps))

    # Create raw data figure for incremental plotting
    raw_fig, ax_sig_ref, ax_norm = create_raw_data_figure(
        taus
    )
    # Set up a run indicator for incremental plotting
    run_indicator_text = "Run #{}/{}"
    text = run_indicator_text.format(0, num_runs)
    run_indicator_obj = kpl.anchored_text(ax_norm, text, loc=kpl.Loc.UPPER_RIGHT)

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()
    for run_ind in range(num_runs):

        print('Run index: {}'. format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize and save the coords we found
        if opti_nv_sig:
            opti_coords = optimize.main_with_cxn(cxn, opti_nv_sig)
            drift = positioning.get_drift(cxn)
            adj_coords = nv_sig['coords'] + numpy.array(drift)
            positioning.set_xyz(cxn, adj_coords)
        else:
            opti_coords = optimize.main_with_cxn(cxn, nv_sig)
        opti_coords_list.append(opti_coords)

        tool_belt.set_filter(cxn, nv_sig, "spin_laser")
        laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

        # Apply the microwaves
        phase = 0
        if do_dq:
            sig_gen_low_cxn = tool_belt.get_server_sig_gen(cxn, States.LOW)
            sig_gen_low_cxn.set_freq(uwave_freq_low)
            sig_gen_low_cxn.set_amp(uwave_power_low)
            sig_gen_low_cxn.uwave_on()
            sig_gen_high_cxn = tool_belt.get_server_sig_gen(cxn, States.HIGH)
            sig_gen_high_cxn.set_freq(uwave_freq_high)
            sig_gen_high_cxn.set_amp(uwave_power_high)
            if do_iq:
                sig_gen_high_cxn.load_iq()
                phase = numpy.pi/2
                arbwavegen_server.load_arb_phases([phase])
            sig_gen_high_cxn.uwave_on()
            
        else:
            sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)
            sig_gen_cxn.set_freq(uwave_freq)
            sig_gen_cxn.set_amp(uwave_power)
            if do_iq:
                sig_gen_cxn.load_iq()
                arbwavegen_server.load_arb_phases([0])
            sig_gen_cxn.uwave_on()
        
        
        
        if do_scc:
            charge_readout_laser_server = tool_belt.get_server_charge_readout_laser(cxn)
            charge_readout_laser_server.load_feedthrough(nv_sig["charge_readout_laser_power"])

        # Load the APD
        counter_server.start_tag_stream()

        # Shuffle the list of indices to use for stepping through the taus
        shuffle(tau_ind_list)

        # start_time = time.time()
        for tau_ind in tau_ind_list:
#        for tau_ind in range(len(taus)):
            # print('Tau: {} ns'. format(taus[tau_ind]))
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
#            print(taus[tau_ind])
            # add the tau indexxes used to a list to save at the end
            tau_index_master_list[run_ind].append(tau_ind)
            # Stream the sequence
            
            if do_dq:
                if do_scc:    
                    seq_args = [
                            taus[tau_ind],
                            readout,
                            polarization_dur,
                            ionization_dur,
                            uwave_pi_pulse_low,
                            uwave_pi_pulse_high,
                            max_uwave_time,
                            state.value,
                            pol_laser_name,
                            readout_laser_name,
                            ion_laser_name,
                            pol_laser_power,
                            ion_laser_power,
                            readout_laser_power,]
                else:
                    seq_args = [taus[tau_ind], polarization_time,
                                readout,  uwave_pi_pulse_low, uwave_pi_pulse_high, max_uwave_time,
                                state.value, laser_name, laser_power]
            else:
                if do_scc:
                    seq_args = [
                            readout,
                            polarization_dur,
                            ionization_dur,
                            taus[tau_ind],
                            max_uwave_time,
                            pol_laser_name,
                            readout_laser_name,
                            ion_laser_name,
                            state.value, 
                            pol_laser_power,
                            ion_laser_power,
                            readout_laser_power,]
                else:
                    seq_args = [taus[tau_ind], polarization_time,
                                readout, max_uwave_time,
                                state.value, laser_name, laser_power]
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # print(seq_args)
            # Clear the tagger buffer of any excess counts
            counter_server.clear_buffer()

            # start_time = time.time()
            counter_server.clear_buffer()
            pulsegen_server.stream_immediate(file_name, num_reps,
                                             seq_args_string)
            new_counts = counter_server.read_counter_modulo_gates(2, 1)
            # print(new_counts)
            sample_counts = new_counts[0]
            if do_dq:
                ref_counts[run_ind, tau_ind] = sample_counts[0]
                sig_counts[run_ind, tau_ind] = sample_counts[1]
            else:
                sig_counts[run_ind, tau_ind] = sample_counts[0]
                ref_counts[run_ind, tau_ind] = sample_counts[1]
            
            


#            run_time = time.time()
#            run_elapsed_time = run_time - start_time
#            start_time = run_time
#            print('Tau: {} ns'.format(taus[tau_ind]))
#            print('Elapsed time {}'.format(run_elapsed_time))

        counter_server.stop_tag_stream()

        ### Incremental plotting

        # Update the run indicator
        text = run_indicator_text.format(run_ind + 1, num_runs)
        run_indicator_obj.txt.set_text(text)

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

        kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
        kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
        kpl.plot_line_update(ax_norm, y=norm_avg_sig)


        # %% Save the data we have incrementally for long measurements

        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    # 'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'do_scc': do_scc,
                    'do_dq': do_dq,
                    'phase':phase,
                    'uwave_freq': uwave_freq,
                    'uwave_freq-units': 'GHz',
                    'uwave_power': uwave_power,
                    'uwave_power-units': 'dBm',
                    'uwave_time_range': uwave_time_range,
                    'uwave_time_range-units': 'ns',
                    'state': state.name,
                    'num_steps': num_steps,
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'tau_index_master_list':tau_index_master_list,
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
        tool_belt.save_raw_data(raw_data, file_path)
        tool_belt.save_figure(raw_fig, file_path)

   
    ### Process and plot the data

    ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, norm_style)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals

    # Raw data
    kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
    kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
    kpl.plot_line_update(ax_norm, y=norm_avg_sig)
    run_indicator_obj.remove()


    #  Plot the data itself and the fitted curve
    if do_cos_fit:
        if do_scc:
            fit_func = tool_belt.inverted_cosexp
        else:
            fit_func = tool_belt.cosexp_1_at_0
        fit_fig, ax, fit_func, popt, pcov = create_cos_fit_figure(
            uwave_time_range, num_steps, uwave_freq, norm_avg_sig, norm_avg_sig_ste,
            fit_func 
        )
        rabi_period = 1/popt[1]
        v_unc = numpy.sqrt(pcov[1][1])
        print(v_unc)
        rabi_period_unc = rabi_period**2 * v_unc
        print('Rabi period measured: {} +/- {} ns\n'.format('%.2f'%rabi_period, '%.2f'%rabi_period_unc))
    else:
        fit_func = tool_belt.cosexp_1_at_0
        fit_fig, ax, fit_func, popt, pcov, pi_pulse,pi_on_2_pulse  = create_piecewise_fit_figure(
            uwave_time_range, num_steps, uwave_freq, norm_avg_sig, norm_avg_sig_ste,
            fit_func 
        )
        print('pi pulse: {}  ns\n'.format('%.2f'%pi_pulse))
        print('pi_on_2 pulse: {}  ns\n'.format('%.2f'%pi_on_2_pulse))
        rabi_period = 2*pi_pulse
        

    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
                'timeElapsed-units': 's',
                'nv_sig': nv_sig,
                # 'nv_sig-units': tool_belt.get_nv_sig_units(),
                'do_scc': do_scc,
                'do_dq': do_dq,
                'phase':phase,
                'uwave_freq': uwave_freq,
                'uwave_freq-units': 'GHz',
                'uwave_power': uwave_power,
                'uwave_power-units': 'dBm',
                'uwave_time_range': uwave_time_range,
                'uwave_time_range-units': 'ns',
                'state': state.name,
                'num_steps': num_steps,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'tau_index_master_list':tau_index_master_list,
                'opti_coords_list': opti_coords_list,
                'opti_coords_list-units': 'V',
                'sig_counts': sig_counts.astype(int).tolist(),
                'sig_counts-units': 'counts',
                'ref_counts': ref_counts.astype(int).tolist(),
                'ref_counts-units': 'counts',
                'norm_avg_sig': norm_avg_sig.astype(float).tolist(),
                'norm_avg_sig-units': 'arb',
                'norm_avg_sig_ste': norm_avg_sig_ste.astype(float).tolist(),
                'norm_avg_sig_ste-units': 'arb'}

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path_fit = tool_belt.get_file_path(__file__, timestamp, nv_name + "-fit")
        tool_belt.save_figure(fit_fig, file_path_fit)
    tool_belt.save_raw_data(raw_data, file_path)

    if (fit_func is not None) and (popt is not None):
        return rabi_period, sig_counts, ref_counts, popt
    else:
        return None, sig_counts, ref_counts


# %% Run the file


if __name__ == '__main__':

    path = 'pc_rabi/branch_master/rabi/2023_01'
    file = '2023_01_27-09_42_22-siena-nv4_2023_01_16'
    data = tool_belt.get_raw_data(file, path)
    kpl.init_kplotlib()

    norm_avg_sig = data['norm_avg_sig']
    uwave_time_range = data['uwave_time_range']
    num_steps = data['num_steps']
    uwave_freq = data['uwave_freq']
    norm_avg_sig_ste =None

    fit_func = hyperfine_rabi_func
    # fit_func = tool_belt.cosexp_1_at_0
    
    
    sig_counts = data['sig_counts']
    ref_counts = data['ref_counts']
    num_reps = data['num_reps']
    nv_sig = data['nv_sig']
    readout = nv_sig['spin_readout_dur']
    ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, NormStyle.POINT_TO_POINT)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals
    
    
    create_piecewise_fit_figure(
         uwave_time_range, num_steps, uwave_freq, norm_avg_sig, norm_avg_sig_ste,
         fit_func 
     )
    
   # create_cos_fit_figure(
   #     uwave_time_range, num_steps, uwave_freq, norm_avg_sig, norm_avg_sig_ste,
   #     fit_func 
   # )


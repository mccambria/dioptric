# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: chemistatcode
"""

import sys
import time
import traceback
import multiprocessing

import time
from datetime import datetime
import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping, brute
from scipy.signal import lombscargle
from matplotlib.widgets import Button, RadioButtons

import lmfit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.tool_belt import curve_fit
from scipy.optimize import basinhopping


def quartic_decay(
    tau,
    baseline,
    quartic_contrast,
    revival_time,
    quartic_decay_time,
    T2_ms,
    T2_exp,
    osc_contrast=None,
    osc_freq1=None,
    osc_freq2=None,
):
    T2_us = 1000 * T2_ms
    envelope = np.exp(-((tau / T2_us) ** T2_exp))
    
    # Create array for revival times
    num_revivals = 3
    revival_indices = np.arange(num_revivals)
    revival_times = revival_indices * revival_time
    
    # Broadcasting to create a 2D array: (tau_points × num_revivals)
    tau_expanded = tau[:, np.newaxis]
    revival_times_expanded = revival_times[np.newaxis, :]
    
    # Calculate exponential part for all revivals simultaneously
    exp_part = np.exp(-((tau_expanded - revival_times_expanded) / quartic_decay_time) ** 4)
    
    # Sum along revival axis
    comb = np.sum(exp_part, axis=1)
    
    if osc_contrast is None:
        mod = quartic_contrast
    else:
        mod = (
            quartic_contrast
            - osc_contrast
            * np.sin(2 * np.pi * osc_freq1 * tau / 2) ** 2
            * np.sin(2 * np.pi * osc_freq2 * tau / 2) ** 2
        )
    
    val = baseline - envelope * mod * comb
    return val


def quartic_decay_fixed_revival(
    tau,
    baseline,
    quartic_contrast,
    quartic_decay_time,
    T2_ms,
    T2_exp,
    osc_contrast=None,
    osc_freq1=None,
    osc_freq2=None,
):
    return quartic_decay(
        tau,
        baseline,
        quartic_contrast,
        50,
        quartic_decay_time,
        T2_ms,
        T2_exp,
        osc_contrast,
        osc_freq1,
        osc_freq2,
    )


def constant(tau):
    norm = 1
    if isinstance(tau, list):
        return [norm] * len(tau)
    elif type(tau) == np.ndarray:
        return np.array([norm] * len(tau))
    else:
        return norm


def create_raw_data_figure(data):
    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["states"])
    sig_counts, ref_counts = counts[0], counts[1]

    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )
    # avg_counts -= norms[:, np.newaxis]

    fig, ax = plt.subplots()
    total_evolution_times = 2 * np.array(taus) / 1e3
    widefield.plot_raw_data(
        ax, nv_list, total_evolution_times, avg_counts, avg_counts_ste
    )
    ax.set_xlabel("Total evolution time (µs)")
    ax.set_ylabel("Counts")
    return fig


def brute_fit_fn_cost(
    x,
    total_evolution_times,
    nv_counts,
    nv_counts_ste,
    fit_fn,
    no_c13_popt,
    osc_contrast_guess,
):
    line = fit_fn(total_evolution_times, *no_c13_popt, osc_contrast_guess, *x)
    
    # Vectorized computation of chi-square
    residuals = nv_counts - line
    weighted_residuals = residuals / nv_counts_ste
    chi_square = np.sum(weighted_residuals ** 2)
    
    return chi_square


def fit(total_evolution_times, nv_counts, nv_counts_ste):
    import time
    timing = {}
    fit_start = time.time()

    # Keep reference to original function for plotting
    fit_fn = quartic_decay_fixed_revival  

    ### Get good guesses
    t0 = time.time()
    baseline_guess = nv_counts[9]
    quartic_contrast_guess = baseline_guess - nv_counts[0]
    log_decay = -np.log((baseline_guess - nv_counts[-6]) / quartic_contrast_guess)
    T2_guess = 0.1 * (log_decay ** (-1 / 3))
    guess_params = [baseline_guess, quartic_contrast_guess, 7, T2_guess, 3]
    bounds = [
        [0, 0, 0, 0, 0],
        [1, 1, 20, 1000, 10],
    ]
    timing['initial_guesses'] = time.time() - t0

    ### PART 1: First fit without C13 coupling
    t0 = time.time()
    
    no_c13_popt, no_c13_pcov, no_c13_red_chi_sq = curve_fit(
        fit_fn,
        total_evolution_times,
        nv_counts,
        guess_params,
        nv_counts_ste,
        bounds=bounds,
    )
    timing['first_curve_fit'] = time.time() - t0

    ### PART 2: Try with different oscillation parameters
    t0 = time.time()
    # Extract contrast for C13 parameters
    osc_contrast_guess = no_c13_popt[1] / 2

    # Prepare for oscillation fitting
    best_popt = no_c13_popt
    best_red_chi_sq = no_c13_red_chi_sq
    best_pcov = no_c13_pcov

    # Create new guess with oscillation parameters
    full_guess_params = list(no_c13_popt) + [osc_contrast_guess, 0.3, 0.1]
    full_bounds = [
        [0, 0, 0, 0, 0, -0.5, 0, 0],
        [1, 1, 20, 1000, 10, 0.5, 1.5, 0.5]
    ]

    # Try different frequency combinations
    freq_combinations = [
        (0.1, 0.1), (0.3, 0.1), (0.5, 0.1), (0.7, 0.1), 
        (0.3, 0.2), (0.5, 0.2), (0.7, 0.2),
        (0.3, 0.3), (0.5, 0.3)
    ]
    
    for freq1, freq2 in freq_combinations:
        try:
            # Update guess for this iteration
            test_guess = list(no_c13_popt) + [osc_contrast_guess, freq1, freq2]
            
            # Try this combination
            popt, pcov, red_chi_sq = curve_fit(
                fit_fn,
                total_evolution_times,
                nv_counts,
                test_guess,
                nv_counts_ste,
                bounds=full_bounds,
            )
            
            # If better than current best, update
            if red_chi_sq < best_red_chi_sq:
                best_popt = popt
                best_pcov = pcov
                best_red_chi_sq = red_chi_sq
        except Exception as e:
            # Skip if fitting fails
            continue
    
    timing['multi_freq_fitting'] = time.time() - t0

    # Select the better model (with or without oscillations)
    if len(best_popt) > 5 and best_red_chi_sq < no_c13_red_chi_sq:
        popt = best_popt
        pcov = best_pcov
        red_chi_sq = best_red_chi_sq
    else:
        popt = no_c13_popt
        pcov = no_c13_pcov
        red_chi_sq = no_c13_red_chi_sq

    
    print(f"Red chi sq: {round(red_chi_sq, 3)}")

    # Add total timing information
    timing['total'] = time.time() - fit_start
    
    # Log timing information
    log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fit_log_filename = f'logs/fit_timing_{log_timestamp}.txt'
    
    with open(fit_log_filename, 'w') as log_file:
        log_file.write("Optimized Multi-Start Fitting Breakdown:\n")
        log_file.write(f"\nInitial parameter guesses: {timing['initial_guesses']:.3f} seconds")
        log_file.write(f"\nFirst curve_fit call: {timing['first_curve_fit']:.3f} seconds")
        log_file.write(f"\nMulti-frequency fitting: {timing['multi_freq_fitting']:.3f} seconds")
        log_file.write(f"\nTotal fit time: {timing['total']:.3f} seconds\n")
        
        # Add fit statistics
        log_file.write("\nFit Statistics:\n")
        log_file.write(f"Reduced Chi-Square: {red_chi_sq:.4f}\n")
        log_file.write(f"\nParameters: {popt}\n")

    
    return popt, pcov, red_chi_sq

# Single image display with CLI
def create_interactive_fit_figures(data, nv_inds=None):
    """
    Creates individual figures for each NV and provides a simple command-line interface for navigation
    """
    # Redirect stderr to suppress font warnings
    import os
    import sys
    
    # Create a temporary file to capture stderr
    null = open(os.devnull, 'w')
    
    # Save the original stderr
    original_stderr = sys.stderr
    
    # Redirect stderr to null device
    sys.stderr = null
        
    # Setup data
    nv_list = data["nv_list"]
    taus = np.array(data["taus"])
    total_evolution_times = 2 * np.array(taus) / 1e3
    
    if nv_inds is None:
        num_nvs = len(nv_list)
        nv_inds = list(range(num_nvs))
    else:
        num_nvs = len(nv_inds)
    
    # Process counts data
    if "norm_counts" in data:
        norm_counts = np.array(data["norm_counts"])
        norm_counts_ste = np.array(data["norm_counts_ste"])
    else:
        counts = np.array(data["counts"])
        sig_counts = counts[0]
        ref_counts = counts[1]
        norm_counts, norm_counts_ste = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )
    
    # Precompute fits for all NVs to avoid lag when displaying
    print(f"Precomputing fits for {num_nvs} NVs...")
    fit_fns = []
    popts = []
    
    for i, nv_ind in enumerate(nv_inds):
        print(f"Fitting NV {nv_ind} ({i+1}/{num_nvs})")
        try:
            nv_counts = norm_counts[nv_ind]
            nv_counts_ste = norm_counts_ste[nv_ind]
            
            fit_fn = quartic_decay_fixed_revival
            popt, pcov, red_chi_sq = fit(
                total_evolution_times, nv_counts, nv_counts_ste
            )
            print(f"NV {nv_ind} - Red chi sq: {round(red_chi_sq, 3)}")
            
            fit_fns.append(fit_fn)
            popts.append(popt)
        except Exception as e:
            print(f"Error fitting NV {nv_ind}: {str(e)}")
            fit_fns.append(None)
            popts.append(None)
    
    # Create a single figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Function to update the plot for a specific NV
    def update_plot(idx):
        nv_ind = nv_inds[idx]
        
        # Clear the current axis
        ax.clear()
        
        # Plot the data points
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]
        kpl.plot_points(ax, total_evolution_times, nv_counts, nv_counts_ste)
        
        # Plot the fit line if available
        if idx < len(fit_fns) and fit_fns[idx] is not None and popts[idx] is not None:
            linspace_taus = np.linspace(0, np.max(total_evolution_times), 1000)
            kpl.plot_line(
                ax,
                linspace_taus,
                fit_fns[idx](linspace_taus, *popts[idx]),
                color=kpl.KplColors.GRAY,
            )
        
        # Update title and labels
        fig.suptitle(f'NV {nv_ind}', fontsize=16)
        ax.set_xlabel("Total evolution time (µs)")
        ax.set_ylabel("Normalized NV$^{-}$ population")
        ax.set_ylim(-0.2, 1.2)
        
        # Make sure the figure is visible and refreshed
        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(0.01)  # Small pause to ensure the UI updates
    
    # Display the first NV
    update_plot(0)
    
    # Simple CLI for navigation
    def navigate_figures():
        current = 0
        
        print("\nFigure Navigation:")
        print("------------------")
        print("Enter 'n' or 'next' for next NV")
        print("Enter 'p' or 'prev' for previous NV")
        print("Enter a number to jump to that NV index")
        print("Enter 'q' or 'quit' to exit navigation mode")
        print(f"Currently showing: NV {nv_inds[current]} (Figure {current+1}/{len(nv_inds)})")
        
        while True:
            command = input("\nCommand (n/p/number/q): ").strip().lower()
            
            if command in ['q', 'quit', 'exit']:
                break
            elif command in ['n', 'next']:
                current = (current + 1) % len(nv_inds)
            elif command in ['p', 'prev', 'previous']:
                current = (current - 1) % len(nv_inds)
            else:
                try:
                    # Try to interpret as a direct NV index
                    nv_idx = int(command)
                    if nv_idx in nv_inds:
                        current = nv_inds.index(nv_idx)
                    else:
                        print(f"NV {nv_idx} not found")
                        continue
                except ValueError:
                    # Try to interpret as a figure number (1-based)
                    try:
                        fig_num = int(command)
                        if 1 <= fig_num <= len(nv_inds):
                            current = fig_num - 1
                        else:
                            print(f"Figure number must be between 1 and {len(nv_inds)}")
                            continue
                    except ValueError:
                        print("Invalid command")
                        continue
            
            # Update the plot for the new current index
            update_plot(current)
            print(f"Showing: NV {nv_inds[current]} (Figure {current+1}/{len(nv_inds)})")
    
    # Start the navigation
    navigate_figures()
    
    # Restore original stderr before returning
    sys.stderr = original_stderr
    null.close() 
    
    return fig

def create_fit_figure(data, axes_pack=None, layout=None, no_legend=True, nv_inds=None):


    total_start_time = time.time()
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Create log file with timestamp
    log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/timing_log_{log_timestamp}.txt'
    
    # Detailed timing dictionaries
    timing = {
        'setup': {},
        'data_processing': {},
        'fitting': {
            'per_nv': [],  # List of dicts for each NV's detailed timings
        },
        'plotting': {
            'grid_setup': 0,
            'individual_plots': [],  # List of dicts for each plot's timings
            'summary_figures': []    # List of dicts for each summary figure's timings
        }
    }

    # Initial setup timing
    setup_start = time.time()
    t0 = time.time()
    nv_list = data["nv_list"]
    timing['setup']['data_access'] = time.time() - t0

    t0 = time.time()
    if nv_inds is None:
        num_nvs = len(nv_list)
        nv_inds = list(range(num_nvs))
    else:
        num_nvs = len(nv_inds)
    timing['setup']['nv_indices'] = time.time() - t0

    t0 = time.time()
    num_steps = data["num_steps"]
    taus = np.array(data["taus"])
    total_evolution_times = 2 * np.array(taus) / 1e3
    num_runs = data["num_runs"]
    timing['setup']['time_calculations'] = time.time() - t0
    timing['setup']['total'] = time.time() - setup_start

    # Data processing timing
    data_proc_start = time.time()
    t0 = time.time()
    if "norm_counts" in data:
        norm_counts = np.array(data["norm_counts"])
        norm_counts_ste = np.array(data["norm_counts_ste"])
        timing['data_processing']['type'] = 'direct_access'
    else:
        t1 = time.time()
        counts = np.array(data["counts"])
        sig_counts = counts[0]
        ref_counts = counts[1]
        timing['data_processing']['array_setup'] = time.time() - t1
        
        t1 = time.time()
        norm_counts, norm_counts_ste = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )
        timing['data_processing']['process_counts'] = time.time() - t1
        timing['data_processing']['type'] = 'processed'
    timing['data_processing']['total'] = time.time() - data_proc_start

    do_fit = True
    if do_fit:
        fit_fns = []
        popts = []

        # Grid setup timing
        t0 = time.time()
        n_plots = len(nv_inds)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        
        fig_fits, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_plots > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        timing['plotting']['grid_setup'] = time.time() - t0

        plt.ion()
        plt.show()
        
        # Individual NV processing timing
        for i, nv_ind in enumerate(nv_inds):
            nv_timing = {
                'data_prep': 0,
                'fitting': {
                    'function_call': 0,
                    'chi_sq_calc': 0
                },
                'plotting': {
                    'points': 0,
                    'line': 0,
                    'formatting': 0,
                    'drawing': 0
                },
                'total': 0
            }
            nv_start = time.time()

            t0 = time.time()
            nv_counts = norm_counts[nv_ind]
            nv_counts_ste = norm_counts_ste[nv_ind]
            nv_timing['data_prep'] = time.time() - t0

            try:
                t0 = time.time()
                fit_fn = quartic_decay_fixed_revival
                popt, pcov, red_chi_sq = fit(
                    total_evolution_times, nv_counts, nv_counts_ste
                )
                nv_timing['fitting']['function_call'] = time.time() - t0
                
                t0 = time.time()
                print(f"Red chi sq: {round(red_chi_sq, 3)}")
                nv_timing['fitting']['chi_sq_calc'] = time.time() - t0

                # Plotting timing
                ax = axes[i]
                
                t0 = time.time()
                kpl.plot_points(ax, total_evolution_times, nv_counts, nv_counts_ste)
                nv_timing['plotting']['points'] = time.time() - t0
                
                t0 = time.time()
                linspace_taus = np.linspace(0, np.max(total_evolution_times), 1000)
                kpl.plot_line(
                    ax,
                    linspace_taus,
                    fit_fn(linspace_taus, *popt),
                    color=kpl.KplColors.GRAY,
                )
                nv_timing['plotting']['line'] = time.time() - t0
                
                t0 = time.time()
                ax.set_title(f'NV {nv_ind}')
                ax.set_xlabel("Total evolution time (µs)")
                ax.set_ylabel("Normalized NV$^{-}$ population")
                nv_timing['plotting']['formatting'] = time.time() - t0
                
                t0 = time.time()
                plt.draw()
                plt.pause(0.1)
                nv_timing['plotting']['drawing'] = time.time() - t0

            except Exception:
                print(traceback.format_exc())
                fit_fn = None
                popt = None
            
            nv_timing['total'] = time.time() - nv_start
            timing['fitting']['per_nv'].append(nv_timing)
            
            fit_fns.append(fit_fn)
            popts.append(popt)

        plt.tight_layout()
        plt.ioff()
        plt.show(block=True)

    # Summary figures timing
    figsize = [6.5, 5.0]
    figsize[0] *= 3
    figsize[1] *= 3
    
    for ind in range(2):
        summary_timing = {
            'setup': 0,
            'plot_fit': 0,
            'axis_formatting': 0,
            'total': 0
        }
        summary_start = time.time()
        
        t0 = time.time()
        fig, axes_pack, layout = kpl.subplot_mosaic(num_nvs, figsize=figsize)
        summary_timing['setup'] = time.time() - t0

        t0 = time.time()
        widefield.plot_fit(
            axes_pack,
            [nv_list[ind] for ind in nv_inds],
            total_evolution_times,
            norm_counts[nv_inds],
            norm_counts_ste[nv_inds],
            fit_fns,
            popts,
            no_legend=no_legend,
        )
        summary_timing['plot_fit'] = time.time() - t0
        
        t0 = time.time()
        ax = axes_pack[layout[-1, 0]]
        kpl.set_shared_ax_xlabel(ax, "Total evolution time (µs)")
        kpl.set_shared_ax_ylabel(ax, "Normalized NV$^{-}$ population")
        ax.set_title(num_runs)
        ax.set_ylim(-0.2, 1.2)
        if ind == 1:
            ax.set_xlim(40, 60)
        summary_timing['axis_formatting'] = time.time() - t0
            
        summary_timing['total'] = time.time() - summary_start
        timing['plotting']['summary_figures'].append(summary_timing)

    total_time = time.time() - total_start_time
    
    # Write timing summary to log file
    with open(log_filename, 'w') as log_file:
        log_file.write("Detailed Timing Summary:\n")
        
        log_file.write("\nSetup Phase:\n")
        for key, value in timing['setup'].items():
            if isinstance(value, (int, float)):
                log_file.write(f"  {key}: {value:.3f} seconds\n")
            else:
                log_file.write(f"  {key}: {value}\n")
            
        log_file.write("\nData Processing Phase:\n")
        for key, value in timing['data_processing'].items():
            if isinstance(value, (int, float)):
                log_file.write(f"  {key}: {value:.3f} seconds\n")
            else:
                log_file.write(f"  {key}: {value}\n")
            
        log_file.write("\nIndividual NV Processing:\n")
        for i, nv_timing in enumerate(timing['fitting']['per_nv']):
            log_file.write(f"\n  NV {nv_inds[i]}:\n")
            log_file.write(f"    Data preparation: {nv_timing['data_prep']:.3f} seconds\n")
            log_file.write(f"    Fitting:\n")
            log_file.write(f"      Function call: {nv_timing['fitting']['function_call']:.3f} seconds\n")
            log_file.write(f"      Chi square calculation: {nv_timing['fitting']['chi_sq_calc']:.3f} seconds\n")
            log_file.write(f"    Plotting:\n")
            log_file.write(f"      Points: {nv_timing['plotting']['points']:.3f} seconds\n")
            log_file.write(f"      Line: {nv_timing['plotting']['line']:.3f} seconds\n")
            log_file.write(f"      Formatting: {nv_timing['plotting']['formatting']:.3f} seconds\n")
            log_file.write(f"      Drawing: {nv_timing['plotting']['drawing']:.3f} seconds\n")
            log_file.write(f"    Total for this NV: {nv_timing['total']:.3f} seconds\n")
            
        log_file.write("\nSummary Figure Timing:\n")
        for i, summary_timing in enumerate(timing['plotting']['summary_figures']):
            log_file.write(f"\n  Figure {i+1}:\n")
            log_file.write(f"    Setup: {summary_timing['setup']:.3f} seconds\n")
            log_file.write(f"    Plot fit: {summary_timing['plot_fit']:.3f} seconds\n")
            log_file.write(f"    Axis formatting: {summary_timing['axis_formatting']:.3f} seconds\n")
            log_file.write(f"    Total: {summary_timing['total']:.3f} seconds\n")
        
        log_file.write(f"\nTotal execution time: {total_time:.3f} seconds\n")
        
        # Add a small message to indicate where timing data was saved
        print(f"Timing data has been saved to: {log_filename}")

    return fig


def create_correlation_figure(nv_list, taus, counts):
    total_evolution_times = 2 * np.array(taus) / 1e3

    # fig, ax = plt.subplots()
    fig, axes_pack = plt.subplots(
        nrows=5, ncols=5, sharex=True, sharey=True, figsize=[10, 10]
    )

    widefield.plot_correlations(axes_pack, nv_list, total_evolution_times, counts)

    ax = axes_pack[-1, 0]
    ax.set_xlabel(" ")
    fig.text(0.55, 0.01, "Total evolution time (µs)", ha="center")
    ax.set_ylabel(" ")
    fig.text(0.01, 0.55, "Correlation coefficient", va="center", rotation="vertical")
    return fig


def calc_T2_times(
    peak_total_evolution_times, peak_contrasts, peak_contrast_errs, baselines
):
    for nv_ind in range(len(peak_contrasts)):
        baseline = baselines[nv_ind]

        def envelope(total_evolution_time, T2):
            return (
                -(baseline - 1) * np.exp(-((total_evolution_time / T2) ** 3)) + baseline
            )

        guess_params = (400,)
        popt, pcov = curve_fit(
            envelope,
            peak_total_evolution_times[nv_ind],
            peak_contrasts[nv_ind],
            p0=guess_params,
            sigma=peak_contrast_errs[nv_ind],
            absolute_sigma=True,
        )
        pste = np.sqrt(np.diag(pcov))
        print(f"{round(popt[0])} +/- {round(pste[0])}")


def main(nv_list, num_steps, num_reps, num_runs, min_tau=None, max_tau=None, taus=None):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "spin_echo.py"

    uwave_ind_list = [0, 1]

    ### Collect the data

    def run_fn(shuffled_step_inds):
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
        ]
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=False,
    )

    ### Process and plot

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "tau-units": "ns",
        "taus": taus,
        "min_tau": min_tau,
        "max_tau": max_tau,
    }

# save data
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    # create figure and save
    raw_fig = None
    fit_fig = None
    interactive_fig = None
    
    try:
        # raw_fig = create_raw_data_figure(raw_data)
        # Traditional grid figure with all NVs
        fit_fig = create_fit_figure(raw_data)
        
        # interactive figure, broken
        interactive_fig = create_interactive_fit_figure(raw_data)
    except Exception:
        print(traceback.format_exc())
        # raw_fig = None
        fit_fig = None
        interactive_fig = None

    ### Clean up and return
    tb.reset_cfm()
    kpl.show()

    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)
    if interactive_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-interactive")
        dm.save_figure(interactive_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Combined file
    data = dm.get_raw_data(file_id=1755199883770)

    split_esr = [12, 13, 14, 61, 116]
    broad_esr = [52, 11]
    weak_esr = [72, 64, 55, 96, 112, 87, 12, 58, 36]
    skip_inds = list(set(split_esr + broad_esr + weak_esr))
    nv_inds = [ind for ind in range(117) if ind not in skip_inds]

    
    figures = create_interactive_fit_figures(data, nv_inds=nv_inds)
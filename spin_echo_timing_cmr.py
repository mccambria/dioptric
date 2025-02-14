# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""

import sys
import time
import traceback

import time
from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping, brute
from scipy.signal import lombscargle

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.tool_belt import curve_fit


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

    # fit_fn = quartic_decay
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

    ### Fit assuming no strongly coupled C13
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

    ### Brute to find correct frequencies
    t0 = time.time()
    osc_contrast_guess = no_c13_popt[1]
    args = (
        total_evolution_times,
        nv_counts,
        nv_counts_ste,
        fit_fn,
        no_c13_popt,
        osc_contrast_guess,
    )
    timing['brute_setup'] = time.time() - t0

    t0 = time.time()
    ranges = [(0, 1.5), (0, 0.5)]
    workers = 6
    popt = brute(
        brute_fit_fn_cost, ranges, Ns=1000, finish=None, workers=workers, args=args
    )

    
    timing['brute_force'] = time.time() - t0

    ### Fine tune with a final fit
    t0 = time.time()
    guess_params.append(osc_contrast_guess)
    guess_params.extend(popt)
    bounds[0].extend([-0.5, 0, 0])
    bounds[1].extend([0.5, 1.5, 0.5])
    timing['final_setup'] = time.time() - t0

    t0 = time.time()
    popt, pcov, red_chi_sq = curve_fit(
        fit_fn,
        total_evolution_times,
        nv_counts,
        guess_params,
        nv_counts_ste,
        bounds=bounds,
    )
    timing['final_curve_fit'] = time.time() - t0

    # Calculate total time
    timing['total'] = time.time() - fit_start

    # Write timing to log file
    log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fit_log_filename = f'logs/fit_timing_{log_timestamp}.txt'
    
    with open(fit_log_filename, 'w') as log_file:
        log_file.write("Fit Function Timing Breakdown:\n")
        log_file.write(f"\nInitial parameter guesses: {timing['initial_guesses']:.3f} seconds")
        log_file.write(f"\nFirst curve_fit call: {timing['first_curve_fit']:.3f} seconds")
        log_file.write(f"\nBrute force setup: {timing['brute_setup']:.3f} seconds")
        log_file.write(f"\nBrute force optimization: {timing['brute_force']:.3f} seconds")
        log_file.write(f"\nFinal fit setup: {timing['final_setup']:.3f} seconds")
        log_file.write(f"\nFinal curve_fit call: {timing['final_curve_fit']:.3f} seconds")
        log_file.write(f"\nTotal fit time: {timing['total']:.3f} seconds\n")

    # Return original results along with timing
    if no_c13_red_chi_sq < red_chi_sq:
        return no_c13_popt, no_c13_pcov, no_c13_red_chi_sq
    else:
        return popt, pcov, red_chi_sq
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

    # creat fugure and save
    raw_fig = None
    try:
        # raw_fig = create_raw_data_figure(raw_data)
        fit_fig = create_fit_figure(raw_data)
    except Exception:
        print(traceback.format_exc())
        # raw_fig = None
        fit_fig = None

    ### Clean up and return
    tb.reset_cfm()
    kpl.show()

    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    ### Fit fn testing

    # fig, ax = plt.subplots()
    # tau_linspace = np.linspace(0, 100, 1000)
    # line = quartic_decay(tau_linspace, 0.5, 0.5, 45, 5, 1000, 1, -0.5, 100, 0.5)
    # kpl.plot_line(ax, tau_linspace, line)
    # kpl.show(block=True)
    # sys.exit()

    ###

    # data = dm.get_raw_data(file_id=1548381879624)

    # Separate files
    # # fmt: off
    # file_ids = [1734158411844, 1734273666255, 1734371251079, 1734461462293, 1734569197701, 1736117258235, 1736254107747, 1736354618206, 1736439112682]
    # file_ids2 = [1736589839249, 1736738087977, 1736932211269, 1737087466998, 1737219491182]
    # # fmt: on
    # file_ids = file_ids[:4]
    # file_ids.extend(file_ids2)
    # data = dm.get_raw_data(file_id=file_ids)

    # Combined file
    data = dm.get_raw_data(file_id=1755199883770)

    split_esr = [12, 13, 14, 61, 116]
    broad_esr = [52, 11]
    weak_esr = [72, 64, 55, 96, 112, 87, 12, 58, 36]
    skip_inds = list(set(split_esr + broad_esr + weak_esr))
    nv_inds = [ind for ind in range(117) if ind not in skip_inds]

    # create_raw_data_figure(data)
    create_fit_figure(data, nv_inds=nv_inds)

    plt.show(block=True)

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:34:26 2020

Repeatedly measure the coutns of an NV after red or green light to determine
the NV0 or NV- average counts.

Can either do a single NV or a list of NVs

USE 515 DM, not AM

@author: agardill
"""
# %%
# import majorroutines.image_sample as image_sample
import copy
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
import scipy.stats as stats

import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt


# %%
# Apply a first pulse (typically green or red) followed by a readout pulse (typically under yellow)
# Repeat num_reps number of times and returns the sum of the counts
def main(nv_sig, init_laser, readout_laser, apd_indices, num_reps):
    with labrad.connect() as cxn:
        counts = main_with_cxn(
            cxn, nv_sig, init_laser, readout_laser, apd_indices, num_reps
        )

    return counts


def main_with_cxn(cxn, nv_sig, init_laser, readout_laser, apd_indices, num_reps):
    tool_belt.reset_cfm(cxn)
    num_reps = int(num_reps)

    init_laser_key = nv_sig[init_laser]
    readout_laser_key = nv_sig[readout_laser]

    # Initial Calculation and setup
    tool_belt.set_filter(cxn, nv_sig, init_laser)

    tool_belt.set_filter(cxn, nv_sig, readout_laser)

    init_laser_power = tool_belt.set_laser_power(cxn, nv_sig, init_laser)
    readout_laser_power = tool_belt.set_laser_power(cxn, nv_sig, readout_laser)

    # Estimate the lenth of the sequance
    seq_file = "simple_readout_two_pulse.py"

    #### Load the measuremnt
    readout_on_pulse_ind = 2
    seq_args = [
        nv_sig["{}_dur".format(init_laser)],
        nv_sig["{}_dur".format(readout_laser)],
        init_laser_key,
        readout_laser_key,
        init_laser_power,
        readout_laser_power,
        readout_on_pulse_ind,
        apd_indices[0],
    ]
    #    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(seq_file, seq_args_string)

    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Clear the buffer
    cxn.apd_tagger.clear_buffer()
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(seq_file, num_reps, seq_args_string)

    counts = cxn.apd_tagger.read_counter_simple(num_reps)
    # print(counts)

    return counts


# %%


def nv_compare_charge_counts(cxn, nv_sig, num_reps=1000, save_data=True):
    apd_indices = [0]

    readout_laser = nv_sig["charge_readout_laser"]

    # green init pulse
    init_laser = nv_sig["nv-_prep_laser"]

    targeting.main(nv_sig, apd_indices)
    nvm_counts = main(nv_sig, init_laser, readout_laser, apd_indices, num_reps)

    # red init pulse
    init_laser = nv_sig["nv0_prep_laser"]

    targeting.main(nv_sig, apd_indices)
    nv0_counts = main(nv_sig, init_laser, readout_laser, apd_indices, num_reps)

    nv0_avg = numpy.average(nv0_counts)
    nv0_ste = stats.sem(nv0_counts)
    nvm_avg = numpy.average(nvm_counts)
    nvm_ste = stats.sem(nvm_counts)

    if save_data:
        # # measure laser powers:
        # green_optical_power_pd, green_optical_power_mW, \
        #         red_optical_power_pd, red_optical_power_mW, \
        #         yellow_optical_power_pd, yellow_optical_power_mW = \
        #         tool_belt.measure_g_r_y_power(
        #                           nv_sig['am_589_power'], nv_sig['nd_filter'])

        timestamp = tool_belt.get_time_stamp()
        raw_data = {
            "timestamp": timestamp,
            "nv_sig": nv_sig,
            # 'green_optical_power_pd': green_optical_power_pd,
            # 'green_optical_power_pd-units': 'V',
            # 'green_optical_power_mW': green_optical_power_mW,
            # 'green_optical_power_mW-units': 'mW',
            # 'red_optical_power_pd': red_optical_power_pd,
            # 'red_optical_power_pd-units': 'V',
            # 'red_optical_power_mW': red_optical_power_mW,
            # 'red_optical_power_mW-units': 'mW',
            # 'yellow_optical_power_pd': yellow_optical_power_pd,
            # 'yellow_optical_power_pd-units': 'V',
            # 'yellow_optical_power_mW': yellow_optical_power_mW,
            # 'yellow_optical_power_mW-units': 'mW',
            "num_runs": num_reps,
            "nv0": nv0_counts,
            "nv0-units": "counts",
            "nvm": nvm_counts,
            "nvm-units": "counts",
            "nv0_avg": nv0_avg,
            "nv0_avg-units": "counts",
            "nv0_ste": nv0_ste,
            "nv0_ste-units": "counts",
            "nvm_avg": nvm_avg,
            "nvm_avg-units": "counts",
            "nvm_ste": nvm_ste,
            "nvm_ste-units": "counts",
        }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_raw_data(raw_data, file_path)

    print(str(nv0_avg) + " +/-" + str(nv0_ste))
    print(str(nvm_avg) + " +/-" + str(nvm_ste))

    return nv0_avg, nv0_ste, nvm_avg, nvm_ste


# %%
def vary_init_pulse_dur(nv_sig, num_reps, apd_indices, duration_list):
    readout_laser = "charge_readout_laser"
    init_laser = "initialization_laser"

    targeting.main(nv_sig, apd_indices)

    num_steps = len(duration_list)
    sig_list = numpy.zeros((num_steps, int(num_reps)))
    ref_list = numpy.zeros((num_steps, int(num_reps)))
    tau_ind_list = list(range(0, num_steps))
    tau_index_master_list = []
    shuffle(tau_ind_list)

    for tau_ind in tau_ind_list:
        tau_index_master_list.append(tau_ind)

        print("init pulse: {} ns".format(duration_list[tau_ind]))
        print("measuring with init pulse...")
        nv_sig["{}_dur".format(init_laser)] = duration_list[tau_ind]
        counts = main(nv_sig, init_laser, readout_laser, apd_indices, num_reps)
        sig_list[tau_ind] = counts

        nv_sig["initialization_laser"] = red_laser
        nv_sig["initialization_laser_power"] = 0.66
        # nv_sig['initialization_laser_dur'] = 1e5

        print("measuring reference...")
        # counts = main(nv_sig, readout_laser, readout_laser, apd_indices, num_reps)
        counts = main(nv_sig, init_laser, readout_laser, apd_indices, num_reps)
        ref_list[tau_ind] = counts

    avg_sig_list = numpy.average(sig_list, axis=1)
    avg_ref_list = numpy.average(ref_list, axis=1)

    avg_norm_list = avg_sig_list / avg_ref_list

    init_color = tool_belt.get_registry_entry_no_cxn(
        "wavelength", ["Config", "Optics", nv_sig[init_laser]]
    )
    readout_color = tool_belt.get_registry_entry_no_cxn(
        "wavelength", ["Config", "Optics", nv_sig[readout_laser]]
    )
    readout_dur = nv_sig["{}_dur".format(readout_laser)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    ax.plot(
        numpy.array(duration_list) * 1e-6,
        avg_sig_list,
        "bo",
        label="init {} nm pulse".format(init_color),
    )
    ax.plot(
        numpy.array(duration_list) * 1e-6,
        avg_ref_list,
        "ko",
        label="init pulse same as readout",
    )
    ax.set_xlabel("initial pulse duration (ms)")
    ax.set_ylabel("Counts")
    ax.set_title(
        "initial {} pulse, with {} ms {} nm readout".format(
            init_color, readout_dur * 1e-6, readout_color
        )
    )
    ax.legend()

    ax = axes[1]
    ax.plot(numpy.array(duration_list) * 1e-6, avg_norm_list, "bo")
    ax.set_xlabel("initial pulse duration (ms)")
    ax.set_ylabel("Normalized signal")
    fig.set_tight_layout(True)

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "init_color": init_color,
        "readout_color": readout_color,
        "num_reps": num_reps,
        "duration_list": duration_list,
        "avg_sig_list": avg_sig_list.tolist(),
        "sig_list": sig_list.astype(int).tolist(),
        "avg_ref_list": avg_ref_list.tolist(),
        "ref_list": ref_list.astype(int).tolist(),
    }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)

    return


# %% Run the files

if __name__ == "__main__":
    apd_indicies = [1]

    nd_yellow = "nd_1.5"
    green_power = 10
    red_power = 120
    sample_name = "sandia"
    green_laser = "integrated_520"  # "cobolt_515"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    nv_sig = {
        "coords": [-0.851, -0.343, 6.17],  # a6_R10c10
        "name": "{}-siv_R21_a6_r10_c10".format(
            sample_name,
        ),  # _r10_c10
        "disable_opt": False,
        "ramp_voltages": True,
        "expected_count_rate": 80,
        "imaging_laser": red_laser,
        "imaging_laser_power": 0.595,  # 6 mW
        "imaging_readout_dur": 1e7,
        "initialization_laser": green_laser,
        "initialization_laser_power": green_power,
        "initialization_laser_dur": 1e5,
        "charge_readout_laser": red_laser,
        "charge_readout_laser_power": 0.69,
        "charge_readout_laser_dur": 500,
        "collection_filter": "715_lp",
        "magnet_angle": None,
    }

    try:
        duration_list = [50, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
        # duration_list= [ 5e5, 6e6]
        vary_init_pulse_dur(nv_sig, 1e4, apd_indicies, duration_list)

    except Exception as exc:
        print(exc)

    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()

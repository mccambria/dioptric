# -*- coding: utf-8 -*-
"""
This file contains functions to control the CFM. Just change the function call
in the main section at the bottom of this file and run the file. Shared or
frequently changed parameters are in the __main__ body and relatively static
parameters are in the function definitions.

Created on Sun Nov 25 14:00:28 2018

@author: mccambria
"""


# %% Imports


import labrad
import numpy
import time
import copy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
import majorroutines.image_sample as image_sample
import majorroutines.image_sample_xz as image_sample_xz
import chargeroutines.image_sample_charge_state_compare as image_sample_charge_state_compare
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.optimize_magnet_angle as optimize_magnet_angle
import majorroutines.rabi as rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.ramsey as ramsey
import majorroutines.t1_dq_main as t1_dq_main
import majorroutines.spin_echo as spin_echo
import majorroutines.lifetime_v2 as lifetime_v2
import minorroutines.time_resolved_readout as time_resolved_readout
import chargeroutines.SPaCE as SPaCE
import chargeroutines.SPaCE_simplified as SPaCE_simplified
import chargeroutines.scc_pulsed_resonance as scc_pulsed_resonance
import chargeroutines.scc_spin_echo as scc_spin_echo
import chargeroutines.super_resolution_pulsed_resonance as super_resolution_pulsed_resonance
import chargeroutines.super_resolution_ramsey as super_resolution_ramsey
import chargeroutines.super_resolution_spin_echo as super_resolution_spin_echo
import chargeroutines.g2_measurement as g2_SCC_branch

# import majorroutines.set_drift_from_reference_image as set_drift_from_reference_image
import debug.test_major_routines as test_major_routines
from utils.tool_belt import States
import time


# %% Major Routines


def do_image_sample(nv_sig, apd_indices):

    # scan_range = 0.5
    # num_steps = 90
    # num_steps = 120
    #
    # scan_range = 0.15
    # num_steps = 60
    #
    # scan_range = 0.75
    # num_steps = 150

    # scan_range = 2
    # num_steps = 160
    # scan_range =.5
    # num_steps = 90
    # scan_range = 0.5
    # num_steps = 120
    # scan_range = 0.05
    # num_steps = 60
    # 80 um / V
    #
    # scan_range = 5.0
    # scan_range = 3
    # scan_range = 1
    # scan_range =4
    # scan_range = 2
    # scan_range = 0.5
    #scan_range = 0.35
    # scan_range = 0.25
    # scan_range = 0.2
    # scan_range = 0.15
    scan_range = 0.1
    # scan_range = 0.05
    # scan_range = 0.025
    # scan_range = 0.012

    #num_steps = 400
    # num_steps = 300
    # num_steps = 200
    # num_steps = 160
    # num_steps = 135
    # num_steps =120
    # num_steps = 90
    num_steps = 60
    # num_steps = 31
    # num_steps = 21

    #individual line pairs:
    # scan_range = 0.16
    # num_steps = 160

    #both line pair sets:
    # scan_range = 0.35
    # num_steps = 160


    # For now we only support square scans so pass scan_range twice
    ret_vals = image_sample.main(nv_sig, scan_range, scan_range, num_steps, apd_indices)
    img_array, x_voltages, y_voltages = ret_vals

    return img_array, x_voltages, y_voltages


def do_subtract_filter_image(nv_sig, apd_indices):
    scan_range = 0.2
    num_steps = 90

    nv_sig['collection_filter'] = "715_lp"
    img_array_siv, x_voltages, y_voltages = image_sample.main(nv_sig, scan_range,
                                          scan_range, num_steps, apd_indices)

    nv_sig['collection_filter'] = "715_sp+630_lp"
    img_array_nv, x_voltages, y_voltages = image_sample.main(nv_sig, scan_range,
                                         scan_range, num_steps, apd_indices)

    img_array_sub = img_array_siv - img_array_nv

    x_num_steps = len(x_voltages)
    x_low = x_voltages[0]
    x_high = x_voltages[x_num_steps-1]
    y_num_steps = len(y_voltages)
    y_low = y_voltages[0]
    y_high = y_voltages[y_num_steps-1]

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2

    readout = nv_sig['imaging_readout_dur']
    readout_sec = readout / 10**9
    img_array_kcps = numpy.copy(img_array_sub)
    img_array_kcps[:] = (img_array_sub[:] / 1000) / readout_sec

    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]

    title = 'SiV filter images - NV filter image'
    fig = tool_belt.create_image_figure(img_array_kcps, img_extent,
                    clickHandler=image_sample.on_click_image, color_bar_label='kcps',
                    title=title)

    time.sleep(1)
    timestamp = tool_belt.get_time_stamp()
    filePath = tool_belt.get_file_path('image_sample.py', timestamp, nv_sig['name'])
    tool_belt.save_figure(fig, filePath)

    return

def do_image_sample_xz(nv_sig, apd_indices):

    scan_range_x = .25
# z code range 3 to 7 if centered at 5
    scan_range_z =1
    num_steps = 60

    image_sample_xz.main(
        nv_sig,
        scan_range_x,
        scan_range_z,
        num_steps,
        apd_indices,
        um_scaled=False,
    )


def do_image_charge_states(nv_sig, apd_indices):

    scan_range = 0.01

    num_steps = 31
    num_reps= 10

    image_sample_charge_state_compare.main(
        nv_sig, scan_range, scan_range, num_steps,num_reps, apd_indices
    )


def do_optimize(nv_sig, apd_indices):

    optimize.main(
        nv_sig,
        apd_indices,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )


def do_optimize_list(nv_sig_list, apd_indices):

    optimize.optimize_list(nv_sig_list, apd_indices)


def do_opti_z(nv_sig_list, apd_indices):

    optimize.opti_z(
        nv_sig_list,
        apd_indices,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )


def do_stationary_count(nv_sig, apd_indices):

    run_time = 1 * 60 * 10 ** 9  # ns

    stationary_count.main(nv_sig, run_time, apd_indices)


def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):

    run_time = 3*60  # s
    diff_window =120# ns

    # g2_measurement.main(
    g2_SCC_branch.main(
        nv_sig, run_time, diff_window, apd_a_index, apd_b_index
    )


def do_resonance(nv_sig, opti_nv_sig,apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 11#101
    num_runs = 2#15
    uwave_power = -10.0

    resonance.main(
        nv_sig,
        apd_indices,
        freq_center,
        freq_range,
        num_steps,
        num_runs,
        uwave_power,
        state=States.HIGH,
        opti_nv_sig = opti_nv_sig
    )


def do_resonance_state(nv_sig, opti_nv_sig, apd_indices, state):

    freq_center = nv_sig["resonance_{}".format(state.name)]
    uwave_power = 10.0

    freq_range = 0.15
    num_steps = 51
    num_runs = 10

    # Zoom
    # freq_range = 0.060
    # num_steps = 51
    # num_runs = 10

    resonance.main(
        nv_sig,
        apd_indices,
        freq_center,
        freq_range,
        num_steps,
        num_runs,
        uwave_power,
        opti_nv_sig = opti_nv_sig
    )


def do_pulsed_resonance(nv_sig, opti_nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps =101
    num_reps = 1e4
    num_runs = 10
    uwave_power = 10
    uwave_pulse_dur = int(40)

    pulsed_resonance.main(
        nv_sig,
        apd_indices,
        freq_center,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        uwave_power,
        uwave_pulse_dur,
        opti_nv_sig = opti_nv_sig
    )


def do_pulsed_resonance_state(nv_sig, opti_nv_sig,apd_indices, state):

    # freq_range = 0.150
    # num_steps = 51
    # num_reps = 10**4
    # num_runs = 8

    # Zoom
    freq_range = 0.1
    # freq_range = 0.120
    num_steps = 51
    num_reps = int(1e4)
    num_runs = 10

    composite = False

    res, _ = pulsed_resonance.state(
        nv_sig,
        apd_indices,
        state,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        composite,
        opti_nv_sig = opti_nv_sig
    )
    nv_sig["resonance_{}".format(state.name)] = res


def do_optimize_magnet_angle(nv_sig, apd_indices):

    # angle_range = [132, 147]
    #    angle_range = [315, 330]
    num_angle_steps = 6
    #    freq_center = 2.7921
    #    freq_range = 0.060
    angle_range = [0, 150]
    #    num_angle_steps = 6
    freq_center = 2.87
    freq_range = 0.3
    num_freq_steps = 101
    num_freq_runs = 10

    # Pulsed
    uwave_power = 10#14.5
    uwave_pulse_dur = 80/2
    num_freq_reps = int(1e4)

    # CW
    #uwave_power = -10.0
    #uwave_pulse_dur = None
    #num_freq_reps = None

    optimize_magnet_angle.main(
        nv_sig,
        apd_indices,
        angle_range,
        num_angle_steps,
        freq_center,
        freq_range,
        num_freq_steps,
        num_freq_reps,
        num_freq_runs,
        uwave_power,
        uwave_pulse_dur,
    )


def do_rabi(nv_sig, opti_nv_sig, apd_indices, state, uwave_time_range=[0, 200]):

    num_steps = 51
    num_reps = int(1e4)
    num_runs = 2

    period = rabi.main(
        nv_sig,
        apd_indices,
        uwave_time_range,
        state,
        num_steps,
        num_reps,
        num_runs,
        opti_nv_sig = opti_nv_sig
    )
    nv_sig["rabi_{}".format(state.name)] = period




def do_lifetime(nv_sig, apd_indices, filter, voltage, reference=False):

    #    num_reps = 100 #MM
    num_reps = 500  # SM
    num_bins = 101
    num_runs = 5
    readout_time_range = [0, 1.0 * 10 ** 6]  # ns
    polarization_time = 60 * 10 ** 3  # ns

    lifetime_v2.main(
        nv_sig,
        apd_indices,
        readout_time_range,
        num_reps,
        num_runs,
        num_bins,
        filter,
        voltage,
        polarization_time,
        reference,
    )



def do_ramsey(nv_sig, opti_nv_sig, apd_indices):

    detuning = 10  # MHz
    precession_time_range = [0, 2 * 10 ** 3]
    num_steps = 101
    num_reps = int( 10 ** 4)
    num_runs = 6

    ramsey.main(
        nv_sig,
        apd_indices,
        detuning,
        precession_time_range,
        num_steps,
        num_reps,
        num_runs,
        opti_nv_sig = opti_nv_sig
    )


def do_spin_echo(nv_sig, apd_indices):

    # T2* in nanodiamond NVs is just a couple us at 300 K
    # In bulk it's more like 100 us at 300 K
    max_time = 100  # us
    num_steps = int(max_time + 1)  # 1 point per us
    precession_time_range = [0, max_time*10**3]


    num_reps = 1e4
    num_runs =100

    #    num_steps = 151
    #    precession_time_range = [0, 10*10**3]
    #    num_reps = int(10.0 * 10**4)
    #    num_runs = 6

    state = States.LOW



    angle = spin_echo.main(
        nv_sig,
        apd_indices,
        precession_time_range,
        num_steps,
        num_reps,
        num_runs,
        state,
    )
    return angle

def do_relaxation(nv_sig, apd_indices, ):
    min_tau = 0
    max_tau_omega = 15e6
    max_tau_gamma = 15e6
    num_steps = 31
    num_reps = 1e4
    num_runs = 20

    t1_exp_array = numpy.array(
        [[
                [States.HIGH, States.HIGH],
                [min_tau, max_tau_gamma],
                num_steps,
                num_reps,
                num_runs,
            ]])

    # t1_exp_array = numpy.array(
    #    [ [
    #             [States.ZERO, States.ZERO],
    #             [min_tau, max_tau_omega],
    #             num_steps,
    #             num_reps,
    #             num_runs,
    #         ],
    #     [
    #             [States.ZERO, States.HIGH],
    #             [min_tau, max_tau_omega],
    #             num_steps,
    #             num_reps,
    #             num_runs,
    #         ],
    #             [
    #             [States.HIGH, States.HIGH],
    #             [min_tau, max_tau_gamma],
    #             num_steps,
    #             num_reps,
    #             num_runs,
    #         ],
    #                 [
    #             [States.HIGH, States.LOW],
    #             [min_tau, max_tau_gamma],
    #             num_steps,
    #             num_reps,
    #             num_runs,
    #         ]] )

    t1_dq_main.main(
            nv_sig,
            apd_indices,
            t1_exp_array,
            num_runs,
            composite_pulses=False,
            scc_readout=False,
        )

def do_time_resolved_readout(nv_sig, apd_indices):

    # nv_sig uses the initialization key for the first pulse
    # and the imaging key for the second

    num_reps = 1000
    num_bins = 2001
    num_runs = 20
    # disp = 0.0001#.05

    bin_centers, binned_samples_sig = time_resolved_readout.main(
        nv_sig,
        apd_indices,
        num_reps,
        num_runs,
        num_bins
    )
    return bin_centers, binned_samples_sig

def do_time_resolved_readout_three_pulses(nv_sig, apd_indices):

    # nv_sig uses the initialization key for the first pulse
    # and the imaging key for the second

    num_reps = 1000
    num_bins = 2001
    num_runs = 20


    bin_centers, binned_samples_sig = time_resolved_readout.main_three_pulses(
        nv_sig,
        apd_indices,
        num_reps,
        num_runs,
        num_bins
    )

    return bin_centers, binned_samples_sig



def do_SPaCE(nv_sig, opti_nv_sig, apd_indices,num_runs, num_steps_a, num_steps_b,
               img_range_1D, img_range_2D, offset, charge_state_threshold = None):
    # dr = 0.025 / numpy.sqrt(2)
    # img_range = [[-dr,-dr],[dr, dr]] #[[x1, y1], [x2, y2]]
    # num_steps = 101
    # num_runs = 50
    # measurement_type = "1D"

    # img_range = 0.075
    # num_steps = 71
    # num_runs = 1
    # measurement_type = "2D"

    # dz = 0
    SPaCE.main(nv_sig, opti_nv_sig, apd_indices,num_runs, num_steps_a, num_steps_b,
               charge_state_threshold, img_range_1D, img_range_2D, offset )


def do_SPaCE_simplified(nv_sig, source_coords, apd_indices):

    # pulse_durs = numpy.linspace(0,0.7e9, 3)
    # pulse_durs = numpy.linspace(0,1.5e9, 30)
    # pulse_durs = numpy.linspace(1e2,1e9, 5)
    # pulse_durs = numpy.array([0,  0.1, ])*1e9
    pulse_powers = numpy.array([0, 0.565])
    pulse_durs= None

    num_reps =int(100)

    SPaCE_simplified.main(nv_sig, source_coords, num_reps, apd_indices,
         pulse_durs, pulse_powers)

def do_SPaCE_simplified_time_resolved_readout(nv_sig, source_coords, apd_indices):

    num_reps =int(1000)
    num_runs = 10
    num_bins = 52
    bin_centers, binned_samples_sig = SPaCE_simplified.main_time_resolved_readout(nv_sig, source_coords,
                                                num_reps, num_runs,num_bins,apd_indices)
    return bin_centers, binned_samples_sig

def do_SPaCE_simplified_scan_init(nv_sig, source_coords_list, init_scan_range,
                                  init_scan_steps, num_runs, apd_indices):



    SPaCE_simplified.main_scan_init(nv_sig, source_coords_list, init_scan_range, init_scan_steps,
                   num_runs,  apd_indices)


def do_scc_resonance(nv_sig, opti_nv_sig, apd_indices, state=States.LOW):
    freq_center = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    uwave_pulse_dur = nv_sig['rabi_{}'.format(state.name)]/2

    freq_range = 0.05
    num_steps = 51
    num_reps = int(10**3)
    num_runs = 30

    scc_pulsed_resonance.main(nv_sig, opti_nv_sig, apd_indices, freq_center, freq_range,
         num_steps, num_reps, num_runs, uwave_power, uwave_pulse_dur, state )

def do_scc_spin_echo(nv_sig, opti_nv_sig, apd_indices, tau_start, tau_stop, state=States.LOW):
    step_size = 1 # us
    num_steps = int((tau_stop - tau_start)/step_size + 1)

    precession_time_range = [tau_start *1e3, tau_stop *1e3]

    num_reps = int(10**3)
    num_runs = 40

    scc_spin_echo.main(nv_sig, opti_nv_sig, apd_indices, precession_time_range,
         num_steps, num_reps, num_runs,
         state )



def do_super_resolution_resonance(nv_sig, opti_nv_sig, apd_indices, state=States.LOW):
    freq_center = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    uwave_pulse_dur = nv_sig['rabi_{}'.format(state.name)]/2

    freq_range = 0.05
    num_steps = 51
    num_reps = int(10**3)
    num_runs = 30

    super_resolution_pulsed_resonance.main(nv_sig, opti_nv_sig, apd_indices, freq_center, freq_range,
         num_steps, num_reps, num_runs, uwave_power, uwave_pulse_dur, state )

def do_super_resolution_ramsey(nv_sig, opti_nv_sig, apd_indices,
                                  tau_start, tau_stop, state=States.LOW):

    detuning = 5  # MHz

    # step_size = 0.05 # us
    # num_steps = int((tau_stop - tau_start)/step_size + 1)
    num_steps = 101
    precession_time_range = [tau_start *1e3, tau_stop *1e3]


    num_reps = int(10**3)
    num_runs = 30

    super_resolution_ramsey.main(nv_sig, opti_nv_sig, apd_indices,
                                    precession_time_range, detuning,
         num_steps, num_reps, num_runs, state )

def do_super_resolution_spin_echo(nv_sig, opti_nv_sig, apd_indices,
                                  tau_start, tau_stop, state=States.LOW):
    step_size = 1 # us
    num_steps = int((tau_stop - tau_start)/step_size + 1)
    print(num_steps)
    precession_time_range = [tau_start *1e3, tau_stop *1e3]


    num_reps = int(10**3)
    num_runs = 20

    super_resolution_spin_echo.main(nv_sig, opti_nv_sig, apd_indices,
                                    precession_time_range,
         num_steps, num_reps, num_runs, state )

def do_sample_nvs(nv_sig_list, apd_indices):

    # g2 parameters
    run_time = 60 * 5
    diff_window = 150

    # PESR parameters
    num_steps = 101
    num_reps = 10 ** 5
    num_runs = 3
    uwave_power = 9.0
    uwave_pulse_dur = 100

    g2 = g2_measurement.main_with_cxn
    pesr = pulsed_resonance.main_with_cxn

    with labrad.connect() as cxn:
        for nv_sig in nv_sig_list:
            g2_zero = g2(
                cxn,
                nv_sig,
                run_time,
                diff_window,
                apd_indices[0],
                apd_indices[1],
            )
            if g2_zero < 0.5:
                pesr(
                    cxn,
                    nv_sig,
                    apd_indices,
                    2.87,
                    0.1,
                    num_steps,
                    num_reps,
                    num_runs,
                    uwave_power,
                    uwave_pulse_dur,
                )


def do_test_major_routines(nv_sig, apd_indices):
    """Run this whenver you make a significant code change. It'll make sure
    you didn't break anything in the major routines.
    """

    test_major_routines.main(nv_sig, apd_indices)


# %% Run the file


if __name__ == "__main__":

    # In debug mode, don't bother sending email notifications about exceptions
    debug_mode = True

    # %% Shared parameters

    # apd_indices = [0]
    apd_indices = [1]
    # apd_indices = [0,1]

    nd_yellow = "nd_0"
    green_power =8000
    nd_green = 'nd_0.4'
    red_power = 120
    sample_name = "rubin"
    green_laser = "integrated_520"#"cobolt_515"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"



    nv_sig = {
            "coords":[-0.854, -0.605,  6.177],
        "name": "{}-nv1".format(sample_name,),
        "disable_opt":False,
        "ramp_voltages": False,
        "expected_count_rate":None,
        "correction_collar": 0.12,



          "spin_laser":green_laser,
          "spin_laser_power": green_power,
         "spin_laser_filter": nd_green,
          "spin_readout_dur": 350,
          "spin_pol_dur": 1000.0,

          "imaging_laser":green_laser,
        "imaging_laser_power": green_power,
         "imaging_laser_filter": nd_green,
          "imaging_readout_dur": 1e7,

         "initialize_laser": green_laser,
           "initialize_laser_power": green_power,
           "initialize_laser_dur":  1e3,
         "CPG_laser": green_laser,
           "CPG_laser_power":red_power,
           "CPG_laser_dur": int(1e6),




         "charge_readout_laser": yellow_laser,
          "charge_readout_laser_power": 0.2, #0.15 for NV
          "charge_readout_laser_filter": "nd_1.0",
          "charge_readout_laser_dur": 50e6, #50e6 for NV

        # "collection_filter": "715_lp",#see only SiV (some NV signal)
        # "collection_filter": "740_bp",#SiV emission only (no NV signal)
        "collection_filter": "715_sp+630_lp", # NV band only
        "magnet_angle": 156,
        "resonance_LOW":2.7790,
        "rabi_LOW":72.2,
        "uwave_power_LOW": 10,  # 15.5 max
        "resonance_HIGH":2.7790,#2.9611,
        "rabi_HIGH":68,
        "uwave_power_HIGH": 10,
    }  # 14.5 max






    nv_sig = nv_sig


    # %% Functions to run
#
    try:

        #tool_belt.init_safe_stop()
        # for dz in [0, 0.15,0.3, 0.45, 0.6, 0.75,0.9, 1.05, 1.2, 1.5, 1.7, 1.85, 2, 2.15, 2.3, 2.45]: #0.5,0.4, 0.3, 0.2, 0.1,0, -0.1,-0.2,-0.3, -0.4, -0.5
            # nv_sig_copy = copy.deepcopy(nv_sig)
            # coords = nv_sig["coords"]
            # new_coords= list(numpy.array(coords)+ numpy.array([0, 0, dz]))
            # # new_coords = numpy.array(coords) +[0, 0, dz]
            # # print(new_coords)
            # nv_sig_copy['coords'] = new_coords
            # do_image_sample(nv_sig_copy, apd_indices)
         #
        #
        # tool_belt.set_drift([0.0, 0.0, tool_belt.get_drift()[2]])  # Keep z
        # tool_belt.set_drift([0.0, 0.0, 0.0])
        # tool_belt.set_xyz(labrad.connect(), [0,0,5])
#
        # do_optimize(nv_sig,apd_indices)

        # do_image_sample(nv_sig, apd_indices)

        # do_stationary_count(nv_sig, apd_indices)


        # do_image_sample_xz(nv_sig, apd_indices)
        # do_image_charge_states(nv_sig, apd_indices)


        # do_subtract_filter_image(nv_sig, apd_indices)
        # nv_sig["collection_filter"] = "740_bp"
        # do_image_sample(nv_sig, apd_indices)
        # do_g2_measurement(nv_sig, 0, 1)

        # num_runs = 20
        # num_steps_a = 81
        # num_steps_b = num_steps_a
        # img_range_1D = None#[[0.042, 0, 0],[0,0,0]]

        # img_range_2D = [0.1, 0.1, 0]
        # offset = [-0.22/83,0.25/83,0]
        # for t in [5e8]:
        #     nv_sig["CPG_laser_dur"] = t

            # do_SPaCE(nv_sig, nv_sig, apd_indices,num_runs, num_steps_a, num_steps_b,
            #  img_range_1D,img_range_2D, offset)

        #do_optimize_magnet_angle(nv_sig, apd_indices)
        # do_resonance(nv_sig, nv_sig, apd_indices,  2.875, 0.2)
        # do_resonance(nv_sig, nv_sig, apd_indices,  2.875, 0.1)
        # do_resonance_state(nv_sig,nv_sig, apd_indices, States.LOW)
        # do_resonance_state(nv_sig,nv_sig, apd_indices, States.HIGH)

        # do_rabi(nv_sig, nv_sig, apd_indices, States.LOW, uwave_time_range=[0, 200])
        do_rabi(nv_sig, nv_sig,apd_indices, States.HIGH, uwave_time_range=[0, 200])

        #do_pulsed_resonance(nv_sig, nv_sig, apd_indices, 2.87, 0.30) ###
        # do_pulsed_resonance_state(nv_sig, nv_sig,apd_indices, States.LOW)
        # do_pulsed_resonance_state(nv_sig, nv_sig,apd_indices, States.HIGH)
        # do_ramsey(nv_sig, opti_nv_sig,apd_indices)

        #do_spin_echo(nv_sig, apd_indices)


        # do_relaxation(nv_sig, apd_indices)


        # Operations that don't need an NV#
        # tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset
        # tool_belt.set_drift([0.0, 0.0, tool_belt.get_drift()[2]])  # Keep z
        # tool_belt.set_xyz(labrad.connect(), [0,0,5])
#-0.243, -0.304,5.423
#ML -0.216, -0.115,5.417
    except Exception as exc:
        # Intercept the exception so we can email it out and re-raise it
        if not debug_mode:
            tool_belt.send_exception_email()
        raise exc

    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        tool_belt.reset_safe_stop()

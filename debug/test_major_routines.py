# -*- coding: utf-8 -*-
"""
Loop through extremely abbreviated versions of all the major routines
to make sure they all run to completion. This should be run whenever
significant changes are made to the code. 

Created on Thu Jun 13 14:30:42 2019

@author: mccambria
"""


import labrad
import majorroutines.image_sample as image_sample
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.pulsed_resonance as pulsed_resonance 
import majorroutines.rabi as rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.t1_double_quantum as t1_double_quantum
import majorroutines.ramsey as ramsey
import majorroutines.spin_echo as spin_echo
import majorroutines.optimize_magnet_angle as optimize_magnet_angle


def main(nv_sig, apd_indices):
    
    with labrad.connect() as cxn:
        
        # g2_measurement
        print('\nTesting g2_measurement...\n')
        run_time = 5
        diff_window = 150
        g2_measurement.main(cxn, nv_sig, run_time,
                            diff_window, apd_indices[0], apd_indices[1])
        
        # image_sample
        print('\nTesting image_sample...\n')
        scan_range = 0.01
        num_scan_steps = 5
        # For now we only support square scans so pass scan_range twice
        image_sample.main(cxn, nv_sig, scan_range, scan_range,
                          num_scan_steps, apd_indices)
        
        # optimize
        print('\nTesting optimize...\n')
        optimize.main(cxn, nv_sig, apd_indices,
                  set_to_opti_coords=False, save_data=True, plot_data=True)
        
        # stationary_count
        print('\nTesting stationary_count...\n')
        run_time = 5
        readout = 100 * 10**6
        stationary_count.main(cxn, nv_sig, run_time, readout, apd_indices)
        
        # resonance
        print('\nTesting resonance...\n')
        freq_center = 2.87
        freq_range = 0.05
        num_steps = 5
        num_runs = 2
        uwave_power = -13.0
        resonance.main(cxn, nv_sig, apd_indices, freq_center, 
                       freq_range, num_steps, num_runs, uwave_power)
        
        # pulsed_resonance
        print('\nTesting pulsed_resonance...\n')
        freq_center = 2.87
        freq_range = 0.05
        num_steps = 5
        num_runs = 2
        uwave_power = -13.0
        pulsed_resonance.main(cxn, nv_sig, apd_indices, freq_center,
                              freq_range, num_steps, num_runs, uwave_power)
        
        # rabi
        print('\nTesting rabi...\n')
        uwave_freq = 2.87
        uwave_power = 9.0
        uwave_time_range = [0, 100]
        do_uwave_gate_number = 0
        num_steps = 10
        num_reps = 10
        num_runs = 2
        rabi.main(cxn, nv_sig, apd_indices, 
                  uwave_freq, uwave_power, uwave_time_range,
                  do_uwave_gate_number,
                  num_steps, num_reps, num_runs)
        
        # t1_double_quantum
        print('\nTesting t1_double_quantum...\n')
        uwave_freq_plus = 2.86
        uwave_freq_minus = 2.88
        uwave_power = 9
        uwave_pi_pulse_plus = 65
        uwave_pi_pulse_minus = 75
        relaxation_time_range = [0, 10*10**3]
        num_steps = 10
        num_reps = 10
        num_runs = 2
        init_read_list = [-1, +1]
        t1_double_quantum.main(cxn, nv_sig, apd_indices,
                     uwave_freq_plus, uwave_freq_minus,
                     uwave_power, uwave_power,
                     uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                     relaxation_time_range, num_steps, num_reps, num_runs, 
                     init_read_list)
        
        # t1_double_quantum
        print('\nTesting t1_double_quantum...\n')
        uwave_freq_plus = 2.86
        uwave_freq_minus = 2.88
        uwave_power = 9
        uwave_pi_pulse_plus = 65
        uwave_pi_pulse_minus = 75
        relaxation_time_range = [0, 10*10**3]
        num_steps = 10
        num_reps = 10
        num_runs = 2
        init_read_list = [-1, +1]
        t1_double_quantum.main(cxn, nv_sig, apd_indices,
                     uwave_freq_plus, uwave_freq_minus,
                     uwave_power, uwave_power,
                     uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                     relaxation_time_range, num_steps, num_reps, num_runs, 
                     init_read_list)
        
        # ramsey
        print('\nTesting t1_double_quantum...\n')
        uwave_freq_plus = 2.86
        uwave_freq_minus = 2.88
        uwave_power = 9
        uwave_pi_pulse_plus = 65
        uwave_pi_pulse_minus = 75
        relaxation_time_range = [0, 10*10**3]
        num_steps = 10
        num_reps = 10
        num_runs = 2
        init_read_list = [-1, +1]
        ramsey.main(cxn, nv_sig, apd_indices,
                     uwave_freq_plus, uwave_freq_minus,
                     uwave_power, uwave_power,
                     uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                     relaxation_time_range, num_steps, num_reps, num_runs, 
                     init_read_list)
        
        # spin_echo
        print('\nTesting optimize_magnet_angle...\n')
        angle_range = [0, 45]
        num_angle_steps = 2
        freq_center = 2.87
        freq_range = 0.2
        num_freq_steps = 3
        num_freq_runs = 1
        uwave_power = 9.0
        spin_echo.main(nv_sig, apd_indices,
                               angle_range, num_angle_steps,
                               freq_center, freq_range,
                               num_freq_steps, num_freq_runs, uwave_power)
        
        # Success!
        print('\nNo crashes went unhandled. Success!')
    
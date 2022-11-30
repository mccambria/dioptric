# -*- coding: utf-8 -*-
"""
Created on Sat May  4 08:34:08 2019

@author: Aedan
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
import utils.tool_belt as tool_belt

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    # %% Parse wiring and args

    # Unpack the args
    readout_time, init_time, depletion_time, ion_time, shelf_time,\
            tau_shrt, tau_long, pi_pulse, pi_on_2_pulse, \
                init_color, depletion_color, \
            green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name, \
            apd_indices, readout_power, shelf_power = args
            
    # Convert all times to int64
    readout_time = numpy.int64(readout_time)
    init_time = numpy.int64(init_time)
    depletion_time = numpy.int64(depletion_time)
    ion_time = numpy.int64(ion_time)
    shelf_time = numpy.int64(shelf_time)
    pi_pulse = numpy.int64(pi_pulse)
    pi_on_2_pulse = numpy.int64(pi_on_2_pulse)
    tau_shrt = numpy.int64(tau_shrt)
    tau_long = numpy.int64(tau_long)
            
        
    wait_time = config['CommonDurations']['uwave_buffer']
    galvo_move_time = config['Positioning']['xy_small_response_delay']
    galvo_move_time = numpy.int64(galvo_move_time)
    
    # delays
    green_delay_time = config['Optics'][green_laser_name]['delay']
    yellow_delay_time = config['Optics'][yellow_laser_name]['delay']
    red_delay_time = config['Optics'][red_laser_name]['delay']
    rf_delay_time = config['Microwaves'][sig_gen_name]['delay']
    
    
    # TESTING
    # wait_time =100
    # galvo_move_time = 500
    # galvo_move_time = numpy.int64(galvo_move_time)
    # green_delay_time = 0
    # yellow_delay_time = 0
    # red_delay_time =0
    # rf_delay_time = 0
    
    total_delay = green_delay_time + yellow_delay_time + red_delay_time + rf_delay_time
    
    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_ao_589_aom = pulser_wiring['ao_{}_am'.format(yellow_laser_name)]
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    

    # %% Write the microwave sequence to be used.

    # In t1, the sequence is just a pi pulse, wait for a relaxation time, then
    # then a second pi pulse

    # I define both the time of this experiment, which is useful for the AOM
    # and gate sequences to dictate the time for them to be LOW
    # And I define the actual uwave experiement to be plugged into the rf
    # sequence. I hope that this formatting works.

    # With future protocols--ramsey, spin echo, etc--it will be easy to use
    # this format of sequence building and just change this section of the file

    uwave_experiment_shrt = pi_on_2_pulse + tau_shrt + pi_pulse + \
                            tau_shrt + pi_on_2_pulse
    uwave_experiment_long = pi_on_2_pulse + tau_long + pi_pulse + \
                            tau_long + pi_on_2_pulse

    # %% Couple calculated values

    readout_scheme = shelf_time + ion_time + wait_time + readout_time
    
    preparation = init_time + galvo_move_time + depletion_time + galvo_move_time
    
        

    # %% Calclate total period. This is fixed for each tau index

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = total_delay + ( preparation +uwave_experiment_shrt +  2* wait_time + readout_scheme)*2 +\
       ( preparation +uwave_experiment_long +  2* wait_time + readout_scheme)*2 

    # %% Define the sequence

    seq = Sequence()

    # APD gating
    train =  [(total_delay + preparation + uwave_experiment_shrt + wait_time + readout_scheme - readout_time, LOW),
             (readout_time, HIGH),
             (wait_time + preparation + uwave_experiment_shrt + wait_time + readout_scheme - readout_time, LOW),
             (readout_time, HIGH),
             (wait_time + preparation + uwave_experiment_long + wait_time + readout_scheme - readout_time, LOW),
             (readout_time, HIGH),
             (wait_time + preparation + uwave_experiment_long + wait_time + readout_scheme - readout_time, LOW),
             (readout_time, HIGH), (100, LOW)]
    
    seq.setDigital(pulser_do_apd_gate, train)


    
    # Yellow laser
    delay = total_delay - yellow_delay_time
    train = [(delay + preparation + uwave_experiment_shrt + wait_time, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW),
             (readout_time, readout_power),
             (wait_time, LOW),
             (preparation + uwave_experiment_shrt + wait_time, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW),
             (readout_time, readout_power),
             (wait_time, LOW),
             (preparation + uwave_experiment_long + wait_time, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW),
             (readout_time, readout_power),
             (wait_time, LOW),
             (preparation + uwave_experiment_long + wait_time, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW),
             (readout_time, readout_power),
             (wait_time, LOW),
             (yellow_delay_time, LOW)]
    seq.setAnalog(pulser_ao_589_aom, train) 
    
    # Clock    

    train = [(total_delay + init_time + 100, LOW), (100, HIGH),
              (galvo_move_time - 100 + depletion_time, LOW), (100, HIGH), 
              (galvo_move_time + uwave_experiment_shrt  - 100 + wait_time + readout_scheme, LOW),
              (100, HIGH), 
              (wait_time - 100 + init_time, LOW), (100, HIGH),
              (galvo_move_time - 100 + depletion_time, LOW), (100, HIGH), 
              (galvo_move_time + uwave_experiment_shrt  - 100 + wait_time + readout_scheme, LOW),
              (100, HIGH), 
              (wait_time - 100 + init_time, LOW), (100, HIGH),
              (galvo_move_time - 100 + depletion_time, LOW), (100, HIGH), 
              (galvo_move_time + uwave_experiment_long  - 100 + wait_time + readout_scheme, LOW),
              (100, HIGH), 
              (wait_time - 100 + init_time, LOW), (100, HIGH),
              (galvo_move_time - 100 + depletion_time, LOW), (100, HIGH), 
              (galvo_move_time + uwave_experiment_long  - 100 + wait_time + readout_scheme, LOW),
              (100, HIGH), (wait_time - 100, LOW)
             ]
    seq.setDigital(pulser_do_clock, train)

    # Pulse the microwave for tau
    delay = total_delay - rf_delay_time
    train = [(delay + preparation, LOW),
             (pi_on_2_pulse, HIGH), (tau_shrt, LOW),
             (pi_pulse, HIGH),
             (tau_shrt, LOW), (pi_on_2_pulse, HIGH),
             (wait_time + readout_scheme + wait_time, LOW),
             (preparation + uwave_experiment_shrt, LOW),
             (wait_time + readout_scheme + wait_time, LOW),
             (preparation, LOW),
             (pi_on_2_pulse, HIGH), (tau_long, LOW),
             (pi_pulse, HIGH),
             (tau_long, LOW), (pi_on_2_pulse, HIGH),
             (wait_time + readout_scheme + wait_time, LOW),
             (preparation + uwave_experiment_long, LOW),
             (wait_time + readout_scheme + wait_time, LOW),
             (rf_delay_time, LOW)
             ]
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    #### Green and Red
    # Green laser
    green_delay = total_delay - green_delay_time
    green_train = [ (green_delay, LOW)]
    
    # Red laser
    red_delay = total_delay - red_delay_time
    red_train = [(red_delay, LOW)]
    
    init_train_on = [(init_time, HIGH)]
    init_train_off = [(init_time, LOW)]
    if init_color < 589 :
        green_train.extend(init_train_on)
        red_train.extend(init_train_off)
    if init_color > 589:
        green_train.extend(init_train_off)
        red_train.extend(init_train_on)
    green_train.extend([(galvo_move_time, LOW)])
    red_train.extend([(galvo_move_time, LOW)])
    deplete_train_on = [(depletion_time, HIGH)]
    deplete_train_off = [(depletion_time, LOW)]
    if depletion_color < 589 :
        green_train.extend(deplete_train_on)
        red_train.extend(deplete_train_off)
    if depletion_color > 589:
        green_train.extend(deplete_train_off)
        red_train.extend(deplete_train_on)
    green_train.extend([(galvo_move_time + uwave_experiment_shrt + wait_time + readout_scheme + wait_time, LOW)])
    red_train.extend([(galvo_move_time + uwave_experiment_shrt + wait_time + shelf_time, LOW), 
             (ion_time, HIGH), 
             (wait_time + readout_time + wait_time, LOW)])

    if init_color < 589 :
        green_train.extend(init_train_on)
        red_train.extend(init_train_off)
    if init_color > 589:
        green_train.extend(init_train_off)
        red_train.extend(init_train_on)
    green_train.extend([(galvo_move_time, LOW)])
    red_train.extend([(galvo_move_time, LOW)])
    if depletion_color < 589 :
        green_train.extend(deplete_train_on)
        red_train.extend(deplete_train_off)
    if depletion_color > 589:
        green_train.extend(deplete_train_off)
        red_train.extend(deplete_train_on)
    green_train.extend([(galvo_move_time + uwave_experiment_shrt + wait_time + readout_scheme + wait_time, LOW)])
    red_train.extend([(galvo_move_time + uwave_experiment_shrt + wait_time + shelf_time, LOW), 
             (ion_time, HIGH), 
             (wait_time + readout_time + wait_time, LOW)])

    if init_color < 589 :
        green_train.extend(init_train_on)
        red_train.extend(init_train_off)
    if init_color > 589:
        green_train.extend(init_train_off)
        red_train.extend(init_train_on)
    green_train.extend([(galvo_move_time, LOW)])
    red_train.extend([(galvo_move_time, LOW)])
    if depletion_color < 589 :
        green_train.extend(deplete_train_on)
        red_train.extend(deplete_train_off)
    if depletion_color > 589:
        green_train.extend(deplete_train_off)
        red_train.extend(deplete_train_on)
    green_train.extend([(galvo_move_time + uwave_experiment_long + wait_time + readout_scheme + wait_time, LOW)])
    red_train.extend([(galvo_move_time + uwave_experiment_long + wait_time + shelf_time, LOW), 
             (ion_time, HIGH), 
             (wait_time + readout_time + wait_time, LOW)])
    
    if init_color < 589 :
        green_train.extend(init_train_on)
        red_train.extend(init_train_off)
    if init_color > 589:
        green_train.extend(init_train_off)
        red_train.extend(init_train_on)
    green_train.extend([(galvo_move_time, LOW)])
    red_train.extend([(galvo_move_time, LOW)])
    if depletion_color < 589 :
        green_train.extend(deplete_train_on)
        red_train.extend(deplete_train_off)
    if depletion_color > 589:
        green_train.extend(deplete_train_off)
        red_train.extend(deplete_train_on)
    green_train.extend([(galvo_move_time + uwave_experiment_long + wait_time + 
                         readout_scheme + wait_time + green_delay, LOW)])
    red_train.extend([(galvo_move_time + uwave_experiment_long + wait_time + shelf_time, LOW), 
             (ion_time, HIGH), 
             (wait_time + readout_time + wait_time + red_delay, LOW)])
    
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            green_laser_name, None, green_train)
 
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            red_laser_name, None, red_train)
    


    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    config = tool_belt.get_config_dict()
            
    seq_args = [500000.0, 10000.0, 10000.0, 500, 0, 100.0, 1000.0, 
                0, 35, 515, 638, 'cobolt_515', 'laserglow_589', 'cobolt_638', 
                'signal_generator_bnc835', 0, 0.3, 0.4]
    # seq_args = [1000, 500, 1000, 100, 0, 1000, 100, 100, 50,
    #             515, 638,
    #             'cobolt_515', 'laserglow_589','cobolt_638', 
    #             'signal_generator_bnc835', 0, 0.5, 1.0]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:49:52 2020

This routine tests rabi under various readout routines: regular green readout,
regular yellow readout, and SCC readout.


@author: agardill
"""
import numpy
import matplotlib.pyplot as plt
import majorroutines.rabi as rabi
import majorroutines.rabi_SCC as rabi_SCC
from utils.tool_belt import States

# %%

def main(nv_sig):
    apd_indices = [0]
    num_steps = 51
    num_reps = 10**3
    num_runs = 1
    state = States.LOW
    uwave_time_range = [0, 200]
    
    # %% Run rabi with SCC readout
    per, sig_counts, ref_counts = rabi_SCC.main(nv_sig, apd_indices, uwave_time_range, state,
         num_steps, num_reps, num_runs)
    
    # Average the counts over the iterations
    avg_sig_counts = numpy.average(sig_counts, axis=0)
    avg_ref_counts = numpy.average(ref_counts, axis=0)

    # Calculate the Rabi data, signal / reference over different Tau
    # Replace x/0=inf with 0
    try:
        norm_avg_sig_scc = avg_sig_counts / avg_ref_counts
    except RuntimeWarning as e:
        print(e)
        inf_mask = numpy.isinf(norm_avg_sig_scc)
        # Assign to 0 based on the passed conditional array
        norm_avg_sig_scc[inf_mask] = 0
    

   # %% Run rabi with green readout
    per, sig_counts, ref_counts = rabi.main(nv_sig, apd_indices, uwave_time_range, state,
         num_steps, num_reps, num_runs)
    
    # Average the counts over the iterations
    avg_sig_counts = numpy.average(sig_counts, axis=0)
    avg_ref_counts = numpy.average(ref_counts, axis=0)
    
    # Replace x/0=inf with 0
    try:
        norm_avg_sig_green = avg_sig_counts / avg_ref_counts
    except RuntimeWarning as e:
        print(e)
        inf_mask = numpy.isinf(norm_avg_sig_green)
        # Assign to 0 based on the passed conditional array
        norm_avg_sig_green[inf_mask] = 0
        
    # %% Run rabi with yellow readout
    # replace SCC readout with shorter duration
    nv_sig['pulsed_SCC_readout_dur'] = nv_sig['pulsed_readout_dur'] 
    # replace the yellow readout power to something brighter
    nv_sig['am_589_power'] = 0.9
    # set the shelf_pulse and ion-pulse to 0 time
    nv_sig['pulsed_shelf_dur'] = 0
    nv_sig['pulsed_ionization_dur'] = 0
    
    per, sig_counts, ref_counts = rabi_SCC.main(nv_sig, apd_indices, uwave_time_range, state,
         num_steps, num_reps, num_runs)
    
    # Average the counts over the iterations
    avg_sig_counts = numpy.average(sig_counts, axis=0)
    avg_ref_counts = numpy.average(ref_counts, axis=0)

    # Calculate the Rabi data, signal / reference over different Tau
    # Replace x/0=inf with 0
    try:
        norm_avg_sig_yellow = avg_sig_counts / avg_ref_counts
    except RuntimeWarning as e:
        print(e)
        inf_mask = numpy.isinf(norm_avg_sig_yellow)
        # Assign to 0 based on the passed conditional array
        norm_avg_sig_yellow[inf_mask] = 0     

    # %% Compare on a signle plot
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(min_uwave_time, max_uwave_time,
                          num=num_steps)
    
        
    fig, ax = plt.subplots(1,1, figsize = (10, 8.5)) 
    ax.plot(taus, norm_avg_sig_green, 'g-', label = 'Standard green readout')
    ax.plot(taus, norm_avg_sig_scc, 'r-', label = 'SSC yellow readout')
#    ax.plot(taus, norm_avg_sig_yellow, 'y-', label = 'Standard yellow readout')
    ax.set_xlabel('Tau (ns)')
    ax.set_ylabel('Normalized contrast (arb.)')
    ax.legend()
    
# %%
if __name__ == '__main__':
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.2, 
            'pulsed_initial_ion_dur': 50*10**3,
            'pulsed_shelf_dur': 100, 'am_589_shelf_power': 0.2,
            'pulsed_ionization_dur': 450, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 10*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 187.8, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}   
    nv_sig = ensemble
    
    main(nv_sig)
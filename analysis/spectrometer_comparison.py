# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:54:20 2019

Spectrometer analysis

@author: Aedan
"""
# %%
import numpy
import matplotlib.pyplot as plt

# %%
counts_p_sec = 100000

int_time = 100

int_time_linspace = numpy.linspace(0,100, 1000)

counts_linspace = numpy.linspace(10, 1000000, 1000)

# %%
def SN_t(t, *params):
    quantum_eff, read_noise, dark_current, columns, rows, counts_p_sec, int_time = params
    
    photon_flux_p_pixel = counts_p_sec / (columns * rows)
    
    signal = t * quantum_eff * photon_flux_p_pixel
    
    N_R = read_noise**2
    N_D = dark_current * t 
    N_S = photon_flux_p_pixel * quantum_eff * t
    
    return signal / numpy.sqrt(N_R + N_D + N_S)

def SN_f(c, *params):
    quantum_eff, read_noise, dark_current, columns, rows, counts_p_sec, int_time = params
        
    photon_flux_p_pixel = c / (columns * rows)

    signal = int_time * quantum_eff * photon_flux_p_pixel
    
    N_R = read_noise**2
    N_D = dark_current * int_time 
    N_S = photon_flux_p_pixel * quantum_eff * int_time
    
    return signal / numpy.sqrt(N_R + N_D + N_S)

# %%
    
horiba_BIDD = (0.9, 4, 0.01, 256, 1024, counts_p_sec, int_time)
horiba_BI = (0.9, 4, 0.004, 256, 1024, counts_p_sec, int_time)
horiba_OE = (0.5, 4, 0.002, 256, 1024, counts_p_sec, int_time)

andor_BIDD = (0.9, 4, 0.008, 256, 1024, counts_p_sec, int_time)
andor_BI = (0.9, 4, 0.03, 255, 1024, counts_p_sec, int_time)
andor_OE = (0.5, 4, 0.0014, 255, 1024, counts_p_sec, int_time)

pi_BIDD = (0.9, 3, 0.03, 100, 1340, counts_p_sec, int_time)
pi_BI = (0.9, 3, 0.001, 100, 1340, counts_p_sec, int_time)
pi_OE = (0.5, 2.5, 0.0008, 100, 1340, counts_p_sec, int_time)

# %%

time_fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Integration time
ax.plot(int_time_linspace, SN_t(int_time_linspace, *horiba_BIDD), label = 'Horiba BIDD')
#ax.plot(int_time_linspace, SN_t(int_time_linspace, *horiba_BI), label = 'Horiba BI')
#ax.plot(int_time_linspace, SN_t(int_time_linspace, *horiba_OE), label = 'Horiba OE')

ax.plot(int_time_linspace, SN_t(int_time_linspace, *andor_BIDD), 'r--', label = 'Andor BIDD')
#ax.plot(int_time_linspace, SN_t(int_time_linspace, *andor_BI), label = 'Andor BI')
#ax.plot(int_time_linspace, SN_t(int_time_linspace, *andor_OE), label = 'Andor OE')

ax.plot(int_time_linspace, SN_t(int_time_linspace, *pi_BIDD), label = 'PI BIDD')
#ax.plot(int_time_linspace, SN_t(int_time_linspace, *pi_BI), label = 'PI BI')
#ax.plot(int_time_linspace, SN_t(int_time_linspace, *pi_OE), label = 'PI FI')

ax.set_xlabel('Integration time (s)')
ax.set_ylabel('S/N ratio')
ax.set_title('S/N ratio vs integration time, given 100 kcps photons')
ax.legend()
time_fig.canvas.draw()
time_fig.canvas.flush_events()

# Photon flux

flux_fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.plot(counts_linspace, SN_f(counts_linspace, *horiba_BIDD), label = 'Horiba BIDD')
#ax.plot(counts_linspace, SN_f(counts_linspace, *horiba_BI), label = 'Horiba BI')
#ax.plot(counts_linspace, SN_f(counts_linspace, *horiba_OE), label = 'Horiba OE')

ax.plot(counts_linspace, SN_f(counts_linspace, *andor_BIDD), 'r--', label = 'Andor BIDD')
#ax.plot(counts_linspace, SN_f(counts_linspace, *andor_BI), label = 'Andor BI')
#ax.plot(counts_linspace, SN_f(counts_linspace, *andor_OE), label = 'Andor OE')

ax.plot(counts_linspace, SN_f(counts_linspace, *pi_BIDD), label = 'PI BIDD')
#ax.plot(counts_linspace, SN_f(counts_linspace, *pi_BI), label = 'PI BI')
#ax.plot(counts_linspace, SN_f(counts_linspace, *pi_OE), label = 'PI FI')

ax.set_xlabel('Collected photon rate (counts/s)')
ax.set_ylabel('S/N ratio')
ax.set_title('S/N ratio vs photon counts, given 100 s integration time')
ax.legend()
flux_fig.canvas.draw()
flux_fig.canvas.flush_events()





import numpy
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import matplotlib.pyplot as plt

SNR_C = 0.01
SNR_SCC = 0.3
t_SCC = 50000 #us
t_C = 0.3 #us


T=(SNR_C**2 * t_SCC - SNR_SCC**2 * t_C ) / (SNR_SCC**2 - SNR_C**2)

print('{} us or longer'.format(T))
# 

kpl.init_kplotlib
    
label_list = ['0.35 V,10 ms', '0.35 V, 5 ms']
t_r_list_ms = [10, 5,]
snr_list = [0.1, 0.04]

# SNR_t = SNR_s * sqrt(T / (te + tr))
def total_snr_per_meas_dur(t, snr, t_r):
    return snr / numpy.sqrt(t + t_r)

smooth_t = numpy.linspace(0, 15, 1000)

fig, ax = plt.subplots()

for i in range(len(label_list)):
    ax.plot(smooth_t, total_snr_per_meas_dur(smooth_t, snr_list[i], t_r_list_ms[i]),
                    label = '{}'.format(label_list[i]))
ax.set_xlabel('Coherence length (ms)')
ax.set_ylabel('SNR ( sqrt(ms)^-1 )')
ax.legend()


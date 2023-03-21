import numpy
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import matplotlib.pyplot as plt

# SNR_C = 0.025
# SNR_SCC = 0.1
# t_SCC = 50000 #us
# t_C = 0.3 #us


# SNR_C = 0.025
# SNR_SCC = 0.5
# t_SCC = 50000 #us
# t_C = 0.3 #us

# SNR_C = 0.012
# SNR_SCC = 0.255
# t_SCC = 10000 #us
# t_C = 0.3 #us


# T=(SNR_C**2 * t_SCC - SNR_SCC**2 * t_C ) / (SNR_SCC**2 - SNR_C**2)

# print('{} us or longer'.format(T))
# 

kpl.init_kplotlib
    
label_list = ['0.15 V, 20 ms','0.2 V, 15 ms','0.25 V, 15 ms','0.25 V, 10 ms', '0.3 V, 5 ms', '0.35 V, 5 ms']
t_r_list_ms = [20, 15, 15, 10, 5, 5]
snr_list = [0.13, 0.22, 0.287, 0.22, 0.228, 0.179]

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


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:33:08 2022

@author: kolkowitz
"""
import numpy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def stretch_exp(x, o, a, d, n):
    return o + a*numpy.exp(-(x/d)**n)
    
folder = 'pc_rabi/branch_master/dynamical_decoupling_xy4/2022_08'
file = '2022_08_23-11_32_09-rubin-nv1'

contrast = 0.167*2

data = tool_belt.get_raw_data(file, folder)
sig_counts = numpy.array(data['sig_counts'])
ref_counts = numpy.array(data['ref_counts'])
precession_time_range = data['precession_time_range']
num_xy4_reps = data['num_xy4_reps']
num_runs = data['num_runs']
num_steps = data['num_steps']

# calc taus
min_precession_time = int(precession_time_range[0])
max_precession_time = int(precession_time_range[1])

taus = numpy.linspace(
    min_precession_time,
    max_precession_time,
    num=num_steps,
)
plot_taus = (taus * 2 *8* num_xy4_reps) / 1000
taus_linspace = numpy.linspace(plot_taus[0], plot_taus[-1], 100
                               )
# calc norm sig and ste
norm_sig = sig_counts / ref_counts
norm_avg_sig = numpy.average(norm_sig, axis=0)
norm_avg_sig_ste = numpy.std(
    norm_sig, axis=0, ddof=1
) / numpy.sqrt(num_runs)

# convert norm sig to coherence
norm_avg_coherence = (1- norm_avg_sig)/(contrast)
norm_avg_coherence_ste = norm_avg_coherence* norm_avg_sig_ste/norm_avg_sig

# fit to stretched exponential
fit_func = lambda x, d, n:stretch_exp(x, 0.5, 1, d, n)
init_params = [100, 1]
opti_params, cov_arr = curve_fit(
    fit_func,
    plot_taus,
    norm_avg_coherence,
    p0=init_params,
    sigma=norm_avg_coherence_ste,
    absolute_sigma=True,
    )

fig, ax = plt.subplots()
ax.errorbar(
        plot_taus,
        norm_avg_coherence,
        yerr=norm_avg_coherence_ste,
        fmt="o",
        color="blue",
        label="data",
    )

ax.plot(
    taus_linspace,
    fit_func(taus_linspace, *opti_params),
    "r",
    label="fit",
)
                    
ax.set_xlabel("Coherence time, T (us)")
ax.set_ylabel("Normalized signal Counts")
ax.set_title('XY4-{}'.format(num_xy4_reps))
ax.legend()
text = "d = {:.2f} us\nn= {:.2f}".format(opti_params[0], opti_params[1])

props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(
    0.55,
    0.9,
    text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=props,
)

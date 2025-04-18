import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
from numpy import pi
from numpy.linalg import eigvals
from scipy.optimize import curve_fit, minimize_scalar

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors
from utils.tool_belt import NormStyle, States

file = "2023_04_03-11_04_08-siena-nv0_2023_03_20"  # 2

data = tool_belt.get_raw_data(file)

comp_taus = data["comp_taus"]
norm_avg_sig_list = data["norm_avg_sig_list"]
norm_avg_sig_ste_list = data["norm_avg_sig_ste_list"]
num_steps_coh = data["num_steps_coh"]
pi_pulse_reps = data["pi_pulse_reps"]
taus_coh = data["taus_coh"]

kpl.init_kplotlib()
fig, ax = plt.subplots()

color_list = [
    KplColors.BLUE,
    KplColors.RED,
    KplColors.GREEN,
    KplColors.ORANGE,
    KplColors.PURPLE,
]
for ind in [0, 1]:
    kpl.plot_points(
        ax,
        comp_taus,
        norm_avg_sig_list[ind],
        yerr=norm_avg_sig_ste_list[ind],
        label="Coherence at T = {} ms".format(taus_coh[ind] / 1e3),
        color=color_list[ind],
    )
ax.set_xlabel(r"Timing between composite pulses (ns)")
ax.set_ylabel("Contrast (arb. units)")
ax.set_title("CPMG-{} {} SCC Measurement".format(pi_pulse_reps, "DQ"))
ax.legend()

fig, ax = plt.subplots()
for ind in [0, 2]:
    kpl.plot_points(
        ax,
        comp_taus,
        norm_avg_sig_list[ind],
        yerr=norm_avg_sig_ste_list[ind],
        label="Coherence at T = {} ms".format(taus_coh[ind] / 1e3),
        color=color_list[ind],
    )
ax.set_xlabel(r"Timing between composite pulses (ns)")
ax.set_ylabel("Contrast (arb. units)")
ax.set_title("CPMG-{} {} SCC Measurement".format(pi_pulse_reps, "DQ"))
ax.legend()


fig, ax = plt.subplots()
for ind in [0, 3]:
    kpl.plot_points(
        ax,
        comp_taus,
        norm_avg_sig_list[ind],
        yerr=norm_avg_sig_ste_list[ind],
        label="Coherence at T = {} ms".format(taus_coh[ind] / 1e3),
        color=color_list[ind],
    )
ax.set_xlabel(r"Timing between composite pulses (ns)")
ax.set_ylabel("Contrast (arb. units)")
ax.set_title("CPMG-{} {} SCC Measurement".format(pi_pulse_reps, "DQ"))
ax.legend()


fig, ax = plt.subplots()
for ind in [0, 4]:
    kpl.plot_points(
        ax,
        comp_taus,
        norm_avg_sig_list[ind],
        yerr=norm_avg_sig_ste_list[ind],
        label="Coherence at T = {} ms".format(taus_coh[ind] / 1e3),
        color=color_list[ind],
    )
ax.set_xlabel(r"Timing between composite pulses (ns)")
ax.set_ylabel("Contrast (arb. units)")
ax.set_title("CPMG-{} {} SCC Measurement".format(pi_pulse_reps, "DQ"))
ax.legend()

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:47:43 2021

@author: kolkowitz
"""

import matplotlib.pyplot as plt
import time

    
plt.ion()
fig, ax = plt.subplots()
plot_times = [0]
plot_temps = [0]
plot_period = 2  # Plot every plot_period seconds
# Don't actually plot yet because matplotlib gets confused trying
# to plot a line with just one data point
history = 600
max_plot_vals = history / plot_period
plot_x_extent = 11
ax.set_xlim(0, plot_x_extent)
# ax.set_ylim(actual-2, actual+2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Temp (K)")
fig.canvas.draw()
fig.canvas.flush_events()
first_plot_call = True

for i in range(10):
    
    plot_times.append(i+1)
    plot_temps.append(i*2 + 1)
    
    if first_plot_call:
        first_plot_call = False
        ax.plot(plot_times, plot_temps)
    else:
        lines = ax.get_lines()
        line = lines[0]
        line.set_xdata(plot_times)
        line.set_ydata(plot_temps)

    # Relim as necessary
    if len(plot_times) > max_plot_vals:
        plot_times.pop(0)
        plot_temps.pop(0)
        min_plot_time = min(plot_times)
        ax.set_xlim(min_plot_time, min_plot_time + plot_x_extent)
    ax.set_ylim(min(plot_temps)-2, max(plot_temps)+2)
        
    # Tell the canvas to redraw from scratch so there's 
    # only one line
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    time.sleep(1)
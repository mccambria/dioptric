# -*- coding: utf-8 -*-
"""
Plotting and exploring hysteresis curves

2021/08/03 mccambria
"""


import matplotlib.pyplot as plt
import numpy as np


def hysteresis_curve(voltage, prev_turning_delta, 
                     prev_turning_voltage, prev_turning_pos,
                     full_scale_voltage, a, b):

    # scale = abs(prev_turning_delta / full_scale_voltage)
    # scale = abs(prev_turning_delta / full_scale_voltage)
    scale = 1
    diff = abs(voltage - prev_turning_voltage)
    response = lambda diff : scale * (a * diff**2 + b * diff)
    # scaled_diff = diff * scale
    # response = lambda diff : scale * (a * scaled_diff**2 + b * scaled_diff)

    if voltage > prev_turning_voltage: 
        return prev_turning_pos + response(diff)
    else:
        return prev_turning_pos - response(diff)


def plot_hysteresis(turning_voltages, full_scale_voltage, a, b):

    step_size = 0.01

    prev_turning_delta = 10
    prev_turning_voltage = 0
    prev_turning_pos = 0
    voltages = []
    positions = []
    for turning_voltage in turning_voltages:
        num_steps = round((turning_voltage - prev_turning_voltage) / step_size)
        num_steps = abs(num_steps)
        segment_voltages = np.linspace(prev_turning_voltage, turning_voltage, 
                                       num_steps)
        segment_positions = []
        for vol in segment_voltages:
            pos = hysteresis_curve(vol, prev_turning_delta, 
                                   prev_turning_voltage, prev_turning_pos,
                                   full_scale_voltage, a, b)
            segment_positions.append(float(pos))
        voltages.extend(segment_voltages)
        positions.extend(segment_positions)
        if turning_voltage > prev_turning_voltage:
            # prev_turning_delta = turning_voltage - prev_turning_voltage
            prev_turning_delta = turning_voltage
        prev_turning_voltage = turning_voltage
        prev_turning_pos = segment_positions[-1]

    fig, ax = plt.subplots()
    ax.plot(voltages, positions)
    ax.set_xlabel('Applied value')
    ax.set_ylabel('Actual value')
    plt.show()



if __name__ == '__main__':

    full_scale_voltage = 10
    a = 0.07
    b = 0.3

    # turning_voltages = [10, 0, 8, 0, 6, 0]  # Loop scale test
    # turning_voltages = [10, 2, 10, 4, 10, 6, 10]  # Reverse loop scale test
    # turning_voltages = [10, 0, 7.5, 2.5, 7.5, 2.5, 7.5]  # Halving the drive
    # turning_voltages = [10, 1, 9, 8, 2, 7, 3, 10]  # Spiral
    turning_voltages = [10, 0, 9, 1, 8, 2, 7, 4]  # Spiral simple
    plot_hysteresis(turning_voltages, full_scale_voltage, a, b)
# -*- coding: utf-8 -*-
"""
Plotting and exploring hysteresis curves

2021/08/03 mccambria
"""


import matplotlib.pyplot as plt
import numpy as np


def invert_hysteresis(position, prev_turning_position, a, b):
    # The hysteresis curve is p(v) = a * v**2 + b * v
    # We want to feedforward using this curve to set the piezo voltage
    # such that the nominal voltage passed by the user functions 
    # linearly and without hysteresis. The goal is to prevent the
    # accumulation of small errors until active feedback (eg 
    # optimizing on an NV) can be performed
    
    # The adjustment voltage we need is obtained by inverting p(v)
    p = position - prev_turning_position
    abs_p = abs(p)
    v = (-b + np.sqrt(b**2 + 4 * a * abs_p)) / (2 * a)
    
    # return position
    return prev_turning_position + (np.sign(p) * v)

def hysteresis_curve(voltage, prev_turning_delta, 
                     prev_turning_voltage, prev_turning_pos,
                     full_scale_voltage, a, b):

    # scale = abs(prev_turning_delta / full_scale_voltage)
    # scale = abs(prev_turning_delta / full_scale_voltage)
    # scale = 1
    # scaled_diff = diff * scale
    # response = lambda diff : scale * (a * scaled_diff**2 + b * scaled_diff)
    diff = voltage - prev_turning_voltage
    response = lambda diff : a * diff**2 + b * diff

    return prev_turning_pos + (np.sign(diff) * response(abs(diff)))


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
            vol = invert_hysteresis(vol, prev_turning_voltage, a, b)
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


if __name__ == '__main__':

    full_scale_voltage = 10
    a = 0.07
    b = 0.3
    
    exps = [
        [10, 0, 8, 0, 6, 0],  # Loop scale test
        [10, 2, 10, 4, 10, 6, 10],  # Reverse loop scale test
        [10, 0, 7.5, 2.5, 7.5, 2.5, 7.5],  # Halving the drive
        [10, 0, 9, 1, 8, 2, 7, 4],  # Spiral simple
        [10, 0, 9, 1, 8, 2, 7, 3, 6, 4, 6],  # Spiral simple 2
        [10, 0, 5, 4, 10],  # Spiral simple 2
        [10]
    ]
    for exp in exps:
        plot_hysteresis(exp, full_scale_voltage, a, b)
    plt.show()
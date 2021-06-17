# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:19:53 2020

@author: kolkowitz
"""
import numpy

x_range = 3
y_range = 3
num_steps = 4
x_center  =2.5
y_center = 2.5
######### Assumes x_range == y_range #########


if x_range != y_range:
    raise ValueError('x_range must equal y_range for now')

x_num_steps = num_steps
y_num_steps = num_steps

# Force the scan to have square pixels by only applying num_steps
# to the shorter axis
half_x_range = x_range / 2
half_y_range = y_range / 2

x_low = x_center - half_x_range
x_high = x_center + half_x_range
y_low = y_center - half_y_range
y_high = y_center + half_y_range

# Apply scale and offset to get the voltages we'll apply to the galvo
# Note that the polar/azimuthal angles, not the actual x/y positions
# are linear in these voltages. For a small range, however, we don't
# really care.
x_voltages_1d = numpy.linspace(x_low, x_high, num_steps)
y_voltages_1d = numpy.linspace(y_low, y_high, num_steps)

######### Works for any x_range, y_range #########

# Winding cartesian product
# The x values are repeated and the y values are mirrored and tiled
# The comments below shows what happens for [1, 2, 3], [4, 5, 6]

# [1, 2, 3] => [1, 2, 3, 3, 2, 1]
x_inter = numpy.concatenate((numpy.flipud(x_voltages_1d), x_voltages_1d
                             ))
# [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
if y_num_steps % 2 == 0:  # Even x size
    x_voltages = numpy.tile(x_inter, int(y_num_steps/2))
else:  # Odd x size
    x_voltages = numpy.tile(x_inter, int(numpy.floor(y_num_steps/2)))
    x_voltages = numpy.concatenate((x_voltages, x_voltages_1d))

# [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
y_voltages = numpy.repeat(y_voltages_1d, x_num_steps)

voltages = numpy.vstack((x_voltages, y_voltages))

print(x_voltages)
print(y_voltages)
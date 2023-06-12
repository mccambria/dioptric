# -*- coding: utf-8 -*-
"""Various simple laser routines for aligning, imaging a sample in
reflection, etc

Created on June 16th, 2019

@author: mccambria
"""


import labrad
import utils.tool_belt as tool_belt
import utils.positioning as positioning


def constant(cxn, laser_name, laser_power=None):

    tool_belt.laser_on(cxn, laser_name, laser_power)
    tool_belt.poll_safe_stop()
    tool_belt.laser_off(cxn, laser_name)


def square_wave(cxn, laser_name, laser_power=None):
    """Run a laser on on a square wave."""

    # period = int(2e4)
    # period = int(350*2)
    # period = int(1000)
    # period = int(0.25e6)
    period = int(1e6)

    seq_file = "square_wave.py"
    
    charge_readout_laser_server = tool_belt.get_server_charge_readout_laser(cxn)
    charge_readout_laser_server.load_feedthrough(1)
    
    seq_args = [period, laser_name, laser_power]
    pulse_gen = tool_belt.get_server_pulse_gen(cxn)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    pulse_gen.stream_immediate(seq_file, -1, seq_args_string)
    tool_belt.poll_safe_stop()
    tool_belt.laser_off(cxn, laser_name)


def arb_duty_cycle(cxn, laser_name, laser_power=None):
    """Run a laser on on a square wave."""

    period_1 = 400
    wait_1 = 1e6
    period_2 = 400
    wait_2 = 1e6

    seq_file = "square_wave_arb_duty_cycle.py"
    seq_args = [wait_1, period_1, wait_2, period_2, laser_name, laser_power]
    pulse_gen = tool_belt.get_pulse_gen_server(cxn)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    pulse_gen.stream_immediate(seq_file, -1, seq_args_string)
    tool_belt.poll_safe_stop()
    tool_belt.laser_off(cxn, laser_name)


def circle(cxn, laser_name, laser_power=None):
    """Run a laser around in a circle"""

    period = int(10e6)
    radius = 1.0
    num_steps = 300

    seq_file = "simple_readout.py"
    seq_args = [period, period, laser_name, laser_power]
    xy_server = positioning.get_server_pos_xy(cxn)
    coords_x, coords_y = positioning.get_scan_circle_2d(0, 0, radius, num_steps)
    xy_server.load_stream_xy(coords_x, coords_y, True)
    pulse_gen = tool_belt.get_server_pulse_gen(cxn)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    pulse_gen.stream_immediate(seq_file, -1, seq_args_string)
    tool_belt.poll_safe_stop()
    tool_belt.laser_off(cxn, laser_name)


if __name__ == "__main__":

    laser_name = "laserglow_532"
    # laser_name = "laser_LGLO_589"
    # laser_name = "integrated_520"
    laser_power = None
    laser_filter = "nd_0.3"
    collection_filter = "nd_0"
    # pos = [-0.306, 0.235, 8.0]
    z_coord = 5.0
    pos = [0.0, 0.0, z_coord]

    tool_belt.init_safe_stop()
    
 
    with labrad.connect() as cxn:

        # positioning.set_xyz(cxn, pos)
        tool_belt.set_filter(
            cxn, optics_name=laser_name, filter_name=laser_filter
        )
        # tool_belt.set_filter(
        #     cxn, optics_name="collection", filter_name=collection_filter
        # )

        # Some parameters you'll need to set in these functions
        constant(cxn, laser_name)
        # square_wave(cxn, laser_name)
        # arb_duty_cycle(cxn, laser_name)
        # circle(cxn, laser_name)

    tool_belt.reset_cfm()
    tool_belt.reset_safe_stop()

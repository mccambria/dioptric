# -*- coding: utf-8 -*-
"""
This file contains functions, classes, and other objects that are useful
in a variety of contexts. Since they are expected to be used in many
files, I put them all in one place so that they don't have to be redefined
in each file.

Created on Fri Nov 23 14:57:08 2018

@author: mccambria
"""


# %% Imports


import matplotlib.pyplot as plt
import threading
import os
import datetime
import numpy
from numpy import exp
import json
import time
import labrad
from tkinter import Tk
from tkinter import filedialog
from git import Repo
from pathlib import Path
from pathlib import PurePath
from enum import Enum, auto
import socket


# %% Constants


class States(Enum):
    LOW = auto()
    ZERO = auto()
    HIGH = auto()

#Digital = {'LOW': 0, 'HIGH': 1}

class Mod_types(Enum):
    DIGITAL = auto()
    ANALOG = auto()

def get_signal_generator_name_no_cxn(state):
    with labrad.connect() as cxn:
        return get_signal_generator_name(cxn, state)

def get_signal_generator_name(cxn, state):
    return get_registry_entry(cxn, 'sig_gen_{}'.format(state.name),
                              ['', 'Config', 'Microwaves'])

def get_signal_generator_cxn(cxn, state):
    signal_generator_name = get_signal_generator_name(cxn, state)
    signal_generator_cxn = eval('cxn.{}'.format(signal_generator_name))
    return signal_generator_cxn


# %% xyz sets


def set_xyz(cxn, coords):
    xy_dtype = eval(get_registry_entry(cxn, 'xy_dtype',
                                       ['', 'Config', 'Positioning']))
    z_dtype = eval(get_registry_entry(cxn, 'z_dtype',
                                       ['', 'Config', 'Positioning']))
    xy_server = get_xy_server(cxn)
    z_server = get_z_server(cxn)
    if xy_dtype is int:
        xy_op = round
    else:
        xy_op = xy_dtype
    if z_dtype is int:
        z_op = round
    else:
        z_op = z_dtype
    xy_server.write_xy(xy_op(coords[0]), xy_op(coords[1]))
    z_server.write_z(z_op(coords[2]))
    # Force some delay before proceeding to account
    # for the effective write time
    time.sleep(0.002)


def set_xyz_center(cxn):
    # MCC Generalize this for Hahn
    set_xyz(cxn, [0, 0, 5])


def set_xyz_on_nv(cxn, nv_sig):
    set_xyz(cxn, nv_sig['coords'])
    
    
# %% Laser utils


def laser_off(cxn, laser_name):
    laser_switch_sub(cxn, False, laser_name)

def laser_on(cxn, laser_name, laser_power=None):
    laser_switch_sub(cxn, True, laser_name, laser_power)
    
def laser_switch_sub(cxn, turn_on, laser_name, laser_power=None):
    
    mod_type = get_registry_entry(cxn, 'mod_type', 
                                  ['', 'Config', 'Optics', laser_name])
    mod_type = eval(mod_type)
    feedthrough = get_registry_entry(cxn, 'feedthrough',
                                     ['', 'Config', 'Optics', laser_name])
    feedthrough = eval(feedthrough)
    
    if mod_type is Mod_types.DIGITAL:
        # Digital, feedthrough
        if feedthrough:
            if turn_on:
                cxn[laser_name].laser_on()
            else:
                cxn[laser_name].laser_off()
        # Digital, no feedthrough
        else:  
            if turn_on:
                laser_chan = get_registry_entry(cxn, 'do_{}_dm'.format(laser_name),
                                     ['', 'Config', 'Wiring', 'PulseStreamer'])
                cxn.pulse_streamer.constant([laser_chan])
    # Analog
    elif mod_type is Mod_types.ANALOG:  
        if turn_on:
            laser_chan = get_registry_entry(cxn, 'do_{}_dm'.format(laser_name),
                                     ['', 'Config', 'Wiring', 'PulseStreamer'])
            if laser_chan == 0:
                cxn.pulse_streamer.constant([], 0.0, laser_power)
            elif laser_chan == 1:
                cxn.pulse_streamer.constant([], laser_power, 0.0)
                
    # If we're turning things off, turn everything off. If we wanted to really
    # do this nicely we'd find a way to only turn off the specific channel,
    # but it's not worth the effort.
    if not turn_on:
        cxn.pulse_streamer.constant([])
            
        

def set_laser_power(cxn, nv_sig=None, laser_key=None,
                    laser_name=None, laser_power=None):
    """
    Set a laser power, or return it for analog modulation.
    Specify either a laser_key/nv_sig or a laser_name/laser_power.
    """
    
    if (nv_sig is not None) and (laser_key is not None):
        laser_name = nv_sig[laser_key]
        power_key = '{}_power'.format(laser_key)
        # If the power isn't specified, then we assume it's set some other way
        if power_key in nv_sig:
            laser_power = nv_sig[power_key]
    elif (laser_name is not None) and (laser_power is not None):
        pass  # All good
    else:
        raise Exception('Specify either a laser_key/nv_sig or a laser_name/laser_power.')
    
    # If the power is controlled by analog modulation, we'll need to pass it
    # to the pulse streamer
    mod_type = get_registry_entry(cxn, 'mod_type',
                                  ['', 'Config', 'Optics', laser_name])
    mod_type = eval(mod_type)
    if mod_type == Mod_types.ANALOG:
        return laser_power
    else:
        laser_server = get_filter_server(cxn, laser_name)
        if (laser_power is not None) and (laser_server is not None):
            laser_server.set_laser_power(laser_power)
        return None  
    

def set_filter(cxn, nv_sig=None, optics_key=None,
               optics_name=None, filter_name=None):
    """
    optics_key should be either 'collection' or a laser key.
    Specify either an optics_key/nv_sig or an optics_name/filter_name.
    """
    
    if (nv_sig is not None) and (optics_key is not None):
        optics_name = nv_sig[optics_key]
        filter_key = '{}_filter'.format(optics_key)
        # Just exit if there's no filter specified in the nv_sig
        if filter_key not in nv_sig:
            return
        filter_name = nv_sig[filter_key]
    elif (optics_name is not None) and (filter_name is not None):
        pass  # All good
    else:
        raise Exception('Specify either an optics_key/nv_sig or an optics_name/filter_name.')
    
    filter_server = get_filter_server(cxn, optics_name)
    if filter_server is None:
        return
    pos = get_registry_entry(cxn, filter_name,
                     ['', 'Config', 'Optics', optics_name, 'FilterMapping'])
    filter_server.set_filter(pos)


def get_filter_server(cxn, optics_name):
    """
    Try to get a filter server. If there isn't one listed on the registry, 
    just return None.
    """
    
    try:
        server_name = get_registry_entry(cxn, 'filter_server',
                                     ['', 'Config', 'Optics', optics_name])
        return getattr(cxn, server_name)
    except Exception:
        return None


def get_laser_server(cxn, laser_name):
    """
    Try to get a laser server. If there isn't one listed on the registry, 
    just return None.
    """

    try:
        server_name = get_registry_entry(cxn, 'laser_server',
                                     ['', 'Config', 'Optics', laser_name])
        return getattr(cxn, server_name)
    except Exception:
        return None
    
    
def process_laser_seq(pulse_streamer, seq, config, 
                      laser_name, laser_power, train):
    """
    Some lasers may require special processing of their Pulse Streamer
    sequence. For example, the Cobolt lasers expect 3.5 V for digital
    modulation, but the Pulse Streamer only supplies 2.6 V.
    """
    
    pulser_wiring = config['Wiring']['PulseStreamer']
    mod_type = config['Optics'][laser_name]['mod_type']
    mod_type = eval(mod_type)
    feedthrough = config['Optics'][laser_name]['feedthrough']
    feedthrough = eval(feedthrough)
#    feedthrough = False
    
    LOW = 0
    HIGH = 1
        
    processed_train = []
    
    if mod_type is Mod_types.DIGITAL:
        # Digital, feedthrough, bookend each pulse with 100 ns clock pulses
        # Assumes we always leave the laser on (or off) for at least 100 ns
        if feedthrough:
            # Collapse the sequence so that no two adjacent elements have the
            # same value
            collapsed_train = []
            ind = 0
            len_train = len(train)
            while ind < len_train:
                el = train[ind]
                dur = el[0]
                val = el[1]
                next_ind = ind+1
                while next_ind < len_train:
                    next_el = train[next_ind]
                    next_dur = next_el[0]
                    next_val = next_el[1]
                    # If the next element shares the same value as the current
                    # one, combine them
                    if next_val == val:
                        dur += next_dur
                        next_ind += 1
                    else:
                        break
                # Append the current pulse and start back 
                # where we left off
                collapsed_train.append((dur, val))
                ind = next_ind
            # Check if this is just supposed to be always on
            if (len(collapsed_train) == 1) and (collapsed_train[0][1] == HIGH):
                if pulse_streamer is not None:
                    pulse_streamer.client[laser_name].turn_on()
                return
            # Set up the bookends
            for ind in range(len(collapsed_train)):
                el = collapsed_train[ind]
                dur = el[0]
                val = el[1]
                # For the first element, just leave things LOW
                # Assumes the laser is off prior to the start of the sequence
                if (ind == 0) and (val is LOW):
                    processed_train.append((dur, LOW))
                    continue
                processed_train.append((100, HIGH))
                processed_train.append((dur-100, LOW))
        # Digital, no feedthrough, do nothing
        else:  
            processed_train = train.copy()
        pulser_laser_mod = pulser_wiring['do_{}_dm'.format(laser_name)]
        seq.setDigital(pulser_laser_mod, processed_train)
        
    # Analog, convert LOW / HIGH to 0.0 / analog voltage
    elif mod_type is Mod_types.ANALOG:  
        power_dict = {LOW: 0.0, HIGH: laser_power}
        for el in train:
            dur = el[0]
            val = el[1]
            processed_train.append((dur, power_dict[val]))
            
        pulser_laser_mod = pulser_wiring['ao_{}_am'.format(laser_name)]
        seq.setAnalog(pulser_laser_mod, processed_train)


# %% Pulse Streamer utils


def encode_seq_args(seq_args):
    # Recast numpy ints to Python ints so json knows what to do
    for ind in range(len(seq_args)):
        el = seq_args[ind]
        if type(el) is numpy.int32:
            seq_args[ind] = int(el)
    return json.dumps(seq_args)

def decode_seq_args(seq_args_string):
    if seq_args_string == '':
        return []
    else:
        return json.loads(seq_args_string)

def get_pulse_streamer_wiring(cxn):
    cxn.registry.cd(['', 'Config', 'Wiring', 'Pulser'])
    sub_folders, keys = cxn.registry.dir()
    if keys == []:
        return {}
    p = cxn.registry.packet()
    for key in keys:
        p.get(key, key=key)  # Return as a dictionary
    wiring = p.send()
    pulse_streamer_wiring = {}
    for key in keys:
        pulse_streamer_wiring[key] = wiring[key]
    return pulse_streamer_wiring

def get_tagger_wiring(cxn):
    cxn.registry.cd(['', 'Config', 'Wiring', 'Tagger'])
    sub_folders, keys = cxn.registry.dir()
    if keys == []:
        return {}
    p = cxn.registry.packet()
    for key in keys:
        p.get(key, key=key)  # Return as a dictionary
    wiring = p.send()
    tagger_wiring = {}
    for key in keys:
        tagger_wiring[key] = wiring[key]
    return tagger_wiring


# %% Matplotlib plotting utils


def create_image_figure(imgArray, imgExtent, clickHandler=None, title=None, 
                color_bar_label='Counts', min_value=None, um_scaled=False):
    """
    Creates a figure containing a single grayscale image and a colorbar.

    Params:
        imgArray: numpy.ndarray
            Rectangular numpy array containing the image data.
            Just zeros if you're going to be writing the image live.
        imgExtent: list(float)
            The extent of the image in the form [left, right, bottom, top]
        clickHandler: function
            Function that fires on clicking in the image

    Returns:
        matplotlib.figure.Figure
    """
    
    if um_scaled:
        axes_label = r'$\mu$m'
    else:
        axes_label = get_registry_entry_no_cxn('xy_units', ['', 'Config', 'Positioning'])
        
    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots()

    fig.set_tight_layout(True)

    # Tell the axes to show a grayscale image
    img = ax.imshow(imgArray, cmap='inferno',
                    extent=tuple(imgExtent), vmin = min_value)

#    if min_value == None:
#        img.autoscale()

    # Add a colorbar
    clb = plt.colorbar(img)
    clb.set_label(color_bar_label)
    # clb.ax.set_tight_layout(True)
    # clb.ax.set_title(color_bar_label)
#    clb.set_label('kcounts/sec', rotation=270)

    # Label axes
    plt.xlabel(axes_label)
    plt.ylabel(axes_label)
    if title:
        plt.title(title)

    # Wire up the click handler to print the coordinates
    if clickHandler is not None:
        fig.canvas.mpl_connect('button_press_event', clickHandler)

    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig


def update_image_figure(fig, imgArray):
    """
    Update the image with the passed image array and redraw the figure.
    Intended to update figures created by create_image_figure.

    The implementation below isn't nearly the fastest way of doing this, but
    it's the easiest and it makes a perfect figure every time (I've found
    that the various update methods accumulate undesirable deviations from
    what is produced by this brute force method).

    Params:
        fig: matplotlib.figure.Figure
            The figure containing the image to update
        imgArray: numpy.ndarray
            The new image data
    """

    # Get the image - Assume it's the first image in the first axes
    axes = fig.get_axes()
    ax = axes[0]
    images = ax.get_images()
    img = images[0]

    # Set the data for the image to display
    img.set_data(imgArray)

    # Check if we should clip or autoscale
    clipAtThousand = False
    if clipAtThousand:
        if numpy.all(numpy.isnan(imgArray)):
            imgMax = 0  # No data yet
        else:
            imgMax = numpy.nanmax(imgArray)
        if imgMax > 1000:
            img.set_clim(None, 1000)
        else:
            img.autoscale()
    else:
        img.autoscale()

    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()


def create_line_plot_figure(vals, xVals=None):
    """
    Creates a figure containing a single line plot

    Params:
        vals: numpy.ndarray
            1D numpy array containing the values to plot
        xVals: numpy.ndarray
            1D numpy array with the x values to plot against
            Default is just the index of the value in vals

    Returns:
        matplotlib.figure.Figure
    """

    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots()

    if xVals is not None:
        ax.plot(xVals, vals)
        ax.set_xlim(xVals[0], xVals[len(xVals) - 1])
    else:
        ax.plot(vals)
        ax.set_xlim(0, len(vals)-1)

    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig


def create_line_plots_figure(vals, xVals=None):
    """
    Creates a figure containing a single line plot

    Params:
        vals: tuple(numpy.ndarray)
            1D numpy array containing the values to plot
        xVals: numpy.ndarray
            1D numpy array with the x values to plot against
            Default is just the index of the value in vals

    Returns:
        matplotlib.figure.Figure
    """

    # Tell matplotlib to generate a figure with len(vals) plots
    fig, ax = plt.subplots(len(vals))

    if xVals is not None:
        ax.plot(xVals, vals)
        ax.set_xlim(xVals[0], xVals[len(xVals) - 1])
    else:
        ax.plot(vals)
        ax.set_xlim(0, len(vals) - 1)

    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig


def update_line_plot_figure(fig, vals):
    """
    Updates a figure created by create_line_plot_figure

    Params:
        vals: numpy.ndarray
            1D numpy array containing the values to plot
    """

    # Get the line - Assume it's the first line in the first axes
    axes = fig.get_axes()
    ax = axes[0]
    lines = ax.get_lines()
    line = lines[0]

    # Set the data for the line to display and rescale
    line.set_ydata(vals)
    ax.relim()
    ax.autoscale_view(scalex=False)

    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()


# %% Math functions


def get_pi_pulse_dur(rabi_period):
    return round(rabi_period / 2)


def get_pi_on_2_pulse_dur(rabi_period):
    return round(rabi_period / 4)

def get_pi_pulse_dur(rabi_period):
    return round(rabi_period / 2)


def get_pi_on_2_pulse_dur(rabi_period):
    return round(rabi_period / 4)


def gaussian(x, *params):
    """
    Calculates the value of a gaussian for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the Gaussian
            0: coefficient that defines the peak height
            1: mean, defines the center of the Gaussian
            2: standard deviation, defines the width of the Gaussian
            3: constant y value to account for background
    """

    coeff, mean, stdev, offset = params
    var = stdev**2  # variance
    centDist = x-mean  # distance from the center
    return offset + coeff**2*numpy.exp(-(centDist**2)/(2*var))


def sinexp(t, offset, amp, freq, decay):
    two_pi = 2*numpy.pi
    half_pi = numpy.pi / 2
    return offset + (amp * numpy.sin((two_pi * freq * t) + half_pi)) * exp(-decay**2 * t)

# This cosexp includes a phase that will be 0 in the ideal case.
#def cosexp(t, offset, amp, freq, phase, decay):
#    two_pi = 2*numpy.pi
#    return offset + (numpy.exp(-t / abs(decay)) * abs(amp) * numpy.cos((two_pi * freq * t) + phase))


def cosexp(t, offset, amp, freq, decay):
    two_pi = 2*numpy.pi
    return offset + (numpy.exp(-t / abs(decay)) * abs(amp) * numpy.cos((two_pi * freq * t)))


def cosexp_1_at_0(t, offset, freq, decay):
    two_pi = 2*numpy.pi
    amp = 1- offset
    return offset + (numpy.exp(-t / abs(decay)) * abs(amp) * numpy.cos((two_pi * freq * t)))

def cosine_sum(t, offset, decay, amp_1, freq_1, amp_2, freq_2, amp_3, freq_3):
    two_pi = 2*numpy.pi

    return offset + numpy.exp(-t / abs(decay)) * (
                amp_1 * numpy.cos(two_pi * freq_1 * t) +
                amp_2 * numpy.cos(two_pi * freq_2 * t) +
                amp_3 * numpy.cos(two_pi * freq_3 * t))

def calc_snr(sig_count, ref_count):
    '''
    Take a list of signal and reference counts, and take their average, then
    calculate a snr.
    inputs:
        sig_count = list
        ref_counts = list
    outputs:
        snr = list
    '''

    sig_count_avg = numpy.average(sig_count)
    ref_count_avg = numpy.average(ref_count)
    dif = sig_count_avg - ref_count_avg
    noise = numpy.sqrt(ref_count_avg)
    snr = dif / noise

    return snr


def get_scan_vals(center, scan_range, num_steps, dtype=float):
    """
    Returns a linspace for a scan centered about specified point
    """

    half_scan_range = scan_range / 2
    low = center - half_scan_range
    high = center + half_scan_range
    scan_vals = numpy.linspace(low, high, num_steps, dtype=dtype)
    # Deduplicate - may be necessary for ints and low scan ranges
    scan_vals = numpy.unique(scan_vals)
    return scan_vals


# %% LabRAD utils


def get_config_dict(cxn=None):
    """Get the shared parameters from the registry. These parameters are not
    specific to any experiment, but are instead used across experiments. They
    may depend on the current alignment (eg aom_delay) or they may just be
    parameters that are referenced by many sequences (eg polarization_dur).
    Generally, they should need to be updated infrequently, unlike the
    shared parameters defined in cfm_control_panel, which change more
    frequently (eg apd_indices).

    We currently have the parameters listed below. All durations (ending in
    _delay or _dur) have units of ns.
        airy_radius: Standard deviation of the Gaussian approximation to
            the Airy disk in nm
        polarization_dur: Duration to illuminate for polarization
        post_polarization_wait_dur: Duration to wait after polarization to
            allow the NV metastable state to decay
        pre_readout_wait_dur: Duration to wait before readout - functionally
            I think this is just for symmetry with post_polarization_wait_dur
        532_aom_delay: Delay between signal to the 532 nm laser AOM and the
            AOM actually opening
        uwave_delay: Delay between signal to uwave switch and the switch
            actually opening - should probably be different for different
            signal generators...
        pulsed_readout_dur: Readout duration if we're looking to determine
            the state directly dorm fluorescence
        continuous_readout_dur: Readout duration if we're just looking to
            see how bright something is
        galvo_delay: Delay between signal to galvo and the galvo settling to
            its new position
        galvo_nm_per_volt: Conversion factor between galvo voltage and xy
            position
        piezo_delay: Delay between signal to objective piezo and the piezo
            settling to its new position
        piezo_nm_per_volt: Conversion factor between objective piezo voltage
            and z position
    """

    if cxn is None:
        with labrad.connect() as cxn:
            return get_config_dict_sub(cxn)
    else:
        return get_config_dict_sub(cxn)


def get_config_dict_sub(cxn):

    config_dict = {}
    populate_config_dict(cxn, ['', 'Config'], config_dict)
    return config_dict


def populate_config_dict(cxn, reg_path, dict_to_populate):
    """Populate the config dictionary recursively"""
    
    # Sub-folders
    cxn.registry.cd(reg_path)
    sub_folders, keys = cxn.registry.dir()
    for el in sub_folders:
        sub_dict = {}
        sub_path = reg_path + [el]
        populate_config_dict(cxn, sub_path, sub_dict)
        dict_to_populate[el] = sub_dict

    # Keys
    if len(keys) == 1:
        cxn.registry.cd(reg_path)
        p = cxn.registry.packet()
        key = keys[0]
        p.get(key)
        val = p.send()['get']
        dict_to_populate[key] = val
    
    elif len(keys) > 1:
        cxn.registry.cd(reg_path)
        p = cxn.registry.packet()
        for key in keys:
            p.get(key)
        vals = p.send()['get']
    
        for ind in range(len(keys)):
            key = keys[ind]
            val = vals[ind]
            dict_to_populate[key] = val
    

def get_nv_sig_units():
    return 'in config'


def get_xy_server(cxn):
    """
    Talk to the registry to get the fine xy control server for this setup.
    eg for rabi it is probably galvo. See optimize for some examples.
    """

    # return an actual reference to the appropriate server so it can just
    # be used directly
    return getattr(cxn, get_registry_entry(cxn, 'xy_server', ['', 'Config', 'Positioning']))


def get_z_server(cxn):
    """Same as get_xy_server but for the fine z control server"""

    return getattr(cxn, get_registry_entry(cxn, 'z_server', ['', 'Config', 'Positioning']))


def get_registry_entry(cxn, key, directory):
    """
    Return the value for the specified key. The directory is specified from
    the top of the registry. Directory as a list
    """

    p = cxn.registry.packet()
    p.cd('', *directory)
    p.get(key)
    return p.send()['get']


def get_registry_entry_no_cxn(key, directory):
    """
    Same as above
    """
    with labrad.connect() as cxn:
        p = cxn.registry.packet()
        p.cd('', *directory)
        p.get(key)
        return p.send()['get']


# %% Open utils


def ask_open_file(file_path):
    """
    Open a file by selecting it through a file window. File window usually
    opens behind Spyder, may need to minimize Spyder to see file number

    file_path: input the file path to the folder of the data, starting after
    the Kolkowitz Lab Group folder

    Returns:
        string: file name of the file to use in program
    """
    # Prompt the user to select a file
    print('Select file \n...')

    root = Tk()
    root.withdraw()
    root.focus_force()
    directory = str("E:/Shared drives/Kolkowitz Lab Group/" + file_path)
    file_name = filedialog.askopenfilename(initialdir = directory,
                                          title = 'choose file to replot', filetypes = (("svg files","*.svg"),("all files","*.*")) )
    return file_name

def get_file_list(path_from_nvdata, file_ends_with,
                 data_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'):
    '''
    Creates a list of all the files in the folder for one experiment, based on
    the ending file name
    '''

    data_dir = Path(data_dir)
    file_path = data_dir / path_from_nvdata

    file_list = []

    for file in os.listdir(file_path):
        if file.endswith(file_ends_with):
            file_list.append(file)

    return file_list


# def get_raw_data(source_name, file_name, sub_folder_name=None,
#                  data_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'):
#     """Returns a dictionary containing the json object from the specified
#     raw data file.
#     """

#     # Parse the source_name if __file__ was passed
#     source_name = os.path.splitext(os.path.basename(source_name))[0]

#     data_dir = Path(data_dir)
#     file_name_ext = '{}.txt'.format(file_name)
#     if sub_folder_name is None:
#         file_path = data_dir / source_name / file_name_ext
#     else:
#         file_path = data_dir / source_name / sub_folder_name / file_name_ext

#     with open(file_path) as file:
#         return json.load(file)


def get_raw_data(path_from_nvdata, file_name,
                 nvdata_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'):
    """Returns a dictionary containing the json object from the specified
    raw data file.
    """

    data_dir = PurePath(nvdata_dir, path_from_nvdata)
    file_name_ext = '{}.txt'.format(file_name)
    file_path = data_dir / file_name_ext

    with open(file_path) as file:
        return json.load(file)


# %%  Save utils


def get_branch_name():
    """Return the name of the active branch of kolkowitz-nv-experiment-v1.0"""
    home_to_repo = Path('Documents/GitHub/kolkowitz-nv-experiment-v1.0')
    repo_path = Path.home() / home_to_repo
    repo = Repo(repo_path)
    return repo.active_branch.name


def get_time_stamp():
    """
    Get a formatted timestamp for file names and metadata.

    Returns:
        string: <year>-<month>-<day>_<hour>-<minute>-<second>
    """

    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.split('.')[0]  # Keep up to seconds
    timestamp = timestamp.replace(':', '_')  # Replace colon with dash
    timestamp = timestamp.replace('-', '_')  # Replace dash with underscore
    timestamp = timestamp.replace(' ', '-')  # Replace space with dash
    return timestamp


def get_folder_dir(source_name, subfolder):

    source_name = os.path.basename(source_name)
    source_name = os.path.splitext(source_name)[0]

    branch_name = get_branch_name()
    pc_name = socket.gethostname()

    # # Check where we should save to
    # if branch_name == 'master':
    #     # master should save without a branch sub-folder
    #     joined_path = os.path.join('E:/Shared drives/Kolkowitz Lab Group/nvdata',
    #                                source_name)
    # else:
    #     # Otherwise we want a branch sub-folder so that we know this data was
    #     # produced by code that's under development
    #     joined_path = os.path.join('E:/Shared drives/Kolkowitz Lab Group/nvdata',
    #                                source_name,
    #                                'branch_{}'.format(branch_name))

    joined_path = os.path.join('E:/Shared drives/Kolkowitz Lab Group/nvdata',
                               'pc_{}'.format(pc_name),
                               'branch_{}'.format(branch_name),
                               source_name)

    if subfolder is not None:
        joined_path = os.path.join(joined_path, subfolder)

    folderDir = os.path.abspath(joined_path)

    # Make the required directory if it doesn't exist already
    if not os.path.isdir(folderDir):
        os.makedirs(folderDir)

    return folderDir

def get_data_path():
    return Path('E:/Shared drives/Kolkowitz Lab Group/nvdata')

def get_data_path():
    return Path('E:/Shared drives/Kolkowitz Lab Group/nvdata')


def get_file_path(source_name, time_stamp='', name='', subfolder=None):
    """
    Get the file path to save to. This will be in a subdirectory of nvdata.

    Params:
        source_name: string
            Source file name - alternatively, __file__ of the caller which will
            be parsed to get the name of the subdirectory we will write to
        time_stamp: string
            Formatted timestamp to include in the file name
        name: string
            The file names consist of <timestamp>_<name>.<ext>
            Ext is supplied by the save functions
        subfolder: string
            Subfolder to save to under file name
    """

    date_folder_name = None  # Init to None
    # Set up the file name
    if (time_stamp != '') and (name != ''):
        fileName = '{}-{}'.format(time_stamp, name)
        #locate the subfolder that matches the month and year when the data is taken
        date_folder_name = '_'.join(time_stamp.split('_')[0:2])
    elif (time_stamp == '') and (name != ''):
        fileName = name
    elif (time_stamp != '') and (name == ''):
        fileName = '{}-{}'.format(time_stamp, 'untitled')
        date_folder_name = '_'.join(time_stamp.split('_')[0:2])
    else:
        fileName = '{}-{}'.format(get_time_stamp(), 'untitled')

    # Create the subfolder combined name, if needed
    subfolder_name = None
    if (subfolder != None) and (date_folder_name != None):
        subfolder_name = str(date_folder_name + '/' + subfolder)
    elif (subfolder == None) and (date_folder_name != None):
        subfolder_name = date_folder_name

    folderDir = get_folder_dir(source_name, subfolder_name)
    fileDir = os.path.abspath(os.path.join(folderDir, fileName))

    return fileDir

#def get_file_path(source_name, time_stamp='', name='', subfolder=None):
#    """
#    Get the file path to save to. This will be in a subdirectory of nvdata.
#
#    Params:
#        source_name: string
#            Source file name - alternatively, __file__ of the caller which will
#            be parsed to get the name of the subdirectory we will write to
#        time_stamp: string
#            Formatted timestamp to include in the file name
#        name: string
#            The file names consist of <timestamp>_<name>.<ext>
#            Ext is supplied by the save functions
#        subfolder: string
#            Subfolder to save to under file name
#    """
#
#    # Set up the file name
#    if (time_stamp != '') and (name != ''):
#        fileName = '{}-{}'.format(time_stamp, name)
#    elif (time_stamp == '') and (name != ''):
#        fileName = name
#    elif (time_stamp != '') and (name == ''):
#        fileName = '{}-{}'.format(time_stamp, 'untitled')
#    else:
#        fileName = '{}-{}'.format(get_time_stamp(), 'untitled')
#
#    folderDir = get_folder_dir(source_name, subfolder)
#
#    fileDir = os.path.abspath(os.path.join(folderDir, fileName))
#
#    return fileDir


def save_figure(fig, file_path):
    """
    Save a matplotlib figure as a png.

    Params:
        fig: matplotlib.figure.Figure
            The figure to save
        file_path: string
            The file path to save to including the file name, excluding the
            extension
    """

    file_path = str(file_path)
    fig.savefig(file_path + '.svg', dpi = 300)


def save_raw_data(rawData, filePath):
    """
    Save raw data in the form of a dictionary to a text file. New lines
    will be printed between entries in the dictionary.

    Params:
        rawData: dict
            The raw data as a dictionary - will be saved via JSON
        filePath: string
            The file path to save to including the file name, excluding the
            extension
    """

    # Add in a few things that should always be saved here. In particular,
    # sharedparameters so we have as snapshot of the configuration and
    # nv_sig_units. If these have already been defined in the routine,
    # then they'll just be overwritten.
    try:
        rawData['nv_sig_units'] = get_nv_sig_units()
        rawData['config'] = get_config_dict()  # Include a snapshot of the config
    except Exception as e:
        print(e)

    with open(filePath + '.txt', 'w') as file:
        json.dump(rawData, file, indent=2)


def get_nv_sig_units():
    return {'coords': 'V', 'expected_count_rate': 'kcps',
        'pulsed_readout_dur': 'ns',
        'pulsed_SCC_readout_dur': 'ns', 'am_589_power': '0-1 V',
        'pulsed_shelf_dur': 'ns', 'am_589_shelf_power': '0-1 V',
        'pulsed_ionization_dur': 'ns', 'cobalt_638_power': 'mW',
        'pulsed_reionization_dur': 'ns', 'cobalt_532_power': 'mW',
        'magnet_angle': 'deg', 'resonance': 'GHz',
        'rabi': 'ns', 'uwave_power': 'dBm'}

# Error messages

def color_ind_err(color_ind):
    if color_ind != 532 or color_ind != 589:
        raise RuntimeError('Value of color_ind must be 532 or 589.'+
                           '\nYou entered {}'.format(color_ind))

def check_laser_power(laser_name, laser_power):
    pass

def aom_ao_589_pwr_err(aom_ao_589_pwr):
    if aom_ao_589_pwr < 0 or aom_ao_589_pwr > 1.0:
        raise RuntimeError('Value for 589 aom must be within 0 to +1 V.'+
                           '\nYou entered {} V'.format(aom_ao_589_pwr))

def ao_638_pwr_err(ao_638_pwr):
    if ao_638_pwr < 0 or ao_638_pwr > 0.9:
        raise RuntimeError('Value for 638 ao must be within 0 to 0.9 V.'+
                           '\nYou entered {} V'.format(ao_638_pwr))

def x_y_image_grid(x_center, y_center, x_range, y_range, num_steps):

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
        x_inter = numpy.concatenate((x_voltages_1d,
                                     numpy.flipud(x_voltages_1d)))
        # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
        if y_num_steps % 2 == 0:  # Even x size
            x_voltages = numpy.tile(x_inter, int(y_num_steps/2))
        else:  # Odd x size
            x_voltages = numpy.tile(x_inter, int(numpy.floor(y_num_steps/2)))
            x_voltages = numpy.concatenate((x_voltages, x_voltages_1d))

        # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
        y_voltages = numpy.repeat(y_voltages_1d, x_num_steps)

        voltages = numpy.vstack((x_voltages, y_voltages))

        return x_voltages, y_voltages
# %% Misc

def opt_power_via_photodiode(color_ind, AO_power_settings = None, nd_filter = None):
    cxn = labrad.connect()
    optical_power_list = []
    if color_ind==532:
        cxn.pulse_streamer.constant([3],0.0, 0.0) # Turn on the green laser
        time.sleep(0.3)
        for i in range(10):
            optical_power_list.append(cxn.photodiode.read_optical_power())
            time.sleep(0.01)
    elif color_ind==589:
        cxn.filter_slider_ell9k.set_filter(nd_filter) # Change the nd filter for the yellow laser
        cxn.pulse_streamer.constant([],0.0, AO_power_settings) # Turn on the yellow laser
        time.sleep(0.3)
        for i in range(10):
            optical_power_list.append(cxn.photodiode.read_optical_power())
            time.sleep(0.01)
    elif color_ind==638:
        cxn.pulse_streamer.constant([7], 0.0, 0.0) # Turn on the red laser
        time.sleep(0.3)
        for i in range(10):
            optical_power_list.append(cxn.photodiode.read_optical_power())
            time.sleep(0.01)

    optical_power = numpy.average(optical_power_list)
    time.sleep(0.1)
    cxn.pulse_streamer.constant([], 0.0, 0.0)
    return optical_power

def calc_optical_power_mW(color_ind, optical_power_V):
    # Values found from experiments. See Notebook entry 3/19/2020 and 3/20/2020
    if color_ind == 532:
        return 11.84* optical_power_V + 0.0493
    elif color_ind == 589:
        return 13.41* optical_power_V + 0.06
    if color_ind == 638:
        return 4.14* optical_power_V + 0.0492

def measure_g_r_y_power(aom_ao_589_pwr, nd_filter):
    green_optical_power_pd = opt_power_via_photodiode(532)

    red_optical_power_pd = opt_power_via_photodiode(638)

    yellow_optical_power_pd = opt_power_via_photodiode(589,
           AO_power_settings = aom_ao_589_pwr, nd_filter = nd_filter)

    # Convert V to mW optical power
    green_optical_power_mW = \
            calc_optical_power_mW(532, green_optical_power_pd)

    red_optical_power_mW = \
            calc_optical_power_mW(638, red_optical_power_pd)

    yellow_optical_power_mW = \
            calc_optical_power_mW(589, yellow_optical_power_pd)

    return green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW

# %% Safe stop (TM mccambria)


"""
Safe stop allows you to listen for a stop command while other things are
happening. This allows you to, say, stop a loop-based routine halfway
through. To use safe stop, call init_safe_stop() and then poll for the
stop command with safe_stop(). It's up to you to actually stop the
routine once you get the signal. Note that there's no way to programmatically
halt safe stop once it's running; the user must press enter.

Safe stop works by setting up a second thread alongside the main
thread. This thread listens for input, and sets a threading event after
the input. A threading event is just a flag used for communication between
threads. safe_stop() simply returns whether the flag is set.
"""


def safe_stop_input():
    """
    This is what the safe stop thread does.
    """

    global SAFESTOPEVENT
    input('Press enter to stop...')
    SAFESTOPEVENT.set()


def check_safe_stop_alive():
    """
    Checks if the safe stop thread is alive.
    """

    global SAFESTOPTHREAD
    try:
        SAFESTOPTHREAD
        return SAFESTOPTHREAD.isAlive()
    except NameError:
        return False


def init_safe_stop():
    """
    Initialize safe stop. Recycles the current instance of safe stop if
    there's one already running.
    """

    global SAFESTOPEVENT
    global SAFESTOPTHREAD
    needNewSafeStop = False

    # Determine if we need a new instance of safe stop or if there's
    # already one running
    try:
        SAFESTOPEVENT
        SAFESTOPTHREAD
        if not SAFESTOPTHREAD.isAlive():
            # Safe stop has already run to completion so start it back up
            needNewSafeStop = True
    except NameError:
        # Safe stop was never initialized so just get a new instance
        needNewSafeStop = True

    if needNewSafeStop:
        SAFESTOPEVENT = threading.Event()
        SAFESTOPTHREAD = threading.Thread(target=safe_stop_input)
        SAFESTOPTHREAD.start()


def safe_stop():
    """
    Check if the user has told us to stop. Call this whenever there's a safe
    break point after initializing safe stop.
    """

    global SAFESTOPEVENT

    try:
        return SAFESTOPEVENT.is_set()
    except Exception:
        print('Stopping. You have to intialize safe stop before checking it.')
        return True


def poll_safe_stop():
    """
    Polls safe stop continuously until the user says stop. Effectively a
    regular blocking input. The problem with just sticking input() in the main
    thread is that you can't have multiple threads looking for input.
    """

    init_safe_stop()
    while True:
        time.sleep(0.1)
        if safe_stop():
            break


# %% State/globals


# Our client is and should be mostly stateless.
# But in some cases it's just easier to share some state across the life of an
# experiment/across experiments. To do this safely and easily we store global
# variables on our LabRAD registry. The globals should only be accessed with
# the getters and setters here so that we can be sure they're implemented
# properly.


def get_drift():
    with labrad.connect() as cxn:
        cxn.registry.cd(['', 'State'])
        drift = cxn.registry.get('DRIFT')
        # MCC where should this stuff live?
        cxn.registry.cd(['', 'Config', 'Positioning'])
        xy_dtype = eval(cxn.registry.get('xy_dtype'))
        z_dtype = eval(cxn.registry.get('z_dtype'))
    len_drift = len(drift)
    if len_drift != 3:
        print('Got drift of length {}.'.format(len_drift))
        print('Setting to length 3.')
        if len_drift < 3:
            for ind in range(3 - len_drift):
                drift.append(0.0)
        elif len_drift > 3:
            drift = drift[0:3]
    # Cast to appropriate type
    # MCC round instead of int?
    drift_to_return = [xy_dtype(drift[0]),
                       xy_dtype(drift[1]),
                       z_dtype(drift[2])]
    return drift_to_return


def set_drift(drift):
    len_drift = len(drift)
    if len_drift != 3:
        print('Attempted to set drift of length {}.'.format(len_drift))
        print('Set drift unsuccessful.')
    # Cast to the proper types

    xy_dtype = eval(get_registry_entry_no_cxn('xy_dtype',
                                              ['', 'SharedParameters']))
    z_dtype = eval(get_registry_entry_no_cxn('z_dtype',
                                             ['', 'SharedParameters']))
    drift = [xy_dtype(drift[0]), xy_dtype(drift[1]), z_dtype(drift[2])]
    with labrad.connect() as cxn:
        cxn.registry.cd(['', 'State'])
        return cxn.registry.set('DRIFT', drift)


def reset_drift():
    set_drift([0.0, 0.0, 0.0])


# %% Reset hardware


def reset_cfm(cxn=None):
    """Reset our cfm so that it's ready to go for a new experiment. Avoids
    unnecessarily resetting components that may suffer hysteresis (ie the
    components that control xyz since these need to be reset in any
    routine where they matter anyway).
    """

    if cxn == None:
        with labrad.connect() as cxn:
            reset_cfm_with_cxn(cxn)
    else:
        reset_cfm_with_cxn(cxn)


def reset_cfm_with_cxn(cxn):
    xyz_servers = [get_xy_server(cxn), get_z_server(cxn)]
    for server in cxn.servers:
        if server in xyz_servers:
            continue
        if hasattr(server, 'reset'):
            server.reset()

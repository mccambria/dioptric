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


# User modules
import PulseStreamer.pulse_streamer_jrpc
from PulseStreamer.pulse_streamer_jrpc import PulseStreamer
from PulseStreamer.pulse_streamer_jrpc import Start
from PulseStreamer.pulse_streamer_jrpc import Mode
from PulseStreamer.Sequence import Sequence

# Library modules
import nidaqmx
import matplotlib.pyplot as plt
import visa  # Docs here: https://pyvisa.readthedocs.io/en/master/
import threading
import os
import datetime
import numpy
import json
import time
import labrad


# %% Singletons


def get_cxn():
    global CXN
    try:
        return CXN
    except NameError:
        CXN = labrad.connect()
        return CXN


def get_pulser(pulserIP):
    """
    Gets the PulseStreamer at the given IP address. Returns None if it's not
    available. Maintains a single instance of the pulser on the kernel.

    Params:
        pulserIP: string
            The IP address of the PulserStreamer

    Returns:
        PulseStreamer: The PulseStreamer, or None if it isn't available
    """

    global PULSER
    try:
        return PULSER
    except Exception:
        try:
            PULSER = PulseStreamer(pulserIP)
            PULSER.isStreaming()
            return PULSER
        except Exception as e:
            del PULSER
            print(e)
            print('Could not get PulseStreamer at ' + pulserIP)
            return None


def get_daq(daqName):
    """
    Gets the DAQ for the given name. Returns None on failure.
    Maintains a single instance of the daq on the kernel.

    Params:
        daqName: string
            The name of the daq

    Returns:
        nidaqmx.system.device.Device: The DAQ we're running
    """
    global DAQ
    try:
        return DAQ
    except Exception:
        try:
            DAQ = nidaqmx.system.device.Device(daqName)
            return DAQ
        except Exception as e:
            del DAQ
            print(e)
            print('Could not get DAQ name ' + daqName)
            return None


def get_task_list():
    """
    Gets the task list. Returns None on failure.
    Maintains a single instance of the daq on the kernel.

    Returns:
        list(Task): The global list of nidaqmx tasks
    """
    global TASKLIST
    try:
        return TASKLIST
    except Exception:
        TASKLIST = []
        return TASKLIST


# %% Range parsing


def parse_range(dimExtent, dimResolution):
    """
    Calculate the number of steps in each direction and the real low/high
    voltages for a dimension based on the dimension's extent and resolution.

    Params:
        dimExtent: tuple(float)
            Low and high voltage limit of the dimension's extent
        dimResolution: float
            Volts per step between samples along the dimension

    Returns:
        float: Actual low voltage limit of the dimension's extent
        float: Actual high voltage limit of the dimension's extent
        int: Number of steps along the dimension
    """

    # dimLow is the actual lowest voltage in the range.
    # dimHighLim is the value which the highest voltage in the range will
    # not exceed.
    dimLow = dimExtent[0]
    dimHighRange = dimExtent[1]

    # Calculate the number of steps we'll make in each direction.
    # The + 1 is to include the starting point in the count.
    dimRangeDiff = dimHighRange - dimLow
    dimNumSteps = int(numpy.floor(dimRangeDiff / dimResolution) + 1)

    # Calculate the actual highest x/y voltages in the sweep.
    # The - 1 accounts for including the starting point in the numSteps.
    dimHigh = dimLow + ((dimNumSteps - 1) * dimResolution)

    return dimLow, dimHigh, dimNumSteps


def parse_center(dimCenter, dimHalfRange, dimResolution):
    """
    Calculate the number of steps in each direction and the real low/high
    x/y values based on the x/y centers.

    Params:
        dimCenter: float
            The center voltage of the dimension
        dimHalfRange: float
            Half the range along the dimension that we'll scan out to
        dimResolution: float
            Volts per step between samples

    Returns:
        float: Actual low voltage limit of the dimension's extent
        float: Actual high voltage limit of the dimension's extent
        int: Number of steps along the dimension
    """

    # Calculate the extent from the given parameters
    dimExtent = (dimCenter - dimHalfRange, dimCenter + dimHalfRange)

    return parse_range(dimExtent, dimResolution)


# %% Matplotlib plotting utils


def create_image_figure(imgArray, imgExtent, clickHandler=None):
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

    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots()

    # Tell the axes to show a grayscale image
    img = ax.imshow(imgArray, cmap='inferno',
                    extent=tuple(imgExtent))

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

    # Add a colorbar
    plt.colorbar(img)

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


def gaussian(x, *params):
    """
    Calculates the value of a gaussian for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the Gaussian
            1: coefficient that defines the peak height
            2: mean, defines the center of the Gaussian
            3: standard deviation, defines the width of the Gaussian
            4: constant y value to account for background
    """

    coeff, mean, stdev, offset = params
    var = stdev**2  # variance
    centDist = x-mean  # distance from the center
    return offset + coeff**2*numpy.exp(-(centDist**2)/(2*var))


# %%  Save utils


def get_time_stamp():
    """
    Get a formatted timestamp for file names and metadata.

    Returns:
        string: <year>-<month>-<day>_<hour>-<minute>-<second>
    """

    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.split('.')[0]  # Ditch microseconds
    timestamp = timestamp.replace(':', '-')  # Replace colon with dash
    timestamp = timestamp.replace(' ', '_')  # Replace space with underscore
    return timestamp


def get_file_path(subDirName, timeStamp, name):
    """
    Get the file path to save to. This will be in a subdirectory of Data.

    Params:
        subDirName: string
            The sub directory of Data to save to
        timeStamp: string
            Formatted timestamp to include in the file name
        name: string
            The file names consist of <timestamp>_<name>.<ext>
            Ext is supplied by the save functions
    """

    # Set up a timestamp
    fileName = timeStamp + '_' + name

    # Find the data directory relative to tool_belt's directory
    currentDir = os.path.dirname(__file__)
    folderDir = os.path.abspath(os.path.join(currentDir, '..',
                                             'Data', subDirName))

    # Make the required directory if it doesn't exist already
    if not os.path.isdir(folderDir):
        os.makedirs(folderDir)

    fileDir = os.path.abspath(os.path.join(folderDir, fileName))

    return fileDir


def save_figure(fig, filePath):
    """
    Save a matplotlib figure as a png.

    Params:
        fig: matplotlib.figure.Figure
            The figure to save
        filePath: string
            The file path to save to including the file name, excluding the
            extension
    """

    fig.savefig(filePath + '.svg')


def save_raw_data(rawData, filePath):
    """
    Save raw data in the form of a dictionary to a text file.

    Params:
        rawData: dict
            The raw data as a dictionary - will be saved via JSON
        filePath: string
            The file path to save to including the file name, excluding the
            extension
    """

    with open(filePath + '.txt', 'w') as file:
        json.dump(rawData, file)


# %%  Pulser Streams


def pulser_all_zero(pulserIP):
    """
    Set Pulsestreamer constant (LOW)

    Params:
        pulserIP: string
            The IP of the PulseStreamer that we'll be using
    """

    # Get pulser
    pulser = get_pulser(pulserIP)
    if pulser is None:
        return

    allZero = (0, [], 0, 0)
    pulser.constant(allZero)


def pulser_all_high(pulserIP):
    """
    Set Pulsestreamer constant (HIGH)

    Params:
        pulserIP: string
            The IP of the PulseStreamer that we'll be using
    """

    # Get pulser
    pulser = get_pulser(pulserIP)
    if pulser is None:
        return

    allHigh = (0, [0, 1, 2, 3, 4, 5, 6, 7], 1.0, 1.0)
    pulser.constant(allHigh)


def pulser_high(pulserIP, chanList):
    """
    Set specified channels high

    Params:
        pulserIP: string
            The IP of the PulseStreamer that we'll be using
        chanList: list(int)
            List of channels to write high
    """

    # Get pulser
    pulser = get_pulser(pulserIP)
    if pulser is None:
        return

    highStream = (0, chanList, 0, 0)
    pulser.constant(highStream)


def pulser_square_wave(pulserIP, period, chanList, count):
    """
    Streams a square wave to the specified digital outputs

    Params:
        pulserIP: string
            The IP of the PulseStreamer that we'll be using
        period: int
            The period of the wave in ns
        chanList: list(int)
            A list of the channels to stream to
        count: int
            The number of times to run the wave, ie the number of samples,
            'infinite' for continuous
    """

    # Get pulser
    pulser = get_pulser(pulserIP)
    if pulser is None:
        return

    # Make sure we're starting from zero
    pulser_all_zero(pulserIP)

    # Set the mode
    pulser.setTrigger(start=Start.IMMEDIATE, mode=Mode.SINGLE)

    # Set the final value
    final = (0, [], 0, 0)  # All zeros

    # Pulse-sequence on all channels
    seq = [(period // 2, chanList, 0, 0),
           (period // 2, [], 0, 0)]

    pulser.stream(seq, count, final)


def get_readout_cont_illum_seq(pulserDODaqClock, pulserDODaqGate,
                               pulserDOAom, period, readout):

    low = 0
    high = 1

    seq = Sequence()

    train = [(100, high), (period - 100, low)]
    seq.setDigitalChannel(pulserDODaqClock, train)

    train = [(period - readout, low), (readout, high)]
    seq.setDigitalChannel(pulserDODaqGate, train)

    train = [(period, high)]
    seq.setDigitalChannel(pulserDOAom, train)
#    seq.setDigitalChannel(4, train)  # Temporary to keep RF ungated

    return seq.getSequence()


def pulser_readout_cont_illum(pulserIP, pulserDODaqClock,
                              pulserDODaqGate, pulserDOAom,
                              period, readout, totalSamples):
    """
    Streams a clock signal (100 ns high, low for the rest of the period)
    and a gate signal (high for the last readout ns of the period). Sets
    the AOM on. Streams totalSamples + 1 periods to bookend samples.

    Params:
        pulserIP: string
            The IP of the PulseStreamer that we'll be using
        pulserDODaqClock: int
            pulser DOs carrying DAQ clock signal
        pulserDODaqGate: int
            pulser DOs carrying DAQ gate signal
        pulserDOAom: int
            pulser DOs carrying AOM gate signal
        period: numpy.int64
            Total period of a sample in ns
        readout: numpy.int64
            Readout time of a sample (time APD channel is ungated) in ns
            If None, does not run the gate channel
        totalSamples: int
            Total number of samples we'll be collecting
    """

    pulser = get_pulser(pulserIP)

    if pulser is None:
        return

    # Make sure we're starting from zero
    pulser_all_zero(pulserIP)

    # Set the PulseStreamer to start as soon as it receives the stream from
    # from the computer
    pulser.setTrigger(start=Start.IMMEDIATE, mode=Mode.SINGLE)

    # Bookend samples
    totalRisingClockEdges = totalSamples + 1

    seq = get_readout_cont_illum_seq(pulserDODaqClock, pulserDODaqGate,
                                     pulserDOAom, period, readout)

    # Run the stream
    pulser.stream(seq, totalRisingClockEdges)


# %% Simple VISA functions


def get_VISA_instr(address):
    """
    Gets the VISA instrument at the specified address. Then use the PyVISA
    library to communicate with the instrument via SCPI commands.

    Params:
        address: string
            Address of the instrument to get. For example, 'ASRL1::INSTR'
            is the address of the first instrument connected via serial.
    """

    try:
        resourceManager = visa.ResourceManager()
        return resourceManager.open_resource(address)
    except Exception:
        print('Could not get VISA instrument at: ' + address)
        return None


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


# %% Resets and clean up


def task_list_close_all():
    """
    Closes and removes all tasks in the task list.
    """

    taskList = get_task_list()
    for task in taskList:
        task.stop()
        task.close()
    taskList.clear()


def clean_up(pulserIP, daqName):
    """
    Do the necessary clean up after running a routine.

    Params:
        pulserIP: string
            The IP of the PulseStreamer that we'll be using
        daqName: string
            The name of the DAQ that we'll be using
    """

    # PulseStreamer clean up
    pulser = get_pulser(pulserIP)
    if pulser is not None:
        pulser.reset()

    # DAQ clean up
    task_list_close_all()

    if check_safe_stop_alive():
        print('\n\nRoutine complete. Press enter to exit.')
        poll_safe_stop()

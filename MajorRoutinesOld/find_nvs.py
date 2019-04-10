# -*- coding: utf-8 -*-
"""
Find the NVs in the input rectangular area. Sweeps the galvo in a winding
pattern over the rectangle and records the fluorescence at each point.

Created on Fri Dec 2 12:24:54 2018

@author: mccambria
"""


# %% Imports


# User modules
import Utils.tool_belt as tool_belt
import Utils.sweep_utils as sweep_utils
import Outputs.galvo as galvo
import Outputs.objective_piezo as objective_piezo
import Outputs.xyz as xyz
import Inputs.apd as apd

# Library modules
import numpy


# %% Functions


def update_image(newSamples, numReadSoFar, *args):
    """
    Update the image figure. Called once per loop through the APD stream read
    function.

    Params:
        newSamples: numpy.ndarray
            Array of new samples to populate the image with
        numReadSoFar: int
            Total number of samples read so far
        args: tuple
            fig: matplotlib.figure.Figure
                The matplotlib figure to update
                Should be from tool_belt.create_image_figure
            imgArray: numpy.ndarray
                Array of NaNs to write the edge counts to
            imgWritePos: list
                Write position in the image, [x,y]
            galvoVoltages: numpy.ndarray(float)
                All voltages to write to the galvo
            bufferPos: list(int)
                Contains the index in galvoVoltages that we've written up to
            streamWriter: nidaqmx.stream_writer
                The galvo stream to write to
    """

    fig, imgArray, imgWritePos, streamWriter, galvoVoltages, bufferPos = args

    # Refill the buffer once we've written bufferPosVal - 1000 samples
    bufferPosVal = bufferPos[0]
    if bufferPosVal is not None:
        if numReadSoFar > bufferPosVal - 1000:
            # Check if there are more than 3000 samples left to write
            if galvoVoltages.shape[1] - bufferPosVal > 3000:
                nextBufferPosVal = bufferPosVal + 3000
                bufferVoltages = galvoVoltages[:, bufferPosVal:nextBufferPosVal]
                bufferPos[0] = nextBufferPosVal
            else:
                bufferVoltages = galvoVoltages[:, bufferPosVal:]
                bufferPos[0] = None
            contBufferVoltages = numpy.ascontiguousarray(bufferVoltages)
            streamWriter.write_many_sample(contBufferVoltages)

    # Write to the image array
    sweep_utils.populate_img_array(newSamples, imgArray, imgWritePos)

    # Update the figure with the updated image array.
    tool_belt.update_image_figure(fig, imgArray)


def on_click_image(event):
    """
    Click handler for images. Prints the click coordinates to the console.
    
    Params:
        event: dictionary
            Dictionary containing event details
    """

    try:
        print("\n    xCenter = %.3f\n    yCenter = %.3f" % 
              (event.xdata, event.ydata))
    except TypeError:
        # Ignore TypeError if you click in the figure but out of the image
        pass


# %% Main


def main(pulserIP, daqName,
         daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd,
         daqDIPulserClock, daqDIPulserGate,
         pulserDODaqClock, pulserDODaqGate, pulserDOAom,
         name, xCenter, yCenter, zCenter, xScanRange, yScanRange,
         scanStepSize, readout, continuous):
    """
    Entry point for the routine

    Params:

        pulserIP: string
            The IP of the PulseStreamer that we'll be using
        daqName: string
            The name of the DAQ that we'll be using

        daqAOGalvoX: int
            DAQ AO carrying galvo X signal
        daqAOGalvoY: int
            DAQ AO carrying galvo Y signal
        piezoSerial: string
            Objective piezo serial number

        daqCIApd: int
            DAQ CI for APD

        daqDIPulserClock: int
            DAQ DI for clock signal from pulser
        daqDIPulserGate: int
            DAQ DI for gate signal from pulser

        pulserDODaqClock: int
            pulser DO carrying DAQ clock signal
        pulserDODaqGate: int
            pulser DO carrying DAQ gate signal
        pulserDOAom: int
            pulser DO carrying AOM gate signal

        name: string
            The file names consist of <date>_<time>_<name>.<ext>
        xCenter: float
            The center x position the galvo will scan around
        yCenter: float
            The center y position the galvo will scan around
        zCenter: float
            Fixed voltage for the piezo
        scanRange: float
            The range, along one direction in both x and y from the center
        scanStepSize: float
            Volts per step between samples
        readout: numpy.int64
            Readout time of a sample (time APD channel is ungated) in ns
        continuous: boolean
            If True, takes images continously and updates the figure live for
            feedback. If False, takes and saves a single image.
    """

    # %% Some initial calculations

    # The galvo's small angle step response is 400 us
    # Let's give ourselves a buffer of 500 us (500000 ns)
    period = readout + numpy.int64(500000)

    xScanCenterPlusMinus = xScanRange / 2
    
    yScanCenterPlusMinus = yScanRange / 2

    # Parse the galvo inputs
    xLow, xHigh, xNumSteps = tool_belt.parse_center(xCenter,
                                                    xScanCenterPlusMinus,
                                                    scanStepSize)
    yLow, yHigh, yNumSteps = tool_belt.parse_center(yCenter,
                                                    yScanCenterPlusMinus,
                                                    scanStepSize)

    # Total number of samples we'll read
    totalNumSamples = xNumSteps * yNumSteps

    # %% Set up the galvo

    galvoVoltages = sweep_utils.calc_voltages(scanStepSize, xLow, yLow,
                                              xNumSteps, yNumSteps)

    # We'll write incrementally if there are more than 4000 samples
    # per channel since the DAQ buffer supports 8191 samples max
    if galvoVoltages.shape[1] > 4000:
        bufferVoltages = galvoVoltages[:, 0:4000]
        bufferPos = [4000]
    else:
        bufferVoltages = galvoVoltages
        bufferPos = [None]

    streamWriter, galvoTask = galvo.stream_write_daq(daqName,
                                                     daqAOGalvoX, daqAOGalvoY,
                                                     daqDIPulserClock,
                                                     bufferVoltages[0],
                                                     bufferVoltages[1],
                                                     period)

    # %% Set the piezo

    objective_piezo.write_single_open_loop(piezoSerial, zCenter)

    # %% Set up the APD

    streamReader, apdTask = apd.stream_read_load_daq(daqName, daqCIApd,
                                                     daqDIPulserClock,
                                                     daqDIPulserGate,
                                                     period)

    # %% Set up the image display

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    imgArray = numpy.empty((yNumSteps, xNumSteps))
    imgArray[:] = numpy.nan
    imgWritePos = []

    # For the image extent, we need to bump out the min/max x/y by half the
    # resolution in each direction so that the center of each pixel properly
    # lies at its x/y voltages.
    halfStep = scanStepSize / 2
    imageExtent = [xHigh + halfStep, xLow - halfStep,
                   yLow - halfStep, yHigh + halfStep]

    fig = tool_belt.create_image_figure(imgArray, imageExtent,
                                        clickHandler=on_click_image)

    # %% Run the PulseStreamer

    tool_belt.pulser_readout_cont_illum(pulserIP, pulserDODaqClock,
                                        pulserDODaqGate, pulserDOAom,
                                        period, readout, totalNumSamples)

    # %% Collect the data

    timeout = ((period*(10**-9)) * totalNumSamples) + 10

    apd.stream_read_daq(streamReader, totalNumSamples, timeout,
                        update_image, fig, imgArray, imgWritePos,
                        streamWriter, galvoVoltages, bufferPos)

    # %% Clean up

    # Close tasks
    taskList = tool_belt.get_task_list()
    for task in taskList:
        if task.name == galvoTask.name:
            taskList.remove(task)
    galvoTask.close()
    for task in taskList:
        if task.name == apdTask.name:
            taskList.remove(task)
    apdTask.close()

    xyz.write_daq(daqName, daqAOGalvoX, daqAOGalvoY, piezoSerial,
                  xCenter, yCenter, zCenter)

    # %% Save the data 

    timeStamp = tool_belt.get_time_stamp()

    rawData = {"timeStamp": timeStamp,
               "name": name,
               "xyzCenters": [xCenter, yCenter, zCenter],
               "xScanRange": xScanRange,
               "yScanRange": yScanRange,
               "scanStepSize": scanStepSize,
               "readout": int(readout),
               "imgResolution": [xNumSteps, yNumSteps],
               "imgArray": imgArray.astype(int).tolist()}

    filePath = tool_belt.get_file_path("find_nvs", timeStamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)

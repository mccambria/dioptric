# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:53:42 2019

@author: mccambria
"""


# %% Imports


# User modules
import Utils.tool_belt as tool_belt
import Outputs.galvo as galvo
import Outputs.objective_piezo as objective_piezo
import Inputs.apd as apd
from PulseStreamer.pulse_streamer_jrpc import Start

# Library modules
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time


# %% Main


def main(pulserIP, daqName,
         daqAOGalvoX, daqAOGalvoY, piezoSerial,
         daqCIApd, daqDIPulserClock, daqDIPulserGate,
         pulserDODaqClock, pulserDODaqGate, pulserDOAom,
         name, xCenter, yCenter, zCenter, scanRange, scanStepSize,
         depthRange, depthStepSize, readout, doPlot=True):
    """
    Entry point for the routine

    Params:

        pulserIP: string
            The IP of the PulseStreamer that we'll be using
        daqName: string
            The name of the DAQ that we'll be using

        daqAOGalvoX: int
            DAQ AOs carrying galvo X signal
        daqAOGalvoY: int
            DAQ AOs carrying galvo Y signal
        piezoSerial: string
            Objective piezo serial number

        daqCIApd: int
            DAQ CI for APD

        daqDIPulserClock: int
            DAQ DI for clock signal from pulser
        daqDIPulserGate: int
            DAQ DI for gate signal from pulser

        pulserDODaqClock: int
            pulser DOs carrying DAQ clock signal
        pulserDODaqGate: int
            pulser DOs carrying DAQ gate signal
        pulserDOAom: int
            pulser DOs carrying AOM gate signal

        name: string
            The file names consist of <date>_<time>_<name>.<ext>
        xCenter: float
            The center x position the galvo will scan around
        yCenter: float
            The center y position the galvo will scan around
        zCenter: float
            The center z position the piezo will scan around
        scanRange: float
            The range, along one direction in both x and y from the center
        scanStepSize: float
            Volts per step between x/y samples
        depthRange: float
            The range, along the z axis from the center
        depthStepSize: float
            Volts per step between z samples
        readout: numpy.int64
            Readout time of a sample (time APD channel is ungated) in ns

    Returns:
        float: The optimized x center
        float: The optimized y center
        float: The optimized z center
    """

    # %% Some initial calculations

    # The galvo's small angle step response is 400 us
    # Let's give ourselves a buffer of 500 us (500000 ns)
    period = readout + numpy.int64(500000)

    scanCenterPlusMinus = scanRange / 2
    depthCenterPlusMinus = depthRange / 2

    # Turn the ranges into a set of more useful values
    xLow, xHigh, xNumSteps = tool_belt.parse_center(xCenter,
                                                    scanCenterPlusMinus,
                                                    scanStepSize)
    yLow, yHigh, yNumSteps = tool_belt.parse_center(yCenter,
                                                    scanCenterPlusMinus,
                                                    scanStepSize)
    zLow, zHigh, zNumSteps = tool_belt.parse_center(zCenter,
                                                    depthCenterPlusMinus,
                                                    depthStepSize)

    if (xLow < -10.0) or (xHigh > 10.0):
        raise ValueError("Galvo voltages are out of range.")

    if (yLow < -10.0) or (yHigh > 10.0):
        print(yLow, yHigh)
        raise ValueError("Galvo voltages are out of range.")

    if (zLow < 0.0) or (zHigh > 100.0):
        raise ValueError("Piezo voltages are out of range.")

    # List to store the optimized centers
    optiCenters = [None, None, None]

    # %% Calculate our voltages

    # Set up vectors for the number of samples in each direction
    # [0, 1, 2, ... length - 1]
    xSteps = numpy.arange(xNumSteps)
    ySteps = numpy.arange(yNumSteps)
    zSteps = numpy.arange(zNumSteps)

    # Apply scale and offset to get the voltages we'll apply to the galvo
    # Note that the polar/azimuthal angles, not the actual x/y positions are
    # linear in these voltages. For a small range, however, we don't really
    # care.
    xVoltages = (scanStepSize * xSteps) + xLow
    yVoltages = (scanStepSize * ySteps) + yLow
    zVoltages = (depthStepSize * zSteps) + zLow

    # %% Collect the x/y counts

    xyNumSteps = xNumSteps+yNumSteps

    # The galvo's small angle step response is 400 us
    # Let's give ourselves a buffer of 500 us (500000 ns)
    period = readout + numpy.int64(500000)

    # Set up the galvo
    xGalvoVoltages = numpy.concatenate([xVoltages,
                                        numpy.full(yNumSteps, xCenter)])
    yGalvoVoltages = numpy.concatenate([numpy.full(xNumSteps, yCenter),
                                        yVoltages])

    streamWriter, galvoTask = galvo.stream_write_daq(daqName,
                                                     daqAOGalvoX, daqAOGalvoY,
                                                     daqDIPulserClock,
                                                     xGalvoVoltages,
                                                     yGalvoVoltages,
                                                     period)

	# Set up the APD
    streamReader, apdTask = apd.stream_read_load_daq(daqName, daqCIApd,
                                                     daqDIPulserClock,
                                                     daqDIPulserGate,
                                                     period)

    objective_piezo.write_single_open_loop(piezoSerial, zCenter)

    # Run the PulseStreamer
    tool_belt.pulser_readout_cont_illum(pulserIP, pulserDODaqClock,
                                        pulserDODaqGate, pulserDOAom,
                                        period, readout, xyNumSteps)

    # Collect the data
    timeout = ((period*(10**-9)) * xyNumSteps) + 10
    xyCounts = apd.stream_read_daq(streamReader, xyNumSteps, timeout)

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

    # %% Collect the z counts

    # If the user said stop, let's just stop
    if tool_belt.safe_stop():
        return optiCenters

    # Base this off the piezo hysteresis and step response
    period = readout + numpy.int64(500000)

    # Set up the galvo
    galvo.write_daq(daqName, daqAOGalvoX, daqAOGalvoY, xCenter, yCenter)
    
	# Set up the APD
    streamReader, apdTask = apd.stream_read_load_daq(daqName, daqCIApd,
                                                     daqDIPulserClock,
                                                     daqDIPulserGate,
                                                     period)
    
    # Set previousSample to 0 since we don't have any samples yet
    previousSample = 0
    
    zCounts = numpy.zeros(zNumSteps, dtype=numpy.uint32)
    
    pulser = tool_belt.get_pulser(pulserIP)

    # Set the PulseStreamer to start on python's command. We can run the
    # loaded stream repeatedly using startNow().
    pulser.setTrigger(start=Start.SOFTWARE)
    
    seq = tool_belt.get_readout_cont_illum_seq(pulserDODaqClock, pulserDODaqGate, 
                                               pulserDOAom, period, readout)
    
    pulser.stream(seq, 1, (0, [pulserDOAom], 0, 0))
    
    objective_piezo.write_single_open_loop(piezoSerial, zVoltages[0])
    time.sleep(0.5)
    
    for ind in range(zNumSteps):
        
        objective_piezo.write_single_open_loop(piezoSerial, zVoltages[ind])
        
        # Start the timing stream
        pulser.startNow()
        
        sample = streamReader.read_one_sample_uint32(timeout)
        
        # The counter task returns the cumulative count over the life of the
        # task. We want the individual count over each sample period.
        zCounts[ind] = sample - previousSample
        previousSample = sample

    # Run the PulseStreamer
    tool_belt.pulser_readout_cont_illum(pulserIP, pulserDODaqClock,
                                        pulserDODaqGate, pulserDOAom,
                                        period, readout, zNumSteps)

    # Close tasks
    for task in taskList:
        if task.name == apdTask.name:
            taskList.remove(task)
    apdTask.close()

	# %% Extract each dimension's counts

    start = 0
    end = xNumSteps
    xCounts = xyCounts[start: end]

    start = xNumSteps
    end = xNumSteps + yNumSteps
    yCounts = xyCounts[start: end]

	# %% Fit Gaussians and plot the data

	# Create 3 plots in the figure, one for each axis
    if doPlot:
        fig, axesPack = plt.subplots(1, 3, figsize=(17, 8.5))
        fig.canvas.draw()
        fig.canvas.flush_events()

	# Pack up
    centersPack = (xCenter, yCenter, zCenter)
    plusMinusPack = (scanCenterPlusMinus, scanCenterPlusMinus, depthCenterPlusMinus)
    voltagesPack = (xVoltages, yVoltages, zVoltages)
    countsPack = (xCounts, yCounts, zCounts)
    titlesPack = ("X Axis", "Y Axis", "Z Axis")

	# Loop over each dimension
    for ind in range(3):

        optimizationFailed = False

		# Unpack for the dimension
        center = centersPack[ind]
        voltagePlusMinus = plusMinusPack[ind]
        voltages = voltagesPack[ind]
        counts = countsPack[ind]
        title = titlesPack[ind]

        # Guess initial Gaussian fit parameters: coeff, mean, stdev, constY
        initParams = (23., center, voltagePlusMinus/3, 50.)

		# Least squares
        try:
            optiParams, varianceArr = curve_fit(tool_belt.gaussian, voltages,
                                                counts, p0=initParams)
        except Exception:
            optimizationFailed = True

        if not optimizationFailed:
            optiCenters[ind] = optiParams[1]

        # Plot the data
        if doPlot:
            ax = axesPack[ind]
            ax.plot(voltages, counts)
            ax.set_title(title)

    		# Plot the fit
            if not optimizationFailed:
                first = voltages[0]
                last = voltages[len(voltages)-1]
                linspaceVoltages = numpy.linspace(first, last, num=1000)
                gaussianFit = tool_belt.gaussian(linspaceVoltages, *optiParams)
                ax.plot(linspaceVoltages, gaussianFit)

                # Add info to the axes
                # a: coefficient that defines the peak height
    			# mu: mean, defines the center of the Gaussian
    			# sigma: standard deviation, defines the width of the Gaussian
    			# offset: constant y value to account for background
                text = "\n".join(("a=" + "%.3f"%(optiParams[0]),
                                  "$\mu$=" + "%.3f"%(optiParams[1]),
                                  "$\sigma$=" + "%.4f"%(optiParams[2]),
                                  "offset=" + "%.3f"%(optiParams[3])))

                props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)

            fig.canvas.draw()
            fig.canvas.flush_events()

	# %% Save the data

    # Don't bother saving the data if we're just using this to find the
    # optimized coordinates
    if doPlot:

        timeStamp = tool_belt.get_time_stamp()

        rawData = {"timeStamp": timeStamp,
                   "name": name,
                   "xyzCenters": [xCenter, yCenter, zCenter],
                   "scanRange": scanRange,
                   "scanStepSize": scanStepSize,
                   "depthRange": depthRange,
                   "depthStepSize": depthStepSize,
                   "readout": int(readout),
                   "imgResolution": [xNumSteps, yNumSteps],
                   "counts": [xCounts.astype(int).tolist(),
                              yCounts.astype(int).tolist(),
                              zCounts.astype(int).tolist()]}

        filePath = tool_belt.get_file_path("find_nv_center", timeStamp, name)
        tool_belt.save_figure(fig, filePath)
        tool_belt.save_raw_data(rawData, filePath)

    # %% Return the optimized centers
    
    print("\n    xCenter = %.3f\n    yCenter = %.3f\n    zCenter = %.1f\n" % 
              tuple(optiCenters))
    
    return optiCenters

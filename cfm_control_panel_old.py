# -*- coding: utf-8 -*-
"""
This file contains functions to control the CFM. Just change the function call
in the main section at the bottom of this file and run the file. For clarity
and ease of use, the do_ functions are intended to be called without arguments.
Microscope configuration parameters are at the top of the file. Function
parameters are at the top of each function.

Created on Sun Nov 25 14:00:28 2018

@author: mccambria
"""


# %% Imports


# User modules
import Utils.tool_belt as tool_belt
import MajorRoutines.find_nvs as find_nvs
import MajorRoutines.find_nv_center as find_nv_center
import MajorRoutines.stationary_count as stationary_count
import MajorRoutines.find_resonance as find_resonance
import MajorRoutines.find_resonance_edit as find_resonance_edit
<<<<<<< HEAD
import MajorRoutines.rabi as rabi
import MajorRoutines.correlation as correlation
=======
import MajorRoutines.Rabi as rabi
import MajorRoutines.pulse_sequence as pulse_sequence
>>>>>>> 2f9cff36273f001463154e5e9bb7b3643d65c398
import Outputs.xyz as xyz

# Library modules
import numpy


# %% Device Identifiers


# The IP address adopted by the PulseStreamer is hardcoded. See the lab wiki
# for information on how to change it
PULSE_STREAMER_IP = "128.104.160.11"
TIME_TAGGER_SERIAL = ""
# DAQ_NAME is the automatically assigned name of the DAQ
DAQ_NAME = "Dev1"
# The IP address of the RF signal generator is also hardcoded. You can
# change it via the signal generator's front panel. The rest of this is the
# pyvisa address of the device.
RF_ADDRESS = "TCPIP0::128.104.160.12::5025::SOCKET"
OBJECTIVE_PIEZO_SERIAL = "119008970"


# %% Wiring


# Wiring constants have the naming scheme:
# <device>_<connection type>_<to/from device>_<purpose (if necessary)>

# Connection types are:
# AO (analog output)
# AI (analog input)
# DO (digital output)
# DI (digital input)
# CI (counter input)
# CO (counter output)

# Outputs and inputs are single ints that map to the specified channel type.

# When wiring the DAQ, it is extremely useful to consult the device pinout,
# which can be accessed from MAX. It'll tell you which ctr channels are wired
# to which PFI channels. It is also useful to check the physical channels on
# the device. Try the following, which uses ci_physical_chans as an example:
# DAQ = nidaqmx.system.device.Device("dev1")
# for chan in DAQ.ci_physical_chans:
#     print(str(chan))

DAQ_AO_GALVO_X = 0
DAQ_AO_GALVO_Y = 1

# PFI channels for counter sources (connected to APD)
DAQ_CI_APD_0 = 8
DAQ_CI_APD_1 = 3
DAQ_CI_APD_3 = 5

DAQ_AI_PHOTOMETER = 0

TAGGER_DI_APD_0 = 0
TAGGER_DI_APD_1 = 1

# The DAQ has its digital inputs labeled as "PFI".
# I'll label them DI here for consistency.
DAQ_DI_PULSER_CLOCK = 12
DAQ_DI_PULSER_START = 10  # PFI 10 on the BNC 2110 maps to ctr 3 src
DAQ_DI_PULSER_GATE_0 = 0 # PFI 0 on the BNC will be used to gate ctr 0
DAQ_DI_PULSER_GATE_1 = 1 # PFI 1 on the BNC will be used to gate ctr 1

PULSER_DO_DAQ_CLOCK = 0
PULSER_DO_DAQ_START = 1
PULSER_DO_DAQ_GATE_0 = 5 # Pulser ch 5 is connected to PFI 0, and will be used to gate ctr 0
PULSER_DO_DAQ_GATE_1 = 2 # Pulser ch 2 is connected to PFI 1, and will be used to gate ctr 1
PULSER_DO_AOM = 3
PULSER_DO_RF = 4




# %% Major Routines


def do_find_nvs(name, xCenter, yCenter, zCenter,
                xScanRange, yScanRange, scanStepSize, readout):

    # Function-specific parameters
    continuous = False

    # Run the function
    find_nvs.main(PULSE_STREAMER_IP, DAQ_NAME,
                  DAQ_AO_GALVO_X, DAQ_AO_GALVO_Y, OBJECTIVE_PIEZO_SERIAL, DAQ_CI_APD_0,
                  DAQ_DI_PULSER_CLOCK, DAQ_DI_PULSER_GATE_0,
                  PULSER_DO_DAQ_CLOCK, PULSER_DO_DAQ_GATE_0, PULSER_DO_AOM,
                  name, xCenter, yCenter, zCenter, xScanRange, yScanRange,
                  scanStepSize, readout, continuous)


def do_find_nv_center(name, xCenter, yCenter, zCenter):

    # Function-specific parameters
    scanRange = 0.04
    scanStepSize = scanRange / 60
    depthRange = 5.0
    depthStepSize = depthRange / 60
    readout = numpy.int64(10 * 10**6)

    # Run the function
    return find_nv_center.main(PULSE_STREAMER_IP, DAQ_NAME,
						DAQ_AO_GALVO_X, DAQ_AO_GALVO_Y, OBJECTIVE_PIEZO_SERIAL,
						DAQ_CI_APD_0, DAQ_DI_PULSER_CLOCK, DAQ_DI_PULSER_GATE_0,
						PULSER_DO_DAQ_CLOCK, PULSER_DO_DAQ_GATE_0, PULSER_DO_AOM,
						name, xCenter, yCenter, zCenter, scanRange,
						scanStepSize, depthRange, depthStepSize, readout)


def do_stationary_count(xCenter, yCenter, zCenter, readout):

    # Function-specific parameters
    numSamples = 1000
    readout = numpy.int64(100 * 10**6)
    period = readout

    # Run the function
    stationary_count.main(PULSE_STREAMER_IP, DAQ_NAME,
                          DAQ_AO_GALVO_X, DAQ_AO_GALVO_Y, OBJECTIVE_PIEZO_SERIAL,
                          DAQ_CI_APD_0, DAQ_DI_PULSER_CLOCK, DAQ_DI_PULSER_GATE_1,
                          PULSER_DO_DAQ_CLOCK, PULSER_DO_DAQ_GATE_1, PULSER_DO_AOM,
                          xCenter, yCenter, zCenter,
                          numSamples, period, readout)


def do_find_resonance(name, xCenter, yCenter, zCenter, readout):

    # Function-specific parameters
    freqCenter = 2.875
    freqRange = 0.3
    freqResolution = 240
    rfPower = -15.0
    readout = 100 * 10**6

    # Run the function
    find_resonance.main(PULSE_STREAMER_IP, DAQ_NAME, RF_ADDRESS,
                        DAQ_AO_GALVO_X, DAQ_AO_GALVO_Y, OBJECTIVE_PIEZO_SERIAL,
                        DAQ_CI_APD_0, DAQ_DI_PULSER_CLOCK, DAQ_DI_PULSER_GATE_0,
                        PULSER_DO_DAQ_CLOCK, PULSER_DO_DAQ_GATE_0,
                        PULSER_DO_AOM, PULSER_DO_RF,
                        name, xCenter, yCenter, zCenter,
                        freqCenter, freqRange, freqResolution, rfPower,
                        readout)

def do_ESR(name, xCenter, yCenter, zCenter):

    # Function-specific parameters
    freqCenter = 2.87
    freqRange = 0.30
    freqResolution = 120
    rfPower = -13.0
    readout1 = 100 * 10**6
    readout2 = readout1
    numRunsToAvg = 1

    # Run the function
    find_resonance_edit.main(PULSE_STREAMER_IP, DAQ_NAME, RF_ADDRESS,
                        DAQ_AO_GALVO_X, DAQ_AO_GALVO_Y, OBJECTIVE_PIEZO_SERIAL,
                        DAQ_CI_APD_0, DAQ_DI_PULSER_CLOCK, DAQ_DI_PULSER_GATE_0,
                        PULSER_DO_DAQ_CLOCK, PULSER_DO_DAQ_GATE_0,
                        PULSER_DO_AOM, PULSER_DO_RF,
                        name, xCenter, yCenter, zCenter,
                        freqCenter, freqRange, freqResolution, rfPower,
                        readout1, readout2, numRunsToAvg)

def do_rabi(name, xCenter, yCenter, zCenter):

    # Function specific parameters
<<<<<<< HEAD
    rfFrequency = 2.875
    rfPower = -13.0
    rfMaxTime = 100000
    timeResolution = 2
    iterations = 1

=======
    AOMDelay = 750
    rfFrequency = 2.907
    rfPower = 5
    rfMinTime = 0
    rfMaxTime = 100
    numTimeSteps = 21
    nSamples = 10**5
    
>>>>>>> 2f9cff36273f001463154e5e9bb7b3643d65c398
    # Run the function
    rabi.main(PULSE_STREAMER_IP, DAQ_NAME, RF_ADDRESS,
                    DAQ_AO_GALVO_X, DAQ_AO_GALVO_Y, OBJECTIVE_PIEZO_SERIAL,
                    DAQ_CI_APD_0, DAQ_CI_APD_1, DAQ_DI_PULSER_CLOCK, DAQ_DI_PULSER_GATE_0, DAQ_DI_PULSER_GATE_1, 
                    PULSER_DO_DAQ_CLOCK, PULSER_DO_DAQ_GATE_0, PULSER_DO_DAQ_GATE_1, 
                    PULSER_DO_AOM, PULSER_DO_RF,
                    name, xCenter, yCenter, zCenter, AOMDelay, rfFrequency, rfPower,
                    rfMinTime, rfMaxTime, numTimeSteps, nSamples)
    
def do_sequence(name, xCenter, yCenter, zCenter):

    # Function-specific parameters
    startTime = 1000
    endTime = 3000
    numSteps = 61
    nSamples = 100000

    # Run the function
    pulse_sequence.main(PULSE_STREAMER_IP, DAQ_NAME, RF_ADDRESS,
                                 DAQ_AO_GALVO_X, DAQ_AO_GALVO_Y, OBJECTIVE_PIEZO_SERIAL, DAQ_CI_APD_0,
                                 DAQ_CI_APD_1, DAQ_DI_PULSER_CLOCK, DAQ_DI_PULSER_GATE_0, DAQ_DI_PULSER_GATE_1,
                                 PULSER_DO_DAQ_CLOCK, PULSER_DO_DAQ_GATE_0, PULSER_DO_DAQ_GATE_1,
                                 PULSER_DO_AOM, PULSER_DO_RF,
                                 name, xCenter, yCenter, zCenter,
                                 startTime, endTime, numSteps,
                                 nSamples)


def do_correlation(name, x_center, y_center, z_center):

    # Function specific parameters
    bin_width = 1000  # In ps
    num_bins = 5000

    # Optimize
    opti_x, opti_y, opti_z = do_find_nv_center(name,
                                               x_center, y_center, z_center)

    input('Switch the APD lines to the proper Time Tagger channels.\n'
          'Then press enter to continue.')

    # Run the function
    correlation.main(PULSE_STREAMER_IP, DAQ_NAME,
                     DAQ_AO_GALVO_X, DAQ_AO_GALVO_Y, OBJECTIVE_PIEZO_SERIAL,
                     TAGGER_DI_APD_0, TAGGER_DI_APD_1,
                     name, opti_x, opti_y, opti_z)


# %% Utility functions


def do_set_xyz(xCenter, yCenter, zCenter):
    xyz.write_daq(DAQ_NAME, DAQ_AO_GALVO_X, DAQ_AO_GALVO_Y, OBJECTIVE_PIEZO_SERIAL,
                  xCenter, yCenter, zCenter)


def do_set_xyz_zero():
    xyz.write_daq(DAQ_NAME, DAQ_AO_GALVO_X, DAQ_AO_GALVO_Y, OBJECTIVE_PIEZO_SERIAL,
                  0.0, 0.0, 50.0)


def do_gate_aom():
    tool_belt.pulser_all_zero(PULSE_STREAMER_IP)
    tool_belt.poll_safe_stop()

def do_ungate_aom():
    tool_belt.pulser_high(PULSE_STREAMER_IP, PULSER_DO_AOM)
    tool_belt.poll_safe_stop()

def do_rf_on():
    tool_belt.pulser_high(PULSE_STREAMER_IP, PULSER_DO_RF)


# %% Script Code


# Functions only run when called. Since this part of the script is not in a
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == "__main__":

    # %% Shared parameters
    # The file has minimal documentation.
    # For more, view the function definitions in their respective file.

    name = "Ayrton9"
<<<<<<< HEAD

#    xCenter = 0.0
#    yCenter = 0.0
#    zCenter = 50.0


    xCenter = -0.039
    yCenter = -0.007
    zCenter = 50.3

#    xCenter = 0.008
#    yCenter = -0.067
#    zCenter = 48.1
=======
    
    xCenter = 0.0
    yCenter = 0.0
    zCenter = 47.4
    
#    xCenter = -0.043
#    yCenter = 0.076
#    zCenter = 47.4


#    xCenter = 0.122
#    yCenter = -0.01
#    zCenter = 47.6
    
#    xCenter = 0.065
#    yCenter = 0.025
#    zCenter = 47.6
>>>>>>> 2f9cff36273f001463154e5e9bb7b3643d65c398


    # 1 V => ~100 um
    # With gold nanoparticles 0.4 is good for broad field
    # 0.04 is good for a single particle

    scanRange = 3.0

    
    xScanRange = scanRange
    yScanRange = scanRange
    scanStepSize = scanRange / 100

    # find_nvs_center sweeps over the full 10 V piezo range right now
    depthRange = 5.0
    depthStepSize = depthRange / 60
    # Note that the piezo has a slew rate of ~25V/s in terms of the ext in
    # 1 volt -> 0.465 um for piezo
#    depthRange = 5.0
#    if zCenter <= 5.0:
#        depthRange = zCenter
#    else:
#        depthRange = 10.0 - zCenter
#    depthStepSize = depthRange / 60

    readout = numpy.int64(10 * 10**6)

    # %% Functions to run

    try:
        # Put the do functions you want to run below. The code in the finally
        # block will run no matter what happens in this try block (i.e. the
        # finally block will run even if there's a crash here).

#        do_set_xyz(xCenter, yCenter, zCenter)
#        do_set_xyz_zero()
<<<<<<< HEAD
#        do_find_nvs(name, xCenter, yCenter, zCenter, xScanRange, yScanRange, scanStepSize, readout)
        do_stationary_count(xCenter, yCenter, zCenter, readout)
#        do_find_nv_center(name, xCenter, yCenter, zCenter, readout)
=======
        do_find_nvs(name, xCenter, yCenter, zCenter, xScanRange, yScanRange, scanStepSize, readout)
#        do_stationary_count(xCenter, yCenter, zCenter, readout)
#        do_find_nv_center(name, xCenter, yCenter, zCenter, scanRange, scanStepSize, depthRange, depthStepSize, readout)
>>>>>>> 2f9cff36273f001463154e5e9bb7b3643d65c398
#        do_find_resonance(name, xCenter, yCenter, zCenter, 100 * 10**6)
#         do_sequence(name, xCenter, yCenter, zCenter)
#        do_ESR(name, xCenter, yCenter, zCenter)
#         do_rabi(name, xCenter, yCenter, zCenter)
        # do_correlation(name, xCenter, yCenter, zCenter)
#        do_gate_aom()
#        do_ungate_aom()
#         pass
    finally:
        tool_belt.clean_up(PULSE_STREAMER_IP, DAQ_NAME)
        # By default we may want some channels ungated
        tool_belt.pulser_high(PULSE_STREAMER_IP, [PULSER_DO_AOM])

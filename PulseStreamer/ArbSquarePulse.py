# -*- coding: utf-8 -*-
"""
This file contains code for the pulse streamer to output a pulse of arbitrary 
duty time to the pulse streamer.

Code taken and modified from Swabian file: Pulse streamer SimplePulses

Created on Fri Dec 28 11:14:15 2018

@author: Gardill
"""

PULSE_STREAMER_IP = "128.104.160.11" #defines pulse streamer IP address
from pulse_streamer_jrpc import PulseStreamer # import Pulse Streamer wrapper class
from pulse_streamer_jrpc import Start, Mode # import enum types  
from Sequence import Sequence #imports sequence definition

# %% Device Identifiers

# The IP address adopted by the PulseStreamer is hardcoded. See the lab wiki
# for information on how to change it
pulser = PulseStreamer(PULSE_STREAMER_IP)

# %% Defining the sequence

#Name the channels to be used. Add PulseStreamer channels as needed
ch_ref = 5 # output channel 5

#define the off/on time

ch_ref_off = 500 #ns
ch_ref_on  = 130 #ns

#define digital levels
HIGH=1
LOW=0

#define every channel by its own
# simply add more pulses with ', (time, HIGH/LOW)'
seq_ref = [(ch_ref_off, LOW), (ch_ref_on, HIGH)]

#create the sequence
seq = Sequence()

seq.setDigitalChannel(ch_ref , seq_ref)
                      
#run the sequence a number of times
n_runs = 100000000
#n_runs = 'INFIITE' # repeat the sequence all the time)

#reset the device - all outputs 0V
pulser.reset()

#set constant state of the device
pulser.constant('CONSTANT_ZERO') #all outputs 0V

# define the final state of the Pulsestreamer - the device will enter this 
#state, when the sequence is finished
final = (0,[],0,0)

#Start via the trigger input and enable the retrigger-function
#start = Start.HARDWARE_RISING
#mode = Mode.NORMAL

#Start the sequence after the upload and disable the retrigger-function
start = Start.IMMEDIATE
mode = Mode.SINGLE

pulser.setTrigger(start=start, mode=mode)

print (" ")
print ("Single period:", (ch_ref_off + ch_ref_on), "ns")
print ("Full time of sequence:", (ch_ref_off + ch_ref_on)/1000000000 * n_runs, "s")
print (" ")
print ("\nGenerated sequence pulse list:")
print ("Data format: list of (duration [ns], digital bit pattern, analog 0, analog 1)")
print(seq.getSequence())
#upload the sequence and arm the device
pulser.stream(seq.getSequence(), n_runs, final)
print ("\nOutput running on Pulse Streamer")

"""
This file shows how to use the PulseStreamer 8/2.
The PulseStreamer 8/2 describes pulses in the form (time, ['ch0', 'ch3'], 0.8, -0.4),
where time is an integer in ns (clock ticks), ['ch0','ch3'] is a list naming the channels
which should be high the last two numbers specify the analog outputs in volt.

You can either communicate with the hardware via JSON-RPC (recommended) or Google-RPC (see RAW_json_grpc folder).

JSON-RPC:
- To use this script, you should have installed the python module <tinyrpc>.
  If you do not have the module installed already, type the following line in your terminal:
  > pip install tinyrpc
- In your work-folder (or in the python path), you should have the follwing files
  <pulse_streamer_jrpc.py> and <tinyrpc3.py>.
"""

import sys

ip_hostname="128.104.160.11" # edit this line to use a specific Pulse Streamer IP address

from pulse_streamer_jrpc import PulseStreamer # import Pulse Streamer wrapper class

from pulse_streamer_jrpc import Start, Mode# import enum types  

from Sequence import Sequence

#python module for scientific computing only used for creating the random pulse and merging signals
import numpy as np


#create Pulsestreamer
"""To set the IP-Address of the Pulsestreamer see
https://www.swabianinstruments.com/static/documentation/PulseStreamer/sections/network.html
"""
pulser = PulseStreamer(ip_hostname)

#define the sequence
ch_pockels   = 0 # output channel 0
ch_camera1   = 1 # output channel 1
ch_camera2   = 2 # output channel 2
ch_ref       = 3 # output channel 3
ch_ref2      = 4 # output channel 4
trigger_high_time = 10 #ns
delay_pockels = 5 #ns
delay_camera1 = 100 #ns
delay_camera2 = 150 #ns
refPeriod     = 20 #ns
refPeriod2    = 1000000 #ns

#define digital levels
HIGH=1
LOW=0

#define every channel by its own
# simply add more pulses with ', (time, HIGH/LOW)'
  #seq_pockels = [(delay_pockels, LOW), (trigger_high_time, HIGH)]
  #seq_camera1 = [(delay_camera1, LOW), (trigger_high_time, HIGH)]
  #seq_camera2 = [(delay_camera2, LOW), (trigger_high_time, HIGH)]

# 10 pulses with a period of refPeriod and refPeriod2
  #seq_ref     = [(int(refPeriod - (refPeriod / 2))  , HIGH), (int(refPeriod / 2), LOW)] * 10
seq_ref2    = [(int(refPeriod2), HIGH), (int(2*refPeriod2), LOW)] * 10

#create the sequence
seq = Sequence()

  #seq.setDigitalChannel(ch_pockels, seq_pockels)
  #seq.setDigitalChannel(ch_camera1, seq_camera1)
  #seq.setDigitalChannel(ch_camera2, seq_camera2)
  #seq.setDigitalChannel(ch_ref, seq_ref)
seq.setDigitalChannel(ch_ref2 , seq_ref2)

#run the sequence a number of times
n_runs = 1000
#n_runs = 'INFIITE' # repeat the sequence all the time

#reset the device - all outputs 0V
pulser.reset()

#set constant state of the device
pulser.constant('CONSTANT_ZERO') #all outputs 0V

# define the final state of the Pulsestreamer - the device will enter this state, when the sequence is finished
final = (0,[],0,0)

#Start via the trigger input and enable the retrigger-function
#start = Start.HARDWARE_RISING
#mode = Mode.NORMAL

#Start the sequence after the upload and disable the retrigger-function
start = Start.IMMEDIATE
mode = Mode.SINGLE

pulser.setTrigger(start=start, mode=mode)

print ("\nGenerated sequence pulse list:")
print ("Data format: list of (duration [ns], digital bit pattern, analog 0, analog 1)")
print(seq.getSequence())
#upload the sequence and arm the device
#pulser.stream(seq.getSequence(), n_runs, final)
print ("\nOutput running on Pulse Streamer")
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 21:08:10 2019

@author: Matt
"""

from PulseStreamer.Sequence import Sequence
import numpy as np

pulserDODaqClock = [0]
pulserDODaqStart = []
pulserDODaqGate = []  # [2]
pulserDODaqSignal = []

# period = 1000
period = np.int64(0.01 * 10**9)
readout = np.int64(0.35 * 10**9)
totalSamples = 500
low = 0
high = 1
halfPeriod = period // 2
seq = Sequence()

# Each channel in the sequence must have the same time length, or else
# getSequence will not properly union the channels.
train = [(period, low)]
trainElem = [(halfPeriod, high), (halfPeriod, low)]
train.extend(trainElem * totalSamples)
for chan in pulserDODaqClock:
    seq.setDigitalChannel(chan, train)

train = [(period, high)]
trainElem = [(period, low)]
train.extend(trainElem * totalSamples)
for chan in pulserDODaqStart:
    seq.setDigitalChannel(chan, train)

trainElem = [(period - readout, low), (readout, high)]
train = trainElem * (totalSamples + 1)
for chan in pulserDODaqGate:
    seq.setDigitalChannel(chan, train)

# train = [(100, low), (100, high), (100, low)]
trainElem = [(100, low), (100, high), (period - 200, low)]
train = trainElem * (totalSamples + 1)
for chan in pulserDODaqSignal:
    seq.setDigitalChannel(chan, train)

fullSeq = seq.getSequence()
index = 0
total = 0
for step in fullSeq:
    index += 1
    print(index)
    total += step[0]
    print(total)

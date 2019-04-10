# -*- coding: utf-8 -*-
"""
g2 simulation

Created on Sun Mar 17 10:25:51 2019

@author: mccambria
"""

import numpy
import matplotlib.pyplot as plt


def intensity(x):
    return (numpy.sin(x))


numInputSamples = 1000
integrationPeriod = 100.0
numTauSamples = 1000
tauRange = 10.0
tau = 0.5

inputSamples = numpy.arange(numInputSamples)
inputStepSize = integrationPeriod / numInputSamples
inputSamples = (inputStepSize * inputSamples)


def plot_gTwo():

    plt.figure()
    
    tauSamples = numpy.arange(numTauSamples)
    tauStepSize = tauRange / numTauSamples
    tauCenterInd = numTauSamples // 2
    tauCenter = tauSamples[tauCenterInd]
    tauSamples = tauSamples - tauCenter
    tauSamples = tauSamples * tauStepSize
    
    outputVals = intensity(inputSamples)
    avgIntensity = numpy.average(outputVals)
    avgIntensitySquared = avgIntensity ** 2
    # avgIntensitySquared = numpy.average(numpy.multiply(outputVals, outputVals))
    
    gTwo = numpy.empty(numTauSamples)

    for index in range(numTauSamples):
        tau = tauSamples[index]
        outputVals = intensity(inputSamples)
        inputDelaySamples = inputSamples + tau
        outputDelayVals = intensity(inputDelaySamples)
        delayProd = numpy.multiply(outputVals, outputDelayVals)
        avgDelayProd = numpy.average(delayProd)
        gTwo[index] = avgDelayProd / avgIntensitySquared
    
    plt.plot(tauSamples, gTwo)


def plot_delay_prod():

    plt.figure()
    
    outputVals = intensity(inputSamples)
    plt.plot(inputSamples, outputVals)

    inputDelaySamples = inputSamples + tau
    outputDelayVals = intensity(inputDelaySamples)
    plt.plot(inputSamples, outputDelayVals)

    delayProd = numpy.multiply(outputVals, outputDelayVals)
    plt.plot(inputSamples, delayProd)


if __name__ == '__main__':
    plot_delay_prod()
    plot_gTwo()

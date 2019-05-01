# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:23:08 2019

@author: mccambria
"""

import rsa_api
import numpy as np

############# Taken from example code here: 
# https://github.com/tektronix/RSA_API/blob/master/Python/Cython%20Version/cython_example.py

def search_connect():
    print('API Version {}'.format(rsa_api.DEVICE_GetAPIVersion_py()))
    try:
        numDevicesFound, deviceIDs, deviceSerial, deviceType = rsa_api.DEVICE_Search_py()
    except rsa_api.RSAError:
        print(rsa_api.RSAError)
        
    print('Number of devices: {}'.format(numDevicesFound))
    if numDevicesFound > 0:
        print('Device serial numbers: {}'.format(deviceSerial[0].decode()))
        print('Device type: {}'.format(deviceType[0].decode()))
        rsa_api.DEVICE_Connect_py(deviceIDs[0])
    else:
        print('No devices found, exiting script.')
        exit()
    rsa_api.CONFIG_Preset_py()

def config_spectrum(center_freq=2.87e9, ref_level=0,
                    freq_span=30e6, resolution_bandwidth=300e3):
    rsa_api.SPECTRUM_SetEnable_py(True)
    rsa_api.CONFIG_SetCenterFreq_py(center_freq)
    rsa_api.CONFIG_SetReferenceLevel_py(ref_level)

    rsa_api.SPECTRUM_SetDefault_py()
    rsa_api.SPECTRUM_SetSettings_py(span=freq_span, rbw=ref_level,
                                    traceLength=801)
    specSet = rsa_api.SPECTRUM_GetSettings_py()
    return specSet

def create_frequency_array(specSet):
    # Create array of frequency data for plotting the spectrum.
    freq = np.arange(specSet['actualStartFreq'], specSet['actualStartFreq']
                     + specSet['actualFreqStepSize'] * specSet['traceLength'],
                     specSet['actualFreqStepSize'])
    return freq

def peak_power_detector(freq, trace):
    peakPower = np.amax(trace)
    peakFreq = freq[np.argmax(trace)]

    return peakPower, peakFreq

########################

search_connect()
center_freq = 2.87e9
ref_level = 0
freq_span = 0.3e9
resolution_bandwidth = 10e3
specSet = config_spectrum(center_freq, ref_level,
                          freq_span, resolution_bandwidth)
trace = rsa_api.SPECTRUM_Acquire_py(rsa_api.SpectrumTraces.SpectrumTrace1,
                                    specSet['traceLength'], 100)
freq = create_frequency_array(specSet)
peakPower, peakFreq = peak_power_detector(freq, trace)

print(peakPower)



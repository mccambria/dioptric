# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:31:30 2019

@author: Aedan
"""
import numpy
import matplotlib.pyplot as plt

folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/spectra/NV/Goeppert-Mayer'
file_off = '2019_11_06-19_44_00_01-off_NV.txt' # Off NV

file_on = '2019_11_06-21_12_13_01-on_NV.txt' # On NV

wavelength_on_list = []
counts_on_list = []

wavelength_off_list = []
counts_off_list = []

# Read in the wavelengths and counts for when we are sitting on the NV
on = open(folder + '/' + file_on, 'r')

on_lines = on.readlines()
for line in on_lines:
    wavelength, counts = line.split()
    wavelength_on_list.append(float(wavelength))
    counts_on_list.append(float(counts))
    
# Read in the wavelengths and counts for when we are sitting on the NV
off = open(folder + '/' + file_off, 'r')

off_lines = off.readlines()
for line in off_lines:
    wavelength, counts = line.split()
    wavelength_off_list.append(float(wavelength))
    counts_off_list.append(float(counts))
    
# the list for on_NV is 22 elements longer than that of Off_NV... huh. For
# now I'll just delete the last 22 elements 
del counts_on_list[len(counts_on_list) - 22:len(counts_on_list)]
    
# turn the lists into arrays

wavelength_on_array = numpy.array(wavelength_on_list)
counts_on_array = numpy.array(counts_on_list)

wavelength_off_array = numpy.array(wavelength_off_list)
counts_off_array = numpy.array(counts_off_list)

# Subtract the two: ON - OFF
counts = counts_on_array - counts_off_array

# pLot the subtracted counts!

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.plot(wavelength_off_array, counts)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Counts (arb.)')
ax.set_title('Subtracted spectra of NV in bulk')



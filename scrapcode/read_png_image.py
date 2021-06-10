# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:22:38 2020

Tests of reading in binary pixel values from png image

@author: Aedan
"""

from PIL import Image
import numpy
import matplotlib.pyplot as plt

#Opens the Image
im_file = Image.open('E:/Shared drives/Kolkowitz Lab Group/SCC_image_test.png')

##Reads the image pixel information
arr = numpy.array(im_file)

##Sets the width, height and maze size variables
width = im_file.size[0]
height = im_file.size[1]
size = width * height

###Defines the mapping array
#map = numpy.zeros([width, height], dtype=numpy.int)

##Prints maze information for debugging
#print ('Maze width:', width)
#print ('Maze height:', height)
#print ('Maze size:', size, '\n')

print(arr)

dark_px_list = []

for row_px in range(width):
    for column_px in range(height):
        if arr[row_px][column_px].all() == 0:
            dark_px_list.append([row_px, column_px])
            
dark_px_arr = numpy.array(dark_px_list)

            
print(dark_px_arr)

plt.imshow(arr)

###Prints mapping array for debugging
#print (DataFrame(map))
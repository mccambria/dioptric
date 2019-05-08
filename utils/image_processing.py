# -*- coding: utf-8 -*-
"""
Image processing utils

Created on Mon May  6 18:22:43 2019

@author: mccambria
"""

import numpy

def convert_to_8bit(img):
    img = img.astype(numpy.float64)
    img -= numpy.nanmin(img)  # Set the lowest value to 0
    img *= (255/numpy.nanmax(img))
    img = img.astype(numpy.uint8)
    return img

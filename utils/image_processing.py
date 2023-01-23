# -*- coding: utf-8 -*-
"""
Image processing utils

Created on Mon May  6 18:22:43 2019

@author: mccambria
"""

import numpy

def convert_to_8bit(img, min_val=0, max_val=255):
    img = img.astype(numpy.float64)
    img -= numpy.nanmin(img)  # Set the lowest value to 0
    img *= (max_val/numpy.nanmax(img))
    img = img.astype(numpy.uint8)
    img += min_val
    return img
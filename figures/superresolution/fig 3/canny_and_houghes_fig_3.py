#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 12:57:14 2022

@author: sissi00
"""


from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from collections import defaultdict
from math import sqrt, atan2, pi


import copy
import matplotlib.pyplot as plt 
import numpy
import json

from scipy import optimize
from pathlib import PurePath

color_1 = '#00a651'
color_2 = '#f7941d'

datadir = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_master/SPaCE/2021_09'

def get_raw_data(path_from_nvdata, file_name,
                 data_dir=datadir):
    """Returns a dictionary containing the json object from the specified
    raw data file.
    """

    data_dir = PurePath(data_dir, path_from_nvdata)
    file_name_ext = '{}.txt'.format(file_name)
    file_path = data_dir / file_name_ext

    with open(file_path) as file:
        return json.load(file)
    
def canny_edge_detector(readout_image_array,num_steps):
    input_pixels = readout_image_array
    width = num_steps
    height = num_steps

    # Blur it to remove noise
    blurred = compute_blur(input_pixels, width, height)

    # Compute the gradient
    gradient, direction = compute_gradient(blurred, width, height)

    # Non-maximum suppression
    filter_out_non_maximum(gradient, direction, width, height)

    #print (numpy.max(gradient))
    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 0.2, 0.6)
    #print (keep)

    return keep



def compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = numpy.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = numpy.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn][ yn] * kernel[a][ b]
            blurred[x][ y] = acc
    return blurred

def compute_gradient(input_pixels, width, height):
    gradient = numpy.zeros((width, height))
    direction = numpy.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1][ y] - input_pixels[x - 1][ y]
                magy = input_pixels[x][ y + 1] - input_pixels[x][ y - 1]
                gradient[x][ y] = sqrt(magx**2 + magy**2)
                direction[x][ y] = atan2(magy, magx)
    return gradient, direction


def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x][ y] if direction[x][ y] >= 0 else direction[x][ y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x][ y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1][ y] > mag or gradient[x + 1][ y] > mag)
                    or (rangle == 1 and (gradient[x - 1][ y - 1] > mag or gradient[x + 1][ y + 1] > mag))
                    or (rangle == 2 and (gradient[x][ y - 1] > mag or gradient[x][ y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1][ y - 1] > mag or gradient[x - 1][ y + 1] > mag))):
                gradient[x][ y] = 0

def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x][ y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a][ y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)


def find_centers(file, do_plot = False):
    data = get_raw_data('', file[:-4])
    
    readout_image_array = data['readout_image_array']
    
    
    img_range = data['img_range_2D'][0]
    num_steps = data['num_steps_a']
    nv_sig = data['nv_sig']
    coords = nv_sig['coords']
    x_coord = coords[0]          
    y_coord = coords[1]
    half_range = img_range / 2
    x_high = x_coord + half_range
    x_low = x_coord - half_range
    y_high = y_coord + half_range
    y_low = y_coord - half_range
    
    
    x = numpy.linspace(x_high, x_low, num_steps)
    y = numpy.linspace(y_high, y_low, num_steps)
    
    half_pix = (x[1]-x[2])/2
    full_pix = 2*half_pix
    print('pixel size = {} V'.format(full_pix))
    
    img_extent = [x_high+abs(half_pix), x_low-abs(half_pix), 
                    y_low-abs(half_pix), y_high+abs(half_pix) ]
    extent = tuple(img_extent)
    if do_plot:
        fig, ax = plt.subplots()
        img =ax.imshow(readout_image_array, extent = extent, cmap = 'inferno')
        clb = plt.colorbar(img)
    empty_array = numpy.zeros([num_steps,num_steps])
    for x_pos, y_pos in canny_edge_detector(readout_image_array,num_steps):
        #print ("x_pos, y_pos", x_pos, y_pos)
        empty_array[x_pos][y_pos] = 1 
        # ax.plot(x[y_pos], y[x_pos],'ro')
    if do_plot:
        fig2, ax2 = plt.subplots()
        img2 =ax2.imshow(empty_array, extent = extent, cmap = 'BuPu_r')
        clb = plt.colorbar(img2) 


    # Find circles
    rmin = 26
    rmax = 31
    steps =90
    threshold = 0.33 # percentage of proper pixels on the circle
    
    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))
    
    acc = defaultdict(int)
    for x_pos, y_pos in canny_edge_detector(readout_image_array,num_steps):
        for r, dx, dy in points:
            a = x_pos - dx
            b = y_pos - dy
            acc[(a, b, r)] += 1
    print(points)
    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x_pos, y_pos, r = k
        #print (v/steps)
        if v / steps >= threshold :#and all((x_pos - xc) ** 2 + (y_pos - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            print(v / steps, x_pos, y_pos, r)
            circles.append((x_pos, y_pos, r))
    
    if do_plot:
        colors = [color_1, color_2]
        for i in range(len(circles)-1):
            x_pos, y_pos, r = circles[i]
            angle = numpy.linspace(0,2*numpy.pi,180)
            x_find = x[y_pos] + r * full_pix * numpy.cos(angle)
            y_find = y[x_pos] + r * full_pix * numpy.sin(angle)
            ax.plot(x_find, y_find,'g-', color = colors[i], linewidth = 1)
            
            ax.plot(x[y_pos], y[x_pos], 'wo',  color = colors[i],)

    b_x = x[circles[0][1]]
    b_y = y[circles[0][0]]
    b_r = circles[0][2]*full_pix 
    
    a_x = x[circles[1][1]]
    a_y = y[circles[1][0]]
    a_r = circles[1][2]*full_pix 
    
    r_diff = numpy.sqrt((b_x - a_x)**2 + (b_y - a_y)**2)
    print('radial dist between points = {} V'.format(r_diff))
    
    return a_x, a_y, a_r, b_x, b_y, b_r

if __name__ == "__main__":
    file = '2021_09_30-13_18_47-johnson-dnv7_2021_09_23.txt'
    find_centers(file, do_plot = True)
        
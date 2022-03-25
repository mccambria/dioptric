#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:38:30 2021

@author: sissi00
"""



import matplotlib.pyplot as plt 
import numpy
import json

#from scipy import optimize
from pathlib import PurePath
from scipy import linalg, optimize

data_dir = '/Users/sissi00/Desktop/AMO lab/Airy circle fitting'

def get_raw_data(path_from_nvdata, file_name,
                 data_dir='/Users/sissi00/Desktop/AMO lab/2021_09_30-13_18_47-johnson-dnv7_2021_09_23'):
    """Returns a dictionary containing the json object from the specified
    raw data file.
    """

    data_dir = PurePath(data_dir, path_from_nvdata)
    file_name_ext = '{}.txt'.format(file_name)
    file_path = data_dir / file_name_ext

    with open(file_path) as file:
        return json.load(file)

def calc_R_l(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return numpy.sqrt((b3_x_l-xc)**2 + (b3_y_l-yc)**2)

def f_2_l(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R_l(*c)
    return Ri - Ri.mean()


def calc_R_r(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return numpy.sqrt((b3_x_r-xc)**2 + (b3_y_r-yc)**2)

def f_2_r(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R_r(*c)
    return Ri - Ri.mean()

# imput these initial guess of center position by eyes
x0_guess_l = -0.018
y0_guess_l = 0.283
#center_range = 5

k = 0.5 # threshold for the value on each pixel

x0_steps = 21 # number of steps we go through with the center
y0_steps = 21 # number of steps we go through with the center
away_b = 6 # defining the number of pixels from the radius we want, preparation for final least square fit

file = '2021_09_30-13_18_47-johnson-dnv7_2021_09_23.txt'
data = get_raw_data('', file[:-4], data_dir = data_dir)

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

img_extent = [x_high+abs(half_pix), x_low-abs(half_pix), 
                y_low-abs(half_pix), y_high+abs(half_pix) ]
extent = tuple(img_extent)
fig, ax = plt.subplots()
img =ax.imshow(readout_image_array, extent = extent)
clb = plt.colorbar(img)

# for the left circle
# finding the best fit result of the circle through stepping through all the points inside the radius 
# and stepping through all the center position in a specific range
x0_guess_range = 3*full_pix # testing half of the range of the x center 
y0_guess_range = 3*full_pix # testing half of the range of the y center
r_max_l = int(0.012/full_pix) # first guess of maximum radius in number of pixels

result_l = numpy.zeros((r_max_l,x0_steps,y0_steps))
radius_l = numpy.linspace(0, r_max_l, r_max_l+1)
x0_list_l = numpy.linspace(x0_guess_l-x0_guess_range, x0_guess_l+x0_guess_range,x0_steps)
y0_list_l = numpy.linspace(y0_guess_l-y0_guess_range, y0_guess_l+y0_guess_range,y0_steps)

    
for x0_i in range (x0_steps):
    x0 = x0_list_l[x0_i]
    for y0_i in range (y0_steps):
        y0 = y0_list_l[y0_i]
        for ri in range(r_max_l):
            r = radius_l[ri]
            result_l[ri][x0_i][y0_i] = 0 
            count = 0
            for i in range (0,num_steps,1):
                for j in range (0,num_steps,1):
                    z_l = (x[i]-x0)**2+(y[j]-y0)**2
                    if z_l <= ((r+1/2)*full_pix)**2 and z_l >= ((r-1/2)*full_pix)**2 : 
                        count += 1
                        if readout_image_array[j][i] > k:
                            result_l[ri][x0_i][y0_i] += 1 
            if count == 0 :
                continue
            else:
                result_l[ri][x0_i][y0_i] = result_l[ri][x0_i][y0_i]/count

indexes = numpy.where(result_l == numpy.amax(result_l))
print ('indexes', indexes)
print ('Best fit result for left ring (x0, y0, radius, count):', x0_list_l[indexes[1]], y0_list_l[indexes[2]], radius_l[indexes[0]]*full_pix)#  
print (x0_list_l[indexes[1]][0], y0_list_l[indexes[2]][0], radius_l[indexes[0]][0]*full_pix) # keep for 2 NV centers
plt.plot (x0_list_l[indexes[1]], y0_list_l[indexes[2]],'c.')

# for best fit point 1, plot a circle fit
b_x1_l=x0_list_l[indexes[1]][0]
b_y1_l=y0_list_l[indexes[2]][0]
b_r1_l=radius_l[indexes[0]][0]*full_pix
BestPoint1_l = numpy.zeros((num_steps,num_steps))

angle_l = numpy.linspace(0,2*numpy.pi,180)
x_r_l = b_x1_l+b_r1_l*numpy.cos(angle_l)
y_r_l = b_y1_l+b_r1_l*numpy.sin(angle_l)
#plt.plot(x_r_l, y_r_l,'g')

# find the pixels that are larger than the threshold within some radial range
b3_x_l = []
b3_y_l = []
for i in range (0,num_steps,1):
    for j in range (0,num_steps,1):
        z_l = (x[i]-b_x1_l)**2+(y[j]-b_y1_l)**2
        if z_l <= (b_r1_l+away_b*full_pix)**2 and z_l >= (b_r1_l-away_b*full_pix)**2 :
            if readout_image_array[j][i] > k:
                b3_x_l.append(x[i])
                b3_y_l.append(y[j])

# the final least square fit plot for the left ring
center_estimate_l = b_x1_l, b_y1_l
fit_2_l = lambda c: f_2_l(c)
center_2_l, cov_x = optimize.leastsq(fit_2_l, center_estimate_l)
print (center_2_l)

xc_2_l, yc_2_l = center_2_l
Ri_2_l       = calc_R_l(xc_2_l, yc_2_l)
R_2_l        = Ri_2_l.mean()
residu_2_l   = sum((Ri_2_l - R_2_l)**2)
print ('Best left circle fit result from least square fit(x0, y0, radius, count):', xc_2_l, yc_2_l, R_2_l)

angle_l = numpy.linspace(0,2*numpy.pi,180)
x_r2_l = xc_2_l + R_2_l * numpy.cos(angle_l)
y_r2_l = yc_2_l + R_2_l * numpy.sin(angle_l)
plt.plot(x_r2_l, y_r2_l,'r')



# for the right ring
x0_guess_r = -0.023
y0_guess_r = 0.284
r_max_r = int(0.012/full_pix) # first guess of maximum radius in number of pixels
x0_guess_range = 3*full_pix # testing half of the range of the x center 
y0_guess_range = 3*full_pix # testing half of the range of the y center

result_r = numpy.zeros((r_max_r,x0_steps,y0_steps))
radius_r = numpy.linspace(0, r_max_r, r_max_r+1)
x0_list_r = numpy.linspace(x0_guess_r-x0_guess_range, x0_guess_r+x0_guess_range,x0_steps)
y0_list_r = numpy.linspace(y0_guess_r-y0_guess_range, y0_guess_r+y0_guess_range,y0_steps)
for x0_i in range (x0_steps):
    x0 = x0_list_r[x0_i]
    for y0_i in range (y0_steps):
        y0 = y0_list_r[y0_i]
        for ri in range(r_max_r):
            r = radius_r[ri]
            result_r[ri][x0_i][y0_i] = 0 
            count = 0
            for i in range (0,num_steps,1):
                for j in range (0,num_steps,1):
                    z_r = (x[i]-x0)**2+(y[j]-y0)**2
                    if z_r <= ((r+1/2)*full_pix)**2 and z_r >= ((r-1/2)*full_pix)**2 :
                        count += 1
                        if readout_image_array[j][i] > k:
                            result_r[ri][x0_i][y0_i] += 1 
            if count == 0 :
                continue
            else:
                result_r[ri][x0_i][y0_i] = result_r[ri][x0_i][y0_i]/count

indexes = numpy.where(result_r == numpy.amax(result_r))
print ('Best fit result for right ring (x0, y0, radius, count):', x0_list_r[indexes[1]], y0_list_r[indexes[2]], radius_r[indexes[0]]*full_pix)#  
print (x0_list_r[indexes[1]][0], y0_list_r[indexes[2]][0], radius_r[indexes[0]][0]*full_pix) # keep for 2 NV centers
plt.plot (x0_list_r[indexes[1]], y0_list_r[indexes[2]],'y.')

# for best fit point 1, plot a circle fit
b_x1_r=x0_list_r[indexes[1]][0]
b_y1_r=y0_list_r[indexes[2]][0]
b_r1_r=radius_r[indexes[0]][0]*full_pix
BestPoint1_R = numpy.zeros((num_steps,num_steps))

angle_r = numpy.linspace(0,2*numpy.pi,180)
x_r_r = b_x1_r+b_r1_r*numpy.cos(angle_r)
y_r_r = b_y1_r+b_r1_r*numpy.sin(angle_r)
#plt.plot(x_r_r, y_r_r,'r')

# find the pixels that are larger than the threshold within some radial range
b3_x_r = []
b3_y_r = []
for i in range (0,num_steps,1):
    for j in range (0,num_steps,1):
        z_r = (x[i]-b_x1_r)**2+(y[j]-b_y1_r)**2
        if z_r <= (b_r1_r+away_b*full_pix)**2 and z_r >= (b_r1_r-away_b*full_pix)**2 :
            if readout_image_array[j][i] > k:
                b3_x_r.append(x[i])
                b3_y_r.append(y[j])

# the final least square fit plot for the right ring
center_estimate_r = b_x1_r, b_y1_r
fit_2_r = lambda c: f_2_r(c)
center_2, cov_x_r = optimize.leastsq(fit_2_r, center_estimate_r)
print (center_2)

xc_2_r, yc_2_r = center_2
Ri_2_r       = calc_R_r(xc_2_r, yc_2_r)
R_2_r        = Ri_2_r.mean()
residu_2_r   = sum((Ri_2_r - R_2_r)**2)
print ('Best right circle fit result from least square fit(x0, y0, radius, count):', xc_2_r, yc_2_r, R_2_r)

angle_r = numpy.linspace(0,2*numpy.pi,180)
x_r2_r = xc_2_r + R_2_r * numpy.cos(angle_r)
y_r2_r = yc_2_r + R_2_r * numpy.sin(angle_r)
plt.plot(x_r2_r, y_r2_r,'y')



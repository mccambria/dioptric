# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:51:46 2019

@author: mccambria
"""

import cv2
import numpy
from matplotlib import pyplot as plt
import json

#directory = 'G:\\Team Drives\\Kolkowitz Lab Group\\nvdata\\image_sample\\'
directory = 'E:\\Team Drives\\Kolkowitz Lab Group\\nvdata\\image_sample\\'
#file_name = '2019-04-29_16-37-06_ayrton12.txt'
#file_name = '2019-04-29_16-37-56_ayrton12.txt'
file_name = '2019-04-29_16-19-11_ayrton12.txt'
file_path = directory + file_name

with open(file_path, 'r') as file:
    data = json.load(file)

img_array = numpy.array(data['img_array'])
readout = data['readout']

# Convert to kcps
img = (img_array / 1000) / (readout / 10**9)

# Convert to 8 bit
img -= numpy.nanmin(img)  # Set the lowest value to 0
img *= (255/numpy.nanmax(img))
img = img.astype(numpy.uint8)
contour_img = numpy.copy(img)
contour_img[:] = 0

# grad = 50
edges = cv2.Canny(img, 40, 50)

ret_vals = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = ret_vals
hierarchy = hierarchy[0]

# contour_img = cv2.drawContours(contour_img, contours, -1, (255,255,255), 1)

# Clean up the contour list
# Only keep a contour if it has no children (ie it is innermost) and
# has a parent (ie it is contained within another contour). findContours
# finds internal and external lines around closed loops in the image
# so this criteria selects only the internal lines. It can be fooled, but it
# should work for our edge-processed images.
contours_temp = []
for ind in range(len(contours)):
    hier = hierarchy[ind]
    cnt = contours[ind]
    child_cnt_id = hier[1]
    parent_cnt_id = hier[3]
    if (child_cnt_id == -1) and (parent_cnt_id != -1):
        contours_temp.append(cnt)
contours = contours_temp

# contour_img = cv2.drawContours(contour_img, contours, -1, (255,255,255), 1)

for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
#    contour_img = cv2.circle(E:\Team Drives\Kolkowitz Lab Group\nvdata\image_sample, center, radius, (255,255,255), 1)
    contour_img[center[1], center[0]] = 255

fig, axes_pack = plt.subplots(1, 3, figsize=(15, 5))
ax = axes_pack[0]
ax.imshow(img, cmap='gray')
ax.set_title('Original')
ax = axes_pack[1]
ax.imshow(edges, cmap='gray')
ax.set_title('Edges')
ax = axes_pack[2]
ax.imshow(contour_img, cmap='gray')
ax.set_title('Circle Fits')

fig.show()
fig.tight_layout()

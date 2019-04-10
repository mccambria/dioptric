# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:25:38 2019

@author: Aedan
"""

import matplotlib.pyplot as plt

#X, Y = np.meshgrid(x, y)
#Z = z.reshape(21, 21)

yScanRange = 0.4
yCenter = 0.0
yImgResolution = 3.0
yScanCenterPlusMinus = yScanRange / 2
yImgStepSize = yScanRange / yImgResolution
yMin = yCenter - yScanCenterPlusMinus

xScanRange = 0.4
xCenter = -0.108
xImgResolution = 3.0
xScanCenterPlusMinus = xScanRange / 2
xImgStepSize = xScanRange / xImgResolution
xMin = xCenter - xScanCenterPlusMinus

Y = []
Y.append(yMin)
i = 1
while i < (yImgResolution + 1):
    yNextPoint = Y[i-1] + yImgStepSize
    Y.append(yNextPoint)
    i += 1
    


X = []
X.append(xMin)
i = 1
while i < (xImgResolution + 1):
    xNextPoint = X[i-1] + xImgStepSize
    X.append(xNextPoint)
    i += 1

print(X, Y)
    

plt.pcolor(X,Y,[[0, 2, 2], [0, 1, 3], [2, 1, 0]])
plt.show()
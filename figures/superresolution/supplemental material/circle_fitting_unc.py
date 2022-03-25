# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:53:17 2022

@author: agard
"""

import numpy

scale = 34.5e3

#Fig 3
# px_v = 0.035/80

# x1 = 36.9
# dx1 = 1.4
# y1 = 41.7
# dy1=1.5

# x2 = 43.9
# dx2=0.9
# y2=39.1
# dy2=1.3

#Fig 4
px_v = 0.05/100
x1 =45.8
dx1 = 1
y1 = 51.0
dy1=1.4

x2 = 56.3
dx2=1.1
y2=51.2
dy2=1.4




#calculate the distance between these two points in x and y
X = abs(x1 - x2)
Y = abs(y1 - y2)

#what are the propegated uncertainties on these distances? we need to add
# uncertanties of the x and y values in quadrature
DX = numpy.sqrt(dx1**2 + dx2**2)
DY = numpy.sqrt(dy1**2 + dy2**2)

# then we need the radial distance between the two. 
R = numpy.sqrt(X**2 + Y**2)
# The uncertianity is a bit more tricky. Let's instead consider the euqation
# R^2 = X^2 + Y^2, which is just simple addition. We cna find the uncertainties 
#of the values of X^2 and Y^2, and then add them in quad.

# So let's find the uncertainties of X^2 and Y^2 with the rule for squaring values:
# X = x^n, dX = |X|n dx/|x|
XS = X**2
DXS = XS*2*DX/X 
YS = Y**2
DYS = YS*2*DY/Y


# then we need to add these squared values together: R^2 = X^2 + Y^2
RS = XS + YS
DRS = numpy.sqrt(DXS**2 + DYS**2)

#finally, we want R, not R^2, so we take the square root of the value, which 
# turns out to follow the following error propegation rule: X = x^n, dX = |X|n dx/|x|
# Rp = numpy.sqrt(RS)
DR = R * 1/2 * DRS /RS

print(R*px_v*scale)
print(DR*px_v*scale)
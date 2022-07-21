# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:53:17 2022

@author: agard
"""

import numpy

scale = 34.5e3

#Fig 3
num_steps_3 = 81
px_v_3 = 0.035/(num_steps_3-1)

xa = 36.83
dxa = 1.36
ya = 41.73
dya=1.59
ra = 27.72
dra = 1.14
costA=0.42379

xb = 43.9
dxb=0.74
yb=39.1
dyb=1.08
rb = 27.64
drb = 0.83
costB=0.44067

#Fig 4
num_steps_4 = 101
px_v_4 = 0.05/(num_steps_4-1)
xc = 45.64
dxc= 1.26
yc = 50.64
dyc= 1.42
rc = 26.29
drc = 1.46
cost_C = 0.43131

xd = 56.15
dxd=1.11
yd=51.02
dyd=1.58
rd = 27.48
drd = 1.20
costD=0.45009

def convert_position(pos_px, pos_px_unc, num_steps, px_v, x_or_y ):
    # because of the flipped y axis, we need to subtract the center point of
    # the array differently for x vs y
    if x_or_y == 'x':
        dif_pos_px = (pos_px - (num_steps-1)/2)
    elif x_or_y == 'y':
        dif_pos_px = ((num_steps-1)/2 - pos_px)
        
    pos_nm = dif_pos_px*px_v*scale
    
    pos_nm_unc = pos_px_unc* px_v*scale
    
    return pos_nm, pos_nm_unc


def convert_distance(dist_px, dist_px_unc, px_v):
    dist_nm = dist_px * px_v * scale
    
    dist_nm_unc = dist_nm * dist_px_unc / dist_px
    return dist_nm, dist_nm_unc


def calc_seperation(x1, x2, y1, y2, dx1, dx2, dy1,dy2, px_v, scale):
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
    
    sep_nm = R*px_v*scale
    sep_nm_unc = DR*px_v*scale
    
    return sep_nm, sep_nm_unc

# calculate
val, unc = convert_position(xa, dxa, num_steps_3, px_v_3, 'x' )
print('xA = {:.4} +/- {:.4}'.format(val, unc))
val, unc = convert_position(ya, dya, num_steps_3, px_v_3, 'y' )
print('yA = {:.4} +/- {:.4}'.format(val, unc))
val, unc = convert_distance(ra, dra, px_v_3)
print('RA = {:.4} +/- {:.4}'.format(val, unc))
val, unc = convert_position(xb, dxb, num_steps_3, px_v_3, 'x' )
print('xB = {:.4} +/- {:.4}'.format(val, unc))
val, unc = convert_position(yb, dyb, num_steps_3, px_v_3, 'y' )
print('yB = {:.4} +/- {:.4}'.format(val, unc))
val, unc = convert_distance(rb, drb, px_v_3)
print('RB = {:.4} +/- {:.4}'.format(val, unc))

val, unc= calc_seperation(xa, xb, ya, yb, dxa, dxb, dya,dyb, px_v_3, scale)
print('Dist between NVA and NVB = {:.4} +/- {:.4}'.format(val, unc))

val, unc = convert_position(xc, dxc, num_steps_4, px_v_4, 'x' )
print('xC = {:.4} +/- {:.4}'.format(val, unc))
val, unc = convert_position(yc, dyc, num_steps_4, px_v_4, 'y' )
print('yC = {:.4} +/- {:.4}'.format(val, unc))
val, unc = convert_distance(rc, drc, px_v_4)
print('RC = {:.4} +/- {:.4}'.format(val, unc))
val, unc = convert_position(xd, dxd, num_steps_4, px_v_4, 'x' )
print('xD = {:.4} +/- {:.4}'.format(val, unc))
val, unc = convert_position(yd, dyd, num_steps_4, px_v_4, 'y' )
print('yD = {:.4} +/- {:.4}'.format(val, unc))
val, unc = convert_distance(rd, drd, px_v_4)
print('RD = {:.4} +/- {:.4}'.format(val, unc))

val, unc= calc_seperation(xc, xd, yc, yd, dxc, dxd, dyc,dyd, px_v_4, scale)
print('Dist between NVC and NVD = {:.4} +/- {:.4}'.format(val, unc))
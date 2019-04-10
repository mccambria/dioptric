# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:34:43 2019

@author: Aedan
"""

import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit

splitting = (0.058, 0., 0.002, 0.046, 0.086, 0.125, 0.153, 0.172, 0.179) #GHz, not angular frequency

angle = (0., 15.*numpy.pi / 180, 20.*numpy.pi / 180, 30.*numpy.pi / 180, 45.*numpy.pi / 180, 
         60.*numpy.pi / 180, 75.*numpy.pi / 180, 90.*numpy.pi / 180, 105.*numpy.pi / 180) #radians

bohrMagneton = 9.27 * 10**(-24) #J/T

h = 6.63 * 10**(-34.) #J s

g = 2 #dimensionless

ampl = 2 * g * bohrMagneton / h

theta = numpy.arange(0, 105 * numpy.pi / 180, 0.01)

def test_func(x, B, a, phase):
    return  B * numpy.absolute(numpy.cos(a * x + phase))

params, params_covariance = curve_fit(test_func, angle, splitting,
                                               p0=[0.1, 2., -3.])

print(params)

magneticField = params[0] * 10**9 / ampl

print(magneticField) #in Teslas

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(angle, splitting,'bo')
ax.plot(theta, test_func(theta, params[0], params[1], params[2]), 'r')
ax.set_xlabel('Angle Orientation of Magnet (radian)')
ax.set_ylabel('Splitting of spin levels (GHz)')
ax.set_title('NV splitting dependance on magnet orientation')
text = "\n".join(("amplitude=" + "%.4f"%(params[0]) + ' GHz',
                      "phase=" + "%.3f"%(params[2]) + ' radian'))

text1 = ("Magnetic Field Magnitude at NV: " + "%.3f"%(magneticField * 10**4) + ' G')

props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)
ax.text(0.5, 0.1, text1, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)

fig.canvas.draw()
fig.canvas.flush_events()
#
fig.savefig('2019-03-21_ESR_MagneticFieldAngleDependance.png')

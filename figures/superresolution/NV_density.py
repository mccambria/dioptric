# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:07:47 2022

@author: kolkowitz
"""
import numpy

n = 2.4 # refractive index of diamond
NA = 1.3 #objective NA
w = 380/2 #1/e2 beam radius in the xy plane

D = 1.76e11 #atoms/um3, atomic density in diamond
NV = 4 # number of NVs in volume

# confocal volume
kappa = 2.33*n/NA
V = (numpy.pi)**(3/2) * kappa * w**3 /1e9
# print(kappa)

print('Confocal volume: {} um^3'.format(V))

# NV density

num_atoms = D * V
NV_D = NV / num_atoms

print('NV density: {} ppb'.format(NV_D*1e9))
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:08:46 2019

@author: Aedan
"""

from scipy.stats import chisquare
import numpy

O = [16, 18, 16, 14, 12, 12]
E = [16, 16, 16, 16, 16, 8]

print(chisquare(O, f_exp=E))

chi_sq_list = []
for el in range(len(O)):
    chi_sq_el = (O[el] - E[el])**2 / E[el]
    chi_sq_list.append(chi_sq_el)
    
print(numpy.sum(chi_sq_list))
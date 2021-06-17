# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:19:38 2021

@author: kolkowitz
"""

import matplotlib.pyplot as plt

x = [0.5,1,1.5,2,2.5,3,4]
y = [0.0063, 0.0074, 0.0089, 0.011, 0.017, 0.09, 0.43]

fig, ax = plt.subplots()
ax.plot(x,y)
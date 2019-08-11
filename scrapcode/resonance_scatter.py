# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:33:37 2019

@author: kolkowitz
"""

import matplotlib.pyplot as plt

res_low = [2.865, 2.8265, 2.8435, 2.7726]
res_high = [2.865, 2.9117, 2.8890, 3.0530]

fig, ax = plt.subplots(figsize=(8.5, 8.5))

ax.scatter(res_high, res_low)

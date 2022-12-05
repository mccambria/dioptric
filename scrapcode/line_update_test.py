# -*- coding: utf-8 -*-
"""
Plot update test

Created on December 5th, 2022

@author: mccambria
"""

import time
import utils.kplotlib as kpl
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 9, 10)
y1 = np.empty(10)
y1[:] = np.nan
y2 = np.empty(10)
y2[:] = np.nan

kpl.init_kplotlib()
fig, axes_pack = plt.subplots(1, 2, figsize=kpl.double_figsize)
ax1, ax2 = axes_pack

kpl.plot_line(ax1, x, y1)
kpl.plot_line(ax2, x, y2)
kpl.tight_layout(fig)
# plt.show()

for ind in range(10):
    time.sleep(1)
    y1_new = np.random.rand(10)
    kpl.plot_line_update(ax1, y=y1_new)
    y2_new = np.random.rand(10)
    kpl.plot_line_update(ax2, y=y2_new)

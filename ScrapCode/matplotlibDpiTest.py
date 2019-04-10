# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 03:16:31 2019

@author: mccambria
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Utils.tool_belt as tool_belt

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots(figsize=(10, 8.5))
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

# plt.show()
# fig.canvas.flush_events()

# timeStamp = tool_belt.get_time_stamp()
# filePath = tool_belt.get_file_path("test2", timeStamp, "MCC")
# tool_belt.save_figure(fig, filePath)

# -*- coding: utf-8 -*-
"""


Created on Tue Feb 26 23:50:19 2019

@author: mccambria
"""

import time
import msvcrt

start  = time.time()

print("Press enter to stop...")
while time.time() - start < 10:
    if msvcrt.kbhit():
        print(msvcrt.getch())
        break

print("DONE")
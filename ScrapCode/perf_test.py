# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:50:43 2019

@author: kolkowitz
"""

import time

start = time.time()
loop = 0
while loop < 1000000:
    loop += 1
    
end = time.time()
print(end-start)
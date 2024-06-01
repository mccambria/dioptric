#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:41:33 2024

@author: sean
"""

import fieldcontrol as fc
instr = fc.initialize()
print(fc.allCurrent(instr))
print(fc.calculate([1,2,3]))
fc.allCurrent(instr,fc.calculate([1,2,3]))
print(fc.allCurrent(instr))

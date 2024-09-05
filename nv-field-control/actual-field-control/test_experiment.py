#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:41:33 2024

@author: sean
"""

import fieldcontrol as fc

a = fc.RS_NGC103(start_open = True)

a.set_current(1, 2)
a.activateChannel(1)
a.activateMaster()

print(a.get_current(1))
print(a.get_current(3))

a.close_connection()
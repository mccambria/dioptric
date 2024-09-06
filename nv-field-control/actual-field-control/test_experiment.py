#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:41:33 2024

@author: sean
"""

import fieldcontrol as fc

a = fc.RS_NGC103(IP='128.32.239.90',start_open = True)

a.activateChannel(1)
a.activateMaster()

a.maintain_current(1, 1)

# a.deactivateMaster()
# a.deactivateAll()

a.close_connection()
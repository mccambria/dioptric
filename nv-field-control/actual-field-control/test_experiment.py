#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:41:33 2024

@author: sean, quinn
"""

import time

import fieldcontrol as fc

a = fc.RS_NGC103(IP='128.32.239.90', start_open = True)

a.activateMaster()
a.hold_current(1, 1.5, activate_channel=True)
# a.hold_current(2, 1, activate_channel=True)
# a.hold_current(3, 0.5, activate_channel=True)

time.sleep(3)

a.hold_current(3, 1,5)

# a._write_command("INST OUT1")
# a._write_command("OUTP:CHAN ON")

time.sleep(5)

# a.deactivateChannel(1)

a.release_current(1, deactivate_channel=True)
# a.release_current(2, deactivate_channel=True)
# a.release_current(3, deactivate_channel=True)

time.sleep(1)

a.deactivateMaster()
a.close_connection()
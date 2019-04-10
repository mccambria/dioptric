# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:10:27 2019

@author: mccambria
"""

import Utils.tool_belt as tool_belt
PULSE_STREAMER_IP = "128.104.160.11"
PULSER_DO_RF = 4

tool_belt.pulser_high(PULSE_STREAMER_IP, [PULSER_DO_RF])
input("Press enter to stop...")

tool_belt.pulser_all_zero(PULSE_STREAMER_IP)

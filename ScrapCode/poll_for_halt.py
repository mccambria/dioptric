# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 09:43:17 2019

@author: Matt
"""

import time
import Utils.tool_belt as tool_belt

start = time.time()

tool_belt.poll_safe_stop()

tool_belt.init_safe_stop()
tool_belt.init_safe_stop()

print("Looping")
loop = 0

while True:
    time.sleep(0.1)
    loop += 1
    print(loop)
    if tool_belt.safe_stop():
        print("SAFESTOP")
        break
    if loop > 30:
        print("failure")
        break

if tool_belt.check_safe_stop_alive():
    tool_belt.poll_safe_stop()

tool_belt.poll_safe_stop()

tool_belt.init_safe_stop()
tool_belt.init_safe_stop()

print("Looping")
loop = 0

while True:
    time.sleep(0.1)
    loop += 1
    print(loop)
    if tool_belt.safe_stop():
        print("SAFESTOP")
        break
    if loop > 30:
        print("failure")
        break

if tool_belt.check_safe_stop_alive():
    tool_belt.poll_safe_stop()

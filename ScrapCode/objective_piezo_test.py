# -*- coding: utf-8 -*-
"""
objective_piezo test

Created on Wed Mar 20 13:26:19 2019

@author: mccambria
"""

import Outputs.objective_piezo as objective_piezo

piezoSerial = '119008970'

#objective_piezo.write_single_open_loop(piezoSerial, 50.0)
print(objective_piezo.read_position(piezoSerial))

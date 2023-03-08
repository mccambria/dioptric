# -*- coding: utf-8 -*-
"""Presentation round unit tests

Created on March 7th, 2023

@author: mccambria
"""

import utils.tool_belt as tool_belt

vals = []
errs = []

expected_results = []

actual_results = [
    tool_belt.presentation_round(val, err) for val, err in zip(vals, errs)
]
test = [exp == act for exp, act in zip(expected_results, actual_results)]

print(test)

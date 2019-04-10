# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:00:46 2019

@author: Aedan
"""

import json

with open('2019-03-09_13-08-08_ayrton4.txt') as json_file:
    data = json.load(json_file)
    for i in data['timestamp']:
        print('Timestamp: ' + i['timestamp'])
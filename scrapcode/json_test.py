# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:18:43 2019

@author: Matt
"""

import json
import numpy

data = {'test': 'hello',
        'data': 155511,
        'arraty': [[1,2,3,4],
                   [6,5,numpy.nan,3]]}

with open('test.txt', 'w') as file:
    last_key = data.keys()[-1]
    file.write('{\n')
    for key in data:
        if key is last_key:
            write_string =
        file.write('  \"{}\": {},\n'.format(key, json.dumps(data[key])))
    file.write('}')

with open('test1.txt', 'w') as file:
    json.dump(data, file, indent=2)
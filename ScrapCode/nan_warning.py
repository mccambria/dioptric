# -*- coding: utf-8 -*-
"""


Created on Thu Mar  7 08:50:06 2019

@author: mccambria
"""

import numpy

nan = numpy.nan

test = numpy.array([[3,nan,nan,nan],
                    [nan,nan,43,nan]])

print(numpy.all(test))
print(numpy.all(numpy.isnan(test)))

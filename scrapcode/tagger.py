# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:45:54 2019

@author: kolkowitz
"""

import TimeTagger
import time

print('getting tagger')
tagger = TimeTagger.createTimeTagger('1740000JEH')
print('resetting tagger')
tagger.reset()
print('collecting data')

apd_chans = [1, 2]

buffer_size = int(10**6 / len(apd_chans))  # A million total
stream = TimeTagger.TimeTagStream(tagger, buffer_size, apd_chans)

time.sleep(2.0)

buffer = stream.getData().getTimestamps()
test = 1

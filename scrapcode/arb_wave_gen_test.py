# -*- coding: utf-8 -*-
"""
Arbitrary waveform generator testing.

Created on Fri Jun 14 14:20:09 2019

@author: mccambria
"""


import labrad


def main(cxn):
    cxn.arbitrary_waveform_generator.test_sin()
    input('Press enter to stop...')
    cxn.arbitrary_waveform_generator.wave_off()


if __name__ == '__main__':
    with labrad.connect() as cxn:
        main(cxn)

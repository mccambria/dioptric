# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:33:25 2019

@author: Matt
"""

class Test():

    serial = 'MCCTEST'

    def run_test(self):
        return self.serial

    def test(self):
        return self.run_test()

if __name__ == '__main__':
    this_test = Test()
    print(this_test.test())
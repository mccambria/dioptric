# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:07:43 2019

@author: mccambria
"""

class hello():
    
    def close_task_0(self):
        print('hello')
    
    def run_close_method(self, apd_index):
        foo = getattr(self, 'close_task_{}'.format(apd_index))
        foo()
        
hi = hello()

hi.run_close_method(0)

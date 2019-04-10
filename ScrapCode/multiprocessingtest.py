# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:29:24 2019

@author: kolkowitz
"""

from multiprocessing import Process
import time
import sys

def f(name):
    time.sleep(5.0)
    print('hello', name)
    sys.stdout.flush()

if __name__ == '__main__':
    print("start")
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
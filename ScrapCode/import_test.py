# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:51:05 2019

@author: Matt
"""

import os
import importlib

seq_lib_dir = 'servers.timing.sequencelib.{}'

def get_seq(seq_file):
    seq = None
    file_name, file_ext = os.path.splitext(seq_file)
    if file_ext == '.py':  # py: import as a module
        module_path = seq_lib_dir.format(file_name)
        seq_module = importlib.import_module(module_path)
        seq = seq_module.get_seq()
    return seq

seq = get_seq('simple_readout.py')
print(seq)

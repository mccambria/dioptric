# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:53:06 2021

@author: kolkowitz
"""

import labrad

with labrad.connect() as cxn:
    
    cxn.z_piezo_kpz101.write_z(5.0)
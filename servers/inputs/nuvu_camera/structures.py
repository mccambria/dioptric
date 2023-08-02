# -*- coding: utf-8 -*-
"""
C type defintions from Nuvu's SDK

Obtained on August 1st, 2023

@author: Nuvu
"""

from ctypes import Structure, POINTER


class NCCAMHANDLE(Structure):
    pass


NCCAM = POINTER(NCCAMHANDLE)


class NCIMAGEHANDLE(Structure):
    pass


NCIMAGE = POINTER(NCIMAGEHANDLE)

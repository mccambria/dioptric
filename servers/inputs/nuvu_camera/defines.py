# -*- coding: utf-8 -*-
"""
Constants from Nuvu's SDK

Obtained on August 1st, 2023

@author: Nuvu
"""
from enum import Enum, auto

NC_AUTO_DETECT = 0x0000FFFF
NC_AUTO_CHANNEL = NC_AUTO_DETECT
NC_AUTO_UNIT = 0x6FFFFFFF
NC_FULL_WIDTH = -1
NC_FULL_HEIGHT = -1
NC_USE_MAC_ADRESS = 0x20000000


class ShutterMode(Enum):
    SHUT_NOT_SET = 0
    OPEN = 1
    CLOSE = 2
    AUTO = 3
    BIAS_DEFAULT = CLOSE


class ReadoutMode(Enum):
    NONE = 0
    EM = 1
    CONV = 2


class TriggerMode(Enum):
    CONT_HIGH_LOW = -3
    EXT_HIGH_LOW_EXP = -2
    EXT_HIGH_LOW = -1
    INTERNAL = 0
    EXT_LOW_HIGH = 1
    EXT_LOW_HIGH_EXP = 2
    CONT_LOW_HIGH = 3


class ProcessingType(Enum):
    NO_PROCESSING = 0x00
    BIAS_SUBTRACTION = 0x01
    PHOTON_COUNTING = 0x02

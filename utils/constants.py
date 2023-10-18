# -*- coding: utf-8 -*-
"""Enums and other constants. Should not import anything other user modules
or else we will probably get a circular import

Created on June 26th, 2023

@author: mccambria
"""

from enum import Enum, IntEnum, auto


class CollectionMode(Enum):
    COUNTER = auto()  # Count all photons incident on a detector (e.g. PMT or APD)
    CAMERA = auto()  # Collect photons onto a camera


class CountFormat(Enum):
    KCPS = auto()  # Count rate in kilo counts per second
    RAW = auto()  # Just the raw number of counts


class LaserKey(Enum):
    IMAGING = auto()
    SPIN = auto()


class ControlStyle(Enum):
    """
    STEP: Manual control with individual function calls
    STREAM: A stream of values can be loaded onto the controller - the controller will step
        through the stream automatically in response to a clock signal from the pulse generator
    SEQUENCE: Controlled  directly from the pulse generator sequence
    """

    STEP = auto()
    STREAM = auto()
    SEQUENCE = auto()


class NVSpinState(Enum):
    LOW = auto()
    ZERO = auto()
    HIGH = auto()


# Normalization style for comparing experimental data to reference data
class NormStyle(Enum):
    SINGLE_VALUED = auto()  # Use a single-valued reference
    POINT_TO_POINT = auto()  # Normalize each signal point by its own reference


class ModTypes(Enum):
    DIGITAL = auto()
    ANALOG = auto()


class Digital(IntEnum):
    LOW = 0
    HIGH = 1


Boltzmann = 8.617e-2  # meV / K

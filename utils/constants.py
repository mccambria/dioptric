# -*- coding: utf-8 -*-
"""Enums, dataclasses other constants. Should not import anything other user modules
or else we will probably get a circular import

Created on June 26th, 2023

@author: mccambria
"""

from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from numbers import Number

from strenum import StrEnum


@dataclass
class NVSig:
    name: str = None
    coords: dict | list = None
    representative: bool = False
    disable_opt: bool = False
    disable_z_opt: bool = False
    threshold: Number = None
    expected_counts: Number = None
    magnet_angle: Number = None
    opti_offset: list[Number] = None  # Only works for global coordinates


class CoordsKey(StrEnum):
    GLOBAL = "global"
    PIXEL = "pixel"


class CollectionMode(Enum):
    COUNTER = auto()  # Count all photons incident on a detector (e.g. PMT or APD)
    CAMERA = auto()  # Collect photons onto a camera


class CountFormat(Enum):
    KCPS = auto()  # Count rate in kilo counts per second
    RAW = auto()  # Just the raw number of counts


class LaserKey(Enum):
    IMAGING = auto()  # Basic imaging
    WIDEFIELD_IMAGING = auto()
    IONIZATION = auto()
    POLARIZATION = auto()  # Charge / spin state polarization
    SPIN_READOUT = auto()  # Standard spin readout
    CHARGE_READOUT = auto()  # Readout of the charge state


class LaserPosMode(Enum):
    SCANNING = auto()
    WIDEFIELD = auto()


class ControlMode(Enum):
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


# Normalization mode for comparing experimental data to reference data
class NormMode(Enum):
    SINGLE_VALUED = auto()  # Use a single-valued reference
    POINT_TO_POINT = auto()  # Normalize each signal point by its own reference


class ModMode(Enum):
    DIGITAL = auto()
    ANALOG = auto()


class Digital(IntEnum):
    LOW = 0
    HIGH = 1


# Ionization pulse type
class IonPulseType(IntEnum):
    SCC = 0
    ION = 1


Boltzmann = 8.617e-2  # meV / K

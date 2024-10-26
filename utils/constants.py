# -*- coding: utf-8 -*-
"""Enums, dataclasses, other constants. Should not import anything other user modules
or else we will probably get a circular import

Created on June 26th, 2023

@author: mccambria
"""

from dataclasses import dataclass
from enum import Enum, IntEnum, auto

from strenum import StrEnum

number = int | float


@dataclass
class NVSig:
    name: str | None = None
    coords: dict | list | None = None
    representative: bool = False
    disable_opt: bool = False
    disable_z_opt: bool = False
    threshold: number | None = None
    expected_counts: number | None = None
    magnet_angle: number | None = None
    opti_offset: list[number] | None = None  # Only works for global coordinates
    # spin_flip: If True, an additional pi pulse will be applied to the NV at
    # the end of a spin experiment prior to readout. Useful for anticorrelations
    # and rejecting common mode noise
    spin_flip: bool = False
    scc_duration: int | None = None
    scc_amp: float | None = None
    nvn_dist_params: tuple | None = None  # bg, amp, sigma


class CollectionMode(Enum):
    COUNTER = auto()  # Count all photons incident on a detector (e.g. APD)
    CAMERA = auto()  # Collect photons onto a camera


class ChargeStateEstimationMode(Enum):
    THRESHOLDING = auto()
    MLE = auto()  # Maximum likelihood estimator for images


class CountFormat(Enum):
    """Deprecated, everything should be raw"""

    KCPS = auto()  # Count rate in kilo counts per second
    RAW = auto()  # Just the raw number of counts


# Laser keys describe virtual lasers, which accomplish one and only one function and
# must be associated with a physical laser in config
class LaserKey(Enum):
    # Scanning virtual lasers
    IMAGING = auto()
    ION = auto()
    SCC = auto()
    CHARGE_POL = auto()
    SPIN_POL = auto()
    SHELVING = auto()
    SPIN_READOUT = auto()  # Standard spin readout
    # Widefield virtual lasers
    WIDEFIELD_SHELVING = auto()
    WIDEFIELD_IMAGING = auto()
    WIDEFIELD_CHARGE_READOUT = auto()
    WIDEFIELD_SPIN_POL = auto()


class PosControlMode(Enum):
    """
    Different ways to control a positioner
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


Boltzmann = 8.617e-2  # meV / K

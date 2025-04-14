import os
import sys
import time
import traceback
from datetime import datetime
from utils.tool_belt import curve_fit

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import font_manager as fm
from matplotlib import rcParams

from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey


def find_optimal_value_geom_mean(step_vals, prep_fidelity, readout_fidelity, goodness_of_fit, weights=(1, 1, 1)):
        """
    Finds the optimal step value using a weighted geometric mean of fidelities and goodness of fit.

    Args:
        step_vals: Array of step values.
        prep_fidelity: Array of preparation fidelities.
        readout_fidelity: Array of readout fidelities.
        goodness_of_fit: Array of goodness-of-fit values (to minimize).
        weights: Tuple of weights (w1, w2, w3) for the metrics.

    Returns:
        optimal_step_val: Step value corresponding to the maximum combined score.
        optimal_prep_fidelity: Preparation fidelity at the optimal step.
        optimal_readout_fidelity: Readout fidelity at the optimal step.
        max_combined_score: Maximum combined score.
    """
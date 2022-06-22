# -*- coding: utf-8 -*-
"""
This file contains standardized functions intended to simplify the creation of plots for publications in a consistent style.

Created on June 22nd, 2022

@author: mccambria
"""


# region Imports


import matplotlib
import matplotlib.pyplot as plt
import threading
import os
import csv
import datetime
import numpy as np
from numpy import exp
import json
import time
import labrad
from tkinter import Tk
from tkinter import filedialog
from git import Repo
from pathlib import Path, PurePath
from enum import Enum, auto
import socket
import smtplib
from email.mime.text import MIMEText
import traceback
import keyring
import math
import utils.common as common
from colorutils import Color

# endregion


# region Colors
# The default color specification is hex, eg "#bcbd22"


kpl_colors = {
    # The following are taken from matplotlib's excellent default palette
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "yellow": "#bcbd22",
    "cyan": "#17becf",
    #
    # The following are good for background lines on plots to mark interesting points
    "gray": "#C0C0C0",
    # If marking an interesting point with a confidence interval, use dark_gray for the main line and light_gray for the interval
    "dark_gray": "#909090",
    "light_gray": "#DCDCDC",
}


def color_mpl_to_color_hex(color_mpl):

    # Trim the alpha value and convert from 0:1 to 0:255
    color_rgb = [255 * val for val in color_mpl[0:3]]
    color_Color = Color(tuple(color_rgb))
    color_hex = color_Color.hex
    return color_hex


def lighten_color_hex(color_hex, saturation_factor=0.3, value_factor=1.2):

    color_Color = Color(hex=color_hex)
    color_hsv = color_Color.hsv
    lightened_hsv = (
        color_hsv[0],
        saturation_factor * color_hsv[1],
        value_factor * color_hsv[2],
    )
    # Threshold to make sure these are valid colors
    lightened_hsv = (
        lightened_hsv[0],
        zero_to_one_threshold(lightened_hsv[1]),
        zero_to_one_threshold(lightened_hsv[2]),
    )
    lightened_Color = Color(hsv=lightened_hsv)
    lightened_hex = lightened_Color.hex
    return lightened_hex


def zero_to_one_threshold(val):
    if val < 0:
        return 0
    elif val > 1:
        return 1
    else:
        return val


# endregion

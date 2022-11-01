# -*- coding: utf-8 -*-
"""
This file contains standardized functions intended to simplify the
creation of plots for publications in a consistent style.

Created on June 22nd, 2022

@author: mccambria
"""


# region Imports


import matplotlib.pyplot as plt
from enum import Enum
from colorutils import Color

# endregion

# region Constants
# These standard values are intended for single-column figures

marker_sizes = {"normal": 7, "small": 6}
line_widths = {"normal": 1.5, "small": 1.25}
marker_edge_widths = line_widths.copy()
font_sizes = {"normal": 17, "small": 13}
figsize = [6.5, 5.0]
double_figsize = [figsize[0] * 2, figsize[1]]
line_style = "solid"
marker_style = "o"

# Default sizes here
marker_size = marker_sizes["normal"]
marker_size_inset = marker_sizes["small"]
line_width = line_widths["normal"]
line_width_inset = line_widths["small"]
marker_edge_width = marker_edge_widths["normal"]
marker_edge_width_inset = marker_edge_widths["small"]
default_font_size = "normal"
default_data_size = "normal"

# endregion


# region Colors
# The default color specification is hex, eg "#bcbd22"


class KplColors(Enum):
    BLUE = "#1f77b4"
    ORANGE = "#ff7f0e"
    GREEN = "#2ca02c"
    RED = "#d62728"
    PURPLE = "#9467bd"
    BROWN = "#8c564b"
    PINK = "#e377c2"
    GRAY = "#7f7f7f"
    YELLOW = "#bcbd22"
    CYAN = "#17becf"
    #
    # The following are good for background lines on plots to mark interesting points
    MEDIUM_GRAY = "#C0C0C0"
    # If marking an interesting point with a confidence interval, use dark_gray for the main line and light_gray for the interval
    DARK_GRAY = "#909090"
    LIGHT_GRAY = "#DCDCDC"

data_color_cycler = [
        KplColors.BLUE.value,
        KplColors.RED.value,
        KplColors.GREEN.value,
        KplColors.ORANGE.value,
        KplColors.PURPLE.value,
        KplColors.BROWN.value,
        KplColors.PINK.value,
        KplColors.GRAY.value,
        KplColors.YELLOW.value,
        KplColors.CYAN.value,
    ]
line_color_cycler = data_color_cycler.copy()


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


def init_kplotlib(font_size="normal", data_size="normal"):
    """
    Runs the default initialization for kplotlib, our default configuration
    of matplotlib
    """

    global active_axes, color_cyclers, default_font_size, default_data_size
    active_axes = []
    color_cyclers = []
    default_font_size = font_size
    default_data_size = data_size

    # Interactive mode so plots update as soon as the event loop runs
    plt.ion()

    ####### Latex setup #######

    preamble = r""
    preamble += r"\newcommand\hmmax{0} \newcommand\bmmax{0}"
    preamble += r"\usepackage{physics} \usepackage{upgreek}"

    # Fonts
    # preamble += r"\usepackage{roboto}"  # Google's free Helvetica
    preamble += r"\usepackage{helvet}"
    # Latin mdoern is default math font but let's be safe
    preamble += r"\usepackage{lmodern}"

    # Sans serif math font, looks better for axis numbers.
    # We preserve \mathrm and \mathit commands so you can still use
    # the serif lmodern font for variables, equations, etc
    preamble += r"\usepackage[mathrmOrig, mathitOrig, helvet]{sfmath}"

    plt.rcParams["text.latex.preamble"] = preamble

    ###########################

    # plt.rcParams["savefig.format"] = "svg"

    plt.rcParams["font.size"] = font_sizes[default_font_size]
    plt.rcParams['figure.figsize'] = figsize

    plt.rc("text", usetex=True)

def tight_layout(fig):

    fig.tight_layout(pad=0.3)

def get_default_color(ax, plot_type):
    """plot_type is data or line"""

    global active_axes, color_cyclers
    if ax not in active_axes:
        active_axes.append(ax)
        color_cyclers.append({"points": data_color_cycler.copy(), "line": line_color_cycler.copy()})
    ax_ind = active_axes.index(ax)
    cycler = color_cyclers[ax_ind][plot_type]
    color = cycler.pop(0)
    return color

def plot_points(ax, x, y, size=None, **kwargs):
    """
    Same as matplotlib's errorbar, but with our defaults. Use for plotting
    data points.
    """

    global default_data_size
    if size is None:
        size = default_data_size

    # Color handling
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = get_default_color(ax, "points")
    if "facecolor" in kwargs:
        face_color = kwargs["markerfacecolor"]
    else:
        face_color = lighten_color_hex(color)

    # Defaults
    params = {
        "linestyle": "none",
        "marker": marker_style,
        "markersize": marker_sizes[size],
        "markeredgewidth": marker_edge_widths[size],
    }

    # Combine passed args and defaults
    params = {**params, **kwargs}
    params["color"] = color
    params["markerfacecolor"] = face_color

    ax.errorbar(x, y, **params)

def plot_line(ax, x, y, size=None, **kwargs):
    """
    Same as matplotlib's plot, but with our defaults. Use for plotting
    continuous lines.
    """

    global default_data_size
    if size is None:
        size = default_data_size

    # Color handling
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = get_default_color(ax, "line")

    # Defaults
    params = {
        "linestyle": line_style,
        "linewidth": line_widths[size],
    }

    # Combine passed args and defaults
    params = {**params, **kwargs}
    params["color"] = color

    ax.plot(x, y, **params)

def text(ax, x, y, text, size=None, **kwargs):
    """x, y are relative to plot dimensions and start from lower left corner"""

    global default_font_size
    if size is None:
        size = default_font_size

    bbox_props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    font_size = font_sizes[size]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=font_size,
        # verticalalignment="top",
        bbox=bbox_props,
    )

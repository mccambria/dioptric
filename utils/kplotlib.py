# -*- coding: utf-8 -*-
"""This file contains standardized functions intended to simplify the
creation of publication-quality plots in a visually appealing, unique,
and consistent style.

Created on June 22nd, 2022

@author: mccambria
"""

# region Imports and constants

import utils.common as common
import matplotlib as mpl
import matplotlib.pyplot as plt
from strenum import StrEnum
from colorutils import Color
import re
from enum import Enum, auto
from strenum import StrEnum
from matplotlib.offsetbox import AnchoredText
import numpy as np


# matplotlib semantic locations for legends and text boxes
class Loc(StrEnum):
    BEST = "best"
    LOWER_LEFT = "lower left"
    UPPER_LEFT = "upper left"
    LOWER_RIGHT = "lower right"
    UPPER_RIGHT = "upper right"


class Size(Enum):
    NORMAL = "NORMAL"
    SMALL = "SMALL"
    TINY = "TINY"


class MarkerSize(float, Enum):
    NORMAL = 7
    SMALL = 6
    TINY = 4


class LineWidth(float, Enum):
    HUGE = 2.5
    BIG = 2.0
    NORMAL = 1.5
    SMALL = 1.25
    TINY = 1.0


class MarkerEdgeWidth(float, Enum):
    NORMAL = 1.5
    SMALL = 1.25
    TINY = 1.0


class FontSize(float, Enum):
    NORMAL = 17
    SMALL = 14
    TINY = 11


class PlotType(Enum):
    POINTS = auto()
    LINE = auto()
    HIST = auto()


class Font(Enum):
    ROBOTO = auto()
    HELVETICA = auto()


# Histogram type, mostly following https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
class HistType(StrEnum):
    INTEGER = "integer"  # Just plot the frequency of each integer
    STEP = "step"  # No space between between, unfilled
    BAR = "bar"  # Space between bins, filled


# Default sizes
marker_size = MarkerSize.NORMAL
marker_size_inset = MarkerSize.SMALL
line_width = LineWidth.NORMAL
line_width_inset = LineWidth.NORMAL
marker_edge_width = MarkerEdgeWidth.NORMAL
marker_edge_width_inset = MarkerEdgeWidth.NORMAL
default_font_size = Size.NORMAL
default_data_size = Size.NORMAL
figsize = [6.5, 5.0]
double_figsize = [figsize[0] * 2, figsize[1]]

# Default styles
line_style = "solid"
marker_style = "o"

# endregion
# region Colors
"""The default color specification is hex, eg "#bcbd22'"""


class KplColors(StrEnum):
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
    BLACK = "000000"


data_color_cycler = [
    KplColors.BLUE,
    KplColors.RED,
    KplColors.GREEN,
    KplColors.ORANGE,
    KplColors.PURPLE,
    KplColors.BROWN,
    KplColors.PINK,
    KplColors.GRAY,
    KplColors.YELLOW,
    KplColors.CYAN,
]
line_color_cycler = data_color_cycler.copy()
hist_color_cycler = data_color_cycler.copy()


def color_mpl_to_color_hex(color_mpl):
    """Convert a named color from a matplotlib color map into hex"""
    # Trim the alpha value and convert from 0:1 to 0:255
    color_rgb = [255 * val for val in color_mpl[0:3]]
    color_Color = Color(tuple(color_rgb))
    color_hex = color_Color.hex
    return color_hex


def lighten_color_hex(color_hex, saturation_factor=0.3, value_factor=1.2):
    """Algorithmically lighten the passed hex color"""

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
    """Clip the passed value such that it is between 0 and 1"""
    if val < 0:
        return 0
    elif val > 1:
        return 1
    else:
        return val


# endregion
# region Miscellaneous


def init_kplotlib(
    font_size=Size.NORMAL,
    data_size=Size.NORMAL,
    font=Font.ROBOTO,
    constrained_layout=True,
    latex=False,
):
    """Runs the initialization for kplotlib, our default configuration
    of matplotlib. Plotting will be faster if latex is False - only set to True
    if you need full access to LaTeX
    """

    ### Misc setup

    # Reset to the default
    mpl.rcParams.update(mpl.rcParamsDefault)

    global active_axes, color_cyclers, default_font_size, default_data_size
    active_axes = []
    color_cyclers = []
    default_font_size = font_size
    default_data_size = data_size

    # Interactive mode so plots update as soon as the event loop runs
    plt.ion()

    ### Latex setup

    preamble = r""
    preamble += r"\newcommand\hmmax{0} \newcommand\bmmax{0}"
    preamble += r"\usepackage{physics} \usepackage{upgreek}"
    if font == Font.ROBOTO:
        preamble += r"\usepackage{roboto}"  # Google's free Helvetica
    elif font == Font.HELVETICA:
        preamble += r"\usepackage{helvet}"

    preamble += r"\usepackage[T1]{fontenc}"
    preamble += r"\usepackage{siunitx}"
    preamble += r"\sisetup{detect-all}"

    # Note: The global usetex setting should remain off. This prevents serif fonts
    # from proliferating (e.g. to axis tick labels). Instead just pass usetex=True
    # as a kwarg to any text-based command as necessary. If really necessary, flip
    # the flag below and use the \mathrm and \mathit macros to keep serifs in check
    if False:  # Global usetex?
        preamble += r"\usepackage[mathrmOrig, mathitOrig]{sfmath}"
        plt.rcParams["text.latex.preamble"] = preamble
        plt.rc("text", usetex=True)
    else:
        plt.rcParams["text.latex.preamble"] = preamble

    ### Other rcparams

    # plt.rcParams["legend.handlelength"] = 0.5
    plt.rcParams["font.family"] = "sans-serif"
    if font == Font.ROBOTO:
        plt.rcParams["font.sans-serif"] = "Roboto"
    if font == Font.HELVETICA:
        plt.rcParams["font.sans-serif"] = "Helvetica"
    plt.rcParams["font.size"] = FontSize[default_font_size.value]
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["image.cmap"] = "inferno"
    plt.rcParams["figure.constrained_layout.use"] = constrained_layout
    plt.rcParams["savefig.format"] = "svg"
    plt.rcParams["figure.max_open_warning"] = 100


def get_default_color(ax, plot_type):
    """Get the default color according to the cycler of the passed plot type.

    plot_type : PlotType(enum)
    """

    global active_axes, color_cyclers
    if ax not in active_axes:
        active_axes.append(ax)
        color_cyclers.append(
            {
                PlotType.POINTS: data_color_cycler.copy(),
                PlotType.LINE: line_color_cycler.copy(),
                PlotType.HIST: hist_color_cycler.copy(),
            }
        )
    ax_ind = active_axes.index(ax)
    cycler = color_cyclers[ax_ind][plot_type]
    color = cycler.pop(0)
    return color


def anchored_text(ax, text, loc, size=None, **kwargs):
    """Add text in default style to the passed ax. To update text call set_text on the
    returned object's txt property

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add the text box to
    text : str
    loc : str or Loc(enum)
        Relative location to anchor the text box to
    size : Size(enum), optional
        Font size, defaults to default_font_size set up in init_kplotlib

    Returns
    -------
    matplotlib.offsetbox.AnchoredText
    """

    global default_font_size
    if size is None:
        size = default_font_size

    font_size = FontSize[size.value]
    text_props = kwargs
    text_props["fontsize"] = font_size
    # text_props = dict(fontsize=font_size)
    text_box = AnchoredText(text, loc, prop=text_props)
    text_box.patch.set_boxstyle("round, pad=0, rounding_size=0.2")
    text_box.patch.set_facecolor("wheat")
    text_box.patch.set_alpha(0.5)
    ax.add_artist(text_box)
    return text_box


def tex_escape(text):
    """Escape TeX characters in the passed text"""
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    regex = re.compile(
        "|".join(
            re.escape(str(key))
            for key in sorted(conv.keys(), key=lambda item: -len(item))
        )
    )
    return regex.sub(lambda match: conv[match.group()], text)


def flush_update(ax=None, fig=None):
    """Call this after making some change to an existing figure to have the figure
    actually update in the window
    """
    if fig is None:
        fig = ax.get_figure()
    fig.canvas.draw()
    fig.canvas.flush_events()


# endregion
# region Plotting


def plot_points(ax, x, y, size=None, **kwargs):
    """Same as matplotlib's errorbar, but with our defaults. Use for plotting
    data points

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : 1D array
        x values to plot
    y : 1D array
        y values to plot
    size : Size(enum), optional
        Data point size, by default data_size passed to init_kplotlib
    kwargs
        Passed on to matplotlib's errorbar function
    """

    global default_data_size
    if size is None:
        size = default_data_size

    # Color handling
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = get_default_color(ax, PlotType.POINTS)
    if "markerfacecolor" in kwargs:
        face_color = kwargs["markerfacecolor"]
    else:
        face_color = lighten_color_hex(color)

    # Defaults
    params = {
        "linestyle": "none",
        "marker": marker_style,
        "markersize": MarkerSize[size.value],
        "markeredgewidth": MarkerEdgeWidth[size.value],
    }

    # Combine passed args and defaults
    params = {**params, **kwargs}
    params["color"] = color
    params["markerfacecolor"] = face_color

    ax.errorbar(x, y, **params)


def plot_line(ax, x, y, size=None, **kwargs):
    """Same as matplotlib's plot, but with our defaults. Use for plotting
    continuous lines

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : 1D array
        x values to plot
    y : 1D array
        y values to plot
    size : Size(enum), optional
        Line width, by default data_size passed to init_kplotlib
    kwargs
        Passed on to matplotlib's plot function
    """

    global default_data_size
    if size is None:
        size = default_data_size

    # Color handling
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = get_default_color(ax, PlotType.LINE)

    # Defaults
    params = {
        "linestyle": line_style,
        "linewidth": LineWidth[size.value],
    }

    # Combine passed args and defaults
    params = {**params, **kwargs}
    params["color"] = color

    ax.plot(x, y, **params)


def plot_line_update(
    ax, line_ind=0, x=None, y=None, relim_x=True, relim_y=True, flush=True
):
    """Updates a plot created by plot_line

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to update
    line_ind : int, optional
        Index of the line in the plot to update, by default 0
    x : 1D array, optional
        New x values to write, by default None
    y : 1D array, optional
        New y values to write, by default None
    relim_x : bool, optional
        Update the x limits of the plot? By default True
    relim_y : bool, optional
        Update the y limits of the plot? By default True
    flush : bool, optional
        Flush the updates (i.e. render them)? If False, call flush_update
        yoursel. By default True
    """

    lines = ax.get_lines()
    line = lines[line_ind]

    if x is not None:
        line.set_xdata(x)
    if y is not None:
        line.set_ydata(y)

    if relim_x or relim_y:
        ax.relim()
        ax.autoscale_view(scalex=relim_x, scaley=relim_y)

    if flush:
        flush_update(ax)


def imshow(
    ax, img_array, title=None, x_label=None, y_label=None, cbar_label=None, **kwargs
):
    """Same as matplotlib's imshow, but with our defaults

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes for the image
    img_array : 2D array
        Values to plot in the image
    title : str, optional
        Image title, by default None
    x_label : str, optional
        x axis label, by default None
    y_label : str, optional
        y axis label, by default None
    cbar_label : str, optional
        Color bar label, by default None
    kwargs
        Passed on to matplotlib's plot function

    Returns
    -------
    matplotlib.image.AxesImage
    """

    fig = ax.get_figure()

    # Get a default aspect ratio
    if "extent" in kwargs and kwargs["extent"] is not None:
        extent = tuple(kwargs["extent"])
        kwargs["extent"] = extent
        if "aspect" not in kwargs:
            height = abs(extent[3] - extent[2])
            width = abs(extent[1] - extent[0])
            aspect = height / width
            kwargs["aspect"] = aspect

    img = ax.imshow(img_array, **kwargs)

    # Colorbar and labels
    clb = fig.colorbar(img)
    if cbar_label is not None:
        clb.set_label(cbar_label)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if title:
        plt.title(title)

    # Click handler
    fig.canvas.mpl_connect("button_press_event", on_click_image)

    return img


def imshow_update(ax, img_array, cmin=None, cmax=None):
    """Update the (first) image in the passed ax

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes for the image
    img_array : 2D array
        Values to plot in the image
    cmin : numeric, optional
        Minimum color bar value, by default None
    cmax : numeric, optional
        Maximum color bar value, by default None
    """
    images = ax.get_images()
    img = images[0]
    img.set_data(img_array)
    if (cmin != None) & (cmax != None):
        img.set_clim(cmin, cmax)
    else:
        img.autoscale()
    img.autoscale()
    flush_update(ax)


def on_click_image(event):
    """Click handler for images from imshow. Prints the click coordinates to
    the console. Event is passed automatically
    """
    x_coord = round(event.xdata, 3)
    y_coord = round(event.ydata, 3)
    try:
        print(f"{x_coord}, {y_coord}")
    except TypeError:
        # Ignore the exc that's raised if the user clicks out of bounds
        pass


def histogram(ax, data, hist_type=HistType.INTEGER, nbins=10, **kwargs):
    """Similar to matplotlib's hist, but with our defaults

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : array
        Data to histogram - will be flattened
    hist_type : HistType(enum), optional
        Histogram type, by default HistType.INTEGER
    nbins : int, optional
        Number of bins, by default 10
    kwargs
        Passed on to matplotlib's plot function

    Returns
    -------
    1D array
        Histogram occurrences (same as numpy histogram)
    1D array
        Histogram bin edges
    """

    # Color handling
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = get_default_color(ax, PlotType.HIST)

    # Copy kwargs and set the color
    params = {**kwargs}
    params["color"] = color

    # Just make a line plot of the frequencies of each integer
    if hist_type == HistType.INTEGER:
        max_data = int(max(data))
        occur, bin_edges = np.histogram(data, np.linspace(0, max_data, max_data + 1))
        x_vals = bin_edges[:-1]
        plot_line(ax, x_vals, occur, **kwargs)
    else:
        occur, bin_edges, _ = ax.hist(data, histtype=hist_type, bins=nbins, **kwargs)

    return occur, bin_edges


# endregion

if __name__ == "__main__":
    # print(cambria_fixed(15))
    # sys.exit()

    # calc_zfs_from_compiled_data()

    init_kplotlib()

    # main()
    x = np.random.randint(10, size=500)
    fig, ax = plt.subplots()
    histogram(ax, x, hist_type=HistType.BAR)

    plt.show(block=True)

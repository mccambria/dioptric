# -*- coding: utf-8 -*-
"""This file contains standardized functions intended to simplify the
creation of publication-quality plots in a visually appealing, unique,
and consistent style.

Created on June 22nd, 2022

@author: mccambria
"""

# region Imports and constants

import utils.common as common
import matplotlib.pyplot as plt
from strenum import StrEnum
from colorutils import Color
import re
from enum import Enum, auto
from strenum import StrEnum
from matplotlib.offsetbox import AnchoredText

# matplotlib semantic locations for legends and text boxes
class Loc(StrEnum):
    BEST = "best"
    LOWER_LEFT = "lower left"
    UPPER_LEFT = "upper left"
    LOWER_RIGHT = "lower right"
    UPPER_RIGHT = "upper right"


class Size(Enum):
    NORMAL = auto()
    SMALL = auto()
    TINY = auto()


class PlotType(Enum):
    DATA = auto()
    LINE = auto()


# Size options
marker_Size = {Size.NORMAL: 7, Size.SMALL: 6, Size.TINY: 4}
line_widths = {Size.NORMAL: 1.5, Size.SMALL: 1.25, Size.TINY: 1.0}
marker_edge_widths = line_widths.copy()
font_Size = {Size.NORMAL: 17, Size.SMALL: 13}

# Default sizes
marker_size = marker_Size[Size.NORMAL]
marker_size_inset = marker_Size[Size.SMALL]
line_width = line_widths[Size.NORMAL]
line_width_inset = line_widths[Size.SMALL]
marker_edge_width = marker_edge_widths[Size.NORMAL]
marker_edge_width_inset = marker_edge_widths[Size.SMALL]
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
    if val < 0:
        return 0
    elif val > 1:
        return 1
    else:
        return val


# endregion
# region Miscellaneous


def init_kplotlib(font_size=Size.NORMAL, data_size=Size.NORMAL, no_latex=False):
    """Runs the default initialization for kplotlib, our default configuration
    of matplotlib. Make sure no_latex is True for faster plotting.
    """

    ### Misc setup

    global active_axes, color_cyclers, default_font_size, default_data_size
    active_axes = []
    color_cyclers = []
    default_font_size = font_size
    default_data_size = data_size

    # Interactive mode so plots update as soon as the event loop runs
    plt.ion()

    ### Latex setup

    if not no_latex:

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
        plt.rc("text", usetex=True)

    ### Other rcparams

    # plt.rcParams["savefig.format"] = "svg"
    # plt.rcParams["legend.handlelength"] = 0.5

    plt.rcParams["font.size"] = font_Size[default_font_size]
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["image.cmap"] = "inferno"


def tight_layout(fig):
    """Tight layout with defaults. Called twice because sometimes things are
    still off after the first call.
    """

    fig.tight_layout(pad=0.3)
    fig.tight_layout(pad=0.3)


def get_default_color(ax, plot_type):
    """Get the default color according to the cycler of the passed plot type.

    plot_type : PlotType(enum)
    """

    global active_axes, color_cyclers
    if ax not in active_axes:
        active_axes.append(ax)
        color_cyclers.append(
            {
                "points": data_color_cycler.copy(),
                "line": line_color_cycler.copy(),
            }
        )
    ax_ind = active_axes.index(ax)
    cycler = color_cyclers[ax_ind][plot_type]
    color = cycler.pop(0)
    return color


def anchored_text(ax, text, loc, size=None, **kwargs):
    """Add text in default style to the passed ax"""

    global default_font_size
    if size is None:
        size = default_font_size

    font_size = font_Size[size]
    text_props = dict(fontsize=font_size)
    text_box = AnchoredText(text, loc, prop=text_props)
    text_box.patch.set_boxstyle("round, pad=0.05")
    text_box.patch.set_facecolor("wheat")
    text_box.patch.set_alpha(0.5)
    ax.add_artist(text_box)


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


def flush_update(ax):
    """Call this after making some change to an existing figure to have the figure
    actually update in the window
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    fig.canvas.flush_events()


# endregion
# region Plotting


def plot_points(ax, x, y, size=None, **kwargs):
    """Same as matplotlib's errorbar, but with our defaults. Use for plotting
    data points

    size : Size(enum)
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
        "markersize": marker_Size[size],
        "markeredgewidth": marker_edge_widths[size],
    }

    # Combine passed args and defaults
    params = {**params, **kwargs}
    params["color"] = color
    params["markerfacecolor"] = face_color

    ax.errorbar(x, y, **params)


def plot_line(ax, x, y, size=None, **kwargs):
    """Same as matplotlib's plot, but with our defaults. Use for plotting
    continuous lines

    size : Size(enum)
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


def plot_line_update(ax, line_ind=0, x=None, y=None):
    """Updates a figure created by plot_line. x and y are the new data to write.
    Either may be None in which case that axis of the plot won't be updated
    """

    # Get the line - assume it's  the first line in the first axes
    lines = ax.get_lines()
    line = lines[line_ind]

    # Set the data for the line to display and rescale
    if x is not None:
        line.set_xdata(x)
    if y is not None:
        line.set_ydata(y)
    ax.relim()
    ax.autoscale_view(scalex=False)

    flush_update(ax)


def imshow(ax, img_array, title=None, axes_labels=None, cbar_label=None, **kwargs):
    """Same as matplotlib's imshow, but with our defaults. Returns the image object"""

    fig = ax.get_figure()

    # Get a default aspect ratio
    if "extent" in kwargs:
        extent = tuple(kwargs["extent"])
        kwargs["extent"] = extent
        if "aspect" not in kwargs:
            height = extent[3] - extent[2]
            width = extent[1] - extent[0]
            aspect = height / width
            kwargs["aspect"] = aspect

    img = ax.imshow(img_array, **kwargs)

    # Colorbar and labels
    clb = fig.colorbar(img)
    if cbar_label is not None:
        clb.set_label(cbar_label)
    if axes_labels is not None:
        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])
    if title:
        plt.title(title)

    # Click handler
    fig.canvas.mpl_connect("button_press_event", on_click_image)

    tight_layout(fig)

    return img


def imshow_update(ax, img_array, cmin=None, cmax=None):
    """Update the first image in the passed ax"""
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


# endregion

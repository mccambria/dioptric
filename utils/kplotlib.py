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


def plot_line_update(ax, x=None, y=None):
    """Updates a figure created by plot_line. x and y are the new data to write.
    Either may be None in which case that axis of the plot won't be updated
    """

    # Get the line - assume it's  the first line in the first axes
    lines = ax.get_lines()
    line = lines[0]

    # Set the data for the line to display and rescale
    if x is not None:
        line.set_xdata(x)
    if y is not None:
        line.set_ydata(y)
    ax.relim()
    ax.autoscale_view(scalex=False)

    # Redraw the canvas and flush the changes to the backend
    # start = time.time()
    fig = ax.get_figure()
    fig.canvas.draw()
    fig.canvas.flush_events()
    # stop = time.time()
    # print(f"Tool time: {stop - start}")


def imshow(
    imgArray,
    imgExtent,
    clickHandler=None,
    title=None,
    color_bar_label="Counts",
    um_scaled=False,
    axes_labels=None,  # ["V", "V"],
    aspect_ratio=None,
    color_map="inferno",
    cmin=None,
    cmax=None,
):
    """Creates a figure containing a single grayscale image and a colorbar.

    Params:
        imgArray: np.ndarray
            Rectangular np array containing the image data.
            Just zeros if you're going to be writing the image live.
        imgExtent: list(float)
            The extent of the image in the form [left, right, bottom, top]
        clickHandler: function
            Function that fires on clicking in the image

    Returns:
        matplotlib.figure.Figure
    """

    # plt.rcParams.update({'font.size': 22})

    # if um_scaled:
    #     axes_label = r"$\mu$m"
    # else:
    if axes_labels == None:
        try:
            a = common.get_registry_entry_no_cxn(
                "xy_units", ["", "Config", "Positioning"]
            )
            axes_labels = [a, a]
        except Exception as exc:
            print(exc)
            axes_labels = ["V", "V"]
    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots()

    # make sure the image is square
    # plt.axis('square')

    fig.set_tight_layout(True)

    # Tell the axes to show a grayscale image
    # print(imgArray)
    img = ax.imshow(
        imgArray,
        cmap=color_map,
        extent=tuple(imgExtent),
        vmin=cmin,  # min_value,
        vmax=cmax,
        aspect=aspect_ratio,
    )

    #    if min_value == None:
    #        img.autoscale()

    # Add a colorbar
    clb = plt.colorbar(img)
    clb.set_label(color_bar_label)
    # clb.ax.set_tight_layout(True)
    # clb.ax.set_title(color_bar_label)
    #    clb.set_label('kcounts/sec', rotation=270)

    # Label axes
    plt.xlabel(axes_labels[0])
    plt.ylabel(axes_labels[1])
    if title:
        plt.title(title)

    # Wire up the click handler to print the coordinates
    if clickHandler is not None:
        fig.canvas.mpl_connect("button_press_event", clickHandler)

    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig


def on_click_image(event):
    """
    Click handler for images. Prints the click coordinates to the console.

    Params:
        event: dictionary
            Dictionary containing event details
    """

    try:
        print("{:.3f}, {:.3f}".format(event.xdata, event.ydata))
    #        print('[{:.3f}, {:.3f}, 50.0],'.format(event.xdata, event.ydata))
    except TypeError:
        # Ignore TypeError if you click in the figure but out of the image
        pass


def imshow_update(fig, imgArray, cmin=None, cmax=1000):
    """Update the image with the passed image array and redraw the figure.
    Intended to update figures created by create_image_figure.

    The implementation below isn't nearly the fastest way of doing this, but
    it's the easiest and it makes a perfect figure every time (I've found
    that the various update methods accumulate undesirable deviations from
    what is produced by this brute force method).

    Params:
        fig: matplotlib.figure.Figure
            The figure containing the image to update
        imgArray: np.ndarray
            The new image data
    """

    # Get the image - Assume it's the first image in the first axes
    axes = fig.get_axes()
    ax = axes[0]
    images = ax.get_images()
    img = images[0]

    # Set the data for the image to display
    img.set_data(imgArray)

    # Check if we should clip or autoscale
    clipAtThousand = False
    if clipAtThousand:
        if np.all(np.isnan(imgArray)):
            imgMax = 0  # No data yet
        else:
            imgMax = np.nanmax(imgArray)
        if imgMax > 1000:
            img.set_clim(None, 1000)
        else:
            img.autoscale()
    elif (cmax != None) & (cmin != None):
        img.set_clim(cmin, cmax)
    else:
        img.autoscale()

    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()


# endregion

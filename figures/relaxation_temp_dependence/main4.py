import errno
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as patches
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
import pandas as pd
import utils.tool_belt as tool_belt
import utils.common as common
from scipy.odr import ODR, Model, RealData
import sys
from pathlib import Path
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
import temp_dependence_fitting
import csv

marker_size = 7
# line_width = 1.5
line_width = 2.5
marker_edge_width = line_width


def round_base_2(val):
    power = round(np.log2(val))
    rounded_val = 2 ** power
    return rounded_val


def bar_gill_replot(file_name, path):

    data_points = []
    with open(path / file_name, newline="") as f:
        raw_data = csv.reader(f)
        prev_point_ind = -1
        new_point = None
        header = True
        for row in raw_data:
            if header:
                header = False
                continue
            point_ind = int(row[3])
            if point_ind != prev_point_ind:
                prev_point_ind = point_ind
                if new_point is not None:
                    data_points.append(new_point)
                new_point = {
                    "temp": int(row[0]),
                    "num_pulses": round_base_2(float(row[1])),
                }
            row_type = row[4].strip()
            val = float(row[2])
            new_point[row_type] = val

    for point in data_points:
        T2 = point["main"]
        if ("ste_above" in point) and ("ste_below" in point):
            avg_ste = (
                (point["ste_above"] - T2) + (T2 - point["ste_below"])
            ) / 2
            point["ste"] = avg_ste
        elif "ste_above" in point:
            point["ste"] = point["ste_above"] - T2
        elif "ste_below" in point:
            point["ste"] = T2 - point["ste_below"]
        else:
            point["ste"] = None

    colors = {
        300: "blue",
        240: "green",
        190: "purple",
        160: "cyan",
        120: "red",
        77: "yellow",
    }
    fig, ax = plt.subplots(figsize=[6.5, 5.0])
    for point in data_points:
        ax.errorbar(
            point["num_pulses"],
            point["main"],
            point["ste"],
            color=colors[point["temp"]],
            marker="o",
        )

    ax.set_yscale("log")
    ax.set_xscale("log")
    fig.tight_layout()


def main(
    file_name,
    path,
    plot_type,
    rates_to_plot,
    temp_range,
    y_range,
    xscale,
    yscale,
    dosave=False,
):

    measured_vals = [580, 152, 39.8, 17.3, 5.92, 3.34]
    measured_vals = [val / 1000 for val in measured_vals]
    measured_errs = [210, 52, 7.7, 4.3, 1.23, 0.41]
    measured_errs = [val / 1000 for val in measured_errs]
    temps = [77, 120, 160, 190, 240, 300]
    authors = []
    authors.extend(["Bar Gill"] * 6)

    fig, ax = temp_dependence_fitting.main(
        file_name,
        path,
        plot_type,
        rates_to_plot,
        temp_range,
        y_range,
        xscale,
        yscale,
        dosave=False,
    )

    colors = {
        "Bar Gill": "blue",
    }
    # markers = [
    #     "o",
    #     "^",
    #     "s",
    #     "X",
    #     "D",
    #     "H",
    # ]
    markers = {"Bar Gill": "o"}

    ms = marker_size ** 2
    num_points = len(measured_vals)
    used_authors = []
    for ind in range(num_points):
        temp = temps[ind]
        val = measured_vals[ind]
        err = measured_errs[ind]
        author = authors[ind]
        color = colors[author]
        marker = markers[author]
        label = None
        if author not in used_authors:
            used_authors.append(author)
            label = author
        ax.errorbar(
            temp,
            val,
            err,
            color=color,
            label=label,
            marker=marker,
            ms=marker_size,
            lw=line_width,
            markeredgewidth=marker_edge_width,
            linestyle="None",
        )

    ax.legend()


if __name__ == "__main__":

    tool_belt.init_matplotlib()
    # matplotlib.rcParams["axes.linewidth"] = 1.0

    file_name = "compiled_data"
    home = common.get_nvdata_dir()
    path = home / "paper_materials/relaxation_temp_dependence"

    plot_type = "T2_max"
    y_range = [1e-3, 10]
    yscale = "log"
    temp_range = [0, 480]
    xscale = "linear"
    rates_to_plot = "both"

    main(
        file_name,
        path,
        plot_type,
        rates_to_plot,
        temp_range,
        y_range,
        xscale,
        yscale,
        dosave=False,
    )

    # file_name = "bar_gill_2012-2a.csv"
    # home = common.get_nvdata_dir()
    # path = home / "paper_materials/relaxation_temp_dependence/ripped_T2_plots"
    # bar_gill_replot(file_name, path)

    plt.show(block=True)

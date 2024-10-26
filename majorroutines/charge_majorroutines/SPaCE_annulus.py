# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:33:57 2020

A routine to take one NV to readout the charge state, after pulsing a laser
at a distance from this readout NV.


@author: agardill
"""

import copy
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
import scipy.stats as stats
from scipy.optimize import curve_fit

import majorroutines.image_sample as image_sample
import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt

# Define the start time to track the time between optimzies
time_start = time.time()


def inverse_sqrt(x, a, o):
    return a * x ** (-1 / 2) + o


def inverse_quarter(x, a):
    return a * x ** (-1 / 4)


def power_fnct(x, a, b):
    # o = 15
    # return numpy.sqrt((a*x**-b)**2 + (o )**2)
    return a * x ** (b)


def inverse_law(x, a):
    return a * x**-1


def exp_decay(x, a, d):
    return a * numpy.exp(-x / d)


def exp_sqrd_decay(x, a, d):
    return a * numpy.exp(-(x**2) / d)


def sq_gaussian(x, *params):
    """
    Calculates the value of a gaussian for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the Gaussian
            0: coefficient that defines the peak height
            1: mean, defines the center of the Gaussian
            2: standard deviation, defines the width of the Gaussian
            3: constant y value to account for background
    """

    coeff, mean, stdev, offset = params
    var = stdev**4  # variance
    centDist = x - mean  # distance from the center
    return offset + coeff**2 * numpy.exp(-(centDist**4) / (4 * var))


# %%
def plot_1D_SpaCE(file_name, file_path, do_plot=True, do_fit=False, do_save=True):
    data = tool_belt.get_raw_data(file_name, file_path)
    timestamp = data["timestamp"]
    nv_sig = data["nv_sig"]
    CPG_pulse_dur = nv_sig["CPG_laser_dur"]
    # dir_1D = nv_sig['dir_1D']
    start_coords = nv_sig["coords"]

    counts = data["readout_counts_avg"]
    coords_voltages = data["coords_voltages"]
    x_voltages = numpy.array([el[0] for el in coords_voltages])
    y_voltages = numpy.array([el[1] for el in coords_voltages])
    num_steps = data["num_steps"]

    start_coords = nv_sig["coords"]

    # calculate the radial distances from the readout NV to the target points
    rad_dist = (
        numpy.sqrt(
            (x_voltages - start_coords[0]) ** 2 + (y_voltages - start_coords[1]) ** 2
        )
        * 35000
    )

    # if dir_1D == 'x':
    #     coord_ind = 0
    # elif dir_1D == 'y':
    #     coord_ind = 1
    # voltages = [i[coord_ind] for i in coords_voltages]
    # voltages = numpy.array(voltages)
    # rad_dist = (voltages - start_coords[coord_ind])*35000
    opti_params = []

    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(rad_dist, counts, "b.")
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("Average counts")
        ax.set_title("{} us pulse".format(CPG_pulse_dur / 10**3))

    if do_fit:
        init_fit = [2, rad_dist[int(num_steps / 2)], 15, 7]
        try:
            opti_params, cov_arr = curve_fit(sq_gaussian, rad_dist, counts, p0=init_fit)
            if do_plot:
                text = r"$C + A^2 e^{-(r - r_0)^4/(2*\sigma^4)}$"
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                ax.text(
                    0.05,
                    0.95,
                    text,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=props,
                )
                lin_radii = numpy.linspace(rad_dist[0], rad_dist[-1], 100)
                ax.plot(lin_radii, sq_gaussian(lin_radii, *opti_params), "r-")
                text = (
                    "A={:.3f} sqrt(counts)\n$r_0$={:.3f} nm\n "
                    "$\sigma$={:.3f} nm\nC={:.3f} counts".format(*opti_params)
                )
                ax.text(
                    0.3,
                    0.1,
                    text,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=props,
                )
            print(opti_params)
        except Exception:
            text = "Peak could not be fit"
            ax.text(
                0.3,
                0.1,
                text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )

    if do_plot and do_save:
        filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
        tool_belt.save_figure(fig, filePath + "-sq_gaussian_fit")

    return rad_dist, counts, opti_params


def gaussian_fit_1D_airy_rings(file_name, file_path, lobe_positions):
    rad_dist, counts, _ = plot_1D_SpaCE(file_name, file_path, do_plot=False)

    data = tool_belt.get_raw_data(file_name, file_path)
    timestamp = data["timestamp"]
    nv_sig = data["nv_sig"]
    dir_1D = nv_sig["dir_1D"]
    num_steps = data["num_steps"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    text = r"$C + A^2 e^{-(r - r_0)^2/(2*\sigma^2)}$"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    ax.set_title(
        "{} us pulse, power set to {}".format(
            nv_sig["CPG_laser_dur"] / 10**3, nv_sig["CPG_laser_power"]
        )
    )

    ax.plot(rad_dist, counts, "b.")
    if dir_1D == "x":
        ax.set_xlabel("x (nm)")
    elif dir_1D == "y":
        ax.set_xlabel("y (nm)")
    ax.set_ylabel("Average counts")

    # get idea of where the center is
    zero_ind = int(num_steps / 2)
    step_size_nm = rad_dist[1] - rad_dist[0]
    # Calculate the steps we'll consider around each ring position
    wings_ind = int(400 / step_size_nm)

    fit_params_list = []
    ring_positions = lobe_positions
    for ri in range(len(ring_positions)):
        fit_fail = False
        ring_r_nm = ring_positions[ri]
        if ring_r_nm > 0:
            ring_ind = int(zero_ind + ring_r_nm / step_size_nm)
        else:
            ring_ind = int(zero_ind - abs(ring_r_nm / step_size_nm))
        ring_range = [ring_ind - wings_ind, ring_ind + wings_ind]
        # ring_ind_high = int(zero_ind + ring_r_nm / step_size_nm)
        # ring_range_high = [ring_ind_high-wings_ind,
        #                     ring_ind_high+wings_ind]
        # ring_ind_low = int(zero_ind - ring_r_nm / step_size_nm)
        # ring_range_low = [ring_ind_low-wings_ind,
        #                     ring_ind_low+wings_ind]
        init_fit = [2, ring_r_nm, 15, 7]
        try:
            opti_params, cov_arr = curve_fit(
                tool_belt.gaussian,
                rad_dist[ring_range[0] : ring_range[1]],
                counts[ring_range[0] : ring_range[1]],
                p0=init_fit,
                bounds=(
                    [-numpy.infty, -(abs(ring_r_nm) + 10), -80, 0],
                    [numpy.infty, abs(ring_r_nm) + 10, 80, 11],
                ),
            )

            lin_radii = numpy.linspace(
                rad_dist[ring_range[0]], rad_dist[ring_range[1]], 100
            )
            ax.plot(lin_radii, tool_belt.gaussian(lin_radii, *opti_params), "r-")
            text = (
                "A={:.3f} sqrt(counts)\n$r_0$={:.3f} nm\n "
                "$\sigma$={:.3f} nm\nC={:.3f} counts".format(*opti_params)
            )
            ax.text(
                0.3 * (ri),
                0.1,
                text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )
            print(opti_params)
        except Exception:
            fit_fail = True
            text = "Peak could not be\nfit for +{} nm lobe".format(ring_r_nm)
            ax.text(
                0.3 * ri,
                0.1,
                text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )

        if not fit_fail:
            fit_params_list.append(opti_params)
        else:
            fit_params_list.append(None)

        # init_fit = [4, -ring_r_nm, 15, 3]
        # try:
        #     opti_params, cov_arr = curve_fit(tool_belt.gaussian,
        #           rad_dist[ring_range_low[0]: ring_range_low[1]],
        #           counts[ring_range_low[0]: ring_range_low[1]],
        #           p0=init_fit,
        #           bounds = ([-numpy.infty, -numpy.infty, -numpy.infty, 0],
        #                     [numpy.infty, 0, 100, 9]))

        #     lin_radii = numpy.linspace(rad_dist[ring_range_low[0]],
        #                         rad_dist[ring_range_low[1]], 100)
        #     ax.plot(lin_radii,
        #            tool_belt.gaussian(lin_radii, *opti_params), 'r-')
        #     text = 'A={:.3f} sqrt(counts)\n$r_0$={:.3f} nm\n ' \
        #         '$\sigma$={:.3f} nm\nC={:.3f} counts'.format(*opti_params)
        #     ax.text(0.5-0.3*(1+ri), 0.1, text, transform=ax.transAxes, fontsize=12,
        #             verticalalignment='top', bbox=props)
        #     print(opti_params)
        # except Exception:
        #     text = 'Peak could not be fit for -{} nm lobe'.format(ring_r_nm)
        #     ax.text(0.5-0.3*(1+ri), 0.1, text, transform=ax.transAxes, fontsize=12,
        #             verticalalignment='top', bbox=props)

    # filePath = tool_belt.get_file_path(__file__, timestamp,
    #                                         nv_sig['name'])
    # tool_belt.save_figure(fig, filePath + '-gaussian_fit')

    return fit_params_list


def plot_2D_space(file, path):
    data = tool_belt.get_raw_data(file, path)
    # try:
    nv_sig = data["nv_sig"]
    CPG_laser_dur = nv_sig["CPG_laser_dur"]
    readout_counts_avg = numpy.array(data["readout_counts_avg"])
    num_steps_b = data["num_steps_b"]
    x_voltages = data["a_voltages_1d"]
    y_voltages = data["b_voltages_1d"]
    x_low = x_voltages[0]
    x_high = x_voltages[-1]
    y_low = y_voltages[0]
    y_high = y_voltages[-1]
    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [
        x_high + half_pixel_size,
        x_low - half_pixel_size,
        y_low - half_pixel_size,
        y_high + half_pixel_size,
    ]

    split_counts = numpy.split(readout_counts_avg, num_steps_b)
    readout_image_array = numpy.vstack(split_counts)
    r = 0
    for i in reversed(range(len(readout_image_array))):
        if r % 2 == 0:
            readout_image_array[i] = list(reversed(readout_image_array[i]))
        r += 1

    title = "SPaCE - {} ms depletion pulse".format(CPG_laser_dur)

    tool_belt.create_image_figure(
        readout_image_array,
        img_extent,
        clickHandler=None,
        title=title,
        color_bar_label="Counts",
        min_value=None,
        um_scaled=False,
    )


# %%


def build_voltages_from_list_xyz(
    start_coords_drift, coords_list_drift, movement_incr, step_size_list
):
    # adding some modifications to incrimentally step to the desired target position
    # calculate the x values we want to step thru
    start_x_value = start_coords_drift[0]
    start_y_value = start_coords_drift[1]
    start_z_value = start_coords_drift[2]

    step_size_x, step_size_y, step_size_z = step_size_list

    num_samples = len(coords_list_drift)

    # we want this list to have the pattern [[readout], [target_1], [target_2]...
    #                                                   [readout_1], [readout_2]...
    #                                                   [readout],
    #                                                   ...]
    # The glavo needs a 0th coord, so we'll pass the readout NV as the "starting" point
    x_points = [start_x_value]
    y_points = [start_y_value]
    z_points = [start_z_value]

    # now create a list of all the coords we want to feed to the galvo
    for i in range(num_samples):
        dx = coords_list_drift[i][0] - start_x_value
        dy = coords_list_drift[i][1] - start_y_value
        dz = coords_list_drift[i][2] - start_z_value

        # how many steps, based on step size, will it take to get to final position?
        # round up
        num_steps_x = numpy.ceil(abs(dx) / step_size_x)
        num_steps_y = numpy.ceil(abs(dy) / step_size_y)
        num_steps_z = numpy.ceil(abs(dz) / step_size_z)

        # max_num_steps = max([num_steps_x, num_steps_y,num_steps_z])
        # move to target in steps based on step size
        for n in range(movement_incr):
            # for the final move, just put in the prefered value to avoid rounding errors
            if n > num_steps_x - 1:
                x_points.append(coords_list_drift[i][0])
            else:
                move_x = (n + 1) * step_size_x * dx / abs(dx)
                incr_x_val = move_x + start_x_value
                x_points.append(incr_x_val)

            if n > num_steps_y - 1:
                y_points.append(coords_list_drift[i][1])
            else:
                move_y = (n + 1) * step_size_y * dy / abs(dy)
                incr_y_val = move_y + start_y_value
                y_points.append(incr_y_val)

            if n > num_steps_z - 1:
                z_points.append(coords_list_drift[i][2])
            else:
                move_z = (n + 1) * step_size_z * dz / abs(dz)
                incr_z_val = move_z + start_z_value
                z_points.append(incr_z_val)

        # readout, step back to NV
        for n in range(movement_incr):
            # for the final move, just put in the prefered value to avoid rounding errors
            if n > num_steps_x - 1:
                x_points.append(start_x_value)
            else:
                move_x = (n + 1) * step_size_x * dx / abs(dx)
                incr_x_val = coords_list_drift[i][0] - move_x
                x_points.append(incr_x_val)
            if n > num_steps_y - 1:
                y_points.append(start_y_value)
            else:
                move_y = (n + 1) * step_size_y * dy / abs(dy)
                incr_y_val = coords_list_drift[i][1] - move_y
                y_points.append(incr_y_val)

            if n > num_steps_z - 1:
                z_points.append(start_z_value)
            else:
                move_z = (n + 1) * step_size_z * dz / abs(dz)
                incr_z_val = coords_list_drift[i][2] - move_z
                z_points.append(incr_z_val)

        # initialize
        x_points.append(start_x_value)
        y_points.append(start_y_value)
        z_points.append(start_z_value)

    return x_points, y_points, z_points


def build_voltages_image(start_coords, img_range_2D, axes, num_steps_a, num_steps_b):
    # Make this arbitrary for building image in x, y, or z

    center_a = start_coords[axes[0]]
    center_b = start_coords[axes[1]]

    # num_steps_a = num_steps
    # num_steps_b = num_steps

    # Force the scan to have square pixels by only applying num_steps
    # to the shorter axis
    half_a_range = img_range_2D[axes[0]] / 2
    half_b_range = img_range_2D[axes[1]] / 2

    a_low = center_a - half_a_range
    a_high = center_a + half_a_range
    b_low = center_b - half_b_range
    b_high = center_b + half_b_range

    # Apply scale and offset to get the voltages we'll apply to the galvo
    # Note that the polar/azimuthal angles, not the actual x/y positions
    # are linear in these voltages. For a small range, however, we don't
    # really care.
    a_voltages_1d = numpy.linspace(a_low, a_high, num_steps_a)
    b_voltages_1d = numpy.linspace(b_low, b_high, num_steps_b)

    # Winding cartesian product
    # The x values are repeated and the y values are mirrored and tiled
    # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

    # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
    a_inter = numpy.concatenate((a_voltages_1d, numpy.flipud(a_voltages_1d)))
    # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
    if num_steps_b % 2 == 0:  # Even x size
        target_a_values = numpy.tile(a_inter, int(num_steps_b / 2))
    else:  # Odd x size
        target_a_values = numpy.tile(a_inter, int(numpy.floor(num_steps_b / 2)))
        target_a_values = numpy.concatenate((target_a_values, a_voltages_1d))

    # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
    target_b_values = numpy.repeat(b_voltages_1d, num_steps_a)

    return target_a_values, target_b_values, a_voltages_1d, b_voltages_1d


def collect_counts(cxn, movement_incr, num_samples, seq_args_string, apd_indices):
    #  Set up the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # prepare and run the sequence
    file_name = "SPaCE_w_movement_steps.py"
    cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    cxn.pulse_streamer.stream_start(num_samples)

    total_samples_list = []
    num_read_so_far = 0

    while num_read_so_far < num_samples:
        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        new_samples = cxn.apd_tagger.read_counter_simple()
        num_new_samples = len(new_samples)

        if num_new_samples > 0:
            for el in new_samples:
                total_samples_list.append(int(el))
            num_read_so_far += num_new_samples

    # The last of the triplet of readout windows is the counts we are interested in
    # readout_counts = total_samples_list[2::3]
    # print(total_samples_list)
    rep_samples = 2 * movement_incr + 1
    readout_counts = total_samples_list[rep_samples - 1 :: rep_samples]
    readout_counts_list = [int(el) for el in readout_counts]

    cxn.apd_tagger.stop_tag_stream()

    return readout_counts_list


# %%
def populate_img_array(valsToAdd, imgArray, run_num):
    """
    We scan the sample in a winding pattern. This function takes a chunk
    of the 1D list returned by this process and places each value appropriately
    in the 2D image array. This allows for real time imaging of the sample's
    fluorescence.

    Note that this function could probably be much faster. At least in this
    context, we don't care if it's fast. The implementation below was
    written for simplicity.

    Params:
        valsToAdd: numpy.ndarray
            The increment of raw data to add to the image array
        imgArray: numpy.ndarray
            The xDim x yDim array of fluorescence counts
        writePos: tuple(int)
            The last x, y write position on the image array. [] will default
            to the bottom right corner. Third index is the run number
    """
    yDim = imgArray.shape[0]
    xDim = imgArray.shape[1]

    # Start with the write position at the start
    writePos = [xDim, yDim - 1, run_num]

    xPos = writePos[0]
    yPos = writePos[1]

    # Figure out what direction we're heading
    headingLeft = (yDim - 1 - yPos) % 2 == 0

    for val in valsToAdd:
        if headingLeft:
            # Determine if we're at the left x edge
            if xPos == 0:
                yPos = yPos - 1
                imgArray[yPos, xPos, run_num] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos - 1
                imgArray[yPos, xPos, run_num] = val
        else:
            # Determine if we're at the right x edge
            if xPos == xDim - 1:
                yPos = yPos - 1
                imgArray[yPos, xPos, run_num] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos + 1
                imgArray[yPos, xPos, run_num] = val
    return


# %%
def data_collection(nv_sig, opti_nv_sig, coords_list, run_num, opti_interval=4):
    with labrad.connect() as cxn:
        ret_vals = data_collection_with_cxn(
            cxn, nv_sig, opti_nv_sig, coords_list, run_num, opti_interval
        )

    readout_counts_array, drift_list = ret_vals

    return readout_counts_array, drift_list


def data_collection_with_cxn(
    cxn, nv_sig, opti_nv_sig, coords_list, run_num, opti_interval=4
):
    """
    Runs a measurement where an initial pulse is pulsed on the start coords,
    then a pulse is set on the first point in the coords list, then the
    counts are recorded on the start coords. The routine steps through
    the coords list

    Here, we run each point individually, and we optimize before each point to
    ensure we're centered on the NV. The optimize function is built into the
    sequence.

    Parameters
    ----------
    cxn :
        labrad connection. See other our other python functions.
    nv_sig : dict
        dictionary containing onformation about the pulse lengths, pusle powers,
        expected count rate, nd filter, color filter, etc
    opti_nv_sig : dict
        dictionary that contains the coordinates of an NV to optimize on
        (parmaeters should include expected count rate, coords, imagine laser,
         and imaging laser duration)
    coords_list : 2D list (float)
        A list of each coordinate that we will pulse the laser at.

    Returns
    -------
    readout_counts_array : numpy.array
        2D array with the raw counts from each run for each target coordinate
        measured on the start coord.
        The first index refers to the coordinate, the secon index refers to the
        run.
    opti_coords_list : list(float)
        A list of the optimized coordinates recorded during the measurement.
        In the form of [[x,y,z],...]

    """
    tool_belt.reset_cfm(cxn)
    xyz_server = tool_belt.get_xyz_server(cxn)

    # Define paramters
    apd_indices = [0]
    drift_list = []
    # Readout array will be a list in this case. This will be a list with
    # dimensions [num_samples].
    readout_counts_list = []

    num_samples = len(coords_list)
    start_coords = nv_sig["coords"]

    init_color = tool_belt.get_registry_entry_no_cxn(
        "wavelength", ["Config", "Optics", nv_sig["initialize_laser"]]
    )
    pulse_color = tool_belt.get_registry_entry_no_cxn(
        "wavelength", ["Config", "Optics", nv_sig["CPG_laser"]]
    )
    readout_color = tool_belt.get_registry_entry_no_cxn(
        "wavelength", ["Config", "Optics", nv_sig["charge_readout_laser"]]
    )
    pulse_time = nv_sig["CPG_laser_dur"]
    initialization_time = nv_sig["initialize_dur"]
    charge_readout_time = nv_sig["charge_readout_dur"]
    charge_readout_laser_power = nv_sig["charge_readout_laser_power"]
    readout_color = tool_belt.get_registry_entry_no_cxn(
        "wavelength", ["Config", "Optics", nv_sig["charge_readout_laser"]]
    )

    # Set the charge readout (assumed to be yellow here) to the correct filter
    if "charge_readout_laser_filter" in nv_sig:
        tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")

    # movement
    xy_delay = tool_belt.get_registry_entry_no_cxn(
        "xy_small_response_delay", ["Config", "Positioning"]
    )
    z_delay = tool_belt.get_registry_entry_no_cxn("z_delay", ["Config", "Positioning"])
    step_size_x = tool_belt.get_registry_entry_no_cxn(
        "xy_incremental_step_size", ["Config", "Positioning"]
    )
    step_size_y = tool_belt.get_registry_entry_no_cxn(
        "xy_incremental_step_size", ["Config", "Positioning"]
    )
    step_size_z = tool_belt.get_registry_entry_no_cxn(
        "z_incremental_step_size", ["Config", "Positioning"]
    )
    step_size_list = [step_size_x, step_size_y, step_size_z]

    # determine max num_steps between NV and each coord
    num_steps_list = []
    for i in range(len(coords_list)):
        # x
        diff = abs(start_coords[0] - coords_list[i][0])
        num_steps_list.append(numpy.ceil(diff / step_size_x))
        # y
        diff = abs(start_coords[1] - coords_list[i][1])
        num_steps_list.append(numpy.ceil(diff / step_size_y))
        # z
        diff = abs(start_coords[2] - coords_list[i][2])
        num_steps_list.append(numpy.ceil(diff / step_size_z))

    movement_incr = int(max(num_steps_list))

    if xy_delay > z_delay:
        movement_delay = xy_delay
    else:
        movement_delay = z_delay

    # define the sequence paramters
    file_name = "SPaCE_w_movement_steps.py"
    seq_args = [
        initialization_time,
        pulse_time,
        charge_readout_time,
        movement_delay,
        charge_readout_laser_power,
        apd_indices[0],
        init_color,
        pulse_color,
        readout_color,
        movement_incr,
    ]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    # print(seq_args)
    # return

    # print the expected run time
    period = ret_vals[0]
    period_s = period / 10**9
    period_s_total = period_s * num_samples + 1
    period_m_total = period_s_total / 60
    print("{} ms pulse time".format(pulse_time / 10**6))
    print("Expected run time for set of points: {:.1f} m".format(period_m_total))
    # return
    tool_belt.init_safe_stop()

    if period_m_total > opti_interval:
        num_optimize = int(numpy.ceil(period_m_total / opti_interval))
        redux_num_samples = int(numpy.floor(num_samples / num_optimize))
        remain_num_samples = int(num_samples % num_optimize)
        i = 0
        while i < num_optimize:
            redux_coords_list = coords_list[
                i * redux_num_samples : (i + 1) * redux_num_samples
            ]
            targeting.main_with_cxn(cxn, opti_nv_sig, apd_indices)
            drift_list.append(tool_belt.get_drift())

            ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)

            drift = numpy.array(tool_belt.get_drift())

            # get the readout coords with drift
            start_coords_drift = start_coords + drift
            coords_list_drift = numpy.array(redux_coords_list) + drift

            # Build the list to step through the coords on readout NV and targets
            x_voltages, y_voltages, z_voltages = build_voltages_from_list_xyz(
                start_coords_drift, coords_list_drift, movement_incr, step_size_list
            )

            # Load the galvo
            xyz_server = tool_belt.get_xyz_server(cxn)
            xyz_server.load_arb_scan_xyz(
                x_voltages, y_voltages, z_voltages, int(period)
            )

            # We'll be lookign for three samples each repetition with how I have
            # the sequence set up
            total_num_samples = (2 * movement_incr + 1) * redux_num_samples
            readout_counts = collect_counts(
                cxn, movement_incr, total_num_samples, seq_args_string, apd_indices
            )

            readout_counts_list.append(readout_counts)
            i += 1
        # then perform final measurements on points that were remainders when dividing up the num_samples
        if remain_num_samples != 0:
            compl_num_samples = num_optimize * redux_num_samples
            remain_coords_list = coords_list[
                compl_num_samples : compl_num_samples + remain_num_samples
            ]

            targeting.main_with_cxn(cxn, opti_nv_sig, apd_indices)
            drift_list.append(tool_belt.get_drift())

            ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)

            drift = numpy.array(tool_belt.get_drift())

            # get the readout coords with drift
            start_coords_drift = start_coords + drift
            coords_list_drift = numpy.array(remain_coords_list) + drift

            # Build the list to step through the coords on readout NV and targets
            x_voltages, y_voltages, z_voltages = build_voltages_from_list_xyz(
                start_coords_drift, coords_list_drift, movement_incr, step_size_list
            )
            # Load the galvo
            xyz_server = tool_belt.get_xyz_server(cxn)
            xyz_server.load_arb_scan_xyz(
                x_voltages, y_voltages, z_voltages, int(period)
            )

            # We'll be lookign for three samples each repetition with how I have
            # the sequence set up
            total_num_samples = (2 * movement_incr + 1) * remain_num_samples
            readout_counts = collect_counts(
                cxn, movement_incr, total_num_samples, seq_args_string, apd_indices
            )

            readout_counts_list.append(readout_counts)
    else:
        # the whole sequence will tkae less time than the intervals between
        # optimize so just run it all at once
        targeting.main_with_cxn(cxn, opti_nv_sig, apd_indices)
        drift_list.append(tool_belt.get_drift())

        drift = numpy.array(tool_belt.get_drift())

        # get the readout coords with drift
        start_coords_drift = start_coords + drift
        coords_list_drift = numpy.array(coords_list) + drift
        # Build the list to step through the coords on readout NV and targets
        x_voltages, y_voltages, z_voltages = build_voltages_from_list_xyz(
            start_coords_drift, coords_list_drift, movement_incr, step_size_list
        )
        # Load the galvo
        xyz_server = tool_belt.get_xyz_server(cxn)
        xyz_server.load_arb_scan_xyz(x_voltages, y_voltages, z_voltages, int(period))

        # We'll be lookign for three samples each repetition with how I have
        # the sequence set up
        total_num_samples = (2 * movement_incr + 1) * num_samples
        readout_counts = collect_counts(
            cxn, movement_incr, total_num_samples, seq_args_string, apd_indices
        )
        readout_counts_list.append(readout_counts)

    return list(numpy.concatenate(readout_counts_list).flat), drift_list


# %%
def main(
    nv_sig,
    opti_nv_sig,
    num_runs,
    num_steps_a,
    num_steps_b=None,
    charge_state_threshold=None,
    img_range_2D=None,
    radial_lengths=None,
    offset_2D=[0, 0, 0],
):
    """
    A measurements to initialize on a single point, then pulse a laser off that
    point, and then read out the charge state on the single point.

    Parameters
    ----------
    nv_sig : dict
        dictionary specific to the nv, which contains parameters liek the
        scc readout length, the yellow readout power, the expected count rate...\
    opti_nv_sig : dict
        dictionary that contains the coordinates of an NV to optimize on 
        (parmaeters should include expected count rate, coords, imagine laser, 
         and imaging laser duration)
    img_range: list (2D)
    
        
    num_steps: int
        number of steps in 1 direction. For 2D image, this is squared for the
        total number of points
    num_runs: int
        the number of repetitions of the same measurement, which will be averaged over
    measurement_type: '1D' or '2D'
        either run a 2D oving target or 1D moving target measurement.
    """
    direction_labels = ["x", "y", "z"]
    axes = []
    direction_title = []

    # Record start time of the measurement
    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    init_color = tool_belt.get_registry_entry_no_cxn(
        "wavelength", ["Config", "Optics", nv_sig["initialize_laser"]]
    )
    pulse_color = tool_belt.get_registry_entry_no_cxn(
        "wavelength", ["Config", "Optics", nv_sig["CPG_laser"]]
    )
    pulse_time = nv_sig["CPG_laser_dur"]

    xy_scale = tool_belt.get_registry_entry_no_cxn(
        "xy_nm_per_unit", ["", "Config", "Positioning"]
    )
    z_scale = tool_belt.get_registry_entry_no_cxn(
        "z_nm_per_unit", ["", "Config", "Positioning"]
    )
    scale_list = [xy_scale / 1e3, xy_scale / 1e3, z_scale / 1e3]

    opti_interval = 4  # min

    if not num_steps_b:
        num_steps_b = num_steps_a

    start_coords = nv_sig["coords"]

    if img_range_2D != None:
        measurement_type = "2D"
        for v in range(len(img_range_2D)):
            val = img_range_2D[v]
            if val != 0:
                direction_title.append(direction_labels[v])
                axes.append(v)
            else:
                stationary_axis = v

        # calculate the list of x and y voltages we'll need to step through
        ret_vals = build_voltages_image(
            start_coords, img_range_2D, axes, num_steps_a, num_steps_b
        )
        a_voltages, b_voltages, a_voltages_1d, b_voltages_1d = ret_vals

        # list the values of the axis that won't move
        c_voltages = numpy.linspace(
            start_coords[stationary_axis],
            start_coords[stationary_axis],
            len(a_voltages),
        )

        # sort which voltage lists go to which axes
        voltage_list = [[], [], []]
        voltage_list[axes[0]] = numpy.array(a_voltages) + offset_2D[axes[0]]
        # print(offset_2D[axes[1]])
        # print((numpy.array(b_voltages) - start_coords[axes[1]])*35)
        # return
        voltage_list[axes[1]] = numpy.array(b_voltages) + offset_2D[axes[1]]
        voltage_list[stationary_axis] = c_voltages + offset_2D[stationary_axis]

        # Determine which coordinated fall within the annulus
        a_0 = a_voltages_1d[int(num_steps_a / 2)]
        b_0 = b_voltages_1d[int(num_steps_b / 2)]
        a_ind = []
        b_ind = []
        for a in range(len(voltage_list[axes[0]])):
            for b in range(len(voltage_list[axes[1]])):
                r = numpy.sqrt(
                    (voltage_list[axes[0]] - a_0) ** 2
                    + (voltage_list[axes[1]] - b_0) ** 2
                )
                if r <= radial_lengths[1] and r > radial_lengths[0]:
                    a_ind.append(a)
                    b_ind.append(b)

        # Combine the x and y voltages together into pairs
        coords_voltages = list(zip(voltage_list[0], voltage_list[1], voltage_list[2]))

        # prep for the figure
        half_range_a = img_range_2D[axes[0]] / 2
        half_range_b = img_range_2D[axes[1]] / 2
        a_low = -half_range_a
        a_high = half_range_a
        b_low = -half_range_b
        b_high = half_range_b
        if axes[0] == 2:
            a_low -= offset_2D[axes[0]]
            a_high -= offset_2D[axes[0]]
        elif axes[1] == 2:
            b_low -= offset_2D[axes[1]]
            b_high -= offset_2D[axes[1]]

        # a_low = -half_range_a + offset_2D[axes[0]]
        # a_high = half_range_a + offset_2D[axes[0]]
        # b_low = -half_range_b + offset_2D[axes[1]]
        # b_high = half_range_b + offset_2D[axes[1]]

        pixel_size_a = a_voltages_1d[1] - a_voltages_1d[0]
        pixel_size_b = b_voltages_1d[1] - b_voltages_1d[0]

        half_pixel_size_a = pixel_size_a / 2
        half_pixel_size_b = pixel_size_b / 2

        img_extent = [
            (a_high + half_pixel_size_a) * scale_list[axes[0]],
            (a_low - half_pixel_size_a) * scale_list[axes[0]],
            (b_low - half_pixel_size_b) * scale_list[axes[1]],
            (b_high + half_pixel_size_b) * scale_list[axes[1]],
        ]

        # Create some empty data lists

        readout_image_array = numpy.empty([num_steps_a, num_steps_b])
        readout_image_array[:] = numpy.nan
        # aspect_r = None
        # if 2 in axes:
        #     aspect_r = "auto"
        # Create the figure
        title = "SPaCE {} vs {} - {} nm init pulse \n{} nm {} ms CPG pulse".format(
            direction_title[0],
            direction_title[1],
            init_color,
            pulse_color,
            pulse_time / 10**6,
        )
        fig_2D = tool_belt.create_image_figure(
            readout_image_array,
            numpy.array(img_extent),
            title=title,
            um_scaled=True,
            aspect_ratio=None,
        )

    drift_list_master = []
    num_samples = len(coords_voltages)
    readout_counts_array = numpy.empty([num_samples, num_runs])

    for n in range(num_runs):
        print("Run {}".format(n))
        # shuffle the voltages that we're stepping thru
        ind_list = list(range(num_samples))
        shuffle(ind_list)

        # shuffle the voltages to run
        coords_voltages_shuffle = []
        for i in ind_list:
            coords_voltages_shuffle.append(coords_voltages[i])
        coords_voltages_shuffle_list = [list(el) for el in coords_voltages_shuffle]

        # ========================== Run the data collection====================#
        ret_vals = data_collection(
            nv_sig, opti_nv_sig, coords_voltages_shuffle_list, n, opti_interval
        )

        readout_counts_list_shfl, drift = ret_vals
        drift_list_master.append(drift)
        readout_counts_list_shfl = numpy.array(readout_counts_list_shfl)
        # unshuffle the raw data
        list_ind = 0
        for f in ind_list:
            readout_counts_array[f][n] = readout_counts_list_shfl[list_ind]
            list_ind += 1

        if type(drift_list_master) != list:
            drift_list_master = drift_list_master.tolist()

        # If a threshold for charge state readout is passed, take each value and assign either 1 or 0
        if charge_state_threshold != None:
            readout_counts_array_charge = readout_counts_array
            for r in range(len(readout_counts_array)):
                row = readout_counts_array[r]
                for c in range(len(row)):
                    current_val = readout_counts_array[r][c]
                    if current_val < charge_state_threshold:
                        set_val = 0
                    elif current_val >= charge_state_threshold:
                        set_val = 1
                    readout_counts_array_charge[r][c] = set_val

            # Need to rotate the matrix, to then only
            # average the runs that have been completed
            readout_counts_array_rot = numpy.rot90(readout_counts_array_charge)
        else:
            # if no threshold, just take the counts and rotate the matrix
            readout_counts_array_rot = numpy.rot90(readout_counts_array)

        # Take the average and ste.
        readout_counts_avg = numpy.average(readout_counts_array_rot[-(n + 1) :], axis=0)
        readout_counts_ste = stats.sem(readout_counts_array_rot[-(n + 1) :], axis=0)
        # Save incrementally
        raw_data = {
            "timestamp": start_timestamp,
            "img_range_2D": img_range_2D,
            "img_range-units": "V",
            "offset_2D": offset_2D,
            "direction_title": direction_title,
            "num_steps_a": num_steps_a,
            "num_steps_b": num_steps_b,
            "num_runs": num_runs,
            "opti_interval": opti_interval,
            "opti_interval-units": "s",
            "nv_sig": nv_sig,
            "nv_sig-units": tool_belt.get_nv_sig_units(),
            "coords_voltages": coords_voltages,
            "coords_voltages-units": "[V, V]",
            "ind_list": ind_list,
            "drift_list_master": drift_list_master,
            "readout_counts_array": readout_counts_array.tolist(),
            "readout_counts_array-units": "counts",
            "readout_counts_avg": readout_counts_avg.tolist(),
            "readout_counts_avg-units": "counts",
            "readout_counts_ste": readout_counts_ste.tolist(),
            "readout_counts_ste-units": "counts",
        }
        if charge_state_threshold != None:
            raw_data["charge_state_threshold"] = charge_state_threshold
            raw_data[
                "readout_counts_array_charge"
            ] = readout_counts_array_charge.tolist()
            raw_data["readout_counts_array_charge-units"] = "counts"

        file_path = tool_belt.get_file_path(
            __file__, start_timestamp, nv_sig["name"], "incremental"
        )

        if measurement_type == "2D":
            # create the img arrays
            # writePos = []
            # create image array from list of  readout counts
            split_counts = numpy.split(readout_counts_avg, num_steps_b)
            readout_image_array = numpy.vstack(split_counts)
            r = 0
            for i in reversed(range(len(readout_image_array))):
                if r % 2 == 0:
                    readout_image_array[i] = list(reversed(readout_image_array[i]))
                r += 1

            # readout_image_array = image_sample.populate_img_array(readout_counts_avg, readout_image_array, writePos)

            tool_belt.update_image_figure(fig_2D, readout_image_array)

            raw_data["a_voltages_1d"] = a_voltages_1d.tolist()
            raw_data["b_voltages_1d"] = b_voltages_1d.tolist()
            raw_data["img_extent"] = img_extent
            raw_data["img_extent-units"] = "V"
            raw_data["readout_image_array"] = readout_image_array.tolist()
            raw_data["readout_image_array-units"] = "counts"

            tool_belt.save_figure(fig_2D, file_path)

        tool_belt.save_raw_data(raw_data, file_path)

    endFunctionTime = time.time()

    # Save
    timeElapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "timeElapsed": timeElapsed,
        "img_range_2D": img_range_2D,
        "img_range-units": "V",
        "offset_2D": offset_2D,
        "direction_title": direction_title,
        "num_steps_a": num_steps_a,
        "num_steps_b": num_steps_b,
        "num_runs": num_runs,
        "opti_interval": opti_interval,
        "opti_interval-units": "s",
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "coords_voltages": coords_voltages,
        "coords_voltages-units": "[V, V]",
        "ind_list": ind_list,
        "drift_list_master": drift_list_master,
        "readout_counts_array": readout_counts_array.tolist(),
        "readout_counts_array-units": "counts",
        "readout_counts_avg": readout_counts_avg.tolist(),
        "readout_counts_avg-units": "counts",
        "readout_counts_ste": readout_counts_ste.tolist(),
        "readout_counts_ste-units": "counts",
    }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])

    if charge_state_threshold != None:
        raw_data["charge_state_threshold"] = charge_state_threshold
        raw_data["readout_counts_array_charge"] = readout_counts_array_charge.tolist()
        raw_data["readout_counts_array_charge-units"] = "counts"

    if measurement_type == "2D":
        raw_data["a_voltages_1d"] = a_voltages_1d.tolist()
        raw_data["b_voltages_1d"] = b_voltages_1d.tolist()
        raw_data["img_extent"] = img_extent
        raw_data["img_extent-units"] = "V"
        raw_data["readout_image_array"] = readout_image_array.tolist()
        raw_data["readout_counts_array-units"] = "counts"

        tool_belt.save_figure(fig_2D, file_path)

    tool_belt.save_raw_data(raw_data, file_path)

    return


# %% Run the files

if __name__ == "__main__":
    path_august = "pc_rabi/branch_master/SPaCE/2021_08"
    path = "pc_rabi/branch_master/SPaCE/2021_09"

    file_name_no_opt = "2021_07_27-15_10_17-johnson-nv1_2021_07_27"
    file_name_no_opt_2 = "2021_07_28-15_21_25-johnson-nv1_2021_07_27"
    file_name_no_opt_dim = "2021_08_03-15_13_57-johnson-nv1_2021_07_27"
    file_name_opt = "2021_07_27-19_03_17-johnson-nv1_2021_07_27"
    file_name_opt_xy = "2021_07_28-14_00_23-johnson-nv1_2021_07_27"
    file_name_opt_mod = "2021_08_03-12_54_18-johnson-nv1_2021_07_27"
    file_name_opt_dim = "2021_08_03-14_48_39-johnson-nv1_2021_07_27"

    file_name_no_opt_400_ms = "2021_08_03-16_21_50-johnson-nv1_2021_07_27"
    file_name_opt_xy_2 = "2021_08_03-18_22_18-johnson-nv1_2021_07_27"

    file = "2021_07_31-18_02_29-johnson-nv1_2021_07_27"

    # ================ 7/27/2021 x and y scans @ 22 mW ================#
    # file_list = ['2021_07_27-04_19_32-johnson-nv1_2021_07_21', # x
    #               '2021_07_27-01_57_21-johnson-nv1_2021_07_21',
    #               '2021_07_26-23_35_05-johnson-nv1_2021_07_21',
    #               '2021_07_26-21_12_36-johnson-nv1_2021_07_21']

    # file_list = ['2021_07_27-05_30_36-johnson-nv1_2021_07_21', # y
    #               '2021_07_27-03_08_29-johnson-nv1_2021_07_21',
    #               '2021_07_27-00_46_16-johnson-nv1_2021_07_21',
    #               '2021_07_26-22_23_54-johnson-nv1_2021_07_21']

    # ================ 7/28/2021 x and y scans @ 33 mW ================#
    # x
    # file_list = ['2021_07_28-00_46_22-johnson-nv1_2021_07_27',
    # '2021_07_28-03_08_32-johnson-nv1_2021_07_27',
    # '2021_07_28-05_30_52-johnson-nv1_2021_07_27',
    # '2021_07_28-07_54_25-johnson-nv1_2021_07_27',
    # '2021_07_27-21_13_36-johnson-nv1_2021_07_27'
    # ]

    # y
    # file_list = ['2021_07_28-01_57_24-johnson-nv1_2021_07_27',
    #              '2021_07_28-04_19_38-johnson-nv1_2021_07_27',
    #              '2021_07_28-06_42_07-johnson-nv1_2021_07_27',
    #              '2021_07_28-09_06_42-johnson-nv1_2021_07_27',
    #              '2021_07_27-22_39_56-johnson-nv1_2021_07_27']

    # ================ 8/2/2021 x and y scans @ 22 mW ================#
    # x
    # file_list = [
    #             '2021_07_30-18_15_21-johnson-nv1_2021_07_27',
    #                '2021_07_30-20_39_48-johnson-nv1_2021_07_27',
    #                '2021_07_31-10_49_04-johnson-nv1_2021_07_27',
    #               '2021_07_31-23_41_32-johnson-nv1_2021_07_27',
    #               '2021_07_31-13_13_29-johnson-nv1_2021_07_27',
    #               '2021_07_31-15_37_56-johnson-nv1_2021_07_27',
    #               '2021_07_31-18_02_29-johnson-nv1_2021_07_27',
    #               '2021_08_01-02_06_04-johnson-nv1_2021_07_27',
    #               '2021_08_01-04_30_39-johnson-nv1_2021_07_27',
    #               '2021_08_01-06_55_19-johnson-nv1_2021_07_27',
    #               '2021_08_01-09_20_02-johnson-nv1_2021_07_27',
    #               '2021_08_01-11_44_49-johnson-nv1_2021_07_27'
    #                ]

    # y
    # file_list = [
    #             '2021_07_30-19_27_35-johnson-nv1_2021_07_27',
    #                '2021_07_30-21_52_01-johnson-nv1_2021_07_27',
    #                '2021_07_31-12_01_16-johnson-nv1_2021_07_27',
    #                '2021_08_01-00_53_48-johnson-nv1_2021_07_27',
    #                '2021_07_31-14_25_41-johnson-nv1_2021_07_27',
    #                '2021_07_31-16_50_12-johnson-nv1_2021_07_27',
    #                '2021_07_31-19_14_50-johnson-nv1_2021_07_27',
    #                '2021_08_01-03_18_22-johnson-nv1_2021_07_27',
    #                '2021_08_01-05_42_58-johnson-nv1_2021_07_27',
    #                '2021_08_01-08_07_40-johnson-nv1_2021_07_27',
    #                '2021_08_01-10_32_24-johnson-nv1_2021_07_27',
    #                '2021_08_01-12_57_13-johnson-nv1_2021_07_27',
    #             ]

    # ================ 8/2/2021 x scans @ 800 us ================#

    # x
    # file_list = [
    #     '2021_08_02-14_21_03-johnson-nv1_2021_07_27',
    #     '2021_08_01-06_55_19-johnson-nv1_2021_07_27',
    #     '2021_08_01-23_52_30-johnson-nv1_2021_07_27',
    #     '2021_08_01-14_39_32-johnson-nv1_2021_07_27'
    #     ]

    # ================ 8/2/2021 y scans @ 1000 us ================#

    # y
    # file_list = [
    #     '2021_08_01-12_57_13-johnson-nv1_2021_07_27',
    #     '2021_08_02-03_29_37-johnson-nv1_2021_07_27',
    #     '2021_08_01-18_16_42-johnson-nv1_2021_07_27'
    #     ]
    # ================ 8/5/2021 x scans @ 22 mW Feature 1A ================#
    # x
    # file_list = [
    # '2021_08_04-20_38_12-johnson-nv2_2021_08_04',
    # '2021_08_04-21_29_32-johnson-nv2_2021_08_04',
    # '2021_08_04-21_55_15-johnson-nv2_2021_08_04',
    # '2021_08_04-22_20_56-johnson-nv2_2021_08_04',
    # '2021_08_04-22_46_41-johnson-nv2_2021_08_04',
    # '2021_08_04-23_12_20-johnson-nv2_2021_08_04',
    # '2021_08_04-23_38_01-johnson-nv2_2021_08_04',
    # '2021_08_05-00_03_43-johnson-nv2_2021_08_04',
    # '2021_08_05-00_29_24-johnson-nv2_2021_08_04',
    # '2021_08_05-17_59_06-johnson-nv2_2021_08_04',
    # '2021_08_05-00_55_06-johnson-nv2_2021_08_04',
    # '2021_08_05-01_20_49-johnson-nv2_2021_08_04',
    # '2021_08_05-01_46_33-johnson-nv2_2021_08_04',
    # '2021_08_05-02_12_17-johnson-nv2_2021_08_04',
    # '2021_08_05-02_38_00-johnson-nv2_2021_08_04',
    # '2021_08_05-03_03_43-johnson-nv2_2021_08_04',
    # '2021_08_05-03_29_28-johnson-nv2_2021_08_04',
    # '2021_08_05-03_55_11-johnson-nv2_2021_08_04',
    # '2021_08_05-09_01_01-johnson-nv2_2021_08_04',
    # '2021_08_05-09_26_52-johnson-nv2_2021_08_04',
    # '2021_08_05-09_52_38-johnson-nv2_2021_08_04',
    # '2021_08_05-10_45_26-johnson-nv2_2021_08_04',
    # '2021_08_05-11_43_11-johnson-nv2_2021_08_04',
    # '2021_08_05-12_12_58-johnson-nv2_2021_08_04',
    # '2021_08_05-15_54_52-johnson-nv2_2021_08_04'
    # ]

    # ================ 8/6/2021 x scans @ 500 us Feature 1A ================#
    # x
    # file_list = [
    # '2021_08_05-23_11_42-johnson-nv2_2021_08_04',
    # '2021_08_05-22_46_43-johnson-nv2_2021_08_04',
    # '2021_08_05-22_25_21-johnson-nv2_2021_08_04',
    # '2021_08_05-17_59_06-johnson-nv2_2021_08_04',
    # '2021_08_05-20_06_21-johnson-nv2_2021_08_04',
    # '2021_08_05-20_55_56-johnson-nv2_2021_08_04',
    # '2021_08_05-21_17_41-johnson-nv2_2021_08_04',
    # '2021_08_05-21_41_06-johnson-nv2_2021_08_04',
    # '2021_08_05-22_01_58-johnson-nv2_2021_08_04'
    # ]

    # ================ 8/8/2021 x scans @ 22 mW Feature 1A, changing time between optimize ================#

    # file_list = [
    #     '2021_08_06-22_42_26-johnson-nv2_2021_08_04',
    #     '2021_08_06-21_23_20-johnson-nv2_2021_08_04',
    #     '2021_08_06-18_01_46-johnson-nv2_2021_08_04',
    #     '2021_08_07-13_42_12-johnson-nv2_2021_08_04',
    #     '2021_08_07-14_53_25-johnson-nv2_2021_08_04',
    #     '2021_08_07-20_57_08-johnson-nv2_2021_08_04',
    #     '2021_08_07-21_55_58-johnson-nv2_2021_08_04',
    #     '2021_08_06-23_50_50-johnson-nv2_2021_08_04',
    #     '2021_08_08-12_40_24-johnson-nv2_2021_08_04',
    #     '2021_08_07-22_57_03-johnson-nv2_2021_08_04',
    #     '2021_08_08-11_40_13-johnson-nv2_2021_08_04',
    #     '2021_08_07-12_14_46-johnson-nv2_2021_08_04',
    #     '2021_08_07-23_57_53-johnson-nv2_2021_08_04'
    #     ]

    # ================ 8/31/2021  ================#

    file_list = [
        "2021_09_01-04_36_52-johnson-nv2_2021_08_30",
        "2021_09_01-05_00_07-johnson-nv2_2021_08_30",
        "2021_09_01-05_23_28-johnson-nv2_2021_08_30",
        "2021_09_01-06_13_43-johnson-nv2_2021_08_30",
        "2021_09_01-07_16_36-johnson-nv2_2021_08_30",
        "2021_09_01-08_32_02-johnson-nv2_2021_08_30",
    ]
    # ================ 9/1/2021  ================#

    file_list = [
        "2021_08_31-19_37_16-johnson-nv2_2021_08_30",
        "2021_08_31-21_39_38-johnson-nv2_2021_08_30",
        "2021_09_01-11_10_15-johnson-nv2_2021_08_30",
    ]

    ########### Fit Gaussian to 1D files ###########
    # widths_master_list = []
    # center_master_list = []
    # height_master_list = []
    # path_list = [path_august,path_august, path]
    # for f in range(len(file_list)):
    #     file_name = file_list[f]
    #     path = path_list[f]
    #     lobe_positions = [975]
    #     ret_vals = plot_1D_SpaCE(file_name, path, do_plot = True, do_fit = True)
    #     widths_master_list.append(ret_vals[2][2])
    #     center_master_list.append(ret_vals[2][1])
    #     height_master_list.append(ret_vals[2][0]**2)

    # #_______ Power _______#
    # x_vals = [ 12.7, 15.8, 18.6, 21.7, 24.4, 26.8, 29.1, 30.9, 31.3]
    # lin_x_vals = numpy.linspace(x_vals[0],
    #                 x_vals[-1], 100)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.plot(x_vals, height_master_list, 'bo')
    # ax.set_xlabel('Pulse power (mW)')
    # ax.set_ylabel('Gaussian amplitude (counts)')
    # ax.set_title('8/6/2021 500 us, x axis, 1A lobe')

    # init_fit = [5, 20]
    # opti_params, cov_arr = curve_fit(exp_sqrd_decay,
    #       x_vals,height_master_list,p0=init_fit)
    # print('Opti params: ' + str(opti_params))
    # text = r'$A e^{-P^2/d}$'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.85, 0.95, text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)
    # ax.plot(lin_x_vals, exp_sqrd_decay(lin_x_vals, *opti_params), 'r-')
    # text = 'A={:.3f} counts\n$d$={:.3f} mW^2'.format(*opti_params)
    # ax.text(0.3, 0.1, text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x_vals, widths_master_list, 'bo')
    # ax.set_xlabel('Pulse power (mW)')
    # ax.set_ylabel('Gaussian sigma (nm)')
    # ax.set_title('8/6/2021 500 us x axis, 1A lobe')

    # init_fit = [300, 10]
    # opti_params, cov_arr = curve_fit(inverse_sqrt,
    #       x_vals,widths_master_list,p0=init_fit)
    # print('Opti params: ' + str(opti_params))
    # ax.plot(lin_x_vals, inverse_sqrt(lin_x_vals, *opti_params), 'g-',
    #         label = "fit inverse power: t ^ -0.372")
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # # ax.legend()

    # text = r'$A / P^{-1/2} + o$'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.85, 0.95, text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)
    # text = 'A={:.3f} nm*mw^-1/2\no = {:.1f} nm'.format(*opti_params)
    # ax.text(0.3, 0.1, text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)

    # init_fit = [750]
    # opti_params, cov_arr = curve_fit(inverse_law,
    #       x_vals,widths_master_list,p0=init_fit)
    # print('Opti params: ' + str(opti_params))
    # text = r'$A / P$'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.85, 0.95, text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)
    # ax.plot(lin_x_vals, inverse_law(lin_x_vals, *opti_params), 'r-')
    # text = 'A={:.3f} nm*mW'.format(*init_fit)
    # ax.text(0.3, 0.1, text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.plot(x_vals, center_master_list, 'bo')
    # ax.set_xlabel('Pulse power (mW)')
    # ax.set_ylabel('Lobe position (nm)')
    # ax.set_title('8/6/2021 500 us, x axis, 1A lobe')

    # _______ Time _______#
    # x_vals = [
    #     100, 150, 1000
    #           ]
    # lin_x_vals = numpy.linspace(x_vals[0],
    #                 x_vals[-1], 100)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x_vals, widths_master_list, 'bo')
    # ax.set_xlabel('Pulse duration (us)')
    # ax.set_ylabel('Sq Gaussian sigma (nm)')
    # ax.set_title('8/5/2021 22 mW, x axis, 1A lobe')
    # ax.set_yscale('log')
    # ax.set_xscale('log')

    # init_fit = [300, -1/4]
    # opti_params, cov_arr = curve_fit(power_fnct,
    #       x_vals,widths_master_list,p0=init_fit)
    # print('Opti params: ' + str(opti_params))
    # ax.plot(lin_x_vals, power_fnct(lin_x_vals, *opti_params), 'g-',
    #         label = "inverse square root")

    # init_fit = [300]
    # opti_params, cov_arr = curve_fit(inverse_quarter,
    #       x_vals,widths_master_list,p0=init_fit)
    # print('Opti params: ' + str(opti_params))
    # ax.plot(lin_x_vals, inverse_quarter(lin_x_vals, *opti_params), 'g-',)
    # ax.legend()

    # text = r'$A / t^{1/4}$'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.85, 0.95, text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)
    # text = 'A={:.3f} nm*us^0.25'.format(*opti_params)
    # ax.text(0.3, 0.1, text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)

    ############# Plot 1D comparisons ##############
    # rad_dist, counts_no_opt = plot_1D_SpaCE(file_name_no_opt_400_ms, path, do_plot = False)
    # rad_dist, counts_opt = plot_1D_SpaCE(file_name_opt_xy_2, path, do_plot = False)

    # rad_dist, counts = plot_1D_SpaCE(file, path_july, do_plot = False)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # # ax.plot(rad_dist, counts, 'b-', label = 'Optimize every run')
    # ax.plot(rad_dist, counts_no_opt, 'b-', label = 'Optimize every run, add 400 ms of green pusle every point')
    # ax.plot(rad_dist, counts_opt, 'r-', label = 'Optimize every point in x and y')
    # ax.set_xlabel('y (nm)')
    # ax.set_ylabel('Average counts')
    # ax.legend()

    # ================ specific for 1D scans ================#
    # file = '2021_08_27-08_18_02-johnson-nv1_2021_08_26'
    # data = tool_belt.get_raw_data(file, path)

    # # img_array = data['readout_image_array']
    # readout_counts_avg = data['readout_counts_avg']
    # nv_sig = data['nv_sig']
    # start_coords = nv_sig['coords']
    # num_steps = data['num_steps']
    # img_range = data['img_range']
    # dz = data['dz']
    # pulse_time = nv_sig['CPG_laser_dur']

    # dx_list =[img_range[0][0], img_range[1][0]]
    # dy_list =[img_range[0][1], img_range[1][1]]
    # low_coords = numpy.array(start_coords) + [dx_list[0], dy_list[0], dz]
    # high_coords = numpy.array(start_coords) + [dx_list[1], dy_list[1], dz]
    # x_voltages = numpy.linspace(low_coords[0],
    #                             high_coords[0], num_steps)
    # y_voltages = numpy.linspace(low_coords[1],
    #                             high_coords[1], num_steps)
    # z_voltages = numpy.linspace(low_coords[2],
    #                             high_coords[2], num_steps)

    # rad_dist = numpy.sqrt((x_voltages - start_coords[0])**2 +( y_voltages - start_coords[1])**2)

    # neg_ints = int(numpy.floor(len(rad_dist)/2))
    # rad_dist[0:neg_ints] = rad_dist[0:neg_ints]*-1

    # fig_1D, ax_1D = plt.subplots(1, 1, figsize=(10, 10))
    # ax_1D.plot(rad_dist*35000,readout_counts_avg, label = nv_sig['name'])
    # ax_1D.set_xlabel('r (nm)')
    # ax_1D.set_ylabel('Average counts')
    # ax_1D.set_title('SPaCE - {} nm init pulse \n{} nm {} ms CPG pulse'.\
    #                                 format(638, 515, pulse_time/10**6,))

    # ================ specific for 2D scans ================#
    file_list = ["2021_09_09-00_41_38-johnson-nv1_2021_09_07"]

    for f in range(len(file_list)):
        file = file_list[f]
        plot_2D_space(file, path)

    ############# Create csv filefor 2D image ##############
    # csv_filename = '{}_{}-us'.format(timestamp,int( CPG_pulse_dur/10**3))

    # tool_belt.save_image_data_csv(img_array, x_voltages, y_voltages, path,
    #                               csv_filename)

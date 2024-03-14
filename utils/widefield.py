0  # -*- coding: utf-8 -*-
"""Various utility functions for widefield imaging and camera data processing

Created on August 15th, 2023

@author: mccambria
"""

# region Imports and constants

import itertools
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy import inf

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CollectionMode, CountFormat, LaserKey, LaserPosMode

# endregion
# region Image processing


def integrate_counts_from_adus(img_array, pixel_coords, radius=None):
    img_array_photons = adus_to_photons(img_array)
    return integrate_counts(img_array_photons, pixel_coords, radius)


def integrate_counts(img_array, pixel_coords, radius=None):
    """Add up the counts around a target set of pixel coordinates in the passed image array.
    Use for getting the total number of photons coming from a target NV.

    Parameters
    ----------
    img_array : ndarray
        Image array in units of photons (convert from ADUs with adus_to_photons)
    pixel_coords : 2-tuple
        Pixel coordinates to integrate around
    radius : _type_, optional
        Radius of disk to integrate over, by default retrieved from config

    Returns
    -------
    float
        Integrated counts (just an estimate, as adus_to_photons is also just an estimate)
    """
    pixel_x = pixel_coords[0]
    pixel_y = pixel_coords[1]

    if radius is None:
        radius = _get_camera_spot_radius()

    # Don't work through all the pixels, just the ones that might be relevant
    left = int(np.floor(pixel_x - radius))
    right = int(np.ceil(pixel_x + radius))
    top = int(np.floor(pixel_y - radius))
    bottom = int(np.ceil(pixel_y + radius))
    x_crop = list(range(left, right + 1))
    y_crop = list(range(top, bottom + 1))
    x_crop_mesh, y_crop_mesh = np.meshgrid(x_crop, y_crop)
    dist = np.sqrt((x_crop_mesh - pixel_x) ** 2 + (y_crop_mesh - pixel_y) ** 2)
    img_array_crop = img_array[top : bottom + 1, left : right + 1]

    counts = np.sum(img_array_crop, where=dist < radius)
    return counts


def adus_to_photons(adus, k_gain=None, em_gain=None, bias_clamp=None):
    """Convert camera analog-to-digital converter units (ADUs) to an
    estimated number of photons. Since the gain stages are noisy, this is
    just an estimate

    Parameters
    ----------
    adus : numeric
        Quantity to convert in ADUs
    k_gain : numeric, optional
        k gain of the camera in e- / ADU, by default retrieved from config
    em_gain : numeric, optional
        Electron-multiplying gain of the camera, by default retrieved from config
    bias_clamp : numeric, optional
        Bias clamp level, i.e. the ADU value for a pixel which receives no light. Used to
        ensure the camera does not return negative values. By default retrieved from config

    Returns
    -------
    numeric
        Quantity converted to photons
    """
    if k_gain is None:
        k_gain = _get_camera_k_gain()
    if em_gain is None:
        em_gain = _get_camera_em_gain()
    if bias_clamp is None:
        bias_clamp = _get_camera_bias_clamp()

    total_gain = k_gain / em_gain
    photons = (adus - bias_clamp) * total_gain
    return photons


def img_str_to_array(img_str):
    """Convert an img_array from a uint16-valued byte string (returned by the camera
    labrad server for speed) into a usable int-valued 2D array

    Parameters
    ----------
    img_str : byte string
        Image array as a byte string - the return value of a camera.read() call

    Returns
    -------
    ndarray
        Image array contructed from the byte string
    """
    shape = get_img_array_shape()
    img_array = np.frombuffer(img_str, dtype=np.uint16).reshape(*shape)
    img_array = img_array.astype(int)
    return img_array


run_ax = 1
rep_ax = 3
run_rep_axes = (run_ax, rep_ax)


def process_counts(counts_array, ref_counts_array=None):
    """Gets average and standard error for counts data structure.
    Assumes counts_array has the structure [nv_ind, run_ind, freq_ind, rep_ind].
    Assumes ref_counts_array has the structure [nv_ind, run_ind, rep_ind].
    Returns the structure [nv_ind, freq_ind] for avg_counts and avg_counts_ste.
    Returns the [nv_ind] for norms.
    """

    counts_array = np.array(counts_array)
    # meas_array = counts_array > 75
    meas_array = counts_array

    avg_counts = np.mean(meas_array, axis=run_rep_axes)
    num_shots = meas_array.shape[rep_ax] * meas_array.shape[run_ax]
    avg_counts_std = np.std(meas_array, axis=run_rep_axes, ddof=1)
    avg_counts_ste = avg_counts_std / np.sqrt(num_shots)

    if ref_counts_array is None:
        return avg_counts, avg_counts_ste

    else:
        norms = np.mean(ref_counts_array, axis=(1, 2))
        # Account for heating by adjusting the norm using the counts from
        # a background spot
        background_nv_ind = 1
        norms *= np.mean(avg_counts[background_nv_ind]) / norms[background_nv_ind]

        return avg_counts, avg_counts_ste, norms


def calc_snr(sig_counts, ref_counts):
    """Calculate SNR for a single shot"""
    avg_sig_counts, avg_sig_counts_ste = process_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste = process_counts(ref_counts)
    noise = np.sqrt(
        np.std(sig_counts, axis=run_rep_axes, ddof=1) ** 2
        + np.std(ref_counts, axis=run_rep_axes, ddof=1) ** 2
    )
    avg_snr = (avg_sig_counts - avg_ref_counts) / noise
    avg_snr_ste = np.sqrt((avg_sig_counts_ste**2 + avg_ref_counts_ste**2)) / noise
    return avg_snr, avg_snr_ste


# endregion
# region Miscellaneous public functions


def rep_loop(num_reps, rep_fn):
    """Loop through the reps for a routine. Handles errors from the camera,
    as well as starting and stopping the pulse generator. Does not handle
    loading the sequence or arming and disarming the camera.

    Parameters
    ----------
    num_reps : int
        Number of reps
    rep_fn : function
        Function to run for each rep
    """
    pulse_gen = tb.get_server_pulse_gen()
    camera = tb.get_server_camera()

    # Try 5 times then give up
    num_attempts = 5
    attempt_ind = 0
    start_rep = 0
    while True:
        try:
            pulse_gen.stream_start()

            for rep_ind in range(start_rep, num_reps):
                rep_fn(rep_ind)

            break

        except Exception as exc:
            pulse_gen.halt()

            nuvu_237 = "NuvuException: 237"
            nuvu_214 = "NuvuException: 214"
            if "NuvuException: 237" in str(exc):
                print(nuvu_237)
            elif "NuvuException: 214" in str(exc):
                print(nuvu_214)
            else:
                raise exc

            attempt_ind += 1
            if attempt_ind == num_attempts:
                raise RuntimeError("Maxed out number of attempts")

            camera.arm()
            start_rep = rep_ind

    pulse_gen.halt()

    if attempt_ind > 0:
        print(f"{attempt_ind} crashes occurred")


def get_repr_nv_sig(nv_list):
    for nv in nv_list:
        try:
            if nv["repr"]:
                return nv
        except:
            pass


def get_nv_num(nv_sig):
    return nv_sig["name"].split("-")[1].split("_")[0][2:]


def get_base_scc_seq_args(nv_list):
    """Return base seq_args for any SCC routine. The base sequence arguments
    are the polarization and ionization AOD coordinates

    Parameters
    ----------
    nv_list : list(nv_sig)
        List of nv signatures to target

    Returns
    -------
    list
        Sequence arguments
    """

    config = common.get_config_dict()
    optics_config = config["Optics"]

    # Polarization
    pol_laser = optics_config[LaserKey.POLARIZATION]["name"]
    drift = pos.get_drift(pol_laser)
    pol_coords_list = [
        pos.get_nv_coords(nv, coords_suffix=pol_laser, drift=drift) for nv in nv_list
    ]

    # Ionization
    ion_laser = optics_config[LaserKey.IONIZATION]["name"]
    drift = pos.get_drift(ion_laser)
    ion_coords_list = [
        pos.get_nv_coords(nv, coords_suffix=ion_laser, drift=drift) for nv in nv_list
    ]

    seq_args = [pol_coords_list, ion_coords_list]
    return seq_args


# endregion
# region Drift tracking


def get_pixel_drift():
    pixel_drift = common.get_registry_entry(["State"], "PIXEL_DRIFT")
    return np.array(pixel_drift)


def set_pixel_drift(drift):
    return common.set_registry_entry(["State"], "PIXEL_DRIFT", drift)


def reset_pixel_drift():
    return set_pixel_drift([0.0, 0.0])


def reset_all_drift():
    reset_pixel_drift()
    pos.reset_drift()
    scanning_optics = _get_scanning_optics()
    for coords_suffix in scanning_optics:
        pos.reset_drift(coords_suffix)


def adjust_pixel_coords_for_drift(pixel_coords, drift=None):
    """Current drift will be retrieved from registry if passed drift is None"""
    if drift is None:
        drift = get_pixel_drift()
    adjusted_coords = (np.array(pixel_coords) + np.array(drift)).tolist()
    return adjusted_coords


def get_nv_pixel_coords(nv_sig, drift_adjust=True, drift=None):
    pixel_coords = nv_sig["pixel_coords"].copy()
    if drift_adjust:
        pixel_coords = adjust_pixel_coords_for_drift(pixel_coords, drift)
    return pixel_coords


def set_all_scanning_drift_from_pixel_drift(pixel_drift=None):
    scanning_optics = _get_scanning_optics()
    for coords_suffix in scanning_optics:
        set_scanning_drift_from_pixel_drift(pixel_drift, coords_suffix)


def _get_scanning_optics():
    config = common.get_config_dict()
    config_optics = config["Optics"]
    scanning_optics = []
    for optic_name in config_optics:
        optic_dict = config_optics[optic_name]
        if "pos_mode" in optic_dict and optic_dict["pos_mode"] == LaserPosMode.SCANNING:
            scanning_optics.append(optic_name)
    return scanning_optics


def set_scanning_drift_from_pixel_drift(pixel_drift=None, coords_suffix=None):
    scanning_drift = pixel_to_scanning_drift(pixel_drift, coords_suffix)
    pos.set_drift(scanning_drift, coords_suffix)


def set_pixel_drift_from_scanning_drift(scanning_drift=None, coords_suffix=None):
    pixel_drift = scanning_to_pixel_drift(scanning_drift, coords_suffix)
    set_pixel_drift(pixel_drift)


def pixel_to_scanning_drift(pixel_drift=None, coords_suffix=None):
    if pixel_drift is None:
        pixel_drift = get_pixel_drift()
    m_x, _, m_y, _ = _pixel_to_scanning_calibration(coords_suffix)
    scanning_drift = pos.get_drift(coords_suffix)
    if len(scanning_drift) > 2:
        z_scanning_drift = scanning_drift[2]
        return [m_x * pixel_drift[0], m_y * pixel_drift[1], z_scanning_drift]
    else:
        return [m_x * pixel_drift[0], m_y * pixel_drift[1]]


def scanning_to_pixel_drift(scanning_drift=None, coords_suffix=None):
    if scanning_drift is None:
        scanning_drift = pos.get_drift(coords_suffix)
    m_x, _, m_y, _ = _scanning_to_pixel_calibration(coords_suffix)
    return [m_x * scanning_drift[0], m_y * scanning_drift[1]]


# endregion
# region Scanning to pixel calibration


def set_nv_scanning_coords_from_pixel_coords(
    nv_sig, coords_suffix=None, drift_adjust=True
):
    pixel_coords = get_nv_pixel_coords(nv_sig, drift_adjust=drift_adjust)
    # pixel_coords = pos.get_nv_coords(
    #     nv_sig, "laser_INTE_520", drift_adjust=drift_adjust
    # )  # MCC
    scanning_coords = pixel_to_scanning_coords(pixel_coords, coords_suffix)
    pos.set_nv_coords(nv_sig, scanning_coords, coords_suffix)
    return scanning_coords


def get_widefield_calibration_nvs():
    module = common.get_config_module()
    nv1 = module.widefield_calibration_nv1.copy()
    nv2 = module.widefield_calibration_nv2.copy()
    return nv1, nv2


def pixel_to_scanning_coords(pixel_coords, coords_suffix=None):
    """Convert camera pixel coordinates to scanning coordinates (e.g. galvo voltages)
    using two calibrated NV coordinate pairs from the config file

    Parameters
    ----------
    pixel_coords : list(numeric)
        Camera pixel coordinates to convert

    Returns
    -------
    list(numeric)
        Scanning coordinates
    """
    m_x, b_x, m_y, b_y = _pixel_to_scanning_calibration(coords_suffix)
    scanning_coords = [m_x * pixel_coords[0] + b_x, m_y * pixel_coords[1] + b_y]
    return scanning_coords


def _pixel_to_scanning_calibration(coords_suffix=None):
    """Get the linear parameters for the conversion"""

    pixel_coords_key = "pixel_coords"
    # pixel_coords_key = "coords-laser_INTE_520"  # MCC

    nv1, nv2 = get_widefield_calibration_nvs()
    nv1_scanning_coords = pos.get_nv_coords(nv1, coords_suffix, drift_adjust=False)
    nv1_pixel_coords = nv1[pixel_coords_key]
    nv2_scanning_coords = pos.get_nv_coords(nv2, coords_suffix, drift_adjust=False)
    nv2_pixel_coords = nv2[pixel_coords_key]

    # Assume (independent) linear relations for both x and y

    scanning_diff = nv2_scanning_coords[0] - nv1_scanning_coords[0]
    pixel_diff = nv2_pixel_coords[0] - nv1_pixel_coords[0]
    m_x = scanning_diff / pixel_diff
    b_x = nv1_scanning_coords[0] - m_x * nv1_pixel_coords[0]

    scanning_diff = nv2_scanning_coords[1] - nv1_scanning_coords[1]
    pixel_diff = nv2_pixel_coords[1] - nv1_pixel_coords[1]
    m_y = scanning_diff / pixel_diff
    b_y = nv1_scanning_coords[1] - m_y * nv1_pixel_coords[1]

    return m_x, b_x, m_y, b_y


def scanning_to_pixel_coords(scanning_coords, coords_suffix=None):
    """Convert scanning coordinates (e.g. galvo voltages) to camera pixel coordinates
    using two calibrated NV coordinate pairs from the config file

    Parameters
    ----------
    scanning_coords : list(numeric)
        Scanning coordinates to convert

    Returns
    -------
    list(numeric)
        Camera pixel coordinates
    """

    m_x, b_x, m_y, b_y = _scanning_to_pixel_calibration(coords_suffix)
    pixel_coords = [m_x * scanning_coords[0] + b_x, m_y * scanning_coords[1] + b_y]
    return pixel_coords


def _scanning_to_pixel_calibration(coords_suffix=None):
    """Get the linear parameters for the conversion"""

    nv1, nv2 = get_widefield_calibration_nvs()
    nv1_scanning_coords = pos.get_nv_coords(nv1, coords_suffix, drift_adjust=False)
    nv1_pixel_coords = nv1["pixel_coords"]
    nv2_scanning_coords = pos.get_nv_coords(nv2, coords_suffix, drift_adjust=False)
    nv2_pixel_coords = nv2["pixel_coords"]

    # Assume (independent) linear relations for both x and y

    pixel_diff = nv2_pixel_coords[0] - nv1_pixel_coords[0]
    scanning_diff = nv2_scanning_coords[0] - nv1_scanning_coords[0]
    m_x = pixel_diff / scanning_diff
    b_x = nv1_pixel_coords[0] - m_x * nv1_scanning_coords[0]

    pixel_diff = nv2_pixel_coords[1] - nv1_pixel_coords[1]
    scanning_diff = nv2_scanning_coords[1] - nv1_scanning_coords[1]
    m_y = pixel_diff / scanning_diff
    b_y = nv1_pixel_coords[1] - m_y * nv1_scanning_coords[1]

    return m_x, b_x, m_y, b_y


# endregion
# region Camera getters - probably only needed internally

# Cache these values to allow fast image processing
_camera_spot_radius = None
_camera_bias_clamp = None
_camera_resolution = None
_camera_em_gain = None
_camera_k_gain = None


def _get_camera_spot_radius():
    global _camera_spot_radius
    if _camera_spot_radius is None:
        _camera_spot_radius = _get_camera_config_val("spot_radius")
    return _camera_spot_radius


def _get_camera_bias_clamp():
    global _camera_bias_clamp
    if _camera_bias_clamp is None:
        _camera_bias_clamp = _get_camera_config_val("bias_clamp")
    return _camera_bias_clamp


def _get_camera_resolution():
    global _camera_resolution
    if _camera_resolution is None:
        _camera_resolution = _get_camera_config_val("resolution")
    return _camera_resolution


def _get_camera_em_gain():
    global _camera_em_gain
    if _camera_em_gain is None:
        _camera_em_gain = _get_camera_config_val("em_gain")
    return _camera_em_gain


def _get_camera_k_gain():
    global _camera_k_gain
    if _camera_k_gain is None:
        readout_mode = _get_camera_readout_mode()
        camera_server_name = common.get_server_name("camera")
        camera_module = import_module(f"servers.inputs.{camera_server_name}")
        k_gain_dict = camera_module.k_gain_dict
        _camera_k_gain = k_gain_dict[readout_mode]
    return _camera_k_gain


def _get_camera_readout_mode():
    return _get_camera_config_val("readout_mode")


def _get_camera_timeout():
    return _get_camera_config_val("timeout")


def _get_camera_temp():
    return _get_camera_config_val("temp")


def _get_camera_readout_mode():
    return _get_camera_config_val("readout_mode")


def _get_camera_roi():
    try:
        return _get_camera_config_val("roi")
    except Exception as exc:
        return None


def _get_camera_config_val(key):
    config = common.get_config_dict()
    return config["Camera"][key]


def get_img_array_shape():
    roi = _get_camera_roi()
    if roi is None:
        shape = _get_camera_resolution()
    else:
        shape = roi[2:]
    return shape


# endregion
# region Plotting


def plot_raw_data(ax, nv_list, x, ys, yerrs=None, subset_inds=None):
    """Plot multiple data sets (with a common set of x vals) with an offset between
    the sets such that they are easier to interpret. Useful for plotting simultaneous
    data from multiple NVs.

    Parameters
    ----------
    ax : matplotlib axes
        axes to plot on
    nv_list : list(nv_sig)
        List of NV sigs to plot
    x : 1D array
        x values to plot
    ys : 2D array
        y values to plot - first dimension divides the data sets up
    yerrs : 2D array
        y errors to plot
    subset_inds : list
        Specify subset_inds if we want to plot just a specific subset of the NVs in nv_list
    """
    num_nvs = len(nv_list)
    if subset_inds is None:
        nv_inds = range(num_nvs)
    else:
        nv_inds = subset_inds
    for nv_ind in nv_inds:
        # if nv_ind != 7:
        #     continue
        nv_sig = nv_list[nv_ind]
        label = get_nv_num(nv_sig)
        yerr = None if yerrs is None else yerrs[nv_ind]
        # if nv_ind == 1:
        #     color = kpl.KplColors.GRAY
        num_colors = len(kpl.data_color_cycler)
        color = kpl.data_color_cycler[nv_ind % num_colors]
        if num_colors <= nv_ind < 2 * num_colors:
            color = kpl.lighten_color_hex(color)
        kpl.plot_points(
            ax,
            x,
            ys[nv_ind],
            yerr=yerr,
            label=label,
            size=kpl.Size.SMALL,
            color=color,
        )

        # MCC
        # kpl.show(block=True)
        # fig, ax = plt.subplots()

    # min_x = min(x)
    # max_x = max(x)
    # excess = 0.08 * (max_x - min_x)
    # ax.set_xlim(min_x - excess, max_x + excess)
    # ncols = 3  # MCC
    # ax.legend(loc=kpl.Loc.LOWER_RIGHT, ncols=ncols)


# Separate plots, shared axes
def plot_fit(
    axes_pack,
    nv_list,
    x,
    ys,
    yerrs=None,
    fns=None,
    popts=None,
    xlim=[None, None],
    norms=None,
):
    """Plot multiple data sets (with a common set of x vals) with an offset between
    the sets such that they are separated and easier to interpret. Useful for
    plotting simultaneous data from multiple NVs.

    Parameters
    ----------
    ax : matplotlib axes
        axes to plot on, flattened
    nv_list : list(nv_sig)
        List of NV sigs to plot
    x : 1D array
        x values to plot
    ys : 2D array
        y values to plot - first dimension divides the data sets up
    yerrs : 2D array
        y errors to plot
    fns : list(function)
        The ith fn is the fit function used to fit the data for the ith NV
    popts : list(list(numeric))
        The ith popt is the curve fit results for the ith NV
    """
    if xlim[0] is None:
        xlim[0] = min(x)
    if xlim[1] is None:
        xlim[1] = max(x)
    x_linspace = np.linspace(*xlim, 1000)
    num_nvs = len(nv_list)
    for nv_ind in range(num_nvs):
        fn = None if fns is None else fns[nv_ind]
        popt = None if popts is None else popts[nv_ind]

        nv_sig = nv_list[nv_ind]
        label = get_nv_num(nv_sig)
        color = kpl.data_color_cycler[nv_ind]

        # MCC
        # if nv_ind == 1:
        #     color = kpl.KplColors.GRAY
        #     ax = axes_pack[-1]
        # elif nv_ind > 1:
        #     ax = axes_pack[nv_ind - 1]
        # else:
        #     ax = axes_pack[nv_ind]
        ax = axes_pack[nv_ind]

        # Include the norm if there is one
        y = np.copy(ys[nv_ind])
        yerr = np.copy(yerrs[nv_ind]) if yerrs is not None else None
        if norms is not None:
            norm = norms[nv_ind]
            y /= norm
            yerr /= norm
        # yerr = None  # MCC

        # Plot the points
        # ls = "none" if fn is not None else "solid"
        ls = "none"
        size = kpl.Size.SMALL
        kpl.plot_points(
            ax, x, y, yerr=yerr, label=label, size=size, color=color, linestyle=ls
        )

        # Plot the fit
        if fn is not None:
            # Include the norm if there is one
            fit_vals = fn(x_linspace, *popt)
            if norms is not None:
                fit_vals /= norm
            kpl.plot_line(ax, x_linspace, fit_vals, color=color)

    for ax in axes_pack:
        ax.spines[["right", "top"]].set_visible(False)

    fig = axes_pack[0].get_figure()
    fig.get_layout_engine().set(h_pad=0, hspace=0, w_pad=0, wspace=0)


def animate(x, nv_list, counts, counts_errs, img_arrays, cmin=None, cmax=None):
    num_steps = img_arrays.shape[0]

    figsize = [6.5, 12.0]
    # fig, axes_pack = plt.subplots(2, 1, height_ratios=(1, 1), figsize=figsize)
    fig = plt.figure(figsize=figsize)
    im_fig, data_fig = fig.subfigures(2, 1)
    im_ax = im_fig.add_subplot()
    layout = np.array(
        [
            ["a", "b", "c", ".", "."],
            ["f", "g", "h", "d", "e"],
            ["k", "l", "m", "i", "j"],
        ]
    )
    data_axes = data_fig.subplot_mosaic(layout, sharex=True, sharey=True)
    data_axes_flat = list(data_axes.values())
    rep_data_ax = data_axes[layout[-1, 0]]

    all_axes = [im_ax]
    all_axes.extend(data_axes_flat)

    # Set up the actual image
    kpl.imshow(im_ax, np.zeros(img_arrays[0].shape), no_cbar=True)

    # Set up the data axis
    # plot_raw_data(data_ax, nv_list, x, counts, counts_errs)
    # plot_fit(data_axes_flat, nv_list, x, counts, counts_errs)

    def data_ax_relim():
        # pass
        x_buffer = 0.05 * (np.max(x) - np.min(x))
        rep_data_ax.set_xlim(np.min(x) - x_buffer, np.max(x) + x_buffer)
        y_buffer = 0.05 * (np.max(counts) - np.min(counts))
        rep_data_ax.set_ylim(np.min(counts) - y_buffer, np.max(counts) + y_buffer)
        # data_ax.set_xlabel("Pulse duration (ns)")
        # data_ax.set_ylabel("Counts")
        data_axes["m"].set_xlabel("Frequency (GHz)")
        data_axes["f"].set_ylabel("Normalized fluorescence")

    data_ax_relim()

    def animate_sub(step_ind):
        # print(step_ind)
        kpl.imshow_update(im_ax, img_arrays[step_ind], cmin, cmax)

        # data_ax.clear()
        # plot_raw_data(
        #     data_ax,
        #     nv_list,
        #     x[0 : step_ind + 1],
        #     counts[:, 0 : step_ind + 1],
        #     counts_errs[:, 0 : step_ind + 1],
        # )
        # data_ax_relim()
        # return [im_ax, data_ax]

        for ax in data_axes_flat:
            ax.clear()
        plot_fit(
            data_axes_flat,
            nv_list,
            x[: step_ind + 1],
            counts[:, : step_ind + 1],
            counts_errs[:, : step_ind + 1],
        )
        data_ax_relim()
        return all_axes

    # animate_sub(10)

    anim = animation.FuncAnimation(
        fig, animate_sub, frames=num_steps, interval=200, blit=False
    )
    timestamp = dm.get_time_stamp()
    anim.save(Path.home() / f"lab/movies/{timestamp}.gif")


def plot_correlations(axes_pack, nv_list, x, counts):
    num_nvs = len(nv_list)

    background_nv_ind = 1

    for nv_ind_1 in range(num_nvs):
        if nv_ind_1 == background_nv_ind:
            continue
        for nv_ind_2 in range(num_nvs):
            if nv_ind_2 == background_nv_ind:
                continue

            ax = axes_pack[
                nv_ind_1 if nv_ind_1 < 1 else nv_ind_1 - 1,
                nv_ind_2 if nv_ind_2 < 1 else nv_ind_2 - 1,
            ]

            if nv_ind_1 == nv_ind_2:
                color = kpl.data_color_cycler[nv_ind_1]
                ax.set_facecolor(color)
                continue

            nv_counts_1 = counts[nv_ind_1]
            nv_counts_2 = counts[nv_ind_2]
            corrs = []
            for step_ind in range(len(x)):
                step_counts_1 = nv_counts_1[:, step_ind, :].flatten()
                step_counts_2 = nv_counts_2[:, step_ind, :].flatten()
                corrs.append(np.corrcoef(step_counts_1, step_counts_2)[0, 1])

            # size = kpl.Size.SMALL
            # kpl.plot_points(ax, x, corrs, size=size)


# endregion


if __name__ == "__main__":
    print(_get_camera_k_gain())

# -*- coding: utf-8 -*-
"""Various utility functions for widefield imaging and camera data processing

Created on August 15th, 2023

@author: mccambria
"""

# region Imports and constants

import itertools
import matplotlib.pyplot as plt
import numpy as np
from numpy import inf
from utils import common
from utils import tool_belt as tb
from utils import positioning as pos
from utils import kplotlib as kpl
from utils.constants import CountFormat
from utils.constants import CollectionMode, LaserKey, LaserPosMode
from importlib import import_module


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
    resolution = _get_camera_resolution()
    roi = _get_camera_roi()
    if roi is None:
        img_array = np.frombuffer(img_str, dtype=np.uint16).reshape(*resolution)
    else:
        offset_x, offset_y, width, height = roi
        bias_clamp = _get_camera_bias_clamp()
        img_array = np.full(resolution, bias_clamp, dtype=np.uint16)
        roi_resolution = roi[2:]
        roi_img_array = np.frombuffer(img_str, dtype=np.uint16).reshape(*roi_resolution)
        img_array[
            offset_y : offset_y + height, offset_x : offset_x + width
        ] = roi_img_array

    img_array = img_array.astype(int)
    return img_array


run_ax = 1
rep_ax = 3
run_rep_axes = (run_ax, rep_ax)


def process_counts(counts_array):
    """Gets average and standard error for counts data structure.
    Assumes the structure [nv_ind, run_ind, freq_ind, rep_ind]
    """

    counts_array = np.array(counts_array)
    # meas_array = counts_array > 75
    meas_array = counts_array

    avg_counts = np.mean(meas_array, axis=run_rep_axes)
    num_shots = meas_array.shape[rep_ax] * meas_array.shape[run_ax]
    avg_counts_std = np.std(meas_array, axis=run_rep_axes, ddof=1)
    avg_counts_ste = avg_counts_std / np.sqrt(num_shots)

    return avg_counts, avg_counts_ste
    # return avg_counts_std, avg_counts_ste  # MCC


def calc_snr(sig_counts, ref_counts):
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


def get_repr_nv_sig(nv_list):
    config = common.get_config_dict()
    repr_nv_ind = config["repr_nv_ind"]
    return nv_list[repr_nv_ind]


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
    pol_coords_list = []
    for nv in nv_list:
        pol_coords = pos.get_nv_coords(nv, coords_suffix=pol_laser)
        pol_coords_list.append(pol_coords)

    # Ionization
    ion_laser = optics_config[LaserKey.IONIZATION]["name"]
    ion_coords_list = []
    for nv in nv_list:
        ion_coords = pos.get_nv_coords(nv, coords_suffix=ion_laser)
        ion_coords_list.append(ion_coords)

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


def set_nv_scanning_coords_from_pixel_coords(nv_sig, coords_suffix=None):
    pixel_coords = get_nv_pixel_coords(nv_sig)
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

    nv1, nv2 = get_widefield_calibration_nvs()
    nv1_scanning_coords = pos.get_nv_coords(nv1, coords_suffix, drift_adjust=False)
    nv1_pixel_coords = nv1["pixel_coords"]
    nv2_scanning_coords = pos.get_nv_coords(nv2, coords_suffix, drift_adjust=False)
    nv2_pixel_coords = nv2["pixel_coords"]

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
    for ind in nv_inds:
        nv_sig = nv_list[ind]
        label = get_nv_num(nv_sig)
        yerr = None if yerrs is None else yerrs[ind]
        kpl.plot_points(ax, x, ys[ind], yerr=yerr, label=label, size=kpl.Size.SMALL)
    # min_x = min(x)
    # max_x = max(x)
    # excess = 0.08 * (max_x - min_x)
    # ax.set_xlim(min_x - excess, max_x + excess)
    ax.legend(loc=kpl.Loc.LOWER_RIGHT)


def plot_fit(
    ax, nv_list, x, ys, yerrs=None, fns=None, popts=None, norms=None, offset=0.10
):
    """Plot multiple data sets (with a common set of x vals) with an offset between
    the sets such that they are separated and easier to interpret. Useful for
    plotting simultaneous data from multiple NVs.

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
    fns : list(function)
        The ith fn is the fit function used to fit the data for the ith NV
    popts : list(list(numeric))
        The ith popt is the curve fit results for the ith NV
    norms : list(numeric)
        The ith factor in the list is used to normalize the ith data set
    offset : numeric
        offset between plotted data sets - default 0.05
    """
    min_x = min(x)
    max_x = max(x)
    x_linspace = np.linspace(min_x, max_x, 1000)
    num_nvs = len(nv_list)
    for nv_ind in range(num_nvs):
        nv_sig = nv_list[nv_ind]
        label = get_nv_num(nv_sig)
        nv_offset = offset * (num_nvs - 1 - nv_ind)
        norm = 1 if norms is None else norms[nv_ind]
        y = ys[nv_ind] / norm + nv_offset
        yerr = None if yerrs is None else yerrs[nv_ind] / norm
        kpl.plot_points(ax, x, y, yerr=yerr, label=label, size=kpl.Size.SMALL)
        fn = fns[nv_ind]
        popt = popts[nv_ind]
        kpl.plot_line(ax, x_linspace, (fn(x_linspace, *popt) / norm) + nv_offset)
    # excess = 0.08 * (max_x - min_x)
    # ax.set_xlim(min_x - excess, max_x + excess)
    ax.legend(loc=kpl.Loc.LOWER_RIGHT)


# endregion


if __name__ == "__main__":
    print(_get_camera_k_gain())

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
# region Plotting


def imshow(ax, img_array, **kwargs):
    """Version of kplotlib's imshow with additional defaults for a camera"""

    cbar_label = "ADUs"
    default_kwargs = {
        "cbar_label": cbar_label,
    }
    passed_kwargs = {**default_kwargs, **kwargs}
    kpl.imshow(ax, img_array, **passed_kwargs)


# endregion
# region Image processing


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
    # Make copies so we don't mutate the originals
    pixel_coords = pixel_coords.copy()
    pixel_x = pixel_coords[0]
    pixel_y = pixel_coords[1]

    if radius is None:
        radius = _get_camera_spot_radius()

    # Don't work through all the pixels, just the ones that might be relevant
    left = int(np.floor(pixel_x - radius))
    right = int(np.ceil(pixel_x + radius))
    top = int(np.floor(pixel_y - radius))
    bottom = int(np.ceil(pixel_y + radius))
    x_crop = np.linspace(left, right, right - left + 1)
    y_crop = np.linspace(top, bottom, bottom - top + 1)
    x_crop_mesh, y_crop_mesh = np.meshgrid(x_crop, y_crop)
    img_array_crop = img_array[top : bottom + 1, left : right + 1]
    dist = np.sqrt((x_crop_mesh - pixel_x) ** 2 + (y_crop_mesh - pixel_y) ** 2)
    inner_pixels = np.where(dist < radius, img_array_crop, np.nan)
    inner_pixels = inner_pixels.flatten()
    inner_pixels = inner_pixels[~np.isnan(inner_pixels)]

    counts = np.sum(inner_pixels)

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

    photons = (adus - bias_clamp) * k_gain / em_gain
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
    img_array = np.frombuffer(img_str, dtype=np.uint16).reshape(*resolution)
    img_array = img_array.astype(int)
    return img_array


def mask_img_array(img_array, nv_list, drift_adjust=True, pixel_drift=None):
    """Mask an image array such that it only contains information about
    NVs in the passed nv_list. Greatly reduces the size of compressed numpy
    array files (npzs). Beware, this works by reference, so the passed
    img_array will be modified

    Parameters
    ----------
    img_array : ndarray
        Image array to mask
    nv_list : list(nv_sig)
        List of nv_sigs for NVs to retain in the masked image
    """

    # Setup
    num_x_pixels = img_array.shape[1]
    num_y_pixels = img_array.shape[0]
    radius = _get_camera_spot_radius

    # Construct the mask by looping through the NVs
    for ind in range(len(nv_list)):
        nv_sig = nv_list[ind]
        pixel_coords = get_nv_pixel_coords(nv_sig, drift_adjust, pixel_drift)
        pixel_x = pixel_coords[0]
        pixel_y = pixel_coords[1]

        x_inds = np.linspace(0, num_x_pixels - 1, num_x_pixels)
        y_inds = np.linspace(0, num_y_pixels - 1, num_y_pixels)
        x_mesh, y_mesh = np.meshgrid(x_inds, y_inds)
        dist = np.sqrt((x_mesh - pixel_x) ** 2 + (y_mesh - pixel_y) ** 2)
        sub_mask = dist < 3 * radius

        if ind == 0:
            mask = sub_mask
        else:
            mask = np.bitwise_or(mask, sub_mask)

    img_array *= mask


def process_img_arrays(img_arrays, nv_list):
    """Turn a nested list of image arrays into a nested list of counts. The
    structure of the nested list of counts will match that of the image arrays"""
    shape = img_arrays.shape
    num_dims = len(shape)
    dims_to_loop = shape[0 : num_dims - 2]  # Last two are the images themselves
    num_nvs = len(nv_list)

    counts_lists = [np.empty(dims_to_loop) for ind in range(num_nvs)]

    sub_indices = [range(el) for el in dims_to_loop]
    indices = itertools.product(*sub_indices)

    for index_tuple in indices:
        img_array = img_arrays[index_tuple]
        img_array_photons = adus_to_photons(img_array)
        for nv_ind in range(num_nvs):
            nv = nv_list[nv_ind]
            pixel_coords = get_nv_pixel_coords(nv)
            counts_list = counts_lists[nv_ind]
            counts_list[index_tuple] = integrate_counts(img_array_photons, pixel_coords)

    return counts_lists


def process_counts(counts_lists):
    """Assumes the structure [nv_ind, run_ind, freq_ind, rep_ind]"""
    run_ax = 1
    rep_ax = 3
    run_rep_axes = (run_ax, rep_ax)

    avg_counts = np.mean(counts_lists, axis=run_rep_axes)
    num_shots = counts_lists.shape[rep_ax] + counts_lists.shape[run_ax]
    avg_counts_std = np.std(counts_lists, axis=run_rep_axes, ddof=1)
    avg_counts_ste = avg_counts_std / np.sqrt(num_shots)

    return avg_counts, avg_counts_ste


# endregion
# region Miscellaneous public functions


def get_base_scc_seq_args(nv_list):
    """Return base seq_args for any SCC routine"""

    nv_sig = nv_list[0]

    # Polarization
    pol_laser_dict = nv_sig[LaserKey.POLARIZATION]
    pol_laser = pol_laser_dict["name"]
    pol_duration = pol_laser_dict["duration"]
    pol_coords_list = []
    for nv in nv_list:
        pol_coords = pos.get_nv_coords(nv, coords_suffix=pol_laser)
        pol_coords_list.append(pol_coords)

    # Ionization
    ion_laser_dict = nv_sig[LaserKey.IONIZATION]
    ion_laser = ion_laser_dict["name"]
    ion_duration = ion_laser_dict["duration"]
    ion_coords_list = []
    for nv in nv_list:
        ion_coords = pos.get_nv_coords(nv, coords_suffix=ion_laser)
        ion_coords_list.append(ion_coords)

    # Readout
    readout_laser_dict = nv_sig[LaserKey.CHARGE_READOUT]
    readout_laser = readout_laser_dict["name"]
    readout_duration = readout_laser_dict["duration"]

    seq_args = [
        pol_laser,
        pol_duration,
        pol_coords_list,
        ion_laser,
        ion_duration,
        ion_coords_list,
        readout_laser,
        readout_duration,
    ]

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
    red_coords = pixel_to_scanning_coords(pixel_coords, coords_suffix)
    pos.set_nv_coords(nv_sig, red_coords, coords_suffix)
    return red_coords


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


def _get_camera_spot_radius():
    return _get_camera_config_val("spot_radius")


def _get_camera_bias_clamp():
    return _get_camera_config_val("bias_clamp")


def _get_camera_resolution():
    return _get_camera_config_val("resolution")


def _get_camera_em_gain():
    return _get_camera_config_val("em_gain")


def _get_camera_readout_mode():
    return _get_camera_config_val("readout_mode")


def _get_camera_k_gain():
    readout_mode = _get_camera_readout_mode()
    camera_server_name = common.get_server_name("camera")
    camera_module = import_module(f"servers.inputs.{camera_server_name}")
    k_gain_dict = camera_module.k_gain_dict
    k_gain = k_gain_dict[readout_mode]
    return k_gain


def _get_camera_timeout():
    return _get_camera_config_val("timeout")


def _get_camera_temp():
    return _get_camera_config_val("temp")


def _get_camera_config_val(key):
    config = common.get_config_dict()
    return config["Camera"][key]


# endregion


if __name__ == "__main__":
    print(_get_camera_k_gain())

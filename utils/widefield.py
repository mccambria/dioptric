# -*- coding: utf-8 -*-
"""Various utility functions for widefield imaging and camera data processing

Created on August 15th, 2023

@author: mccambria
"""

# region Imports and constants

import dataclasses
import itertools
from functools import cache
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy import inf
from scipy.special import gamma
from scipy.stats import poisson

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import (
    CollectionMode,
    CoordsKey,
    CountFormat,
    LaserKey,
    LaserPosMode,
    NVSig,
)
from utils.tool_belt import determine_threshold

# endregion
# region Image processing


def integrate_counts_from_adus(img_array, pixel_coords, radius=None):
    img_array_photons = adus_to_photons(img_array)
    return integrate_counts(img_array_photons, pixel_coords, radius)


@cache
def _calc_dist_matrix(radius=None):
    if radius is None:
        radius = _get_camera_spot_radius()
    left = -radius
    right = +radius
    top = -radius
    bottom = +radius
    x_crop = list(range(left, right + 1))
    y_crop = list(range(top, bottom + 1))
    x_crop_mesh, y_crop_mesh = np.meshgrid(x_crop, y_crop)
    return np.sqrt((x_crop_mesh) ** 2 + (y_crop_mesh) ** 2)


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
    left = round(pixel_x - radius)
    right = round(pixel_x + radius)
    top = round(pixel_y - radius)
    bottom = round(pixel_y + radius)
    img_array_crop = img_array[top : bottom + 1, left : right + 1]
    dist = _calc_dist_matrix()

    counts = np.sum(img_array_crop, where=dist < radius)
    return counts


def adus_to_photons(adus, k_gain=None, em_gain=None, baseline=None):
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
    baseline : numeric, optional
        Baseline, or bias clamp level, i.e. the ADU value for a pixel which receives no light.
        Used to ensure the camera does not return negative values. By default retrieved from config

    Returns
    -------
    numeric
        Quantity converted to photons
    """
    if k_gain is None:
        k_gain = _get_camera_k_gain()
    if em_gain is None:
        em_gain = _get_camera_em_gain()
    if baseline is None:
        baseline = _get_camera_bias_clamp()

    total_gain = k_gain / em_gain
    photons = (adus - baseline) * total_gain
    return photons


@cache
def _img_array_iris(shape):
    roi = _get_camera_roi()  # offsetX, offsetY, width, height
    offsetX, offsetY, width, height = roi
    iris_radius = np.sqrt((height / 2) ** 2 + (width / 2) ** 2) + 10
    center_x = offsetX + width // 2
    center_y = height // 2

    iris = np.empty(shape)
    for ind in range(shape[0]):
        for jnd in range(shape[1]):
            dist = np.sqrt((jnd - center_x) ** 2 + (ind - center_y) ** 2)
            iris[ind, jnd] = dist > iris_radius

    return iris


def img_str_to_array(img_str):
    """Convert an img_array from a uint16-valued byte string (returned by the camera
    labrad server for speed) into a usable int-valued 2D array. Also subtracts off bias
    not captured on Nuvu's end camera software

    Parameters
    ----------
    img_str : byte string
        Image array as a byte string - the return value of a camera.read() call

    Returns
    -------
    ndarray
        Image array contructed from the byte string
    """
    shape = _get_img_str_shape()
    img_array = np.frombuffer(img_str, dtype=np.uint16).reshape(*shape)
    img_array = img_array.astype(int)

    # Subtract off correlated readout noise (see wiki 4/19/24)
    roi = _get_camera_roi()  # offsetX, offsetY, width, height
    if roi is None:
        baseline = _get_camera_bias_clamp()
    else:
        offset_x = roi[0]
        width = roi[2]
        iris = _img_array_iris(img_array.shape)
        bg_pixels = np.where(iris, img_array, np.nan)
        baseline = np.nanmean(bg_pixels)
        img_array = img_array[0:, offset_x : offset_x + width]
    return img_array, baseline


run_ax = 1
rep_ax = 3
run_rep_axes = (run_ax, rep_ax)


def average_counts(sig_counts, ref_counts=None):
    """Gets average and standard error for counts data structure.
    Counts arrays must have the structure [nv_ind, run_ind, freq_ind, rep_ind].
    Returns the structure [nv_ind, freq_ind] for avg_counts and avg_counts_ste.
    Returns the [nv_ind] for norms.
    """
    _validate_counts_structure(sig_counts)
    _validate_counts_structure(ref_counts)

    avg_counts = np.mean(sig_counts, axis=run_rep_axes)
    num_shots = sig_counts.shape[rep_ax] * sig_counts.shape[run_ax]
    avg_counts_std = np.std(sig_counts, axis=run_rep_axes, ddof=1)
    avg_counts_ste = avg_counts_std / np.sqrt(num_shots)

    if ref_counts is None:
        norms = None
    else:
        ms0_ref_counts = ref_counts[:, :, :, 0::2]
        ms1_ref_counts = ref_counts[:, :, :, 1::2]
        norms = [
            np.mean(ms0_ref_counts, axis=(1, 2, 3)),
            np.mean(ms1_ref_counts, axis=(1, 2, 3)),
        ]

    return avg_counts, avg_counts_ste, norms


def threshold_counts(
    nv_list, sig_counts, ref_counts=None, dual_thresh_range=None, dynamic_thresh=False
):
    """Only actually thresholds counts for NVs with thresholds specified in their sigs.
    If there's no threshold, then the raw counts are just averaged as normal."""
    _validate_counts_structure(sig_counts)
    _validate_counts_structure(ref_counts)

    if dynamic_thresh:
        thresholds = []
        num_nvs = len(nv_list)
        for ind in range(num_nvs):
            # combined_counts = np.append(
            #     sig_counts[ind].flatten(), ref_counts[ind].flatten()
            # )
            # threshold = determine_threshold(combined_counts)
            threshold = determine_threshold(sig_counts[ind], no_print=True)
            thresholds.append(threshold)
    else:
        thresholds = [1.2 * nv.threshold for nv in nv_list]  # MCC
    print(thresholds)

    thresholds = np.array(thresholds)
    thresholds = thresholds[:, np.newaxis, np.newaxis, np.newaxis]

    if dual_thresh_range is None:
        sig_states_array = tb.threshold(sig_counts, thresholds)
    else:
        half_range = dual_thresh_range / 2
        sig_states_array = tb.dual_threshold(
            sig_counts, thresholds - half_range, thresholds + half_range
        )
    if ref_counts is not None:
        if dual_thresh_range is None:
            ref_states_array = tb.threshold(ref_counts, thresholds)
        else:
            ref_states_array = tb.dual_threshold(
                ref_counts, thresholds - half_range, thresholds + half_range
            )
    else:
        ref_states_array = None

    return sig_states_array, ref_states_array


def poisson_pmf_cont(k, mean):
    return mean**k * np.exp(-mean) / gamma(k + 1)


@cache
def _get_mle_radius():
    return round(1.5 * _get_camera_spot_radius())


@cache
def _calc_mesh_grid(radius=None):
    radius = _get_mle_radius()
    half_range = radius
    one_ax_linspace = np.linspace(-half_range, half_range, 2 * half_range + 1)
    x_crop_mesh, y_crop_mesh = np.meshgrid(one_ax_linspace, one_ax_linspace)
    return x_crop_mesh, y_crop_mesh


@cache
def _calc_nvn_count_distribution(nvn_dist_params, subpixel_offset=(0, 0)):
    x_crop_mesh, y_crop_mesh = _calc_mesh_grid()
    bg, amp, sigma = nvn_dist_params
    return bg + amp * np.exp(
        -(
            ((x_crop_mesh - subpixel_offset[0]) ** 2)
            + ((y_crop_mesh - subpixel_offset[1]) ** 2)
        )
        / (2 * sigma**2)
    )


@cache
def _calc_nv0_count_distribution(nvn_dist_params):
    bg = nvn_dist_params[0]
    return bg


def charge_state_mle_single(nv_sig, img_array):
    nvn_dist_params = nv_sig.nvn_dist_params
    if nvn_dist_params is None:
        return None

    x0, y0 = get_nv_pixel_coords(nv_sig)
    radius = _get_mle_radius()
    half_range = radius
    left = round(x0 - half_range)
    right = round(x0 + half_range)
    top = round(y0 - half_range)
    bottom = round(y0 + half_range)
    img_array_crop = img_array[top : bottom + 1, left : right + 1]
    img_array_crop = np.where(img_array_crop >= 0, img_array_crop, 0)

    subpixel_offset = (x0 - round(x0), y0 - round(y0))
    nvn_count_distribution = _calc_nvn_count_distribution(
        nvn_dist_params, subpixel_offset
    )
    nv0_count_distribution = _calc_nv0_count_distribution(nvn_dist_params)

    nvn_probs = poisson_pmf_cont(img_array_crop, nvn_count_distribution)
    nv0_probs = poisson_pmf_cont(img_array_crop, nv0_count_distribution)
    nvn_prob = np.nanprod(nvn_probs)
    nv0_prob = np.nanprod(nv0_probs)
    return int(nvn_prob > nv0_prob)


def charge_state_mle(nv_list, img_array):
    """Maximum likelihood estimator of state based on image"""

    states = [charge_state_mle_single(nv, img_array) for nv in nv_list]

    return states


def process_counts(nv_list, sig_counts, ref_counts=None, threshold=True):
    """Alias for threshold_counts with a more generic name"""
    _validate_counts_structure(sig_counts)
    _validate_counts_structure(ref_counts)
    if threshold:
        sig_states_array, ref_states_array = threshold_counts(
            nv_list, sig_counts, ref_counts
        )
        return average_counts(sig_states_array, ref_states_array)
    else:
        return average_counts(sig_counts, ref_counts)


def calc_snr(sig_counts, ref_counts):
    """Calculate SNR for a single shot"""
    avg_contrast, avg_contrast_ste = calc_contrast(sig_counts, ref_counts)
    noise = np.sqrt(
        np.std(sig_counts, axis=run_rep_axes, ddof=1) ** 2
        + np.std(ref_counts, axis=run_rep_axes, ddof=1) ** 2
    )
    avg_snr = avg_contrast / noise
    avg_snr_ste = avg_contrast_ste / noise
    return avg_snr, avg_snr_ste


def calc_contrast(sig_counts, ref_counts):
    """Calculate contrast for a single shot"""
    _validate_counts_structure(sig_counts)
    _validate_counts_structure(ref_counts)
    avg_sig_counts, avg_sig_counts_ste, _ = average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste, _ = average_counts(ref_counts)
    avg_contrast = avg_sig_counts - avg_ref_counts
    avg_contrast_ste = np.sqrt((avg_sig_counts_ste**2 + avg_ref_counts_ste**2))
    return avg_contrast, avg_contrast_ste


def _validate_counts_structure(counts):
    """Make sure the structure of the counts object for a single experiment is valid
    for further processing. The structure is [nv_ind, run_ind, freq_ind, rep_ind]."""
    if counts is None:
        return
    if not isinstance(counts, np.ndarray):
        raise RuntimeError("Passed counts object is not a numpy array.")
    if counts.ndim != 4:
        raise RuntimeError("Passed counts object has the wrong number of dimensions.")


# endregion
# region Miscellaneous public functions


def get_default_keys_to_compress(raw_data):
    keys_to_compress = []
    if "img_arrays" in raw_data:
        keys_to_compress.append("img_arrays")
    if "mean_img_arrays" in raw_data:
        keys_to_compress.append("mean_img_arrays")
    return keys_to_compress


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


def get_repr_nv_sig(nv_list: list[NVSig]) -> NVSig:
    for nv in nv_list:
        try:
            if nv.representative:
                return nv
        except Exception:
            pass


def get_nv_num(nv_sig):
    nv_name = nv_sig.name  # of the form <sample_name>-nv<num>_<date_str>
    nv_name_part = nv_name.split("-")[1]  # nv<num>_<date_str>
    nv_num = nv_name_part.split("_")[0][2:]
    nv_num = int(nv_num)
    return nv_num


def get_base_scc_seq_args(nv_list: list[NVSig], uwave_ind_list: list[int]):
    """Return base seq_args for any SCC routine. The base sequence arguments
    are the minimum info required for state preparation and SCC

    Parameters
    ----------
    nv_list : list[NVSig]
        List of nv signatures to target
    uwave_ind : int
        Index of the microwave chain to run for state prep

    Returns
    -------
    list
        Sequence arguments
    """

    pol_coords_list = get_coords_list(nv_list, LaserKey.CHARGE_POL)
    scc_coords_list = get_coords_list(nv_list, LaserKey.SCC)
    scc_duration_list = get_scc_duration_list(nv_list)
    spin_flip_ind_list = get_spin_flip_ind_list(nv_list)

    seq_args = [
        pol_coords_list,
        scc_coords_list,
        scc_duration_list,
        spin_flip_ind_list,
        uwave_ind_list,
    ]
    return seq_args


def get_coords_list(
    nv_list: list[NVSig], laser_key, drift_adjust=True, include_inds=None
):
    laser_name = tb.get_laser_name(laser_key)
    drift = pos.get_drift(laser_name) if drift_adjust else None
    coords_list = [
        pos.get_nv_coords(
            nv, coords_key=laser_name, drift_adjust=drift_adjust, drift=drift
        )
        for nv in nv_list
    ]
    if include_inds is not None:
        coords_list = [coords_list[ind] for ind in include_inds]
    return coords_list


def get_spin_flip_ind_list(nv_list: list[NVSig]):
    num_nvs = len(nv_list)
    return [ind for ind in range(num_nvs) if nv_list[ind].spin_flip]


def get_scc_duration_list(nv_list: list[NVSig]):
    scc_duration_list = []
    for nv in nv_list:
        scc_duration = nv.scc_duration
        if scc_duration is None:
            config = common.get_config_dict()
            scc_duration = config["Optics"][LaserKey.SCC]["duration"]
        if not (scc_duration % 4 == 0 and scc_duration >= 16):
            raise RuntimeError("SCC pulse duration not valid for OPX.")
        scc_duration_list.append(scc_duration)
    return scc_duration_list


# endregion
# region Drift tracking


@cache
def get_pixel_drift():
    pixel_drift = common.get_registry_entry(["State"], "DRIFT-pixel")
    return np.array(pixel_drift)


def set_pixel_drift(drift):
    get_pixel_drift.cache_clear()
    return common.set_registry_entry(["State"], "DRIFT-pixel", drift)


def reset_pixel_drift():
    return set_pixel_drift([0.0, 0.0])


def reset_all_drift():
    reset_pixel_drift()
    pos.reset_drift()
    scanning_optics = _get_scanning_optics()
    for coords_key in scanning_optics:
        pos.reset_drift(coords_key)


def adjust_pixel_coords_for_drift(pixel_coords, drift=None):
    """Current drift will be retrieved from registry if passed drift is None"""
    if drift is None:
        drift = get_pixel_drift()
    adjusted_coords = (np.array(pixel_coords) + np.array(drift)).tolist()
    return adjusted_coords


def get_nv_pixel_coords(nv_sig: NVSig, drift_adjust=True, drift=None):
    pixel_coords = nv_sig.coords[CoordsKey.PIXEL]
    if drift_adjust:
        pixel_coords = adjust_pixel_coords_for_drift(pixel_coords, drift)
    return pixel_coords


def set_all_scanning_drift_from_pixel_drift(pixel_drift=None):
    scanning_optics = _get_scanning_optics()
    for coords_key in scanning_optics:
        set_scanning_drift_from_pixel_drift(pixel_drift, coords_key)


def _get_scanning_optics():
    config = common.get_config_dict()
    config_optics = config["Optics"]
    scanning_optics = []
    for optic_name in config_optics:
        val = config_optics[optic_name]
        if (
            isinstance(val, dict)
            and "pos_mode" in val
            and val["pos_mode"] == LaserPosMode.SCANNING
        ):
            scanning_optics.append(optic_name)
    return scanning_optics


def set_scanning_drift_from_pixel_drift(pixel_drift=None, coords_key=CoordsKey.GLOBAL):
    scanning_drift = pixel_to_scanning_drift(pixel_drift, coords_key)
    pos.set_drift(scanning_drift, coords_key)


def set_pixel_drift_from_scanning_drift(
    scanning_drift=None, coords_key=CoordsKey.GLOBAL
):
    pixel_drift = scanning_to_pixel_drift(scanning_drift, coords_key)
    set_pixel_drift(pixel_drift)


def pixel_to_scanning_drift(pixel_drift=None, coords_key=CoordsKey.GLOBAL):
    if pixel_drift is None:
        pixel_drift = get_pixel_drift()
    m_x, _, m_y, _ = _pixel_to_scanning_calibration(coords_key)
    scanning_drift = pos.get_drift(coords_key)
    if len(scanning_drift) > 2:
        z_scanning_drift = scanning_drift[2]
        return [m_x * pixel_drift[0], m_y * pixel_drift[1], z_scanning_drift]
    else:
        return [m_x * pixel_drift[0], m_y * pixel_drift[1]]


def scanning_to_pixel_drift(scanning_drift=None, coords_key=CoordsKey.GLOBAL):
    if scanning_drift is None:
        scanning_drift = pos.get_drift(coords_key)
    m_x, _, m_y, _ = _scanning_to_pixel_calibration(coords_key)
    return [m_x * scanning_drift[0], m_y * scanning_drift[1]]


# endregion
# region Scanning to pixel calibration


def set_nv_scanning_coords_from_pixel_coords(
    nv_sig, coords_key: str | CoordsKey = CoordsKey.GLOBAL, drift_adjust=True
):
    pixel_coords = get_nv_pixel_coords(nv_sig, drift_adjust=drift_adjust)
    # pixel_coords = pos.get_nv_coords(
    #     nv_sig, "laser_INTE_520", drift_adjust=drift_adjust
    # )  # MCC
    scanning_coords = pixel_to_scanning_coords(pixel_coords, coords_key)
    pos.set_nv_coords(nv_sig, scanning_coords, coords_key)
    return scanning_coords


def get_widefield_calibration_nvs():
    module = common.get_config_module()
    nv1 = NVSig(coords=module.widefield_calibration_coords1)
    nv2 = NVSig(coords=module.widefield_calibration_coords2)
    return nv1, nv2


def pixel_to_scanning_coords(pixel_coords, coords_key=CoordsKey.GLOBAL):
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
    m_x, b_x, m_y, b_y = _pixel_to_scanning_calibration(coords_key)
    scanning_coords = [m_x * pixel_coords[0] + b_x, m_y * pixel_coords[1] + b_y]
    return scanning_coords


def _pixel_to_scanning_calibration(coords_key=CoordsKey.GLOBAL):
    """Get the linear parameters for the conversion"""

    nv1, nv2 = get_widefield_calibration_nvs()
    nv1_scanning_coords = pos.get_nv_coords(nv1, coords_key, drift_adjust=False)
    nv1_pixel_coords = get_nv_pixel_coords(nv1, drift_adjust=False)
    nv2_scanning_coords = pos.get_nv_coords(nv2, coords_key, drift_adjust=False)
    nv2_pixel_coords = get_nv_pixel_coords(nv2, drift_adjust=False)

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


def scanning_to_pixel_coords(scanning_coords, coords_key=CoordsKey.GLOBAL):
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

    m_x, b_x, m_y, b_y = _scanning_to_pixel_calibration(coords_key)
    pixel_coords = [m_x * scanning_coords[0] + b_x, m_y * scanning_coords[1] + b_y]
    return pixel_coords


def _scanning_to_pixel_calibration(coords_key=CoordsKey.GLOBAL):
    """Get the linear parameters for the conversion"""

    nv1, nv2 = get_widefield_calibration_nvs()
    nv1_scanning_coords = pos.get_nv_coords(nv1, coords_key, drift_adjust=False)
    nv1_pixel_coords = get_nv_pixel_coords(nv1, drift_adjust=False)
    nv2_scanning_coords = pos.get_nv_coords(nv2, coords_key, drift_adjust=False)
    nv2_pixel_coords = get_nv_pixel_coords(nv2, drift_adjust=False)

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


@cache
def _get_camera_spot_radius():
    return _get_camera_config_val("spot_radius")


@cache
def _get_camera_bias_clamp():
    return _get_camera_config_val("bias_clamp")


@cache
def _get_camera_resolution():
    return _get_camera_config_val("resolution")


@cache
def _get_camera_em_gain():
    return _get_camera_config_val("em_gain")


@cache
def _get_camera_k_gain():
    readout_mode = _get_camera_readout_mode()
    camera_server_name = common.get_server_name("camera")
    camera_module = import_module(f"servers.inputs.{camera_server_name}")
    k_gain_dict = camera_module.k_gain_dict
    _camera_k_gain = k_gain_dict[readout_mode]
    return _camera_k_gain


@cache
def _get_camera_readout_mode():
    return _get_camera_config_val("readout_mode")


@cache
def _get_camera_timeout():
    return _get_camera_config_val("timeout")


@cache
def _get_camera_temp():
    return _get_camera_config_val("temp")


@cache
def _get_camera_readout_mode():
    return _get_camera_config_val("readout_mode")


@cache
def _get_camera_roi():
    try:
        return _get_camera_config_val("roi")
    except Exception:
        return None


def _get_camera_config_val(key):
    config = common.get_config_dict()
    return config["Camera"][key]


def _get_img_str_shape():
    resolution = _get_camera_resolution()
    roi = _get_camera_roi()
    if roi is None:
        shape = resolution
    else:
        shape = (roi[3], resolution[0])
    return shape


def get_img_array_shape():
    resolution = _get_camera_resolution()
    roi = _get_camera_roi()
    if roi is None:
        shape = resolution
    else:
        shape = roi[2:]
    return shape


@cache
def get_camera_scale():
    return _get_camera_config_val("scale")


# endregion
# region Plotting


def draw_circles_on_nvs(ax, nv_list, drift=None):
    scale = get_camera_scale()
    pixel_coords_list = [get_nv_pixel_coords(nv, drift=drift) for nv in nv_list]
    for ind in range(len(pixel_coords_list)):
        pixel_coords = pixel_coords_list[ind]
        color = kpl.data_color_cycler[ind]
        kpl.draw_circle(ax, pixel_coords, color=color, radius=0.7 * scale, label=ind)
    num_nvs = len(nv_list)
    ncols = (num_nvs // 5) + (1 if num_nvs % 5 > 0 else 0)
    ncols = 5
    ax.legend(loc=kpl.Loc.LOWER_CENTER, ncols=ncols, markerscale=0.9)


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
        # if nv_ind not in [3]:
        #     continue
        # if nv_ind not in [0, 1, 2, 4, 6, 11, 14]:
        #     continue
        yerr = None if yerrs is None else yerrs[nv_ind]
        nv_sig = nv_list[nv_ind]
        # nv_num = get_nv_num(nv_sig)
        nv_num = nv_ind
        color = kpl.data_color_cycler[nv_num]
        kpl.plot_points(
            ax,
            x,
            ys[nv_ind],
            yerr=yerr,
            label=str(nv_num),
            size=kpl.Size.SMALL,
            color=color,
        )

        # MCC
        # ax.legend()
        # kpl.show(block=True)
        # fig, ax = plt.subplots()

    # min_x = min(x)
    # max_x = max(x)
    # excess = 0.08 * (max_x - min_x)
    # ax.set_xlim(min_x - excess, max_x + excess)
    ncols = round(num_nvs / 5)
    ax.legend(ncols=ncols)


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
    no_legend=False,
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
    if isinstance(axes_pack, dict):
        axes_pack = list(axes_pack.values())
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
        nv_num = nv_ind
        # nv_num = get_nv_num(nv_sig)
        color = kpl.data_color_cycler[nv_num]

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
        # size = kpl.Size.SMALL
        size = kpl.Size.XSMALL
        # size = kpl.Size.TINY
        label = str(nv_num)
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

        if not no_legend:
            loc = kpl.Loc.UPPER_LEFT
            # loc = kpl.Loc.UPPER_LEFT if nv_ind in [0, 1, 4, 6] else "upper center"
            ax.legend(loc=loc)

    for ax in axes_pack:
        ax.spines[["right", "top"]].set_visible(False)

    fig = axes_pack[0].get_figure()
    fig.get_layout_engine().set(h_pad=0, hspace=0, w_pad=0, wspace=0)


def downsample_img_array(img_array, downsample_factor):
    shape = img_array.shape
    downsampled_shape = (
        int(np.floor(shape[0] / downsample_factor)),
        int(np.floor(shape[1] / downsample_factor)),
    )

    # Clip the original img_array so that its dimensions are an integer
    # multiple of downsample_factor
    clip_shape = (
        downsample_factor * downsampled_shape[0],
        downsample_factor * downsampled_shape[1],
    )
    img_array = img_array[: clip_shape[0], : clip_shape[1]]

    downsampled_img_array = np.zeros(downsampled_shape)
    for ind in range(downsample_factor):
        for jnd in range(downsample_factor):
            downsampled_img_array += img_array[
                ind::downsample_factor, jnd::downsample_factor
            ]

    return downsampled_img_array


def animate(x, nv_list, counts, counts_errs, img_arrays, cmin=None, cmax=None):
    num_steps = img_arrays.shape[0]

    figsize = [12.8, 6.0]
    # fig, axes_pack = plt.subplots(2, 1, height_ratios=(1, 1), figsize=figsize)
    fig = plt.figure(figsize=figsize)
    im_fig, data_fig = fig.subfigures(1, 2, width_ratios=(6, 6.5))
    im_ax = im_fig.add_subplot()
    num_nvs = len(nv_list)
    layout = kpl.calc_mosaic_layout(num_nvs, num_rows=3)
    data_axes = data_fig.subplot_mosaic(layout, sharex=True, sharey=True)
    data_axes_flat = list(data_axes.values())
    rep_data_ax = data_axes[layout[-1, 0]]

    # all_axes = [fig, im_fig, data_fig, im_ax]
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
        # rep_data_ax.set_xlim(0, np.max(x) + x_buffer)
        # ax.set_xticks((0, 100, 200))
        y_buffer = 0.05 * (np.max(counts) - np.min(counts))
        rep_data_ax.set_ylim(np.min(counts) - y_buffer, np.max(counts) + y_buffer)

        ax = data_axes[layout[-1, 0]]
        ax.set_xlabel(" ")
        xlabel = "Frequency (GHz)"
        # xlabel = "Pulse duration (ns)"
        data_fig.text(0.55, 0.01, xlabel, ha="center")
        ylabel = "Change in fraction in NV$^{-}$"
        ax.set_ylabel(" ")
        data_fig.text(0.005, 0.55, ylabel, va="center", rotation="vertical")

        # data_axes[layout[0, 1]].legend(bbox_to_anchor=(1.05, 1), loc=kpl.Loc.UPPER_LEFT)

    # ax.set_xlabel(" ")
    # ax.set_ylabel(" ")
    # # label = "Normalized fraction in NV$^{-}$"
    # label = "Change in fraction in NV$^{-}$"
    # fig.text(0.005, 0.55, label, va="center", rotation="vertical")

    data_ax_relim()

    def animate_sub(step_ind):
        # print(step_ind)
        kpl.imshow_update(im_ax, img_arrays[step_ind], cmin, cmax)
        im_ax.axis("off")

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
            no_legend=False,
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
    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    kpl.imshow(ax, _img_array_iris((250, 512)))
    kpl.show(block=True)

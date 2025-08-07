# -*- coding: utf-8 -*-
"""
Search for NV triplet-to-singlet wavelength

Created on August 5th, 2025

@author: mccambria
"""

import time
import traceback
from random import shuffle

import numpy as np
from matplotlib import pyplot as plt

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb


def create_raw_figure(wavelengths, relative_diff, relative_diff_err):
    fig, ax = plt.subplots()
    kpl.plot_points(ax, wavelengths, relative_diff, relative_diff_err)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Contrast (arb. units)")


def create_fit_figure():
    pass


def main(min_wavelength, max_wavelength, num_steps, num_runs):
    wavelengths = np.linspace(min_wavelength, max_wavelength, num_steps)
    num_exps = 2

    etalon_range = 10
    etalon_spacing = 1
    # if range = 10, then the settings will be set to [45, 55] so 11 data points in total
    etalon_settings = np.arange(
        50 - etalon_range, 50 + (etalon_range + 1), etalon_spacing
    )
    num_etalon = len(etalon_settings)

    voltages = np.empty((num_runs, num_steps, num_etalon, num_exps))

    step_ind_list = list(range(0, num_steps))
    step_ind_master_list = [None for ind in range(num_runs)]
    shutter_channel = 1

    shutter = common.get_server_by_name("shutter_STAN_sr474")
    multimeter = common.get_server_by_name("multimeter_MULT_mp730028")
    tisapph = common.get_server_by_name("tisapph_M2_solstis")

    ### Collect the data
    dm_folder = common.get_data_manager_folder()

    # Row 0 - Voltage, Row 1 - Wavelength Bin
    data_to_save = np.empty((2, num_runs, num_steps, num_etalon, num_exps))

    try:
        # Runs loop
        for run_ind in range(num_runs):
            print(f"Run {run_ind}")
            shuffle(step_ind_list)

            # Steps loop
            num_steps_completed = 0

            for step_ind in step_ind_list:
                print(f"Number steps completed {num_steps_completed}")

                # Set coarse wavelength
                wavelength = wavelengths[step_ind]
                tisapph.set_wavelength_nm(wavelength)
                tisapph.tune_etalon_relative(50)

                # Save wavelength data
                data_to_save[1, run_ind, num_steps_completed, :, :] = wavelength

                # Decrement etalon voltage from 50% sweep
                for eind in range(int(num_etalon / 2) - 1, -1, -etalon_spacing):
                    # Get reference voltage
                    shutter.close(shutter_channel)
                    time.sleep(0.04)
                    voltage = multimeter.measure()
                    voltages[run_ind, step_ind, eind, 0] = voltage

                    shutter.open(shutter_channel)
                    tisapph.tune_etalon_relative(etalon_settings[eind])

                    # Get signal voltage
                    time.sleep(0.05)
                    voltage = multimeter.measure()
                    voltages[run_ind, step_ind, eind, 1] = voltage

                # Set back to 50%
                tisapph.tune_etalon_relative(50)

                # Increment etalon voltage from 50% sweep
                for eind in range(int(num_etalon / 2), num_etalon, etalon_spacing):
                    # Get reference voltage
                    shutter.close(shutter_channel)
                    time.sleep(0.04)
                    voltage = multimeter.measure()
                    voltages[run_ind, step_ind, eind, 0] = voltage

                    shutter.open(shutter_channel)
                    tisapph.tune_etalon_relative(etalon_settings[eind])

                    # Get signal voltage
                    time.sleep(0.05)
                    voltage = multimeter.measure()
                    voltages[run_ind, step_ind, eind, 1] = voltage

                num_steps_completed += 1

            ### Move on to the next run

            # Record step order
            step_ind_master_list[run_ind] = step_ind_list.copy()

    except Exception:
        print(traceback.format_exc())

    ### Return

    raw_data = {
        "num_steps": num_steps,
        "num_runs": num_runs,
        "min_wavelength": min_wavelength,
        "max_wavelength": max_wavelength,
        "wavelengths": wavelengths,
        "step_ind_master_list": step_ind_master_list,
        "voltages": voltages,
        "voltages-units": "photons",
    }

    ### Process and plot

    try:
        avg_voltages = np.mean(voltages, axis=1)
        ste_voltages = np.std(voltages, axis=1) / np.sqrt(num_runs)
        ref_voltages = avg_voltages[0]
        sig_voltages = avg_voltages[1]
        ref_voltages_ste = ste_voltages[0]
        sig_voltages_ste = ste_voltages[1]

        diff = ref_voltages - sig_voltages
        relative_diff = (ref_voltages - sig_voltages) / ref_voltages
        diff_err = np.sqrt(ref_voltages_ste**2 + sig_voltages_ste**2)
        relative_diff_err = np.abs(relative_diff) * np.sqrt(
            (diff_err / diff) ** 2 + (ref_voltages_ste / ref_voltages) ** 2
        )

        raw_fig = create_raw_figure(wavelengths, relative_diff, relative_diff_err)
        # fit_fig = create_fit_figure(wavelengths, relative_diff)
        fit_fig = None
    except Exception:
        print(traceback.format_exc())
        raw_fig = None
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
    }

    repr_nv_name = "implanted_chinese"
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    kpl.show(block=True)

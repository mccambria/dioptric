# -*- coding: utf-8 -*-
"""
Config file for the PC purcell

Created July 20th, 2023
@author: mccambria
@author: sbchand
"""

from pathlib import Path

import numpy as np

from config.default import config
from utils.constants import (
    ChargeStateEstimationMode,
    CollectionMode,
    CoordsKey,
    CountFormat,
    ModMode,
    PosControlMode,
    VirtualLaserKey,
)

home = Path.home()

# region Widefield calibration coords

green_laser = "laser_INTE_520"
yellow_laser = "laser_OPTO_589"
red_laser = "laser_COBO_638"
green_laser_aod = "laser_INTE_520_aod"
red_laser_aod = "laser_COBO_638_aod"

## cryo
thor_galvos = "pos_xy_THOR_gvs212"
cryo_green_laser = "laser_COBO_520"



calibration_coords_pixel = [
    [13.905, 11.931],
    [124.563, 242.424],
    [240.501, 17.871],
]
calibration_coords_green = [
    [119.616, 121.469],
    [109.97, 94.057],
    [93.88, 118.204],
]
calibration_coords_red = [
    [82.15, 83.705],
    [75.199, 61.034],
    [61.32, 79.784],
]

# Create the dictionaries using the provided lists
calibration_coords_nv1 = {
    CoordsKey.PIXEL: calibration_coords_pixel[0],
    green_laser_aod: calibration_coords_green[0],
    red_laser_aod: calibration_coords_red[0],
}

calibration_coords_nv2 = {
    CoordsKey.PIXEL: calibration_coords_pixel[1],
    green_laser_aod: calibration_coords_green[1],
    red_laser_aod: calibration_coords_red[1],
}

calibration_coords_nv3 = {
    CoordsKey.PIXEL: calibration_coords_pixel[2],
    green_laser_aod: calibration_coords_green[2],
    red_laser_aod: calibration_coords_red[2],
}

pixel_to_sample_affine_transformation_matrix = [
    [0.01476835, -0.00148369, -1.42104908],
    [0.00140560, 0.01479702, -1.73286644],
]
# endregion
# region Base config
# Add on to the default config
config |= {
    ###
    "apd_indices": [0],  # APD indices for the tagger
    "count_format": CountFormat.RAW,
    "collection_mode": CollectionMode.CAMERA,
    "collection_mode_counter": CollectionMode.COUNTER,  # TODO: remove this line when set up in new computer
    # "charge_state_estimation_mode": ChargeStateEstimationMode.MLE,
    "charge_state_estimation_mode": ChargeStateEstimationMode.THRESHOLDING,
    "windows_repo_path": home / "GitHub/dioptric",
    "disable_z_drift_compensation": False,
    ###
    # Common durations are in ns
    "CommonDurations": {
        "default_pulse_duration": 1000,
        "aod_access_time": 11e3,  # access time in specs is 10us
        "widefield_operation_buffer": 1e3,
        "uwave_buffer": 0,
        "iq_buffer": 0,
        "iq_delay": 136,  # SBC measured using NVs 4/18/2025
        "temp_reading_interval": 15 * 60,  # for PID
        # "iq_delay": 140,  # SBC measured using NVs 4/18/2025
    },
    ###
    "DeviceIDs": {
        "arb_wave_gen_visa_address": "TCPIP0::128.104.ramp_to_zero_duration.119::5025::SOCKET",
        "daq0_name": "Dev1",
        "filter_slider_THOR_ell9k_com": "COM13",
        "gcs_dll_path": home
        / "GitHub/dioptric/servers/outputs/GCSTranslator/PI_GCS2_DLL_x64.dll",
        "objective_piezo_model": "E709",
        "objective_piezo_serial": "0119008970",
        "piezo_controller_E727_model": "E727",
        "piezo_controller_E727_serial": "0121089079",
        "pulse_gen_SWAB_82_ip_1": "192.168.0.111",
        "pulse_gen_SWAB_82_ip_2": "192.168.0.160",
        "rotation_stage_THOR_ell18k_com": "COM8",
        "sig_gen_BERK_bnc835_visa": "TCPIP::128.104.ramp_to_zero_duration.114::inst0::INSTR",
        "sig_gen_STAN_sg394_visa": "TCPIP::192.168.0.120::inst0::INSTR",
        "sig_gen_STAN_sg394_2_visa": "TCPIP::192.168.0.121::inst0::INSTR",
        "sig_gen_TEKT_tsg4104a_visa": "TCPIP0::128.104.ramp_to_zero_duration.112::5025::SOCKET",
        "tagger_SWAB_20_1_serial": "1740000JEH",
        "tagger_SWAB_20_2_serial": "1948000SIP",
        "QM_opx_args": {
            "host": "192.168.0.117",
            "port": 9510,
            "cluster_name": "kolkowitz_nv_lab",
        },
    },
    ###
    "Microwaves": {
        "PhysicalSigGens": {
            "sig_gen_BERK_bnc835": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394": {"delay": 104, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394_2": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_TEKT_tsg4104a": {"delay": 57},
        },
        "iq_comp_amp": 0.5,
        "iq_delay": 140,
        "VirtualSigGens": {
            0: {
                "physical_name": "sig_gen_STAN_sg394",
                "uwave_power": 9.6,
                "frequency": 2.800,
                "rabi_period": 144,
                "pi_pulse": 72,
                "pi_on_2_pulse": 36,
            },
            # sig gen 1 is iq molulated
            1: {
                "physical_name": "sig_gen_STAN_sg394_2",
                "uwave_power": 9.6,
                "frequency": 2.8360,
                "rabi_period": 144,
                "pi_pulse": 72,
                "pi_on_2_pulse": 36,
            },
        },
    },
    ###
    "Camera": {
        "server_name": "camera_NUVU_hnu512gamma",
        "resolution": (512, 512),
        "spot_radius": 2.5,  # Radius for integrating NV counts in a camera image
        "bias_clamp": 300,  # (changing this won't actually change the value on the camera currently)
        "em_gain": 5000,
        # "em_gain": 1000,
        # "em_gain": 10,
        "temp": -60,
        "timeout": 60e3,  # ms
        # "timeout": -1,  # No timeout
        # Readout mode specifies EM vs conventional, as well as vertical and horizontal readout frequencies.
        # See camera server file for details
        "readout_mode": 1,  # 16 for double horizontal readout rate (em mode)
        # "readout_mode": 6,  # Fast conventional
        "roi": (122, 126, 250, 250),  # offsetX, offsetY, width, height
        # "roi": None,  # offsetX, offsetY, width, height
        "scale": 5 / 0.6,  # pixels / micron
    },
    ###
    "Optics": {
        "PhysicalLasers": {
            green_laser: {
                "delay": 0,
                "mod_mode": ModMode.DIGITAL,
                "positioner": green_laser_aod,
            },
            red_laser: {
                "delay": 0,
                "mod_mode": ModMode.DIGITAL,
                "positioner": red_laser_aod,
            },
            yellow_laser: {
                "delay": 0,
                "mod_mode": ModMode.ANALOG,
            },
            cryo_green_laser: {
                "delay": 0,
                "mod_mode": ModMode.DIGITAL,
                "positioner": thor_galvos,
            },
        },
        "VirtualLasers": {
            # LaserKey.IMAGING: {"physical_name": green_laser, "duration": 50e6},
            VirtualLaserKey.IMAGING: {
                # "physical_name": green_laser,
                "physical_name": cryo_green_laser,
                "duration": 12e6,
            },
            # SBC: created for calibration only
            VirtualLaserKey.RED_IMAGING: {
                "physical_name": red_laser,
                "duration": 1e6,
            },
            VirtualLaserKey.SPIN_READOUT: {
                "physical_name": green_laser,
                "duration": 300,
            },
            # LaserKey.CHARGE_POL: {"physical_name": green_laser, "duration": 10e3},
            VirtualLaserKey.CHARGE_POL: {
                "physical_name": green_laser,
                "duration": 1e3,  # Works better for Deep NVs (Johnson)
            },
            # LaserKey.CHARGE_POL: {"physical_name": green_laser, "duration": 60},
            VirtualLaserKey.SPIN_POL: {
                "physical_name": green_laser,
                "duration": 10e3,
            },
            VirtualLaserKey.SHELVING: {
                "physical_name": green_laser,
                "duration": 60,
            },
            VirtualLaserKey.ION: {
                "physical_name": red_laser,
                "duration": 1e3,
            },
            # SCC: 180 mW, 0.13 V, no shelving
            # LaserKey.SCC: {"physical_name": red_laser, "duration": 248},
            VirtualLaserKey.SCC: {
                "physical_name": red_laser,
                "duration": 124,
            },
            # LaserKey.SCC: {"physical_name": green_laser, "duration": 200},
            VirtualLaserKey.WIDEFIELD_SHELVING: {
                "physical_name": yellow_laser,
                "duration": 60,
            },
            VirtualLaserKey.WIDEFIELD_IMAGING: {
                "physical_name": yellow_laser,
                "duration": 12e6,
                # "duration": 24e6,
            },
            # LaserKey.WIDEFIELD_SPIN_POL: {"physical_name": yellow_laser, "duration": 10e3},
            VirtualLaserKey.WIDEFIELD_SPIN_POL: {
                "physical_name": yellow_laser,
                "duration": 100e3,
            },
            # LaserKey.WIDEFIELD_SPIN_POL: {"physical_name": yellow_laser, "duration": 1e6},
            VirtualLaserKey.WIDEFIELD_CHARGE_READOUT: {
                "physical_name": yellow_laser,
                # "duration": 200e6,
                # "duration": 60e6,
                "duration": 50e6,
                # "duration": 24e6,  # for red calibration
            },
            # LaserKey.WIDEFIELD_CHARGE_READOUT: {"physical_name": yellow_laser, "duration": 100e6},
        },
        #
        "PulseSettings": {
            "scc_shelving_pulse": False,  # Example setting
        },  # Whether or not to include a shelving pulse in SCC
    },
    ###
    "Positioning": {
        "Positioners": {
            CoordsKey.SAMPLE: {
                "physical_name": "pos_xyz_PI_p616_3c",
                "control_mode": PosControlMode.STREAM,
                "delay": int(1e6),  # 5 ms for PIFOC xyz
                "nm_per_unit": 1000,
                "optimize_range": 0.09,
                "units": "Voltage (V)",
                "opti_virtual_laser_key": VirtualLaserKey.IMAGING,
            },
            CoordsKey.Z: {
                "physical_name": "pos_xyz_PI_p616_3c",
                "control_mode": PosControlMode.STREAM,
                "delay": int(1e6),  # 5 ms for PIFOC xyz
                "nm_per_unit": 1000,
                # "optimize_range": 0.09,
                "optimize_range": 0.24,
                "units": "Voltage (V)",
                "opti_virtual_laser_key": VirtualLaserKey.IMAGING,
            },
            green_laser_aod: {
                "control_mode": PosControlMode.SEQUENCE,
                "delay": int(400e3),  # 400 us for galvo
                "nm_per_unit": 1000,
                "optimize_range": 1.2,
                "units": "MHz",
                "opti_virtual_laser_key": VirtualLaserKey.IMAGING,
                "aod": True,
            },
            red_laser_aod: {
                "control_mode": PosControlMode.SEQUENCE,
                "delay": int(400e3),  # 400 us for galvo
                "nm_per_unit": 1000,
                "optimize_range": 2.4,
                "units": "MHz",
                "opti_virtual_laser_key": VirtualLaserKey.ION,
                "aod": True,
            },
            thor_galvos: {
                "physical_name": "pos_xy_THOR_gvs212",
                "control_mode": PosControlMode.STREAM,
                "delay": int(400e3),  # 400 us for galvo
                "nm_per_unit": 1000,
                "optimize_range": 0.1,
                "units": "Voltage (V)",
                "opti_virtual_laser_key": VirtualLaserKey.IMAGING,
            },
        },
        "calibration_coords_nv1": calibration_coords_nv1,
        "calibration_coords_nv2": calibration_coords_nv2,
        "calibration_coords_nv3": calibration_coords_nv3,
        "pixel_to_sample_affine_transformation_matrix": pixel_to_sample_affine_transformation_matrix,
    },
    ###
    "Servers": {  # Bucket for miscellaneous servers not otherwise listed above
        "pulse_gen": "QM_opx",
        "camera": "camera_NUVU_hnu512gamma",
        "thorslm": "slm_THOR_exulus_hd2",
        "pulse_streamer": "pulse_gen_SWAB_82",
        "counter": "tagger_SWAB_20",
    },
    ###
    "Wiring": {
        "Daq": {
            # https://docs-be.ni.com/bundle/ni-67xx-scb-68a-labels/raw/resource/enus/371806a.pdf
            "ao_galvo_x": "dev1/AO22",
            "ao_galvo_y": "dev1/AO31",
            "ao_piezo_stage_P616_3c_x": "dev1/AO25",
            "ao_piezo_stage_P616_3c_y": "dev1/AO27",
            "ao_piezo_stage_P616_3c_z": "dev1/AO29",
            "ao_objective_piezo": "dev1/AO21",
            "voltage_range_factor": 10.0,
            "di_clock": "PFI12",
        },
        "Piezo_Controller_E727": {
            "piezo_controller_channel_x": 4,
            "piezo_controller_channel_y": 5,
            "piezo_controller_channel_z": 6,
            "voltage_range_factor": 10.0,
            "scaling_offset": 50.0,
            "scaling_gain": 0.5,
        },
        "PulseGen": {
            # "do_laser_INTE_520_dm": 3,
            # "do_laser_OPTO_589_dm": 3,
            # "do_laser_COBO_638_dm": 7,
            # "do_sig_gen_BERK_bnc835_gate": 1,
            # "do_sig_gen_STAN_sg394_gate": 10,
            # "do_apd_gate": 5,
            # "do_sample_clock": 0,
            # "do_camera_trigger": 5,
            # clocks / gates
            "do_sample_clock": 0,  # 125 MHz-compatible sample clock out to Tagger
            "do_apd_gate": 1,  # gate line to Tagger
            # "do_camera_trigger": 6,  # optional
            "do_laser_INTE_520_dm": 2,  # green  TTL
            "do_laser_COBO_638_dm": 3,  # red TTL
            # microwaves (TTL gate to SGs)
            # "do_sig_gen_BERK_bnc835_gate": 4,
            "do_sig_gen_STAN_sg394_2_gate": 4,
            "do_sig_gen_STAN_sg394_gate": 5,
            # analog (for the yellow AOM amplitude)
            "ao_laser_OPTO_589_am": 0,  # yellow analog modulation
        },
        "Tagger": {
            "di_clock": 1,
            "di_apd_gate": 2,
            "di_apd_0": 3,
            "di_apd_1": 4,
        },
    },
}

# endregion



if __name__ == "__main__":
    key = "pixel_to_sample_affine_transformation_matrix"
    mat = np.array(config["Positioning"][key])
    mat[:, 2] = [0, 0]
    print(mat)
    # generate_iq_pulses(["pi_pulse", "pi_on_2_pulse"], [0, 90])

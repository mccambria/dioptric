# -*- coding: utf-8 -*-
"""
Config file for the PC rabi

Created July 20th, 2023

@author: mccambria

updated by @author: sbchand
"""

from pathlib import Path

import numpy as np

from config.default import config
from utils.constants import (
    ChargeStateEstimationMode,
    CollectionMode,
    ControlMode,
    CoordsKey,
    CountFormat,
    DriftMode,
    LaserKey,
    LaserPosMode,
    ModMode,
)

home = Path.home()

# region Widefield calibration coords

green_laser = "laser_INTE_520"
yellow_laser = "laser_OPTO_589"
red_laser = "laser_COBO_638"

# clibration coords region
calibration_pixel_coords_list = [
    [73.767, 91.394],
    [142.545, 210.183],
    [234.876, 116.946],
]
calibration_green_coords_list = [
    [115.779, 106.042],
    [107.253, 118.327],
    [98.279, 107.437],
]
calibration_red_coords_list = [
    [79.256, 72.672],
    [72.849, 82.663],
    [65.189, 74.169],
]
# calibration_pixel_coords_list = [
#     [125.000, 160.887],
#     [75.302, 95.265],
#     [199.053, 94.250],
# ]
# calibration_green_coords_list = [
#     [109.504, 113.073],
#     [115.361, 106.287],
#     [102.128, 105.29],
# ]
# calibration_red_coords_list = [
#     [74.367, 78.506],
#     [78.919, 72.896],
#     [68.183, 72.361],
# ]
# Create the dictionaries using the provided lists
widefield_calibration_coords1 = {
    # CoordsKey.GLOBAL: [0, 0, 2],
    CoordsKey.PIXEL: calibration_pixel_coords_list[0],
    green_laser: calibration_green_coords_list[0],
    red_laser: calibration_red_coords_list[0],
}

widefield_calibration_coords2 = {
    # CoordsKey.GLOBAL: [0, 0, 2],
    CoordsKey.PIXEL: calibration_pixel_coords_list[1],
    green_laser: calibration_green_coords_list[1],
    red_laser: calibration_red_coords_list[1],
}

widefield_calibration_coords3 = {
    # CoordsKey.GLOBAL: [0, 0, 2],
    CoordsKey.PIXEL: calibration_pixel_coords_list[2],
    green_laser: calibration_green_coords_list[2],
    red_laser: calibration_red_coords_list[2],
}

# if coord key is CoordsKey.GLOBAL
affine_pixel2voltage = [
    [-0.01472387, 0.00052569, 1.28717911],
    [0.00040197, -0.01455135, 1.73876545],
    [0.00000000, 0.00000000, 1.00000000],
]

# endregion
# region Base config

# Add on to the default config
config |= {
    ###
    "count_format": CountFormat.RAW,
    "collection_mode": CollectionMode.CAMERA,
    # "charge_state_estimation_mode": ChargeStateEstimationMode.MLE,
    "charge_state_estimation_mode": ChargeStateEstimationMode.THRESHOLDING,
    "windows_repo_path": home / "GitHub/dioptric",
    ###
    # Common durations are in ns
    "CommonDurations": {
        "default_pulse_duration": 1000,
        "aod_access_time": 12e3,
        # "aod_access_time": 30e3,
        "widefield_operation_buffer": 1e3,
        "uwave_buffer": 16,
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
        "pulse_gen_SWAB_82_ip": "192.168.0.111",
        "rotation_stage_THOR_ell18k_com": "COM8",
        "sig_gen_BERK_bnc835_visa": "TCPIP::128.104.ramp_to_zero_duration.114::inst0::INSTR",
        "sig_gen_STAN_sg394_visa": "TCPIP::192.168.0.120::inst0::INSTR",
        "sig_gen_STAN_sg394_2_visa": "TCPIP::192.168.0.121::inst0::INSTR",
        "sig_gen_TEKT_tsg4104a_visa": "TCPIP0::128.104.ramp_to_zero_duration.112::5025::SOCKET",
        "tagger_SWAB_20_serial": "1740000JEH",
        "QM_opx_args": {
            "host": "192.168.0.117",
            "port": 9510,
            "cluster_name": "kolkowitz_nv_lab",
        },
    },
    ###
    "Microwaves": {
        "sig_gen_BERK_bnc835": {"delay": 151, "fm_mod_bandwidth": 100000.0},
        "sig_gen_STAN_sg394": {"delay": 104, "fm_mod_bandwidth": 100000.0},
        "sig_gen_TEKT_tsg4104a": {"delay": 57},
        "iq_comp_amp": 0.5,
        "iq_delay": 630,
        "sig_gen_0": {
            "name": "sig_gen_STAN_sg394",
            # "uwave_power": 6.05,
            "uwave_power": 2.3,
            "frequency": 2.8585669247525622,
            # "frequency": 2.9304468840166678,
            "rabi_period": 128,
            "iq_delay": 140,
        },
        "sig_gen_1": {
            "name": "sig_gen_STAN_sg394_2",
            "uwave_power": 8.1,
            "frequency": 2.812,
            # "frequency": 3.05,
            # "frequency": 2.79,
            # "frequency": 2.8874701085827104,
            "rabi_period": 128,
        },
    },
    ###
    "Camera": {
        "resolution": (512, 512),
        "spot_radius": 2,  # Radius for integrating NV counts in a camera image
        "bias_clamp": 300,  # (changing this won't actually change the value on the camera currently)
        "em_gain": 5000,
        # "em_gain": 1000,
        # "em_gain": 10,
        "temp": -60,
        "timeout": 30e3,  # ms
        # "timeout": -1,  # No timeout
        # Readout mode specifies EM vs conventional, as well as vertical and horizontal readout frequencies.
        # See camera server file for details
        "readout_mode": 1,  # 16 for double horizontal readout rate (em mode)
        # "readout_mode": 6,  # Fast conventional
        # "roi": (121, 110, 250, 250),  # offsetX, offsetY, width, height"roi": (121, 110, 250, 250),  # offsetX, offsetY, width, height
        "roi": (130, 105, 250, 250),  # offsetX, offsetY, width, height
        # "roi": None,  # offsetX, offsetY, width, height
        "scale": 24,  # pixels / micron
    },
    ###
    "Optics": {
        # Physical lasers
        green_laser: {
            "delay": 0,
            "mod_mode": ModMode.DIGITAL,
            "pos_mode": LaserPosMode.SCANNING,
            "aod": True,
            "default_aod_suffix": "charge_pol",
            "opti_laser_key": LaserKey.IMAGING,
        },
        red_laser: {
            "delay": 0,
            "mod_mode": ModMode.DIGITAL,
            "pos_mode": LaserPosMode.SCANNING,
            "aod": True,
            "default_aod_suffix": "scc",
            "opti_laser_key": LaserKey.ION,
        },
        yellow_laser: {
            "delay": 0,
            "mod_mode": ModMode.ANALOG,
            "pos_mode": LaserPosMode.WIDEFIELD,
        },
        CoordsKey.GLOBAL: {
            "opti_laser_key": LaserKey.IMAGING,
        },
        # Virtual lasers
        # LaserKey.IMAGING: {"name": green_laser, "duration": 50e6},
        LaserKey.IMAGING: {"name": green_laser, "duration": 12e6},
        LaserKey.SPIN_READOUT: {"name": green_laser, "duration": 300},
        # LaserKey.CHARGE_POL: {"name": green_laser, "duration": 10e3},
        LaserKey.CHARGE_POL: {"name": green_laser, "duration": 1e3},
        # LaserKey.CHARGE_POL: {"name": green_laser, "duration": 60},
        LaserKey.SPIN_POL: {"name": green_laser, "duration": 10e3},
        LaserKey.SHELVING: {"name": green_laser, "duration": 60},
        LaserKey.ION: {"name": red_laser, "duration": 10e3},
        # SCC: 180 mW, 0.13 V, no shelving
        # LaserKey.SCC: {"name": red_laser, "duration": 248},
        LaserKey.SCC: {"name": red_laser, "duration": 124},
        # LaserKey.SCC: {"name": green_laser, "duration": 200},
        LaserKey.WIDEFIELD_SHELVING: {"name": yellow_laser, "duration": 60},
        LaserKey.WIDEFIELD_IMAGING: {"name": yellow_laser, "duration": 12e6},
        # LaserKey.WIDEFIELD_SPIN_POL: {"name": yellow_laser, "duration": 10e3},
        LaserKey.WIDEFIELD_SPIN_POL: {"name": yellow_laser, "duration": 100e3},
        # LaserKey.WIDEFIELD_SPIN_POL: {"name": yellow_laser, "duration": 1e6},
        LaserKey.WIDEFIELD_CHARGE_READOUT: {"name": yellow_laser, "duration": 36e6},
        # LaserKey.WIDEFIELD_CHARGE_READOUT: {"name": yellow_laser, "duration": 100e6},
        "scc_green_shelving_pulse": False,  # Whether or not to include a shelving pulse in SCC
        "scc_yellow_shelving_pulse": False,
    },
    ###
    "Positioning": {
        "drift_mode": DriftMode.GLOBAL,
        green_laser: {
            "xy_control_mode": ControlMode.SEQUENCE,
            "xy_delay": int(400e3),  # 400 us for galvo
            "xy_dtype": float,
            "xy_nm_per_unit": 1000,
            "xy_optimize_range": 0.8,
            "xy_units": "MHz",
            "z_dtype": float,
        },
        red_laser: {
            "xy_control_mode": ControlMode.SEQUENCE,
            "xy_delay": int(400e3),  # 400 us for galvo
            "xy_dtype": float,
            "xy_nm_per_unit": 1000,
            "xy_optimize_range": 2.4,
            "xy_units": "MHz",
            "z_dtype": float,
        },
        CoordsKey.GLOBAL: {
            "z_control_mode": ControlMode.STREAM,
            "z_delay": int(1e6),  # 5 ms for PIFOC xyz
            "z_dtype": float,
            "z_nm_per_unit": 1000,
            "z_optimize_range": 0.09,
            "z_units": "Voltage (V)",
            "xy_control_mode": ControlMode.STREAM,
            "xy_delay": int(1e6),
            "xy_dtype": float,
            "xy_nm_per_unit": 1000,
            "xy_optimize_range": 0.09,
            "xy_small_response_delay": 800,
            "xy_units": "V",
            "xy_positional_accuracy": 0.002,
            "xy_timeout": 1,
        },
        "widefield_calibration_coords1": widefield_calibration_coords1,
        "widefield_calibration_coords2": widefield_calibration_coords2,
        "widefield_calibration_coords3": widefield_calibration_coords3,
        "AffineCalibration_pixel2voltage": affine_pixel2voltage,
    },
    ###
    "Servers": {
        "counter": "QM_opx",
        "magnet_rotation": "rotation_stage_thor_ell18k",
        "pos_xy": "piezo_stage_P616_3c_purcell",
        "pos_z": "piezo_stage_P616_3c_purcell",
        # "pos_xyz": "piezo_stage_P616_3c_purcell",
        # "pos_z": None,
        "pulse_gen": "QM_opx",
        "sig_gen_LOW": "sig_gen_STAN_sg394",
        "sig_gen_HIGH": "sig_gen_STAN_sg394_2",
        "sig_gen_0": "sig_gen_STAN_sg394",
        "sig_gen_1": "sig_gen_STAN_sg394_2",
        "tagger": "QM_opx",
        "camera": "camera_NUVU_hnu512gamma",
        "thorslm": "slm_THOR_exulus_hd2",
    },
    ###
    "Wiring": {
        "Daq": {
            # "ao_galvo_x": "dev1/AO0",
            # "ao_galvo_y": "dev1/AO1",
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
            "do_apd_gate": 5,
            "do_laser_INTE_520_dm": 3,
            "do_laser_OPTO_589_dm": 3,
            "do_laser_COBO_638_dm": 7,
            "do_sample_clock": 0,
            "do_sig_gen_BERK_bnc835_gate": 1,
            "do_sig_gen_STAN_sg394_gate": 10,
            "do_camera_trigger": 5,
        },
        "Tagger": {"di_apd_0": 2, "di_apd_1": 4, "di_apd_gate": 3, "di_clock": 1},
    },
}

# endregion
# region OPX config

default_pulse_duration = config["CommonDurations"]["default_pulse_duration"]
default_int_freq = 75e6
rabi_period_0 = config["Microwaves"]["sig_gen_0"]["rabi_period"]
rabi_period_1 = config["Microwaves"]["sig_gen_1"]["rabi_period"]
ramp_to_zero_duration = 64
iq_buffer = 0

opx_config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0, "delay": 0},
                2: {"offset": 0.0, "delay": 0},
                3: {"offset": 0.0, "delay": 0},
                4: {"offset": 0.0, "delay": 0},
                5: {"offset": 0.0, "delay": 0},
                6: {"offset": 0.0, "delay": 0},
                7: {"offset": 0.0, "delay": 0},
                8: {"offset": 0.0, "delay": 0},
                9: {"offset": 0.0, "delay": 0},
                10: {"offset": 0.0, "delay": 0},
            },
            "digital_outputs": {
                1: {},  #
                2: {},  #
                3: {},  #
                4: {},  #
                5: {},  #
                6: {},  #
                7: {},  #
                8: {},  #
                9: {},  #
                10: {},  #
            },
            "analog_inputs": {
                1: {"offset": 0},  # APD0
                2: {"offset": 0},  # APD1
            },
        },
    },
    "elements": {
        # region Bare channels
        "do1": {
            "digitalInputs": {"chan": {"port": ("con1", 1), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do2": {
            "digitalInputs": {"chan": {"port": ("con1", 2), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do3": {
            "digitalInputs": {"chan": {"port": ("con1", 3), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do4": {
            "digitalInputs": {"chan": {"port": ("con1", 4), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do5": {
            "digitalInputs": {"chan": {"port": ("con1", 5), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do6": {
            "digitalInputs": {"chan": {"port": ("con1", 6), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do7": {
            "digitalInputs": {"chan": {"port": ("con1", 7), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do8": {
            "digitalInputs": {"chan": {"port": ("con1", 8), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do9": {
            "digitalInputs": {"chan": {"port": ("con1", 9), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do10": {
            "digitalInputs": {"chan": {"port": ("con1", 10), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "ao1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao2": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao3": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao4": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao5": {
            "singleInput": {"port": ("con1", 5)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao6": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao7": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao8": {
            "singleInput": {"port": ("con1", 8)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao9": {
            "singleInput": {"port": ("con1", 9)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao10": {
            "singleInput": {"port": ("con1", 10)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        # endregion
        # region Actual "elements", or physical things to control
        "do_laser_COBO_638_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 1), "delay": 16, "buffer": 0}},
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "scc": "do_scc",
                "ion": "do_ion",
            },
        },
        "ao_laser_OPTO_589_am": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": 0,
            "operations": {
                "on": "yellow_imaging",
                "off": "ao_off",
                "charge_readout": "yellow_charge_readout",
                "spin_pol": "yellow_spin_pol",
                "shelving": "yellow_shelving",
            },
        },
        "ao_laser_OPTO_589_am_sticky": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": 0,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "on": "yellow_imaging",
                "off": "ao_off",
                "charge_readout": "yellow_charge_readout",
            },
        },
        "do_laser_INTE_520_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 4), "delay": 0, "buffer": 0}},
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "charge_pol": "do_charge_pol",
                "spin_pol": "do_green_spin_pol",
                "shelving": "do_shelving",
                "scc": "do_scc",
            },
        },
        "do_sig_gen_STAN_sg394_dm": {
            "digitalInputs": {
                "chan": {
                    "port": ("con1", 10),
                    "delay": 0,
                    # "delay": config["Microwaves"]["sig_gen_0"]["iq_delay"]
                    # + iq_buffer // 2,
                    "buffer": 0,
                }
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "pi_pulse": "do_pi_pulse_0",
                "pi_on_2_pulse": "do_pi_on_2_pulse_0",
                "pi_on_2_pulse_b": "do_pi_on_2_pulse_b_0",
            },
        },
        "ao_sig_gen_STAN_sg394_i": {
            "singleInput": {"port": ("con1", 9)},
            "intermediate_frequency": 0,
            # "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "iq_pi_pulse_0",
                "pi_on_2_pulse": "iq_pi_on_2_pulse_0",
            },
        },
        "ao_sig_gen_STAN_sg394_q": {
            "singleInput": {"port": ("con1", 10)},
            "intermediate_frequency": 0,
            # "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "iq_pi_pulse_0",
                "pi_on_2_pulse": "iq_pi_on_2_pulse_0",
            },
        },
        "do_sig_gen_STAN_sg394_2_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 9), "delay": 0, "buffer": 0}},
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "pi_pulse": "do_pi_pulse_1",
                "pi_on_2_pulse": "do_pi_on_2_pulse_1",
                "pi_on_2_pulse_b": "do_pi_on_2_pulse_b_1",
            },
        },
        "do_camera_trigger": {
            "digitalInputs": {"chan": {"port": ("con1", 5), "delay": 0, "buffer": 0}},
            "sticky": {
                "analog": True,
                "digital": True,
                "duration": ramp_to_zero_duration,
            },
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "ao_laser_COBO_638_x": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": 75e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "aod_cw-opti": "red_aod_cw-opti",
                "aod_cw-ion": "red_aod_cw-ion",
                "aod_cw-scc": "red_aod_cw-scc",
                "continue": "ao_off",
            },
        },
        "ao_laser_COBO_638_y": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": 75e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "aod_cw-opti": "red_aod_cw-opti",
                "aod_cw-ion": "red_aod_cw-ion",
                "aod_cw-scc": "red_aod_cw-scc",
                "continue": "ao_off",
            },
        },
        "ao_laser_INTE_520_x": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": 110e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "aod_cw-opti": "green_aod_cw-opti",
                "aod_cw-charge_pol": "green_aod_cw-charge_pol",
                "aod_cw-spin_pol": "green_aod_cw-spin_pol",
                "aod_cw-shelving": "green_aod_cw-shelving",
                "aod_cw-scc": "green_aod_cw-scc",
                "continue": "ao_off",
            },
        },
        "ao_laser_INTE_520_y": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": 110e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "aod_cw-opti": "green_aod_cw-opti",
                "aod_cw-charge_pol": "green_aod_cw-charge_pol",
                "aod_cw-spin_pol": "green_aod_cw-spin_pol",
                "aod_cw-shelving": "green_aod_cw-shelving",
                "aod_cw-scc": "green_aod_cw-scc",
                "continue": "ao_off",
            },
        },
        # endregion
    },
    # region Pulses
    "pulses": {
        ### Analog
        # Green
        "green_aod_cw-opti": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw-opti"},
        },
        "green_aod_cw-charge_pol": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw-charge_pol"},
        },
        "green_aod_cw-spin_pol": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw-spin_pol"},
        },
        "green_aod_cw-shelving": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw-shelving"},
        },
        "green_aod_cw-scc": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw-scc"},
        },
        # Red
        "red_aod_cw-opti": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "red_aod_cw-opti"},
        },
        "red_aod_cw-ion": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "red_aod_cw-ion"},
        },
        "red_aod_cw-scc": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "red_aod_cw-scc"},
        },
        # Yellow
        "yellow_imaging": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "yellow_imaging"},
        },
        "yellow_charge_readout": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "yellow_charge_readout"},
        },
        "yellow_spin_pol": {
            "operation": "control",
            "length": config["Optics"][LaserKey.WIDEFIELD_SPIN_POL]["duration"],
            "waveforms": {"single": "yellow_spin_pol"},
        },
        "yellow_shelving": {
            "operation": "control",
            "length": config["Optics"][LaserKey.WIDEFIELD_SHELVING]["duration"],
            "waveforms": {"single": "yellow_shelving"},
        },
        #
        "ao_cw": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "cw"},
        },
        "ao_off": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "off"},
        },
        "iq_pi_pulse_0": {
            "operation": "control",
            "length": int(rabi_period_0 / 2) + iq_buffer,
            "waveforms": {"single": "cw"},
        },
        "iq_pi_on_2_pulse_0": {
            "operation": "control",
            "length": int(rabi_period_0 / 4) + iq_buffer,
            # "length": 20,
            "waveforms": {"single": "cw"},
        },
        ### Digital
        "do_on": {
            "operation": "control",
            "length": default_pulse_duration,
            "digital_marker": "on",
        },
        "do_off": {
            "operation": "control",
            "length": default_pulse_duration,
            "digital_marker": "off",
        },
        "do_short_pulse": {
            "operation": "control",
            "length": default_pulse_duration,
            "digital_marker": "square",
        },
        "do_scc": {
            "operation": "control",
            "length": config["Optics"][LaserKey.SCC]["duration"],
            "digital_marker": "on",
        },
        "do_ion": {
            "operation": "control",
            "length": config["Optics"][LaserKey.ION]["duration"],
            "digital_marker": "on",
        },
        "do_charge_pol": {
            "operation": "control",
            "length": config["Optics"][LaserKey.CHARGE_POL]["duration"],
            "digital_marker": "on",
        },
        "do_shelving": {
            "operation": "control",
            "length": config["Optics"][LaserKey.SHELVING]["duration"],
            "digital_marker": "on",
        },
        "do_green_spin_pol": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "on",
        },
        "do_pi_pulse_0": {
            "operation": "control",
            "length": int(rabi_period_0 / 2) + 4,
            "digital_marker": "on",
        },
        "do_pi_on_2_pulse_0": {
            "operation": "control",
            "length": int(rabi_period_0 / 4) + 4,
            # "length": int(rabi_period_0 / 2) + 4,
            # "length": 20,
            "digital_marker": "on",
        },
        "do_pi_on_2_pulse_b_0": {
            "operation": "control",
            "length": int(rabi_period_0 / 4),
            # "length": int(rabi_period_0 / 2),
            # "length": 20,
            "digital_marker": "on",
        },
        "do_pi_pulse_1": {
            "operation": "control",
            "length": int(rabi_period_1 / 2) + 4,
            "digital_marker": "on",
        },
        "do_pi_on_2_pulse_1": {
            "operation": "control",
            "length": int(rabi_period_1 / 4) + 4,
            # "length": int(rabi_period_1 / 2) + 4,
            # "length": 20,
            "digital_marker": "on",
        },
        "do_pi_on_2_pulse_b_1": {
            "operation": "control",
            "length": int(rabi_period_1 / 4),
            # "length": int(rabi_period_1 / 2),
            # "length": 20,
            "digital_marker": "on",
        },
        ### Mixed
    },
    # endregion
    # region Waveforms
    ### Analog
    "waveforms": {
        # Green AOD
        "green_aod_cw-opti": {"type": "constant", "sample": 0.09},
        # "green_aod_cw-opti": {"type": "constant", "sample": 0.07},
        # "green_aod_cw-opti": {"type": "constant", "sample": 0.05},
        # "green_aod_cw-opti": {"type": "constant", "sample": 0.03},
        # "green_aod_cw-charge_pol": {"type": "constant", "sample": 0.13},
        # "green_aod_cw-charge_pol": {"type": "constant", "sample": 0.06},  # Negative
        "green_aod_cw-charge_pol": {"type": "constant", "sample": 0.11},
        "green_aod_cw-spin_pol": {"type": "constant", "sample": 0.05},
        "green_aod_cw-shelving": {"type": "constant", "sample": 0.05},
        "green_aod_cw-scc": {"type": "constant", "sample": 0.15},
        # Red AOD
        # "red_aod_cw-opti": {"type": "constant", "sample": 0.10},
        "red_aod_cw-opti": {"type": "constant", "sample": 0.12},
        # "red_aod_cw-ion": {"type": "constant", "sample": 0.09},
        "red_aod_cw-ion": {"type": "constant", "sample": 0.12},
        # "red_aod_cw-scc": {"type": "constant", "sample": 0.135},
        "red_aod_cw-scc": {"type": "constant", "sample": 0.12},
        # Yellow AOM
        "yellow_imaging": {"type": "constant", "sample": 0.45},  # 0.35
        # "yellow_imaging": {"type": "constant", "sample": 0.50},  # 0.35
        "yellow_charge_readout": {"type": "constant", "sample": 0.38},  # 50e6
        # "yellow_charge_readout": {"type": "constant", "sample": 0.25},  # 100e6
        "yellow_spin_pol": {"type": "constant", "sample": 0.39},
        "yellow_shelving": {"type": "constant", "sample": 0.33},
        # Other
        "aod_cw": {"type": "constant", "sample": 0.35},
        "cw": {"type": "constant", "sample": 0.5},
        "off": {"type": "constant", "sample": 0.0},
    },
    ### Digital, format is list of tuples: (on/off, ns)
    "digital_waveforms": {
        "on": {"samples": [(1, 0)]},
        "off": {"samples": [(0, 0)]},
        "square": {"samples": [(1, 100), (0, 100)]},
    },
    # endregion
}
# endregion

ref_img_array = np.array([])


if __name__ == "__main__":  #
    # print(config)
    print(config["DeviceIDs"]["gcs_dll_path"])

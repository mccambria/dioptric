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


calibration_coords_pixel = [
    [27.44, 23.014],
    [108.384, 227.38],
    [227.438, 19.199],
]
calibration_coords_green = [
    [118.088, 97.57],
    [107.064, 118.434],
    [96.815, 94.848],
]
calibration_coords_red = [
    [80.674, 64.866],
    [72.142, 81.956],
    [63.271, 62.874],
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

# pixel_to_sample_affine_transformation_matrix = [
#     [-0.01472387, 0.00052569, 1.28717911],
#     [0.00040197, -0.01455135, 1.73876545],
# ]

pixel_to_sample_affine_transformation_matrix = [
    [-0.01478989, -0.00127903, 3.85073374],
    [0.00138559, -0.01471450, 5.74068819],
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
    "disable_z_drift_compensation": False,
    ###
    # Common durations are in ns
    "CommonDurations": {
        "default_pulse_duration": 1000,
        "aod_access_time": 11e3,  # access time in specs is 10us
        "widefield_operation_buffer": 1e3,
        "uwave_buffer": 16,
        "iq_buffer": 0,
        "iq_delay": 136,  # SBC measured using NVs 4/18/2025
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
        "PhysicalSigGens": {
            "sig_gen_BERK_bnc835": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394": {"delay": 104, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394_2": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_TEKT_tsg4104a": {"delay": 57},
        },
        "iq_comp_amp": 0.5,
        "iq_delay": 140,  # SBC measured using NVs 4/18/2025
        "VirtualSigGens": {
            0: {
                "physical_name": "sig_gen_STAN_sg394",
                # "uwave_power": 2.3,
                "uwave_power": 8.7,
                # "frequency": 2.779138,  # rubin shallow NVs O1 ms=-1
                # "frequency": 2.909381,  # rubin shallow NV O3 ms=+1
                "frequency": 2.730700,
                "rabi_period": 128,
            },
            # sig gen 1 is iq molulated
            1: {
                "physical_name": "sig_gen_STAN_sg394_2",
                "uwave_power": 8.7,
                # "uwave_power": 9.6,
                # "frequency": 2.779138,   # rubin shallow NVs O1 ms=-1
                # "frequency": 2.964545,  # rubin shallow NV O1 ms=+1
                # "frequency": 2.842478,  # rubin shallow NV O3 ms=-1
                "frequency": 2.730700,  # lower esr peak for both orientation
                "rabi_period": 208,
                "pi_pulse": 104,
                "pi_on_2_pulse": 56,
                # "rabi_period": 52,
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
        "timeout": 30e3,  # ms
        # "timeout": -1,  # No timeout
        # Readout mode specifies EM vs conventional, as well as vertical and horizontal readout frequencies.
        # See camera server file for details
        "readout_mode": 1,  # 16 for double horizontal readout rate (em mode)
        # "readout_mode": 6,  # Fast conventional
        "roi": (134, 105, 250, 250),  # offsetX, offsetY, width, height
        # "roi": None,  # offsetX, offsetY, width, height
        "scale": 5 / 0.6,  # pixels / micron
        # "scale": 1 / 0.072,  # pixels / micron
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
        },
        "VirtualLasers": {
            # LaserKey.IMAGING: {"physical_name": green_laser, "duration": 50e6},
            VirtualLaserKey.IMAGING: {
                "physical_name": green_laser,
                "duration": 12e6,
            },
            VirtualLaserKey.SPIN_READOUT: {
                "physical_name": green_laser,
                "duration": 300,
            },
            # LaserKey.CHARGE_POL: {"physical_name": green_laser, "duration": 10e3},
            VirtualLaserKey.CHARGE_POL: {
                "physical_name": green_laser,
                "duration": 1e3,  # Works better for shallow NVs (Cannon)
                # "duration": 500,  # Works better for shallow NVs (Cannon)
                # "duration": 10e3, #Works better for Deep NVs (Johnson)
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
                # "duration": 12e6,
                "duration": 24e6,
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
                "duration": 60e6,
                # "duration": 30e6,
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
                "optimize_range": 2.4,
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
virtual_sig_gens_dict = config["Microwaves"]["VirtualSigGens"]
num_sig_gens = len(virtual_sig_gens_dict)
rabi_period_0 = virtual_sig_gens_dict[0]["rabi_period"]
rabi_period_1 = virtual_sig_gens_dict[1]["rabi_period"]
pi_pulse_1 = virtual_sig_gens_dict[1]["pi_pulse"]
pi_on_2_pulse_1 = virtual_sig_gens_dict[1]["pi_on_2_pulse"]
ramp_to_zero_duration = 64
virtual_lasers_dict = config["Optics"]["VirtualLasers"]
iq_buffer = config["CommonDurations"]["iq_buffer"]
iq_delay = config["CommonDurations"]["iq_delay"]


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
            "digitalInputs": {"chan": {"port": ("con1", 10), "delay": 0, "buffer": 0}},
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "pi_pulse": "do_pi_pulse_0",
                "pi_on_2_pulse": "do_pi_on_2_pulse_0",
            },
        },
        "do_sig_gen_STAN_sg394_2_dm": {
            # 230 ns I channel latency measured 3/26/25 MCC and Saroj using oscilloscope
            "digitalInputs": {
                # "chan": {"port": ("con1", 9), "delay": 230, "buffer": 0}
                "chan": {"port": ("con1", 9), "delay": iq_delay, "buffer": 0}
            },
            "operations": {
                "iq_test": "do_iq_test",
                "on": "do_on",
                "off": "do_off",
                "pi_pulse": "do_pi_pulse_1",
                "pi_on_2_pulse": "do_pi_on_2_pulse_1",
            },
        },
        # region Microwave iq modulation
        # Additional operations are generated algorithmically with generate_iq_pulses()
        "ao_sig_gen_STAN_sg394_i": {
            "singleInput": {"port": ("con1", 5)},
            "intermediate_frequency": 0,
            "operations": {
                "iq_test": "iq_test",
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "ao_iq_pi_pulse_0",
                "pi_on_2_pulse": "ao_iq_pi_on_2_pulse_0",
            },
        },
        "ao_sig_gen_STAN_sg394_q": {
            "singleInput": {"port": ("con1", 8)},
            "intermediate_frequency": 0,
            "operations": {
                "iq_test": "iq_test",
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "ao_iq_pi_pulse_0",
                "pi_on_2_pulse": "ao_iq_pi_on_2_pulse_0",
            },
        },
        "ao_sig_gen_STAN_sg394_2_i": {
            "singleInput": {"port": ("con1", 9)},
            "intermediate_frequency": 0,
            "operations": {
                "iq_test": "iq_test",
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "ao_iq_pi_pulse_1",
                "pi_on_2_pulse": "ao_iq_pi_on_2_pulse_1",
            },
        },
        "ao_sig_gen_STAN_sg394_2_q": {
            "singleInput": {"port": ("con1", 10)},
            "intermediate_frequency": 0,
            "operations": {
                "iq_test": "iq_test",
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "ao_iq_pi_pulse_1",
                "pi_on_2_pulse": "ao_iq_pi_on_2_pulse_1",
            },
        },
        # endregion
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
                "aod_cw": "red_aod_cw-scc",
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
                "aod_cw": "red_aod_cw-scc",
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
                "aod_cw": "green_aod_cw-charge_pol",
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
                "aod_cw": "green_aod_cw-charge_pol",
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
            "length": virtual_lasers_dict[VirtualLaserKey.WIDEFIELD_SPIN_POL][
                "duration"
            ],
            "waveforms": {"single": "yellow_spin_pol"},
        },
        "yellow_shelving": {
            "operation": "control",
            "length": virtual_lasers_dict[VirtualLaserKey.WIDEFIELD_SHELVING][
                "duration"
            ],
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
        "iq_test": {
            "operation": "control",
            "length": 10000,
            "waveforms": {"single": "cw"},
        },
        "ao_iq_pi_pulse_0": {
            "operation": "control",
            "length": int(rabi_period_0 / 2) + 2 * iq_buffer,
            "waveforms": {"single": "cw"},
        },
        "ao_iq_pi_on_2_pulse_0": {
            "operation": "control",
            "length": int(rabi_period_0 / 4) + 2 * iq_buffer,
            "waveforms": {"single": "cw"},
        },
        "ao_iq_pi_pulse_1": {
            "operation": "control",
            "length": int(pi_pulse_1) + 2 * iq_buffer,
            "waveforms": {"single": "cw"},
        },
        "ao_iq_pi_on_2_pulse_1": {
            "operation": "control",
            "length": int(pi_on_2_pulse_1) + 2 * iq_buffer,
            "waveforms": {"single": "cw"},
        },
        ### Digital
        "do_iq_test": {
            "operation": "control",
            "length": 10000,
            "digital_marker": "on",
        },
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
            "length": virtual_lasers_dict[VirtualLaserKey.SCC]["duration"],
            "digital_marker": "on",
        },
        "do_ion": {
            "operation": "control",
            "length": virtual_lasers_dict[VirtualLaserKey.ION]["duration"],
            "digital_marker": "on",
        },
        "do_charge_pol": {
            "operation": "control",
            "length": virtual_lasers_dict[VirtualLaserKey.CHARGE_POL]["duration"],
            "digital_marker": "on",
        },
        "do_shelving": {
            "operation": "control",
            "length": virtual_lasers_dict[VirtualLaserKey.SHELVING]["duration"],
            "digital_marker": "on",
        },
        "do_green_spin_pol": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "on",
        },
        "do_esr_pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "on",
        },
        "do_pi_pulse_0": {
            "operation": "control",
            "length": int(rabi_period_0 / 2),
            "digital_marker": "on",
        },
        "do_pi_on_2_pulse_0": {
            "operation": "control",
            "length": int(rabi_period_0 / 4),
            "digital_marker": "on",
        },
        "do_pi_pulse_1": {
            "operation": "control",
            "length": int(pi_pulse_1),
            "digital_marker": "on",
        },
        "do_pi_on_2_pulse_1": {
            "operation": "control",
            "length": int(pi_on_2_pulse_1),
            "digital_marker": "on",
        },
        ### Mixed
    },
    # endregion
    # region Waveforms
    ### Analog
    "waveforms": {
        # Green AOD
        "green_aod_cw-opti": {"type": "constant", "sample": 0.11},
        # "green_aod_cw-opti": {"type": "constant", "sample": 0.07},
        # "green_aod_cw-charge_pol": {"type": "constant", "sample": 0.06},  # Negative
        "green_aod_cw-charge_pol": {"type": "constant", "sample": 0.139},  # median
        # "green_aod_cw-charge_pol": {"type": "constant", "sample": 0.15},  # median
        "green_aod_cw-spin_pol": {"type": "constant", "sample": 0.05},
        "green_aod_cw-shelving": {"type": "constant", "sample": 0.05},
        "green_aod_cw-scc": {"type": "constant", "sample": 0.15},
        # Red AOD
        # "red_aod_cw-opti": {"type": "constant", "sample": 0.10},
        "red_aod_cw-opti": {"type": "constant", "sample": 0.15},
        # "red_aod_cw-ion": {"type": "constant", "sample": 0.09},
        "red_aod_cw-ion": {"type": "constant", "sample": 0.15},
        "red_aod_cw-scc": {"type": "constant", "sample": 0.15},
        # "red_aod_cw-scc": {"type": "constant", "sample": 0.12},  # rubin
        # Yellow AOM
        "yellow_imaging": {"type": "constant", "sample": 0.45},  # 0.35
        # "yellow_imaging": {"type": "constant", "sample": 0.50},  # 0.35
        # "yellow_charge_readout": {"type": "constant", "sample": 0.396},  # 50ms 160NVs
        # "yellow_charge_readout": {"type": "constant", "sample": 0.4129},  # 50ms 160NVs
        # "yellow_charge_readout": {"type": "constant", "sample": 0.3472},  # 50ms 117NVs
        # "yellow_charge_readout": {"type": "constant", "sample": 0.3741},  # 50ms 117NVs
        # "yellow_charge_readout": {"type": "constant", "sample": 0.325},  # 100ms 117NVs
        # "yellow_charge_readout": {"type": "constant", "sample": 0.37240},  # 150NVs all
        # "yellow_charge_readout": {"type": "constant", "sample": 0.35736},  # 75NVs new
        "yellow_charge_readout": {"type": "constant", "sample": 0.3600},  # 75NVs new
        # "yellow_charge_readout": {"type": "constant", "sample": 0.32350},  # 35NV/185MHz
        # "yellow_charge_readout": {"type": "constant", "sample": 0.32238},  # 48NV/68MHz
        "yellow_spin_pol": {"type": "constant", "sample": 0.44},  # 75 NVs
        # "yellow_spin_pol": {"type": "constant", "sample": 0.42},
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


# Centralized pulse error values (from bootstrap tomography)
# def correct_pulse_params_by_phase(phase_deg, base_amp=0.5):
#     # Determine which pulse axis the phase aligns with
#     pulse_errors = {
#         "phi_prime": -0.091249,
#         "chi_prime": 0.047725,
#         "phi": 0.08768,
#         "chi": -0.046245,
#         "vz": -0.041764,
#         "ez": -0.081076,
#         "epsilon_z_prime": -0.010492,
#         "nu_x_prime": -0.001294,
#         "nu_z_prime": -0.15388,
#         "epsilon_y": 0.014685,
#         "nu_x": -0.018655,
#     }
#     # return amp_correction, phase_correction_rad
#     phase_mod = phase_deg % 360

#     if np.isclose(phase_mod, 0, atol=5) or np.isclose(phase_mod, 180, atol=5):
#         # X-aligned pulses
#         angle_error = pulse_errors.get("phi", 0)
#         tilt = pulse_errors.get("ez", 0)
#     elif np.isclose(phase_mod, 90, atol=5) or np.isclose(phase_mod, 270, atol=5):
#         # Y-aligned pulses
#         angle_error = pulse_errors.get("chi", 0)
#         tilt = pulse_errors.get("epsilon_y", 0)
#     else:
#         raise ValueError(
#             f"Unsupported phase: {phase_deg}. Expected near 0, 90, 180, 270 degrees."
#         )

#     amp_correction = base_amp * (1.0 / (1 + angle_error))
#     phase_correction_rad = -np.arctan2(tilt, 1.0)
#     print(phase_correction_rad)
#     return amp_correction, phase_correction_rad


def correct_pulse_params_by_phase(phase_deg, base_amp=0.5):
    # Centralized pulse error values from bootstrap
    pulse_errors = {
        "phi": 0.08768,  # amplitude error on X
        "chi": -0.046245,  # amplitude error on Y
        "ez": -0.081076,  # Z-axis tilt for X
        "epsilon_y": 0.014685,  # Z-axis tilt for Y
    }

    # Normalize phase to [0, 360)
    phase_mod = phase_deg % 360

    # Determine closest axis: X-like or Y-like
    # X-like: around 0° or 180°, Y-like: around 90° or 270°
    # Use shortest angular distance modulo 180°
    # distance_to_x = min(abs(phase_mod - 0), abs(phase_mod - 180))
    # distance_to_y = min(abs(phase_mod - 90), abs(phase_mod - 270))

    # if distance_to_x <= distance_to_y:
    # X-aligned rotation
    angle_error = pulse_errors.get("phi", 0)
    tilt = pulse_errors.get("ez", 0)
    # else:
    # Y-aligned rotation
    angle_error = pulse_errors.get("chi", 0)
    tilt = pulse_errors.get("epsilon_y", 0)

    # Apply amplitude and phase correction
    amp_correction = base_amp * (1.0 / (1 + angle_error))
    phase_correction_rad = -np.arctan2(tilt, 1.0)

    return amp_correction, phase_correction_rad


def generate_iq_pulses(pulse_names, phases):
    """Adds iq pulses to opx_config for the passed phases to match the microwave pulses
    already defined manually in the config. The pulses names that are intended for use
    are of the form f"{pulse_name}_{phase}" with duration equal to that of the pulse
    with pulse_name defined on the corresponding digital modulation element for a
    given microwave channel

    Parameters
    ----------
    pulse_names : list(str)
        List of microwave pulse names
    phases : list(int)
        List of phases in degrees. Expects integers.
    """
    # Define the waveforms
    amp = 0.5

    for phase in phases:
        # amp_corr, phase_corr = correct_pulse_params_by_phase(phase, base_amp=0.5)
        # corrected_phase_rad = np.round(np.deg2rad(phase) + phase_corr)  # note: addition

        # i_comp = np.cos(corrected_phase_rad) * amp_corr
        # q_comp = np.sin(corrected_phase_rad) * amp_corr
        # print(
        #     f"Phase {phase}° → corrected amp: {amp_corr:.3f}, phase (deg): {np.rad2deg(corrected_phase_rad):.2f}"
        # )
        i_comp = np.cos(np.deg2rad(phase)) * amp
        q_comp = np.sin(np.deg2rad(phase)) * amp
        opx_config["waveforms"][f"i_{phase}"] = {"type": "constant", "sample": i_comp}
        opx_config["waveforms"][f"q_{phase}"] = {"type": "constant", "sample": q_comp}

    # Define the pulses and add the pulses to the elements
    for comp in ["i", "q"]:
        for pulse_name in pulse_names:
            for phase in phases:
                for chan in range(num_sig_gens):
                    # Define the pulse
                    full_pulse_name = f"ao_{comp}_{pulse_name}_{phase}_{chan}"
                    # print(full_pulse_name)
                    length = opx_config["pulses"][f"do_{pulse_name}_{chan}"]["length"]
                    # print(length)
                    opx_config["pulses"][full_pulse_name] = {
                        "operation": "control",
                        "length": length,
                        "waveforms": {"single": f"{comp}_{phase}"},
                    }
                    # Add the pulse to the element
                    dev = virtual_sig_gens_dict[chan]["physical_name"]
                    opx_config["elements"][f"ao_{dev}_{comp}"]["operations"][
                        f"{pulse_name}_{phase}"
                    ] = full_pulse_name


# ref_img_array = np.array([])
# generate_iq_pulses(["pi_pulse", "pi_on_2_pulse"], [0, 90, 180, 270])
# fmt: off
phases =[0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342, 360]
# fmt:on
generate_iq_pulses(["pi_pulse", "pi_on_2_pulse"], phases)


if __name__ == "__main__":
    key = "pixel_to_sample_affine_transformation_matrix"
    mat = np.array(config["Positioning"][key])
    mat[:, 2] = [0, 0]
    print(mat)
    # generate_iq_pulses(["pi_pulse", "pi_on_2_pulse"], [0, 90])

# -*- coding: utf-8 -*-
"""
Config file for the PC rabi

Created July 20th, 2023

@author: mccambria
"""

from utils.constants import ModMode, ControlMode, CountFormat
from utils.constants import CollectionMode, LaserKey, LaserPosMode
from pathlib import Path
import numpy as np

home = Path.home()

# region Widefield calibration NVs

green_laser = "laser_INTE_520"
yellow_laser = "laser_OPTO_589"
red_laser = "laser_COBO_638"

pixel_coords_key = "pixel_coords"
green_coords_key = f"coords-{green_laser}"
red_coords_key = f"coords-{red_laser}"

# Imaging laser dicts
yellow_laser_dict = {"name": yellow_laser, "readout_dur": 50e6, "num_reps": 1}
green_laser_dict = {"name": green_laser, "readout_dur": 10e6, "num_reps": 10}
red_laser_dict = {"name": red_laser, "readout_dur": 1e6, "num_reps": 1}

sample_name = "johnson"
z_coord = 2.9

widefield_calibration_nv_shell = {
    "coords": [None, None, z_coord],
    "name": f"{sample_name}-nvref",
    "disable_opt": False,
    "disable_z_opt": True,
    "expected_count_rate": None,
    #
    LaserKey.IMAGING: yellow_laser_dict,
    # LaserKey.IMAGING: green_laser_dict,
    # LaserKey.IMAGING: red_laser_dict,
    #
    LaserKey.SPIN_READOUT: {
        "name": green_laser,
        "pol_dur": 2e3,
        "readout_dur": 440,
    },
    LaserKey.IONIZATION: {
        "name": red_laser,
        "ion_dur": 2e3,
    },  # 50 mW setting for 10 mW on table
    LaserKey.POLARIZATION: {
        "name": green_laser,
    },
}

widefield_calibration_nv_shell = {
    "name": "widefield_calibration_nv1",
    "disable_opt": False,
    "disable_z_opt": True,
    "expected_count_rate": 20500,
    LaserKey.IMAGING: {
        "name": "laser_INTE_520",
        "readout_dur": 1e7,
        "num_reps": 100,
        "filter": "nd_0",
    },
    "collection_filter": None,
    "magnet_angle": None,
}
widefield_calibration_nv1 = widefield_calibration_nv_shell.copy()
widefield_calibration_nv2 = widefield_calibration_nv_shell.copy()
widefield_calibration_nv2["name"] = "widefield_calibration_nv2"
widefield_calibration_nv1["disable_z_opt"] = False
widefield_calibration_nv2["disable_z_opt"] = True

# Coords
widefield_calibration_nv1[pixel_coords_key] = [267.929, 290.489]
widefield_calibration_nv1[green_coords_key] = [110.011, 110.845]
# widefield_calibration_nv1[red_coords_key] = [73.8, 76.2]
widefield_calibration_nv1[red_coords_key] = [73.8, 74.6]
widefield_calibration_nv2[pixel_coords_key] = [217.197, 331.628]
widefield_calibration_nv2[green_coords_key] = [108.3, 112.002]
# widefield_calibration_nv2[red_coords_key] = [72.6, 77.1]
widefield_calibration_nv2[red_coords_key] = [72.6, 75.5]

# endregion
# region Base config

config = {
    ###
    "apd_indices": [0],
    "count_format": CountFormat.RAW,
    "collection_mode": CollectionMode.CAMERA,
    "camera_spot_radius": 6,  # Distance to first Airy zero in units of camera pixels for diffraction-limited spot
    "nv_sig_units": "{'coords': 'V', 'expected_count_rate': 'kcps', 'durations': 'ns', 'magnet_angle': 'deg', 'resonance': 'GHz', 'rabi': 'ns', 'uwave_power': 'dBm'}",
    "shared_email": "kolkowitznvlab@gmail.com",
    # Access the OS-specific keys with getters from common
    "windows_nvdata_path": Path("E:/Shared drives/Kolkowitz Lab Group/nvdata"),
    "linux_nvdata_path": home / "E/nvdata",
    "windows_repo_path": home / "Documents/GitHub/dioptric",
    "linux_repo_path": home / "Documents/GitHub/dioptric",
    ###
    "CommonDurations": {
        "cw_meas_buffer": 5000,
        "pol_to_uwave_wait_dur": 5000,
        "scc_ion_readout_buffer": 10000,
        "uwave_buffer": 1000,
        "uwave_to_readout_wait_dur": 5000,
        "aod_rise_time": int(12e3),
        "aod_end_buffer": int(2e3),
        "default_pulse_duration": 1000,
        "aod_access_time": 20e3,
    },
    ###
    "DeviceIDs": {
        "arb_wave_gen_visa_address": "TCPIP0::128.104.160.119::5025::SOCKET",
        "daq0_name": "Dev1",
        "filter_slider_THOR_ell9k_com": "COM13",
        "gcs_dll_path": home
        / "Documents/GitHub/dioptric/servers/outputs/GCSTranslator/PI_GCS2_DLL_x64.dll",
        "objective_piezo_model": "E709",
        "objective_piezo_serial": "0119008970",
        "pulse_gen_SWAB_82_ip": "192.168.0.111",
        "rotation_stage_ell18k_address": "COM6",
        "sig_gen_BERK_bnc835_visa": "TCPIP::128.104.160.114::inst0::INSTR",
        "sig_gen_STAN_sg394_visa": "TCPIP::192.168.0.112::inst0::INSTR",
        "sig_gen_TEKT_tsg4104a_visa": "TCPIP0::128.104.160.112::5025::SOCKET",
        "tagger_SWAB_20_serial": "1740000JEH",
        "QM_opx_ip": "192.168.0.117",
    },
    ###
    "Microwaves": {
        "sig_gen_BERK_bnc835": {"delay": 151, "fm_mod_bandwidth": 100000.0},
        "sig_gen_STAN_sg394": {"delay": 104, "fm_mod_bandwidth": 100000.0},
        "sig_gen_TEKT_tsg4104a": {"delay": 57},
        "iq_comp_amp": 0.5,
        "iq_delay": 630,
        "sig_gen_HIGH": "sig_gen_TEKT_tsg4104a",
        "sig_gen_LOW": "sig_gen_TEKT_tsg4104a",
    },
    ###
    "Camera": {
        "resolution": (512, 512),
        "max_frame_rate": 15,  # In Hz - faster than this and the camera may crash
    },
    ###
    "Optics": {
        "collection": {
            "filter_server": "filter_slider_ell9k_3",
            "filter_mapping": {
                "514_notch+630_lp": 0,
                "740_bp": 1,
                "715_lp": 2,
                "no_filter": 3,
            },
        },
        "laser_INTE_520": {
            "delay": 250,
            "mod_mode": ModMode.DIGITAL,
            "pos_mode": LaserPosMode.SCANNING,
            "filter_server": "filter_slider_THOR_ell9k",
            "filter_mapping": {"nd_0": 0, "nd_0.3": 1, "nd_0.7": 2, "nd_1.0": 3},
        },
        # "laser_OPTO_589": {
        #     "delay": 2500,
        #     "mod_mode": ModMode.DIGITAL,
        #     "pos_mode": LaserPosMode.WIDEFIELD,
        #     "filter_server": "filter_slider_ell9k",
        #     "filter_mapping": {"nd_0": 0, "nd_0.5": 1, "nd_1.0": 2, "nd_1.5": 3},
        # },
        "laser_OPTO_589": {
            "delay": 2500,
            "mod_mode": ModMode.ANALOG,
            "pos_mode": LaserPosMode.WIDEFIELD,
            "filter_server": "filter_slider_ell9k",
            "filter_mapping": {"nd_0": 0, "nd_0.5": 1, "nd_1.0": 2, "nd_1.5": 3},
        },
        "laser_COBO_638": {
            "delay": 250,
            "mod_mode": ModMode.DIGITAL,
            "pos_mode": LaserPosMode.SCANNING,
            "filter_server": "filter_slider_THOR_ell9k",
            "filter_mapping": {"nd_0": 0, "nd_0.3": 1, "nd_0.7": 2, "nd_1.0": 3},
        },
    },
    ###
    "Positioning": {
        #
        "xy_control_mode-laser_INTE_520": ControlMode.SEQUENCE,
        "xy_delay-laser_INTE_520": int(400e3),  # 400 us for galvo
        "xy_dtype-laser_INTE_520": float,
        "xy_nm_per_unit-laser_INTE_520": 1000,
        "xy_optimize_range-laser_INTE_520": 0.3,
        "xy_units-laser_INTE_520": "MHz",
        #
        "xy_control_mode-laser_COBO_638": ControlMode.SEQUENCE,
        "xy_delay-laser_COBO_638": int(400e3),  # 400 us for galvo
        "xy_dtype-laser_COBO_638": float,
        "xy_nm_per_unit-laser_COBO_638": 1000,
        "xy_optimize_range-laser_COBO_638": 0.2,
        "xy_units-laser_COBO_638": "MHz",
        #
        "z_control_mode": ControlMode.STREAM,
        "z_delay": int(5e6),  # 5 ms for PIFOC
        "z_dtype": float,
        "z_nm_per_unit": 1000,
        "z_optimize_range": 0.3,
        "z_units": "Voltage (V)",
        "widefield_calibration_nv1": widefield_calibration_nv1.copy(),
        "widefield_calibration_nv2": widefield_calibration_nv2.copy(),
    },
    ###
    "Servers": {
        "counter": "QM_opx",
        "magnet_rotation": "rotation_stage_thor_ell18k",
        # "pos_xy": "pos_xyz_THOR_gvs212_PI_pifoc",
        "pos_z": "pos_z_PI_pifoc",
        "pulse_gen": "QM_opx",
        "sig_gen_HIGH": "sig_gen_STAN_sg394",
        "sig_gen_LOW": "sig_gen_STAN_sg394",
        "tagger": "QM_opx",
        "camera": "camera_NUVU_hnu512gamma",
    },
    ###
    "Wiring": {
        "Daq": {
            "ao_galvo_x": "dev1/AO0",
            "ao_galvo_y": "dev1/AO1",
            "ao_objective_piezo": "dev1/AO2",
            "di_clock": "PFI12",
        },
        "PulseGen": {
            "do_apd_gate": 5,
            "do_laser_INTE_520_dm": 3,
            "do_laser_OPTO_589_dm": 3,
            "do_laser_COBO_638_dm": 7,
            "do_sample_clock": 0,
            "do_sig_gen_BERK_bnc835_gate": 1,
            "do_sig_gen_STAN_sg394_gate": 10,
            "do_camera_trigger": 7,
        },
        "Tagger": {"di_apd_0": 2, "di_apd_1": 4, "di_apd_gate": 3, "di_clock": 1},
    },
}

# endregion
# region OPX variables

analog_output_delay = 136  # ns

# "Intermediate" frequencies
default_int_freq = 100e6  #
NV_IF_freq = 40e6  # in units of Hz
NV2_IF_freq = 45e6
NV_LO_freq = 2.83e9  # in units of Hz

# Pulses lengths
initialization_len = 200  # in ns
meas_len = 100  # in ns
long_meas_len = 100  # in ns

# MW parameters
mw_amp_NV = 0.5  # in units of volts
mw_len_NV = 200  # in units of ns

aom_amp = 0.5

pi_amp_NV = 0.1  # in units of volts
pi_len_NV = 100  # in units of ns

pi_half_amp_NV = pi_amp_NV / 2  # in units of volts
pi_half_len_NV = pi_len_NV  # in units of ns

# Readout parameters
signal_threshold = -200

# Delays
detection_delay = 36  # keep at 36ns minimum
mw_delay = 0

# uwave length. doesn't really matter
uwave_len = 100

green_laser_delay = 0
red_laser_delay = 0
apd_0_delay = 0
apd_1_delay = 0
uwave_delay = 0
aod_delay = 0
yellow_aom_delay = 0
tsg4104_I_delay = 0
tsg4104_Q_delay = 0
delays = [
    green_laser_delay,
    red_laser_delay,
    apd_0_delay,
    apd_1_delay,
    uwave_delay,
    aod_delay,
    yellow_aom_delay,
    tsg4104_I_delay,
    tsg4104_Q_delay,
]

min_delay = 150  # we use 100 with the pulse streamer. doesn't matter. just wanted it higher than 136 analog delay

common_delay = max(delays) + min_delay

green_laser_total_delay = common_delay - green_laser_delay
red_laser_total_delay = common_delay - red_laser_delay
apd_0_total_delay = common_delay - apd_0_delay
apd_1_total_delay = common_delay - apd_0_delay
uwave_total_delay = common_delay - uwave_delay
NV_total_delay = common_delay - mw_delay
NV2_total_delay = common_delay - mw_delay
AOD_total_delay = common_delay - aod_delay
yellow_AOM_total_delay = common_delay - yellow_aom_delay
tsg4104_I_total_delay = common_delay - tsg4104_I_delay
tsg4104_Q_total_delay = common_delay - tsg4104_Q_delay

default_pulse_duration = config["CommonDurations"]["default_pulse_duration"]

# endregion
# region OPX config

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
                2: {},  # apd 1 virtual gate
                3: {},  # apd 2 virtual gate
                4: {},  # apd 2 virtual gate
                5: {},  # clock
                6: {},  # clock
                7: {},  # tsg4104 sig gen switch
                8: {},  # cobolt 638
                9: {},  # cobolt 515
                10: {},  # cobolt 515
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
            "operations": {"cw": "ao_cw"},
        },
        "ao2": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw"},
        },
        "ao3": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw"},
        },
        "ao4": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw"},
        },
        "ao5": {
            "singleInput": {"port": ("con1", 5)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw"},
        },
        "ao6": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw"},
        },
        "ao7": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw"},
        },
        "ao8": {
            "singleInput": {"port": ("con1", 8)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw"},
        },
        "ao9": {
            "singleInput": {"port": ("con1", 9)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw"},
        },
        "ao10": {
            "singleInput": {"port": ("con1", 10)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw"},
        },
        # endregion
        # region Actual "elements", or physical things to control
        "do_laser_COBO_638_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 1), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off", "ionize": "do_ionization"},
        },
        # "do_laser_OPTO_589_dm": {
        #     "digitalInputs": {"chan": {"port": ("con1", 3), "delay": 0, "buffer": 0}},
        #     "operations": {"on": "do_on", "off": "do_off"},
        # },
        "ao_laser_OPTO_589_am": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": 0,
            "operations": {
                "on": "ao_cw",
                "off": "ao_off",
                "charge_state_readout": "charge_state_readout",
            },
        },
        "do_laser_INTE_520_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 4), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do_sig_gen_STAN_sg394_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 2), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do_camera_trigger": {
            "digitalInputs": {"chan": {"port": ("con1", 5), "delay": 0, "buffer": 0}},
            "sticky": {"analog": True, "digital": True, "duration": 160},
            # "hold_offset": {"duration": 200},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "ao_laser_COBO_638_x": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": 75e6,
            # "sticky": {"analog": True, "duration": 160},
            "operations": {"aod_cw": "red_aod_cw"},
        },
        "ao_laser_COBO_638_y": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": 75e6,
            # "sticky": {"analog": True, "duration": 160},
            "operations": {"aod_cw": "red_aod_cw"},
        },
        "ao_laser_INTE_520_x": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": 110e6,
            # "sticky": {"analog": True},
            "operations": {"aod_cw": "green_aod_cw"},
        },
        "ao_laser_INTE_520_y": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": 110e6,
            # "sticky": {"analog": True},
            "operations": {"aod_cw": "green_aod_cw"},
        },
        # endregion
    },
    # region Pulses
    "pulses": {
        ### Analog
        "green_aod_cw": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw"},
        },
        "red_aod_cw": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "red_aod_cw"},
        },
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
        "charge_state_readout": {
            "operation": "control",
            "length": 10e6,
            "waveforms": {"single": "charge_state_readout"},
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
        "do_ionization": {
            "operation": "control",
            "length": 200,
            "digital_marker": "on",
        },
        ### Mixed
    },
    # endregion
    # region Waveforms
    ### Analog
    "waveforms": {
        "aod_cw": {"type": "constant", "sample": 0.35},
        "red_aod_cw": {"type": "constant", "sample": 0.32},
        # "green_aod_cw": {"type": "constant", "sample": 0.35},
        "green_aod_cw": {"type": "constant", "sample": 0.19},
        "cw": {"type": "constant", "sample": 0.5},
        "off": {"type": "constant", "sample": 0.0},
        "cw_0.5": {"type": "constant", "sample": 0.5},
        "cw_0.45": {"type": "constant", "sample": 0.45},
        "cw_0.4": {"type": "constant", "sample": 0.4},
        "charge_state_readout": {"type": "constant", "sample": 0.5},
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


if __name__ == "__main__":
    # print(config)
    print(config["DeviceIDs"]["gcs_dll_path"])

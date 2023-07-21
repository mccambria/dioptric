# -*- coding: utf-8 -*-
"""
Config file for the PC rabi

Created July 20th, 2023

@author: mccambria
"""

from utils.constants import ModTypes
from utils.positioning import ControlStyle
from pathlib import Path

home = Path.home()

# region Base config

config = {
    ###
    "apd_indices": [0],
    "nv_sig_units": "{'coords': 'V', 'expected_count_rate': 'kcps', 'durations': 'ns', 'magnet_angle': 'deg', 'resonance': 'GHz', 'rabi': 'ns', 'uwave_power': 'dBm'}",
    "shared_email": "kolkowitznvlab@gmail.com",
    "windows_nvdata_path": Path("E:/Shared drives/Kolkowitz Lab Group/nvdata"),
    "linux_nvdata_path": home / "E/nvdata",
    "windows_repo_path": home / "Documents/GitHub/dioptric",
    "linux_repo_path": home / "Documents/GitHub/dioptric",
    ###
    "CommonDurations": {
        "cw_meas_buffer": 5000,
        "pol_to_uwave_wait_dur": 5000,
        "scc_ion_readout_buffer": 1000,
        "uwave_buffer": 1000,
        "uwave_to_readout_wait_dur": 1000,
    },
    ###
    "DeviceIDs": {
        "piezo_stage_616_3cd_model": "E727",
        "piezo_stage_616_3cd_serial": "0121089079",
        "rotation_stage_ell18k_address": "COM5",
        "signal_generator_tsg4104a_visa_address": "TCPIP0::128.104.160.112::5025::SOCKET",
        "QM_opx_ip": "128.104.160.117",
    },
    ###
    "Microwaves": {
        "sig_gen_TEKT_tsg4104a": {"delay": 260},
        "iq_comp_amp": 0.5,
        "iq_delay": 0,
        "sig_gen_HIGH": "sig_gen_TEKT_tsg4104a",
        "sig_gen_LOW": "sig_gen_TEKT_tsg4104a",
    },
    ###
    "Optics": {
        "cobolt_515": {
            "delay": 400,
            "feedthrough": False,
            "mod_type": ModTypes.DIGITAL,
        },
        "cobolt_638": {
            "delay": 300,
            "feedthrough": False,
            "mod_type": ModTypes.DIGITAL,
        },
        "laserglow_589": {
            "delay": 1750,
            "feedthrough": False,
            "mod_type": ModTypes.ANALOG,
        },
    },
    ###
    "PhotonCollection": {
        "qm_opx_max_readout_time": 5000000,
    },
    ###
    "Positioning": {
        "pos_xy_server": "pos_xyz_PI_616_3cd_digital",
        "pos_xyz_server": "pos_xyz_PI_616_3cd_digital",
        "pos_z_server": "pos_xyz_PI_616_3cd_digital",
        "xy_control_style": ControlStyle.STEP,
        "xy_delay": 50000000,
        "xy_dtype": "float",
        "xy_nm_per_unit": 1000,
        "xy_optimize_range": 0.95,
        "xy_server": "pos_xyz_PI_616_3cd_digital",
        "xy_small_response_delay": 800,
        "xy_units": "um",
        "xyz_positional_accuracy": 0.002,
        "xyz_server": "pos_xyz_PI_616_3cd_digital",
        "xyz_timeout": 1,
        "z_control_style": ControlStyle.STEP,
        "z_delay": 50000000,
        "z_dtype": "float",
        "z_nm_per_unit": 1000,
        "z_optimize_range": 4,
        "z_server": "pos_xyz_PI_616_3cd_digital",
        "z_small_response_delay": 50000000,
        "z_units": "nm",
    },
    ###
    "Servers": {
        "arb_wave_gen": "QM_opx",
        "counter": "QM_opx",
        "magnet_rotation": "rotation_stage_THOR_ell18k",
        "pos_xy": "pos_xyz_PI_616_3cd_digital",
        "pos_xyz": "pos_xyz_PI_616_3cd_digital",
        "pos_z": "pos_xyz_PI_616_3cd_digital",
        "pulse_gen": "QM_opx",
        "sig_gen_HIGH": "sig_gen_TEKT_tsg4104a",
        "sig_gen_LOW": "sig_gen_TEKT_tsg4104a",
        "tagger": "QM_opx",
    },
    ###
    "Wiring": {
        "PulseGen": {
            "do_apd_0_gate": 1,
            "do_apd_1_gate": 0,
            "do_integrated_520_dm": 5,
            "do_sample_clock": 0,
        },
        "QmOpx": {
            "ao_laserglow_589_am": 5,
            "do_cobolt_515_dm": 9,
        },
        "Tagger": {
            "di_apd_0": 10,
            "di_apd_1": 10,
            "di_apd_gate": 10,
        },
    },
}

# endregion

# -*- coding: utf-8 -*-
"""
Config file for the PC rabi

Created July 20th, 2023

@author: mccambria
"""

from utils.constants import ModTypes, ControlStyle, CountFormat, CollectionMode
from pathlib import Path

home = Path.home()

# region Base config

config = {
    ###
    "apd_indices": [0],
    "count_format": CountFormat.RAW,
    "collection_mode": CollectionMode.WIDEFIELD,
    "camera_spot_radius": 8,  # Distance to first Airy zero in units of camera pixels for diffraction-limited spot
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
    },
    ###
    "DeviceIDs": {
        "arb_wave_gen_visa_address": "TCPIP0::128.104.160.119::5025::SOCKET",
        "daq0_name": "Dev1",
        "filter_slider_THOR_ell9k_2_com": "COM11",
        "filter_slider_THOR_ell9k_3_com": "COM9",
        "filter_slider_THOR_ell9k_com": "COM5",
        "gcs_dll_path": home
        / "Documents/GitHub/dioptric/servers/outputs/GCSTranslator/PI_GCS2_DLL_x64.dll",
        "objective_piezo_model": "E709",
        "objective_piezo_serial": "0119008970",
        "piezo_stage_626_2cd_model": "E727",
        "piezo_stage_626_2cd_serial": "0116058375",
        "pulse_gen_SWAB_82_ip": "192.168.0.111",
        "rotation_stage_ell18k_address": "COM6",
        "sig_gen_BERK_bnc835_visa": "TCPIP::128.104.160.114::inst0::INSTR",
        "sig_gen_STAN_sg394_visa": "TCPIP::192.168.0.112::inst0::INSTR",
        "sig_gen_TEKT_tsg4104a_visa": "TCPIP0::128.104.160.112::5025::SOCKET",
        "tagger_SWAB_20_serial": "1740000JEH",
        "temp_ctrl_tc200": "COM10",
        "z_piezo_kpz101_serial": "29502179",
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
    "Optics": {
        "cobolt_515": {
            "delay": 120,
            "mod_type": ModTypes.DIGITAL,
        },
        "cobolt_638": {
            "delay": 80,
            "mod_type": ModTypes.DIGITAL,
        },
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
            "mod_type": ModTypes.DIGITAL,
        },
        "laser_LGLO_589": {
            "delay": 2500,
            "mod_type": ModTypes.DIGITAL,
            "filter_server": "filter_slider_ell9k",
            "filter_mapping": {"nd_0": 0, "nd_0.5": 1, "nd_1.0": 2, "nd_1.5": 3},
        },
        "laserglow_532": {
            "delay": 1030,
            "mod_type": ModTypes.DIGITAL,
            "filter_server": "filter_slider_ell9k_2",
            "filter_mapping": {"nd_2.0": 0, "nd_1.0": 1, "nd_0.5": 2, "nd_0": 3},
        },
    },
    ###
    "Positioning": {
        "xy_control_style": ControlStyle.STREAM,
        "xy_delay": int(400e3),  # 400 us for galvo
        "xy_dtype": float,
        "xy_nm_per_unit": 1000,
        "xy_optimize_range": 0.95,
        "xy_small_response_delay": 800,
        "xy_units": "Voltage (V)",
        "z_control_style": ControlStyle.STREAM,
        "z_delay": int(5e6),  # 5 ms for PIFOC
        "z_dtype": float,
        "z_nm_per_unit": 1000,
        "z_optimize_range": 4,
        "z_units": "Voltage (V)",
        "NV1_pixel_coords": [308.158, 309.335],
        "NV1_scanning_coords": [0.155, 0],
        "NV2_pixel_coords": [124.633, 196.258],
        "NV2_scanning_coords": [-0.135, 0.162],
    },
    ###
    "Servers": {
        "arb_wave_gen": "awg_KEYS_33622A",
        "charge_readout_laser": "laser_LGLO_589",
        "counter": "tagger_SWAB_20",
        "magnet_rotation": "rotation_stage_thor_ell18k",
        "pos_xy": "pos_xyz_THOR_gvs212_PI_pifoc",
        "pos_xyz": "pos_xyz_THOR_gvs212_PI_pifoc",
        "pos_z": "pos_xyz_THOR_gvs212_PI_pifoc",
        "pulse_gen": "pulse_gen_SWAB_82",
        "sig_gen_HIGH": "sig_gen_STAN_sg394",
        "sig_gen_LOW": "sig_gen_BERK_bnc835",
        "sig_gen_omni": "sig_gen_BERK_bnc835",
        "sig_gen_single": "sig_gen_STAN_sg394",
        "tagger": "tagger_SWAB_20",
        "camera": "camera_NUVU_hnu512gamma",
    },
    ###
    "Wiring": {
        "Daq": {
            "ai_photodiode": "Dev1/AI0",
            "ai_thermistor_ref": "dev1/AI1",
            "ao_galvo_x": "dev1/AO0",
            "ao_galvo_y": "dev1/AO1",
            "ao_laser_LGLO_589_feedthrough": "dev1/AO3",
            "ao_objective_piezo": "dev1/AO2",
            "ao_piezo_stage_626_2cd_x": "dev1/AO0",
            "ao_piezo_stage_626_2cd_y": "dev1/AO1",
            "ao_uwave_sig_gen_mod": "",
            "ao_z_piezo_kpz101": "dev1/AO2",
            "di_clock": "PFI12",
            "di_laser_LGLO_589_feedthrough": "PFI0",
        },
        "Piezo_stage_E727": {"piezo_stage_channel_x": 4, "piezo_stage_channel_y": 5},
        "PulseGen": {
            "ao_fm_sig_gen_BERK_bnc835": 1,
            "ao_fm_sig_gen_STAN_sg394": 0,
            "ao_laser_LGLO_589_am": 1,
            "do_apd_gate": 5,
            "do_arb_wave_trigger": 2,
            "do_laser_COBO_638_dm": 7,
            "do_laser_INTE_520_dm": 3,
            "do_laser_LGLO_589_am": 6,
            "do_sample_clock": 0,
            "do_sig_gen_BERK_bnc835_gate": 1,
            "do_sig_gen_STAN_sg394_gate": 4,
            "do_camera_trigger": 7,
        },
        "Tagger": {"di_apd_0": 2, "di_apd_1": 4, "di_apd_gate": 3, "di_clock": 1},
    },
}

# endregion

if __name__ == "__main__":
    # print(config)
    print(config["DeviceIDs"]["gcs_dll_path"])

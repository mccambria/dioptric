# -*- coding: utf-8 -*-
"""
Config file for the PC rabi

Created July 20th, 2023

@author: mccambria
"""

from pathlib import Path

from config.default import config
from utils.constants import (
    CollectionMode,
    CoordsKey,
    CountFormat,
    LaserPosMode,
    ModMode,
    PosControlMode,
    VirtualLaserKey,
)

home = Path.home()

# region Widefield calibration coords

green_laser = "laser_INTE_520"
yellow_laser = "laser_OPTO_589"
red_laser = "laser_COBO_638"

pixel_coords_key = "pixel_coords"
green_coords_key = f"coords-{green_laser}"
red_coords_key = f"coords-{red_laser}"

widefield_calibration_nv1 = {
    pixel_coords_key: [107.64, 76.7],
    green_coords_key: [111.85, 109.792],
    red_coords_key: [75.946, 75.332],
}
widefield_calibration_nv2 = {
    pixel_coords_key: [74.828, 109.09],
    green_coords_key: [110.7, 110.817],
    red_coords_key: [75.007, 76.233],
}


# endregion
# region Base config

# Add on to the default config
config |= {
    ###
    "count_format": CountFormat.RAW,
    "collection_mode": CollectionMode.CAMERA,
    ###
    # Common durations are in ns
    "CommonDurations": {
        "uwave_buffer": 1000,
        "default_pulse_duration": 1000,
        "aod_access_time": 20e3,
        "widefield_operation_buffer": 10e3,
    },
    ###
    "DeviceIDs": {
        "arb_wave_gen_visa_address": "TCPIP0::128.104.ramp_to_zero_duration.119::5025::SOCKET",
        "daq0_name": "Dev1",
        "filter_slider_THOR_ell9k_com": "COM13",
        "gcs_dll_path": home
        / "Documents/GitHub/dioptric/servers/outputs/GCSTranslator/PI_GCS2_DLL_x64.dll",
        "objective_piezo_model": "E709",
        "objective_piezo_serial": "0119008970",
        "pulse_gen_SWAB_82_ip": "192.168.0.111",
        "rotation_stage_THOR_ell18k_com": "COM8",
        "sig_gen_BERK_bnc835_visa": "TCPIP::128.104.ramp_to_zero_duration.114::inst0::INSTR",
        "sig_gen_STAN_sg394_visa": "TCPIP::192.168.0.120::inst0::INSTR",
        "sig_gen_STAN_sg394_2_visa": "TCPIP::192.168.0.121::inst0::INSTR",
        "sig_gen_TEKT_tsg4104a_visa": "TCPIP0::128.104.ramp_to_zero_duration.112::5025::SOCKET",
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
        "sig_gen_0": {
            "name": "sig_gen_STAN_sg394",
            # "frequency": 2.87,
            # "rabi_period": 96,
            "frequency": 2.8476285621036683,
            "rabi_period": 144,
            # "rabi_period": 32,
            # "uwave_power": 9,
            "uwave_power": 8.1,
            "iq_delay": 140,
        },
        "sig_gen_1": {
            "name": "sig_gen_STAN_sg394_2",
            "frequency": 2.8924815710559706,
            "rabi_period": 144,
            "uwave_power": 7.5,
        },
    },
    ###
    "Camera": {
        "resolution": (512, 512),
        "spot_radius": 5,  # Radius for integrating NV counts in a camera image
        "bias_clamp": 300,  # (changing this won't actually change the value on the camera currently)
        "em_gain": 1000,
        "temp": -60,
        "timeout": 1000,  # ms
        # Readout mode specifies EM vs conventional, as well as vertical and horizontal readout frequencies.
        # See camera server file for details
        "readout_mode": 1,  # 16 for double horizontal readout rate
        "roi": (226, 185, 150, 150),  # offsetX, offsetY, width, height
    },
    ###
    "Optics": {
        "laser_INTE_520": {
            "delay": 0,
            "mod_mode": ModMode.DIGITAL,
            "pos_mode": LaserPosMode.SCANNING,
            "aod": True,
        },
        "laser_OPTO_589": {
            "delay": 0,
            "mod_mode": ModMode.ANALOG,
            "pos_mode": LaserPosMode.WIDEFIELD,
        },
        "laser_COBO_638": {
            "delay": 0,
            "mod_mode": ModMode.DIGITAL,
            "pos_mode": LaserPosMode.SCANNING,
            "aod": True,
        },
        VirtualLaserKey.IMAGING: {"name": "laser_INTE_520", "duration": 50e6},
        VirtualLaserKey.WIDEFIELD_IMAGING: {
            "name": "laser_OPTO_589",
            "duration": 30e6,
        },  # 35e6
        VirtualLaserKey.SPIN_READOUT: {"name": "laser_INTE_520", "duration": 300},
        VirtualLaserKey.POLARIZATION: {"name": "laser_INTE_520", "duration": 10e3},
        VirtualLaserKey.IONIZATION: {"name": "laser_COBO_638", "duration": 112},
        VirtualLaserKey.CHARGE_READOUT: {
            "name": "laser_OPTO_589",
            "duration": 30e6,
            # "duration": 100e6,
        },  # 35e6, 0.09
    },
    ###
    "Positioning": {
        green_laser: {
            "xy_control_mode": PosControlMode.SEQUENCE,
            "xy_delay": int(400e3),  # 400 us for galvo
            "xy_dtype": float,
            "xy_nm_per_unit": 1000,
            "xy_optimize_range": 0.5,
            "xy_units": "MHz",
        },
        red_laser: {
            "xy_control_mode": PosControlMode.SEQUENCE,
            "xy_delay": int(400e3),  # 400 us for galvo
            "xy_dtype": float,
            "xy_nm_per_unit": 1000,
            "xy_optimize_range": 1.2,
            "xy_units": "MHz",
        },
        CoordsKey.GLOBAL: {
            "z_control_mode": PosControlMode.STREAM,
            "z_delay": int(5e6),  # 5 ms for PIFOC
            "z_dtype": float,
            "z_nm_per_unit": 1000,
            "z_optimize_range": 0.3,
            "z_units": "Voltage (V)",
        },
        "widefield_calibration_nv1": widefield_calibration_nv1.copy(),
        "widefield_calibration_nv2": widefield_calibration_nv2.copy(),
    },
    ###
    "Servers": {
        "counter": "QM_opx",
        "magnet_rotation": "rotation_stage_thor_ell18k",
        "pos_z": "pos_z_PI_pifoc",
        "pulse_gen": "QM_opx",
        "sig_gen_LOW": "sig_gen_STAN_sg394",
        "sig_gen_HIGH": "sig_gen_STAN_sg394_2",
        "sig_gen_0": "sig_gen_STAN_sg394",
        "sig_gen_1": "sig_gen_STAN_sg394_2",
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
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "ionize": "do_ionization",
                "long_ionize": "do_long_ionization",
            },
        },
        "ao_laser_OPTO_589_am": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": 0,
            "operations": {
                "on": "yellow_imaging",
                "off": "ao_off",
                "charge_readout": "yellow_charge_readout",
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
                "polarize": "do_polarization",
            },
        },
        "do_sig_gen_STAN_sg394_dm": {
            "digitalInputs": {
                "chan": {
                    "port": ("con1", 10),
                    # "delay": 0,
                    "delay": config["Microwaves"]["sig_gen_0"]["iq_delay"]
                    + iq_buffer // 2,
                    "buffer": 0,
                }
            },
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "pi_pulse": "do_pi_pulse_0",
                "pi_on_2_pulse": "do_pi_on_2_pulse_0",
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
            "digitalInputs": {"chan": {"port": ("con1", 3), "delay": 0, "buffer": 0}},
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "pi_pulse": "do_pi_pulse_1",
                "pi_on_2_pulse": "do_pi_on_2_pulse_1",
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
            "operations": {"aod_cw": "red_aod_cw", "continue": "ao_off"},
        },
        "ao_laser_COBO_638_y": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": 75e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {"aod_cw": "red_aod_cw", "continue": "ao_off"},
        },
        "ao_laser_INTE_520_x": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": 110e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {"aod_cw": "green_aod_cw", "continue": "ao_off"},
        },
        "ao_laser_INTE_520_y": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": 110e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {"aod_cw": "green_aod_cw", "continue": "ao_off"},
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
        "do_ionization": {
            "operation": "control",
            "length": config["Optics"][VirtualLaserKey.IONIZATION]["duration"],
            "digital_marker": "on",
        },
        "do_long_ionization": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "on",
        },
        "do_polarization": {
            "operation": "control",
            "length": config["Optics"][VirtualLaserKey.POLARIZATION]["duration"],
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
            "length": int(rabi_period_1 / 2),
            "digital_marker": "on",
        },
        "do_pi_on_2_pulse_1": {
            "operation": "control",
            "length": int(rabi_period_1 / 4),
            "digital_marker": "on",
        },
        ### Mixed
    },
    # endregion
    # region Waveforms
    ### Analog
    "waveforms": {
        "aod_cw": {"type": "constant", "sample": 0.35},
        # "red_aod_cw": {"type": "constant", "sample": 0.32},
        # "red_aod_cw": {"type": "constant", "sample": 0.41},
        "red_aod_cw": {"type": "constant", "sample": 0.17},
        # "red_aod_cw": {"type": "constant", "sample": 0.14},  # MCC
        # "red_aod_cw": {"type": "constant", "sample": 0.19},  # MCC
        "green_aod_cw": {"type": "constant", "sample": 0.19},
        "yellow_imaging": {"type": "constant", "sample": 0.20},  # 0.35
        # "yellow_charge_readout": {"type": "constant", "sample": 0.085},  # 30e6
        # "yellow_charge_readout": {"type": "constant", "sample": 0.085},  # 30e6
        "yellow_charge_readout": {"type": "constant", "sample": 0.080},  # 30e6
        # "yellow_charge_readout": {"type": "constant", "sample": 0.075},  # 100e6
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


if __name__ == "__main__":
    # print(config)
    print(config["DeviceIDs"]["gcs_dll_path"])

# -*- coding: utf-8 -*-
"""
Config file for the PC Carr

Created June 20th, 2023

@author: mccambria
"""

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

default_len = 1000

# endregion
# region OPX config

opx_config = {
    "version": 1,
    # region Elements
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0, "delay": NV_total_delay},  # will be I for sig gen
                2: {"offset": 0.0, "delay": NV_total_delay},  # will be Q for sig gen
                3: {"offset": 0.0, "delay": AOD_total_delay},  # AOD_1X
                4: {"offset": 0.0, "delay": AOD_total_delay},  # AOD_1Y
                5: {"offset": 0.0, "delay": yellow_AOM_total_delay},  # yellow AOM
                6: {"offset": 0.0, "delay": tsg4104_I_total_delay},  # sig gen tsg4104 I
                7: {"offset": 0.0, "delay": tsg4104_Q_total_delay},  # sig gen tsg4104 Q
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
    # endregion
    # region Elements
    "elements": {
        # Region Bare channels
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
        "laserglow_589_x": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw"},
        },
        "sig_gen_TEKT_tsg4104a": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 7),
                    "delay": uwave_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {"uwave_on": "do_on", "uwave_off": "do_off"},
        },
        "cobolt_515": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 9),
                    "delay": green_laser_total_delay,
                    "buffer": 0,
                },
            },
            "operations": {"laser_on": "do_on", "laser_off": "do_off"},
        },
        "do_sample_clock": {
            "digitalInputs": {
                "marker": {"port": ("con1", 5), "delay": common_delay, "buffer": 0},
            },
            "operations": {
                "clock_pulse": "do_short_pulse",
            },
        },
        ###
        # "do_apd_1_gate": {
        #     "singleInput": {"port": ("opx1", 2)},
        #     "digitalInputs": {
        #         "marker": {
        #             "port": ("opx1", 3),
        #             "delay": apd_1_total_delay,
        #             "buffer": 0,
        #         },
        #     },
        #     "operations": {
        #         "readout": "do_on",
        #     },
        #     "outputs": {"out1": ("opx1", 2)},
        #     "outputPulseParameters": {
        #         "signalThreshold": signal_threshold,
        #         "signalPolarity": "Below",
        #         "derivativeThreshold": 1800,
        #         "derivativePolarity": "Below",
        #     },
        #     "time_of_flight": detection_delay,
        #     "smearing": 15,
        # },
    },
    # endregion
    # region Pulses
    "pulses": {
        ### Analog
        "ao_cw": {
            "operation": "control",
            "length": default_len,
            "waveforms": {"single": "cw"},
        },
        ### Digital
        "do_on": {
            "operation": "control",
            "length": default_len,
            "digital_marker": "on",
        },
        "do_off": {
            "operation": "control",
            "length": default_len,
            "digital_marker": "off",
        },
        "do_short_pulse": {
            "operation": "control",
            "length": default_len,
            "digital_marker": "square",
        },
        ### Mixed
    },
    # endregion
    # region Waveforms
    ### Analog
    "waveforms": {
        "cw": {"type": "constant", "sample": 0.5},
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
